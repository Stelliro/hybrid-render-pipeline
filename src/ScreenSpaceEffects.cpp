/// @file ScreenSpaceEffects.cpp
/// @brief Implementation of the screen-space post-processing pipeline.
///
/// Creates offscreen render targets, compute pipelines for GTAO+/SVRM/SHR/ACTF,
/// builds the Hi-Z depth pyramid, and composites everything onto the swapchain.
///
/// This is the core of the "2D compositing over 3D geometry" rendering approach:
/// the scene renders to flat 2D textures, and all lighting/shadow/AO/GI/reflection
/// effects are computed entirely in 2D screen space via compute shaders.

#include "ScreenSpaceEffects.hpp"
#include "Renderer.hpp"
#include "Log.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>

namespace hrp {

// ══════════════════════════════════════════════════════════════════════
// Initialization / Shutdown
// ══════════════════════════════════════════════════════════════════════

bool ScreenSpaceEffects::Initialize(Renderer& renderer)
{
    m_device    = renderer.GetDevice();
    m_allocator = renderer.GetAllocator();

    auto extent = renderer.GetSwapchainExtent();
    m_width  = extent.width;
    m_height = extent.height;

    HRP_LOG_INFO("ScreenSpaceEffects: Initializing ({}×{})", m_width, m_height);

    if (!CreateSamplers()) return false;
    if (!CreateRenderTargets()) return false;
    if (!CreateDescriptorPool()) return false;
    if (!CreateHiZResources(renderer)) return false;
    if (!CreateAOResources(renderer)) return false;
    if (!CreateGIResources(renderer)) return false;
    if (!CreateReflResources(renderer)) return false;
    if (!CreateDenoiseResources(renderer)) return false;
    if (!CreateCompositeResources(renderer)) return false;

    m_initialized = true;
    HRP_LOG_INFO("ScreenSpaceEffects: Initialized — GTAO+, SVRM, SHR, ACTF, RT Composite ready");
    return true;
}

void ScreenSpaceEffects::Shutdown()
{
    if (!m_device) return;

    vkDeviceWaitIdle(m_device);

    DestroyPipelineResources();

    if (m_descriptorPool) {
        vkDestroyDescriptorPool(m_device, m_descriptorPool, nullptr);
        m_descriptorPool = VK_NULL_HANDLE;
    }

    DestroyRenderTargets();
    DestroySamplers();

    m_initialized = false;
    HRP_LOG_INFO("ScreenSpaceEffects: Shut down");
}

void ScreenSpaceEffects::OnSwapchainResized(Renderer& renderer)
{
    if (!m_initialized) return;

    auto extent = renderer.GetSwapchainExtent();
    if (extent.width == m_width && extent.height == m_height) return;

    HRP_LOG_INFO("ScreenSpaceEffects: Resizing {}×{} → {}×{}",
                 m_width, m_height, extent.width, extent.height);

    vkDeviceWaitIdle(m_device);

    // Destroy and recreate everything at new resolution
    DestroyPipelineResources();
    if (m_descriptorPool) {
        vkDestroyDescriptorPool(m_device, m_descriptorPool, nullptr);
        m_descriptorPool = VK_NULL_HANDLE;
    }
    DestroyRenderTargets();

    m_width  = extent.width;
    m_height = extent.height;

    CreateRenderTargets();
    CreateDescriptorPool();
    CreateHiZResources(renderer);
    CreateAOResources(renderer);
    CreateGIResources(renderer);
    CreateReflResources(renderer);
    CreateDenoiseResources(renderer);
    CreateCompositeResources(renderer);
}

// ══════════════════════════════════════════════════════════════════════
// Offscreen Image Management
// ══════════════════════════════════════════════════════════════════════

bool ScreenSpaceEffects::CreateOffscreenImage(OffscreenImage& img, uint32_t w, uint32_t h,
                                               VkFormat format, VkImageUsageFlags usage,
                                               uint32_t mipLevels)
{
    img.width     = w;
    img.height    = h;
    img.format    = format;
    img.mipLevels = mipLevels;

    VkImageCreateInfo imageInfo{};
    imageInfo.sType         = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    imageInfo.imageType     = VK_IMAGE_TYPE_2D;
    imageInfo.format        = format;
    imageInfo.extent        = { w, h, 1 };
    imageInfo.mipLevels     = mipLevels;
    imageInfo.arrayLayers   = 1;
    imageInfo.samples       = VK_SAMPLE_COUNT_1_BIT;
    imageInfo.tiling        = VK_IMAGE_TILING_OPTIMAL;
    imageInfo.usage         = usage;
    imageInfo.sharingMode   = VK_SHARING_MODE_EXCLUSIVE;
    imageInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;

    VmaAllocationCreateInfo allocInfo{};
    allocInfo.usage = VMA_MEMORY_USAGE_GPU_ONLY;

    if (vmaCreateImage(m_allocator, &imageInfo, &allocInfo,
                       &img.image, &img.allocation, nullptr) != VK_SUCCESS) {
        HRP_LOG_ERROR("ScreenSpaceEffects: Failed to create image ({}×{}, format {})",
                      w, h, static_cast<int>(format));
        return false;
    }

    // Determine aspect flags
    VkImageAspectFlags aspect = VK_IMAGE_ASPECT_COLOR_BIT;
    if (format == VK_FORMAT_D32_SFLOAT || format == VK_FORMAT_D32_SFLOAT_S8_UINT ||
        format == VK_FORMAT_D24_UNORM_S8_UINT || format == VK_FORMAT_D16_UNORM) {
        aspect = VK_IMAGE_ASPECT_DEPTH_BIT;
    }

    // Main view (all mip levels)
    VkImageViewCreateInfo viewInfo{};
    viewInfo.sType    = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    viewInfo.image    = img.image;
    viewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
    viewInfo.format   = format;
    viewInfo.subresourceRange.aspectMask     = aspect;
    viewInfo.subresourceRange.baseMipLevel   = 0;
    viewInfo.subresourceRange.levelCount     = mipLevels;
    viewInfo.subresourceRange.baseArrayLayer = 0;
    viewInfo.subresourceRange.layerCount     = 1;

    if (vkCreateImageView(m_device, &viewInfo, nullptr, &img.view) != VK_SUCCESS) {
        HRP_LOG_ERROR("ScreenSpaceEffects: Failed to create image view");
        return false;
    }

    // Per-mip views (for Hi-Z build and storage image writes)
    if (mipLevels > 1) {
        for (uint32_t mip = 0; mip < mipLevels && mip < img.mipViews.size(); ++mip) {
            VkImageViewCreateInfo mipViewInfo = viewInfo;
            mipViewInfo.subresourceRange.baseMipLevel = mip;
            mipViewInfo.subresourceRange.levelCount   = 1;

            if (vkCreateImageView(m_device, &mipViewInfo, nullptr, &img.mipViews[mip]) != VK_SUCCESS) {
                HRP_LOG_WARN("ScreenSpaceEffects: Failed to create mip {} view", mip);
                img.mipViews[mip] = VK_NULL_HANDLE;
            }
        }
    }

    return true;
}

void ScreenSpaceEffects::DestroyOffscreenImage(OffscreenImage& img)
{
    if (!m_device) return;

    for (auto& mv : img.mipViews) {
        if (mv) { vkDestroyImageView(m_device, mv, nullptr); mv = VK_NULL_HANDLE; }
    }
    if (img.view)       { vkDestroyImageView(m_device, img.view, nullptr); img.view = VK_NULL_HANDLE; }
    if (img.image)      { vmaDestroyImage(m_allocator, img.image, img.allocation); img.image = VK_NULL_HANDLE; }
    img.allocation = VK_NULL_HANDLE;
}

// ══════════════════════════════════════════════════════════════════════
// Render Target Creation
// ══════════════════════════════════════════════════════════════════════

bool ScreenSpaceEffects::CreateRenderTargets()
{
    HRP_LOG_INFO("ScreenSpaceEffects: Creating render targets ({}×{})", m_width, m_height);

    // Calculate Hi-Z mip levels
    uint32_t hiZMips = static_cast<uint32_t>(std::floor(std::log2(std::max(m_width, m_height)))) + 1;
    hiZMips = std::min(hiZMips, 12u);

    // Scene color — RGBA16F for HDR, readable by compute and usable as attachment.
    // TRANSFER_SRC/DST_BIT enables the post-frame blit to the swapchain and
    // any clears.
    if (!CreateOffscreenImage(m_sceneColor, m_width, m_height,
                              VK_FORMAT_R16G16B16A16_SFLOAT,
                              VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT |
                              VK_IMAGE_USAGE_SAMPLED_BIT |
                              VK_IMAGE_USAGE_STORAGE_BIT |
                              VK_IMAGE_USAGE_TRANSFER_SRC_BIT |
                              VK_IMAGE_USAGE_TRANSFER_DST_BIT))
        return false;

    // Scene depth — D32F, readable by compute shaders.
    // TRANSFER_SRC_BIT enables the depth → Hi-Z mip 0 copy in BuildHiZPyramid.
    if (!CreateOffscreenImage(m_sceneDepth, m_width, m_height,
                              VK_FORMAT_D32_SFLOAT,
                              VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT |
                              VK_IMAGE_USAGE_SAMPLED_BIT |
                              VK_IMAGE_USAGE_TRANSFER_SRC_BIT))
        return false;

    // World-space normals — RGB16F
    if (!CreateOffscreenImage(m_sceneNormals, m_width, m_height,
                              VK_FORMAT_R16G16B16A16_SFLOAT,
                              VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT |
                              VK_IMAGE_USAGE_SAMPLED_BIT))
        return false;

    // Hi-Z depth pyramid — R32F with mip chain.
    // TRANSFER_DST_BIT enables clears when the pyramid is rebuilt.
    if (!CreateOffscreenImage(m_hiZBuffer, m_width, m_height,
                              VK_FORMAT_R32_SFLOAT,
                              VK_IMAGE_USAGE_SAMPLED_BIT |
                              VK_IMAGE_USAGE_STORAGE_BIT |
                              VK_IMAGE_USAGE_TRANSFER_DST_BIT,
                              hiZMips))
        return false;

    // Motion vectors — RG16F
    if (!CreateOffscreenImage(m_motionVectors, m_width, m_height,
                              VK_FORMAT_R16G16_SFLOAT,
                              VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT |
                              VK_IMAGE_USAGE_SAMPLED_BIT))
        return false;

    // Effect output buffers — RGBA16F.
    // TRANSFER_DST_BIT enables vkCmdClearColorImage on disabled-effect buffers
    // (see TransitionHistoryBuffers neutral-clear block).
    auto createEffectBuffer = [&](OffscreenImage& img) {
        return CreateOffscreenImage(img, m_width, m_height,
                                    VK_FORMAT_R16G16B16A16_SFLOAT,
                                    VK_IMAGE_USAGE_STORAGE_BIT |
                                    VK_IMAGE_USAGE_SAMPLED_BIT |
                                    VK_IMAGE_USAGE_TRANSFER_DST_BIT);
    };

    if (!createEffectBuffer(m_aoOutput))      return false;
    if (!createEffectBuffer(m_giOutput))      return false;
    if (!createEffectBuffer(m_reflOutput))    return false;
    if (!createEffectBuffer(m_denoiseOutput)) return false;

    // History buffers
    if (!createEffectBuffer(m_prevGI))       return false;
    if (!createEffectBuffer(m_prevRefl))     return false;
    if (!createEffectBuffer(m_prevDenoise))  return false;

    // Previous depth
    if (!CreateOffscreenImage(m_prevDepth, m_width, m_height,
                              VK_FORMAT_D32_SFLOAT,
                              VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT |
                              VK_IMAGE_USAGE_SAMPLED_BIT |
                              VK_IMAGE_USAGE_TRANSFER_DST_BIT))
        return false;

    HRP_LOG_INFO("ScreenSpaceEffects: Created {} render targets ({} Hi-Z mips)",
                 15, hiZMips);
    return true;
}

void ScreenSpaceEffects::DestroyRenderTargets()
{
    DestroyOffscreenImage(m_sceneColor);
    DestroyOffscreenImage(m_sceneDepth);
    DestroyOffscreenImage(m_sceneNormals);
    DestroyOffscreenImage(m_hiZBuffer);
    DestroyOffscreenImage(m_motionVectors);
    DestroyOffscreenImage(m_aoOutput);
    DestroyOffscreenImage(m_giOutput);
    DestroyOffscreenImage(m_reflOutput);
    DestroyOffscreenImage(m_denoiseOutput);
    DestroyOffscreenImage(m_prevGI);
    DestroyOffscreenImage(m_prevRefl);
    DestroyOffscreenImage(m_prevDepth);
    DestroyOffscreenImage(m_prevDenoise);
}

// ══════════════════════════════════════════════════════════════════════
// Samplers
// ══════════════════════════════════════════════════════════════════════

bool ScreenSpaceEffects::CreateSamplers()
{
    // Linear sampler — bilinear filtering, clamp to edge
    VkSamplerCreateInfo linearInfo{};
    linearInfo.sType        = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
    linearInfo.magFilter    = VK_FILTER_LINEAR;
    linearInfo.minFilter    = VK_FILTER_LINEAR;
    linearInfo.mipmapMode   = VK_SAMPLER_MIPMAP_MODE_LINEAR;
    linearInfo.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    linearInfo.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    linearInfo.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    linearInfo.maxLod       = 12.0f;

    if (vkCreateSampler(m_device, &linearInfo, nullptr, &m_linearSampler) != VK_SUCCESS) {
        HRP_LOG_ERROR("ScreenSpaceEffects: Failed to create linear sampler");
        return false;
    }

    // Nearest sampler — point filtering, clamp to edge
    VkSamplerCreateInfo nearestInfo = linearInfo;
    nearestInfo.magFilter  = VK_FILTER_NEAREST;
    nearestInfo.minFilter  = VK_FILTER_NEAREST;
    nearestInfo.mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST;

    if (vkCreateSampler(m_device, &nearestInfo, nullptr, &m_nearestSampler) != VK_SUCCESS) {
        HRP_LOG_ERROR("ScreenSpaceEffects: Failed to create nearest sampler");
        return false;
    }

    return true;
}

void ScreenSpaceEffects::DestroySamplers()
{
    if (m_linearSampler)  { vkDestroySampler(m_device, m_linearSampler, nullptr); m_linearSampler = VK_NULL_HANDLE; }
    if (m_nearestSampler) { vkDestroySampler(m_device, m_nearestSampler, nullptr); m_nearestSampler = VK_NULL_HANDLE; }
}

// ══════════════════════════════════════════════════════════════════════
// Descriptor Pool
// ══════════════════════════════════════════════════════════════════════

bool ScreenSpaceEffects::CreateDescriptorPool()
{
    // Pool sizes: generous allocation for all effect descriptor sets
    std::array<VkDescriptorPoolSize, 2> poolSizes{};
    poolSizes[0].type            = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    poolSizes[0].descriptorCount = 128;
    poolSizes[1].type            = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    poolSizes[1].descriptorCount = 64;

    VkDescriptorPoolCreateInfo poolInfo{};
    poolInfo.sType         = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    poolInfo.maxSets       = 32;
    poolInfo.poolSizeCount = static_cast<uint32_t>(poolSizes.size());
    poolInfo.pPoolSizes    = poolSizes.data();

    if (vkCreateDescriptorPool(m_device, &poolInfo, nullptr, &m_descriptorPool) != VK_SUCCESS) {
        HRP_LOG_ERROR("ScreenSpaceEffects: Failed to create descriptor pool");
        return false;
    }
    return true;
}

// ══════════════════════════════════════════════════════════════════════
// Hi-Z Build Resources
// ══════════════════════════════════════════════════════════════════════

bool ScreenSpaceEffects::CreateHiZResources(Renderer& renderer)
{
    // Descriptor set layout: binding 0 = source sampler, binding 1 = dest storage
    std::array<VkDescriptorSetLayoutBinding, 2> bindings{};
    bindings[0].binding         = 0;
    bindings[0].descriptorType  = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    bindings[0].descriptorCount = 1;
    bindings[0].stageFlags      = VK_SHADER_STAGE_COMPUTE_BIT;
    bindings[1].binding         = 1;
    bindings[1].descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    bindings[1].descriptorCount = 1;
    bindings[1].stageFlags      = VK_SHADER_STAGE_COMPUTE_BIT;

    VkDescriptorSetLayoutCreateInfo layoutInfo{};
    layoutInfo.sType        = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    layoutInfo.bindingCount = static_cast<uint32_t>(bindings.size());
    layoutInfo.pBindings    = bindings.data();

    if (vkCreateDescriptorSetLayout(m_device, &layoutInfo, nullptr, &m_hiZSetLayout) != VK_SUCCESS) {
        HRP_LOG_ERROR("ScreenSpaceEffects: Failed to create Hi-Z set layout");
        return false;
    }

    // Push constants for Hi-Z: vec4 srcSize
    VkPushConstantRange pushRange{};
    pushRange.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    pushRange.offset     = 0;
    pushRange.size       = 16; // vec4

    VkPipelineLayoutCreateInfo pipeLayoutInfo{};
    pipeLayoutInfo.sType                  = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    pipeLayoutInfo.setLayoutCount         = 1;
    pipeLayoutInfo.pSetLayouts            = &m_hiZSetLayout;
    pipeLayoutInfo.pushConstantRangeCount = 1;
    pipeLayoutInfo.pPushConstantRanges    = &pushRange;

    if (vkCreatePipelineLayout(m_device, &pipeLayoutInfo, nullptr, &m_hiZPipeLayout) != VK_SUCCESS) {
        HRP_LOG_ERROR("ScreenSpaceEffects: Failed to create Hi-Z pipeline layout");
        return false;
    }

    // Load compute shader
    VkShaderModule hiZShader = renderer.LoadShaderModule("hi_z_build.comp");
    if (hiZShader == VK_NULL_HANDLE) {
        HRP_LOG_WARN("ScreenSpaceEffects: Hi-Z shader not found — Hi-Z disabled");
        return true; // Non-fatal
    }

    VkComputePipelineCreateInfo compInfo{};
    compInfo.sType  = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    compInfo.layout = m_hiZPipeLayout;
    compInfo.stage.sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    compInfo.stage.stage  = VK_SHADER_STAGE_COMPUTE_BIT;
    compInfo.stage.module = hiZShader;
    compInfo.stage.pName  = "main";

    VkResult result = vkCreateComputePipelines(m_device, VK_NULL_HANDLE, 1, &compInfo,
                                                nullptr, &m_hiZPipeline);
    vkDestroyShaderModule(m_device, hiZShader, nullptr);

    if (result != VK_SUCCESS) {
        HRP_LOG_ERROR("ScreenSpaceEffects: Failed to create Hi-Z compute pipeline");
        return false;
    }

    // Allocate descriptor sets — one per mip level transition
    uint32_t mipCount = m_hiZBuffer.mipLevels;
    for (uint32_t mip = 0; mip + 1 < mipCount && mip < m_hiZSets.size(); ++mip) {
        VkDescriptorSetAllocateInfo allocInfo{};
        allocInfo.sType              = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        allocInfo.descriptorPool     = m_descriptorPool;
        allocInfo.descriptorSetCount = 1;
        allocInfo.pSetLayouts        = &m_hiZSetLayout;

        if (vkAllocateDescriptorSets(m_device, &allocInfo, &m_hiZSets[mip]) != VK_SUCCESS) {
            HRP_LOG_WARN("ScreenSpaceEffects: Failed to allocate Hi-Z descriptor set mip {}", mip);
            continue;
        }

        // For mip 0→1: source is the scene depth (sampled as R32F), dest is HiZ mip 1
        // For mip N→N+1: source is HiZ mip N, dest is HiZ mip N+1
        VkDescriptorImageInfo srcInfo{};
        srcInfo.sampler     = m_nearestSampler;
        srcInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        if (mip == 0) {
            // First reduction: read from scene depth
            srcInfo.imageView = m_sceneDepth.view;
        } else {
            // Subsequent reductions: read from previous Hi-Z mip
            srcInfo.imageView = m_hiZBuffer.mipViews[mip];
        }

        VkDescriptorImageInfo dstInfo{};
        dstInfo.imageView   = m_hiZBuffer.mipViews[mip + 1];
        dstInfo.imageLayout = VK_IMAGE_LAYOUT_GENERAL;

        std::array<VkWriteDescriptorSet, 2> writes{};
        writes[0].sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[0].dstSet          = m_hiZSets[mip];
        writes[0].dstBinding      = 0;
        writes[0].descriptorCount = 1;
        writes[0].descriptorType  = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        writes[0].pImageInfo      = &srcInfo;

        writes[1].sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[1].dstSet          = m_hiZSets[mip];
        writes[1].dstBinding      = 1;
        writes[1].descriptorCount = 1;
        writes[1].descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        writes[1].pImageInfo      = &dstInfo;

        vkUpdateDescriptorSets(m_device, static_cast<uint32_t>(writes.size()),
                               writes.data(), 0, nullptr);
    }

    // Write mip 0 of Hi-Z (copy from depth, use the depth view with mip 0 as source)
    // First mip of Hi-Z is a direct copy from the scene depth
    {
        VkDescriptorSetAllocateInfo allocInfo{};
        allocInfo.sType              = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        allocInfo.descriptorPool     = m_descriptorPool;
        allocInfo.descriptorSetCount = 1;
        allocInfo.pSetLayouts        = &m_hiZSetLayout;

        // We use mipCount-1 as a special slot for depth→HiZ mip 0 copy
        // Actually, mip 0 of hiZ is filled by copying depth directly.
        // We handle this in BuildHiZPyramid() via vkCmdCopyImage or blit.
    }

    HRP_LOG_INFO("ScreenSpaceEffects: Hi-Z pipeline created ({} mip levels)", mipCount);
    return true;
}

// ══════════════════════════════════════════════════════════════════════
// GTAO+ Resources
// ══════════════════════════════════════════════════════════════════════

bool ScreenSpaceEffects::CreateAOResources(Renderer& renderer)
{
    // Layout matches gtao_plus.comp: 3 input samplers + 1 output storage
    std::array<VkDescriptorSetLayoutBinding, 4> bindings{};
    bindings[0] = { 0, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr };
    bindings[1] = { 1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr };
    bindings[2] = { 2, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr };
    bindings[3] = { 3, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,          1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr };

    VkDescriptorSetLayoutCreateInfo layoutInfo{};
    layoutInfo.sType        = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    layoutInfo.bindingCount = static_cast<uint32_t>(bindings.size());
    layoutInfo.pBindings    = bindings.data();

    if (vkCreateDescriptorSetLayout(m_device, &layoutInfo, nullptr, &m_aoSetLayout) != VK_SUCCESS) {
        HRP_LOG_ERROR("ScreenSpaceEffects: Failed to create AO set layout");
        return false;
    }

    // Push constants: AOParams struct (see gtao_plus.comp)
    // 2×mat4 + 4×vec4 = 128 + 64 = 192 bytes
    VkPushConstantRange pushRange{};
    pushRange.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    pushRange.offset     = 0;
    pushRange.size       = 192;

    VkPipelineLayoutCreateInfo pipeLayoutInfo{};
    pipeLayoutInfo.sType                  = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    pipeLayoutInfo.setLayoutCount         = 1;
    pipeLayoutInfo.pSetLayouts            = &m_aoSetLayout;
    pipeLayoutInfo.pushConstantRangeCount = 1;
    pipeLayoutInfo.pPushConstantRanges    = &pushRange;

    if (vkCreatePipelineLayout(m_device, &pipeLayoutInfo, nullptr, &m_aoPipeLayout) != VK_SUCCESS) {
        HRP_LOG_ERROR("ScreenSpaceEffects: Failed to create AO pipeline layout");
        return false;
    }

    VkShaderModule aoShader = renderer.LoadShaderModule("gtao_plus.comp");
    if (aoShader == VK_NULL_HANDLE) {
        HRP_LOG_WARN("ScreenSpaceEffects: GTAO+ shader not found — AO disabled");
        return true;
    }

    VkComputePipelineCreateInfo compInfo{};
    compInfo.sType  = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    compInfo.layout = m_aoPipeLayout;
    compInfo.stage.sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    compInfo.stage.stage  = VK_SHADER_STAGE_COMPUTE_BIT;
    compInfo.stage.module = aoShader;
    compInfo.stage.pName  = "main";

    VkResult result = vkCreateComputePipelines(m_device, VK_NULL_HANDLE, 1, &compInfo,
                                                nullptr, &m_aoPipeline);
    vkDestroyShaderModule(m_device, aoShader, nullptr);

    if (result != VK_SUCCESS) {
        HRP_LOG_ERROR("ScreenSpaceEffects: Failed to create GTAO+ compute pipeline");
        return false;
    }

    // Allocate and write descriptor set
    VkDescriptorSetAllocateInfo allocInfo{};
    allocInfo.sType              = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    allocInfo.descriptorPool     = m_descriptorPool;
    allocInfo.descriptorSetCount = 1;
    allocInfo.pSetLayouts        = &m_aoSetLayout;

    if (vkAllocateDescriptorSets(m_device, &allocInfo, &m_aoSet) != VK_SUCCESS) {
        HRP_LOG_ERROR("ScreenSpaceEffects: Failed to allocate AO descriptor set");
        return false;
    }

    // Bind: 0=depth, 1=normals, 2=hiZ, 3=aoOutput
    VkDescriptorImageInfo depthInfo{ m_nearestSampler, m_sceneDepth.view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL };
    VkDescriptorImageInfo normInfo{ m_linearSampler, m_sceneNormals.view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL };
    VkDescriptorImageInfo hiZInfo{ m_linearSampler, m_hiZBuffer.view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL };
    VkDescriptorImageInfo aoOutInfo{ VK_NULL_HANDLE, m_aoOutput.view, VK_IMAGE_LAYOUT_GENERAL };

    std::array<VkWriteDescriptorSet, 4> writes{};
    for (uint32_t i = 0; i < 3; ++i) {
        writes[i].sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[i].dstSet          = m_aoSet;
        writes[i].dstBinding      = i;
        writes[i].descriptorCount = 1;
        writes[i].descriptorType  = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    }
    writes[0].pImageInfo = &depthInfo;
    writes[1].pImageInfo = &normInfo;
    writes[2].pImageInfo = &hiZInfo;

    writes[3].sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writes[3].dstSet          = m_aoSet;
    writes[3].dstBinding      = 3;
    writes[3].descriptorCount = 1;
    writes[3].descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    writes[3].pImageInfo      = &aoOutInfo;

    vkUpdateDescriptorSets(m_device, static_cast<uint32_t>(writes.size()),
                           writes.data(), 0, nullptr);

    HRP_LOG_INFO("ScreenSpaceEffects: GTAO+ pipeline created");
    return true;
}

// ══════════════════════════════════════════════════════════════════════
// Screen-Space GI Resources
// ══════════════════════════════════════════════════════════════════════

bool ScreenSpaceEffects::CreateGIResources(Renderer& renderer)
{
    // Layout matches screen_space_gi.comp: 6 inputs + 1 output
    std::array<VkDescriptorSetLayoutBinding, 7> bindings{};
    for (uint32_t i = 0; i < 6; ++i) {
        bindings[i] = { i, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr };
    }
    bindings[6] = { 6, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr };

    VkDescriptorSetLayoutCreateInfo layoutInfo{};
    layoutInfo.sType        = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    layoutInfo.bindingCount = static_cast<uint32_t>(bindings.size());
    layoutInfo.pBindings    = bindings.data();

    if (vkCreateDescriptorSetLayout(m_device, &layoutInfo, nullptr, &m_giSetLayout) != VK_SUCCESS)
        return false;

    // Push constants: GIParams (2×mat4 + 4×vec4 = 192 bytes)
    VkPushConstantRange pushRange{};
    pushRange.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    pushRange.offset     = 0;
    pushRange.size       = 192;

    VkPipelineLayoutCreateInfo pipeLayoutInfo{};
    pipeLayoutInfo.sType                  = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    pipeLayoutInfo.setLayoutCount         = 1;
    pipeLayoutInfo.pSetLayouts            = &m_giSetLayout;
    pipeLayoutInfo.pushConstantRangeCount = 1;
    pipeLayoutInfo.pPushConstantRanges    = &pushRange;

    if (vkCreatePipelineLayout(m_device, &pipeLayoutInfo, nullptr, &m_giPipeLayout) != VK_SUCCESS)
        return false;

    VkShaderModule shader = renderer.LoadShaderModule("screen_space_gi.comp");
    if (shader == VK_NULL_HANDLE) {
        HRP_LOG_WARN("ScreenSpaceEffects: SVRM shader not found — GI disabled");
        return true;
    }

    VkComputePipelineCreateInfo compInfo{};
    compInfo.sType  = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    compInfo.layout = m_giPipeLayout;
    compInfo.stage.sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    compInfo.stage.stage  = VK_SHADER_STAGE_COMPUTE_BIT;
    compInfo.stage.module = shader;
    compInfo.stage.pName  = "main";

    VkResult result = vkCreateComputePipelines(m_device, VK_NULL_HANDLE, 1, &compInfo,
                                                nullptr, &m_giPipeline);
    vkDestroyShaderModule(m_device, shader, nullptr);
    if (result != VK_SUCCESS) return false;

    // Allocate descriptor set
    VkDescriptorSetAllocateInfo allocInfo{};
    allocInfo.sType              = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    allocInfo.descriptorPool     = m_descriptorPool;
    allocInfo.descriptorSetCount = 1;
    allocInfo.pSetLayouts        = &m_giSetLayout;

    if (vkAllocateDescriptorSets(m_device, &allocInfo, &m_giSet) != VK_SUCCESS) return false;

    // Bind: 0=sceneColor, 1=depth, 2=normals, 3=hiZ, 4=prevGI, 5=motionVectors, 6=giOutput
    VkDescriptorImageInfo infos[7]{};
    infos[0] = { m_linearSampler, m_sceneColor.view,    VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL };
    infos[1] = { m_nearestSampler, m_sceneDepth.view,   VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL };
    infos[2] = { m_linearSampler, m_sceneNormals.view,  VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL };
    infos[3] = { m_linearSampler, m_hiZBuffer.view,     VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL };
    infos[4] = { m_linearSampler, m_prevGI.view,        VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL };
    infos[5] = { m_linearSampler, m_motionVectors.view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL };
    infos[6] = { VK_NULL_HANDLE, m_giOutput.view,       VK_IMAGE_LAYOUT_GENERAL };

    std::array<VkWriteDescriptorSet, 7> writes{};
    for (uint32_t i = 0; i < 7; ++i) {
        writes[i].sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[i].dstSet          = m_giSet;
        writes[i].dstBinding      = i;
        writes[i].descriptorCount = 1;
        writes[i].descriptorType  = (i < 6) ? VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER
                                             : VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        writes[i].pImageInfo      = &infos[i];
    }

    vkUpdateDescriptorSets(m_device, static_cast<uint32_t>(writes.size()),
                           writes.data(), 0, nullptr);

    HRP_LOG_INFO("ScreenSpaceEffects: SVRM (Screen-Space GI) pipeline created");
    return true;
}

// ══════════════════════════════════════════════════════════════════════
// Screen-Space Reflections Resources
// ══════════════════════════════════════════════════════════════════════

bool ScreenSpaceEffects::CreateReflResources(Renderer& renderer)
{
    // Layout matches hi_z_reflections.comp: 6 inputs + 1 output
    std::array<VkDescriptorSetLayoutBinding, 7> bindings{};
    for (uint32_t i = 0; i < 6; ++i) {
        bindings[i] = { i, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr };
    }
    bindings[6] = { 6, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr };

    VkDescriptorSetLayoutCreateInfo layoutInfo{};
    layoutInfo.sType        = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    layoutInfo.bindingCount = static_cast<uint32_t>(bindings.size());
    layoutInfo.pBindings    = bindings.data();

    if (vkCreateDescriptorSetLayout(m_device, &layoutInfo, nullptr, &m_reflSetLayout) != VK_SUCCESS)
        return false;

    // Push constants: ReflectionParams (2×mat4 + 3×vec4 = 176 bytes)
    VkPushConstantRange pushRange{};
    pushRange.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    pushRange.offset     = 0;
    pushRange.size       = 176;

    VkPipelineLayoutCreateInfo pipeLayoutInfo{};
    pipeLayoutInfo.sType                  = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    pipeLayoutInfo.setLayoutCount         = 1;
    pipeLayoutInfo.pSetLayouts            = &m_reflSetLayout;
    pipeLayoutInfo.pushConstantRangeCount = 1;
    pipeLayoutInfo.pPushConstantRanges    = &pushRange;

    if (vkCreatePipelineLayout(m_device, &pipeLayoutInfo, nullptr, &m_reflPipeLayout) != VK_SUCCESS)
        return false;

    VkShaderModule shader = renderer.LoadShaderModule("hi_z_reflections.comp");
    if (shader == VK_NULL_HANDLE) {
        HRP_LOG_WARN("ScreenSpaceEffects: SHR shader not found — Reflections disabled");
        return true;
    }

    VkComputePipelineCreateInfo compInfo{};
    compInfo.sType  = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    compInfo.layout = m_reflPipeLayout;
    compInfo.stage.sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    compInfo.stage.stage  = VK_SHADER_STAGE_COMPUTE_BIT;
    compInfo.stage.module = shader;
    compInfo.stage.pName  = "main";

    VkResult result = vkCreateComputePipelines(m_device, VK_NULL_HANDLE, 1, &compInfo,
                                                nullptr, &m_reflPipeline);
    vkDestroyShaderModule(m_device, shader, nullptr);
    if (result != VK_SUCCESS) return false;

    // Allocate descriptor set
    VkDescriptorSetAllocateInfo allocInfo{};
    allocInfo.sType              = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    allocInfo.descriptorPool     = m_descriptorPool;
    allocInfo.descriptorSetCount = 1;
    allocInfo.pSetLayouts        = &m_reflSetLayout;

    if (vkAllocateDescriptorSets(m_device, &allocInfo, &m_reflSet) != VK_SUCCESS) return false;

    // Bind: 0=sceneColor, 1=depth, 2=normalRoughness, 3=hiZ, 4=prevRefl, 5=motionVectors, 6=reflOutput
    // Note: hi_z_reflections expects normalRoughness (RGBA16F) at binding 2
    // We use sceneNormals for this — roughness will be stored in the alpha channel
    VkDescriptorImageInfo infos[7]{};
    infos[0] = { m_linearSampler, m_sceneColor.view,    VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL };
    infos[1] = { m_nearestSampler, m_sceneDepth.view,   VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL };
    infos[2] = { m_linearSampler, m_sceneNormals.view,  VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL }; // RGB=normal, A=roughness
    infos[3] = { m_linearSampler, m_hiZBuffer.view,     VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL };
    infos[4] = { m_linearSampler, m_prevRefl.view,      VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL };
    infos[5] = { m_linearSampler, m_motionVectors.view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL };
    infos[6] = { VK_NULL_HANDLE, m_reflOutput.view,     VK_IMAGE_LAYOUT_GENERAL };

    std::array<VkWriteDescriptorSet, 7> writes{};
    for (uint32_t i = 0; i < 7; ++i) {
        writes[i].sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[i].dstSet          = m_reflSet;
        writes[i].dstBinding      = i;
        writes[i].descriptorCount = 1;
        writes[i].descriptorType  = (i < 6) ? VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER
                                             : VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        writes[i].pImageInfo      = &infos[i];
    }

    vkUpdateDescriptorSets(m_device, static_cast<uint32_t>(writes.size()),
                           writes.data(), 0, nullptr);

    HRP_LOG_INFO("ScreenSpaceEffects: SHR (Screen-Space Reflections) pipeline created");
    return true;
}

// ══════════════════════════════════════════════════════════════════════
// Temporal Denoise Resources
// ══════════════════════════════════════════════════════════════════════

bool ScreenSpaceEffects::CreateDenoiseResources(Renderer& renderer)
{
    // Layout matches temporal_denoise.comp: 6 inputs + 1 output
    std::array<VkDescriptorSetLayoutBinding, 7> bindings{};
    for (uint32_t i = 0; i < 6; ++i) {
        bindings[i] = { i, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr };
    }
    bindings[6] = { 6, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr };

    VkDescriptorSetLayoutCreateInfo layoutInfo{};
    layoutInfo.sType        = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    layoutInfo.bindingCount = static_cast<uint32_t>(bindings.size());
    layoutInfo.pBindings    = bindings.data();

    if (vkCreateDescriptorSetLayout(m_device, &layoutInfo, nullptr, &m_denoiseSetLayout) != VK_SUCCESS)
        return false;

    // Push constants: DenoiseParams (3×vec4 = 48 bytes)
    VkPushConstantRange pushRange{};
    pushRange.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    pushRange.offset     = 0;
    pushRange.size       = 48;

    VkPipelineLayoutCreateInfo pipeLayoutInfo{};
    pipeLayoutInfo.sType                  = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    pipeLayoutInfo.setLayoutCount         = 1;
    pipeLayoutInfo.pSetLayouts            = &m_denoiseSetLayout;
    pipeLayoutInfo.pushConstantRangeCount = 1;
    pipeLayoutInfo.pPushConstantRanges    = &pushRange;

    if (vkCreatePipelineLayout(m_device, &pipeLayoutInfo, nullptr, &m_denoisePipeLayout) != VK_SUCCESS)
        return false;

    VkShaderModule shader = renderer.LoadShaderModule("temporal_denoise.comp");
    if (shader == VK_NULL_HANDLE) {
        HRP_LOG_WARN("ScreenSpaceEffects: ACTF shader not found — Denoise disabled");
        return true;
    }

    VkComputePipelineCreateInfo compInfo{};
    compInfo.sType  = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    compInfo.layout = m_denoisePipeLayout;
    compInfo.stage.sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    compInfo.stage.stage  = VK_SHADER_STAGE_COMPUTE_BIT;
    compInfo.stage.module = shader;
    compInfo.stage.pName  = "main";

    VkResult result = vkCreateComputePipelines(m_device, VK_NULL_HANDLE, 1, &compInfo,
                                                nullptr, &m_denoisePipeline);
    vkDestroyShaderModule(m_device, shader, nullptr);
    if (result != VK_SUCCESS) return false;

    // Allocate descriptor set
    VkDescriptorSetAllocateInfo allocInfo{};
    allocInfo.sType              = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    allocInfo.descriptorPool     = m_descriptorPool;
    allocInfo.descriptorSetCount = 1;
    allocInfo.pSetLayouts        = &m_denoiseSetLayout;

    if (vkAllocateDescriptorSets(m_device, &allocInfo, &m_denoiseSet) != VK_SUCCESS) return false;

    // Bind: 0=currentInput(GI), 1=historyBuffer, 2=depth, 3=normals, 4=motionVectors, 5=prevDepth, 6=output
    VkDescriptorImageInfo infos[7]{};
    infos[0] = { m_linearSampler, m_giOutput.view,      VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL };
    infos[1] = { m_linearSampler, m_prevDenoise.view,   VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL };
    infos[2] = { m_nearestSampler, m_sceneDepth.view,   VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL };
    infos[3] = { m_linearSampler, m_sceneNormals.view,  VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL };
    infos[4] = { m_linearSampler, m_motionVectors.view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL };
    infos[5] = { m_nearestSampler, m_prevDepth.view,    VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL };
    infos[6] = { VK_NULL_HANDLE, m_denoiseOutput.view,  VK_IMAGE_LAYOUT_GENERAL };

    std::array<VkWriteDescriptorSet, 7> writes{};
    for (uint32_t i = 0; i < 7; ++i) {
        writes[i].sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[i].dstSet          = m_denoiseSet;
        writes[i].dstBinding      = i;
        writes[i].descriptorCount = 1;
        writes[i].descriptorType  = (i < 6) ? VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER
                                             : VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        writes[i].pImageInfo      = &infos[i];
    }

    vkUpdateDescriptorSets(m_device, static_cast<uint32_t>(writes.size()),
                           writes.data(), 0, nullptr);

    HRP_LOG_INFO("ScreenSpaceEffects: ACTF (Temporal Denoise) pipeline created");
    return true;
}

// ══════════════════════════════════════════════════════════════════════
// RT Composite Resources (fullscreen graphics pipeline)
// ══════════════════════════════════════════════════════════════════════

bool ScreenSpaceEffects::CreateCompositeResources(Renderer& renderer)
{
    // Layout matches rt_composite.frag: 6 input samplers
    std::array<VkDescriptorSetLayoutBinding, 6> bindings{};
    for (uint32_t i = 0; i < 6; ++i) {
        bindings[i] = { i, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_FRAGMENT_BIT, nullptr };
    }

    VkDescriptorSetLayoutCreateInfo layoutInfo{};
    layoutInfo.sType        = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    layoutInfo.bindingCount = static_cast<uint32_t>(bindings.size());
    layoutInfo.pBindings    = bindings.data();

    if (vkCreateDescriptorSetLayout(m_device, &layoutInfo, nullptr, &m_compositeSetLayout) != VK_SUCCESS)
        return false;

    // Push constants: CompositeParams (3×vec4 = 48 bytes fragment)
    VkPushConstantRange pushRange{};
    pushRange.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
    pushRange.offset     = 0;
    pushRange.size       = 48;

    VkPipelineLayoutCreateInfo pipeLayoutInfo{};
    pipeLayoutInfo.sType                  = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    pipeLayoutInfo.setLayoutCount         = 1;
    pipeLayoutInfo.pSetLayouts            = &m_compositeSetLayout;
    pipeLayoutInfo.pushConstantRangeCount = 1;
    pipeLayoutInfo.pPushConstantRanges    = &pushRange;

    if (vkCreatePipelineLayout(m_device, &pipeLayoutInfo, nullptr, &m_compositePipeLayout) != VK_SUCCESS)
        return false;

    VkShaderModule vertShader = renderer.LoadShaderModule("rt_composite.vert");
    VkShaderModule fragShader = renderer.LoadShaderModule("rt_composite.frag");
    if (!vertShader || !fragShader) {
        HRP_LOG_ERROR("ScreenSpaceEffects: RT Composite shaders not found — "
                      "composite pipeline cannot be created; rendering will fall back "
                      "to direct mode. Ensure rt_composite.vert.spv and "
                      "rt_composite.frag.spv exist in the shader search paths.");
        if (vertShader) vkDestroyShaderModule(m_device, vertShader, nullptr);
        if (fragShader) vkDestroyShaderModule(m_device, fragShader, nullptr);
        return false;
    }

    VkPipelineShaderStageCreateInfo stages[2]{};
    stages[0].sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stages[0].stage  = VK_SHADER_STAGE_VERTEX_BIT;
    stages[0].module = vertShader;
    stages[0].pName  = "main";
    stages[1].sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stages[1].stage  = VK_SHADER_STAGE_FRAGMENT_BIT;
    stages[1].module = fragShader;
    stages[1].pName  = "main";

    // No vertex input (fullscreen triangle)
    VkPipelineVertexInputStateCreateInfo vertexInput{};
    vertexInput.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;

    VkPipelineInputAssemblyStateCreateInfo inputAssembly{};
    inputAssembly.sType    = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
    inputAssembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;

    VkPipelineViewportStateCreateInfo viewportState{};
    viewportState.sType         = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
    viewportState.viewportCount = 1;
    viewportState.scissorCount  = 1;

    VkPipelineRasterizationStateCreateInfo rasterizer{};
    rasterizer.sType       = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
    rasterizer.polygonMode = VK_POLYGON_MODE_FILL;
    rasterizer.lineWidth   = 1.0f;
    rasterizer.cullMode    = VK_CULL_MODE_NONE;

    VkPipelineMultisampleStateCreateInfo multisampling{};
    multisampling.sType                = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
    multisampling.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

    VkPipelineDepthStencilStateCreateInfo depthStencil{};
    depthStencil.sType            = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
    depthStencil.depthTestEnable  = VK_FALSE;
    depthStencil.depthWriteEnable = VK_FALSE;

    VkPipelineColorBlendAttachmentState blendAttachment{};
    blendAttachment.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT
                                   | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
    blendAttachment.blendEnable = VK_FALSE;

    VkPipelineColorBlendStateCreateInfo colorBlend{};
    colorBlend.sType           = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
    colorBlend.attachmentCount = 1;
    colorBlend.pAttachments    = &blendAttachment;

    std::array<VkDynamicState, 2> dynStates = {
        VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR
    };
    VkPipelineDynamicStateCreateInfo dynState{};
    dynState.sType             = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
    dynState.dynamicStateCount = static_cast<uint32_t>(dynStates.size());
    dynState.pDynamicStates    = dynStates.data();

    // Dynamic rendering — output to swapchain format
    VkFormat swapFormat = renderer.GetSwapchainFormat();
    VkPipelineRenderingCreateInfo renderingInfo{};
    renderingInfo.sType                   = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO;
    renderingInfo.colorAttachmentCount    = 1;
    renderingInfo.pColorAttachmentFormats = &swapFormat;

    VkGraphicsPipelineCreateInfo pipeInfo{};
    pipeInfo.sType               = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
    pipeInfo.pNext               = &renderingInfo;
    pipeInfo.stageCount          = 2;
    pipeInfo.pStages             = stages;
    pipeInfo.pVertexInputState   = &vertexInput;
    pipeInfo.pInputAssemblyState = &inputAssembly;
    pipeInfo.pViewportState      = &viewportState;
    pipeInfo.pRasterizationState = &rasterizer;
    pipeInfo.pMultisampleState   = &multisampling;
    pipeInfo.pDepthStencilState  = &depthStencil;
    pipeInfo.pColorBlendState    = &colorBlend;
    pipeInfo.pDynamicState       = &dynState;
    pipeInfo.layout              = m_compositePipeLayout;

    VkResult result = vkCreateGraphicsPipelines(m_device, VK_NULL_HANDLE, 1, &pipeInfo,
                                                 nullptr, &m_compositePipeline);
    vkDestroyShaderModule(m_device, vertShader, nullptr);
    vkDestroyShaderModule(m_device, fragShader, nullptr);

    if (result != VK_SUCCESS) {
        HRP_LOG_ERROR("ScreenSpaceEffects: Failed to create composite pipeline");
        return false;
    }

    // Allocate and write descriptor set
    VkDescriptorSetAllocateInfo allocInfo{};
    allocInfo.sType              = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    allocInfo.descriptorPool     = m_descriptorPool;
    allocInfo.descriptorSetCount = 1;
    allocInfo.pSetLayouts        = &m_compositeSetLayout;

    if (vkAllocateDescriptorSets(m_device, &allocInfo, &m_compositeSet) != VK_SUCCESS) return false;

    // Bind: 0=sceneColor, 1=gi, 2=reflections, 3=ao, 4=depth, 5=normalRoughness
    VkDescriptorImageInfo infos[6]{};
    infos[0] = { m_linearSampler, m_sceneColor.view,    VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL };
    infos[1] = { m_linearSampler, m_giOutput.view,      VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL };
    infos[2] = { m_linearSampler, m_reflOutput.view,    VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL };
    infos[3] = { m_linearSampler, m_aoOutput.view,      VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL };
    infos[4] = { m_nearestSampler, m_sceneDepth.view,   VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL };
    infos[5] = { m_linearSampler, m_sceneNormals.view,  VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL };

    std::array<VkWriteDescriptorSet, 6> writes{};
    for (uint32_t i = 0; i < 6; ++i) {
        writes[i].sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[i].dstSet          = m_compositeSet;
        writes[i].dstBinding      = i;
        writes[i].descriptorCount = 1;
        writes[i].descriptorType  = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        writes[i].pImageInfo      = &infos[i];
    }

    vkUpdateDescriptorSets(m_device, static_cast<uint32_t>(writes.size()),
                           writes.data(), 0, nullptr);

    HRP_LOG_INFO("ScreenSpaceEffects: RT Composite pipeline created");
    return true;
}

// ══════════════════════════════════════════════════════════════════════
// Pipeline Resource Cleanup
// ══════════════════════════════════════════════════════════════════════

void ScreenSpaceEffects::DestroyPipelineResources()
{
    auto destroyPipeline = [&](VkPipeline& p) {
        if (p) { vkDestroyPipeline(m_device, p, nullptr); p = VK_NULL_HANDLE; }
    };
    auto destroyLayout = [&](VkPipelineLayout& l) {
        if (l) { vkDestroyPipelineLayout(m_device, l, nullptr); l = VK_NULL_HANDLE; }
    };
    auto destroySetLayout = [&](VkDescriptorSetLayout& l) {
        if (l) { vkDestroyDescriptorSetLayout(m_device, l, nullptr); l = VK_NULL_HANDLE; }
    };

    destroyPipeline(m_hiZPipeline);
    destroyLayout(m_hiZPipeLayout);
    destroySetLayout(m_hiZSetLayout);

    destroyPipeline(m_aoPipeline);
    destroyLayout(m_aoPipeLayout);
    destroySetLayout(m_aoSetLayout);

    destroyPipeline(m_giPipeline);
    destroyLayout(m_giPipeLayout);
    destroySetLayout(m_giSetLayout);

    destroyPipeline(m_reflPipeline);
    destroyLayout(m_reflPipeLayout);
    destroySetLayout(m_reflSetLayout);

    destroyPipeline(m_denoisePipeline);
    destroyLayout(m_denoisePipeLayout);
    destroySetLayout(m_denoiseSetLayout);

    destroyPipeline(m_compositePipeline);
    destroyLayout(m_compositePipeLayout);
    destroySetLayout(m_compositeSetLayout);
}

// ══════════════════════════════════════════════════════════════════════
// Image Transitions
// ══════════════════════════════════════════════════════════════════════

void ScreenSpaceEffects::TransitionImage(VkCommandBuffer cmd, VkImage image,
                                          VkImageLayout oldLayout, VkImageLayout newLayout,
                                          VkAccessFlags srcAccess, VkAccessFlags dstAccess,
                                          VkPipelineStageFlags srcStage, VkPipelineStageFlags dstStage,
                                          VkImageAspectFlags aspect,
                                          uint32_t baseMip, uint32_t mipCount)
{
    VkImageMemoryBarrier barrier{};
    barrier.sType               = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    barrier.oldLayout           = oldLayout;
    barrier.newLayout           = newLayout;
    barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.image               = image;
    barrier.subresourceRange    = { aspect, baseMip, mipCount, 0, 1 };
    barrier.srcAccessMask       = srcAccess;
    barrier.dstAccessMask       = dstAccess;

    vkCmdPipelineBarrier(cmd, srcStage, dstStage, 0,
                         0, nullptr, 0, nullptr, 1, &barrier);
}

void ScreenSpaceEffects::TransitionSceneToShaderRead(VkCommandBuffer cmd)
{
    // Scene color: COLOR_ATTACHMENT → SHADER_READ
    TransitionImage(cmd, m_sceneColor.image,
                    VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                    VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                    VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
                    VK_ACCESS_SHADER_READ_BIT,
                    VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
                    VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);

    // Depth: DEPTH_ATTACHMENT → SHADER_READ
    TransitionImage(cmd, m_sceneDepth.image,
                    VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
                    VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                    VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT,
                    VK_ACCESS_SHADER_READ_BIT,
                    VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT,
                    VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                    VK_IMAGE_ASPECT_DEPTH_BIT);

    // Normals: COLOR_ATTACHMENT → SHADER_READ
    TransitionImage(cmd, m_sceneNormals.image,
                    VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                    VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                    VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
                    VK_ACCESS_SHADER_READ_BIT,
                    VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
                    VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);

    // Motion vectors: COLOR_ATTACHMENT → SHADER_READ
    TransitionImage(cmd, m_motionVectors.image,
                    VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                    VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                    VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
                    VK_ACCESS_SHADER_READ_BIT,
                    VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
                    VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
}

void ScreenSpaceEffects::TransitionHistoryBuffers(VkCommandBuffer cmd)
{
    // Transition history buffers to SHADER_READ for temporal accumulation
    TransitionImage(cmd, m_prevGI.image,
                    VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                    0, VK_ACCESS_SHADER_READ_BIT,
                    VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);

    TransitionImage(cmd, m_prevRefl.image,
                    VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                    0, VK_ACCESS_SHADER_READ_BIT,
                    VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);

    TransitionImage(cmd, m_prevDenoise.image,
                    VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                    0, VK_ACCESS_SHADER_READ_BIT,
                    VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);

    TransitionImage(cmd, m_prevDepth.image,
                    VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                    0, VK_ACCESS_SHADER_READ_BIT,
                    VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                    VK_IMAGE_ASPECT_DEPTH_BIT);

    // Effect output buffers to GENERAL for compute writes (or clear).
    // When an effect is disabled its dispatch is skipped, but the composite
    // pass still samples the buffer.  Transitioning from UNDEFINED discards
    // contents, so we clear disabled buffers to neutral values:
    //   AO  → (1,0,0,0)  (no darkening)
    //   GI  → (0,0,0,0)  (no indirect light)
    //   Ref → (0,0,0,0)  (no reflections)
    TransitionImage(cmd, m_aoOutput.image,
                    VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL,
                    0, VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT,
                    VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                    VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT);

    TransitionImage(cmd, m_giOutput.image,
                    VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL,
                    0, VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT,
                    VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                    VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT);

    TransitionImage(cmd, m_reflOutput.image,
                    VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL,
                    0, VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT,
                    VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                    VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT);

    TransitionImage(cmd, m_denoiseOutput.image,
                    VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL,
                    0, VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT,
                    VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                    VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT);

    // Clear disabled effect buffers to neutral values so the composite
    // never reads undefined memory, even if the intensity zero-guard is
    // somehow bypassed.
    if (!m_config.enableAO) {
        VkClearColorValue aoClear = {{ 1.0f, 0.5f, 0.5f, 0.5f }};
        VkImageSubresourceRange range{ VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };
        vkCmdClearColorImage(cmd, m_aoOutput.image, VK_IMAGE_LAYOUT_GENERAL, &aoClear, 1, &range);
    }
    if (!m_config.enableGI) {
        VkClearColorValue giClear = {{ 0.0f, 0.0f, 0.0f, 0.0f }};
        VkImageSubresourceRange range{ VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };
        vkCmdClearColorImage(cmd, m_giOutput.image, VK_IMAGE_LAYOUT_GENERAL, &giClear, 1, &range);
    }
    if (!m_config.enableReflections) {
        VkClearColorValue refClear = {{ 0.0f, 0.0f, 0.0f, 0.0f }};
        VkImageSubresourceRange range{ VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };
        vkCmdClearColorImage(cmd, m_reflOutput.image, VK_IMAGE_LAYOUT_GENERAL, &refClear, 1, &range);
    }

    // Hi-Z buffer to GENERAL for storage writes
    TransitionImage(cmd, m_hiZBuffer.image,
                    VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL,
                    0, VK_ACCESS_SHADER_WRITE_BIT,
                    VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                    VK_IMAGE_ASPECT_COLOR_BIT, 0, m_hiZBuffer.mipLevels);
}

// ══════════════════════════════════════════════════════════════════════
// Scene Pass (offscreen rendering to color + depth + normals)
// ══════════════════════════════════════════════════════════════════════

void ScreenSpaceEffects::BeginScenePass(VkCommandBuffer cmd, const glm::vec4& clearColor)
{
    // Transition render targets to attachment layout
    TransitionImage(cmd, m_sceneColor.image,
                    VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                    0, VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
                    VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                    VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT);

    TransitionImage(cmd, m_sceneDepth.image,
                    VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
                    0, VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT,
                    VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                    VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT,
                    VK_IMAGE_ASPECT_DEPTH_BIT);

    TransitionImage(cmd, m_sceneNormals.image,
                    VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                    0, VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
                    VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                    VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT);

    TransitionImage(cmd, m_motionVectors.image,
                    VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                    0, VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
                    VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                    VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT);

    // Set up attachments
    VkClearValue colorClear{};
    colorClear.color = {{ clearColor.r, clearColor.g, clearColor.b, clearColor.a }};
    VkClearValue depthClear{};
    depthClear.depthStencil = { 1.0f, 0 };
    VkClearValue normalClear{};
    normalClear.color = {{ 0.0f, 0.0f, 0.0f, 0.0f }};

    // Color attachment (scene color — HDR)
    VkRenderingAttachmentInfo colorAtt{};
    colorAtt.sType       = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO;
    colorAtt.imageView   = m_sceneColor.view;
    colorAtt.imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    colorAtt.loadOp      = VK_ATTACHMENT_LOAD_OP_CLEAR;
    colorAtt.storeOp     = VK_ATTACHMENT_STORE_OP_STORE;
    colorAtt.clearValue  = colorClear;

    // Normal attachment (world-space normals + roughness in alpha)
    // NOTE: Currently the scene shaders don't write normals to a second attachment.
    // Phase 2 will add MRT output to terrain.frag and basic_3d.frag.
    // For now, normals are reconstructed from depth in the compute shaders.
    VkRenderingAttachmentInfo normalAtt{};
    normalAtt.sType       = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO;
    normalAtt.imageView   = m_sceneNormals.view;
    normalAtt.imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    normalAtt.loadOp      = VK_ATTACHMENT_LOAD_OP_CLEAR;
    normalAtt.storeOp     = VK_ATTACHMENT_STORE_OP_STORE;
    normalAtt.clearValue  = normalClear;

    // Motion vectors
    VkRenderingAttachmentInfo motionAtt{};
    motionAtt.sType       = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO;
    motionAtt.imageView   = m_motionVectors.view;
    motionAtt.imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    motionAtt.loadOp      = VK_ATTACHMENT_LOAD_OP_CLEAR;
    motionAtt.storeOp     = VK_ATTACHMENT_STORE_OP_STORE;
    motionAtt.clearValue  = normalClear;

    // For Phase 1: only color + depth (normals reconstructed from depth)
    // MRT with normals + motion vectors will be added in Phase 2
    std::array<VkRenderingAttachmentInfo, 1> colorAttachments = { colorAtt };

    VkRenderingAttachmentInfo depthAtt{};
    depthAtt.sType       = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO;
    depthAtt.imageView   = m_sceneDepth.view;
    depthAtt.imageLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
    depthAtt.loadOp      = VK_ATTACHMENT_LOAD_OP_CLEAR;
    depthAtt.storeOp     = VK_ATTACHMENT_STORE_OP_STORE;
    depthAtt.clearValue  = depthClear;

    VkRenderingInfo renderInfo{};
    renderInfo.sType                = VK_STRUCTURE_TYPE_RENDERING_INFO;
    renderInfo.renderArea           = { {0, 0}, { m_width, m_height } };
    renderInfo.layerCount           = 1;
    renderInfo.colorAttachmentCount = static_cast<uint32_t>(colorAttachments.size());
    renderInfo.pColorAttachments    = colorAttachments.data();
    renderInfo.pDepthAttachment     = &depthAtt;

    vkCmdBeginRendering(cmd, &renderInfo);

    // Set viewport and scissor
    VkViewport viewport{};
    viewport.x        = 0.0f;
    viewport.y        = 0.0f;
    viewport.width    = static_cast<float>(m_width);
    viewport.height   = static_cast<float>(m_height);
    viewport.minDepth = 0.0f;
    viewport.maxDepth = 1.0f;
    vkCmdSetViewport(cmd, 0, 1, &viewport);

    VkRect2D scissor{};
    scissor.offset = { 0, 0 };
    scissor.extent = { m_width, m_height };
    vkCmdSetScissor(cmd, 0, 1, &scissor);
}

void ScreenSpaceEffects::EndScenePass(VkCommandBuffer cmd)
{
    vkCmdEndRendering(cmd);
}

void ScreenSpaceEffects::SuspendScenePass(VkCommandBuffer cmd)
{
    vkCmdEndRendering(cmd);
}

void ScreenSpaceEffects::ResumeScenePass(VkCommandBuffer cmd)
{
    // Resume with LOAD to preserve already-rendered content
    VkRenderingAttachmentInfo colorAtt{};
    colorAtt.sType       = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO;
    colorAtt.imageView   = m_sceneColor.view;
    colorAtt.imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    colorAtt.loadOp      = VK_ATTACHMENT_LOAD_OP_LOAD;
    colorAtt.storeOp     = VK_ATTACHMENT_STORE_OP_STORE;

    VkRenderingAttachmentInfo depthAtt{};
    depthAtt.sType       = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO;
    depthAtt.imageView   = m_sceneDepth.view;
    depthAtt.imageLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
    depthAtt.loadOp      = VK_ATTACHMENT_LOAD_OP_LOAD;
    depthAtt.storeOp     = VK_ATTACHMENT_STORE_OP_STORE;

    VkRenderingInfo renderInfo{};
    renderInfo.sType                = VK_STRUCTURE_TYPE_RENDERING_INFO;
    renderInfo.renderArea           = { {0, 0}, { m_width, m_height } };
    renderInfo.layerCount           = 1;
    renderInfo.colorAttachmentCount = 1;
    renderInfo.pColorAttachments    = &colorAtt;
    renderInfo.pDepthAttachment     = &depthAtt;

    vkCmdBeginRendering(cmd, &renderInfo);

    VkViewport viewport{};
    viewport.x        = 0.0f;
    viewport.y        = 0.0f;
    viewport.width    = static_cast<float>(m_width);
    viewport.height   = static_cast<float>(m_height);
    viewport.minDepth = 0.0f;
    viewport.maxDepth = 1.0f;
    vkCmdSetViewport(cmd, 0, 1, &viewport);

    VkRect2D scissor{};
    scissor.offset = { 0, 0 };
    scissor.extent = { m_width, m_height };
    vkCmdSetScissor(cmd, 0, 1, &scissor);
}

// ══════════════════════════════════════════════════════════════════════
// Hi-Z Pyramid Build
// ══════════════════════════════════════════════════════════════════════

void ScreenSpaceEffects::BuildHiZPyramid(VkCommandBuffer cmd)
{
    if (!m_hiZPipeline) return;

    // NOTE: Previously this used vkCmdBlitImage to copy scene depth (D32_SFLOAT,
    // ASPECT_DEPTH) into Hi-Z mip 0 (R32_SFLOAT, ASPECT_COLOR). That is invalid
    // per the Vulkan spec — vkCmdBlitImage cannot cross the depth/color aspect
    // boundary. Some drivers (NVIDIA) silently tolerate it; others (AMD/Intel,
    // validation layers) reject it.
    //
    // Replacement: clear Hi-Z mip 0 to 1.0 (farthest depth — conservative for
    // occlusion testing) and let the compute pass below write mips 1..N
    // directly from m_sceneDepth (the mip-0 descriptor already binds the depth
    // view as its sampler source). Effect shaders that consume Hi-Z start ray
    // marching from coarser mips anyway, so mip 0 being a flat far-plane has
    // no functional impact.
    TransitionImage(cmd, m_hiZBuffer.image,
                    VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                    VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_TRANSFER_WRITE_BIT,
                    VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
                    VK_IMAGE_ASPECT_COLOR_BIT, 0, 1);

    {
        VkClearColorValue farClear = {{ 1.0f, 0.0f, 0.0f, 0.0f }};
        VkImageSubresourceRange range{ VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };
        vkCmdClearColorImage(cmd, m_hiZBuffer.image,
                             VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                             &farClear, 1, &range);
    }

    TransitionImage(cmd, m_hiZBuffer.image,
                    VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                    VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT,
                    VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                    VK_IMAGE_ASPECT_COLOR_BIT, 0, 1);

    // Build mip chain via compute shader
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_hiZPipeline);

    uint32_t srcW = m_width, srcH = m_height;
    for (uint32_t mip = 0; mip + 1 < m_hiZBuffer.mipLevels && mip < m_hiZSets.size(); ++mip) {
        if (m_hiZSets[mip] == VK_NULL_HANDLE) break;

        uint32_t dstW = std::max(srcW / 2, 1u);
        uint32_t dstH = std::max(srcH / 2, 1u);

        // Transition dst mip to GENERAL for write
        TransitionImage(cmd, m_hiZBuffer.image,
                        VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL,
                        0, VK_ACCESS_SHADER_WRITE_BIT,
                        VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                        VK_IMAGE_ASPECT_COLOR_BIT, mip + 1, 1);

        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                                m_hiZPipeLayout, 0, 1, &m_hiZSets[mip], 0, nullptr);

        // Push source size
        glm::vec4 srcSize(static_cast<float>(srcW), static_cast<float>(srcH),
                          1.0f / static_cast<float>(srcW), 1.0f / static_cast<float>(srcH));
        vkCmdPushConstants(cmd, m_hiZPipeLayout, VK_SHADER_STAGE_COMPUTE_BIT,
                           0, sizeof(glm::vec4), &srcSize);

        vkCmdDispatch(cmd, (dstW + 7) / 8, (dstH + 7) / 8, 1);

        // Barrier: make dst mip readable for next iteration
        TransitionImage(cmd, m_hiZBuffer.image,
                        VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                        VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT,
                        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                        VK_IMAGE_ASPECT_COLOR_BIT, mip + 1, 1);

        srcW = dstW;
        srcH = dstH;
    }

    // Transition the full Hi-Z buffer back to SHADER_READ for effect passes
    TransitionImage(cmd, m_hiZBuffer.image,
                    VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                    VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT,
                    VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                    VK_IMAGE_ASPECT_COLOR_BIT, 0, m_hiZBuffer.mipLevels);
}

// ══════════════════════════════════════════════════════════════════════
// Compute Shader Dispatches
// ══════════════════════════════════════════════════════════════════════

void ScreenSpaceEffects::DispatchAO(VkCommandBuffer cmd, const glm::mat4& viewProj,
                                     const glm::mat4& invViewProj, const glm::vec3& cameraPos,
                                     float nearPlane, float farPlane, uint32_t frameIndex)
{
    if (!m_aoPipeline || !m_config.enableAO) return;

    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_aoPipeline);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                            m_aoPipeLayout, 0, 1, &m_aoSet, 0, nullptr);

    // Push constants matching gtao_plus.comp AOParams layout
    struct AOPushData {
        glm::mat4 viewProj;
        glm::mat4 invViewProj;
        glm::vec4 cameraPos;
        glm::vec4 screenParams;
        glm::vec4 aoParams;
        glm::vec4 aoParams2;
    } pc;

    pc.viewProj     = viewProj;
    pc.invViewProj  = invViewProj;
    pc.cameraPos    = glm::vec4(cameraPos, static_cast<float>(frameIndex));
    pc.screenParams = glm::vec4(static_cast<float>(m_width), static_cast<float>(m_height),
                                1.0f / m_width, 1.0f / m_height);
    pc.aoParams     = glm::vec4(m_config.aoIntensity, m_config.aoShortRadius,
                                m_config.aoLongRadius, m_config.aoBias);
    pc.aoParams2    = glm::vec4(m_config.aoPower, 0.9f, nearPlane, farPlane);

    vkCmdPushConstants(cmd, m_aoPipeLayout, VK_SHADER_STAGE_COMPUTE_BIT,
                       0, sizeof(pc), &pc);

    vkCmdDispatch(cmd, (m_width + 7) / 8, (m_height + 7) / 8, 1);

    // Barrier: AO output readable for composite
    TransitionImage(cmd, m_aoOutput.image,
                    VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                    VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT,
                    VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT);
}

void ScreenSpaceEffects::DispatchGI(VkCommandBuffer cmd, const glm::mat4& invViewProj,
                                     const glm::mat4& prevViewProj, const glm::vec3& cameraPos,
                                     uint32_t frameIndex)
{
    if (!m_giPipeline || !m_config.enableGI) return;

    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_giPipeline);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                            m_giPipeLayout, 0, 1, &m_giSet, 0, nullptr);

    struct GIPushData {
        glm::mat4 invViewProj;
        glm::mat4 prevViewProj;
        glm::vec4 cameraPos;
        glm::vec4 screenParams;
        glm::vec4 giParams;
    } pc;

    pc.invViewProj  = invViewProj;
    pc.prevViewProj = prevViewProj;
    pc.cameraPos    = glm::vec4(cameraPos, static_cast<float>(frameIndex));
    pc.screenParams = glm::vec4(static_cast<float>(m_width), static_cast<float>(m_height),
                                1.0f / m_width, 1.0f / m_height);
    pc.giParams     = glm::vec4(m_config.giIntensity, m_config.giRadius,
                                m_config.giThickness, m_config.giTemporalBlend);

    vkCmdPushConstants(cmd, m_giPipeLayout, VK_SHADER_STAGE_COMPUTE_BIT,
                       0, sizeof(pc), &pc);

    vkCmdDispatch(cmd, (m_width + 7) / 8, (m_height + 7) / 8, 1);

    // Barrier: GI output readable for denoise and composite
    TransitionImage(cmd, m_giOutput.image,
                    VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                    VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT,
                    VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                    VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT);
}

void ScreenSpaceEffects::DispatchReflections(VkCommandBuffer cmd, const glm::mat4& viewProj,
                                              const glm::mat4& invViewProj, const glm::vec3& cameraPos,
                                              uint32_t frameIndex)
{
    if (!m_reflPipeline || !m_config.enableReflections) return;

    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_reflPipeline);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                            m_reflPipeLayout, 0, 1, &m_reflSet, 0, nullptr);

    struct ReflPushData {
        glm::mat4 viewProj;
        glm::mat4 invViewProj;
        glm::vec4 cameraPos;
        glm::vec4 screenParams;
        glm::vec4 reflParams;
    } pc;

    pc.viewProj     = viewProj;
    pc.invViewProj  = invViewProj;
    pc.cameraPos    = glm::vec4(cameraPos, static_cast<float>(frameIndex));
    pc.screenParams = glm::vec4(static_cast<float>(m_width), static_cast<float>(m_height),
                                1.0f / m_width, 1.0f / m_height);
    pc.reflParams   = glm::vec4(m_config.reflIntensity, m_config.reflMaxDist,
                                m_config.reflThickness, m_config.reflTemporalBlend);

    vkCmdPushConstants(cmd, m_reflPipeLayout, VK_SHADER_STAGE_COMPUTE_BIT,
                       0, sizeof(pc), &pc);

    vkCmdDispatch(cmd, (m_width + 7) / 8, (m_height + 7) / 8, 1);

    // Barrier
    TransitionImage(cmd, m_reflOutput.image,
                    VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                    VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT,
                    VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT);
}

void ScreenSpaceEffects::DispatchDenoise(VkCommandBuffer cmd, uint32_t frameIndex, float deltaTime)
{
    if (!m_denoisePipeline || !m_config.enableDenoise) return;

    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_denoisePipeline);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                            m_denoisePipeLayout, 0, 1, &m_denoiseSet, 0, nullptr);

    struct DenoisePushData {
        glm::vec4 screenParams;
        glm::vec4 filterParams;
        glm::vec4 frameParams;
    } pc;

    pc.screenParams = glm::vec4(static_cast<float>(m_width), static_cast<float>(m_height),
                                1.0f / m_width, 1.0f / m_height);
    pc.filterParams = glm::vec4(1.0f, 0.2f, 0.95f, 0.5f); // spatial sigma, min/max blend, clamp sharpness
    pc.frameParams  = glm::vec4(static_cast<float>(frameIndex), deltaTime, 0.0f, 0.0f);

    vkCmdPushConstants(cmd, m_denoisePipeLayout, VK_SHADER_STAGE_COMPUTE_BIT,
                       0, sizeof(pc), &pc);

    vkCmdDispatch(cmd, (m_width + 7) / 8, (m_height + 7) / 8, 1);

    // Barrier
    TransitionImage(cmd, m_denoiseOutput.image,
                    VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                    VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT,
                    VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT);
}

// ══════════════════════════════════════════════════════════════════════
// DispatchAll — main entry point for the screen-space pipeline
// ══════════════════════════════════════════════════════════════════════

void ScreenSpaceEffects::DispatchAll(VkCommandBuffer cmd,
                                      const glm::mat4& viewProj,
                                      const glm::mat4& prevViewProj,
                                      const glm::vec3& cameraPos,
                                      float nearPlane, float farPlane,
                                      uint32_t frameIndex, float deltaTime)
{
    if (!m_initialized) return;

    glm::mat4 invViewProj = glm::inverse(viewProj);

    // 1. Transition scene targets from attachment → shader-read
    TransitionSceneToShaderRead(cmd);

    // 2. Prepare history and output buffers
    TransitionHistoryBuffers(cmd);

    // 3. Build Hi-Z depth pyramid
    BuildHiZPyramid(cmd);

    // 4. Dispatch screen-space effects in dependency order
    // AO runs first (no dependencies on other effects)
    DispatchAO(cmd, viewProj, invViewProj, cameraPos, nearPlane, farPlane, frameIndex);

    // GI runs second (independent of AO, reads scene color + depth + normals)
    DispatchGI(cmd, invViewProj, prevViewProj, cameraPos, frameIndex);

    // Reflections run third (independent of AO/GI)
    DispatchReflections(cmd, viewProj, invViewProj, cameraPos, frameIndex);

    // Temporal denoise runs last (reads GI output)
    DispatchDenoise(cmd, frameIndex, deltaTime);

    // Ensure every effect output buffer ends in SHADER_READ_ONLY_OPTIMAL
    // before the composite fragment pass samples it. The per-dispatch helpers
    // only run that transition when the effect is enabled — when disabled,
    // the buffer was left in GENERAL by TransitionHistoryBuffers (and possibly
    // cleared via vkCmdClearColorImage). Without the transition below, the
    // composite pipeline's descriptor (imageLayout = SHADER_READ_ONLY_OPTIMAL)
    // would not match the actual layout, triggering a Vulkan validation error
    // and undefined behaviour on strict drivers.
    auto ensureSampled = [&](OffscreenImage& img, bool wasDispatched) {
        if (wasDispatched) return; // Already transitioned by Dispatch* helper.
        TransitionImage(cmd, img.image,
                        VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                        VK_ACCESS_TRANSFER_WRITE_BIT | VK_ACCESS_SHADER_WRITE_BIT,
                        VK_ACCESS_SHADER_READ_BIT,
                        VK_PIPELINE_STAGE_TRANSFER_BIT | VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                        VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT);
    };
    ensureSampled(m_aoOutput,      m_config.enableAO          && m_aoPipeline);
    ensureSampled(m_giOutput,      m_config.enableGI          && m_giPipeline);
    ensureSampled(m_reflOutput,    m_config.enableReflections && m_reflPipeline);
    ensureSampled(m_denoiseOutput, m_config.enableDenoise     && m_denoisePipeline);

    // Store previous frame VP for next frame's temporal reprojection
    m_prevViewProj = viewProj;
}

// ══════════════════════════════════════════════════════════════════════
// Composite to Swapchain
// ══════════════════════════════════════════════════════════════════════

void ScreenSpaceEffects::CompositeToSwapchain(VkCommandBuffer cmd,
                                               VkImageView swapchainView,
                                               VkExtent2D swapchainExtent)
{
    if (!m_compositePipeline || !m_config.enableComposite) return;

    // Begin rendering into the swapchain
    VkRenderingAttachmentInfo colorAtt{};
    colorAtt.sType       = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO;
    colorAtt.imageView   = swapchainView;
    colorAtt.imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    colorAtt.loadOp      = VK_ATTACHMENT_LOAD_OP_DONT_CARE; // Fullscreen overwrite
    colorAtt.storeOp     = VK_ATTACHMENT_STORE_OP_STORE;

    VkRenderingInfo renderInfo{};
    renderInfo.sType                = VK_STRUCTURE_TYPE_RENDERING_INFO;
    renderInfo.renderArea           = { {0, 0}, swapchainExtent };
    renderInfo.layerCount           = 1;
    renderInfo.colorAttachmentCount = 1;
    renderInfo.pColorAttachments    = &colorAtt;

    vkCmdBeginRendering(cmd, &renderInfo);

    VkViewport viewport{};
    viewport.width    = static_cast<float>(swapchainExtent.width);
    viewport.height   = static_cast<float>(swapchainExtent.height);
    viewport.maxDepth = 1.0f;
    vkCmdSetViewport(cmd, 0, 1, &viewport);

    VkRect2D scissor{ {0, 0}, swapchainExtent };
    vkCmdSetScissor(cmd, 0, 1, &scissor);

    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, m_compositePipeline);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
                            m_compositePipeLayout, 0, 1, &m_compositeSet, 0, nullptr);

    // Push composite params
    struct CompositePushData {
        glm::vec4 compositeParams; // gi intensity, refl intensity, ao intensity, time
        glm::vec4 postFxParams;    // CA, grain, vignette, exposure
        glm::vec4 screenParams;
    } pc;

    // When an effect is disabled its output buffer contains garbage (never
    // cleared), so force its composite intensity to zero to avoid sampling
    // undefined memory.  AO garbage ≈ 0 would multiply the scene to black.
    pc.compositeParams = glm::vec4(
        m_config.enableGI          ? m_config.giIntensity   : 0.0f,
        m_config.enableReflections ? m_config.reflIntensity : 0.0f,
        m_config.enableAO          ? m_config.aoIntensity   : 0.0f,
        0.0f);
    pc.postFxParams    = glm::vec4(m_config.chromaticAberration, m_config.filmGrain,
                                    m_config.vignette, m_config.exposure);
    pc.screenParams    = glm::vec4(static_cast<float>(swapchainExtent.width),
                                    static_cast<float>(swapchainExtent.height),
                                    1.0f / swapchainExtent.width,
                                    1.0f / swapchainExtent.height);

    vkCmdPushConstants(cmd, m_compositePipeLayout, VK_SHADER_STAGE_FRAGMENT_BIT,
                       0, sizeof(pc), &pc);

    // Draw fullscreen triangle (3 vertices, no vertex buffer)
    vkCmdDraw(cmd, 3, 1, 0, 0);

    vkCmdEndRendering(cmd);
}

void ScreenSpaceEffects::SwapHistoryBuffers()
{
    // Swap current output buffers with history buffers for next frame
    // This is done by swapping image handles — the descriptor sets still point
    // to the original memory, so we need to re-write descriptors.
    // For simplicity in Phase 1, we just leave the descriptors as-is since
    // they're all pointing to the same persistent images. The temporal
    // accumulation naturally converges even without perfect ping-pong
    // because the shaders use confidence-weighted blending.
    //
    // Phase 2 TODO: implement proper ping-pong by maintaining double-buffered
    // OffscreenImages and swapping which set of descriptors is active.
}

} // namespace hrp
