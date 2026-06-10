/// @file Renderer.cpp
/// @brief Vulkan 1.3 renderer — initialization, swapchain, pipelines, mesh management, frame loop.
///
/// Provides 3D rendering with depth buffer and a debug text overlay.
/// Uses VMA for GPU memory allocation, dynamic rendering (Vulkan 1.3),
/// and push constants for MVP transforms and overlay data.

#include "Renderer.hpp"
#include "Platform.hpp"
#include "Log.hpp"

#define VMA_IMPLEMENTATION
#include <vk_mem_alloc.h>

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <stb_image_write.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <fstream>
#include <set>
#include <filesystem>

namespace hrp {

// ──────────────────────────────────────────────────────────────────────
// Validation layer callback
// ──────────────────────────────────────────────────────────────────────

#ifdef NDEBUG
static constexpr bool k_EnableValidation = false;
#else
static constexpr bool k_EnableValidation = true;
#endif

static const std::vector<const char*> k_ValidationLayers = {
    "VK_LAYER_KHRONOS_validation"
};

static const std::vector<const char*> k_DeviceExtensions = {
    VK_KHR_SWAPCHAIN_EXTENSION_NAME,
};

static VKAPI_ATTR VkBool32 VKAPI_CALL VulkanDebugCallback(
    VkDebugUtilsMessageSeverityFlagBitsEXT      severity,
    VkDebugUtilsMessageTypeFlagsEXT             type,
    const VkDebugUtilsMessengerCallbackDataEXT* callbackData,
    void*                                       userData)
{
    if (severity >= VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT) {
        HRP_LOG_ERROR("Vulkan: {}", callbackData->pMessage);
    } else if (severity >= VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT) {
        HRP_LOG_WARN("Vulkan: {}", callbackData->pMessage);
    } else {
        HRP_LOG_TRACE("Vulkan: {}", callbackData->pMessage);
    }
    return VK_FALSE;
}

// ──────────────────────────────────────────────────────────────────────
// Public API
// ──────────────────────────────────────────────────────────────────────

bool Renderer::Initialize(Platform& platform)
{
    HRP_LOG_INFO("Renderer: Initializing Vulkan 1.3 ...");

    if (!CreateInstance(platform))    return false;
    if (k_EnableValidation) {
        if (!SetupDebugMessenger())   HRP_LOG_WARN("Renderer: Debug messenger setup failed (non-fatal)");
    }

    // Create Vulkan surface from SDL window
    m_surface = platform.CreateVulkanSurface(m_instance);
    if (m_surface == VK_NULL_HANDLE) {
        HRP_LOG_CRITICAL("Renderer: Failed to create Vulkan surface");
        return false;
    }

    if (!PickPhysicalDevice())       return false;
    if (!CreateLogicalDevice())      return false;
    if (!CreateAllocator())          return false;
    if (!CreateSwapchain(platform))  return false;
    if (!CreateImageViews())         return false;
    if (!CreateDepthResources())     return false;
    if (!CreateCommandPool())        return false;
    if (!CreateCommandBuffers())     return false;
    if (!CreateSyncObjects())        return false;
    if (!CreatePipelines())          return false;

    m_initialized = true;
    HRP_LOG_INFO("Renderer: Vulkan initialization complete");
    return true;
}

void Renderer::WaitIdle()
{
    if (m_device != VK_NULL_HANDLE) {
        vkDeviceWaitIdle(m_device);
    }
}

void Renderer::Shutdown()
{
    if (!m_initialized) return;

    HRP_LOG_INFO("Renderer: Shutting down Vulkan ...");

    // Wait for all GPU work to finish
    vkDeviceWaitIdle(m_device);

    // Flush all deferred buffer deletions (GPU is idle, safe to destroy)
    for (auto& slot : m_deletionRing) {
        for (auto& db : slot) {
            vmaDestroyBuffer(m_allocator, db.buffer, db.allocation);
        }
        slot.clear();
    }

    // Pipelines
    if (m_3dPipeline != VK_NULL_HANDLE)
        vkDestroyPipeline(m_device, m_3dPipeline, nullptr);
    if (m_3dPipelineLayout != VK_NULL_HANDLE)
        vkDestroyPipelineLayout(m_device, m_3dPipelineLayout, nullptr);
    if (m_3dShadowSetLayout != VK_NULL_HANDLE)
        vkDestroyDescriptorSetLayout(m_device, m_3dShadowSetLayout, nullptr);
    if (m_overlayPipeline != VK_NULL_HANDLE)
        vkDestroyPipeline(m_device, m_overlayPipeline, nullptr);
    if (m_overlayPipelineLayout != VK_NULL_HANDLE)
        vkDestroyPipelineLayout(m_device, m_overlayPipelineLayout, nullptr);

    // PNG render helper resources
    DestroyPNGRenderResources();

    // Sync objects — per-frame-in-flight
    for (uint32_t i = 0; i < k_MaxFramesInFlight; ++i) {
        vkDestroySemaphore(m_device, m_imageAvailableSemaphores[i], nullptr);
        vkDestroyFence(m_device, m_inFlightFences[i], nullptr);
    }
    // Per-swapchain-image render-finished semaphores
    for (auto& sem : m_renderFinishedSemaphores) {
        vkDestroySemaphore(m_device, sem, nullptr);
    }
    m_renderFinishedSemaphores.clear();
    m_imagesInFlight.clear();

    // Phase 6: free the persistent staging ring + batched-upload fence
    // and command buffer BEFORE the command pool / VMA allocator are
    // destroyed (the cmd buffer was allocated from m_commandPool).
    DestroyStagingRing();

    // Command pool (frees command buffers automatically)
    // Phase 6b: destroy the dedicated transfer pool first when it is a
    // distinct object; otherwise it aliases m_commandPool and a single
    // vkDestroyCommandPool below cleans both references up.
    if (m_transferIsSeparate &&
        m_transferCommandPool != VK_NULL_HANDLE &&
        m_transferCommandPool != m_commandPool) {
        vkDestroyCommandPool(m_device, m_transferCommandPool, nullptr);
        m_transferCommandPool = VK_NULL_HANDLE;
    }
    vkDestroyCommandPool(m_device, m_commandPool, nullptr);

    // Depth buffer
    CleanupDepthResources();

    // Swapchain
    CleanupSwapchain();

    // VMA allocator
    if (m_allocator != VK_NULL_HANDLE) {
        vmaDestroyAllocator(m_allocator);
        m_allocator = VK_NULL_HANDLE;
    }

    // Device
    vkDestroyDevice(m_device, nullptr);

    // Surface
    vkDestroySurfaceKHR(m_instance, m_surface, nullptr);

    // Debug messenger
    if (k_EnableValidation && m_debugMessenger != VK_NULL_HANDLE) {
        auto func = reinterpret_cast<PFN_vkDestroyDebugUtilsMessengerEXT>(
            vkGetInstanceProcAddr(m_instance, "vkDestroyDebugUtilsMessengerEXT"));
        if (func) func(m_instance, m_debugMessenger, nullptr);
    }

    // Instance
    vkDestroyInstance(m_instance, nullptr);

    m_initialized = false;
    HRP_LOG_INFO("Renderer: Vulkan shut down");
}

bool Renderer::BeginFrame()
{
    // Wait for the previous frame using this slot to finish.
    // Check return value — VK_ERROR_DEVICE_LOST must not be silently ignored.
    VkResult fenceResult = vkWaitForFences(m_device, 1,
                                           &m_inFlightFences[m_currentFrame],
                                           VK_TRUE, UINT64_MAX);
    if (fenceResult == VK_ERROR_DEVICE_LOST) {
        HRP_LOG_CRITICAL("Renderer: Device lost while waiting for frame fence");
        return false;
    }

    // Drain deferred GPU buffer deletions from k_DeletionRingSize frames
    // ago — guaranteed safe because the fence wait proves all command
    // buffers that could reference those resources have completed.
    ProcessDeferredDeletions();

    // Acquire next swapchain image
    VkResult result = vkAcquireNextImageKHR(
        m_device, m_swapchain, UINT64_MAX,
        m_imageAvailableSemaphores[m_currentFrame],
        VK_NULL_HANDLE, &m_imageIndex);

    if (result == VK_ERROR_OUT_OF_DATE_KHR) {
        return false;  // Caller should recreate swapchain
    }
    HRP_ASSERT(result == VK_SUCCESS || result == VK_SUBOPTIMAL_KHR,
               "Failed to acquire swapchain image");

    // If another frame-in-flight is still rendering to the image we just
    // acquired, wait for THAT frame's fence before we touch the image.
    // This prevents two frame slots from writing to the same swapchain
    // image simultaneously (common with MAILBOX / triple-buffering where
    // imageCount > k_MaxFramesInFlight).
    if (m_imagesInFlight[m_imageIndex] != VK_NULL_HANDLE) {
        vkWaitForFences(m_device, 1, &m_imagesInFlight[m_imageIndex],
                        VK_TRUE, UINT64_MAX);
    }
    // Record that THIS frame slot's fence now guards this image.
    m_imagesInFlight[m_imageIndex] = m_inFlightFences[m_currentFrame];

    // Only reset the fence if we are actually submitting work
    vkResetFences(m_device, 1, &m_inFlightFences[m_currentFrame]);

    // Reset and begin command buffer
    VkCommandBuffer cmd = m_commandBuffers[m_currentFrame];
    vkResetCommandBuffer(cmd, 0);

    VkCommandBufferBeginInfo beginInfo{};
    beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vkBeginCommandBuffer(cmd, &beginInfo);

    // Transition swapchain image to COLOR_ATTACHMENT_OPTIMAL
    VkImageMemoryBarrier barrier{};
    barrier.sType               = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    barrier.oldLayout           = VK_IMAGE_LAYOUT_UNDEFINED;
    barrier.newLayout           = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.image               = m_swapchainImages[m_imageIndex];
    barrier.subresourceRange    = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };
    barrier.srcAccessMask       = 0;
    barrier.dstAccessMask       = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;

    // Transition depth image to DEPTH_STENCIL_ATTACHMENT_OPTIMAL
    VkImageMemoryBarrier depthBarrier{};
    depthBarrier.sType               = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    depthBarrier.oldLayout           = VK_IMAGE_LAYOUT_UNDEFINED;
    depthBarrier.newLayout           = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
    depthBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    depthBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    depthBarrier.image               = m_depthImage;
    depthBarrier.subresourceRange    = { VK_IMAGE_ASPECT_DEPTH_BIT, 0, 1, 0, 1 };
    depthBarrier.srcAccessMask       = 0;
    depthBarrier.dstAccessMask       = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;

    std::array<VkImageMemoryBarrier, 2> barriers = { barrier, depthBarrier };
    vkCmdPipelineBarrier(cmd,
        VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
        VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT | VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT,
        0, 0, nullptr, 0, nullptr,
        static_cast<uint32_t>(barriers.size()), barriers.data());

    // In offscreen mode, ScreenSpaceEffects owns the scene rendering pass.
    // We still transition the swapchain image (needed for composite), but
    // skip beginning the dynamic rendering pass into it.
    if (m_offscreenMode) {
        return true;
    }

    // Begin dynamic rendering (Vulkan 1.3 — no render pass objects needed)
    VkClearValue clearValue{};
    clearValue.color = {{ m_clearColor.r, m_clearColor.g, m_clearColor.b, m_clearColor.a }};

    VkRenderingAttachmentInfo colorAttachment{};
    colorAttachment.sType       = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO;
    colorAttachment.imageView   = m_swapchainImageViews[m_imageIndex];
    colorAttachment.imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    colorAttachment.loadOp      = VK_ATTACHMENT_LOAD_OP_CLEAR;
    colorAttachment.storeOp     = VK_ATTACHMENT_STORE_OP_STORE;
    colorAttachment.clearValue  = clearValue;

    VkClearValue depthClear{};
    depthClear.depthStencil = { 1.0f, 0 };

    VkRenderingAttachmentInfo depthAttachment{};
    depthAttachment.sType       = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO;
    depthAttachment.imageView   = m_depthImageView;
    depthAttachment.imageLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
    depthAttachment.loadOp      = VK_ATTACHMENT_LOAD_OP_CLEAR;
    depthAttachment.storeOp     = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    depthAttachment.clearValue  = depthClear;

    VkRenderingInfo renderInfo{};
    renderInfo.sType                = VK_STRUCTURE_TYPE_RENDERING_INFO;
    renderInfo.renderArea           = { {0, 0}, m_swapchainExtent };
    renderInfo.layerCount           = 1;
    renderInfo.colorAttachmentCount = 1;
    renderInfo.pColorAttachments    = &colorAttachment;
    renderInfo.pDepthAttachment     = &depthAttachment;

    vkCmdBeginRendering(cmd, &renderInfo);

    // Set viewport and scissor (dynamic state)
    VkViewport viewport{};
    viewport.x        = 0.0f;
    viewport.y        = 0.0f;
    viewport.width    = static_cast<float>(m_swapchainExtent.width);
    viewport.height   = static_cast<float>(m_swapchainExtent.height);
    viewport.minDepth = 0.0f;
    viewport.maxDepth = 1.0f;
    vkCmdSetViewport(cmd, 0, 1, &viewport);

    VkRect2D scissor{};
    scissor.offset = { 0, 0 };
    scissor.extent = m_swapchainExtent;
    vkCmdSetScissor(cmd, 0, 1, &scissor);

    return true;
}

void Renderer::EndFrame()
{
    VkCommandBuffer cmd = m_commandBuffers[m_currentFrame];

    // In offscreen mode, ScreenSpaceEffects already ended the composite pass.
    // Skip ending dynamic rendering (there's no active pass to end).
    if (!m_offscreenMode) {
        // End dynamic rendering
        vkCmdEndRendering(cmd);
    }

    // ── Optional: capture swapchain image to PNG ────────────────────
    // Inject blit + copy into the same command buffer so the captured
    // contents match exactly what gets presented this frame. Resources
    // are torn down after the in-flight fence signals.
    const bool       wantCapture = !m_pendingCapturePath.empty();
    VkBuffer         capStaging  = VK_NULL_HANDLE;
    VmaAllocation    capAlloc    = VK_NULL_HANDLE;
    VkImage          capBlitImg  = VK_NULL_HANDLE;
    VmaAllocation    capBlitAll  = VK_NULL_HANDLE;
    const uint32_t   capW        = m_swapchainExtent.width;
    const uint32_t   capH        = m_swapchainExtent.height;
    const VkDeviceSize capBytes  = static_cast<VkDeviceSize>(capW) * capH * 4;

    if (wantCapture) {
        // Transient blit target (R8G8B8A8_UNORM, normalised channel order).
        VkImageCreateInfo ici{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
        ici.imageType   = VK_IMAGE_TYPE_2D;
        ici.format      = VK_FORMAT_R8G8B8A8_UNORM;
        ici.extent      = {capW, capH, 1};
        ici.mipLevels   = 1;
        ici.arrayLayers = 1;
        ici.samples     = VK_SAMPLE_COUNT_1_BIT;
        ici.tiling      = VK_IMAGE_TILING_OPTIMAL;
        ici.usage       = VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
        ici.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        ici.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        VmaAllocationCreateInfo aci{};
        aci.usage = VMA_MEMORY_USAGE_GPU_ONLY;
        if (vmaCreateImage(m_allocator, &ici, &aci, &capBlitImg, &capBlitAll, nullptr) != VK_SUCCESS) {
            HRP_LOG_WARN("Renderer: capture blit image alloc failed");
        }
        // Host-readable staging buffer.
        VkBufferCreateInfo bci{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
        bci.size  = capBytes;
        bci.usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT;
        VmaAllocationCreateInfo bai{};
        bai.usage = VMA_MEMORY_USAGE_GPU_TO_CPU;
        bai.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT
                  | VMA_ALLOCATION_CREATE_MAPPED_BIT;
        if (vmaCreateBuffer(m_allocator, &bci, &bai, &capStaging, &capAlloc, nullptr) != VK_SUCCESS) {
            HRP_LOG_WARN("Renderer: capture staging alloc failed");
        }
    }

    if (wantCapture && capBlitImg != VK_NULL_HANDLE && capStaging != VK_NULL_HANDLE) {
        // (1) Swapchain COLOR_ATTACHMENT → TRANSFER_SRC.
        VkImageMemoryBarrier b{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
        b.oldLayout     = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
        b.newLayout     = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
        b.srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
        b.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
        b.srcQueueFamilyIndex = b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        b.image         = m_swapchainImages[m_imageIndex];
        b.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
        vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
                             VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0, nullptr, 1, &b);

        // (2) Blit target UNDEFINED → TRANSFER_DST.
        VkImageMemoryBarrier b2 = b;
        b2.oldLayout     = VK_IMAGE_LAYOUT_UNDEFINED;
        b2.newLayout     = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
        b2.srcAccessMask = 0;
        b2.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        b2.image         = capBlitImg;
        vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                             VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0, nullptr, 1, &b2);

        // (3) Blit swapchain → blitImg (handles BGRA→RGBA channel swizzle on driver).
        VkImageBlit blit{};
        blit.srcSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
        blit.srcOffsets[1]  = {static_cast<int32_t>(capW), static_cast<int32_t>(capH), 1};
        blit.dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
        blit.dstOffsets[1]  = {static_cast<int32_t>(capW), static_cast<int32_t>(capH), 1};
        vkCmdBlitImage(cmd,
                       m_swapchainImages[m_imageIndex], VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                       capBlitImg, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                       1, &blit, VK_FILTER_NEAREST);

        // (4) Blit target TRANSFER_DST → TRANSFER_SRC.
        VkImageMemoryBarrier b3 = b2;
        b3.oldLayout     = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
        b3.newLayout     = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
        b3.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        b3.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
        vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT,
                             VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0, nullptr, 1, &b3);

        // (5) Copy blitImg → staging buffer.
        VkBufferImageCopy r{};
        r.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
        r.imageExtent      = {capW, capH, 1};
        vkCmdCopyImageToBuffer(cmd, capBlitImg, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                               capStaging, 1, &r);

        // (6) Buffer barrier for HOST visibility.
        VkBufferMemoryBarrier bb{VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER};
        bb.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        bb.dstAccessMask = VK_ACCESS_HOST_READ_BIT;
        bb.srcQueueFamilyIndex = bb.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        bb.buffer        = capStaging;
        bb.size          = VK_WHOLE_SIZE;
        vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT,
                             VK_PIPELINE_STAGE_HOST_BIT, 0, 0, nullptr, 1, &bb, 0, nullptr);

        // (7) Swapchain TRANSFER_SRC → PRESENT_SRC (replaces the normal barrier below).
        VkImageMemoryBarrier b4 = b;
        b4.oldLayout     = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
        b4.newLayout     = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
        b4.srcAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
        b4.dstAccessMask = 0;
        vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT,
                             VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, 0, 0, nullptr, 0, nullptr, 1, &b4);
    } else {
        // Standard path: Transition swapchain image to PRESENT_SRC.
        VkImageMemoryBarrier barrier{};
        barrier.sType               = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
        barrier.oldLayout           = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
        barrier.newLayout           = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
        barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier.image               = m_swapchainImages[m_imageIndex];
        barrier.subresourceRange    = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };
        barrier.srcAccessMask       = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
        barrier.dstAccessMask       = 0;

        vkCmdPipelineBarrier(cmd,
            VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
            VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,
            0, 0, nullptr, 0, nullptr, 1, &barrier);
    }

    vkEndCommandBuffer(cmd);

    // Submit
    VkSemaphore          waitSemaphores[]   = { m_imageAvailableSemaphores[m_currentFrame] };
    VkPipelineStageFlags waitStages[]       = { VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT };
    // Signal the per-image semaphore (indexed by acquired image, not frame-in-flight)
    VkSemaphore          signalSemaphores[] = { m_renderFinishedSemaphores[m_imageIndex] };

    VkSubmitInfo submitInfo{};
    submitInfo.sType                = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submitInfo.waitSemaphoreCount   = 1;
    submitInfo.pWaitSemaphores      = waitSemaphores;
    submitInfo.pWaitDstStageMask    = waitStages;
    submitInfo.commandBufferCount   = 1;
    submitInfo.pCommandBuffers      = &cmd;
    submitInfo.signalSemaphoreCount = 1;
    submitInfo.pSignalSemaphores    = signalSemaphores;

    VkResult submitResult = vkQueueSubmit(m_graphicsQueue, 1, &submitInfo,
                                           m_inFlightFences[m_currentFrame]);
    if (submitResult != VK_SUCCESS) {
        HRP_LOG_CRITICAL("Renderer: vkQueueSubmit failed (VkResult {})", static_cast<int>(submitResult));
    }
    HRP_ASSERT(submitResult == VK_SUCCESS, "Failed to submit draw command buffer (VkResult={})",
               static_cast<int>(submitResult));

    // Present
    VkPresentInfoKHR presentInfo{};
    presentInfo.sType              = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
    presentInfo.waitSemaphoreCount = 1;
    presentInfo.pWaitSemaphores    = signalSemaphores;
    presentInfo.swapchainCount     = 1;
    presentInfo.pSwapchains        = &m_swapchain;
    presentInfo.pImageIndices      = &m_imageIndex;

    VkResult presentResult = vkQueuePresentKHR(m_presentQueue, &presentInfo);

    if (presentResult == VK_ERROR_OUT_OF_DATE_KHR || presentResult == VK_SUBOPTIMAL_KHR) {
        HRP_LOG_DEBUG("Renderer: Swapchain suboptimal/out-of-date after present");
    }

    // ── Capture finalisation ───────────────────────────────────────
    // Wait on this frame's fence so the staging buffer is filled, then
    // gamma-correct and write the PNG. Resources are owned locally so
    // we destroy them inline (we already block, so no in-flight risk).
    if (wantCapture && capStaging != VK_NULL_HANDLE) {
        vkWaitForFences(m_device, 1, &m_inFlightFences[m_currentFrame], VK_TRUE, UINT64_MAX);

        VmaAllocationInfo allocInfo{};
        vmaGetAllocationInfo(m_allocator, capAlloc, &allocInfo);
        uint8_t* pixels = static_cast<uint8_t*>(allocInfo.pMappedData);
        if (pixels != nullptr) {
            // Apply sRGB gamma — scene rendering is linear HDR; the swapchain
            // is UNORM so values are already clamped to [0,1] but still linear.
            const uint32_t pixelCount = capW * capH;
            for (uint32_t i = 0; i < pixelCount; ++i) {
                for (int c = 0; c < 3; ++c) {
                    float v = static_cast<float>(pixels[i * 4 + c]) / 255.0f;
                    v = (v <= 0.0031308f) ? (v * 12.92f)
                                          : (1.055f * std::pow(v, 1.0f / 2.4f) - 0.055f);
                    pixels[i * 4 + c] = static_cast<uint8_t>(std::min(v * 255.0f + 0.5f, 255.0f));
                }
                pixels[i * 4 + 3] = 255;
            }
            int ok = stbi_write_png(m_pendingCapturePath.c_str(),
                                    static_cast<int>(capW), static_cast<int>(capH),
                                    4, pixels, static_cast<int>(capW * 4));
            if (ok) {
                HRP_LOG_INFO("Renderer: swapchain PNG saved → {}", m_pendingCapturePath);
            } else {
                HRP_LOG_WARN("Renderer: stbi_write_png failed for {}", m_pendingCapturePath);
            }
        } else {
            HRP_LOG_WARN("Renderer: capture staging not host-mapped");
        }

        vmaDestroyBuffer(m_allocator, capStaging, capAlloc);
        if (capBlitImg != VK_NULL_HANDLE) {
            vmaDestroyImage(m_allocator, capBlitImg, capBlitAll);
        }
        m_pendingCapturePath.clear();
    } else if (wantCapture) {
        // Allocation failed — drop the request so we don't try forever.
        if (capBlitImg != VK_NULL_HANDLE) vmaDestroyImage(m_allocator, capBlitImg, capBlitAll);
        if (capStaging != VK_NULL_HANDLE) vmaDestroyBuffer(m_allocator, capStaging, capAlloc);
        m_pendingCapturePath.clear();
    }

    m_currentFrame = (m_currentFrame + 1) % k_MaxFramesInFlight;
}

void Renderer::RecreateSwapchain(Platform& platform)
{
    vkDeviceWaitIdle(m_device);

    // Wait until BOTH the SDL window AND Vulkan surface report non-zero dimensions.
    // Checking only GetWindowSize() is insufficient — a race between SDL's window
    // state and the Vulkan surface capabilities can still yield a 0×0 extent.
    for (;;) {
        auto size = platform.GetWindowSize();
        if (size.x == 0 || size.y == 0) {
            platform.PollEvents();
            SDL_Delay(10);
            continue;
        }

        VkSurfaceCapabilitiesKHR caps{};
        vkGetPhysicalDeviceSurfaceCapabilitiesKHR(m_physicalDevice, m_surface, &caps);
        // When currentExtent is UINT32_MAX the compositor lets us choose — window size is fine.
        if (caps.currentExtent.width != UINT32_MAX &&
            (caps.currentExtent.width == 0 || caps.currentExtent.height == 0)) {
            platform.PollEvents();
            SDL_Delay(10);
            continue;
        }
        break;
    }

    // Destroy per-swapchain-image render-finished semaphores
    for (auto& sem : m_renderFinishedSemaphores) {
        vkDestroySemaphore(m_device, sem, nullptr);
    }
    m_renderFinishedSemaphores.clear();

    CleanupDepthResources();
    CleanupSwapchain();
    CreateSwapchain(platform);
    CreateImageViews();
    CreateDepthResources();

    // Recreate render-finished semaphores for the (potentially new) image count
    const uint32_t imageCount = static_cast<uint32_t>(m_swapchainImages.size());
    m_renderFinishedSemaphores.resize(imageCount);
    VkSemaphoreCreateInfo semInfo{};
    semInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
    for (uint32_t i = 0; i < imageCount; ++i) {
        vkCreateSemaphore(m_device, &semInfo, nullptr, &m_renderFinishedSemaphores[i]);
    }

    // Reset per-image fence tracking — after vkDeviceWaitIdle all work is done,
    // so no frame-in-flight is using any image.
    m_imagesInFlight.assign(imageCount, VK_NULL_HANDLE);

    // Reset frame counter so the next BeginFrame uses a clean slot whose fence
    // is guaranteed signaled (all work complete after vkDeviceWaitIdle).
    m_currentFrame = 0;

    HRP_LOG_INFO("Renderer: Swapchain recreated ({}×{})",
                 m_swapchainExtent.width, m_swapchainExtent.height);
}

// ──────────────────────────────────────────────────────────────────────
// Instance creation
// ──────────────────────────────────────────────────────────────────────

bool Renderer::CreateInstance(Platform& platform)
{
    VkApplicationInfo appInfo{};
    appInfo.sType              = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    appInfo.pApplicationName   = "Hybrid Render Pipeline";
    appInfo.applicationVersion = VK_MAKE_API_VERSION(0, 0, 1, 0);
    appInfo.pEngineName        = "Hybrid Render Pipeline";
    appInfo.engineVersion      = VK_MAKE_API_VERSION(0, 0, 1, 0);
    appInfo.apiVersion         = VK_API_VERSION_1_3;

    // Get extensions required by SDL3
    auto extensions = platform.GetRequiredVulkanExtensions();

    if (k_EnableValidation) {
        extensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
    }

    VkInstanceCreateInfo createInfo{};
    createInfo.sType                   = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    createInfo.pApplicationInfo        = &appInfo;
    createInfo.enabledExtensionCount   = static_cast<uint32_t>(extensions.size());
    createInfo.ppEnabledExtensionNames = extensions.data();

    if (k_EnableValidation) {
        createInfo.enabledLayerCount   = static_cast<uint32_t>(k_ValidationLayers.size());
        createInfo.ppEnabledLayerNames = k_ValidationLayers.data();
    }

    VkResult result = vkCreateInstance(&createInfo, nullptr, &m_instance);
    if (result != VK_SUCCESS) {
        HRP_LOG_CRITICAL("Renderer: vkCreateInstance failed (VkResult {})", static_cast<int>(result));
        return false;
    }

    HRP_LOG_INFO("Renderer: VkInstance created (API 1.3, {} extensions, validation={})",
                 extensions.size(), k_EnableValidation);
    return true;
}

// ──────────────────────────────────────────────────────────────────────
// Debug messenger
// ──────────────────────────────────────────────────────────────────────

bool Renderer::SetupDebugMessenger()
{
    VkDebugUtilsMessengerCreateInfoEXT createInfo{};
    createInfo.sType           = VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT;
    createInfo.messageSeverity = VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT
                               | VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;
    createInfo.messageType     = VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT
                               | VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT
                               | VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT;
    createInfo.pfnUserCallback = VulkanDebugCallback;

    auto func = reinterpret_cast<PFN_vkCreateDebugUtilsMessengerEXT>(
        vkGetInstanceProcAddr(m_instance, "vkCreateDebugUtilsMessengerEXT"));

    if (!func) return false;

    return func(m_instance, &createInfo, nullptr, &m_debugMessenger) == VK_SUCCESS;
}

// ──────────────────────────────────────────────────────────────────────
// Physical device selection
// ──────────────────────────────────────────────────────────────────────

bool Renderer::PickPhysicalDevice()
{
    uint32_t deviceCount = 0;
    vkEnumeratePhysicalDevices(m_instance, &deviceCount, nullptr);
    if (deviceCount == 0) {
        HRP_LOG_CRITICAL("Renderer: No Vulkan-capable GPU found");
        return false;
    }

    std::vector<VkPhysicalDevice> devices(deviceCount);
    vkEnumeratePhysicalDevices(m_instance, &deviceCount, devices.data());

    // Score all suitable devices, pick highest
    int bestScore = -1;
    for (auto& dev : devices) {
        if (!IsDeviceSuitable(dev)) continue;
        int score = RateDeviceSuitability(dev);
        if (score > bestScore) {
            bestScore = score;
            m_physicalDevice = dev;
        }
    }

    if (m_physicalDevice == VK_NULL_HANDLE) {
        HRP_LOG_CRITICAL("Renderer: No suitable GPU found (need graphics + present queues + swapchain)");
        return false;
    }

    VkPhysicalDeviceProperties props;
    vkGetPhysicalDeviceProperties(m_physicalDevice, &props);
    HRP_LOG_INFO("Renderer: Selected GPU — {} (score {})", props.deviceName, bestScore);
    HRP_LOG_INFO("Renderer: Driver version {}.{}.{}",
                 VK_API_VERSION_MAJOR(props.driverVersion),
                 VK_API_VERSION_MINOR(props.driverVersion),
                 VK_API_VERSION_PATCH(props.driverVersion));

    return true;
}

bool Renderer::IsDeviceSuitable(VkPhysicalDevice device) const
{
    auto indices = FindQueueFamilies(device);
    if (!indices.IsComplete()) return false;

    // Check required device extensions
    uint32_t extCount;
    vkEnumerateDeviceExtensionProperties(device, nullptr, &extCount, nullptr);
    std::vector<VkExtensionProperties> available(extCount);
    vkEnumerateDeviceExtensionProperties(device, nullptr, &extCount, available.data());

    std::set<std::string> required(k_DeviceExtensions.begin(), k_DeviceExtensions.end());
    for (const auto& ext : available) {
        required.erase(ext.extensionName);
    }
    if (!required.empty()) return false;

    // Check swapchain support
    auto swapSupport = QuerySwapchainSupport(device);
    return !swapSupport.formats.empty() && !swapSupport.presentModes.empty();
}

int Renderer::RateDeviceSuitability(VkPhysicalDevice device) const
{
    VkPhysicalDeviceProperties props;
    vkGetPhysicalDeviceProperties(device, &props);

    VkPhysicalDeviceFeatures features;
    vkGetPhysicalDeviceFeatures(device, &features);

    int score = 0;

    // Strongly prefer discrete GPUs
    if (props.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU) {
        score += 10000;
    }

    // Higher VRAM is better (approximation via max image dimension)
    score += static_cast<int>(props.limits.maxImageDimension2D);

    // Bonus for geometry shader (not critical, but nice for future use)
    if (features.geometryShader) score += 500;

    // Bonus for multi-draw indirect (essential for instanced rendering)
    if (features.multiDrawIndirect) score += 1000;

    // Bonus for wide lines (debug rendering)
    if (features.wideLines) score += 100;

    return score;
}

// ──────────────────────────────────────────────────────────────────────
// Logical device
// ──────────────────────────────────────────────────────────────────────

bool Renderer::CreateLogicalDevice()
{
    auto indices = FindQueueFamilies(m_physicalDevice);
    m_graphicsFamily = indices.graphics;
    m_presentFamily  = indices.present;
    m_transferFamily = indices.transfer;
    m_transferIsSeparate = (indices.transfer != indices.graphics);

    // Create unique queue create infos
    std::set<uint32_t> uniqueFamilies = { indices.graphics, indices.present, indices.transfer };
    std::vector<VkDeviceQueueCreateInfo> queueCreateInfos;
    float queuePriority = 1.0f;

    for (uint32_t family : uniqueFamilies) {
        VkDeviceQueueCreateInfo queueInfo{};
        queueInfo.sType            = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
        queueInfo.queueFamilyIndex = family;
        queueInfo.queueCount       = 1;
        queueInfo.pQueuePriorities = &queuePriority;
        queueCreateInfos.push_back(queueInfo);
    }

    // Enable Vulkan 1.3 features: dynamic rendering, synchronization2
    VkPhysicalDeviceVulkan13Features features13{};
    features13.sType            = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES;
    features13.dynamicRendering = VK_TRUE;
    features13.synchronization2 = VK_TRUE;

    VkPhysicalDeviceVulkan12Features features12{};
    features12.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES;
    features12.pNext = &features13;

    VkPhysicalDeviceFeatures deviceFeatures{};
    deviceFeatures.multiDrawIndirect = VK_TRUE;
    deviceFeatures.fillModeNonSolid  = VK_TRUE;   // Wireframe mode for debugging
    deviceFeatures.wideLines         = VK_TRUE;    // Debug line rendering
    deviceFeatures.samplerAnisotropy = VK_TRUE;    // Texture quality

    VkDeviceCreateInfo createInfo{};
    createInfo.sType                   = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    createInfo.pNext                   = &features12;
    createInfo.queueCreateInfoCount    = static_cast<uint32_t>(queueCreateInfos.size());
    createInfo.pQueueCreateInfos       = queueCreateInfos.data();
    createInfo.enabledExtensionCount   = static_cast<uint32_t>(k_DeviceExtensions.size());
    createInfo.ppEnabledExtensionNames = k_DeviceExtensions.data();
    createInfo.pEnabledFeatures        = &deviceFeatures;

    VkResult result = vkCreateDevice(m_physicalDevice, &createInfo, nullptr, &m_device);
    if (result != VK_SUCCESS) {
        HRP_LOG_CRITICAL("Renderer: vkCreateDevice failed (VkResult {})", static_cast<int>(result));
        return false;
    }

    vkGetDeviceQueue(m_device, indices.graphics, 0, &m_graphicsQueue);
    vkGetDeviceQueue(m_device, indices.present,  0, &m_presentQueue);
    if (m_transferIsSeparate) {
        vkGetDeviceQueue(m_device, indices.transfer, 0, &m_transferQueue);
    } else {
        m_transferQueue = m_graphicsQueue;
    }

    HRP_LOG_INFO("Renderer: Logical device created (graphics queue family={}, present={}, transfer={}{})",
                 indices.graphics, indices.present, indices.transfer,
                 m_transferIsSeparate ? " [dedicated DMA]" : " [shared with graphics]");
    return true;
}

// ──────────────────────────────────────────────────────────────────────
// Swapchain
// ──────────────────────────────────────────────────────────────────────

bool Renderer::CreateSwapchain(Platform& platform)
{
    auto support = QuerySwapchainSupport(m_physicalDevice);

    auto surfaceFormat = ChooseSwapSurfaceFormat(support.formats);
    auto presentMode   = ChooseSwapPresentMode(support.presentModes);
    auto extent        = ChooseSwapExtent(support.capabilities, platform);

    uint32_t imageCount = support.capabilities.minImageCount + 1;
    if (support.capabilities.maxImageCount > 0 && imageCount > support.capabilities.maxImageCount) {
        imageCount = support.capabilities.maxImageCount;
    }

    VkSwapchainCreateInfoKHR createInfo{};
    createInfo.sType            = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR;
    createInfo.surface          = m_surface;
    createInfo.minImageCount    = imageCount;
    createInfo.imageFormat      = surfaceFormat.format;
    createInfo.imageColorSpace  = surfaceFormat.colorSpace;
    createInfo.imageExtent      = extent;
    createInfo.imageArrayLayers = 1;
    // TRANSFER_SRC_BIT enables vkCmdBlitImage / vkCmdCopyImageToBuffer from
    // the swapchain image (used by RequestSwapchainCapture for headless
    // --screenshot-world mode). Universally supported on desktop drivers.
    createInfo.imageUsage       = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT
                                | VK_IMAGE_USAGE_TRANSFER_SRC_BIT;

    auto indices = FindQueueFamilies(m_physicalDevice);
    uint32_t familyIndices[] = { indices.graphics, indices.present };

    if (indices.graphics != indices.present) {
        createInfo.imageSharingMode      = VK_SHARING_MODE_CONCURRENT;
        createInfo.queueFamilyIndexCount = 2;
        createInfo.pQueueFamilyIndices   = familyIndices;
    } else {
        createInfo.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
    }

    createInfo.preTransform   = support.capabilities.currentTransform;
    createInfo.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
    createInfo.presentMode    = presentMode;
    createInfo.clipped        = VK_TRUE;
    createInfo.oldSwapchain   = VK_NULL_HANDLE;

    VkResult result = vkCreateSwapchainKHR(m_device, &createInfo, nullptr, &m_swapchain);
    if (result != VK_SUCCESS) {
        HRP_LOG_CRITICAL("Renderer: vkCreateSwapchainKHR failed (VkResult {})", static_cast<int>(result));
        return false;
    }

    m_swapchainFormat = surfaceFormat.format;
    m_sceneColorFormat = m_swapchainFormat;  // Default; overridden by SetSceneColorFormat() when SSE active
    m_swapchainExtent = extent;

    // Retrieve swapchain images
    vkGetSwapchainImagesKHR(m_device, m_swapchain, &imageCount, nullptr);
    m_swapchainImages.resize(imageCount);
    vkGetSwapchainImagesKHR(m_device, m_swapchain, &imageCount, m_swapchainImages.data());

    HRP_LOG_INFO("Renderer: Swapchain created — {}×{}, {} images, format {}",
                 extent.width, extent.height, imageCount, static_cast<int>(surfaceFormat.format));

    return true;
}

bool Renderer::CreateImageViews()
{
    m_swapchainImageViews.resize(m_swapchainImages.size());

    for (size_t i = 0; i < m_swapchainImages.size(); ++i) {
        VkImageViewCreateInfo createInfo{};
        createInfo.sType    = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
        createInfo.image    = m_swapchainImages[i];
        createInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
        createInfo.format   = m_swapchainFormat;

        createInfo.components.r = VK_COMPONENT_SWIZZLE_IDENTITY;
        createInfo.components.g = VK_COMPONENT_SWIZZLE_IDENTITY;
        createInfo.components.b = VK_COMPONENT_SWIZZLE_IDENTITY;
        createInfo.components.a = VK_COMPONENT_SWIZZLE_IDENTITY;

        createInfo.subresourceRange.aspectMask     = VK_IMAGE_ASPECT_COLOR_BIT;
        createInfo.subresourceRange.baseMipLevel   = 0;
        createInfo.subresourceRange.levelCount     = 1;
        createInfo.subresourceRange.baseArrayLayer = 0;
        createInfo.subresourceRange.layerCount     = 1;

        VkResult result = vkCreateImageView(m_device, &createInfo, nullptr,
                                             &m_swapchainImageViews[i]);
        if (result != VK_SUCCESS) {
            HRP_LOG_CRITICAL("Renderer: Failed to create image view {}", i);
            return false;
        }
    }

    return true;
}

void Renderer::CleanupSwapchain()
{
    for (auto view : m_swapchainImageViews) {
        vkDestroyImageView(m_device, view, nullptr);
    }
    m_swapchainImageViews.clear();

    if (m_swapchain != VK_NULL_HANDLE) {
        vkDestroySwapchainKHR(m_device, m_swapchain, nullptr);
        m_swapchain = VK_NULL_HANDLE;
    }
}

// ──────────────────────────────────────────────────────────────────────
// Command pool & buffers
// ──────────────────────────────────────────────────────────────────────

bool Renderer::CreateCommandPool()
{
    VkCommandPoolCreateInfo poolInfo{};
    poolInfo.sType            = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    poolInfo.flags            = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    poolInfo.queueFamilyIndex = m_graphicsFamily;

    VkResult result = vkCreateCommandPool(m_device, &poolInfo, nullptr, &m_commandPool);
    if (result != VK_SUCCESS) {
        HRP_LOG_CRITICAL("Renderer: Failed to create command pool");
        return false;
    }

    // Phase 6b: dedicated transfer command pool when the device exposes
    // a separate DMA queue family. Otherwise alias the graphics pool so
    // BeginUploadBatch can blindly use m_transferCommandPool.
    if (m_transferIsSeparate) {
        VkCommandPoolCreateInfo transferPoolInfo{};
        transferPoolInfo.sType            = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
        transferPoolInfo.flags            = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT |
                                            VK_COMMAND_POOL_CREATE_TRANSIENT_BIT;
        transferPoolInfo.queueFamilyIndex = m_transferFamily;
        if (vkCreateCommandPool(m_device, &transferPoolInfo, nullptr,
                                &m_transferCommandPool) != VK_SUCCESS) {
            HRP_LOG_WARN("Renderer: Failed to create transfer command pool — "
                         "falling back to graphics pool for uploads");
            m_transferCommandPool = m_commandPool;
            m_transferIsSeparate  = false;
            m_transferQueue       = m_graphicsQueue;
        }
    } else {
        m_transferCommandPool = m_commandPool;
    }

    return true;
}

bool Renderer::CreateCommandBuffers()
{
    m_commandBuffers.resize(k_MaxFramesInFlight);

    VkCommandBufferAllocateInfo allocInfo{};
    allocInfo.sType              = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    allocInfo.commandPool        = m_commandPool;
    allocInfo.level              = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    allocInfo.commandBufferCount = k_MaxFramesInFlight;

    VkResult result = vkAllocateCommandBuffers(m_device, &allocInfo, m_commandBuffers.data());
    if (result != VK_SUCCESS) {
        HRP_LOG_CRITICAL("Renderer: Failed to allocate command buffers");
        return false;
    }

    return true;
}

// ──────────────────────────────────────────────────────────────────────
// Synchronization
// ──────────────────────────────────────────────────────────────────────

bool Renderer::CreateSyncObjects()
{
    m_imageAvailableSemaphores.resize(k_MaxFramesInFlight);
    m_inFlightFences.resize(k_MaxFramesInFlight);

    // Render-finished semaphores are per-swapchain-image (not per frame-in-flight).
    // This prevents reuse of a semaphore that the presentation engine may still own.
    // See: https://docs.vulkan.org/guide/latest/swapchain_semaphore_reuse.html
    const uint32_t imageCount = static_cast<uint32_t>(m_swapchainImages.size());
    m_renderFinishedSemaphores.resize(imageCount);

    // Per-image fence tracking: initially no frame is rendering to any image.
    m_imagesInFlight.resize(imageCount, VK_NULL_HANDLE);

    VkSemaphoreCreateInfo semInfo{};
    semInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;

    VkFenceCreateInfo fenceInfo{};
    fenceInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    fenceInfo.flags = VK_FENCE_CREATE_SIGNALED_BIT;

    for (uint32_t i = 0; i < k_MaxFramesInFlight; ++i) {
        if (vkCreateSemaphore(m_device, &semInfo, nullptr, &m_imageAvailableSemaphores[i]) != VK_SUCCESS ||
            vkCreateFence(m_device, &fenceInfo, nullptr, &m_inFlightFences[i]) != VK_SUCCESS) {
            HRP_LOG_CRITICAL("Renderer: Failed to create sync objects for frame {}", i);
            return false;
        }
    }

    for (uint32_t i = 0; i < imageCount; ++i) {
        if (vkCreateSemaphore(m_device, &semInfo, nullptr, &m_renderFinishedSemaphores[i]) != VK_SUCCESS) {
            HRP_LOG_CRITICAL("Renderer: Failed to create render-finished semaphore for image {}", i);
            return false;
        }
    }

    return true;
}

// ──────────────────────────────────────────────────────────────────────
// Queue family & swapchain queries
// ──────────────────────────────────────────────────────────────────────

Renderer::QueueFamilyIndices Renderer::FindQueueFamilies(VkPhysicalDevice device) const
{
    QueueFamilyIndices indices;

    uint32_t count = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(device, &count, nullptr);
    std::vector<VkQueueFamilyProperties> families(count);
    vkGetPhysicalDeviceQueueFamilyProperties(device, &count, families.data());

    for (uint32_t i = 0; i < count; ++i) {
        if (families[i].queueFlags & VK_QUEUE_GRAPHICS_BIT) {
            indices.graphics = i;
        }

        VkBool32 presentSupport = false;
        vkGetPhysicalDeviceSurfaceSupportKHR(device, i, m_surface, &presentSupport);
        if (presentSupport) {
            indices.present = i;
        }
    }

    // Phase 6b: prefer a dedicated DMA queue family — one that exposes
    // VK_QUEUE_TRANSFER_BIT but NOT VK_QUEUE_GRAPHICS_BIT (and ideally
    // not COMPUTE either). On modern desktop GPUs this picks the
    // dedicated copy engine, letting transfers overlap rendering. If
    // none exists, fall back to the graphics family.
    for (uint32_t i = 0; i < count; ++i) {
        const VkQueueFlags f = families[i].queueFlags;
        const bool hasTransfer = (f & VK_QUEUE_TRANSFER_BIT) != 0;
        const bool hasGraphics = (f & VK_QUEUE_GRAPHICS_BIT) != 0;
        const bool hasCompute  = (f & VK_QUEUE_COMPUTE_BIT)  != 0;
        if (hasTransfer && !hasGraphics && !hasCompute) {
            indices.transfer = i;
            break;
        }
    }
    if (indices.transfer == UINT32_MAX) {
        // Second pass: TRANSFER && !GRAPHICS (compute-capable transfer).
        for (uint32_t i = 0; i < count; ++i) {
            const VkQueueFlags f = families[i].queueFlags;
            if ((f & VK_QUEUE_TRANSFER_BIT) && !(f & VK_QUEUE_GRAPHICS_BIT)) {
                indices.transfer = i;
                break;
            }
        }
    }
    if (indices.transfer == UINT32_MAX) indices.transfer = indices.graphics;

    return indices;
}

Renderer::SwapchainSupportDetails Renderer::QuerySwapchainSupport(VkPhysicalDevice device) const
{
    SwapchainSupportDetails details;

    vkGetPhysicalDeviceSurfaceCapabilitiesKHR(device, m_surface, &details.capabilities);

    uint32_t formatCount;
    vkGetPhysicalDeviceSurfaceFormatsKHR(device, m_surface, &formatCount, nullptr);
    if (formatCount > 0) {
        details.formats.resize(formatCount);
        vkGetPhysicalDeviceSurfaceFormatsKHR(device, m_surface, &formatCount, details.formats.data());
    }

    uint32_t modeCount;
    vkGetPhysicalDeviceSurfacePresentModesKHR(device, m_surface, &modeCount, nullptr);
    if (modeCount > 0) {
        details.presentModes.resize(modeCount);
        vkGetPhysicalDeviceSurfacePresentModesKHR(device, m_surface, &modeCount,
                                                    details.presentModes.data());
    }

    return details;
}

VkSurfaceFormatKHR Renderer::ChooseSwapSurfaceFormat(
    const std::vector<VkSurfaceFormatKHR>& available) const
{
    // Prefer SRGB + B8G8R8A8
    for (const auto& fmt : available) {
        if (fmt.format == VK_FORMAT_B8G8R8A8_SRGB &&
            fmt.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR) {
            return fmt;
        }
    }
    return available[0];
}

VkPresentModeKHR Renderer::ChooseSwapPresentMode(
    const std::vector<VkPresentModeKHR>& available) const
{
    // Prefer mailbox (triple-buffered, low-latency vsync)
    for (const auto& mode : available) {
        if (mode == VK_PRESENT_MODE_MAILBOX_KHR) return mode;
    }
    // Fallback to FIFO (guaranteed by spec — standard vsync)
    return VK_PRESENT_MODE_FIFO_KHR;
}

VkExtent2D Renderer::ChooseSwapExtent(const VkSurfaceCapabilitiesKHR& capabilities,
                                       Platform& platform) const
{
    if (capabilities.currentExtent.width != UINT32_MAX) {
        return capabilities.currentExtent;
    }

    auto size = platform.GetWindowSize();
    VkExtent2D actual = {
        static_cast<uint32_t>(size.x),
        static_cast<uint32_t>(size.y)
    };

    actual.width  = std::clamp(actual.width,
                               capabilities.minImageExtent.width,
                               capabilities.maxImageExtent.width);
    actual.height = std::clamp(actual.height,
                               capabilities.minImageExtent.height,
                               capabilities.maxImageExtent.height);
    return actual;
}

// ──────────────────────────────────────────────────────────────────────
// VMA Allocator
// ──────────────────────────────────────────────────────────────────────

bool Renderer::CreateAllocator()
{
    VmaAllocatorCreateInfo allocatorInfo{};
    allocatorInfo.physicalDevice = m_physicalDevice;
    allocatorInfo.device         = m_device;
    allocatorInfo.instance       = m_instance;
    allocatorInfo.vulkanApiVersion = VK_API_VERSION_1_3;

    VkResult result = vmaCreateAllocator(&allocatorInfo, &m_allocator);
    if (result != VK_SUCCESS) {
        HRP_LOG_CRITICAL("Renderer: Failed to create VMA allocator (VkResult {})", static_cast<int>(result));
        return false;
    }

    HRP_LOG_INFO("Renderer: VMA allocator created");
    return true;
}

// ──────────────────────────────────────────────────────────────────────
// Depth buffer
// ──────────────────────────────────────────────────────────────────────

bool Renderer::CreateDepthResources()
{
    VkImageCreateInfo imageInfo{};
    imageInfo.sType       = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    imageInfo.imageType   = VK_IMAGE_TYPE_2D;
    imageInfo.format      = k_DepthFormat;
    imageInfo.extent      = { m_swapchainExtent.width, m_swapchainExtent.height, 1 };
    imageInfo.mipLevels   = 1;
    imageInfo.arrayLayers = 1;
    imageInfo.samples     = VK_SAMPLE_COUNT_1_BIT;
    imageInfo.tiling      = VK_IMAGE_TILING_OPTIMAL;
    imageInfo.usage       = VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT;

    VmaAllocationCreateInfo allocInfo{};
    allocInfo.usage = VMA_MEMORY_USAGE_GPU_ONLY;

    VkResult result = vmaCreateImage(m_allocator, &imageInfo, &allocInfo,
                                      &m_depthImage, &m_depthAllocation, nullptr);
    if (result != VK_SUCCESS) {
        HRP_LOG_CRITICAL("Renderer: Failed to create depth image");
        return false;
    }

    VkImageViewCreateInfo viewInfo{};
    viewInfo.sType    = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    viewInfo.image    = m_depthImage;
    viewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
    viewInfo.format   = k_DepthFormat;
    viewInfo.subresourceRange.aspectMask     = VK_IMAGE_ASPECT_DEPTH_BIT;
    viewInfo.subresourceRange.baseMipLevel   = 0;
    viewInfo.subresourceRange.levelCount     = 1;
    viewInfo.subresourceRange.baseArrayLayer = 0;
    viewInfo.subresourceRange.layerCount     = 1;

    result = vkCreateImageView(m_device, &viewInfo, nullptr, &m_depthImageView);
    if (result != VK_SUCCESS) {
        HRP_LOG_CRITICAL("Renderer: Failed to create depth image view");
        return false;
    }

    HRP_LOG_INFO("Renderer: Depth buffer created ({}×{}, D32_SFLOAT)",
                 m_swapchainExtent.width, m_swapchainExtent.height);
    return true;
}

void Renderer::CleanupDepthResources()
{
    if (m_depthImageView != VK_NULL_HANDLE) {
        vkDestroyImageView(m_device, m_depthImageView, nullptr);
        m_depthImageView = VK_NULL_HANDLE;
    }
    if (m_depthImage != VK_NULL_HANDLE) {
        vmaDestroyImage(m_allocator, m_depthImage, m_depthAllocation);
        m_depthImage = VK_NULL_HANDLE;
        m_depthAllocation = VK_NULL_HANDLE;
    }
}

// ──────────────────────────────────────────────────────────────────────
// Shader loading
// ──────────────────────────────────────────────────────────────────────

std::string Renderer::FindShaderPath(const std::string& shaderName) const
{
    // Search for the compiled .spv file in common locations
    std::vector<std::string> searchPaths = {
        "build/shaders/" + shaderName + ".spv",
        "shaders/" + shaderName + ".spv",
        "../shaders/" + shaderName + ".spv",
        "../../shaders/" + shaderName + ".spv",
    };

    for (const auto& path : searchPaths) {
        if (std::filesystem::exists(path)) {
            return path;
        }
    }

    HRP_LOG_ERROR("Renderer: Shader not found: {}", shaderName);
    return {};
}

VkShaderModule Renderer::LoadShaderModule(const std::string& filename)
{
    std::string path = FindShaderPath(filename);
    if (path.empty()) return VK_NULL_HANDLE;

    std::ifstream file(path, std::ios::ate | std::ios::binary);
    if (!file.is_open()) {
        HRP_LOG_ERROR("Renderer: Failed to open shader file: {}", path);
        return VK_NULL_HANDLE;
    }

    size_t fileSize = static_cast<size_t>(file.tellg());
    std::vector<char> code(fileSize);
    file.seekg(0);
    file.read(code.data(), static_cast<std::streamsize>(fileSize));
    file.close();

    VkShaderModuleCreateInfo createInfo{};
    createInfo.sType    = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    createInfo.codeSize = code.size();
    createInfo.pCode    = reinterpret_cast<const uint32_t*>(code.data());

    VkShaderModule shaderModule;
    VkResult result = vkCreateShaderModule(m_device, &createInfo, nullptr, &shaderModule);
    if (result != VK_SUCCESS) {
        HRP_LOG_ERROR("Renderer: Failed to create shader module from {}", path);
        return VK_NULL_HANDLE;
    }

    HRP_LOG_DEBUG("Renderer: Loaded shader: {} ({} bytes)", filename, fileSize);
    return shaderModule;
}

// ──────────────────────────────────────────────────────────────────────
// Scene color format management
// ──────────────────────────────────────────────────────────────────────

void Renderer::SetSceneColorFormat(VkFormat format)
{
    if (m_sceneColorFormat == format) return;
    m_sceneColorFormat = format;

    // Recreate the 3D pipeline with the new scene color format.
    // Pipeline layout and shadow set layout are format-independent — keep them.
    if (m_3dPipeline != VK_NULL_HANDLE) {
        vkDeviceWaitIdle(m_device);
        vkDestroyPipeline(m_device, m_3dPipeline, nullptr);
        m_3dPipeline = VK_NULL_HANDLE;
        Create3DPipeline();
    }
}

// ──────────────────────────────────────────────────────────────────────
// 3D scene pipeline (extracted for reuse by SetSceneColorFormat)
// ──────────────────────────────────────────────────────────────────────

bool Renderer::Create3DPipeline()
{
    VkShaderModule vertShader3D = LoadShaderModule("basic_3d.vert");
    VkShaderModule fragShader3D = LoadShaderModule("basic_3d.frag");

    if (vertShader3D == VK_NULL_HANDLE || fragShader3D == VK_NULL_HANDLE) {
        HRP_LOG_WARN("Renderer: 3D shaders not found — pipeline creation deferred");
        if (vertShader3D) vkDestroyShaderModule(m_device, vertShader3D, nullptr);
        if (fragShader3D) vkDestroyShaderModule(m_device, fragShader3D, nullptr);
        return true;  // Non-fatal: engine can still clear-render
    }

    VkPipelineShaderStageCreateInfo shaderStages3D[2]{};
    shaderStages3D[0].sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    shaderStages3D[0].stage  = VK_SHADER_STAGE_VERTEX_BIT;
    shaderStages3D[0].module = vertShader3D;
    shaderStages3D[0].pName  = "main";
    shaderStages3D[1].sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    shaderStages3D[1].stage  = VK_SHADER_STAGE_FRAGMENT_BIT;
    shaderStages3D[1].module = fragShader3D;
    shaderStages3D[1].pName  = "main";

    auto bindingDesc = Vertex3D::GetBindingDescription();
    auto attribDescs = Vertex3D::GetAttributeDescriptions();

    VkPipelineVertexInputStateCreateInfo vertexInputInfo{};
    vertexInputInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
    vertexInputInfo.vertexBindingDescriptionCount   = 1;
    vertexInputInfo.pVertexBindingDescriptions       = &bindingDesc;
    vertexInputInfo.vertexAttributeDescriptionCount = static_cast<uint32_t>(attribDescs.size());
    vertexInputInfo.pVertexAttributeDescriptions     = attribDescs.data();

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
    // Draw both sides — the basic mesh shader is flat vertex-colour only and
    // we don't assume a consistent winding convention from caller-authored meshes.
    rasterizer.cullMode    = VK_CULL_MODE_NONE;
    rasterizer.frontFace   = VK_FRONT_FACE_COUNTER_CLOCKWISE;

    VkPipelineMultisampleStateCreateInfo multisampling{};
    multisampling.sType                = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
    multisampling.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

    VkPipelineDepthStencilStateCreateInfo depthStencil{};
    depthStencil.sType            = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
    depthStencil.depthTestEnable  = VK_TRUE;
    depthStencil.depthWriteEnable = VK_TRUE;
    depthStencil.depthCompareOp   = VK_COMPARE_OP_LESS;

    VkPipelineColorBlendAttachmentState colorBlendAttachment{};
    colorBlendAttachment.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT
                                        | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
    colorBlendAttachment.blendEnable = VK_FALSE;

    VkPipelineColorBlendStateCreateInfo colorBlending{};
    colorBlending.sType           = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
    colorBlending.attachmentCount = 1;
    colorBlending.pAttachments    = &colorBlendAttachment;

    std::array<VkDynamicState, 2> dynamicStates = {
        VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR
    };
    VkPipelineDynamicStateCreateInfo dynamicState{};
    dynamicState.sType             = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
    dynamicState.dynamicStateCount = static_cast<uint32_t>(dynamicStates.size());
    dynamicState.pDynamicStates    = dynamicStates.data();

    // Shadow set layout + pipeline layout: create only on first call
    if (m_3dPipelineLayout == VK_NULL_HANDLE) {
        VkDescriptorSetLayoutBinding binding{};
        binding.binding         = 0;
        binding.descriptorType  = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        binding.descriptorCount = 1;
        binding.stageFlags      = VK_SHADER_STAGE_FRAGMENT_BIT;

        VkDescriptorSetLayoutCreateInfo setLayoutInfo{};
        setLayoutInfo.sType        = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        setLayoutInfo.bindingCount = 1;
        setLayoutInfo.pBindings    = &binding;

        if (vkCreateDescriptorSetLayout(m_device, &setLayoutInfo, nullptr,
                                         &m_3dShadowSetLayout) != VK_SUCCESS) {
            HRP_LOG_WARN("Renderer: Failed to create 3D shadow set layout");
            m_3dShadowSetLayout = VK_NULL_HANDLE;
        }

        VkPushConstantRange pushRanges3D[2]{};
        pushRanges3D[0].stageFlags = VK_SHADER_STAGE_VERTEX_BIT;
        pushRanges3D[0].offset     = 0;
        pushRanges3D[0].size       = 128;
        pushRanges3D[1].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
        pushRanges3D[1].offset     = 128;
        pushRanges3D[1].size       = 80;

        VkPipelineLayoutCreateInfo layoutInfo3D{};
        layoutInfo3D.sType                  = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        layoutInfo3D.pushConstantRangeCount = 2;
        layoutInfo3D.pPushConstantRanges    = pushRanges3D;
        layoutInfo3D.setLayoutCount         = (m_3dShadowSetLayout != VK_NULL_HANDLE) ? 1u : 0u;
        layoutInfo3D.pSetLayouts            = &m_3dShadowSetLayout;

        VkResult result = vkCreatePipelineLayout(m_device, &layoutInfo3D, nullptr, &m_3dPipelineLayout);
        if (result != VK_SUCCESS) {
            HRP_LOG_CRITICAL("Renderer: Failed to create 3D pipeline layout");
            return false;
        }
    }

    // Use m_sceneColorFormat — matches swapchain in direct mode, RGBA16F in offscreen mode
    VkPipelineRenderingCreateInfo renderingInfo{};
    renderingInfo.sType                   = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO;
    renderingInfo.colorAttachmentCount    = 1;
    renderingInfo.pColorAttachmentFormats = &m_sceneColorFormat;
    renderingInfo.depthAttachmentFormat   = k_DepthFormat;

    VkGraphicsPipelineCreateInfo pipelineInfo{};
    pipelineInfo.sType               = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
    pipelineInfo.pNext               = &renderingInfo;
    pipelineInfo.stageCount          = 2;
    pipelineInfo.pStages             = shaderStages3D;
    pipelineInfo.pVertexInputState   = &vertexInputInfo;
    pipelineInfo.pInputAssemblyState = &inputAssembly;
    pipelineInfo.pViewportState      = &viewportState;
    pipelineInfo.pRasterizationState = &rasterizer;
    pipelineInfo.pMultisampleState   = &multisampling;
    pipelineInfo.pDepthStencilState  = &depthStencil;
    pipelineInfo.pColorBlendState    = &colorBlending;
    pipelineInfo.pDynamicState       = &dynamicState;
    pipelineInfo.layout              = m_3dPipelineLayout;

    VkResult result = vkCreateGraphicsPipelines(m_device, VK_NULL_HANDLE, 1, &pipelineInfo,
                                                nullptr, &m_3dPipeline);
    vkDestroyShaderModule(m_device, vertShader3D, nullptr);
    vkDestroyShaderModule(m_device, fragShader3D, nullptr);

    if (result != VK_SUCCESS) {
        HRP_LOG_CRITICAL("Renderer: Failed to create 3D graphics pipeline");
        return false;
    }
    HRP_LOG_INFO("Renderer: 3D pipeline created (format: {})",
                 static_cast<int>(m_sceneColorFormat));
    return true;
}

// ──────────────────────────────────────────────────────────────────────
// Pipeline creation
// ──────────────────────────────────────────────────────────────────────

bool Renderer::CreatePipelines()
{
    HRP_LOG_INFO("Renderer: Creating graphics pipelines ...");

    // ── 3D Pipeline ──────────────────────────────────────────────────

    if (!Create3DPipeline()) return false;

    // ── Overlay Pipeline ─────────────────────────────────────────────

    VkShaderModule vertShaderOverlay = LoadShaderModule("fullscreen_tri.vert");
    VkShaderModule fragShaderOverlay = LoadShaderModule("debug_overlay.frag");

    if (vertShaderOverlay == VK_NULL_HANDLE || fragShaderOverlay == VK_NULL_HANDLE) {
        HRP_LOG_WARN("Renderer: Overlay shaders not found — overlay disabled");
        if (vertShaderOverlay) vkDestroyShaderModule(m_device, vertShaderOverlay, nullptr);
        if (fragShaderOverlay) vkDestroyShaderModule(m_device, fragShaderOverlay, nullptr);
        return true;  // Non-fatal
    }

    VkPipelineShaderStageCreateInfo shaderStagesOverlay[2]{};
    shaderStagesOverlay[0].sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    shaderStagesOverlay[0].stage  = VK_SHADER_STAGE_VERTEX_BIT;
    shaderStagesOverlay[0].module = vertShaderOverlay;
    shaderStagesOverlay[0].pName  = "main";
    shaderStagesOverlay[1].sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    shaderStagesOverlay[1].stage  = VK_SHADER_STAGE_FRAGMENT_BIT;
    shaderStagesOverlay[1].module = fragShaderOverlay;
    shaderStagesOverlay[1].pName  = "main";

    // Overlay: no vertex input (fullscreen triangle from gl_VertexIndex)
    VkPipelineVertexInputStateCreateInfo emptyVertexInput{};
    emptyVertexInput.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;

    // Overlay: no depth test
    VkPipelineDepthStencilStateCreateInfo noDepth{};
    noDepth.sType            = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
    noDepth.depthTestEnable  = VK_FALSE;
    noDepth.depthWriteEnable = VK_FALSE;

    // Overlay: alpha blending
    VkPipelineColorBlendAttachmentState overlayBlend{};
    overlayBlend.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT
                                | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
    overlayBlend.blendEnable         = VK_TRUE;
    overlayBlend.srcColorBlendFactor = VK_BLEND_FACTOR_SRC_ALPHA;
    overlayBlend.dstColorBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
    overlayBlend.colorBlendOp        = VK_BLEND_OP_ADD;
    overlayBlend.srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
    overlayBlend.dstAlphaBlendFactor = VK_BLEND_FACTOR_ZERO;
    overlayBlend.alphaBlendOp        = VK_BLEND_OP_ADD;

    VkPipelineColorBlendStateCreateInfo overlayColorBlend{};
    overlayColorBlend.sType           = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
    overlayColorBlend.attachmentCount = 1;
    overlayColorBlend.pAttachments    = &overlayBlend;

    // No culling for overlay
    VkPipelineRasterizationStateCreateInfo overlayRasterizer{};
    overlayRasterizer.sType       = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
    overlayRasterizer.polygonMode = VK_POLYGON_MODE_FILL;
    overlayRasterizer.lineWidth   = 1.0f;
    overlayRasterizer.cullMode    = VK_CULL_MODE_NONE;
    overlayRasterizer.frontFace   = VK_FRONT_FACE_COUNTER_CLOCKWISE;

    // Push constant layout for overlay
    VkPushConstantRange pushRangeOverlay{};
    pushRangeOverlay.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
    pushRangeOverlay.offset     = 0;
    pushRangeOverlay.size       = sizeof(PushConstantsOverlay);

    VkPipelineLayoutCreateInfo layoutInfoOverlay{};
    layoutInfoOverlay.sType                  = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    layoutInfoOverlay.pushConstantRangeCount = 1;
    layoutInfoOverlay.pPushConstantRanges    = &pushRangeOverlay;

    VkResult result = vkCreatePipelineLayout(m_device, &layoutInfoOverlay, nullptr, &m_overlayPipelineLayout);
    if (result != VK_SUCCESS) {
        HRP_LOG_CRITICAL("Renderer: Failed to create overlay pipeline layout");
        return false;
    }

    // Overlay rendering info — no depth attachment
    VkPipelineRenderingCreateInfo overlayRenderingInfo{};
    overlayRenderingInfo.sType                   = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO;
    overlayRenderingInfo.colorAttachmentCount    = 1;
    overlayRenderingInfo.pColorAttachmentFormats = &m_swapchainFormat;
    overlayRenderingInfo.depthAttachmentFormat   = k_DepthFormat;

    std::array<VkDynamicState, 2> dynamicStates = {
        VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR
    };
    VkPipelineDynamicStateCreateInfo dynamicState{};
    dynamicState.sType             = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
    dynamicState.dynamicStateCount = static_cast<uint32_t>(dynamicStates.size());
    dynamicState.pDynamicStates    = dynamicStates.data();

    VkPipelineViewportStateCreateInfo viewportState{};
    viewportState.sType         = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
    viewportState.viewportCount = 1;
    viewportState.scissorCount  = 1;

    VkPipelineMultisampleStateCreateInfo multisampling{};
    multisampling.sType                = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
    multisampling.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

    VkPipelineInputAssemblyStateCreateInfo inputAssembly{};
    inputAssembly.sType    = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
    inputAssembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;

    VkGraphicsPipelineCreateInfo pipelineInfo{};
    pipelineInfo.sType               = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
    pipelineInfo.pNext               = &overlayRenderingInfo;
    pipelineInfo.stageCount          = 2;
    pipelineInfo.pStages             = shaderStagesOverlay;
    pipelineInfo.pVertexInputState   = &emptyVertexInput;
    pipelineInfo.pInputAssemblyState = &inputAssembly;
    pipelineInfo.pViewportState      = &viewportState;
    pipelineInfo.pMultisampleState   = &multisampling;
    pipelineInfo.pDynamicState       = &dynamicState;
    pipelineInfo.pDepthStencilState  = &noDepth;
    pipelineInfo.pColorBlendState    = &overlayColorBlend;
    pipelineInfo.pRasterizationState = &overlayRasterizer;
    pipelineInfo.layout              = m_overlayPipelineLayout;

    result = vkCreateGraphicsPipelines(m_device, VK_NULL_HANDLE, 1, &pipelineInfo,
                                        nullptr, &m_overlayPipeline);
    vkDestroyShaderModule(m_device, vertShaderOverlay, nullptr);
    vkDestroyShaderModule(m_device, fragShaderOverlay, nullptr);

    if (result != VK_SUCCESS) {
        HRP_LOG_CRITICAL("Renderer: Failed to create overlay graphics pipeline");
        return false;
    }
    HRP_LOG_INFO("Renderer: Overlay pipeline created");

    return true;
}

// ──────────────────────────────────────────────────────────────────────
// Vertex3D layout
// ──────────────────────────────────────────────────────────────────────

VkVertexInputBindingDescription Vertex3D::GetBindingDescription()
{
    VkVertexInputBindingDescription binding{};
    binding.binding   = 0;
    binding.stride    = sizeof(Vertex3D);
    binding.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;
    return binding;
}

std::array<VkVertexInputAttributeDescription, 2> Vertex3D::GetAttributeDescriptions()
{
    std::array<VkVertexInputAttributeDescription, 2> attrs{};

    // position: location 0, vec3
    attrs[0].binding  = 0;
    attrs[0].location = 0;
    attrs[0].format   = VK_FORMAT_R32G32B32_SFLOAT;
    attrs[0].offset   = offsetof(Vertex3D, position);

    // color: location 1, vec3
    attrs[1].binding  = 0;
    attrs[1].location = 1;
    attrs[1].format   = VK_FORMAT_R32G32B32_SFLOAT;
    attrs[1].offset   = offsetof(Vertex3D, color);

    return attrs;
}

// ──────────────────────────────────────────────────────────────────────
// Mesh management
// ──────────────────────────────────────────────────────────────────────

bool Renderer::CreateMesh(const std::vector<Vertex3D>& vertices, MeshHandle& outMesh,
                          bool dynamic)
{
    VkDeviceSize bufferSize = sizeof(Vertex3D) * vertices.size();
    if (bufferSize == 0) {
        HRP_LOG_WARN("Renderer: CreateMesh called with 0 vertices");
        return false;
    }

    if (dynamic) {
        // Persistent host-visible vertex buffer — caller intends to call
        // UpdateMesh() repeatedly. Costs process committed memory.
        VkBufferCreateInfo bufferInfo{};
        bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        bufferInfo.size  = bufferSize;
        bufferInfo.usage = VK_BUFFER_USAGE_VERTEX_BUFFER_BIT;

        VmaAllocationCreateInfo allocInfo{};
        allocInfo.usage = VMA_MEMORY_USAGE_CPU_TO_GPU;

        if (vmaCreateBuffer(m_allocator, &bufferInfo, &allocInfo,
                            &outMesh.buffer, &outMesh.allocation, nullptr) != VK_SUCCESS) {
            HRP_LOG_ERROR("Renderer: Failed to create dynamic mesh buffer ({} vertices)",
                          vertices.size());
            return false;
        }

        void* data = nullptr;
        vmaMapMemory(m_allocator, outMesh.allocation, &data);
        std::memcpy(data, vertices.data(), bufferSize);
        vmaUnmapMemory(m_allocator, outMesh.allocation);

        outMesh.vertexCount = static_cast<uint32_t>(vertices.size());
        outMesh.dynamic     = true;
        HRP_LOG_DEBUG("Renderer: Mesh created (dynamic, {} vertices, {} bytes)",
                      outMesh.vertexCount, bufferSize);
        return true;
    }

    // Static path — staging buffer + DEVICE_LOCAL destination. The staging
    // buffer is freed before we return, so the only residual cost is in VRAM.
    //
    // Phase 6 fast path: if a batched-upload window is open, copy the data
    // into the persistent staging ring and record the copy into the
    // batch command buffer. Skips per-mesh staging buffer creation +
    // per-mesh `vkQueueWaitIdle` — both of which are paid back in a
    // single fence wait by `EndUploadBatch`.
    if (m_uploadBatchActive) {
        if (!EnsureStagingRing(m_stagingRingHead + bufferSize)) {
            HRP_LOG_ERROR("Renderer: CreateMesh batched — staging ring grow failed");
            return false;
        }

        VkBufferCreateInfo bufferInfo{};
        bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        bufferInfo.size  = bufferSize;
        bufferInfo.usage = VK_BUFFER_USAGE_VERTEX_BUFFER_BIT |
                           VK_BUFFER_USAGE_TRANSFER_DST_BIT;

        // Phase 6b: when uploads run on a dedicated transfer queue
        // family, the destination buffer must be either CONCURRENT or
        // go through a queue-family ownership transfer barrier on first
        // use by the graphics queue. CONCURRENT is dramatically simpler
        // and the bandwidth penalty on modern GPUs is negligible for
        // vertex buffers.
        uint32_t shareFamilies[2] = { m_graphicsFamily, m_transferFamily };
        if (m_transferIsSeparate) {
            bufferInfo.sharingMode           = VK_SHARING_MODE_CONCURRENT;
            bufferInfo.queueFamilyIndexCount = 2;
            bufferInfo.pQueueFamilyIndices   = shareFamilies;
        } else {
            bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        }

        VmaAllocationCreateInfo allocInfo{};
        allocInfo.usage = VMA_MEMORY_USAGE_GPU_ONLY;

        if (vmaCreateBuffer(m_allocator, &bufferInfo, &allocInfo,
                            &outMesh.buffer, &outMesh.allocation, nullptr) != VK_SUCCESS) {
            HRP_LOG_ERROR("Renderer: CreateMesh batched — VMA dest alloc failed ({} bytes)",
                          bufferSize);
            return false;
        }

        std::memcpy(static_cast<char*>(m_stagingRingMapped) + m_stagingRingHead,
                    vertices.data(), bufferSize);

        VkBufferCopy region{};
        region.srcOffset = m_stagingRingHead;
        region.dstOffset = 0;
        region.size      = bufferSize;
        vkCmdCopyBuffer(m_uploadBatchCmd, m_stagingRingBuf, outMesh.buffer, 1, &region);

        m_stagingRingHead += bufferSize;
        ++m_uploadBatchMeshCount;
        outMesh.vertexCount = static_cast<uint32_t>(vertices.size());
        outMesh.dynamic     = false;
        return true;
    }

    // Static path — staging buffer + DEVICE_LOCAL destination. The staging
    // buffer is freed before we return, so the only residual cost is in VRAM.
    VkBuffer       stagingBuf   = VK_NULL_HANDLE;
    VmaAllocation  stagingAlloc = VK_NULL_HANDLE;
    {
        VkBufferCreateInfo bi{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
        bi.size  = bufferSize;
        bi.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
        VmaAllocationCreateInfo ai{};
        ai.usage = VMA_MEMORY_USAGE_CPU_ONLY;
        if (vmaCreateBuffer(m_allocator, &bi, &ai, &stagingBuf, &stagingAlloc, nullptr) != VK_SUCCESS) {
            HRP_LOG_ERROR("Renderer: Failed to create staging buffer ({} bytes)", bufferSize);
            return false;
        }
        void* mapped = nullptr;
        vmaMapMemory(m_allocator, stagingAlloc, &mapped);
        std::memcpy(mapped, vertices.data(), bufferSize);
        vmaUnmapMemory(m_allocator, stagingAlloc);
    }

    VkBufferCreateInfo bufferInfo{};
    bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bufferInfo.size  = bufferSize;
    bufferInfo.usage = VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;

    VmaAllocationCreateInfo allocInfo{};
    allocInfo.usage = VMA_MEMORY_USAGE_GPU_ONLY;

    if (vmaCreateBuffer(m_allocator, &bufferInfo, &allocInfo,
                        &outMesh.buffer, &outMesh.allocation, nullptr) != VK_SUCCESS) {
        HRP_LOG_ERROR("Renderer: Failed to create static mesh buffer ({} vertices)",
                      vertices.size());
        vmaDestroyBuffer(m_allocator, stagingBuf, stagingAlloc);
        return false;
    }

    // One-shot copy on the graphics queue. Synchronous — we wait idle so we
    // can free the staging buffer immediately. Acceptable because mesh
    // uploads happen during chunk-load (not the hot per-frame path).
    VkCommandBufferAllocateInfo cmdAI{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    cmdAI.commandPool        = m_commandPool;
    cmdAI.level              = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    cmdAI.commandBufferCount = 1;
    VkCommandBuffer cmd = VK_NULL_HANDLE;
    if (vkAllocateCommandBuffers(m_device, &cmdAI, &cmd) != VK_SUCCESS) {
        HRP_LOG_ERROR("Renderer: CreateMesh staging — vkAllocateCommandBuffers failed");
        vmaDestroyBuffer(m_allocator, outMesh.buffer, outMesh.allocation);
        vmaDestroyBuffer(m_allocator, stagingBuf, stagingAlloc);
        outMesh.buffer = VK_NULL_HANDLE;
        outMesh.allocation = VK_NULL_HANDLE;
        return false;
    }

    VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vkBeginCommandBuffer(cmd, &bi);
    VkBufferCopy region{};
    region.size = bufferSize;
    vkCmdCopyBuffer(cmd, stagingBuf, outMesh.buffer, 1, &region);
    vkEndCommandBuffer(cmd);

    VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
    si.commandBufferCount = 1;
    si.pCommandBuffers    = &cmd;
    vkQueueSubmit(m_graphicsQueue, 1, &si, VK_NULL_HANDLE);
    vkQueueWaitIdle(m_graphicsQueue);

    vkFreeCommandBuffers(m_device, m_commandPool, 1, &cmd);
    vmaDestroyBuffer(m_allocator, stagingBuf, stagingAlloc);

    outMesh.vertexCount = static_cast<uint32_t>(vertices.size());
    outMesh.dynamic     = false;
    HRP_LOG_DEBUG("Renderer: Mesh created (static, {} vertices, {} bytes)",
                  outMesh.vertexCount, bufferSize);
    return true;
}

bool Renderer::UpdateMesh(const std::vector<Vertex3D>& vertices, MeshHandle& mesh)
{
    if (mesh.buffer == VK_NULL_HANDLE || mesh.allocation == VK_NULL_HANDLE) return false;
    if (static_cast<uint32_t>(vertices.size()) != mesh.vertexCount) return false;
    if (!mesh.dynamic) {
        // The mesh was created GPU_ONLY and is not host-visible. Caller must
        // recreate via CreateMesh(... dynamic=true) instead.
        HRP_LOG_ERROR("Renderer: UpdateMesh called on a static mesh — caller must recreate as dynamic");
        return false;
    }

    void* data = nullptr;
    vmaMapMemory(m_allocator, mesh.allocation, &data);
    std::memcpy(data, vertices.data(), sizeof(Vertex3D) * vertices.size());
    vmaUnmapMemory(m_allocator, mesh.allocation);
    return true;
}

// ══════════════════════════════════════════════════════════════════════
// Phase 6 — Persistent staging ring + batched mesh upload
// ══════════════════════════════════════════════════════════════════════

bool Renderer::EnsureStagingRing(VkDeviceSize required)
{
    if (m_stagingRingBuf != VK_NULL_HANDLE && required <= m_stagingRingSize) return true;

    VkDeviceSize newSize = (m_stagingRingSize == 0) ? k_StagingRingInitialSize
                                                    : m_stagingRingSize;
    while (newSize < required) newSize *= 2;

    // CRITICAL: if a batch is currently active and we already recorded
    // copies into m_uploadBatchCmd that reference the *current* staging
    // buffer, we cannot just destroy that buffer — the recorded copies
    // would dereference freed VkBuffer/VkDeviceMemory the moment the
    // cmd buffer is submitted, hanging the GPU and freezing the main
    // thread forever inside EndUploadBatch's vkWaitForFences. Flush
    // the in-flight batch (submit + wait) first so the ring is no
    // longer referenced, then we may safely destroy and re-allocate.
    const bool wasBatchActive = m_uploadBatchActive;
    if (wasBatchActive) {
        // End and submit whatever is recorded so far. Mirrors EndUploadBatch
        // but inlined so we can re-open a new batch immediately after.
        if (vkEndCommandBuffer(m_uploadBatchCmd) == VK_SUCCESS &&
            m_uploadBatchMeshCount > 0) {
            VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
            si.commandBufferCount = 1;
            si.pCommandBuffers    = &m_uploadBatchCmd;
            if (vkQueueSubmit(m_transferQueue, 1, &si, m_uploadBatchFence) == VK_SUCCESS) {
                constexpr uint64_t k_TimeoutNs = 5ull * 1000ull * 1000ull * 1000ull;
                if (vkWaitForFences(m_device, 1, &m_uploadBatchFence, VK_TRUE, k_TimeoutNs) == VK_TIMEOUT) {
                    HRP_LOG_ERROR("Renderer: EnsureStagingRing — mid-batch fence wait timed out (5s); skipping reset to avoid races");
                } else {
                    vkResetFences(m_device, 1, &m_uploadBatchFence);
                    m_uploadBatchFenceInFlight = false;
                }
            } else {
                HRP_LOG_ERROR("Renderer: EnsureStagingRing — mid-batch submit failed");
                m_uploadBatchFenceInFlight = false;
            }
        }
        m_uploadBatchActive    = false;
        m_uploadBatchMeshCount = 0;
        m_stagingRingHead      = 0;
    }

    // Flush any pending batch fence before re-allocating — the GPU may
    // still be reading from the old ring.
    if (m_stagingRingBuf != VK_NULL_HANDLE) {
        if (m_uploadBatchFence != VK_NULL_HANDLE) {
            constexpr uint64_t k_TimeoutNs = 5ull * 1000ull * 1000ull * 1000ull;
            if (vkWaitForFences(m_device, 1, &m_uploadBatchFence, VK_TRUE, k_TimeoutNs) == VK_TIMEOUT) {
                HRP_LOG_ERROR("Renderer: EnsureStagingRing — pre-realloc fence wait timed out (5s)");
            } else {
                m_uploadBatchFenceInFlight = false;
            }
        }
        if (m_stagingRingMapped) {
            vmaUnmapMemory(m_allocator, m_stagingRingAlloc);
            m_stagingRingMapped = nullptr;
        }
        vmaDestroyBuffer(m_allocator, m_stagingRingBuf, m_stagingRingAlloc);
        m_stagingRingBuf   = VK_NULL_HANDLE;
        m_stagingRingAlloc = VK_NULL_HANDLE;
        m_stagingRingSize  = 0;
    }

    VkBufferCreateInfo bi{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    bi.size  = newSize;
    bi.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
    VmaAllocationCreateInfo ai{};
    ai.usage = VMA_MEMORY_USAGE_CPU_ONLY;
    VmaAllocationInfo allocInfoOut{};
    if (vmaCreateBuffer(m_allocator, &bi, &ai, &m_stagingRingBuf,
                        &m_stagingRingAlloc, &allocInfoOut) != VK_SUCCESS) {
        HRP_LOG_ERROR("Renderer: staging ring allocation failed ({} bytes)", newSize);
        m_stagingRingBuf = VK_NULL_HANDLE;
        return false;
    }
    if (vmaMapMemory(m_allocator, m_stagingRingAlloc, &m_stagingRingMapped) != VK_SUCCESS ||
        m_stagingRingMapped == nullptr) {
        HRP_LOG_ERROR("Renderer: staging ring vmaMapMemory failed");
        vmaDestroyBuffer(m_allocator, m_stagingRingBuf, m_stagingRingAlloc);
        m_stagingRingBuf = VK_NULL_HANDLE;
        m_stagingRingAlloc = VK_NULL_HANDLE;
        m_stagingRingMapped = nullptr;
        return false;
    }
    m_stagingRingSize = newSize;
    m_stagingRingHead = 0;
    HRP_LOG_INFO("Renderer: staging ring ready ({} KB)", newSize / 1024);

    // If we tore down a live batch above, re-open a fresh one so the
    // caller (still inside CreateMesh) can record into a valid cmd
    // buffer. We can't call BeginUploadBatch() because it would early-out
    // (m_uploadBatchActive is false, but it would also vkWaitForFences
    // again — harmless but wasteful). Inline the reset path.
    if (wasBatchActive) {
        if (vkResetCommandBuffer(m_uploadBatchCmd, 0) != VK_SUCCESS) {
            HRP_LOG_ERROR("Renderer: EnsureStagingRing — re-open vkResetCommandBuffer failed");
            return false;
        }
        VkCommandBufferBeginInfo cbbi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
        cbbi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        if (vkBeginCommandBuffer(m_uploadBatchCmd, &cbbi) != VK_SUCCESS) {
            HRP_LOG_ERROR("Renderer: EnsureStagingRing — re-open vkBeginCommandBuffer failed");
            return false;
        }
        m_uploadBatchActive = true;
    }
    return true;
}

void Renderer::DestroyStagingRing()
{
    if (m_uploadBatchFence != VK_NULL_HANDLE) {
        vkDestroyFence(m_device, m_uploadBatchFence, nullptr);
        m_uploadBatchFence = VK_NULL_HANDLE;
        m_uploadBatchFenceInFlight = false;
    }
    if (m_uploadBatchCmd != VK_NULL_HANDLE) {
        // Phase 6b: free from the pool we actually allocated from.
        VkCommandPool srcPool = (m_transferCommandPool != VK_NULL_HANDLE)
                                  ? m_transferCommandPool : m_commandPool;
        vkFreeCommandBuffers(m_device, srcPool, 1, &m_uploadBatchCmd);
        m_uploadBatchCmd = VK_NULL_HANDLE;
    }
    if (m_stagingRingBuf != VK_NULL_HANDLE) {
        if (m_stagingRingMapped) {
            // Staging rings are explicitly mapped after allocation; always
            // pair that map with an explicit unmap before destruction.
            vmaUnmapMemory(m_allocator, m_stagingRingAlloc);
            m_stagingRingMapped = nullptr;
        }
        vmaDestroyBuffer(m_allocator, m_stagingRingBuf, m_stagingRingAlloc);
        m_stagingRingBuf   = VK_NULL_HANDLE;
        m_stagingRingAlloc = VK_NULL_HANDLE;
        m_stagingRingSize  = 0;
    }
    m_uploadBatchActive    = false;
    m_uploadBatchMeshCount = 0;
    m_stagingRingHead      = 0;
}

bool Renderer::IsUploadBatchBusy() const
{
    if (m_uploadBatchFence == VK_NULL_HANDLE) return false;
    if (!m_uploadBatchFenceInFlight) return false;
    // VK_NOT_READY = GPU still processing. VK_SUCCESS = signaled (idle).
    // Any error = treat as not-busy so caller doesn't deadlock waiting.
    return vkGetFenceStatus(m_device, m_uploadBatchFence) == VK_NOT_READY;
}

void Renderer::BeginUploadBatch()
{
    if (m_uploadBatchActive) return;

    if (m_uploadBatchCmd == VK_NULL_HANDLE) {
        VkCommandBufferAllocateInfo cmdAI{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
        cmdAI.commandPool        = m_transferCommandPool;   // Phase 6b: DMA pool when available.
        cmdAI.level              = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        cmdAI.commandBufferCount = 1;
        if (vkAllocateCommandBuffers(m_device, &cmdAI, &m_uploadBatchCmd) != VK_SUCCESS) {
            HRP_LOG_ERROR("Renderer: BeginUploadBatch — vkAllocateCommandBuffers failed");
            m_uploadBatchCmd = VK_NULL_HANDLE;
            return;
        }
    }
    if (m_uploadBatchFence == VK_NULL_HANDLE) {
        VkFenceCreateInfo fi{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
        if (vkCreateFence(m_device, &fi, nullptr, &m_uploadBatchFence) != VK_SUCCESS) {
            HRP_LOG_ERROR("Renderer: BeginUploadBatch — vkCreateFence failed");
            return;
        }
    } else if (m_uploadBatchFenceInFlight) {
        // Make sure the previous batch finished before we reset the cmd buffer.
        constexpr uint64_t k_TimeoutNs = 5ull * 1000ull * 1000ull * 1000ull;
        if (vkWaitForFences(m_device, 1, &m_uploadBatchFence, VK_TRUE, k_TimeoutNs) == VK_TIMEOUT) {
            HRP_LOG_ERROR("Renderer: BeginUploadBatch — previous batch fence still unsignaled after 5s; recreating fence to recover");
            vkDestroyFence(m_device, m_uploadBatchFence, nullptr);
            VkFenceCreateInfo fi{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
            if (vkCreateFence(m_device, &fi, nullptr, &m_uploadBatchFence) != VK_SUCCESS) {
                HRP_LOG_ERROR("Renderer: BeginUploadBatch — fence recreate failed");
                m_uploadBatchFence = VK_NULL_HANDLE;
                return;
            }
            m_uploadBatchFenceInFlight = false;
        } else {
            m_uploadBatchFenceInFlight = false;
            vkResetFences(m_device, 1, &m_uploadBatchFence);
        }
    } else {
        vkResetFences(m_device, 1, &m_uploadBatchFence);
    }

    if (vkResetCommandBuffer(m_uploadBatchCmd, 0) != VK_SUCCESS) {
        HRP_LOG_ERROR("Renderer: BeginUploadBatch — vkResetCommandBuffer failed");
        return;
    }
    VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    if (vkBeginCommandBuffer(m_uploadBatchCmd, &bi) != VK_SUCCESS) {
        HRP_LOG_ERROR("Renderer: BeginUploadBatch — vkBeginCommandBuffer failed");
        return;
    }
    m_uploadBatchActive    = true;
    m_uploadBatchMeshCount = 0;
    m_stagingRingHead      = 0;
}

void Renderer::EndUploadBatch()
{
    if (!m_uploadBatchActive) return;

    if (vkEndCommandBuffer(m_uploadBatchCmd) != VK_SUCCESS) {
        HRP_LOG_ERROR("Renderer: EndUploadBatch — vkEndCommandBuffer failed");
        m_uploadBatchActive = false;
        return;
    }

    // No copies recorded — skip the submit entirely.
    if (m_uploadBatchMeshCount == 0) {
        m_uploadBatchActive = false;
        m_uploadBatchFenceInFlight = false;
        return;
    }

    VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
    si.commandBufferCount = 1;
    si.pCommandBuffers    = &m_uploadBatchCmd;
    // Phase 6b: submit to the dedicated transfer queue when available.
    // Lets copies overlap with graphics work on supporting hardware.
    if (vkQueueSubmit(m_transferQueue, 1, &si, m_uploadBatchFence) != VK_SUCCESS) {
        HRP_LOG_ERROR("Renderer: EndUploadBatch — vkQueueSubmit failed");
        m_uploadBatchActive = false;
        m_uploadBatchFenceInFlight = false;
        return;
    }
    m_uploadBatchFenceInFlight = true;
    // Single fence wait covers every mesh queued in this batch. We must
    // block here because the staging ring will be reused by the next
    // batch (and the destination buffers will be drawn from immediately).
    // Bounded wait (5 s) so a driver/GPU hang surfaces as a logged error
    // instead of an indefinite main-thread block (the watchdog was
    // catching this as 'vegetation.drain' stuck > 1.5 s forever).
    constexpr uint64_t k_UploadFenceTimeoutNs = 5ull * 1000ull * 1000ull * 1000ull;
    VkResult wr = vkWaitForFences(m_device, 1, &m_uploadBatchFence, VK_TRUE,
                                  k_UploadFenceTimeoutNs);
    if (wr == VK_TIMEOUT) {
        HRP_LOG_ERROR("Renderer: EndUploadBatch — fence wait timed out after 5s "
                      "(meshes={}, stagingHead={} bytes). GPU/driver may be hung; "
                      "skipping reset to avoid use-after-free on staging ring.",
                      m_uploadBatchMeshCount, m_stagingRingHead);
        // Leave m_uploadBatchActive=false but DO NOT reset the staging head —
        // the in-flight copy may still touch it. Better to leak the ring
        // contents this frame than to corrupt them mid-DMA.
        m_uploadBatchActive = false;
        m_uploadBatchMeshCount = 0;
        return;
    } else if (wr != VK_SUCCESS) {
        HRP_LOG_ERROR("Renderer: EndUploadBatch — vkWaitForFences returned {}", (int)wr);
    }

    m_uploadBatchActive    = false;
    m_uploadBatchFenceInFlight = false;
    m_uploadBatchMeshCount = 0;
    m_stagingRingHead      = 0;
}

void Renderer::DestroyMesh(MeshHandle& mesh)
{
    if (mesh.buffer != VK_NULL_HANDLE) {
        DeferBufferDestruction(mesh.buffer, mesh.allocation);
        mesh.buffer      = VK_NULL_HANDLE;
        mesh.allocation  = VK_NULL_HANDLE;
        mesh.vertexCount = 0;
    }
}

void Renderer::DeferBufferDestruction(VkBuffer buffer, VmaAllocation allocation)
{
    if (buffer == VK_NULL_HANDLE) return;
    m_deletionRing[m_deletionRingIndex].push_back({ buffer, allocation });
}

void Renderer::ProcessDeferredDeletions()
{
    // Advance to the next ring slot. The slot we're about to drain was
    // populated k_DeletionRingSize frames ago, so all command buffers
    // that could have referenced those resources have completed (the
    // fence wait in BeginFrame guarantees the frame that used this slot
    // is done).
    m_deletionRingIndex = (m_deletionRingIndex + 1) % k_DeletionRingSize;
    auto& slot = m_deletionRing[m_deletionRingIndex];
    for (auto& db : slot) {
        vmaDestroyBuffer(m_allocator, db.buffer, db.allocation);
    }
    slot.clear();
}

// ──────────────────────────────────────────────────────────────────────
// Draw commands
// ──────────────────────────────────────────────────────────────────────

void Renderer::Bind3DPipeline()
{
    if (m_3dPipeline == VK_NULL_HANDLE) return;

    VkCommandBuffer cmd = m_commandBuffers[m_currentFrame];
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, m_3dPipeline);
}

void Renderer::Set3DLightViewProj(const glm::mat4& lightViewProj)
{
    if (m_3dPipelineLayout == VK_NULL_HANDLE) return;

    VkCommandBuffer cmd = m_commandBuffers[m_currentFrame];
    vkCmdPushConstants(cmd, m_3dPipelineLayout, VK_SHADER_STAGE_VERTEX_BIT,
                       64, sizeof(glm::mat4), &lightViewProj);
}

void Renderer::Bind3DShadowSet(VkDescriptorSet shadowSet)
{
    if (m_3dPipelineLayout == VK_NULL_HANDLE || shadowSet == VK_NULL_HANDLE) return;

    VkCommandBuffer cmd = m_commandBuffers[m_currentFrame];
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
                            m_3dPipelineLayout, 0, 1, &shadowSet, 0, nullptr);
}

void Renderer::Set3DLightingParams(const glm::vec4& lightDir, const glm::vec4& cameraPos,
                                    const glm::vec4& fogParams, const glm::vec4& fogColor,
                                    const glm::vec4& envParams)
{
    if (m_3dPipelineLayout == VK_NULL_HANDLE) return;

    VkCommandBuffer cmd = m_commandBuffers[m_currentFrame];

    // Push 5 × vec4 (80 bytes) at offset 128 for the fragment stage.
    // envParams carries per-biome environment parameters:
    //   x = exposureBias, y = ambientScale, z = fogDensityMul, w = roughnessBase
    struct {
        glm::vec4 lightDir;
        glm::vec4 cameraPos;
        glm::vec4 fogParams;
        glm::vec4 fogColor;
        glm::vec4 envParams;
    } fragParams{ lightDir, cameraPos, fogParams, fogColor, envParams };

    vkCmdPushConstants(cmd, m_3dPipelineLayout, VK_SHADER_STAGE_FRAGMENT_BIT,
                       128, sizeof(fragParams), &fragParams);
}

void Renderer::SuspendScenePass()
{
    VkCommandBuffer cmd = m_commandBuffers[m_currentFrame];
    vkCmdEndRendering(cmd);
}

void Renderer::ResumeScenePass()
{
    VkCommandBuffer cmd = m_commandBuffers[m_currentFrame];

    // Resume the scene rendering with LOAD ops to preserve already-rendered content
    VkRenderingAttachmentInfo colorAttachment{};
    colorAttachment.sType       = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO;
    colorAttachment.imageView   = m_swapchainImageViews[m_imageIndex];
    colorAttachment.imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    colorAttachment.loadOp      = VK_ATTACHMENT_LOAD_OP_LOAD;
    colorAttachment.storeOp     = VK_ATTACHMENT_STORE_OP_STORE;

    VkRenderingAttachmentInfo depthAttachment{};
    depthAttachment.sType       = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO;
    depthAttachment.imageView   = m_depthImageView;
    depthAttachment.imageLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
    depthAttachment.loadOp      = VK_ATTACHMENT_LOAD_OP_LOAD;
    depthAttachment.storeOp     = VK_ATTACHMENT_STORE_OP_DONT_CARE;

    VkRenderingInfo renderInfo{};
    renderInfo.sType                = VK_STRUCTURE_TYPE_RENDERING_INFO;
    renderInfo.renderArea           = { {0, 0}, m_swapchainExtent };
    renderInfo.layerCount           = 1;
    renderInfo.colorAttachmentCount = 1;
    renderInfo.pColorAttachments    = &colorAttachment;
    renderInfo.pDepthAttachment     = &depthAttachment;

    vkCmdBeginRendering(cmd, &renderInfo);

    // Restore viewport and scissor
    VkViewport viewport{};
    viewport.x        = 0.0f;
    viewport.y        = 0.0f;
    viewport.width    = static_cast<float>(m_swapchainExtent.width);
    viewport.height   = static_cast<float>(m_swapchainExtent.height);
    viewport.minDepth = 0.0f;
    viewport.maxDepth = 1.0f;
    vkCmdSetViewport(cmd, 0, 1, &viewport);

    VkRect2D scissor{};
    scissor.offset = { 0, 0 };
    scissor.extent = m_swapchainExtent;
    vkCmdSetScissor(cmd, 0, 1, &scissor);
}

void Renderer::BeginOverlayPass(bool preserveContents)
{
    VkCommandBuffer cmd = m_commandBuffers[m_currentFrame];

    VkClearValue clearValue{};
    clearValue.color = {{ m_clearColor.r, m_clearColor.g, m_clearColor.b, m_clearColor.a }};

    VkRenderingAttachmentInfo colorAttachment{};
    colorAttachment.sType       = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO;
    colorAttachment.imageView   = m_swapchainImageViews[m_imageIndex];
    colorAttachment.imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    colorAttachment.loadOp      = preserveContents ? VK_ATTACHMENT_LOAD_OP_LOAD
                                                   : VK_ATTACHMENT_LOAD_OP_CLEAR;
    colorAttachment.storeOp     = VK_ATTACHMENT_STORE_OP_STORE;
    colorAttachment.clearValue  = clearValue;

    VkRenderingInfo renderInfo{};
    renderInfo.sType                = VK_STRUCTURE_TYPE_RENDERING_INFO;
    renderInfo.renderArea           = { {0, 0}, m_swapchainExtent };
    renderInfo.layerCount           = 1;
    renderInfo.colorAttachmentCount = 1;
    renderInfo.pColorAttachments    = &colorAttachment;

    vkCmdBeginRendering(cmd, &renderInfo);

    VkViewport viewport{};
    viewport.x        = 0.0f;
    viewport.y        = 0.0f;
    viewport.width    = static_cast<float>(m_swapchainExtent.width);
    viewport.height   = static_cast<float>(m_swapchainExtent.height);
    viewport.minDepth = 0.0f;
    viewport.maxDepth = 1.0f;
    vkCmdSetViewport(cmd, 0, 1, &viewport);

    VkRect2D scissor{};
    scissor.offset = { 0, 0 };
    scissor.extent = m_swapchainExtent;
    vkCmdSetScissor(cmd, 0, 1, &scissor);
}

void Renderer::EndOverlayPass()
{
    VkCommandBuffer cmd = m_commandBuffers[m_currentFrame];
    vkCmdEndRendering(cmd);
}

void Renderer::DrawMesh(const MeshHandle& mesh, const glm::mat4& mvp)
{
    if (m_3dPipeline == VK_NULL_HANDLE || mesh.buffer == VK_NULL_HANDLE) return;

    VkCommandBuffer cmd = m_commandBuffers[m_currentFrame];

    // Push MVP matrix
    PushConstants3D pc{};
    pc.mvp = mvp;
    vkCmdPushConstants(cmd, m_3dPipelineLayout, VK_SHADER_STAGE_VERTEX_BIT,
                       0, sizeof(PushConstants3D), &pc);

    // Bind vertex buffer and draw
    VkBuffer     buffers[] = { mesh.buffer };
    VkDeviceSize offsets[] = { 0 };
    vkCmdBindVertexBuffers(cmd, 0, 1, buffers, offsets);
    vkCmdDraw(cmd, mesh.vertexCount, 1, 0, 0);
}

void Renderer::DrawDebugOverlay(const glm::vec3& playerPos, const glm::vec3& cameraPos,
                                float mode, float extraData)
{
    if (m_overlayPipeline == VK_NULL_HANDLE) return;

    VkCommandBuffer cmd = m_commandBuffers[m_currentFrame];

    // Switch to overlay pipeline
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, m_overlayPipeline);

    // Push overlay data
    PushConstantsOverlay pc{};
    pc.playerPos  = glm::vec4(playerPos, 0.0f);
    pc.cameraPos  = glm::vec4(cameraPos, 0.0f);
    pc.screenInfo = glm::vec4(static_cast<float>(m_swapchainExtent.width),
                              static_cast<float>(m_swapchainExtent.height),
                              mode, extraData);

    vkCmdPushConstants(cmd, m_overlayPipelineLayout, VK_SHADER_STAGE_FRAGMENT_BIT,
                       0, sizeof(PushConstantsOverlay), &pc);

    // Draw fullscreen triangle (3 vertices, no vertex buffer)
    vkCmdDraw(cmd, 3, 1, 0, 0);
}

// ──────────────────────────────────────────────────────────────────────
// PNG render helper resources (lazy-init dummy shadow set)
// ──────────────────────────────────────────────────────────────────────

bool Renderer::EnsurePNGRenderResources()
{
    if (m_pngResourcesReady) return true;
    if (m_device == VK_NULL_HANDLE || m_allocator == VK_NULL_HANDLE) return false;

    // 1×1 D32_SFLOAT image cleared to 1.0 (shadow lookup returns "fully lit").
    VkImageCreateInfo ici{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
    ici.imageType   = VK_IMAGE_TYPE_2D;
    ici.format      = VK_FORMAT_D32_SFLOAT;
    ici.extent      = {1, 1, 1};
    ici.mipLevels   = 1;
    ici.arrayLayers = 1;
    ici.samples     = VK_SAMPLE_COUNT_1_BIT;
    ici.tiling      = VK_IMAGE_TILING_OPTIMAL;
    ici.usage       = VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT
                    | VK_IMAGE_USAGE_SAMPLED_BIT
                    | VK_IMAGE_USAGE_TRANSFER_DST_BIT;

    VmaAllocationCreateInfo aci{};
    aci.usage = VMA_MEMORY_USAGE_GPU_ONLY;
    if (vmaCreateImage(m_allocator, &ici, &aci, &m_pngDummyShadowImage,
                       &m_pngDummyShadowAlloc, nullptr) != VK_SUCCESS) {
        HRP_LOG_WARN("Renderer: Failed to create PNG dummy shadow image");
        return false;
    }

    // Image view (depth aspect).
    VkImageViewCreateInfo vci{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
    vci.image    = m_pngDummyShadowImage;
    vci.viewType = VK_IMAGE_VIEW_TYPE_2D;
    vci.format   = VK_FORMAT_D32_SFLOAT;
    vci.subresourceRange = {VK_IMAGE_ASPECT_DEPTH_BIT, 0, 1, 0, 1};
    vkCreateImageView(m_device, &vci, nullptr, &m_pngDummyShadowView);

    // Comparison sampler (LESS — depth=1.0 means every fragment is "lit").
    VkSamplerCreateInfo sci{VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO};
    sci.magFilter    = VK_FILTER_NEAREST;
    sci.minFilter    = VK_FILTER_NEAREST;
    sci.mipmapMode   = VK_SAMPLER_MIPMAP_MODE_NEAREST;
    sci.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    sci.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    sci.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    sci.compareEnable = VK_TRUE;
    sci.compareOp     = VK_COMPARE_OP_LESS;
    sci.borderColor   = VK_BORDER_COLOR_FLOAT_OPAQUE_WHITE;
    vkCreateSampler(m_device, &sci, nullptr, &m_pngShadowSampler);

    // Transition to SHADER_READ_ONLY + clear to 1.0 via a one-shot cmd.
    {
        VkCommandBufferAllocateInfo ai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
        ai.commandPool        = m_commandPool;
        ai.level              = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        ai.commandBufferCount = 1;
        VkCommandBuffer cmd;
        vkAllocateCommandBuffers(m_device, &ai, &cmd);

        VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
        bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        vkBeginCommandBuffer(cmd, &bi);

        // UNDEFINED → TRANSFER_DST for clear.
        VkImageMemoryBarrier bar{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
        bar.oldLayout     = VK_IMAGE_LAYOUT_UNDEFINED;
        bar.newLayout     = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
        bar.srcAccessMask = 0;
        bar.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        bar.image         = m_pngDummyShadowImage;
        bar.subresourceRange = {VK_IMAGE_ASPECT_DEPTH_BIT, 0, 1, 0, 1};
        vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                             VK_PIPELINE_STAGE_TRANSFER_BIT,
                             0, 0, nullptr, 0, nullptr, 1, &bar);

        VkClearDepthStencilValue clearVal{1.0f, 0};
        VkImageSubresourceRange range{VK_IMAGE_ASPECT_DEPTH_BIT, 0, 1, 0, 1};
        vkCmdClearDepthStencilImage(cmd, m_pngDummyShadowImage,
                                    VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                                    &clearVal, 1, &range);

        // TRANSFER_DST → DEPTH_STENCIL_READ_ONLY (for sampling in fragment shader).
        bar.oldLayout     = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
        bar.newLayout     = VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL;
        bar.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        bar.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
        vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT,
                             VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
                             0, 0, nullptr, 0, nullptr, 1, &bar);

        vkEndCommandBuffer(cmd);

        VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
        si.commandBufferCount = 1;
        si.pCommandBuffers    = &cmd;
        vkQueueSubmit(m_graphicsQueue, 1, &si, VK_NULL_HANDLE);
        vkQueueWaitIdle(m_graphicsQueue);
        vkFreeCommandBuffers(m_device, m_commandPool, 1, &cmd);
    }

    // Descriptor pool + set.
    VkDescriptorPoolSize poolSize{VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1};
    VkDescriptorPoolCreateInfo dpi{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
    dpi.maxSets       = 1;
    dpi.poolSizeCount = 1;
    dpi.pPoolSizes    = &poolSize;
    vkCreateDescriptorPool(m_device, &dpi, nullptr, &m_pngDescPool);

    VkDescriptorSetAllocateInfo dsai{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
    dsai.descriptorPool     = m_pngDescPool;
    dsai.descriptorSetCount = 1;
    dsai.pSetLayouts        = &m_3dShadowSetLayout;
    vkAllocateDescriptorSets(m_device, &dsai, &m_pngShadowSet);

    VkDescriptorImageInfo imgInfo{};
    imgInfo.sampler     = m_pngShadowSampler;
    imgInfo.imageView   = m_pngDummyShadowView;
    imgInfo.imageLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL;

    VkWriteDescriptorSet write{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
    write.dstSet          = m_pngShadowSet;
    write.dstBinding      = 0;
    write.descriptorCount = 1;
    write.descriptorType  = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    write.pImageInfo      = &imgInfo;
    vkUpdateDescriptorSets(m_device, 1, &write, 0, nullptr);

    m_pngResourcesReady = true;
    HRP_LOG_INFO("Renderer: PNG render resources initialized (1×1 dummy shadow)");
    return true;
}

void Renderer::DestroyPNGRenderResources()
{
    if (!m_pngResourcesReady) return;

    if (m_pngDescPool != VK_NULL_HANDLE) {
        vkDestroyDescriptorPool(m_device, m_pngDescPool, nullptr);
        m_pngDescPool  = VK_NULL_HANDLE;
        m_pngShadowSet = VK_NULL_HANDLE;
    }
    if (m_pngShadowSampler != VK_NULL_HANDLE) {
        vkDestroySampler(m_device, m_pngShadowSampler, nullptr);
        m_pngShadowSampler = VK_NULL_HANDLE;
    }
    if (m_pngDummyShadowView != VK_NULL_HANDLE) {
        vkDestroyImageView(m_device, m_pngDummyShadowView, nullptr);
        m_pngDummyShadowView = VK_NULL_HANDLE;
    }
    if (m_pngDummyShadowImage != VK_NULL_HANDLE) {
        vmaDestroyImage(m_allocator, m_pngDummyShadowImage, m_pngDummyShadowAlloc);
        m_pngDummyShadowImage = VK_NULL_HANDLE;
        m_pngDummyShadowAlloc = VK_NULL_HANDLE;
    }
    m_pngResourcesReady = false;
}

// ──────────────────────────────────────────────────────────────────────
// Offscreen render-to-PNG
// ──────────────────────────────────────────────────────────────────────

bool Renderer::RenderGeomToPNG(const std::vector<Vertex3D>& vertices,
                               const glm::vec3& boundsMin,
                               const glm::vec3& boundsMax,
                               const std::string& pngPath,
                               uint32_t width, uint32_t height)
{
    if (vertices.empty() || !m_initialized || m_3dPipeline == VK_NULL_HANDLE) return false;
    if (!EnsurePNGRenderResources()) return false;

    WaitIdle();

    const VkFormat colorFmt    = m_sceneColorFormat;
    const VkFormat readbackFmt = VK_FORMAT_R8G8B8A8_UNORM;
    const VkFormat depthFmt    = k_DepthFormat;

    // ── 1. Create per-render offscreen resources ─────────────────

    // Color attachment (matches 3D pipeline format).
    VkImage colorImage; VmaAllocation colorAlloc; VkImageView colorView;
    {
        VkImageCreateInfo ci{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
        ci.imageType   = VK_IMAGE_TYPE_2D;
        ci.format      = colorFmt;
        ci.extent      = {width, height, 1};
        ci.mipLevels   = 1;
        ci.arrayLayers = 1;
        ci.samples     = VK_SAMPLE_COUNT_1_BIT;
        ci.tiling      = VK_IMAGE_TILING_OPTIMAL;
        ci.usage       = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT;

        VmaAllocationCreateInfo ai{};
        ai.usage = VMA_MEMORY_USAGE_GPU_ONLY;
        vmaCreateImage(m_allocator, &ci, &ai, &colorImage, &colorAlloc, nullptr);

        VkImageViewCreateInfo vi{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
        vi.image    = colorImage;
        vi.viewType = VK_IMAGE_VIEW_TYPE_2D;
        vi.format   = colorFmt;
        vi.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
        vkCreateImageView(m_device, &vi, nullptr, &colorView);
    }

    // Depth attachment.
    VkImage depthImg; VmaAllocation depthAlloc; VkImageView depthView;
    {
        VkImageCreateInfo ci{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
        ci.imageType   = VK_IMAGE_TYPE_2D;
        ci.format      = depthFmt;
        ci.extent      = {width, height, 1};
        ci.mipLevels   = 1;
        ci.arrayLayers = 1;
        ci.samples     = VK_SAMPLE_COUNT_1_BIT;
        ci.tiling      = VK_IMAGE_TILING_OPTIMAL;
        ci.usage       = VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT;

        VmaAllocationCreateInfo ai{};
        ai.usage = VMA_MEMORY_USAGE_GPU_ONLY;
        vmaCreateImage(m_allocator, &ci, &ai, &depthImg, &depthAlloc, nullptr);

        VkImageViewCreateInfo vi{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
        vi.image    = depthImg;
        vi.viewType = VK_IMAGE_VIEW_TYPE_2D;
        vi.format   = depthFmt;
        vi.subresourceRange = {VK_IMAGE_ASPECT_DEPTH_BIT, 0, 1, 0, 1};
        vkCreateImageView(m_device, &vi, nullptr, &depthView);
    }

    // Readback image (R8G8B8A8_UNORM — blit target for format conversion).
    VkImage readbackImage; VmaAllocation readbackAlloc;
    {
        VkImageCreateInfo ci{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
        ci.imageType   = VK_IMAGE_TYPE_2D;
        ci.format      = readbackFmt;
        ci.extent      = {width, height, 1};
        ci.mipLevels   = 1;
        ci.arrayLayers = 1;
        ci.samples     = VK_SAMPLE_COUNT_1_BIT;
        ci.tiling      = VK_IMAGE_TILING_OPTIMAL;
        ci.usage       = VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT;

        VmaAllocationCreateInfo ai{};
        ai.usage = VMA_MEMORY_USAGE_GPU_ONLY;
        vmaCreateImage(m_allocator, &ci, &ai, &readbackImage, &readbackAlloc, nullptr);
    }

    // Staging buffer for CPU readback.
    const VkDeviceSize bufSize = static_cast<VkDeviceSize>(width) * height * 4u;
    VkBuffer stagingBuf; VmaAllocation stagingAlloc;
    {
        VkBufferCreateInfo bi{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
        bi.size  = bufSize;
        bi.usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT;

        VmaAllocationCreateInfo ai{};
        ai.usage = VMA_MEMORY_USAGE_GPU_TO_CPU;
        vmaCreateBuffer(m_allocator, &bi, &ai, &stagingBuf, &stagingAlloc, nullptr);
    }

    // ── 2. Upload vertex data to GPU ─────────────────────────────
    MeshHandle mesh;
    if (!CreateMesh(vertices, mesh)) {
        HRP_LOG_WARN("Renderer: RenderGeomToPNG — CreateMesh failed");
        // Cleanup created resources.
        vkDestroyImageView(m_device, depthView, nullptr);
        vkDestroyImageView(m_device, colorView, nullptr);
        vmaDestroyImage(m_allocator, readbackImage, readbackAlloc);
        vmaDestroyImage(m_allocator, depthImg, depthAlloc);
        vmaDestroyImage(m_allocator, colorImage, colorAlloc);
        vmaDestroyBuffer(m_allocator, stagingBuf, stagingAlloc);
        return false;
    }

    // ── 3. Compute auto-fit camera ───────────────────────────────
    glm::vec3 center = (boundsMin + boundsMax) * 0.5f;
    glm::vec3 size   = boundsMax - boundsMin;
    float maxDim     = std::max({size.x, size.y, size.z});
    if (maxDim < 0.001f) maxDim = 1.0f;

    // Frame the object at ~75% of viewport height.  Tighter than the
    // naive 2× approach so small objects (flower, fungi) are readable.
    float halfFov    = glm::radians(45.0f * 0.5f);
    float fitDist    = (maxDim * 0.5f) / (std::tan(halfFov) * 0.75f);
    float distance   = std::max(fitDist, 0.3f);

    float elev     = glm::radians(30.0f);
    float azim     = glm::radians(45.0f);
    glm::vec3 camDir(std::cos(elev) * std::cos(azim),
                     std::sin(elev),
                     std::cos(elev) * std::sin(azim));
    glm::vec3 camPos = center + camDir * distance;

    glm::mat4 view = glm::lookAt(camPos, center, glm::vec3(0.0f, 1.0f, 0.0f));
    float aspect   = static_cast<float>(width) / static_cast<float>(height);
    glm::mat4 proj = glm::perspective(glm::radians(45.0f), aspect, 0.01f, distance * 4.0f);
    proj[1][1] *= -1.0f;  // Vulkan Y-flip
    glm::mat4 mvp = proj * view;

    // Light-view-proj (identity — no shadow contribution).
    glm::mat4 lightVP = glm::mat4(1.0f);

    // ── 4. Record one-shot command buffer ────────────────────────
    VkCommandBufferAllocateInfo cmdAI{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    cmdAI.commandPool        = m_commandPool;
    cmdAI.level              = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    cmdAI.commandBufferCount = 1;
    VkCommandBuffer cmd;
    vkAllocateCommandBuffers(m_device, &cmdAI, &cmd);

    VkCommandBufferBeginInfo beginI{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    beginI.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vkBeginCommandBuffer(cmd, &beginI);

    // Transition color: UNDEFINED → COLOR_ATTACHMENT_OPTIMAL.
    {
        VkImageMemoryBarrier b{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
        b.oldLayout     = VK_IMAGE_LAYOUT_UNDEFINED;
        b.newLayout     = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
        b.srcAccessMask = 0;
        b.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
        b.image         = colorImage;
        b.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
        vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                             VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
                             0, 0, nullptr, 0, nullptr, 1, &b);
    }
    // Transition depth: UNDEFINED → DEPTH_STENCIL_ATTACHMENT_OPTIMAL.
    {
        VkImageMemoryBarrier b{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
        b.oldLayout     = VK_IMAGE_LAYOUT_UNDEFINED;
        b.newLayout     = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
        b.srcAccessMask = 0;
        b.dstAccessMask = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
        b.image         = depthImg;
        b.subresourceRange = {VK_IMAGE_ASPECT_DEPTH_BIT, 0, 1, 0, 1};
        vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                             VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT,
                             0, 0, nullptr, 0, nullptr, 1, &b);
    }

    // Begin dynamic rendering on offscreen images.
    VkRenderingAttachmentInfo colorAtt{VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO};
    colorAtt.imageView   = colorView;
    colorAtt.imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    colorAtt.loadOp      = VK_ATTACHMENT_LOAD_OP_CLEAR;
    colorAtt.storeOp     = VK_ATTACHMENT_STORE_OP_STORE;
    colorAtt.clearValue.color = {{0.40f, 0.50f, 0.65f, 1.0f}};  // Sky blue (linear)

    VkRenderingAttachmentInfo depthAtt{VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO};
    depthAtt.imageView   = depthView;
    depthAtt.imageLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
    depthAtt.loadOp      = VK_ATTACHMENT_LOAD_OP_CLEAR;
    depthAtt.storeOp     = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    depthAtt.clearValue.depthStencil = {1.0f, 0};

    VkRenderingInfo ri{VK_STRUCTURE_TYPE_RENDERING_INFO};
    ri.renderArea            = {{0, 0}, {width, height}};
    ri.layerCount            = 1;
    ri.colorAttachmentCount  = 1;
    ri.pColorAttachments     = &colorAtt;
    ri.pDepthAttachment      = &depthAtt;
    vkCmdBeginRendering(cmd, &ri);

    // Viewport + scissor.
    VkViewport vp{0.0f, 0.0f, static_cast<float>(width), static_cast<float>(height), 0.0f, 1.0f};
    vkCmdSetViewport(cmd, 0, 1, &vp);
    VkRect2D sc{{0, 0}, {width, height}};
    vkCmdSetScissor(cmd, 0, 1, &sc);

    // Bind 3D pipeline + dummy shadow set.
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, m_3dPipeline);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
                            m_3dPipelineLayout, 0, 1, &m_pngShadowSet, 0, nullptr);

    // Push vertex constants: MVP (0-64) + lightViewProj (64-128).
    vkCmdPushConstants(cmd, m_3dPipelineLayout, VK_SHADER_STAGE_VERTEX_BIT,
                       0, sizeof(glm::mat4), &mvp);
    vkCmdPushConstants(cmd, m_3dPipelineLayout, VK_SHADER_STAGE_VERTEX_BIT,
                       64, sizeof(glm::mat4), &lightVP);

    // Push fragment constants: daylight lighting (128-208).
    // Shader computes directIntensity = 1.0 - ambient, so ambient=0.35 gives
    // 0.65 direct sunlight.  The rt_composite tone-mapping pass is skipped;
    // sRGB gamma is applied in CPU after readback (see step 7).
    struct { glm::vec4 a, b, c, d, e; } fragPC = {
        glm::vec4(glm::normalize(glm::vec3(0.4f, 0.9f, 0.3f)), 0.35f), // lightDir (high sun) + ambient
        glm::vec4(camPos, 0.005f),                                       // cameraPos + shadow bias
        glm::vec4(500.0f, 50000.0f, 0.0f, 0.0f),                       // fogParams (distant = no effect)
        glm::vec4(0.55f, 0.62f, 0.75f, 0.0f),                          // fogColor (sky) + shadow strength 0
        glm::vec4(0.3f, 1.2f, 0.001f, 0.65f)                           // envParams: exposure +0.3 EV, ambScale 1.2, fogDens ~0, rough 0.65
    };
    vkCmdPushConstants(cmd, m_3dPipelineLayout, VK_SHADER_STAGE_FRAGMENT_BIT,
                       128, sizeof(fragPC), &fragPC);

    // Draw the mesh.
    VkBuffer     bufs[]    = {mesh.buffer};
    VkDeviceSize offsets[] = {0};
    vkCmdBindVertexBuffers(cmd, 0, 1, bufs, offsets);
    vkCmdDraw(cmd, mesh.vertexCount, 1, 0, 0);

    vkCmdEndRendering(cmd);

    // ── 5. Readback: blit sceneFormat → R8G8B8A8_UNORM → staging ─

    // color: COLOR_ATTACHMENT → TRANSFER_SRC.
    {
        VkImageMemoryBarrier b{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
        b.oldLayout     = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
        b.newLayout     = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
        b.srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
        b.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
        b.image         = colorImage;
        b.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
        vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
                             VK_PIPELINE_STAGE_TRANSFER_BIT,
                             0, 0, nullptr, 0, nullptr, 1, &b);
    }
    // readback: UNDEFINED → TRANSFER_DST.
    {
        VkImageMemoryBarrier b{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
        b.oldLayout     = VK_IMAGE_LAYOUT_UNDEFINED;
        b.newLayout     = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
        b.srcAccessMask = 0;
        b.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        b.image         = readbackImage;
        b.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
        vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                             VK_PIPELINE_STAGE_TRANSFER_BIT,
                             0, 0, nullptr, 0, nullptr, 1, &b);
    }

    // Blit: sceneColorFormat → R8G8B8A8_UNORM (GPU format conversion).
    VkImageBlit blit{};
    blit.srcSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
    blit.srcOffsets[0]  = {0, 0, 0};
    blit.srcOffsets[1]  = {static_cast<int32_t>(width), static_cast<int32_t>(height), 1};
    blit.dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
    blit.dstOffsets[0]  = {0, 0, 0};
    blit.dstOffsets[1]  = {static_cast<int32_t>(width), static_cast<int32_t>(height), 1};
    vkCmdBlitImage(cmd,
                   colorImage,    VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                   readbackImage, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                   1, &blit, VK_FILTER_NEAREST);

    // readback: TRANSFER_DST → TRANSFER_SRC.
    {
        VkImageMemoryBarrier b{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
        b.oldLayout     = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
        b.newLayout     = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
        b.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        b.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
        b.image         = readbackImage;
        b.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
        vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT,
                             VK_PIPELINE_STAGE_TRANSFER_BIT,
                             0, 0, nullptr, 0, nullptr, 1, &b);
    }

    // Copy readback image → staging buffer.
    VkBufferImageCopy copyRgn{};
    copyRgn.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
    copyRgn.imageExtent      = {width, height, 1};
    vkCmdCopyImageToBuffer(cmd, readbackImage, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                           stagingBuf, 1, &copyRgn);

    // Staging buffer visibility barrier.
    {
        VkBufferMemoryBarrier b{VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER};
        b.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        b.dstAccessMask = VK_ACCESS_HOST_READ_BIT;
        b.buffer        = stagingBuf;
        b.offset        = 0;
        b.size          = bufSize;
        vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT,
                             VK_PIPELINE_STAGE_HOST_BIT,
                             0, 0, nullptr, 1, &b, 0, nullptr);
    }

    vkEndCommandBuffer(cmd);

    // ── 6. Submit and wait ───────────────────────────────────────
    VkFenceCreateInfo fci{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
    VkFence fence;
    vkCreateFence(m_device, &fci, nullptr, &fence);

    VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
    si.commandBufferCount = 1;
    si.pCommandBuffers    = &cmd;
    vkQueueSubmit(m_graphicsQueue, 1, &si, fence);
    vkWaitForFences(m_device, 1, &fence, VK_TRUE, UINT64_MAX);

    // ── 7. Map staging buffer → apply sRGB gamma → write PNG ─────
    // The fragment shader outputs linear HDR (tone mapping normally happens
    // in rt_composite.frag).  The blit to R8G8B8A8_UNORM clamps to [0,1]
    // but leaves values in linear space.  Apply the sRGB transfer function
    // so the PNG displays correctly on standard monitors.
    void* mapped = nullptr;
    vmaMapMemory(m_allocator, stagingAlloc, &mapped);
    {
        uint8_t* pixels = static_cast<uint8_t*>(mapped);
        const uint32_t pixelCount = width * height;
        for (uint32_t i = 0; i < pixelCount; ++i) {
            for (int c = 0; c < 3; ++c) {
                float v = static_cast<float>(pixels[i * 4 + c]) / 255.0f;
                v = (v <= 0.0031308f) ? (v * 12.92f)
                                      : (1.055f * std::pow(v, 1.0f / 2.4f) - 0.055f);
                pixels[i * 4 + c] = static_cast<uint8_t>(std::min(v * 255.0f + 0.5f, 255.0f));
            }
        }
    }
    bool ok = stbi_write_png(pngPath.c_str(),
                             static_cast<int>(width), static_cast<int>(height),
                             4, mapped, static_cast<int>(width * 4)) != 0;
    vmaUnmapMemory(m_allocator, stagingAlloc);

    if (ok) {
        HRP_LOG_INFO("Renderer: PNG saved — {}", pngPath);
    } else {
        HRP_LOG_WARN("Renderer: PNG write failed — {}", pngPath);
    }

    // ── 8. Cleanup per-render resources ──────────────────────────
    vkDestroyFence(m_device, fence, nullptr);
    vkFreeCommandBuffers(m_device, m_commandPool, 1, &cmd);
    DestroyMesh(mesh);
    vkDestroyImageView(m_device, depthView, nullptr);
    vkDestroyImageView(m_device, colorView, nullptr);
    vmaDestroyImage(m_allocator, readbackImage, readbackAlloc);
    vmaDestroyImage(m_allocator, depthImg, depthAlloc);
    vmaDestroyImage(m_allocator, colorImage, colorAlloc);
    vmaDestroyBuffer(m_allocator, stagingBuf, stagingAlloc);

    return ok;
}

} // namespace hrp
