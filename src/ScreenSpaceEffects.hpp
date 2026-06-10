#pragma once

/// @file ScreenSpaceEffects.hpp
/// @brief Screen-space post-processing infrastructure — manages intermediate
///        render targets and dispatches compute-based screen-space effects.
///
/// This system implements the "2D compositing" rendering pipeline:
///   1. Scene renders to offscreen color + depth + normal buffers (not swapchain)
///   2. Hi-Z depth pyramid is built from the depth buffer
///   3. Screen-space compute shaders run over the 2D buffers:
///      - GTAO+ (ambient occlusion + bent normals)
///      - SVRM (screen-space global illumination)
///      - SHR (screen-space reflections)
///      - ACTF (temporal denoising)
///   4. RT composite pass blends all results onto the swapchain
///
/// The compute shaders treat the 3D scene as flat 2D textures — shadows,
/// lighting, and effects are computed entirely in screen space. This is
/// the efficient "2D rendering over 3D geometry" approach.
///
/// Integration points:
///   - Renderer creates offscreen targets and calls this system
///   - Engine::RenderInGame() inserts the compute dispatch between
///     scene rendering and UI overlay

#include <vulkan/vulkan.h>
#include <vk_mem_alloc.h>
#include <glm/glm.hpp>
#include <array>
#include <string>

namespace hrp {

class Renderer;

/// Configuration for screen-space effects.
struct ScreenSpaceConfig {
    bool  enableAO          = false;  ///< GTAO+ ambient occlusion  (disabled: requires MRT normals — Phase 2)
    bool  enableGI          = false;  ///< SVRM screen-space GI     (disabled: requires MRT normals — Phase 2)
    bool  enableReflections = false;  ///< SHR screen-space reflections (disabled: requires MRT normals — Phase 2)
    bool  enableDenoise     = false;  ///< ACTF temporal denoising  (disabled: no input without AO/GI/reflections)
    bool  enableComposite   = true;   ///< RT composite pass

    float aoIntensity       = 1.0f;   ///< AO strength multiplier
    float giIntensity       = 0.8f;   ///< GI strength multiplier
    float reflIntensity     = 0.6f;   ///< Reflection strength multiplier

    float aoShortRadius     = 4.0f;   ///< GTAO+ short-range radius (texels)
    float aoLongRadius      = 24.0f;  ///< GTAO+ long-range radius (texels)
    float aoBias            = 0.01f;  ///< AO depth bias
    float aoPower           = 1.5f;   ///< AO power curve

    float giRadius          = 0.2f;   ///< GI trace radius (UV space)
    float giThickness       = 0.05f;  ///< GI depth thickness for hit test
    float giTemporalBlend   = 0.9f;   ///< GI temporal accumulation factor

    float reflMaxDist       = 0.4f;   ///< Reflection max trace distance (UV)
    float reflThickness     = 0.03f;  ///< Reflection depth thickness
    float reflTemporalBlend = 0.85f;  ///< Reflection temporal blend

    float chromaticAberration = 0.0f; ///< CA strength
    float filmGrain         = 0.0f;  ///< Film grain strength
    float vignette          = 0.3f;   ///< Vignette strength
    float exposure          = 1.0f;   ///< Final exposure
};

/// Manages GPU resources for an offscreen render target (image + view + memory).
struct OffscreenImage {
    VkImage       image      = VK_NULL_HANDLE;
    VmaAllocation allocation = VK_NULL_HANDLE;
    VkImageView   view       = VK_NULL_HANDLE;
    VkFormat      format     = VK_FORMAT_UNDEFINED;
    uint32_t      width      = 0;
    uint32_t      height     = 0;
    uint32_t      mipLevels  = 1;

    /// Additional views for specific mip levels (used by Hi-Z).
    std::array<VkImageView, 12> mipViews{};
};

class ScreenSpaceEffects {
public:
    /// Creates all GPU resources: offscreen targets, compute pipelines,
    /// descriptor sets. Call after Renderer::Initialize().
    bool Initialize(Renderer& renderer);

    /// Destroys all GPU resources.
    void Shutdown();

    /// Called on swapchain recreation to resize all offscreen targets.
    void OnSwapchainResized(Renderer& renderer);

    /// Returns true after successful Initialize().
    [[nodiscard]] bool IsInitialized() const { return m_initialized; }

    /// Returns mutable config for settings UI.
    [[nodiscard]] ScreenSpaceConfig& GetConfig() { return m_config; }
    [[nodiscard]] const ScreenSpaceConfig& GetConfig() const { return m_config; }

    // ── Offscreen render targets (used by Renderer for scene pass) ────

    /// The offscreen color image the scene renders into (instead of swapchain).
    [[nodiscard]] VkImageView   GetSceneColorView()   const { return m_sceneColor.view; }
    [[nodiscard]] VkImage       GetSceneColorImage()  const { return m_sceneColor.image; }
    [[nodiscard]] VkFormat      GetSceneColorFormat()  const { return m_sceneColor.format; }

    /// The depth buffer (scene writes depth here; compute shaders read it).
    [[nodiscard]] VkImageView   GetDepthView()        const { return m_sceneDepth.view; }
    [[nodiscard]] VkImage       GetDepthImage()       const { return m_sceneDepth.image; }

    /// World-space normal buffer (written during scene pass via MRT or a separate pass).
    [[nodiscard]] VkImageView   GetNormalView()       const { return m_sceneNormals.view; }
    [[nodiscard]] VkImage       GetNormalImage()      const { return m_sceneNormals.image; }

    // ── Core pipeline ─────────────────────────────────────────────────

    /// Transitions scene color/depth/normal from attachment to shader-read,
    /// builds the Hi-Z pyramid, dispatches all enabled screen-space compute
    /// shaders, then composites the result into the current swapchain image.
    ///
    /// @param cmd           Active command buffer (between BeginFrame/EndFrame)
    /// @param viewProj      Current frame VP matrix
    /// @param prevViewProj  Previous frame VP matrix (for reprojection)
    /// @param cameraPos     Camera world position
    /// @param nearPlane     Camera near plane distance
    /// @param farPlane      Camera far plane distance
    /// @param frameIndex    Current frame counter (for temporal jitter)
    /// @param deltaTime     Frame delta time
    void DispatchAll(VkCommandBuffer cmd,
                     const glm::mat4& viewProj,
                     const glm::mat4& prevViewProj,
                     const glm::vec3& cameraPos,
                     float nearPlane, float farPlane,
                     uint32_t frameIndex, float deltaTime);

    // ── Begin/End offscreen scene pass ────────────────────────────────

    /// Begins the offscreen scene rendering pass (writes to color + depth + normals).
    /// The Renderer calls this instead of rendering directly to the swapchain.
    void BeginScenePass(VkCommandBuffer cmd, const glm::vec4& clearColor);

    /// Ends the offscreen scene rendering pass.
    void EndScenePass(VkCommandBuffer cmd);

    /// Suspend/Resume for shadow map insertion (mirrors Renderer API).
    void SuspendScenePass(VkCommandBuffer cmd);
    void ResumeScenePass(VkCommandBuffer cmd);

    /// Composites the final result to the given swapchain image view.
    /// Called after DispatchAll().
    void CompositeToSwapchain(VkCommandBuffer cmd,
                              VkImageView swapchainView,
                              VkExtent2D swapchainExtent);

private:
    bool m_initialized = false;
    ScreenSpaceConfig m_config;

    // ── Cached Vulkan handles ─────────────────────────────────────────

    VkDevice     m_device    = VK_NULL_HANDLE;
    VmaAllocator m_allocator = VK_NULL_HANDLE;
    uint32_t     m_width     = 0;
    uint32_t     m_height    = 0;

    // ── Offscreen render targets ──────────────────────────────────────

    OffscreenImage m_sceneColor;        ///< RGBA16F scene color
    OffscreenImage m_sceneDepth;        ///< D32_SFLOAT depth
    OffscreenImage m_sceneNormals;      ///< RGB16F world-space normals
    OffscreenImage m_hiZBuffer;         ///< R32F hierarchical depth (mip chain)
    OffscreenImage m_motionVectors;     ///< RG16F screen-space motion

    // ── Effect output buffers ─────────────────────────────────────────

    OffscreenImage m_aoOutput;          ///< RGBA16F: R=AO, GBA=bent normal
    OffscreenImage m_giOutput;          ///< RGBA16F: RGB=indirect, A=confidence
    OffscreenImage m_reflOutput;        ///< RGBA16F: RGB=reflected, A=confidence
    OffscreenImage m_denoiseOutput;     ///< RGBA16F: denoised result

    // ── History buffers (for temporal accumulation) ───────────────────

    OffscreenImage m_prevGI;            ///< Previous frame GI
    OffscreenImage m_prevRefl;          ///< Previous frame reflections
    OffscreenImage m_prevDepth;         ///< Previous frame depth
    OffscreenImage m_prevDenoise;       ///< Previous frame denoised

    // ── Sampler ───────────────────────────────────────────────────────

    VkSampler m_linearSampler  = VK_NULL_HANDLE;  ///< Bilinear, clamp-to-edge
    VkSampler m_nearestSampler = VK_NULL_HANDLE;  ///< Nearest, clamp-to-edge

    // ── Descriptor infrastructure ─────────────────────────────────────

    VkDescriptorPool m_descriptorPool = VK_NULL_HANDLE;

    // Hi-Z build pass
    VkDescriptorSetLayout m_hiZSetLayout    = VK_NULL_HANDLE;
    VkPipelineLayout      m_hiZPipeLayout   = VK_NULL_HANDLE;
    VkPipeline            m_hiZPipeline     = VK_NULL_HANDLE;
    std::array<VkDescriptorSet, 12> m_hiZSets{};  ///< One per mip level

    // GTAO+
    VkDescriptorSetLayout m_aoSetLayout    = VK_NULL_HANDLE;
    VkPipelineLayout      m_aoPipeLayout   = VK_NULL_HANDLE;
    VkPipeline            m_aoPipeline     = VK_NULL_HANDLE;
    VkDescriptorSet       m_aoSet          = VK_NULL_HANDLE;

    // Screen-Space GI
    VkDescriptorSetLayout m_giSetLayout    = VK_NULL_HANDLE;
    VkPipelineLayout      m_giPipeLayout   = VK_NULL_HANDLE;
    VkPipeline            m_giPipeline     = VK_NULL_HANDLE;
    VkDescriptorSet       m_giSet          = VK_NULL_HANDLE;

    // Hi-Z Reflections
    VkDescriptorSetLayout m_reflSetLayout  = VK_NULL_HANDLE;
    VkPipelineLayout      m_reflPipeLayout = VK_NULL_HANDLE;
    VkPipeline            m_reflPipeline   = VK_NULL_HANDLE;
    VkDescriptorSet       m_reflSet        = VK_NULL_HANDLE;

    // Temporal Denoise
    VkDescriptorSetLayout m_denoiseSetLayout  = VK_NULL_HANDLE;
    VkPipelineLayout      m_denoisePipeLayout = VK_NULL_HANDLE;
    VkPipeline            m_denoisePipeline   = VK_NULL_HANDLE;
    VkDescriptorSet       m_denoiseSet        = VK_NULL_HANDLE;

    // RT Composite (fullscreen graphics pipeline)
    VkDescriptorSetLayout m_compositeSetLayout  = VK_NULL_HANDLE;
    VkPipelineLayout      m_compositePipeLayout = VK_NULL_HANDLE;
    VkPipeline            m_compositePipeline   = VK_NULL_HANDLE;
    VkDescriptorSet       m_compositeSet        = VK_NULL_HANDLE;

    // ── Previous frame state ──────────────────────────────────────────

    glm::mat4 m_prevViewProj = glm::mat4(1.0f);

    // ── Resource creation helpers ─────────────────────────────────────

    bool CreateOffscreenImage(OffscreenImage& img, uint32_t w, uint32_t h,
                              VkFormat format, VkImageUsageFlags usage,
                              uint32_t mipLevels = 1);
    void DestroyOffscreenImage(OffscreenImage& img);

    bool CreateRenderTargets();
    void DestroyRenderTargets();

    bool CreateSamplers();
    void DestroySamplers();

    bool CreateDescriptorPool();
    bool CreateHiZResources(Renderer& renderer);
    bool CreateAOResources(Renderer& renderer);
    bool CreateGIResources(Renderer& renderer);
    bool CreateReflResources(Renderer& renderer);
    bool CreateDenoiseResources(Renderer& renderer);
    bool CreateCompositeResources(Renderer& renderer);

    void DestroyPipelineResources();

    // ── Dispatch helpers ──────────────────────────────────────────────

    void BuildHiZPyramid(VkCommandBuffer cmd);
    void DispatchAO(VkCommandBuffer cmd, const glm::mat4& viewProj,
                    const glm::mat4& invViewProj, const glm::vec3& cameraPos,
                    float nearPlane, float farPlane, uint32_t frameIndex);
    void DispatchGI(VkCommandBuffer cmd, const glm::mat4& invViewProj,
                    const glm::mat4& prevViewProj, const glm::vec3& cameraPos,
                    uint32_t frameIndex);
    void DispatchReflections(VkCommandBuffer cmd, const glm::mat4& viewProj,
                             const glm::mat4& invViewProj, const glm::vec3& cameraPos,
                             uint32_t frameIndex);
    void DispatchDenoise(VkCommandBuffer cmd, uint32_t frameIndex, float deltaTime);

    // ── Barrier helpers ───────────────────────────────────────────────

    void TransitionImage(VkCommandBuffer cmd, VkImage image,
                         VkImageLayout oldLayout, VkImageLayout newLayout,
                         VkAccessFlags srcAccess, VkAccessFlags dstAccess,
                         VkPipelineStageFlags srcStage, VkPipelineStageFlags dstStage,
                         VkImageAspectFlags aspect = VK_IMAGE_ASPECT_COLOR_BIT,
                         uint32_t baseMip = 0, uint32_t mipCount = 1);

    void TransitionSceneToShaderRead(VkCommandBuffer cmd);
    void TransitionHistoryBuffers(VkCommandBuffer cmd);
    void SwapHistoryBuffers();
};

} // namespace hrp
