#pragma once

/// @file Renderer.hpp
/// @brief Vulkan 1.3 renderer — instance, device, swapchain, pipelines, and mesh management.
///
/// Current capabilities:
///   - VkInstance creation with validation layers (debug builds)
///   - Physical device selection (prefer discrete GPU with RT support)
///   - Logical device + graphics queue
///   - Swapchain creation and recreation on resize
///   - VMA-based memory allocation
///   - Depth buffer for 3D rendering
///   - 3D graphics pipeline (position + color vertices, MVP push constant)
///   - Debug overlay pipeline (bitmap-font text via fullscreen triangle)
///   - Mesh creation/destruction for scene geometry
///   - Per-frame command buffer recording with dynamic rendering
///   - Frame synchronization (fences + semaphores)

#include "Types.hpp"

#include <vk_mem_alloc.h>
#include <vulkan/vulkan.h>
#include <glm/glm.hpp>
#include <vector>
#include <array>
#include <string>
#include <deque>

namespace hrp {

class Platform;

// ── Vertex format for 3D geometry ────────────────────────────────────

struct Vertex3D {
    glm::vec3 position;
    glm::vec3 color;

    static VkVertexInputBindingDescription GetBindingDescription();
    static std::array<VkVertexInputAttributeDescription, 2> GetAttributeDescriptions();
};

// ── Mesh handle — references a GPU vertex buffer ─────────────────────

struct MeshHandle {
    VkBuffer      buffer     = VK_NULL_HANDLE;
    VmaAllocation allocation = VK_NULL_HANDLE;
    uint32_t      vertexCount = 0;
    bool          dynamic     = false;  ///< true when host-visible (UpdateMesh-able).
};

// ── Push constant structures ─────────────────────────────────────────

struct PushConstants3D {
    glm::mat4 mvp;   // 64 bytes — fits in minimum guaranteed range
};

struct PushConstantsOverlay {
    glm::vec4 playerPos;    // xyz = player position, w = unused
    glm::vec4 cameraPos;    // xyz = camera position, w = unused
    glm::vec4 screenInfo;   // x = width, y = height, zw = unused
};

class Renderer {
public:
    /// Initializes the full Vulkan stack: instance → device → VMA → swapchain → pipelines.
    bool Initialize(Platform& platform);

    /// Waits for GPU idle, then destroys all Vulkan objects in reverse order.
    void Shutdown();

    /// Begins a new frame: waits on fence, acquires swapchain image, begins command buffer.
    /// Returns false if the swapchain is out of date and needs recreation.
    bool BeginFrame();

    /// Ends the frame: ends command buffer, submits, presents.
    void EndFrame();

    /// Recreates the swapchain (call after window resize or suboptimal present).
    void RecreateSwapchain(Platform& platform);

    /// Returns the Vulkan instance (needed by Platform for surface creation).
    [[nodiscard]] VkInstance GetInstance() const { return m_instance; }

    /// True after Initialize() succeeds.
    [[nodiscard]] bool IsInitialized() const { return m_initialized; }

    /// Waits for the GPU to finish all submitted work.
    void WaitIdle();

    /// Returns current swapchain extent (for aspect ratio calculations).
    [[nodiscard]] VkExtent2D GetSwapchainExtent() const { return m_swapchainExtent; }

    // ── Public getters for UISystem / TerrainTextures ─────────────────

    [[nodiscard]] VkPhysicalDevice GetPhysicalDevice() const { return m_physicalDevice; }
    [[nodiscard]] VkDevice         GetDevice()         const { return m_device; }
    [[nodiscard]] uint32_t         GetGraphicsFamily()  const { return m_graphicsFamily; }
    [[nodiscard]] VkQueue          GetGraphicsQueue()   const { return m_graphicsQueue; }
    [[nodiscard]] VkFormat         GetSwapchainFormat()  const { return m_swapchainFormat; }
    [[nodiscard]] VkFormat         GetSceneColorFormat() const { return m_sceneColorFormat; }
    [[nodiscard]] static constexpr VkFormat GetDepthFormat() { return k_DepthFormat; }

    /// Sets the color format used by scene-rendering pipelines (3D objects,
    /// terrain, skybox).  When ScreenSpaceEffects is active this is RGBA16F;
    /// otherwise it matches the swapchain format.  Triggers recreation of the
    /// Renderer-owned 3D pipeline; subsystems that init later (Skybox,
    /// TerrainTextures) will pick up the new format from GetSceneColorFormat().
    void SetSceneColorFormat(VkFormat format);
    [[nodiscard]] VkCommandBuffer  GetCurrentCommandBuffer() const { return m_commandBuffers[m_currentFrame]; }
    [[nodiscard]] VmaAllocator     GetAllocator()       const { return m_allocator; }
    [[nodiscard]] VkCommandPool    GetCommandPool()     const { return m_commandPool; }

    /// Descriptor set layout used by the 3D pipeline's shadow map binding (set 0, depth-compare sampler).
    /// External code that wants to call Bind3DShadowSet() must allocate the descriptor set with this layout.
    [[nodiscard]] VkDescriptorSetLayout Get3DShadowSetLayout() const { return m_3dShadowSetLayout; }

    // ── Mesh management ──────────────────────────────────────────────

    /// Creates a GPU vertex buffer from CPU vertex data.
    ///
    /// `dynamic == false` (default): allocates DEVICE_LOCAL (VRAM) memory
    /// and uploads via a one-shot staging buffer. The staging buffer is
    /// freed before this returns, so process committed memory does not
    /// retain a copy of the vertex data. `UpdateMesh()` will fail on these
    /// meshes — recreate them instead.
    ///
    /// `dynamic == true`: allocates persistently-mapped host-visible memory
    /// (CPU_TO_GPU). `UpdateMesh()` works. Use only for meshes that are
    /// re-uploaded frequently (particles, falling-tree animation, etc.) —
    /// host-visible memory shows up in process Working Set / Private Bytes.
    bool CreateMesh(const std::vector<Vertex3D>& vertices, MeshHandle& outMesh,
                    bool dynamic = false);

    /// Re-uploads CPU vertex data to an existing GPU mesh buffer. The mesh
    /// must have been created with `dynamic == true`. The new vertex count
    /// must match the original.
    bool UpdateMesh(const std::vector<Vertex3D>& vertices, MeshHandle& mesh);

    /// Destroys a mesh and frees its GPU memory.
    /// Destruction is deferred until all in-flight frames have completed
    /// to prevent GPU use-after-free when a previous frame's command
    /// buffer is still referencing the vertex buffer.
    void DestroyMesh(MeshHandle& mesh);

    // ── Phase 6 — batched mesh upload (persistent staging ring) ────
    /// Open a batched upload window. While open, every static `CreateMesh`
    /// call records its copy into a single per-batch transfer command
    /// buffer using a slice of the persistent staging ring instead of
    /// allocating a per-mesh staging buffer + waiting on `vkQueueWaitIdle`.
    /// Idempotent if already active. Pair with `EndUploadBatch`.
    void BeginUploadBatch();

    /// Submit and wait on the batched transfer command buffer (single
    /// fence wait covers every mesh queued since `BeginUploadBatch`).
    /// After this returns, every mesh uploaded inside the batch is
    /// safe to draw. Frees no staging — the ring is persistent.
    void EndUploadBatch();

    /// True while a batch is open.
    [[nodiscard]] bool IsUploadBatchActive() const { return m_uploadBatchActive; }

    /// Non-blocking check: true if the most recent batch's GPU work has not
    /// yet completed. Lets callers (e.g. VegetationSystem::DrainReadyChunkVegetation)
    /// skip a frame instead of blocking inside BeginUploadBatch's fence wait.
    /// Always false when no fence has ever been signaled (i.e. first batch).
    [[nodiscard]] bool IsUploadBatchBusy() const;

    /// Queues a raw VkBuffer + VmaAllocation for deferred destruction.
    /// Used by subsystems (TerrainTextures, VegetationSystem) that manage
    /// their own mesh handles but share the same VMA allocator.
    void DeferBufferDestruction(VkBuffer buffer, VmaAllocation allocation);

    // ── Scene pass management (for shadow map insertion) ────────────

    /// Temporarily ends the main scene dynamic rendering pass.
    /// Call before an off-screen pass (e.g. shadow map), then ResumeScenePass().
    void SuspendScenePass();

    /// Re-starts the main scene dynamic rendering pass after a suspend.
    /// Uses LOAD_OP_LOAD to preserve whatever was already rendered.
    void ResumeScenePass();

    // ── Draw commands (call between BeginFrame/EndFrame) ─────────────

    /// Binds the 3D pipeline. Call before DrawMesh().
    void Bind3DPipeline();

    /// Pushes the light-space view-projection matrix for vertex-stage shadow
    /// coordinate output (push constant bytes 64–127). Call after Bind3DPipeline().
    void Set3DLightViewProj(const glm::mat4& lightViewProj);

    /// Binds the shadow map descriptor set at set 0 for the 3D pipeline.
    /// Call after Bind3DPipeline() and before DrawMesh().
    void Bind3DShadowSet(VkDescriptorSet shadowSet);

    /// Sets fragment-stage lighting + environment parameters for the 3D pipeline.
    /// Call after Bind3DPipeline() and before DrawMesh(). Params persist
    /// across all subsequent DrawMesh calls until the pipeline is re-bound.
    /// @param envParams  Per-biome environment: x=exposureBias, y=ambientScale,
    ///                   z=fogDensityMul, w=roughnessBase. CPU-blended from
    ///                   the player's current biome and day/night cycle.
    void Set3DLightingParams(const glm::vec4& lightDir, const glm::vec4& cameraPos,
                             const glm::vec4& fogParams, const glm::vec4& fogColor,
                             const glm::vec4& envParams);

    /// Draws a mesh with the given MVP matrix (push constant).
    void DrawMesh(const MeshHandle& mesh, const glm::mat4& mvp);

    /// Draws the debug text overlay showing player/camera positions.
    /// @param mode      0 = debug HUD, 1 = main menu, 2 = loading screen.
    /// @param extraData  Mode-specific data (e.g. loading progress 0–1 for mode 2).
    void DrawDebugOverlay(const glm::vec3& playerPos, const glm::vec3& cameraPos,
                          float mode = 0.0f, float extraData = 0.0f);

    // ── Offscreen render-to-PNG ──────────────────────────────────────────────

    /// Renders CPU vertex data to an offscreen image and saves as PNG.
    /// Automatically computes a 3/4-view camera that frames the geometry.
    /// @param vertices    Triangle list (Vertex3D, 3 verts per tri).
    /// @param boundsMin   AABB minimum of the geometry.
    /// @param boundsMax   AABB maximum of the geometry.
    /// @param pngPath     Output file path.
    /// @param width       Image width in pixels (default 1024).
    /// @param height      Image height in pixels (default 1024).
    /// @return True if PNG was written successfully.
    bool RenderGeomToPNG(const std::vector<Vertex3D>& vertices,
                         const glm::vec3& boundsMin,
                         const glm::vec3& boundsMax,
                         const std::string& pngPath,
                         uint32_t width = 1024, uint32_t height = 1024);

    // ── Post-processing controls ─────────────────────────────────────────────

    void SetClearColor(const glm::vec4& color) { m_clearColor = color; }
    [[nodiscard]] const glm::vec4& GetClearColor() const { return m_clearColor; }

    /// Enables offscreen rendering mode: BeginFrame/EndFrame skip starting/ending
    /// the scene dynamic rendering pass, leaving that to ScreenSpaceEffects.
    /// When enabled, the caller is responsible for beginning/ending rendering.
    void SetOffscreenMode(bool enabled) { m_offscreenMode = enabled; }
    [[nodiscard]] bool IsOffscreenMode() const { return m_offscreenMode; }

    /// Request that the next EndFrame() copy the post-render swapchain image
    /// out to a PNG at the given path (sRGB-corrected). One-shot: cleared after
    /// the next successful EndFrame. Used by --screenshot-world headless mode.
    /// The swapchain must have been created with TRANSFER_SRC_BIT (it is, by
    /// default). Capture allocates a transient blit image + staging buffer
    /// per call and adds ~1 frame of latency.
    void RequestSwapchainCapture(const std::string& pngPath) { m_pendingCapturePath = pngPath; }
    [[nodiscard]] bool HasPendingCapture() const { return !m_pendingCapturePath.empty(); }

    /// Begins a rendering pass on the swapchain for UI overlay (ForgeUI).
    /// In offscreen mode, ScreenSpaceEffects owns the scene pass but ForgeUI
    /// still needs an active rendering pass on the swapchain to draw into.
    /// @param preserveContents  If true, uses LOAD_OP_LOAD to keep composite
    ///                          output; if false, clears to m_clearColor.
    void BeginOverlayPass(bool preserveContents = false);

    /// Ends the overlay rendering pass started by BeginOverlayPass().
    void EndOverlayPass();

    // ── Swapchain image accessors (for composite pass) ───────────────────────

    [[nodiscard]] VkImageView GetCurrentSwapchainImageView() const {
        return m_swapchainImageViews[m_imageIndex];
    }
    [[nodiscard]] VkImage GetCurrentSwapchainImage() const {
        return m_swapchainImages[m_imageIndex];
    }

private:
    // ── Vulkan core objects ──────────────────────────────────────────

    VkInstance               m_instance       = VK_NULL_HANDLE;
    VkDebugUtilsMessengerEXT m_debugMessenger = VK_NULL_HANDLE;
    VkPhysicalDevice         m_physicalDevice = VK_NULL_HANDLE;
    VkDevice                 m_device         = VK_NULL_HANDLE;
    VkQueue                  m_graphicsQueue  = VK_NULL_HANDLE;
    VkQueue                  m_presentQueue   = VK_NULL_HANDLE;
    uint32_t                 m_graphicsFamily = 0;
    uint32_t                 m_presentFamily  = 0;
    uint32_t                 m_transferFamily = 0;     ///< Phase 6b: dedicated DMA family if available, else == m_graphicsFamily.
    bool                     m_transferIsSeparate = false; ///< True when m_transferFamily differs from m_graphicsFamily.
    VkQueue                  m_transferQueue  = VK_NULL_HANDLE; ///< Phase 6b. Same as m_graphicsQueue when not separate.

    // ── VMA allocator ────────────────────────────────────────────────

    VmaAllocator m_allocator = VK_NULL_HANDLE;

    // ── Surface & swapchain ──────────────────────────────────────────

    VkSurfaceKHR             m_surface        = VK_NULL_HANDLE;
    VkSwapchainKHR           m_swapchain      = VK_NULL_HANDLE;
    VkFormat                 m_swapchainFormat     = VK_FORMAT_B8G8R8A8_SRGB;
    VkFormat                 m_sceneColorFormat    = VK_FORMAT_B8G8R8A8_SRGB; ///< Format used by scene pipelines
    VkExtent2D               m_swapchainExtent{ 0, 0 };
    std::vector<VkImage>     m_swapchainImages;
    std::vector<VkImageView> m_swapchainImageViews;

    // ── Depth buffer ─────────────────────────────────────────────────

    VkImage       m_depthImage      = VK_NULL_HANDLE;
    VmaAllocation m_depthAllocation = VK_NULL_HANDLE;
    VkImageView   m_depthImageView  = VK_NULL_HANDLE;
    static constexpr VkFormat k_DepthFormat = VK_FORMAT_D32_SFLOAT;

    // ── Pipelines ────────────────────────────────────────────────────

    VkPipelineLayout m_3dPipelineLayout      = VK_NULL_HANDLE;
    VkPipeline       m_3dPipeline            = VK_NULL_HANDLE;
    VkDescriptorSetLayout m_3dShadowSetLayout = VK_NULL_HANDLE;  ///< Shadow sampler layout for 3D pipeline
    VkPipelineLayout m_overlayPipelineLayout = VK_NULL_HANDLE;
    VkPipeline       m_overlayPipeline       = VK_NULL_HANDLE;

    // ── Command pools & buffers ──────────────────────────────────────

    VkCommandPool                m_commandPool = VK_NULL_HANDLE;
    VkCommandPool                m_transferCommandPool = VK_NULL_HANDLE; ///< Phase 6b: pool on the transfer family. Equal-handle to m_commandPool when not separate.
    std::vector<VkCommandBuffer> m_commandBuffers;

    // ── Phase 6 — persistent staging ring + batched mesh upload ──────
    static constexpr VkDeviceSize k_StagingRingInitialSize = 32u * 1024u * 1024u; ///< 32 MB initial — sized for welded per-chunk tree meshes.
    VkBuffer       m_stagingRingBuf   = VK_NULL_HANDLE;
    VmaAllocation  m_stagingRingAlloc = VK_NULL_HANDLE;
    VkDeviceSize   m_stagingRingSize  = 0;
    VkDeviceSize   m_stagingRingHead  = 0;     ///< Bytes consumed in the current batch.
    void*          m_stagingRingMapped = nullptr; ///< Persistently mapped pointer.
    VkCommandBuffer m_uploadBatchCmd  = VK_NULL_HANDLE;
    VkFence         m_uploadBatchFence = VK_NULL_HANDLE;
    bool            m_uploadBatchActive = false;
    bool            m_uploadBatchFenceInFlight = false;
    uint32_t        m_uploadBatchMeshCount = 0;

    /// Ensure the staging ring is at least `required` bytes. Reallocates
    /// (waiting for any pending batch fence) if not. Lazy: first call
    /// creates the ring at `k_StagingRingInitialSize`.
    bool EnsureStagingRing(VkDeviceSize required);
    /// Tear down the staging ring + batched-upload command buffer + fence.
    /// Called from `Shutdown`.
    void DestroyStagingRing();

    // ── Synchronization ──────────────────────────────────────────────

    static constexpr uint32_t k_MaxFramesInFlight = 2;
    uint32_t m_currentFrame  = 0;
    uint32_t m_imageIndex    = 0;

    /// Per-frame-in-flight semaphores (signaled by vkAcquireNextImageKHR).
    std::vector<VkSemaphore> m_imageAvailableSemaphores;
    /// Per-swapchain-image semaphores (signaled by vkQueueSubmit, waited by vkQueuePresentKHR).
    /// Indexed by m_imageIndex to avoid reuse while the presentation engine still owns one.
    std::vector<VkSemaphore> m_renderFinishedSemaphores;
    std::vector<VkFence>     m_inFlightFences;

    /// Per-swapchain-image fence tracking — maps each image to the fence of the
    /// frame-in-flight that is currently rendering to it.  VK_NULL_HANDLE means
    /// no frame-in-flight is using that image.  This prevents two different frame
    /// slots from rendering to the same image simultaneously (possible with
    /// MAILBOX / triple-buffering when imageCount > k_MaxFramesInFlight).
    std::vector<VkFence>     m_imagesInFlight;

    // ── State ────────────────────────────────────────────────────────

    bool     m_initialized = false;
    bool     m_offscreenMode = false;  ///< When true, BeginFrame/EndFrame skip scene rendering begin/end.

    std::string m_pendingCapturePath;  ///< Set by RequestSwapchainCapture; consumed by EndFrame.
    glm::vec4 m_clearColor{ 0.02f, 0.02f, 0.05f, 1.0f };  // Near-black

    // ── Deferred GPU resource deletion ────────────────────────────────
    /// Buffers slated for destruction are held for k_MaxFramesInFlight + 1
    /// frames to guarantee all in-flight command buffers have completed.
    struct DeferredBuffer {
        VkBuffer      buffer;
        VmaAllocation allocation;
    };
    static constexpr uint32_t k_DeletionRingSize = k_MaxFramesInFlight + 1;
    std::array<std::vector<DeferredBuffer>, k_DeletionRingSize> m_deletionRing;
    uint32_t m_deletionRingIndex = 0;
    void ProcessDeferredDeletions();

    // ── PNG render helper resources (lazy-init in RenderGeomToPNG) ────
    bool              m_pngResourcesReady      = false;
    VkImage           m_pngDummyShadowImage    = VK_NULL_HANDLE;
    VmaAllocation     m_pngDummyShadowAlloc    = VK_NULL_HANDLE;
    VkImageView       m_pngDummyShadowView     = VK_NULL_HANDLE;
    VkSampler         m_pngShadowSampler       = VK_NULL_HANDLE;
    VkDescriptorPool  m_pngDescPool            = VK_NULL_HANDLE;
    VkDescriptorSet   m_pngShadowSet           = VK_NULL_HANDLE;
    bool EnsurePNGRenderResources();
    void DestroyPNGRenderResources();

    // ── Internal helpers ─────────────────────────────────────────────

    bool CreateInstance(Platform& platform);
    bool SetupDebugMessenger();
    bool PickPhysicalDevice();
    bool CreateLogicalDevice();
    bool CreateAllocator();
    bool CreateSwapchain(Platform& platform);
    bool CreateImageViews();
    bool CreateDepthResources();
    bool CreateCommandPool();
    bool CreateCommandBuffers();
    bool CreateSyncObjects();
    bool CreatePipelines();
    bool Create3DPipeline();  ///< Creates (or recreates) the 3D scene pipeline

    void CleanupSwapchain();
    void CleanupDepthResources();

public:
    /// Loads a SPIR-V shader module from disk.
    /// Public so TerrainTextures (and future subsystems) can load their own shaders.
    VkShaderModule LoadShaderModule(const std::string& filename);

private:

    /// Searches for the shader directory relative to CWD or executable.
    std::string FindShaderPath(const std::string& shaderName) const;

    // ── Vulkan query helpers ─────────────────────────────────────────

    struct QueueFamilyIndices {
        uint32_t graphics = UINT32_MAX;
        uint32_t present  = UINT32_MAX;
        uint32_t transfer = UINT32_MAX;   ///< Phase 6b: prefers a dedicated DMA queue family (TRANSFER_BIT && !GRAPHICS_BIT). Falls back to graphics.
        [[nodiscard]] bool IsComplete() const {
            return graphics != UINT32_MAX && present != UINT32_MAX;
        }
    };

    QueueFamilyIndices FindQueueFamilies(VkPhysicalDevice device) const;

    struct SwapchainSupportDetails {
        VkSurfaceCapabilitiesKHR        capabilities{};
        std::vector<VkSurfaceFormatKHR> formats;
        std::vector<VkPresentModeKHR>   presentModes;
    };

    SwapchainSupportDetails QuerySwapchainSupport(VkPhysicalDevice device) const;

    VkSurfaceFormatKHR ChooseSwapSurfaceFormat(const std::vector<VkSurfaceFormatKHR>& available) const;
    VkPresentModeKHR   ChooseSwapPresentMode(const std::vector<VkPresentModeKHR>& available) const;
    VkExtent2D         ChooseSwapExtent(const VkSurfaceCapabilitiesKHR& capabilities,
                                        Platform& platform) const;

    bool IsDeviceSuitable(VkPhysicalDevice device) const;
    int  RateDeviceSuitability(VkPhysicalDevice device) const;
};

} // namespace hrp
