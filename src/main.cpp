/// @file main.cpp
/// @brief Hybrid Render Pipeline — demo entry point.
///
/// Renders a procedural FBM heightmap terrain + sky dome through the full
/// hybrid pipeline (forward geometry pass → Hi-Z → GTAO+ → SVRM GI →
/// SHR reflections → ACTF temporal denoise → composite).
///
/// The demo's purpose is to prove the pipeline boots end-to-end and to give
/// new users something visible to start hacking from. Shaders, meshes, and
/// scene content are intentionally minimal — there is no shadow pass (a
/// 1×1 "always lit" dummy shadow map is bound), no skybox shader (a vertex-
/// coloured inverted sphere stands in), and no input besides window-close.
///
/// TODO list for anyone extending this:
///   - Wire a real shadow pass using shadow_depth.vert/.frag against the
///     terrain mesh and bind the resulting depth image instead of the dummy.
///   - Replace the sky-dome mesh with a fullscreen-triangle pipeline driving
///     skybox.frag for the full procedural-stars/clouds sky.
///   - Drive Camera::ProcessMouseMovement / ProcessMouseScroll from the SDL
///     events the Platform module already collects (Platform::GetFrameEvents).

#include "Platform.hpp"
#include "Renderer.hpp"
#include "ScreenSpaceEffects.hpp"
#include "Camera.hpp"
#include "Log.hpp"
#include "Types.hpp"

#define VMA_IMPLEMENTATION
#include <vk_mem_alloc.h>
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <stb_image_write.h>

#include <SDL3/SDL.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include <chrono>
#include <cmath>
#include <cstdint>
#include <vector>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

using namespace hrp;

// ──────────────────────────────────────────────────────────────────────
// FBM noise — value noise with 5 octaves, used to drive the heightmap
// ──────────────────────────────────────────────────────────────────────

static float hash2D(int x, int y)
{
    uint32_t n = uint32_t(x) * 0x27d4eb2du ^ uint32_t(y) * 0x165667b1u;
    n = (n ^ (n >> 13)) * 0x85ebca6bu;
    n = (n ^ (n >> 16));
    return float(n) / 4294967295.0f;
}

static float smoothNoise(float x, float y)
{
    int ix = int(std::floor(x));
    int iy = int(std::floor(y));
    float fx = x - float(ix);
    float fy = y - float(iy);
    fx = fx * fx * (3.0f - 2.0f * fx);
    fy = fy * fy * (3.0f - 2.0f * fy);
    float v00 = hash2D(ix,     iy);
    float v10 = hash2D(ix + 1, iy);
    float v01 = hash2D(ix,     iy + 1);
    float v11 = hash2D(ix + 1, iy + 1);
    float a = v00 + (v10 - v00) * fx;
    float b = v01 + (v11 - v01) * fx;
    return a + (b - a) * fy;
}

static float fbm(float x, float y, int octaves)
{
    float total = 0.0f, amp = 1.0f, freq = 1.0f, maxAmp = 0.0f;
    for (int i = 0; i < octaves; ++i) {
        total  += smoothNoise(x * freq, y * freq) * amp;
        maxAmp += amp;
        amp    *= 0.5f;
        freq   *= 2.0f;
    }
    return total / maxAmp;
}

// ──────────────────────────────────────────────────────────────────────
// Mesh generators
// ──────────────────────────────────────────────────────────────────────

static std::vector<Vertex3D> GenerateTerrainMesh(int gridSize, float worldSize, float heightScale)
{
    std::vector<float> heights(size_t(gridSize) * size_t(gridSize));
    for (int y = 0; y < gridSize; ++y) {
        for (int x = 0; x < gridSize; ++x) {
            float u = float(x) / float(gridSize - 1);
            float v = float(y) / float(gridSize - 1);
            // Domain warping bumps a flat FBM into something more interesting.
            float wx = u * 6.0f + 0.5f * fbm(u * 4.0f + 13.0f, v * 4.0f, 3);
            float wy = v * 6.0f + 0.5f * fbm(u * 4.0f, v * 4.0f + 7.0f, 3);
            float h  = fbm(wx, wy, 5);
            // Ridge transform for sharper peaks.
            h = 1.0f - std::fabs(h * 2.0f - 1.0f);
            h = h * h;
            heights[size_t(y) * size_t(gridSize) + size_t(x)] = h * heightScale;
        }
    }

    auto colorAtHeight = [heightScale](float h) -> glm::vec3 {
        float t = h / heightScale;
        if (t < 0.18f)      return glm::mix(glm::vec3(0.10f, 0.22f, 0.32f), glm::vec3(0.55f, 0.50f, 0.30f), t / 0.18f);            // shallow water → sand
        else if (t < 0.45f) return glm::mix(glm::vec3(0.42f, 0.55f, 0.25f), glm::vec3(0.30f, 0.42f, 0.22f), (t - 0.18f) / 0.27f);  // grass → forest
        else if (t < 0.65f) return glm::mix(glm::vec3(0.30f, 0.42f, 0.22f), glm::vec3(0.45f, 0.38f, 0.28f), (t - 0.45f) / 0.20f);  // forest → soil
        else if (t < 0.85f) return glm::mix(glm::vec3(0.45f, 0.38f, 0.28f), glm::vec3(0.50f, 0.48f, 0.45f), (t - 0.65f) / 0.20f);  // soil → rock
        else                return glm::mix(glm::vec3(0.50f, 0.48f, 0.45f), glm::vec3(0.96f, 0.96f, 0.98f), (t - 0.85f) / 0.15f);  // rock → snow
    };

    const float half = worldSize * 0.5f;
    const float step = worldSize / float(gridSize - 1);

    auto vertAt = [&](int x, int y) -> Vertex3D {
        Vertex3D v;
        v.position = glm::vec3(-half + float(x) * step,
                               heights[size_t(y) * size_t(gridSize) + size_t(x)],
                               -half + float(y) * step);
        v.color    = colorAtHeight(v.position.y);
        return v;
    };

    std::vector<Vertex3D> verts;
    verts.reserve(size_t(gridSize - 1) * size_t(gridSize - 1) * 6);
    for (int y = 0; y < gridSize - 1; ++y) {
        for (int x = 0; x < gridSize - 1; ++x) {
            auto a = vertAt(x,     y);
            auto b = vertAt(x + 1, y);
            auto c = vertAt(x,     y + 1);
            auto d = vertAt(x + 1, y + 1);
            verts.push_back(a); verts.push_back(c); verts.push_back(b);
            verts.push_back(b); verts.push_back(c); verts.push_back(d);
        }
    }
    return verts;
}

static std::vector<Vertex3D> GenerateSkyDomeMesh(float radius, int rings, int sectors)
{
    std::vector<glm::vec3> pos;
    pos.reserve(size_t(rings + 1) * size_t(sectors + 1));
    for (int r = 0; r <= rings; ++r) {
        float phi  = (float(r) / float(rings)) * float(M_PI);
        float py   = std::cos(phi) * radius;
        float rRad = std::sin(phi) * radius;
        for (int s = 0; s <= sectors; ++s) {
            float theta = (float(s) / float(sectors)) * 2.0f * float(M_PI);
            pos.emplace_back(rRad * std::cos(theta), py, rRad * std::sin(theta));
        }
    }

    auto colorAt = [radius](const glm::vec3& p) -> glm::vec3 {
        float t = (p.y / radius + 1.0f) * 0.5f;  // 0 (bottom) → 1 (top)
        glm::vec3 horizon(0.95f, 0.78f, 0.55f);  // warm
        glm::vec3 mid    (0.50f, 0.65f, 0.92f);  // sky blue
        glm::vec3 zenith (0.12f, 0.25f, 0.62f);  // deep blue
        return (t < 0.5f) ? glm::mix(horizon, mid,    t * 2.0f)
                          : glm::mix(mid,     zenith, (t - 0.5f) * 2.0f);
    };

    std::vector<Vertex3D> verts;
    verts.reserve(size_t(rings) * size_t(sectors) * 6);
    for (int r = 0; r < rings; ++r) {
        for (int s = 0; s < sectors; ++s) {
            int i00 = r * (sectors + 1) + s;
            int i10 = i00 + (sectors + 1);
            int i01 = i00 + 1;
            int i11 = i10 + 1;
            Vertex3D a{ pos[i00], colorAt(pos[i00]) };
            Vertex3D b{ pos[i10], colorAt(pos[i10]) };
            Vertex3D c{ pos[i01], colorAt(pos[i01]) };
            Vertex3D d{ pos[i11], colorAt(pos[i11]) };
            // Inverted winding so the inside faces of the sphere are visible.
            verts.push_back(a); verts.push_back(b); verts.push_back(c);
            verts.push_back(c); verts.push_back(b); verts.push_back(d);
        }
    }
    return verts;
}

// ──────────────────────────────────────────────────────────────────────
// Dummy shadow map — 1×1 depth image bound through Renderer's 3D shadow
// set layout. Sampler uses VK_COMPARE_OP_ALWAYS so shadow comparison is
// always "lit". Replace with a real shadow pass for actual shadows.
// ──────────────────────────────────────────────────────────────────────

class DummyShadowMap {
public:
    bool Create(Renderer& renderer)
    {
        m_device = renderer.GetDevice();

        // 1×1 D32_SFLOAT image
        VkImageCreateInfo ic{};
        ic.sType         = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
        ic.imageType     = VK_IMAGE_TYPE_2D;
        ic.format        = VK_FORMAT_D32_SFLOAT;
        ic.extent        = { 1, 1, 1 };
        ic.mipLevels     = 1;
        ic.arrayLayers   = 1;
        ic.samples       = VK_SAMPLE_COUNT_1_BIT;
        ic.tiling        = VK_IMAGE_TILING_OPTIMAL;
        ic.usage         = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
        ic.sharingMode   = VK_SHARING_MODE_EXCLUSIVE;
        ic.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;

        VmaAllocationCreateInfo ac{};
        ac.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;

        if (vmaCreateImage(renderer.GetAllocator(), &ic, &ac, &m_image, &m_alloc, nullptr) != VK_SUCCESS) {
            HRP_LOG_ERROR("DummyShadowMap: vmaCreateImage failed");
            return false;
        }

        // Transition UNDEFINED → SHADER_READ_ONLY_OPTIMAL (no upload needed; sampler always-passes anyway)
        TransitionToShaderRead(renderer);

        VkImageViewCreateInfo vc{};
        vc.sType      = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
        vc.image      = m_image;
        vc.viewType   = VK_IMAGE_VIEW_TYPE_2D;
        vc.format     = VK_FORMAT_D32_SFLOAT;
        vc.subresourceRange.aspectMask     = VK_IMAGE_ASPECT_DEPTH_BIT;
        vc.subresourceRange.baseMipLevel   = 0;
        vc.subresourceRange.levelCount     = 1;
        vc.subresourceRange.baseArrayLayer = 0;
        vc.subresourceRange.layerCount     = 1;
        if (vkCreateImageView(m_device, &vc, nullptr, &m_view) != VK_SUCCESS) {
            HRP_LOG_ERROR("DummyShadowMap: vkCreateImageView failed");
            return false;
        }

        VkSamplerCreateInfo sc{};
        sc.sType        = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
        sc.magFilter    = VK_FILTER_LINEAR;
        sc.minFilter    = VK_FILTER_LINEAR;
        sc.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER;
        sc.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER;
        sc.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER;
        sc.borderColor  = VK_BORDER_COLOR_FLOAT_OPAQUE_WHITE;
        sc.compareEnable = VK_TRUE;
        sc.compareOp     = VK_COMPARE_OP_ALWAYS;  // always "lit"
        if (vkCreateSampler(m_device, &sc, nullptr, &m_sampler) != VK_SUCCESS) {
            HRP_LOG_ERROR("DummyShadowMap: vkCreateSampler failed");
            return false;
        }

        VkDescriptorPoolSize ps{};
        ps.type            = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        ps.descriptorCount = 1;
        VkDescriptorPoolCreateInfo pc{};
        pc.sType         = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
        pc.maxSets       = 1;
        pc.poolSizeCount = 1;
        pc.pPoolSizes    = &ps;
        if (vkCreateDescriptorPool(m_device, &pc, nullptr, &m_pool) != VK_SUCCESS) {
            HRP_LOG_ERROR("DummyShadowMap: vkCreateDescriptorPool failed");
            return false;
        }

        VkDescriptorSetLayout layout = renderer.Get3DShadowSetLayout();
        VkDescriptorSetAllocateInfo ai{};
        ai.sType              = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        ai.descriptorPool     = m_pool;
        ai.descriptorSetCount = 1;
        ai.pSetLayouts        = &layout;
        if (vkAllocateDescriptorSets(m_device, &ai, &m_set) != VK_SUCCESS) {
            HRP_LOG_ERROR("DummyShadowMap: vkAllocateDescriptorSets failed");
            return false;
        }

        VkDescriptorImageInfo ii{};
        ii.sampler     = m_sampler;
        ii.imageView   = m_view;
        ii.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        VkWriteDescriptorSet w{};
        w.sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        w.dstSet          = m_set;
        w.dstBinding      = 0;
        w.descriptorCount = 1;
        w.descriptorType  = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        w.pImageInfo      = &ii;
        vkUpdateDescriptorSets(m_device, 1, &w, 0, nullptr);

        return true;
    }

    void Destroy(Renderer& renderer)
    {
        if (m_pool)    { vkDestroyDescriptorPool(m_device, m_pool,    nullptr); m_pool    = VK_NULL_HANDLE; }
        if (m_sampler) { vkDestroySampler       (m_device, m_sampler, nullptr); m_sampler = VK_NULL_HANDLE; }
        if (m_view)    { vkDestroyImageView     (m_device, m_view,    nullptr); m_view    = VK_NULL_HANDLE; }
        if (m_image)   { vmaDestroyImage(renderer.GetAllocator(), m_image, m_alloc); m_image = VK_NULL_HANDLE; m_alloc = VK_NULL_HANDLE; }
    }

    [[nodiscard]] VkDescriptorSet Set() const { return m_set; }

private:
    void TransitionToShaderRead(Renderer& renderer)
    {
        VkCommandBufferAllocateInfo cbai{};
        cbai.sType              = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        cbai.commandPool        = renderer.GetCommandPool();
        cbai.level              = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        cbai.commandBufferCount = 1;
        VkCommandBuffer cmd = VK_NULL_HANDLE;
        vkAllocateCommandBuffers(m_device, &cbai, &cmd);

        VkCommandBufferBeginInfo bi{};
        bi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        vkBeginCommandBuffer(cmd, &bi);

        VkImageMemoryBarrier br{};
        br.sType               = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
        br.oldLayout           = VK_IMAGE_LAYOUT_UNDEFINED;
        br.newLayout           = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        br.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        br.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        br.image               = m_image;
        br.subresourceRange    = { VK_IMAGE_ASPECT_DEPTH_BIT, 0, 1, 0, 1 };
        br.srcAccessMask       = 0;
        br.dstAccessMask       = VK_ACCESS_SHADER_READ_BIT;
        vkCmdPipelineBarrier(cmd,
            VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
            0, 0, nullptr, 0, nullptr, 1, &br);

        vkEndCommandBuffer(cmd);

        VkSubmitInfo si{};
        si.sType              = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        si.commandBufferCount = 1;
        si.pCommandBuffers    = &cmd;
        vkQueueSubmit(renderer.GetGraphicsQueue(), 1, &si, VK_NULL_HANDLE);
        vkQueueWaitIdle(renderer.GetGraphicsQueue());
        vkFreeCommandBuffers(m_device, renderer.GetCommandPool(), 1, &cmd);
    }

    VkDevice              m_device  = VK_NULL_HANDLE;
    VkImage               m_image   = VK_NULL_HANDLE;
    VmaAllocation         m_alloc   = VK_NULL_HANDLE;
    VkImageView           m_view    = VK_NULL_HANDLE;
    VkSampler             m_sampler = VK_NULL_HANDLE;
    VkDescriptorPool      m_pool    = VK_NULL_HANDLE;
    VkDescriptorSet       m_set     = VK_NULL_HANDLE;
};

// ──────────────────────────────────────────────────────────────────────
// main
// ──────────────────────────────────────────────────────────────────────

int main(int /*argc*/, char** /*argv*/)
{
    LogInit();
    HRP_LOG_INFO("Hybrid Render Pipeline — demo starting");

    Platform platform;
    if (!platform.Initialize("Hybrid Render Pipeline — Demo", 1600, 900, WindowMode::Windowed)) {
        HRP_LOG_CRITICAL("Platform init failed");
        return 1;
    }

    Renderer renderer;
    if (!renderer.Initialize(platform)) {
        HRP_LOG_CRITICAL("Renderer init failed");
        return 1;
    }
    renderer.SetOffscreenMode(true);  // ScreenSpaceEffects owns the scene pass
    renderer.SetSceneColorFormat(VK_FORMAT_R16G16B16A16_SFLOAT);
    renderer.SetClearColor(glm::vec4(0.50f, 0.65f, 0.92f, 1.0f));  // sky blue fallback

    ScreenSpaceEffects sse;
    if (!sse.Initialize(renderer)) {
        HRP_LOG_CRITICAL("ScreenSpaceEffects init failed");
        return 1;
    }

    DummyShadowMap shadowMap;
    if (!shadowMap.Create(renderer)) {
        HRP_LOG_CRITICAL("Dummy shadow map setup failed");
        return 1;
    }

    // Build scene geometry
    auto terrainVerts = GenerateTerrainMesh(/*gridSize=*/192, /*worldSize=*/240.0f, /*heightScale=*/36.0f);
    auto skyVerts     = GenerateSkyDomeMesh(/*radius=*/450.0f, /*rings=*/24, /*sectors=*/32);

    MeshHandle terrainMesh{};
    MeshHandle skyMesh{};
    if (!renderer.CreateMesh(terrainVerts, terrainMesh) || !renderer.CreateMesh(skyVerts, skyMesh)) {
        HRP_LOG_CRITICAL("Mesh upload failed");
        return 1;
    }

    Camera camera;
    camera.SetTarget(glm::vec3(0.0f, 8.0f, 0.0f));
    camera.SetMaxDistance(160.0f);
    camera.SetDistance(95.0f);
    camera.SetPitch(22.0f);
    camera.SetFarPlane(800.0f);

    glm::mat4 prevViewProj = glm::mat4(1.0f);
    uint32_t  frameIndex   = 0;
    auto      startTime    = std::chrono::high_resolution_clock::now();
    auto      lastTime     = startTime;

    while (!platform.ShouldQuit()) {
        platform.PollEvents();
        if (platform.WasResized()) {
            renderer.RecreateSwapchain(platform);
            sse.OnSwapchainResized(renderer);
            platform.ClearResizeFlag();
        }

        // Drive camera from mouse drag events (left button held).
        // TODO: replace with the Input module once you wire it.
        static bool   s_dragging  = false;
        static float  s_lastX     = 0.0f;
        static float  s_lastY     = 0.0f;
        for (const SDL_Event& ev : platform.GetFrameEvents()) {
            if (ev.type == SDL_EVENT_MOUSE_BUTTON_DOWN && ev.button.button == SDL_BUTTON_LEFT) {
                s_dragging = true;
                s_lastX = ev.button.x;
                s_lastY = ev.button.y;
            } else if (ev.type == SDL_EVENT_MOUSE_BUTTON_UP && ev.button.button == SDL_BUTTON_LEFT) {
                s_dragging = false;
            } else if (ev.type == SDL_EVENT_MOUSE_MOTION && s_dragging) {
                camera.ProcessMouseMovement(ev.motion.x - s_lastX, ev.motion.y - s_lastY);
                s_lastX = ev.motion.x;
                s_lastY = ev.motion.y;
            } else if (ev.type == SDL_EVENT_MOUSE_WHEEL) {
                camera.ProcessMouseScroll(int(ev.wheel.y > 0 ? +1 : -1));
            }
        }

        auto now = std::chrono::high_resolution_clock::now();
        float dt = std::chrono::duration<float>(now - lastTime).count();
        float t  = std::chrono::duration<float>(now - startTime).count();
        lastTime = now;

        camera.Update();

        const auto extent = renderer.GetSwapchainExtent();
        const float aspect = float(extent.width) / float(std::max(extent.height, 1u));

        const glm::mat4 view = camera.GetViewMatrix();
        const glm::mat4 proj = camera.GetProjectionMatrix(aspect);
        const glm::mat4 viewProj = proj * view;

        // Sky mesh travels with the camera so its centre is always at the eye.
        const glm::mat4 skyModel = glm::translate(glm::mat4(1.0f), camera.GetPosition());
        const glm::mat4 skyMVP   = viewProj * skyModel;

        // Slowly rotate the sun for visible lighting change.
        const float sunAngle = 0.35f + 0.18f * t;
        const glm::vec3 sunDir = glm::normalize(glm::vec3(std::cos(sunAngle), 0.62f, std::sin(sunAngle)));

        if (!renderer.BeginFrame()) {
            renderer.RecreateSwapchain(platform);
            sse.OnSwapchainResized(renderer);
            continue;
        }

        const VkCommandBuffer cmd = renderer.GetCurrentCommandBuffer();

        sse.BeginScenePass(cmd, glm::vec4(0.50f, 0.65f, 0.92f, 1.0f));

        renderer.Bind3DPipeline();
        renderer.Set3DLightViewProj(glm::mat4(1.0f));  // identity — shadow set is always-lit
        renderer.Bind3DShadowSet(shadowMap.Set());
        renderer.Set3DLightingParams(
            /*lightDir =*/ glm::vec4(sunDir, 0.55f),
            /*cameraPos=*/ glm::vec4(camera.GetPosition(), 0.005f),
            /*fogParams=*/ glm::vec4(80.0f, 380.0f, 0.0f, 1.0f),
            /*fogColor =*/ glm::vec4(0.55f, 0.68f, 0.88f, 0.0f),  // a=shadowStrength=0 (no shadows)
            /*envParams=*/ glm::vec4(0.0f, 1.0f, 1.0f, 0.55f));

        // Sky first (it writes the farthest depth, terrain over-writes it)
        renderer.DrawMesh(skyMesh,     skyMVP);
        renderer.DrawMesh(terrainMesh, viewProj);

        sse.EndScenePass(cmd);

        sse.DispatchAll(cmd, viewProj, prevViewProj, camera.GetPosition(),
                        camera.GetNearPlane(), camera.GetFarPlane(),
                        frameIndex, dt);

        sse.CompositeToSwapchain(cmd, renderer.GetCurrentSwapchainImageView(), extent);

        renderer.EndFrame();

        prevViewProj = viewProj;
        ++frameIndex;
    }

    renderer.WaitIdle();
    renderer.DestroyMesh(skyMesh);
    renderer.DestroyMesh(terrainMesh);
    shadowMap.Destroy(renderer);

    sse.Shutdown();
    renderer.Shutdown();
    platform.Shutdown();
    LogShutdown();
    return 0;
}
