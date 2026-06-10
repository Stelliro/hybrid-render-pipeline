/// @file Platform.cpp
/// @brief SDL3 window management and Vulkan surface creation.

#include "Platform.hpp"
#include "Log.hpp"

#include <SDL3/SDL_vulkan.h>

namespace hrp {

bool Platform::Initialize(const std::string& title, int width, int height, WindowMode mode)
{
    HRP_LOG_INFO("Platform: Initializing SDL3 ...");

    if (!SDL_Init(SDL_INIT_VIDEO | SDL_INIT_GAMEPAD)) {
        HRP_LOG_CRITICAL("Platform: SDL_Init failed — {}", SDL_GetError());
        return false;
    }

    m_width  = width;
    m_height = height;
    m_currentMode = mode;

    // Determine SDL window flags
    Uint64 flags = SDL_WINDOW_VULKAN | SDL_WINDOW_RESIZABLE | SDL_WINDOW_HIGH_PIXEL_DENSITY;

    switch (mode) {
        case WindowMode::Borderless:
            flags |= SDL_WINDOW_BORDERLESS;
            break;
        case WindowMode::Fullscreen:
            flags |= SDL_WINDOW_FULLSCREEN;
            break;
        case WindowMode::Windowed:
        default:
            break;
    }

    m_window = SDL_CreateWindow(title.c_str(), width, height, flags);
    if (!m_window) {
        HRP_LOG_CRITICAL("Platform: SDL_CreateWindow failed — {}", SDL_GetError());
        SDL_Quit();
        return false;
    }

    HRP_LOG_INFO("Platform: Window created — \"{}\" ({}×{}, mode={})",
                 title, width, height, static_cast<int>(mode));

    SDL_StartTextInput(m_window);

    return true;
}

void Platform::Shutdown()
{
    HRP_LOG_INFO("Platform: Shutting down ...");

    if (m_window) {
        SDL_StopTextInput(m_window);
        SDL_DestroyWindow(m_window);
        m_window = nullptr;
    }

    SDL_Quit();
    HRP_LOG_INFO("Platform: SDL3 shut down");
}

void Platform::PollEvents()
{
    m_frameEvents.clear();
    m_wasResized = false;

    SDL_Event event;
    while (SDL_PollEvent(&event)) {
        m_frameEvents.push_back(event);

        switch (event.type) {
            case SDL_EVENT_QUIT:
                m_shouldQuit = true;
                break;

            case SDL_EVENT_WINDOW_RESIZED:
                m_width  = event.window.data1;
                m_height = event.window.data2;
                m_wasResized = true;
                HRP_LOG_DEBUG("Platform: Window resized to {}×{}", m_width, m_height);
                break;

            case SDL_EVENT_WINDOW_MINIMIZED:
                HRP_LOG_DEBUG("Platform: Window minimized");
                break;

            case SDL_EVENT_WINDOW_RESTORED:
                HRP_LOG_DEBUG("Platform: Window restored");
                break;

            default:
                break;
        }
    }
}

glm::ivec2 Platform::GetWindowSize() const
{
    return { m_width, m_height };
}

std::vector<const char*> Platform::GetRequiredVulkanExtensions() const
{
    Uint32 count = 0;
    const char* const* names = SDL_Vulkan_GetInstanceExtensions(&count);

    if (!names) {
        HRP_LOG_ERROR("Platform: SDL_Vulkan_GetInstanceExtensions failed — {}", SDL_GetError());
        return {};
    }

    std::vector<const char*> extensions(names, names + count);
    return extensions;
}

VkSurfaceKHR Platform::CreateVulkanSurface(VkInstance instance) const
{
    VkSurfaceKHR surface = VK_NULL_HANDLE;

    if (!SDL_Vulkan_CreateSurface(m_window, instance, nullptr, &surface)) {
        HRP_LOG_ERROR("Platform: SDL_Vulkan_CreateSurface failed — {}", SDL_GetError());
        return VK_NULL_HANDLE;
    }

    HRP_LOG_INFO("Platform: Vulkan surface created successfully");
    return surface;
}

void Platform::SetWindowMode(WindowMode mode)
{
    if (mode == m_currentMode) return;

    switch (mode) {
        case WindowMode::Windowed:
            SDL_SetWindowFullscreen(m_window, false);
            SDL_SetWindowBordered(m_window, true);
            break;
        case WindowMode::Borderless:
            SDL_SetWindowFullscreen(m_window, false);
            SDL_SetWindowBordered(m_window, false);
            break;
        case WindowMode::Fullscreen:
            SDL_SetWindowFullscreen(m_window, true);
            break;
    }

    m_currentMode = mode;
    m_wasResized = true;
    HRP_LOG_INFO("Platform: Window mode changed to {}", static_cast<int>(mode));
}

} // namespace hrp
