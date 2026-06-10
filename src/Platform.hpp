#pragma once

/// @file Platform.hpp
/// @brief SDL3 window management and Vulkan surface creation.
///
/// The Platform module is the lowest-level component of the Hybrid Render Pipeline.
/// It manages the OS window via SDL3 and provides a VkSurfaceKHR for the renderer.
/// First to initialize, last to shut down.

#include "Types.hpp"
#include "Log.hpp"        // HRP_LOG_INFO/WARN/ERROR (callers expect these via Platform.hpp)
#include "ErrorCodes.hpp" // HRP_WARN_ONCE

#include <SDL3/SDL.h>
#include <vulkan/vulkan.h>
#include <string>
#include <vector>

namespace hrp {

class Platform {
public:
    /// Creates the SDL3 window with SDL_WINDOW_VULKAN flag and initializes SDL subsystems.
    /// @return true on success, false on fatal error.
    bool Initialize(const std::string& title, int width, int height, WindowMode mode);

    /// Destroys the SDL window and SDL context. Called last during shutdown.
    void Shutdown();

    /// Pumps the SDL event queue. Called once per frame on the main thread.
    /// Sets m_shouldQuit if the user requested window close.
    void PollEvents();

    /// Returns the current drawable dimensions in pixels (DPI-aware).
    [[nodiscard]] glm::ivec2 GetWindowSize() const;

    /// Returns the raw SDL_Window pointer (needed by Vulkan surface creation).
    [[nodiscard]] SDL_Window* GetWindow() const { return m_window; }

    /// Returns the list of Vulkan instance extensions required by SDL3.
    [[nodiscard]] std::vector<const char*> GetRequiredVulkanExtensions() const;

    /// Creates a VkSurfaceKHR from the SDL window. Called by the Renderer during init.
    /// @param instance A valid VkInstance.
    /// @return The created surface, or VK_NULL_HANDLE on failure.
    VkSurfaceKHR CreateVulkanSurface(VkInstance instance) const;

    /// Toggles between windowed, borderless, and exclusive fullscreen.
    void SetWindowMode(WindowMode mode);

    /// Returns true if the window was resized this frame (triggers swapchain recreation).
    [[nodiscard]] bool WasResized() const { return m_wasResized; }

    /// Clears the resize flag. Called by the renderer after handling.
    void ClearResizeFlag() { m_wasResized = false; }

    /// Returns true if the user (or OS) requested the application to quit.
    [[nodiscard]] bool ShouldQuit() const { return m_shouldQuit; }

    /// Returns the SDL events captured this frame (for the Input module to process).
    [[nodiscard]] const std::vector<SDL_Event>& GetFrameEvents() const { return m_frameEvents; }

private:
    SDL_Window* m_window     = nullptr;
    int         m_width      = 0;
    int         m_height     = 0;
    bool        m_wasResized = false;
    bool        m_shouldQuit = false;
    WindowMode  m_currentMode = WindowMode::Windowed;

    /// Events captured during PollEvents() — consumed by the Input module.
    std::vector<SDL_Event> m_frameEvents;
};

} // namespace hrp
