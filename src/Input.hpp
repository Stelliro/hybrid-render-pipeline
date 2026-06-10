#pragma once

/// @file Input.hpp
/// @brief SDL3 input state management — keyboard, mouse, and gamepad.
///
/// Processes SDL events captured by the Platform module and provides a
/// clean, frame-consistent query API for game logic.

#include "Types.hpp"

#include <SDL3/SDL.h>
#include <glm/glm.hpp>
#include <array>

namespace hrp {

class Platform;  // forward declaration

class Input {
public:
    /// Processes all SDL3 events from the Platform's event queue.
    /// Updates internal key/button state arrays. Called once per frame.
    void Update(const Platform& platform);

    // ── Keyboard (SDL scancodes) ─────────────────────────────────────

    /// True every frame the key is held.
    [[nodiscard]] bool IsKeyDown(SDL_Scancode key) const;

    /// True only on the frame the key was first pressed.
    [[nodiscard]] bool IsKeyPressed(SDL_Scancode key) const;

    /// True only on the frame the key was released.
    [[nodiscard]] bool IsKeyReleased(SDL_Scancode key) const;

    // ── Mouse ────────────────────────────────────────────────────────

    /// True every frame the button is held. Uses SDL_BUTTON_LEFT, etc.
    [[nodiscard]] bool IsMouseButtonDown(uint8_t button) const;

    /// True only on the frame the button was first clicked.
    [[nodiscard]] bool IsMouseButtonPressed(uint8_t button) const;

    /// Pixel coordinates within the window.
    [[nodiscard]] glm::vec2 GetMousePosition() const { return m_mousePos; }

    /// Per-frame cursor delta (for camera control).
    [[nodiscard]] glm::vec2 GetMouseDelta() const { return m_mouseDelta; }

    /// Scroll wheel: +1 up, -1 down, 0 no movement.
    [[nodiscard]] int GetMouseWheelScroll() const { return m_mouseWheel; }

    // ── Gamepad (SDL GameController API) ─────────────────────────────

    /// Returns axis value in [-1.0, 1.0] with dead-zone applied.
    [[nodiscard]] float GetAxisValue(SDL_GamepadAxis axis) const;

    /// True every frame the button is held.
    [[nodiscard]] bool IsGamepadButtonDown(SDL_GamepadButton button) const;

    /// True only on the frame the button was first pressed.
    [[nodiscard]] bool IsGamepadButtonPressed(SDL_GamepadButton button) const;

    /// True if any gamepad is connected.
    [[nodiscard]] bool IsGamepadConnected() const { return m_gamepad != nullptr; }

    // ── Utility ──────────────────────────────────────────────────────

    /// Locks/unlocks cursor to center for FPS-style camera control.
    /// Requires the SDL window handle (SDL3 changed this from a global call).
    void SetMouseRelativeMode(bool captured, SDL_Window* window);

    /// Returns true if relative mouse mode is active.
    [[nodiscard]] bool IsMouseRelativeMode() const { return m_relativeMode; }

private:
    static constexpr int k_MaxKeys = SDL_SCANCODE_COUNT;
    static constexpr int k_MaxMouseButtons = 6;
    static constexpr int k_MaxGamepadButtons = SDL_GAMEPAD_BUTTON_COUNT;
    static constexpr float k_DeadZone = 0.15f;

    // Keyboard state: current and previous frame
    std::array<bool, k_MaxKeys> m_keyCurrent{};
    std::array<bool, k_MaxKeys> m_keyPrevious{};

    // Mouse state
    std::array<bool, k_MaxMouseButtons> m_mouseCurrent{};
    std::array<bool, k_MaxMouseButtons> m_mousePrevious{};
    glm::vec2 m_mousePos{ 0.0f };
    glm::vec2 m_mouseDelta{ 0.0f };
    int       m_mouseWheel = 0;

    // Gamepad state
    SDL_Gamepad* m_gamepad = nullptr;
    std::array<bool, k_MaxGamepadButtons> m_gpCurrent{};
    std::array<bool, k_MaxGamepadButtons> m_gpPrevious{};

    bool m_relativeMode = false;

    /// Tries to open the first available gamepad.
    void TryOpenGamepad();
};

} // namespace hrp
