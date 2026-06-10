/// @file Input.cpp
/// @brief SDL3 input processing — keyboard, mouse, gamepad.

#include "Input.hpp"
#include "Platform.hpp"
#include "Log.hpp"

#include <algorithm>
#include <cmath>

namespace hrp {

void Input::Update(const Platform& platform)
{
    // Snapshot previous state
    m_keyPrevious   = m_keyCurrent;
    m_mousePrevious = m_mouseCurrent;
    m_gpPrevious    = m_gpCurrent;

    // Reset per-frame accumulators
    m_mouseDelta = { 0.0f, 0.0f };
    m_mouseWheel = 0;

    // Process all SDL events captured by the Platform this frame
    for (const SDL_Event& event : platform.GetFrameEvents()) {
        switch (event.type) {

            // ── keyboard ─────────────────────────────────────────────
            case SDL_EVENT_KEY_DOWN:
                if (event.key.scancode < k_MaxKeys) {
                    m_keyCurrent[event.key.scancode] = true;
                }
                break;

            case SDL_EVENT_KEY_UP:
                if (event.key.scancode < k_MaxKeys) {
                    m_keyCurrent[event.key.scancode] = false;
                }
                break;

            // ── mouse buttons ────────────────────────────────────────
            case SDL_EVENT_MOUSE_BUTTON_DOWN:
                if (event.button.button < k_MaxMouseButtons) {
                    m_mouseCurrent[event.button.button] = true;
                }
                break;

            case SDL_EVENT_MOUSE_BUTTON_UP:
                if (event.button.button < k_MaxMouseButtons) {
                    m_mouseCurrent[event.button.button] = false;
                }
                break;

            // ── mouse motion ─────────────────────────────────────────
            case SDL_EVENT_MOUSE_MOTION:
                m_mousePos.x = event.motion.x;
                m_mousePos.y = event.motion.y;
                m_mouseDelta.x += event.motion.xrel;
                m_mouseDelta.y += event.motion.yrel;
                break;

            // ── mouse wheel ──────────────────────────────────────────
            case SDL_EVENT_MOUSE_WHEEL:
                m_mouseWheel += static_cast<int>(event.wheel.y);
                break;

            // ── gamepad ──────────────────────────────────────────────
            case SDL_EVENT_GAMEPAD_ADDED:
                if (!m_gamepad) {
                    TryOpenGamepad();
                }
                break;

            case SDL_EVENT_GAMEPAD_REMOVED:
                if (m_gamepad) {
                    HRP_LOG_INFO("Input: Gamepad disconnected");
                    SDL_CloseGamepad(m_gamepad);
                    m_gamepad = nullptr;
                    m_gpCurrent.fill(false);
                }
                break;

            case SDL_EVENT_GAMEPAD_BUTTON_DOWN:
                if (event.gbutton.button < k_MaxGamepadButtons) {
                    m_gpCurrent[event.gbutton.button] = true;
                }
                break;

            case SDL_EVENT_GAMEPAD_BUTTON_UP:
                if (event.gbutton.button < k_MaxGamepadButtons) {
                    m_gpCurrent[event.gbutton.button] = false;
                }
                break;

            default:
                break;
        }
    }
}

// ── Keyboard queries ─────────────────────────────────────────────────

bool Input::IsKeyDown(SDL_Scancode key) const
{
    return (key < k_MaxKeys) && m_keyCurrent[key];
}

bool Input::IsKeyPressed(SDL_Scancode key) const
{
    return (key < k_MaxKeys) && m_keyCurrent[key] && !m_keyPrevious[key];
}

bool Input::IsKeyReleased(SDL_Scancode key) const
{
    return (key < k_MaxKeys) && !m_keyCurrent[key] && m_keyPrevious[key];
}

// ── Mouse queries ────────────────────────────────────────────────────

bool Input::IsMouseButtonDown(uint8_t button) const
{
    return (button < k_MaxMouseButtons) && m_mouseCurrent[button];
}

bool Input::IsMouseButtonPressed(uint8_t button) const
{
    return (button < k_MaxMouseButtons) && m_mouseCurrent[button] && !m_mousePrevious[button];
}

// ── Gamepad queries ──────────────────────────────────────────────────

float Input::GetAxisValue(SDL_GamepadAxis axis) const
{
    if (!m_gamepad) return 0.0f;

    float raw = static_cast<float>(SDL_GetGamepadAxis(m_gamepad, axis)) / 32767.0f;

    // Apply dead-zone
    if (std::abs(raw) < k_DeadZone) return 0.0f;

    // Remap from [deadzone, 1.0] to [0.0, 1.0]
    float sign = (raw > 0.0f) ? 1.0f : -1.0f;
    float remapped = (std::abs(raw) - k_DeadZone) / (1.0f - k_DeadZone);
    return sign * std::clamp(remapped, 0.0f, 1.0f);
}

bool Input::IsGamepadButtonDown(SDL_GamepadButton button) const
{
    return (button < k_MaxGamepadButtons) && m_gpCurrent[button];
}

bool Input::IsGamepadButtonPressed(SDL_GamepadButton button) const
{
    return (button < k_MaxGamepadButtons) && m_gpCurrent[button] && !m_gpPrevious[button];
}

// ── Utility ──────────────────────────────────────────────────────────

void Input::SetMouseRelativeMode(bool captured, SDL_Window* window)
{
    SDL_SetWindowRelativeMouseMode(window, captured);
    if (captured) {
        SDL_HideCursor();
    } else {
        SDL_ShowCursor();
    }
    m_relativeMode = captured;
}

void Input::TryOpenGamepad()
{
    int count = 0;
    SDL_JoystickID* joysticks = SDL_GetGamepads(&count);
    if (joysticks && count > 0) {
        m_gamepad = SDL_OpenGamepad(joysticks[0]);
        if (m_gamepad) {
            HRP_LOG_INFO("Input: Gamepad connected — {}",
                         SDL_GetGamepadName(m_gamepad));
        }
    }
    SDL_free(joysticks);
}

} // namespace hrp
