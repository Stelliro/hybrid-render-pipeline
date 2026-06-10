#pragma once

/// @file Types.hpp
/// @brief Common type aliases used across the Hybrid Render Pipeline.

#include <cstdint>
#include <glm/glm.hpp>
#include <glm/gtc/quaternion.hpp>

namespace hrp {

// ── Transform ────────────────────────────────────────────────────────

struct Transform {
    glm::vec3 position{ 0.0f };
    glm::quat rotation{ 1.0f, 0.0f, 0.0f, 0.0f };  // Identity
    glm::vec3 scale{ 1.0f };

    [[nodiscard]] glm::mat4 ToMatrix() const;
};

// ── Window mode ──────────────────────────────────────────────────────

enum class WindowMode : uint8_t {
    Windowed,
    Borderless,
    Fullscreen,
};

} // namespace hrp
