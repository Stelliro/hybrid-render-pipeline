#pragma once

/// @file Camera.hpp
/// @brief Third-person orbital camera that follows a target position.
///
/// The camera orbits around a target point (typically the player) using
/// yaw/pitch angles controlled by mouse input. Supports zoom via scroll wheel.
/// Provides view and projection matrices for the 3D rendering pipeline.

#include <glm/glm.hpp>

namespace hrp {

class Camera {
public:
    /// Sets the position the camera orbits around (e.g., player position).
    void SetTarget(const glm::vec3& target) { m_target = target; }

    /// Processes mouse movement to rotate the camera orbit.
    /// @param xOffset  Horizontal mouse delta (pixels).
    /// @param yOffset  Vertical mouse delta (pixels).
    void ProcessMouseMovement(float xOffset, float yOffset);

    /// Processes scroll wheel for zoom in/out.
    /// @param scroll  +1 zoom in, -1 zoom out.
    void ProcessMouseScroll(int scroll);

    /// Recalculates the camera world position from target + yaw/pitch/distance.
    void Update();

    /// Returns the view matrix (world → camera space).
    [[nodiscard]] glm::mat4 GetViewMatrix() const;

    /// Returns the perspective projection matrix.
    /// @param aspectRatio  Width / height of the viewport.
    [[nodiscard]] glm::mat4 GetProjectionMatrix(float aspectRatio) const;

    /// Returns the camera's world-space position.
    [[nodiscard]] glm::vec3 GetPosition() const { return m_position; }

    /// Returns the normalized forward direction (target - position).
    [[nodiscard]] glm::vec3 GetForward() const;

    /// Returns the normalized right direction (perpendicular to forward + world up).
    [[nodiscard]] glm::vec3 GetRight() const;

    /// Returns the yaw angle in degrees (horizontal rotation around Y axis).
    [[nodiscard]] float GetYaw() const { return m_yaw; }

    /// Returns the pitch angle in degrees (vertical rotation).
    [[nodiscard]] float GetPitch() const { return m_pitch; }

    /// Returns the current orbit distance from target.
    [[nodiscard]] float GetDistance() const { return m_distance; }

    /// Returns true when zoomed all the way in (first-person mode).
    [[nodiscard]] bool IsFirstPerson() const { return m_distance <= m_fpThreshold; }

    /// Sets the yaw angle directly (degrees).
    void SetYaw(float yaw) { m_yaw = yaw; }

    /// Sets the pitch angle directly (degrees, clamped to min/max).
    void SetPitch(float pitch) { m_pitch = glm::clamp(pitch, m_minPitch, m_maxPitch); }

    /// Sets the orbit distance directly (clamped to min/max).
    void SetDistance(float dist) { m_distance = glm::clamp(dist, m_minDistance, m_maxDistance); }

    /// Sets the maximum orbit distance (useful for aerial camera during world loading).
    void SetMaxDistance(float maxD) { m_maxDistance = maxD; }

    /// Sets the horizontal shoulder offset (+right, -left, 0 center).
    void SetShoulderOffset(float offset) { m_shoulderOffset = offset; }

    /// Returns the current shoulder offset.
    [[nodiscard]] float GetShoulderOffset() const { return m_shoulderOffset; }

    /// Sets the vertical field of view (degrees), clamped to [30, 150].
    void SetFOV(float fovDegrees) { m_fov = glm::clamp(fovDegrees, 30.0f, 150.0f); }

    /// Returns the current vertical field of view (degrees).
    [[nodiscard]] float GetFOV() const { return m_fov; }

    /// Sets mouse sensitivity multiplier.
    void SetSensitivity(float sens) { m_sensitivity = glm::clamp(sens, 0.01f, 2.0f); }

    /// Sets the far clipping plane distance.
    void SetFarPlane(float farPlane) { m_farPlane = farPlane; }

    /// Returns the current far clipping plane distance.
    [[nodiscard]] float GetFarPlane() const { return m_farPlane; }

    /// Returns the current near clipping plane distance.
    [[nodiscard]] float GetNearPlane() const { return m_nearPlane; }

private:
    glm::vec3 m_target{ 0.0f, 1.0f, 0.0f };
    glm::vec3 m_position{ 0.0f, 5.0f, 8.0f };

    float m_distance    = 6.0f;      ///< Distance from target.
    float m_yaw         = 0.0f;      ///< Horizontal angle (degrees around Y axis).
    float m_pitch       = 12.0f;     ///< Vertical angle (degrees, 0 = level, 90 = top-down).
    float m_sensitivity = 0.15f;     ///< Mouse sensitivity multiplier.

    float m_minPitch    = -60.0f;    ///< Minimum pitch (allows looking well above horizon).
    float m_maxPitch    = 85.0f;     ///< Maximum pitch (near top-down).
    float m_minDistance  = 0.0f;      ///< Minimum zoom distance (0 = first person).
    float m_maxDistance  = 6.0f;      ///< Maximum gameplay zoom distance (kept tight for stability — see Engine creation/finalize sites).
    float m_zoomSpeed    = 1.5f;      ///< Scroll wheel zoom speed.
    float m_fpThreshold  = 0.5f;      ///< Distance at or below which first-person activates.
    float m_shoulderOffset = 0.75f;  ///< Horizontal shoulder offset (+right, -left, 0 center).

    glm::vec3 m_fpLookDir{ 0.0f, 0.0f, -1.0f };      ///< First-person look direction (set by Update).
    glm::vec3 m_shoulderTarget{ 0.0f, 1.0f, 0.0f };   ///< Shoulder-offset look-at point (third-person).

    float m_fov         = 60.0f;     ///< Vertical field of view (degrees).
    float m_nearPlane   = 0.1f;      ///< Near clipping plane.
    float m_farPlane    = 500.0f;    ///< Far clipping plane.
};

} // namespace hrp
