/// @file Camera.cpp
/// @brief Third-person orbital camera implementation.

#include "Camera.hpp"

#include <glm/gtc/matrix_transform.hpp>
#include <algorithm>
#include <cmath>

namespace hrp {

void Camera::ProcessMouseMovement(float xOffset, float yOffset)
{
    m_yaw   -= xOffset * m_sensitivity;   // Mouse right → yaw decreases (negative from 0)
    m_pitch += yOffset * m_sensitivity;   // Mouse up (neg yOffset) → pitch decreases → camera lowers → look toward horizon

    // Wrap yaw to [0, 360)
    if (m_yaw >= 360.0f) m_yaw -= 360.0f;
    if (m_yaw < 0.0f)    m_yaw += 360.0f;

    // Clamp pitch
    m_pitch = std::clamp(m_pitch, m_minPitch, m_maxPitch);
}

void Camera::ProcessMouseScroll(int scroll)
{
    m_distance -= static_cast<float>(scroll) * m_zoomSpeed;
    m_distance = std::clamp(m_distance, m_minDistance, m_maxDistance);
}

void Camera::Update()
{
    float yawRad   = glm::radians(m_yaw);
    float pitchRad = glm::radians(m_pitch);

    if (IsFirstPerson()) {
        // First-person: camera sits at the target (eye level)
        m_position = m_target;

        // Compute a look direction from yaw/pitch so GetViewMatrix works
        float cosPitch = std::cos(pitchRad);
        m_fpLookDir.x = -cosPitch * std::sin(yawRad);
        m_fpLookDir.y = -std::sin(pitchRad);
        m_fpLookDir.z = -cosPitch * std::cos(yawRad);
    } else {
        // Third-person: orbit around the target
        float cosPitch = std::cos(pitchRad);
        m_position.x = m_target.x + m_distance * cosPitch * std::sin(yawRad);
        m_position.y = m_target.y + m_distance * std::sin(pitchRad);
        m_position.z = m_target.z + m_distance * cosPitch * std::cos(yawRad);

        // Apply shoulder offset (shift camera sideways, look-at stays near player)
        if (std::abs(m_shoulderOffset) > 0.001f) {
            // Right vector on the XZ plane from current yaw
            glm::vec3 right(std::cos(yawRad), 0.0f, -std::sin(yawRad));
            glm::vec3 offset = right * m_shoulderOffset;
            m_position += offset;
            // Look-at stays at/near player so camera peers past the shoulder
            m_shoulderTarget = m_target + offset * 0.15f;
        } else {
            m_shoulderTarget = m_target;
        }

        m_fpLookDir = glm::vec3(0.0f); // unused in third-person
    }
}

glm::mat4 Camera::GetViewMatrix() const
{
    if (IsFirstPerson()) {
        // Look in the direction computed during Update()
        return glm::lookAt(m_position, m_position + m_fpLookDir, glm::vec3(0.0f, 1.0f, 0.0f));
    }
    return glm::lookAt(m_position, m_shoulderTarget, glm::vec3(0.0f, 1.0f, 0.0f));
}

glm::mat4 Camera::GetProjectionMatrix(float aspectRatio) const
{
    glm::mat4 proj = glm::perspective(glm::radians(m_fov), aspectRatio, m_nearPlane, m_farPlane);
    // Vulkan clip space: Y is inverted compared to OpenGL
    proj[1][1] *= -1.0f;
    return proj;
}

glm::vec3 Camera::GetForward() const
{
    glm::vec3 dir = m_target - m_position;
    float len = glm::length(dir);
    return (len > 0.0001f) ? dir / len : glm::vec3(0.0f, 0.0f, -1.0f);
}

glm::vec3 Camera::GetRight() const
{
    glm::vec3 forward = GetForward();
    glm::vec3 worldUp(0.0f, 1.0f, 0.0f);
    glm::vec3 right = glm::normalize(glm::cross(forward, worldUp));
    return right;
}

} // namespace hrp
