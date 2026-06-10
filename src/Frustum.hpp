#pragma once

/// @file Frustum.hpp
/// @brief Shared frustum plane extraction and AABB testing.
///
/// Extracted from ChunkManager.hpp and OcclusionCullingSystem.hpp to
/// provide a single definition of hrp::Frustum used by both terrain
/// streaming (ChunkManager) and occlusion culling (OcclusionCullingSystem).

#include <glm/glm.hpp>
#include <array>
#include <cmath>

namespace hrp {

// ══════════════════════════════════════════════════════════════════════
// Frustum (6-plane)
// ══════════════════════════════════════════════════════════════════════

/// Six planes extracted from a view-projection matrix (Gribb-Hartmann method).
/// Each plane is (a, b, c, d) where ax + by + cz + d = 0, normal = (a,b,c).
struct Frustum {
    std::array<glm::vec4, 6> planes;  ///< Left, Right, Bottom, Top, Near, Far.

    /// Extract frustum planes from a view-projection matrix (mutates this).
    void ExtractFromVP(const glm::mat4& vp);

    /// Factory: extract frustum planes from a View-Projection matrix.
    static Frustum Extract(const glm::mat4& vp) {
        Frustum f;
        f.ExtractFromVP(vp);
        return f;
    }

    /// Tests an AABB against the frustum (binary).
    /// @return true if the AABB is at least partially inside (not fully outside).
    [[nodiscard]] bool TestAABB(const glm::vec3& aabbMin,
                                const glm::vec3& aabbMax) const;

    /// Tests an AABB against the frustum (tri-state).
    /// @return -1 = fully outside, 0 = intersecting, 1 = fully inside.
    [[nodiscard]] int TestAABBEx(const glm::vec3& aabbMin,
                                 const glm::vec3& aabbMax) const {
        int result = 1; // Assume fully inside
        for (int i = 0; i < 6; ++i) {
            const glm::vec3 normal(planes[i]);
            float d = planes[i].w;

            // P-vertex: furthest along the normal
            glm::vec3 pVertex = aabbMin;
            if (normal.x >= 0) pVertex.x = aabbMax.x;
            if (normal.y >= 0) pVertex.y = aabbMax.y;
            if (normal.z >= 0) pVertex.z = aabbMax.z;

            // N-vertex: closest to the plane
            glm::vec3 nVertex = aabbMax;
            if (normal.x >= 0) nVertex.x = aabbMin.x;
            if (normal.y >= 0) nVertex.y = aabbMin.y;
            if (normal.z >= 0) nVertex.z = aabbMin.z;

            // If N-vertex is outside → fully outside this plane
            if (glm::dot(normal, nVertex) + d > 0.0f) {
                return -1;
            }
            // If P-vertex is outside → partially outside (intersecting)
            if (glm::dot(normal, pVertex) + d > 0.0f) {
                result = 0;
            }
        }
        return result;
    }
};

} // namespace hrp
