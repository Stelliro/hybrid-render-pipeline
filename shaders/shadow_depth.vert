/// @file shadow_depth.vert
/// @brief Shadow map depth pass — transforms terrain vertices by light-space MVP.
///
/// Used in the shadow mapping pass where the scene is rendered from the sun's
/// point of view. Only position is needed; all other vertex attributes are ignored
/// but must be declared to match the TerrainVertex stride.

#version 450

// ── Vertex attributes (must match TerrainVertex layout) ──────────────
layout(location = 0) in vec3 inPosition;
layout(location = 1) in vec3 inNormal;          // Unused but required for stride
layout(location = 2) in vec2 inTexCoord;         // Unused but required for stride
layout(location = 3) in vec4 inBiomeData;        // Unused but required for stride
layout(location = 4) in vec4 inSurfaceMaterial;  // Unused but required for stride

// ── Push constant: light-space MVP ───────────────────────────────────
layout(push_constant) uniform PushConstants {
    mat4 lightMVP;
} pc;

void main()
{
    gl_Position = pc.lightMVP * vec4(inPosition, 1.0);
}
