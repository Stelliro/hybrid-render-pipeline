/// @file terrain.vert
/// @brief Terrain vertex shader — world-space output for triplanar texturing,
///        biome environment profiles, surface material mapping, shadow mapping,
///        and volumetric light in the fragment shader.
///
/// Inputs per-vertex:
///   position        (vec3) — world-space XYZ
///   normal          (vec3) — surface normal
///   texCoord        (vec2) — planar UV for fallback
///   biomeData       (vec4) — x=primary biome, y=secondary biome, z=blend, w=slope
///   surfaceMaterial (vec4) — x=primary material layer, y=secondary, z=blend, w=depth
///
/// Push constants:
///   MVP matrix (64 bytes, vertex stage)
///   Fragment params include lightViewProj for shadow mapping

#version 450

// ── Vertex attributes ────────────────────────────────────────────────
layout(location = 0) in vec3 inPosition;
layout(location = 1) in vec3 inNormal;
layout(location = 2) in vec2 inTexCoord;
layout(location = 3) in vec4 inBiomeData;       // x=biome0, y=biome1, z=blendFactor, w=slopeFactor
layout(location = 4) in vec4 inSurfaceMaterial;  // x=matLayer0, y=matLayer1, z=matBlend, w=depthExposure

// ── Outputs to fragment shader ───────────────────────────────────────
layout(location = 0) out vec3 fragWorldPos;
layout(location = 1) out vec3 fragNormal;
layout(location = 2) out vec2 fragTexCoord;
layout(location = 3) out vec4 fragBiomeData;
layout(location = 4) out vec4 fragLightSpacePos;     // Position in shadow map space
layout(location = 5) out vec4 fragSurfaceMaterial;    // Surface material data for texture sampling

// ── Push constants ───────────────────────────────────────────────────
layout(push_constant) uniform PushConstants {
    mat4 mvp;                           // bytes 0–63: camera MVP
    // Fragment push constants follow at offset 64 — we only read lightViewProj here
    layout(offset = 144) mat4 lightViewProj;  // bytes 144–207: light-space VP for shadow mapping
} pc;

void main()
{
    gl_Position         = pc.mvp * vec4(inPosition, 1.0);
    fragWorldPos        = inPosition;
    fragNormal          = inNormal;
    fragTexCoord        = inTexCoord;
    fragBiomeData       = inBiomeData;
    fragLightSpacePos   = pc.lightViewProj * vec4(inPosition, 1.0);
    fragSurfaceMaterial = inSurfaceMaterial;
}
