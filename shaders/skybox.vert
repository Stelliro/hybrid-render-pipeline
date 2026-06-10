/// @file skybox.vert
/// @brief Skybox vertex shader — fullscreen triangle outputting view-space ray directions.
///
/// Generates a screen-covering triangle from gl_VertexIndex. Uses the inverse
/// view-projection matrix (push constant) to reconstruct world-space ray
/// directions for procedural sky rendering.
///
/// Depth is written as 1.0 (farthest) so all scene geometry naturally occludes the sky.

#version 450

layout(location = 0) out vec2 outUV;
layout(location = 1) out vec3 outViewRay;

// Push constants: inverse view-projection matrix
layout(push_constant) uniform PushConstants {
    mat4  invViewProj;   // bytes 0–63
    vec4  sunDir;        // bytes 64–79  (xyz = direction toward sun, w = intensity)
    vec4  cameraPos;     // bytes 80–95  (xyz = world pos, w = time)
    vec4  skyParams;     // bytes 96–111 (x = dayProgress, y = cloudCover, z = starIntensity, w = ringAngle)
    vec4  skyParams2;    // bytes 112–127 (x = meteorTimer, y = meteorActive, z = seed+yearProgress, w = planetPhase)
    vec4  skyParams3;    // bytes 128–143 (x = alienCraftTimer, y = alienCraftType, z = alienCraftSeed, w = reserved)
} pc;

void main()
{
    // Generate fullscreen triangle from vertex index
    outUV = vec2((gl_VertexIndex << 1) & 2, gl_VertexIndex & 2);
    vec2 ndc = outUV * 2.0 - 1.0;

    // Write at maximum depth (1.0) — sky is always behind everything
    gl_Position = vec4(ndc, 1.0, 1.0);

    // Reconstruct world-space view ray from NDC
    vec4 worldPos = pc.invViewProj * vec4(ndc, 1.0, 1.0);
    outViewRay = worldPos.xyz / worldPos.w - pc.cameraPos.xyz;
}
