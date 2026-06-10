/// @file basic_3d.vert
/// @brief 3D vertex shader — MVP transform with per-vertex color, world position,
///        and shadow-map coordinate output for shadow receiving.
///
/// Used for rendering player model, debug geometry, obstacles, and vegetation.
/// Takes position + color per vertex, transforms by a push-constant MVP matrix.
/// Passes world position and light-space position to fragment shader.

#version 450

layout(location = 0) in vec3 inPosition;
layout(location = 1) in vec3 inColor;

layout(location = 0) out vec3 fragColor;
layout(location = 1) out vec3 fragWorldPos;
layout(location = 2) out vec4 fragLightSpacePos;

layout(push_constant) uniform PushConstants {
    mat4 mvp;            // bytes 0–63
    mat4 lightViewProj;  // bytes 64–127
} pc;

void main()
{
    gl_Position      = pc.mvp * vec4(inPosition, 1.0);
    fragColor        = inColor;
    fragWorldPos     = inPosition;
    fragLightSpacePos = pc.lightViewProj * vec4(inPosition, 1.0);
}
