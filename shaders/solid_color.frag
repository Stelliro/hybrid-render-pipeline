/// @file solid_color.frag
/// @brief Simple solid colour output — placeholder for early bringup.
///
/// Outputs a constant dark colour matching the engine's clear colour.
/// Will be replaced by deferred lighting, PBR, and post-processing shaders.

#version 450

layout(location = 0) in vec2 inUV;
layout(location = 0) out vec4 outColor;

layout(push_constant) uniform PushConstants {
    vec4 color;
} pc;

void main()
{
    outColor = pc.color;
}
