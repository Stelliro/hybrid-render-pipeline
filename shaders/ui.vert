#version 450

// ═══════════════════════════════════════════════════════════════════════
//  ui.vert — 2D UI vertex shader for Hybrid Render Pipeline custom UI
//
//  Transforms screen-space pixel coordinates to Vulkan NDC.
//  Vertex format: pos(vec2) + uv(vec2) + color(packed u32 ABGR).
// ═══════════════════════════════════════════════════════════════════════

layout(push_constant) uniform PushConstants {
    vec2 scale;      // vec2(2.0 / viewport_width, 2.0 / viewport_height)
    vec2 translate;  // vec2(-1.0, -1.0)
} pc;

layout(location = 0) in vec2  inPos;
layout(location = 1) in vec2  inUV;
layout(location = 2) in uint  inColor;  // ABGR packed

layout(location = 0) out vec2  fragUV;
layout(location = 1) out vec4  fragColor;

void main()
{
    // Unpack ABGR → linear RGBA
    fragColor.r = float((inColor >>  0u) & 0xFFu) / 255.0;
    fragColor.g = float((inColor >>  8u) & 0xFFu) / 255.0;
    fragColor.b = float((inColor >> 16u) & 0xFFu) / 255.0;
    fragColor.a = float((inColor >> 24u) & 0xFFu) / 255.0;

    fragUV      = inUV;
    gl_Position = vec4(inPos * pc.scale + pc.translate, 0.0, 1.0);
}
