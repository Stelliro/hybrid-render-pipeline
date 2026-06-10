#version 450

// ═══════════════════════════════════════════════════════════════════════
//  ui.frag — 2D UI fragment shader for Hybrid Render Pipeline custom UI
//
//  Samples a font atlas texture (alpha channel) and modulates by vertex
//  color.  For solid-color geometry, the atlas contains a white pixel
//  at UV (0,0) so the texture factor is (1,1,1,1).
// ═══════════════════════════════════════════════════════════════════════

layout(set = 0, binding = 0) uniform sampler2D fontAtlas;

layout(location = 0) in vec2  fragUV;
layout(location = 1) in vec4  fragColor;

layout(location = 0) out vec4 outColor;

void main()
{
    vec4 texel = texture(fontAtlas, fragUV);
    outColor   = fragColor * texel;
}
