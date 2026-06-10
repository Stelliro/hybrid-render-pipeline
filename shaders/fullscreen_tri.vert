/// @file fullscreen_tri.vert
/// @brief Fullscreen triangle vertex shader — no vertex buffer needed.
///
/// Generates a screen-covering triangle from gl_VertexIndex (0, 1, 2).
/// Used for post-processing, deferred lighting, and clear-to-colour passes.

#version 450

layout(location = 0) out vec2 outUV;

void main()
{
    // Generate fullscreen triangle from vertex index
    // Index 0 → (-1, -1), Index 1 → (3, -1), Index 2 → (-1, 3)
    outUV = vec2((gl_VertexIndex << 1) & 2, gl_VertexIndex & 2);
    gl_Position = vec4(outUV * 2.0 - 1.0, 0.0, 1.0);
}
