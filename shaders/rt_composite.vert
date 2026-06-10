/// @file rt_composite.vert
/// @brief Fullscreen triangle vertex shader for RT composite pass.
///
/// Generates a fullscreen triangle from gl_VertexIndex (no vertex buffer).
/// Outputs UV coordinates for the fragment shader to sample all RT buffers.
///
/// Triangle vertices (CCW):
///   v0 = (-1, -1)  uv = (0, 0)
///   v1 = ( 3, -1)  uv = (2, 0)
///   v2 = (-1,  3)  uv = (0, 2)

#version 450

layout(location = 0) out vec2 outUV;

void main()
{
    // Fullscreen triangle — no vertex buffer needed
    outUV = vec2((gl_VertexIndex << 1) & 2, gl_VertexIndex & 2);
    gl_Position = vec4(outUV * 2.0 - 1.0, 0.0, 1.0);
}
