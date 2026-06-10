/// @file shadow_depth.frag
/// @brief Minimal fragment shader for shadow depth pass.
///
/// Some Vulkan drivers require a fragment shader even for depth-only rendering.
/// This shader does nothing — depth is written automatically by the fixed-function
/// depth test. No colour output is produced.

#version 450

void main()
{
    // Depth is written automatically by the pipeline's depth test.
    // No colour output — the shadow pass has zero colour attachments.
}
