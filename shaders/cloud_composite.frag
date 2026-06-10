/// @file cloud_composite.frag
/// @brief Composites the volumetric cloud texture onto the scene.
///
/// Reads the cloud ray-march output (scattering RGB + transmittance A)
/// and applies it to the scene color via:
///   finalColor = sceneColor * cloudTransmittance + cloudScattering
///
/// Uses a fullscreen triangle (no vertex buffer) paired with fullscreen_tri.vert.

#version 450

layout(location = 0) in vec2 inUV;
layout(location = 0) out vec4 outColor;

// ── Bindings ─────────────────────────────────────────────────────────

layout(set = 0, binding = 0) uniform sampler2D sceneColor;       // Current scene
layout(set = 0, binding = 1) uniform sampler2D cloudTexture;     // RGB=scattering, A=transmittance

// ── Push Constants ───────────────────────────────────────────────────

layout(push_constant) uniform CloudCompositeParams {
    float cloudIntensity;    // Global cloud rendering intensity [0,1]
    float _pad0;
    float _pad1;
    float _pad2;
} params;

void main()
{
    vec3 scene = texture(sceneColor, inUV).rgb;
    vec4 cloud = texture(cloudTexture, inUV);

    vec3  cloudScatter      = cloud.rgb;
    float cloudTransmittance = cloud.a;

    // Apply cloud with intensity control
    vec3 composited = scene * mix(1.0, cloudTransmittance, params.cloudIntensity)
                    + cloudScatter * params.cloudIntensity;

    outColor = vec4(composited, 1.0);
}
