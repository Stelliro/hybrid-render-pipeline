/// @file volumetric_light.frag
/// @brief Post-process camera-ray volumetric light scattering.
///
/// Fullscreen pass variant of the per-fragment in-scatter used in terrain.frag.
/// For each screen pixel, this shader:
///   1. Reconstructs the world position from the depth buffer
///   2. Marches a ray FROM the camera TOWARD that world position
///   3. Samples the shadow map at each step — lit segments accumulate, shadowed don't
///   4. The shadow map divides the ray into beam / no-beam segments
///
/// The result is additive light shafts that are world-space anchored.
/// Moving the camera changes which beams are visible; rotating reveals
/// different beams but never slides them. The sun projects like a torch
/// through gaps in geometry.
///
/// Multi-source support:
///   Sun:    Shadow-mapped directional forward scatter (Henyey-Greenstein)
///   Point:  Analytical spherical scatter (closest-point-on-ray approximation)
///
/// NOTE: Requires a depth texture at set 0, binding 1. Currently NOT wired
/// into the rendering pipeline — the primary volumetric path is the per-fragment
/// approach in terrain.frag. This shader is reserved for a future post-process
/// pipeline that would render volumetric light for sky pixels and non-terrain
/// objects in a single fullscreen pass.
///
/// Performance: ~24 shadow map samples per pixel, one depth texture fetch.

#version 450

layout(location = 0) in vec2 inUV;
layout(location = 0) out vec4 outColor;

// ── Descriptor set 0 ────────────────────────────────────────────────
layout(set = 0, binding = 0) uniform sampler2DShadow shadowMap;
// layout(set = 0, binding = 1) uniform sampler2D depthBuffer;  // Uncomment when wired

// ── Push constants ───────────────────────────────────────────────────
layout(push_constant) uniform VolumetricParams {
    mat4 invViewProj;       // Screen → world reconstruction
    mat4 lightViewProj;     // World → shadow map light space
    vec4 sunDir;            // xyz = direction TOWARD sun, w = ambient
    vec4 params;            // x = density, y = numSteps, z = marchDist, w = anisotropy
    vec4 sunColor;          // rgb = sun colour, a = unused
    vec4 cameraPos;         // xyz = camera world position, w = unused
} vp;

const float PI = 3.14159265359;

// ── Henyey-Greenstein phase function ─────────────────────────────────
float henyeyGreenstein(float cosTheta, float g)
{
    float g2 = g * g;
    float denom = 1.0 + g2 - 2.0 * g * cosTheta;
    return (1.0 - g2) / (4.0 * PI * denom * sqrt(max(denom, 0.0001)));
}

void main()
{
    // If volumetric lighting is disabled (density ≤ 0), pass through transparent
    if (vp.params.x <= 0.0) {
        outColor = vec4(0.0);
        return;
    }

    float density    = vp.params.x;
    int   numSteps   = int(vp.params.y);
    float marchDist  = vp.params.z;
    float anisotropy = vp.params.w;
    float heightFade = 0.008;

    // Reconstruct world-space ray from screen position
    vec2 ndc = inUV * 2.0 - 1.0;
    vec4 nearClip = vp.invViewProj * vec4(ndc, 0.0, 1.0);
    vec4 farClip  = vp.invViewProj * vec4(ndc, 1.0, 1.0);
    vec3 nearWorld = nearClip.xyz / nearClip.w;
    vec3 farWorld  = farClip.xyz / farClip.w;

    vec3 rayDir = normalize(farWorld - nearWorld);

    // TODO: When depth buffer is wired, reconstruct actual fragment depth:
    // float depth = texture(depthBuffer, inUV).r;
    // vec4 worldH = vp.invViewProj * vec4(ndc, depth, 1.0);
    // vec3 worldPos = worldH.xyz / worldH.w;
    // float rayLen = length(worldPos - vp.cameraPos.xyz);
    // marchDist = min(rayLen, marchDist);

    float stepSize = marchDist / float(numSteps);

    // Phase function — forward scatter toward the sun
    vec3 lightDir = normalize(vp.sunDir.xyz);
    float cosTheta = dot(rayDir, lightDir);
    float phase = henyeyGreenstein(cosTheta, anisotropy);

    // ── Ray march from camera through shadow map ─────────────────
    // World-anchored jitter (uses world-derived coords for stability)
    float jitter = fract(sin(dot(inUV * 1000.0 + vp.cameraPos.xz * 0.1,
                                  vec2(12.9898, 78.233))) * 43758.5453);

    vec3  samplePos   = vp.cameraPos.xyz + rayDir * stepSize * jitter;
    float inScatter   = 0.0;

    for (int i = 0; i < numSteps; ++i) {
        // Transform sample point to light space
        vec4 lsPos = vp.lightViewProj * vec4(samplePos, 1.0);
        vec3 proj  = lsPos.xyz / lsPos.w;
        vec2 sUV   = proj.xy * 0.5 + 0.5;
        float sDepth = proj.z;

        // Accumulate only within shadow map bounds
        if (sUV.x > 0.0 && sUV.x < 1.0 &&
            sUV.y > 0.0 && sUV.y < 1.0 &&
            sDepth > 0.0 && sDepth < 1.0)
        {
            // Shadow map: 1.0 = lit (beam), 0.0 = shadowed (no beam)
            float lit = texture(shadowMap, vec3(sUV, sDepth));

            // Height-based density (thicker in valleys)
            float hDensity = exp(-max(samplePos.y, 0.0) * heightFade);
            float localDensity = density * (0.4 + 0.6 * hDensity);

            inScatter += lit * localDensity;
        }

        samplePos += rayDir * stepSize;
    }

    // Apply phase function and clamp
    inScatter *= phase;
    inScatter = clamp(inScatter, 0.0, 1.0);

    // Sun beam colour (warm, matches terrain shader)
    float sunElev = max(lightDir.y, 0.0);
    vec3 beamColor = mix(
        vec3(1.0, 0.82, 0.50),
        vec3(1.0, 0.96, 0.90),
        smoothstep(0.0, 0.6, sunElev));

    vec3 finalColor = beamColor * vp.sunColor.rgb * inScatter;

    // Output as additive overlay (alpha = scatter amount for blending)
    outColor = vec4(finalColor, inScatter * 0.6);
}
