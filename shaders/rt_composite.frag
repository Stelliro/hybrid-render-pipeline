/// @file rt_composite.frag
/// @brief Composites all RT approximation buffers (GI, Reflections, AO)
///        into the final scene color with physically-based blending.
///
/// This is the final pass that combines:
///   - Base scene color (forward-rendered)
///   - Screen-space GI (indirect diffuse)
///   - Hi-Z Reflections (indirect specular)
///   - GTAO+ (ambient occlusion + bent normals)
///   - Temporal denoised results
///
/// Uses a fullscreen triangle (no vertex buffer) to write the composited
/// result. The composition follows energy-conservation principles:
///   finalColor = (albedo * (ambientGI + directLight) * AO) + reflections
///
/// Also applies final post-processing:
///   - Chromatic aberration (subtle, camera-lens simulation)
///   - Film grain (optional, for cinematic feel)
///   - Vignette (subtle darkening at edges)

#version 450

layout(location = 0) in vec2 inUV;
layout(location = 0) out vec4 outColor;

// ── Bindings ─────────────────────────────────────────────────────────

layout(set = 0, binding = 0) uniform sampler2D sceneColor;
layout(set = 0, binding = 1) uniform sampler2D giBuffer;            // RGB=indirect, A=confidence
layout(set = 0, binding = 2) uniform sampler2D reflectionBuffer;    // RGB=reflected, A=confidence
layout(set = 0, binding = 3) uniform sampler2D aoBuffer;            // R=AO, GBA=bent normal
layout(set = 0, binding = 4) uniform sampler2D depthBuffer;
layout(set = 0, binding = 5) uniform sampler2D normalRoughness;     // RGB=normal, A=roughness

// ── Push Constants ───────────────────────────────────────────────────

layout(push_constant) uniform CompositeParams {
    vec4 compositeParams;    // x = gi intensity, y = reflection intensity, z = ao intensity, w = time
    vec4 postFxParams;       // x = chromatic aberration, y = film grain, z = vignette, w = exposure
    vec4 screenParams;       // x = width, y = height, z = 1/width, w = 1/height
} params;

// ── Constants ────────────────────────────────────────────────────────

const float PI = 3.14159265358979;

// ── Hash for film grain ──────────────────────────────────────────────

float hash12(vec2 p, float seed)
{
    vec3 p3 = fract(vec3(p.xyx) * 0.1031 + seed);
    p3 += dot(p3, p3.yzx + 33.33);
    return fract((p3.x + p3.y) * p3.z);
}

float bayer4(ivec2 p)
{
    const float bayer[16] = float[](
         0.0,  8.0,  2.0, 10.0,
        12.0,  4.0, 14.0,  6.0,
         3.0, 11.0,  1.0,  9.0,
        15.0,  7.0, 13.0,  5.0
    );
    ivec2 q = p & ivec2(3);
    return bayer[q.y * 4 + q.x] / 16.0;
}

// ══════════════════════════════════════════════════════════════════════
// ACES Filmic Tone Mapping (matches skybox.frag)
// ══════════════════════════════════════════════════════════════════════

vec3 ACESFilm(vec3 x)
{
    float a = 2.51;
    float b = 0.03;
    float c = 2.43;
    float d = 0.59;
    float e = 0.14;
    return clamp((x * (a * x + b)) / (x * (c * x + d) + e), 0.0, 1.0);
}

// ══════════════════════════════════════════════════════════════════════
// Main
// ══════════════════════════════════════════════════════════════════════

void main()
{
    vec2 uv = inUV;

    // ── Read all buffers ─────────────────────────────────────────────

    vec4 scene      = texture(sceneColor, uv);
    vec4 gi         = texture(giBuffer, uv);
    vec4 reflection = texture(reflectionBuffer, uv);
    vec4 ao         = texture(aoBuffer, uv);
    float depth     = texture(depthBuffer, uv).r;

    vec3 baseColor = scene.rgb;

    // Sky pixels — no RT composition needed, but still apply tone mapping
    // since all shaders (including skybox) now output linear HDR.
    if (depth >= 0.9999)
    {
        float exposure = params.postFxParams.w;
        baseColor *= exposure;
        baseColor = ACESFilm(baseColor);
        baseColor = pow(baseColor, vec3(1.0 / 2.2));
        outColor = vec4(baseColor, 1.0);
        return;
    }

    vec4  normalRough = texture(normalRoughness, uv);
    float roughness   = normalRough.a;

    // ── Apply Ambient Occlusion ──────────────────────────────────────
    // AO modulates both direct and indirect lighting

    float aoValue    = ao.r;
    vec3  bentNormal = normalize(ao.gba * 2.0 - 1.0);

    float aoIntensity = params.compositeParams.z;
    float appliedAO = mix(1.0, aoValue, aoIntensity);

    baseColor *= appliedAO;

    // ── Apply Global Illumination ────────────────────────────────────
    // GI adds indirect diffuse light, modulated by AO and confidence

    float giIntensity  = params.compositeParams.x;
    float giConfidence = gi.a;
    vec3  indirectLight = gi.rgb * giIntensity * giConfidence;

    // Bent normal modulation — GI contributes more from unoccluded directions
    float bentAlignment = max(dot(bentNormal, normalize(gi.rgb + 0.001)), 0.0);
    indirectLight *= mix(0.6, 1.0, bentAlignment);

    // AO attenuates indirect light too
    indirectLight *= appliedAO;

    baseColor += indirectLight;

    // ── Apply Reflections ────────────────────────────────────────────
    // Reflections are additive, weighted by confidence and material roughness

    float reflIntensity  = params.compositeParams.y;
    float reflConfidence = reflection.a;

    // Roughness-based reflection weight
    float reflWeight = (1.0 - roughness * roughness) * reflIntensity * reflConfidence;

    baseColor += reflection.rgb * reflWeight;

    // ── Post-Processing ──────────────────────────────────────────────

    // Chromatic aberration (subtle radial)
    float caStrength = params.postFxParams.x;
    if (caStrength > 0.001)
    {
        vec2  center  = vec2(0.5);
        vec2  fromCenter = uv - center;
        float caFactor = length(fromCenter) * caStrength;

        vec2 uvR = uv + fromCenter * caFactor * 1.0;
        vec2 uvB = uv - fromCenter * caFactor * 1.0;

        float r = texture(sceneColor, uvR).r;
        float b = texture(sceneColor, uvB).b;

        // Only apply CA to the original base, not the composited result
        // to keep RT effects clean. Apply as a subtle color shift.
        baseColor.r = mix(baseColor.r, r, 0.3 * caStrength);
        baseColor.b = mix(baseColor.b, b, 0.3 * caStrength);
    }

    // Vignette
    float vignetteStrength = params.postFxParams.z;
    if (vignetteStrength > 0.001)
    {
        vec2  center = vec2(0.5);
        float dist   = length(uv - center);
        float vignette = 1.0 - smoothstep(0.4, 0.9, dist) * vignetteStrength;
        baseColor *= vignette;
    }

    // Exposure
    float exposure = params.postFxParams.w;
    baseColor *= exposure;

    // ── Tone mapping (ACES) and gamma ────────────────────────────────

    baseColor = ACESFilm(baseColor);
    baseColor = pow(baseColor, vec3(1.0 / 2.2));

    // Film grain (perceptual — applied in gamma space so noise is uniform)
    float grainStrength = params.postFxParams.y;
    if (grainStrength > 0.001)
    {
        float grain = hash12(uv * params.screenParams.xy, params.compositeParams.w);
        grain = (grain - 0.5) * grainStrength;

        // Suppress grain in deep shadow so night scenes read as flat darkness
        // with clean silhouettes instead of crawling static. Grain ramps in
        // over [0.06, 0.30] luminance — invisible at night, full in lit areas.
        float baseLum = dot(baseColor, vec3(0.2126, 0.7152, 0.0722));
        float grainWeight = smoothstep(0.06, 0.30, baseLum);
        baseColor += vec3(grain) * grainWeight;
    }

    // Output dithering — prevents visible color banding without adding a
    // moving random grain floor to night scenes. Deep shadows fade the dither
    // down aggressively because sparse ±1/255 noise is very visible there.
    {
        float baseLum = dot(baseColor, vec3(0.2126, 0.7152, 0.0722));
        float shadowFade = smoothstep(0.025, 0.18, baseLum);
        float ordered = bayer4(ivec2(gl_FragCoord.xy)) - 0.5;
        baseColor += vec3(ordered * shadowFade * 0.35 / 255.0);
    }

    outColor = vec4(baseColor, 1.0);
}
