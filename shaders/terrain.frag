/// @file terrain.frag
/// @brief Terrain fragment shader — per-biome environment-aware lighting with
///        surface material map texture sampling, camera-ray volumetric shadow
///        segmentation, triplanar mapping, smooth material blending, PCF soft
///        shadows, Cook-Torrance specular, and atmospheric scattering.
///
/// Dual data model:
///   biomeData       → Drives environment profiles (ambient, fog, exposure, SSS).
///                     Biome indices are used for lighting/atmosphere parameters.
///   surfaceMaterial → Drives actual texture layer sampling and blending.
///                     Material indices come from the TerrainMaterialMap which
///                     accounts for voxel destruction depth, sub-biome variation,
///                     and procedural noise for organic material transitions.
///
/// Environment Lighting System:
///   Each biome has a day and night lighting profile controlling ambient color,
///   exposure, fog density, shadow depth, roughness, and SSS intensity.
///   All parameters are blended per-fragment between the two active biomes
///   using the vertex-interpolated biome blend factor. The day/night profiles
///   are smoothly interpolated via the nightFade factor computed from ambient.
///   This means walking from bright open grassland into a dark dense forest
///   transitions seamlessly at per-pixel resolution.
///
/// Surface Material System:
///   The surface material map provides per-vertex primary + secondary texture
///   layer indices with a blend factor. When terrain is voxel-destructed,
///   the depth exposure value drives subsurface material revelation (topsoil →
///   subsoil → bedrock). Material transitions use noise-perturbed blending
///   for organic edges without hard lines.
///
/// Volumetric Lighting ("Camera-Ray Shadow Segmentation"):
///   For each fragment, a ray is marched FROM the camera TOWARD the surface.
///   At each sample point, the shadow map is queried: where the sun is NOT
///   blocked, in-scattered light accumulates. Where shadows exist, the ray
///   segment contributes nothing. This divides the view ray into lit and unlit
///   segments, creating world-anchored volumetric light shafts that stay fixed
///   in space — moving the camera reveals different beams but never slides them.
///
/// Texture array layers (bound at set=0, binding=0):
///   0 = Grass         (Grassland)
///   1 = Forest floor  (Forest)
///   2 = Dirt/Mud      (Wetland)
///   3 = Sand          (Beach)
///   4 = Rock          (Highland/Mountain)
///   5 = Snow          (Snow)
///   6 = Deep water    (DeepWater)
///   7 = Shallow water (ShallowWater)
///   8 = Cliff rock    (slope-driven overlay — steep surfaces)
///   9 = Riverbed      10 = CaveWall     11 = Erosion
///  12 = Sediment      13 = WonderRock   14 = MossyStone

#version 450

// ── Inputs from vertex shader ────────────────────────────────────────
layout(location = 0) in vec3 fragWorldPos;
layout(location = 1) in vec3 fragNormal;
layout(location = 2) in vec2 fragTexCoord;
layout(location = 3) in vec4 fragBiomeData;       // x=biome0, y=biome1, z=blend, w=slope
layout(location = 4) in vec4 fragLightSpacePos;   // From shadow map VP transform
layout(location = 5) in vec4 fragSurfaceMaterial;  // x=matLayer0, y=matLayer1, z=matBlend, w=depthExposure

// ── Output ───────────────────────────────────────────────────────────
layout(location = 0) out vec4 outColor;

// ── Descriptor set 0: terrain texture array + sampler ────────────────
layout(set = 0, binding = 0) uniform sampler2DArray terrainTextures;

// ── Descriptor set 1: shadow map (depth comparison sampler) ──────────
layout(set = 1, binding = 0) uniform sampler2DShadow shadowMap;

// ── Terrain parameters (push constant, after MVP in vertex stage) ────
layout(push_constant) uniform TerrainParams {
    layout(offset = 64) vec4 lightDir;       // xyz = sun direction, w = ambient strength
    vec4 terrainParams;   // x = texScale, y = triplanarSharpness, z = slopeThreshold, w = slopeBlendRange
    vec4 fogParams;       // x = fogStart, y = fogEnd, z = godRayDensity (0=off), w = shadowMapSize
    vec4 fogColor;        // rgb = fog color, a = shadow strength
    vec4 cameraPos;       // xyz = camera position, w = shadow bias
    mat4 lightViewProj;   // bytes 144–207: light-space VP (also used by vertex stage)
} tp;

// ── Constants ────────────────────────────────────────────────────────
const int LAYER_SNOW  = 5;
const int LAYER_CLIFF = 8;
const float PI = 3.14159265359;

// ══════════════════════════════════════════════════════════════════════
// Per-biome environment lighting profiles
// ══════════════════════════════════════════════════════════════════════
//
// Each biome defines a distinct visual atmosphere for both day and night.
// Parameters are looked up per-fragment using the biome layer index and
// blended between the two active biomes for seamless transitions.
//
// Profile parameters:
//   ambientTint   — Biome-specific colored ambient light (RGB)
//   ambientScale  — Multiplier on base ambient intensity (0.5=dark, 1.0=normal)
//   fogDensityMul — Fog density multiplier (0.5=clear, 2.0=thick haze)
//   exposureBias  — EV offset for tone mapping (-1.0=darker, +1.0=brighter)
//   roughnessBase — Surface roughness for this biome (0.3=smooth, 0.95=rough)
//   shadowDepth   — Shadow darkness multiplier (1.0=normal, 1.5=deep shadows)
//   sssScale      — Subsurface scattering intensity (0=none, 1.0=full SSS)
//
// Night profiles have separate values allowing each biome to feel unique
// in low light — forests become impenetrably dark while snow fields
// retain gentle moonlit visibility from high albedo reflection.

/// Environment lighting profile for a biome (day or night).
struct EnvProfile {
    vec3  ambientTint;
    float ambientScale;
    float fogDensityMul;
    float exposureBias;
    float roughnessBase;
    float shadowDepth;
    float sssScale;
};

/// Look up the DAY lighting profile for a biome texture layer index.
EnvProfile getEnvProfileDay(int biome)
{
    EnvProfile p;

    // Defaults
    p.ambientTint   = vec3(0.50, 0.55, 0.60);
    p.ambientScale  = 1.0;
    p.fogDensityMul = 1.0;
    p.exposureBias  = 0.0;
    p.roughnessBase = 0.85;
    p.shadowDepth   = 1.0;
    p.sssScale      = 0.0;

    if (biome == 0) {
        // Grassland — warm, bright, open sky, gentle golden tones
        p.ambientTint   = vec3(0.55, 0.60, 0.50);
        p.ambientScale  = 1.10;
        p.fogDensityMul = 0.85;
        p.exposureBias  = 0.15;
        p.roughnessBase = 0.82;
        p.shadowDepth   = 0.90;
        p.sssScale      = 0.60;
    }
    else if (biome == 1) {
        // Forest — DARK canopy-filtered light, green tint, deep shadows
        p.ambientTint   = vec3(0.30, 0.42, 0.25);
        p.ambientScale  = 0.55;
        p.fogDensityMul = 1.40;
        p.exposureBias  = -0.60;
        p.roughnessBase = 0.90;
        p.shadowDepth   = 1.50;
        p.sssScale      = 1.00;
    }
    else if (biome == 2) {
        // Wetland — murky, muted tones, thick low haze, uncomfortable
        p.ambientTint   = vec3(0.38, 0.42, 0.35);
        p.ambientScale  = 0.65;
        p.fogDensityMul = 1.80;
        p.exposureBias  = -0.40;
        p.roughnessBase = 0.55;
        p.shadowDepth   = 1.20;
        p.sssScale      = 0.40;
    }
    else if (biome == 3) {
        // Beach — bright, warm, reflective sand, wide open sky
        p.ambientTint   = vec3(0.72, 0.68, 0.55);
        p.ambientScale  = 1.20;
        p.fogDensityMul = 0.70;
        p.exposureBias  = 0.30;
        p.roughnessBase = 0.70;
        p.shadowDepth   = 0.80;
        p.sssScale      = 0.15;
    }
    else if (biome == 4) {
        // Highland/Mountain — cool clear air, high contrast, crisp
        p.ambientTint   = vec3(0.48, 0.52, 0.62);
        p.ambientScale  = 0.95;
        p.fogDensityMul = 0.50;
        p.exposureBias  = 0.10;
        p.roughnessBase = 0.88;
        p.shadowDepth   = 1.10;
        p.sssScale      = 0.05;
    }
    else if (biome == 5) {
        // Snow — very bright, high albedo, cold blue-white, dazzling
        p.ambientTint   = vec3(0.70, 0.75, 0.85);
        p.ambientScale  = 1.30;
        p.fogDensityMul = 0.65;
        p.exposureBias  = 0.50;
        p.roughnessBase = 0.40;
        p.shadowDepth   = 0.75;
        p.sssScale      = 0.10;
    }
    else if (biome == 6) {
        // Deep water — dark blue, submerged feel, heavy atmospheric
        p.ambientTint   = vec3(0.20, 0.30, 0.50);
        p.ambientScale  = 0.50;
        p.fogDensityMul = 2.00;
        p.exposureBias  = -0.70;
        p.roughnessBase = 0.25;
        p.shadowDepth   = 0.60;
        p.sssScale      = 0.30;
    }
    else if (biome == 7) {
        // Shallow water — bright aqua reflections, sparkling
        p.ambientTint   = vec3(0.40, 0.55, 0.60);
        p.ambientScale  = 0.85;
        p.fogDensityMul = 1.30;
        p.exposureBias  = -0.20;
        p.roughnessBase = 0.30;
        p.shadowDepth   = 0.70;
        p.sssScale      = 0.25;
    }
    else if (biome == 8) {
        // Cliff rock — exposed crags, cool grey, moderate shadows
        p.ambientTint   = vec3(0.45, 0.48, 0.55);
        p.ambientScale  = 0.90;
        p.fogDensityMul = 0.60;
        p.exposureBias  = 0.05;
        p.roughnessBase = 0.92;
        p.shadowDepth   = 1.15;
        p.sssScale      = 0.0;
    }
    else if (biome == 9) {
        // Corruption — The Blighted Scar: oppressive dark purple haze,
        // sickly desaturated ambient, thick choking fog, deep shadows
        p.ambientTint   = vec3(0.25, 0.15, 0.30);
        p.ambientScale  = 0.60;
        p.fogDensityMul = 2.00;
        p.exposureBias  = -0.40;
        p.roughnessBase = 0.80;
        p.shadowDepth   = 1.40;
        p.sssScale      = 0.0;
    }

    return p;
}

/// Look up the NIGHT lighting profile for a biome texture layer index.
/// Night profiles are dramatically different — each biome has a unique
/// character in darkness: forests are near-black, snow reflects moonlight,
/// wetlands glow with bioluminescence hints, corruption pulses faintly.
EnvProfile getEnvProfileNight(int biome)
{
    EnvProfile p;

    // Defaults — near-black with faintest cool blue hint (true darkness)
    p.ambientTint   = vec3(0.03, 0.04, 0.08);
    p.ambientScale  = 0.25;
    p.fogDensityMul = 1.20;
    p.exposureBias  = -1.00;
    p.roughnessBase = 0.85;
    p.shadowDepth   = 0.20;
    p.sssScale      = 0.0;

    if (biome == 0) {
        // Grassland night — dark open fields, faint silver-blue hint from planet
        p.ambientTint   = vec3(0.04, 0.05, 0.10);
        p.ambientScale  = 0.35;
        p.fogDensityMul = 1.00;
        p.exposureBias  = -0.80;
        p.roughnessBase = 0.82;
        p.shadowDepth   = 0.30;
        p.sssScale      = 0.0;
    }
    else if (biome == 1) {
        // Forest night — IMPENETRABLY dark, canopy blocks all light, terrifying
        p.ambientTint   = vec3(0.01, 0.02, 0.03);
        p.ambientScale  = 0.10;
        p.fogDensityMul = 1.60;
        p.exposureBias  = -2.00;
        p.roughnessBase = 0.92;
        p.shadowDepth   = 0.10;
        p.sssScale      = 0.0;
    }
    else if (biome == 2) {
        // Wetland night — very dark, faint bioluminescence hint only
        p.ambientTint   = vec3(0.02, 0.04, 0.06);
        p.ambientScale  = 0.20;
        p.fogDensityMul = 2.20;
        p.exposureBias  = -1.20;
        p.roughnessBase = 0.55;
        p.shadowDepth   = 0.15;
        p.sssScale      = 0.08;
    }
    else if (biome == 3) {
        // Beach night — dark shore, pale sand barely visible under planet-light
        p.ambientTint   = vec3(0.05, 0.06, 0.12);
        p.ambientScale  = 0.40;
        p.fogDensityMul = 0.90;
        p.exposureBias  = -0.60;
        p.roughnessBase = 0.70;
        p.shadowDepth   = 0.25;
        p.sssScale      = 0.0;
    }
    else if (biome == 4) {
        // Highland/Mountain night — cold dark peaks under clear starlit sky
        p.ambientTint   = vec3(0.03, 0.04, 0.10);
        p.ambientScale  = 0.35;
        p.fogDensityMul = 0.40;
        p.exposureBias  = -0.60;
        p.roughnessBase = 0.88;
        p.shadowDepth   = 0.30;
        p.sssScale      = 0.0;
    }
    else if (biome == 5) {
        // Snow night — snow reflects faint planet-light, brightest night biome
        // but still genuinely dark — no glowing snow fields
        p.ambientTint   = vec3(0.06, 0.08, 0.16);
        p.ambientScale  = 0.60;
        p.fogDensityMul = 0.70;
        p.exposureBias  = -0.20;
        p.roughnessBase = 0.40;
        p.shadowDepth   = 0.35;
        p.sssScale      = 0.0;
    }
    else if (biome == 6) {
        // Deep water night — pitch-black abyss, deeply unsettling
        p.ambientTint   = vec3(0.005, 0.01, 0.03);
        p.ambientScale  = 0.10;
        p.fogDensityMul = 2.50;
        p.exposureBias  = -2.00;
        p.roughnessBase = 0.25;
        p.shadowDepth   = 0.05;
        p.sssScale      = 0.0;
    }
    else if (biome == 7) {
        // Shallow water night — dark water, barely perceptible ripples
        p.ambientTint   = vec3(0.02, 0.04, 0.08);
        p.ambientScale  = 0.25;
        p.fogDensityMul = 1.50;
        p.exposureBias  = -1.00;
        p.roughnessBase = 0.30;
        p.shadowDepth   = 0.15;
        p.sssScale      = 0.0;
    }
    else if (biome == 8) {
        // Cliff rock night — stark cold exposed rock, minimal light
        p.ambientTint   = vec3(0.02, 0.03, 0.06);
        p.ambientScale  = 0.30;
        p.fogDensityMul = 0.50;
        p.exposureBias  = -0.70;
        p.roughnessBase = 0.92;
        p.shadowDepth   = 0.25;
        p.sssScale      = 0.0;
    }
    else if (biome == 9) {
        // Corruption night — abyssal void, faint sickly purple glow
        // from the blight itself, nearly opaque fog
        p.ambientTint   = vec3(0.04, 0.01, 0.06);
        p.ambientScale  = 0.15;
        p.fogDensityMul = 2.50;
        p.exposureBias  = -1.50;
        p.roughnessBase = 0.80;
        p.shadowDepth   = 0.10;
        p.sssScale      = 0.0;
    }

    return p;
}

/// Blend two environment profiles using a linear interpolation factor.
EnvProfile blendEnvProfiles(EnvProfile a, EnvProfile b, float t)
{
    EnvProfile r;
    r.ambientTint   = mix(a.ambientTint,   b.ambientTint,   t);
    r.ambientScale  = mix(a.ambientScale,  b.ambientScale,  t);
    r.fogDensityMul = mix(a.fogDensityMul, b.fogDensityMul, t);
    r.exposureBias  = mix(a.exposureBias,  b.exposureBias,  t);
    r.roughnessBase = mix(a.roughnessBase, b.roughnessBase, t);
    r.shadowDepth   = mix(a.shadowDepth,   b.shadowDepth,   t);
    r.sssScale      = mix(a.sssScale,      b.sssScale,      t);
    return r;
}

/// Compute the final environment profile for this fragment by blending
/// between two biomes (per-vertex blend) and between day/night (time).
EnvProfile computeFragmentEnv(int biome0, int biome1, float biomeMix, float nightFade)
{
    // Day profiles for both biomes
    EnvProfile day0 = getEnvProfileDay(biome0);
    EnvProfile day1 = getEnvProfileDay(biome1);
    // Night profiles for both biomes
    EnvProfile ngt0 = getEnvProfileNight(biome0);
    EnvProfile ngt1 = getEnvProfileNight(biome1);

    // Blend between biomes
    EnvProfile dayBlend = blendEnvProfiles(day0, day1, biomeMix);
    EnvProfile ngtBlend = blendEnvProfiles(ngt0, ngt1, biomeMix);

    // Blend between day and night
    return blendEnvProfiles(dayBlend, ngtBlend, nightFade);
}

// ══════════════════════════════════════════════════════════════════════
// Per-fragment noise for organic biome edge distortion
// ══════════════════════════════════════════════════════════════════════

/// Hash-based pseudo-random gradient noise (GPU-friendly, no texture).
/// Used to perturb biome blend edges at the fragment level so transitions
/// look organic rather than following vertex-grid lines.
vec2 hash22(vec2 p)
{
    p = vec2(dot(p, vec2(127.1, 311.7)),
             dot(p, vec2(269.5, 183.3)));
    return -1.0 + 2.0 * fract(sin(p) * 43758.5453123);
}

float gradientNoise(vec2 p)
{
    vec2 i = floor(p);
    vec2 f = fract(p);
    // Quintic Hermite interpolation (matches Ken Perlin's improved curve)
    vec2 u = f * f * f * (f * (f * 6.0 - 15.0) + 10.0);

    return mix(mix(dot(hash22(i + vec2(0.0, 0.0)), f - vec2(0.0, 0.0)),
                   dot(hash22(i + vec2(1.0, 0.0)), f - vec2(1.0, 0.0)), u.x),
               mix(dot(hash22(i + vec2(0.0, 1.0)), f - vec2(0.0, 1.0)),
                   dot(hash22(i + vec2(1.0, 1.0)), f - vec2(1.0, 1.0)), u.x), u.y);
}

/// Multi-octave FBM for biome edge perturbation (3 octaves).
float biomeEdgeNoise(vec2 p)
{
    float val  = 0.0;
    float amp  = 1.0;
    float freq = 1.0;
    for (int i = 0; i < 3; i++) {
        val  += gradientNoise(p * freq) * amp;
        amp  *= 0.5;
        freq *= 2.17;
    }
    return val; // approximately in [-1, 1]
}

// ══════════════════════════════════════════════════════════════════════
// Triplanar texture sampling
// ══════════════════════════════════════════════════════════════════════

vec4 sampleTriplanar(int layerIdx, vec3 worldPos, vec3 blendW, float scale)
{
    vec4 xP = texture(terrainTextures, vec3(worldPos.yz * scale, float(layerIdx)));
    vec4 yP = texture(terrainTextures, vec3(worldPos.xz * scale, float(layerIdx)));
    vec4 zP = texture(terrainTextures, vec3(worldPos.xy * scale, float(layerIdx)));
    return xP * blendW.x + yP * blendW.y + zP * blendW.z;
}

// ══════════════════════════════════════════════════════════════════════
// Shadow mapping — PCF soft shadows (the "subtraction" of sunlight)
// ══════════════════════════════════════════════════════════════════════

/// Percentage-Closer Filtering for soft shadow edges.
/// Samples the shadow map in a rotated Poisson disk pattern to avoid
/// regular grid artifacts. Returns 0.0 = full shadow, 1.0 = full light.
float calculateShadow(vec4 lightSpacePos, vec3 normal, vec3 lightDir)
{
    // Perspective divide
    vec3 projCoords = lightSpacePos.xyz / lightSpacePos.w;

    // Convert from NDC [-1,1] to UV [0,1] BEFORE bounds check.
    // NDC xy are in [-1,1]; the bounds check must use the [0,1] UV space.
    vec2 shadowUV = projCoords.xy * 0.5 + 0.5;
    float currentDepth = projCoords.z; // Vulkan NDC z is already [0,1]

    // Outside shadow map → fully lit (no shadow data)
    if (shadowUV.x < 0.0 || shadowUV.x > 1.0 ||
        shadowUV.y < 0.0 || shadowUV.y > 1.0 ||
        currentDepth > 1.0 || currentDepth < 0.0)
        return 1.0;

    // Slope-scaled bias to prevent shadow acne on angled surfaces
    float bias = tp.cameraPos.w; // Base bias from push constant
    float slopeBias = max(bias * (1.0 - dot(normal, lightDir)), bias * 0.5);
    currentDepth -= slopeBias;

    // PCF: 13-sample rotated Poisson disk for soft shadow edges
    // This creates the soft penumbra where sun beams fade into shadow
    float shadowMapSize = tp.fogParams.w;
    float texelSize = 1.0 / shadowMapSize;

    // Poisson disk offsets (pre-computed for consistent soft edges)
    const vec2 poissonDisk[13] = vec2[13](
        vec2( 0.0,       0.0),
        vec2(-0.9406,   -0.2836),
        vec2( 0.9452,   -0.7684),
        vec2(-0.0940,   -0.9289),
        vec2( 0.3432,    0.8458),
        vec2(-0.6714,    0.6393),
        vec2( 0.7126,    0.1916),
        vec2(-0.3518,   -0.4731),
        vec2( 0.1380,   -0.3878),
        vec2(-0.7630,   -0.7490),
        vec2( 0.4424,    0.3350),
        vec2(-0.2860,    0.2916),
        vec2( 0.0634,    0.5866)
    );

    // Rotate the disk per-pixel using world position for noise
    float angle = fract(sin(dot(fragWorldPos.xz, vec2(12.9898, 78.233))) * 43758.5453) * PI * 2.0;
    float cosA = cos(angle);
    float sinA = sin(angle);

    float shadow = 0.0;
    float pcfRadius = 1.5;

    for (int i = 0; i < 13; ++i) {
        vec2 offset = poissonDisk[i] * texelSize * pcfRadius;
        // Rotate the offset
        vec2 rotated = vec2(
            offset.x * cosA - offset.y * sinA,
            offset.x * sinA + offset.y * cosA
        );
        shadow += texture(shadowMap, vec3(shadowUV + rotated, currentDepth));
    }
    shadow /= 13.0;

    // Fade shadow at the edges of the shadow map to avoid hard cutoffs
    float fadeRange = 0.05;
    float fadeFactor = 1.0;
    fadeFactor *= smoothstep(0.0, fadeRange, shadowUV.x);
    fadeFactor *= smoothstep(0.0, fadeRange, 1.0 - shadowUV.x);
    fadeFactor *= smoothstep(0.0, fadeRange, shadowUV.y);
    fadeFactor *= smoothstep(0.0, fadeRange, 1.0 - shadowUV.y);
    shadow = mix(1.0, shadow, fadeFactor);

    // Apply shadow strength (configurable: 0 = no shadow, 1 = pitch black)
    float strength = tp.fogColor.a;
    return mix(1.0, shadow, strength);
}

// ══════════════════════════════════════════════════════════════════════
// Cook-Torrance specular (simplified for terrain — wet rock sheen)
// ══════════════════════════════════════════════════════════════════════

/// GGX Normal Distribution Function
float distributionGGX(float NdotH, float roughness)
{
    float a  = roughness * roughness;
    float a2 = a * a;
    float denom = NdotH * NdotH * (a2 - 1.0) + 1.0;
    return a2 / (PI * denom * denom);
}

/// Schlick-GGX geometry function
float geometrySchlickGGX(float NdotV, float roughness)
{
    float k = (roughness + 1.0);
    k = k * k / 8.0;
    return NdotV / (NdotV * (1.0 - k) + k);
}

/// Smith's geometry function (combined for light and view)
float geometrySmith(float NdotV, float NdotL, float roughness)
{
    return geometrySchlickGGX(NdotV, roughness) * geometrySchlickGGX(NdotL, roughness);
}

/// Fresnel-Schlick approximation
vec3 fresnelSchlick(float cosTheta, vec3 F0)
{
    return F0 + (1.0 - F0) * pow(clamp(1.0 - cosTheta, 0.0, 1.0), 5.0);
}

// ══════════════════════════════════════════════════════════════════════
// Camera-ray volumetric in-scatter (shadow-segmented light shafts)
// ══════════════════════════════════════════════════════════════════════

/// Henyey-Greenstein phase function — models anisotropic atmospheric scattering.
///   g = 0 → isotropic, g → 1 → tight forward scatter (narrow shafts).
float henyeyGreenstein(float cosTheta, float g)
{
    float g2 = g * g;
    float denom = 1.0 + g2 - 2.0 * g * cosTheta;
    return (1.0 - g2) / (4.0 * PI * denom * sqrt(max(denom, 0.0001)));
}

/// Marches a ray FROM the camera TOWARD the fragment through world space.
/// At each step, the shadow map is sampled: lit regions accumulate in-scatter,
/// shadowed regions contribute nothing. The shadow map naturally divides the
/// ray into beam/no-beam segments.
///
/// The result is world-anchored — beams stay fixed in space regardless of
/// camera rotation. Moving the camera changes which beams are visible,
/// but never slides them.
///
/// @param fragPos       World-space fragment position (ray endpoint).
/// @param camPos        World-space camera position (ray origin).
/// @param lightDir      Normalized direction TOWARD the sun.
/// @param godRayDensity Scattering density coefficient (0 = disabled, ~0.015 = default).
/// @return              Additive in-scatter color (sun-tinted light shafts).
vec3 calculateSunInScatter(vec3 fragPos, vec3 camPos, vec3 lightDir, float godRayDensity)
{
    // Early-out when god rays are disabled (setting = Off)
    if (godRayDensity <= 0.0001) return vec3(0.0);

    // Volumetric parameters
    const int   NUM_STEPS    = 24;     // Steps per ray (quality vs. performance)
    const float MAX_MARCH    = 200.0;  // Max ray march distance — covers full loaded terrain
    const float SUN_ANISO    = 0.76;   // Forward-scatter anisotropy (0=iso, 1=tight beams)
    const float HEIGHT_FADE  = 0.008;  // Density fades with altitude (thicker in valleys)

    vec3  rayVec   = fragPos - camPos;
    float rayLen   = length(rayVec);
    if (rayLen < 0.1) return vec3(0.0);

    vec3  rayDir   = rayVec / rayLen;
    float marchDist = min(rayLen, MAX_MARCH);
    float stepSize  = marchDist / float(NUM_STEPS);

    // Phase function — concentrates beams when looking toward the sun
    float cosTheta = dot(rayDir, lightDir);
    float phase    = henyeyGreenstein(cosTheta, SUN_ANISO);

    // World-anchored jitter to reduce banding (uses world-space coords,
    // NOT screen coords, so pattern stays fixed in the world)
    float jitter = fract(sin(
        dot(floor(fragPos.xz * 2.0), vec2(12.9898, 78.233)) +
        dot(floor(camPos.xz * 0.3), vec2(4.1414, 27.6159))
    ) * 43758.5453);

    vec3  samplePos = camPos + rayDir * stepSize * jitter;
    float inScatter = 0.0;
    float shadowMapSize = tp.fogParams.w;

    for (int i = 0; i < NUM_STEPS; ++i)
    {
        // Transform sample point to light space
        vec4 lsPos = tp.lightViewProj * vec4(samplePos, 1.0);
        vec3 proj  = lsPos.xyz / lsPos.w;
        vec2 sUV   = proj.xy * 0.5 + 0.5;
        float sDepth = proj.z;

        // Accumulate only within shadow map bounds
        if (sUV.x > 0.0 && sUV.x < 1.0 &&
            sUV.y > 0.0 && sUV.y < 1.0 &&
            sDepth > 0.0 && sDepth < 1.0)
        {
            // Shadow map comparison: 1.0 = sunlit (beam here), 0.0 = shadowed (no beam)
            float lit = texture(shadowMap, vec3(sUV, sDepth));

            // Height-based density: thicker fog in valleys, thinner high up
            float heightDensity = exp(-max(samplePos.y, 0.0) * HEIGHT_FADE);
            float localDensity  = godRayDensity * (0.4 + 0.6 * heightDensity);

            inScatter += lit * localDensity;
        }

        samplePos += rayDir * stepSize;
    }

    // Apply phase function
    inScatter *= phase;

    // Sun beam color (warm, shifts with sun elevation like surface lighting)
    float sunElev = max(lightDir.y, 0.0);
    vec3 beamColor = mix(
        vec3(1.0, 0.82, 0.50),   // Low sun → warm golden beams
        vec3(1.0, 0.96, 0.90),   // High sun → pale white beams
        smoothstep(0.0, 0.6, sunElev)
    );

    return beamColor * clamp(inScatter, 0.0, 1.0);
}

/// Analytical point light atmospheric scattering (no shadow map needed).
/// Computes how much a point light contributes to the atmosphere along the
/// view ray using the closest-point-on-ray approximation. Different light
/// sources create distinct scattering signatures:
///   Torch:  warm orange, small radius, focused
///   Fire:   hot orange-red, medium radius, wider
///   Moon:   cool blue, large radius, very soft
///
/// @param fragPos     World-space fragment position.
/// @param camPos      World-space camera position.
/// @param lightPos    World-space light position.
/// @param lightColor  RGB color of the light source.
/// @param lightRadius Falloff radius (world units).
/// @param intensity   Light intensity multiplier.
/// @return            Additive scatter contribution.
vec3 calculatePointLightScatter(
    vec3  fragPos,
    vec3  camPos,
    vec3  lightPos,
    vec3  lightColor,
    float lightRadius,
    float intensity)
{
    vec3  rayVec = fragPos - camPos;
    float rayLen = length(rayVec);
    if (rayLen < 0.1) return vec3(0.0);

    vec3 rayDir = rayVec / rayLen;

    // Closest point on the view ray to the light source
    float t = clamp(dot(lightPos - camPos, rayDir), 0.0, rayLen);
    vec3  closestPt   = camPos + rayDir * t;
    float distToLight = length(closestPt - lightPos);

    // Smooth Gaussian-like falloff within radius
    float normDist = distToLight / max(lightRadius, 0.1);
    float scatter  = exp(-normDist * normDist * 2.0);

    // Distance attenuation from light source (inverse square, clamped)
    float atten = 1.0 / (1.0 + distToLight * distToLight * 0.05);

    // Ray length modulation — longer rays through the light's sphere accumulate more
    float rayThrough = max(lightRadius - distToLight, 0.0);
    float lengthFactor = rayThrough / max(lightRadius, 0.1);

    return lightColor * scatter * atten * lengthFactor * intensity;
}

// ══════════════════════════════════════════════════════════════════════
// Main fragment shader
// ══════════════════════════════════════════════════════════════════════

void main()
{
    // ── Unpack parameters ────────────────────────────────────────
    float texScale           = tp.terrainParams.x;
    float triplanarSharpness = tp.terrainParams.y;
    float slopeThreshold     = tp.terrainParams.z;
    float slopeBlendRange    = tp.terrainParams.w;

    vec3 normal = normalize(fragNormal);
    vec3 viewDir = normalize(tp.cameraPos.xyz - fragWorldPos);

    // ── Multi-scale normal micro-detail perturbation ────────────
    // Three frequency bands of procedural normal perturbation create
    // realistic surface roughness: macro bumps (rocks/mounds), medium
    // grain (soil clumps/pebbles), and fine grain (sand/dust).
    // Each band fades with distance to prevent shimmer artifacts.
    {
        float camDist = length(fragWorldPos - tp.cameraPos.xyz);

        // Band 1: Macro bumps — broad undulations (freq 0.15, visible to ~200 units)
        float macroFade = 1.0 - smoothstep(80.0, 200.0, camDist);
        vec2 macroUV = fragWorldPos.xz * 0.15;
        float mnx = gradientNoise(macroUV + vec2(0.0, 100.0));
        float mnz = gradientNoise(macroUV + vec2(100.0, 0.0));
        normal = normalize(normal + vec3(mnx, 0.0, mnz) * 0.06 * macroFade);

        // Band 2: Medium grain — soil clumps, pebbles (freq 0.5, visible to ~120 units)
        float medFade = 1.0 - smoothstep(40.0, 120.0, camDist);
        vec2 medUV = fragWorldPos.xz * 0.5;
        float mx = gradientNoise(medUV + vec2(7.3, 130.0));
        float mz = gradientNoise(medUV + vec2(130.0, 7.3));
        normal = normalize(normal + vec3(mx, 0.0, mz) * 0.10 * medFade);

        // Band 3: Fine grain — sand, dust particles (freq 2.0, visible to ~40 units)
        float fineFade = 1.0 - smoothstep(10.0, 40.0, camDist);
        vec2 fineUV = fragWorldPos.xz * 2.0;
        float fx = gradientNoise(fineUV + vec2(55.0, 200.0));
        float fz = gradientNoise(fineUV + vec2(200.0, 55.0));
        normal = normalize(normal + vec3(fx, 0.0, fz) * 0.12 * fineFade);
    }

    // ── Triplanar blend weights ──────────────────────────────────
    vec3 blendW = abs(normal);
    blendW = pow(blendW, vec3(triplanarSharpness));
    float wSum = blendW.x + blendW.y + blendW.z;
    blendW /= max(wSum, 0.001);

    // ── Biome data (environment profiles — lighting, fog, atmosphere) ──
    int   biome0   = int(fragBiomeData.x + 0.5);
    int   biome1   = int(fragBiomeData.y + 0.5);
    float biomeMix = fragBiomeData.z;

    // ── Surface material data (drives actual texture sampling) ───
    int   matLayer0     = int(fragSurfaceMaterial.x + 0.5);
    int   matLayer1     = int(fragSurfaceMaterial.y + 0.5);
    float matBlend      = fragSurfaceMaterial.z;
    float depthExposure = fragSurfaceMaterial.w;

    // Per-fragment noise perturbation to break up straight biome edges.
    // Applied to BOTH biome blend and material blend for consistent edges.
    float edgeNoise = 0.0;
    if (biome0 != biome1 && biomeMix > 0.001) {
        edgeNoise = biomeEdgeNoise(fragWorldPos.xz * 0.04 + vec2(55.5, 66.6));
        float edgeProximity = 1.0 - abs(biomeMix - 0.5) * 2.0;
        biomeMix += edgeNoise * 0.18 * (0.3 + 0.7 * edgeProximity);
        biomeMix = clamp(biomeMix, 0.0, 1.0);
    }

    // Material blend noise perturbation for organic material transitions
    if (matLayer0 != matLayer1 && matBlend > 0.001) {
        // Use slightly different frequency for material edges vs biome edges
        float matNoise = biomeEdgeNoise(fragWorldPos.xz * 0.06 + vec2(33.3, 88.8));
        float matProximity = 1.0 - abs(matBlend - 0.5) * 2.0;
        matBlend += matNoise * 0.22 * (0.3 + 0.7 * matProximity);
        matBlend = clamp(matBlend, 0.0, 1.0);
    }

    // ── Sample surface material textures (multi-scale for detail) ─
    // Three-scale sampling: base (1x), medium detail (4x), fine detail (12x).
    // Medium detail adds soil/pebble-scale texture variation.
    // Fine detail adds grain-level realism at close range (distance-faded).
    float camDistTex = length(fragWorldPos - tp.cameraPos.xyz);

    vec4 color0 = sampleTriplanar(matLayer0, fragWorldPos, blendW, texScale);
    vec4 color1 = sampleTriplanar(matLayer1, fragWorldPos, blendW, texScale);

    // Medium detail overlay at 4× scale — soil clumps, pebble patterns
    vec4 detail0_med = sampleTriplanar(matLayer0, fragWorldPos, blendW, texScale * 4.0);
    vec4 detail1_med = sampleTriplanar(matLayer1, fragWorldPos, blendW, texScale * 4.0);
    float medDetailFade = 1.0 - smoothstep(60.0, 180.0, camDistTex);
    color0 = mix(color0, color0 * detail0_med * 2.0, 0.22 * medDetailFade);
    color1 = mix(color1, color1 * detail1_med * 2.0, 0.22 * medDetailFade);

    // Fine detail overlay at 12× scale — granular texture (sand grains, moss, dirt particles)
    float fineDetailFade = 1.0 - smoothstep(15.0, 50.0, camDistTex);
    if (fineDetailFade > 0.01) {
        vec4 detail0_fine = sampleTriplanar(matLayer0, fragWorldPos, blendW, texScale * 12.0);
        vec4 detail1_fine = sampleTriplanar(matLayer1, fragWorldPos, blendW, texScale * 12.0);
        color0 = mix(color0, color0 * detail1_fine * 2.0, 0.12 * fineDetailFade);
        color1 = mix(color1, color1 * detail0_fine * 2.0, 0.12 * fineDetailFade);
    }

    vec4 materialColor = mix(color0, color1, matBlend);

    // Depth exposure darkening — exposed subsurface is slightly darker
    // to simulate the visual difference between weathered surface and
    // freshly exposed earth/rock from voxel destruction.
    if (depthExposure > 0.01) {
        float depthDarken = 1.0 - depthExposure * 0.15;
        materialColor.rgb *= depthDarken;
    }

    // ── Slope-driven cliff overlay ───────────────────────────────
    float slope = 1.0 - normal.y;
    float cliffBlend = smoothstep(slopeThreshold, slopeThreshold + slopeBlendRange, slope);
    if (cliffBlend > 0.001) {
        vec4 cliffColor = sampleTriplanar(LAYER_CLIFF, fragWorldPos, blendW, texScale * 0.8);
        materialColor = mix(materialColor, cliffColor, cliffBlend);
    }

    // ── Height-based snow dusting ────────────────────────────────
    float heightFactor = smoothstep(50.0, 70.0, fragWorldPos.y);
    float snowAngle    = smoothstep(0.6, 0.85, normal.y);
    float snowBlend    = heightFactor * snowAngle * 0.45;
    if (snowBlend > 0.01 && biome0 != LAYER_SNOW) {
        vec4 snowColor = sampleTriplanar(LAYER_SNOW, fragWorldPos, blendW, texScale * 1.2);
        materialColor = mix(materialColor, snowColor, snowBlend);
    }

    // ── Material properties (per-biome environment-aware) ──────────
    vec3  lightDir = normalize(tp.lightDir.xyz);
    float ambient  = tp.lightDir.w;
    float nightFade = smoothstep(0.04, 0.005, ambient); // 1.0 = full night

    // Compute per-fragment environment profile from biome blend + time
    EnvProfile env = computeFragmentEnv(biome0, biome1, biomeMix, nightFade);

    // Biome-driven roughness as the base, with overlays for wetness/cliff/snow
    float roughness = env.roughnessBase;
    float metallic  = 0.0;
    // Wet areas (water-adjacent, low elevation) are smoother
    float wetness = smoothstep(0.0, 5.0, -fragWorldPos.y + 0.0);
    roughness = mix(roughness, 0.25, wetness * 0.7);
    // Rock/cliff surfaces
    roughness = mix(roughness, 0.65, cliffBlend * 0.5);
    // Snow overlay
    roughness = mix(roughness, 0.35, snowBlend);

    // Depth exposure modulates roughness — freshly dug rock is rougher
    roughness = mix(roughness, 0.7, depthExposure * 0.4);

    vec3 albedo = materialColor.rgb;

    // ══════════════════════════════════════════════════════════════
    // LIGHTING — Environment-aware "Subtractive Sun Beam" approach
    //   Biome profiles modulate ambient, exposure, shadow depth,
    //   and fog to create unique atmosphere per environment.
    // ══════════════════════════════════════════════════════════════

    // ── Shadow calculation (the "subtraction") ───────────────────
    float shadow = calculateShadow(fragLightSpacePos, normal, lightDir);

    // ── Hemisphere ambient light (biome-tinted) ──────────────────
    // The environment profile provides biome-specific ambient tint and
    // intensity. Forest gets dark green-filtered light, snow biome gets
    // bright cold-blue reflected light, wetland gets murky muted tones.
    vec3 skyAmb = env.ambientTint * ambient * env.ambientScale;
    // Ground bounce is derived from ambient tint, warmer and dimmer
    vec3 gndAmb = env.ambientTint * vec3(0.7, 0.6, 0.5) * ambient * env.ambientScale * 0.4;
    vec3 hemiAmb = mix(gndAmb, skyAmb, normal.y * 0.5 + 0.5);

    // Shadow darkening of ambient (biome-modulated: deeper in forest)
    float shadowAmbientFade = mix(0.50, 1.0, shadow);
    float deepShadowFade    = mix(0.35, 1.0, shadow);
    float shadowDepthBlend  = clamp(env.shadowDepth - 1.0, 0.0, 1.0);
    shadowAmbientFade = mix(shadowAmbientFade, deepShadowFade, shadowDepthBlend);
    hemiAmb *= clamp(shadowAmbientFade, 0.3, 1.0);

    // ── Ambient occlusion (enhanced with biome shadow depth) ─────
    float aoSlope  = mix(0.70, 1.0, normal.y);
    float aoHeight = smoothstep(-25.0, 10.0, fragWorldPos.y) * 0.2 + 0.8;
    float aoDetail = mix(0.85, 1.0, dot(normalize(fragNormal), vec3(0, 1, 0)));
    float ao = aoSlope * aoHeight * aoDetail;
    // Biome shadow depth scales the AO effect — forest has deeper occlusion
    ao = pow(ao, 0.8 + 0.4 * env.shadowDepth);

    // ── Cook-Torrance BRDF ───────────────────────────────────────
    vec3 halfVec = normalize(lightDir + viewDir);
    float NdotL  = max(dot(normal, lightDir), 0.0);
    float NdotV  = max(dot(normal, viewDir),  0.001);
    float NdotH  = max(dot(normal, halfVec),  0.0);
    float HdotV  = max(dot(halfVec, viewDir), 0.0);

    // Night-aware wrap lighting — wrap factor drops to near-zero at night
    // so planet-light barely wraps at all (true darkness requirement).
    float sunHeightFactor = clamp(lightDir.y * 6.0 + 0.5, 0.0, 1.0);
    float wrapFactor = 0.25 * mix(sunHeightFactor, 0.0, nightFade);
    wrapFactor *= mix(1.0, 0.6, clamp(env.shadowDepth - 1.0, 0.0, 1.0));
    float wrappedNdotL = (NdotL + wrapFactor) / (1.0 + wrapFactor);
    wrappedNdotL = max(wrappedNdotL, 0.0);

    // Specular (Cook-Torrance)
    vec3 F0 = mix(vec3(0.04), albedo, metallic);
    float D = distributionGGX(NdotH, roughness);
    float G = geometrySmith(NdotV, NdotL, roughness);
    vec3  F = fresnelSchlick(HdotV, F0);

    vec3 specular = (D * G * F) / max(4.0 * NdotV * NdotL, 0.001);

    // Diffuse (Lambert, energy-conserving with specular)
    vec3 kD = (vec3(1.0) - F) * (1.0 - metallic);
    vec3 diffuse = kD * albedo / PI;

    // ── Sun colour (warm sunlight, subtly shifts with sun angle) ─
    float sunElevation = max(lightDir.y, 0.0);
    vec3 daySunColor = mix(vec3(1.0, 0.75, 0.45),    // Golden hour
                           vec3(1.0, 0.97, 0.92),     // High noon
                           smoothstep(0.0, 0.6, sunElevation));
    vec3 nightPlanetColor = vec3(0.20, 0.25, 0.45);
    vec3 sunColor = mix(daySunColor, nightPlanetColor, nightFade);

    // ── Combine lighting ─────────────────────────────────────────
    // Direct-light intensity — at night direct light is negligible.
    // Planet-reflected light is orders of magnitude weaker than sunlight.
    float directIntensity = mix(1.0 - ambient, max(ambient * 0.5, 0.001), nightFade);
    vec3 directLight = sunColor * directIntensity * (diffuse + specular) * wrappedNdotL * shadow;

    // Final illumination = ambient + direct, modulated by AO
    vec3 litColor = (hemiAmb * albedo + directLight) * ao;

    // ── Distance to camera (used by SSS, fog, and volumetric) ────
    float dist = length(fragWorldPos - tp.cameraPos.xyz);

    // ── Subsurface scattering (environment-aware vegetation translucency)
    // The env profile's sssScale controls how much SSS this biome gets.
    // Grassland and forest have high SSS; snow and rock have almost none.
    if (env.sssScale > 0.01) {
        // Direct backlit translucency (sun behind foliage)
        float sssBack = pow(max(dot(-lightDir, viewDir), 0.0), 3.0) * 0.22;

        // Wrap translucency (light wrapping around leaf edges)
        float sssWrap = pow(max(dot(normal, -lightDir) * 0.5 + 0.5, 0.0), 2.0) * 0.10;

        // Distance fade (SSS most visible at close range)
        float sssDist = 1.0 - smoothstep(30.0, 100.0, dist);

        // Chlorophyll absorption: green-yellow backlit, deeper green wrap
        vec3 sss = vec3(0.40, 0.60, 0.12) * sssBack
                 + vec3(0.25, 0.45, 0.10) * sssWrap;
        // Night-dim SSS: no sun means no subsurface translucency
        float sssNightDim = 1.0 - nightFade * 0.92;
        litColor += sss * env.sssScale * shadow * sssDist * sssNightDim;
    }

    // ── Rim lighting (biome-tinted edge highlight) ───────────────
    float rim = 1.0 - max(dot(normal, viewDir), 0.0);
    rim = pow(rim, 4.0) * 0.08;
    litColor += skyAmb * rim;

    // ── Atmospheric distance fog (biome-density-modulated) ───────
    // The environment profile scales fog density per biome — thick
    // low haze in wetlands, thin clear air in mountains, moderate
    // in forests. Fog color inherits biome tint at close range but
    // converges to the CPU-set horizon color at distance.
    float biomeFogStart = tp.fogParams.x / env.fogDensityMul;
    float biomeFogEnd   = tp.fogParams.y / env.fogDensityMul;

    float fogFactor  = smoothstep(biomeFogStart, biomeFogEnd, dist);

    // Height-based fog (thicker in valleys — fog pools at low elevation)
    // Biome fog density amplifies the valley fog effect
    float heightFog = 1.0 - smoothstep(-15.0, 25.0, fragWorldPos.y);
    fogFactor       = max(fogFactor, heightFog * 0.25 * env.fogDensityMul);

    // ── Horizon dissolution (fog-of-war boundary) ────────────────
    // Uses the original (unmodified) fogEnd for the fog-of-war boundary
    // since that represents the actual loaded terrain extent.
    // Factor 1.5 instead of 3.0 — preserves texture detail at medium
    // distances while still fading cleanly at the boundary.
    float distNorm = dist / max(tp.fogParams.y, 1.0);
    float expFog   = 1.0 - exp(-distNorm * distNorm * 1.5);
    fogFactor      = max(fogFactor, expFog);

    // Hard guarantee at the fog-of-war boundary
    float hardEdge = smoothstep(tp.fogParams.y * 0.85, tp.fogParams.y, dist);
    fogFactor = max(fogFactor, hardEdge);

    // Mie scattering — fog is brighter when looking toward the sun
    // Completely killed at night — no sun means no Mie scattering
    float sunViewDot = max(dot(normalize(fragWorldPos - tp.cameraPos.xyz), lightDir), 0.0);
    float mieScatter = pow(sunViewDot, 8.0) * 0.3 * (1.0 - nightFade);

    // Biome-tinted near-fog: close fog inherits a subtle tint from the
    // environment (green haze in forest, murky in wetland) that fades
    // to the neutral horizon fog color at distance.
    float nearFogBlend = 1.0 - smoothstep(biomeFogStart, biomeFogEnd * 0.5, dist);
    vec3 biomeFogTint  = mix(tp.fogColor.rgb, tp.fogColor.rgb * (env.ambientTint * 1.5 + 0.5), nearFogBlend * 0.25);
    vec3 fogWithMie    = mix(biomeFogTint, sunColor * 1.2, mieScatter);

    fogFactor = clamp(fogFactor, 0.0, 1.0);
    litColor  = mix(litColor, fogWithMie, fogFactor);

    // ══════════════════════════════════════════════════════════════
    // VOLUMETRIC LIGHT — Camera-ray in-scatter (shadow-segmented)
    //   Rays cast FROM camera THROUGH scene, divided by shadow map.
    //   Beams are world-anchored: camera rotation reveals different
    //   beams but never slides them. Sun projects like a torch.
    // ══════════════════════════════════════════════════════════════

    float godRayDensity = tp.fogParams.z;
    vec3 volumetric = calculateSunInScatter(
        fragWorldPos, tp.cameraPos.xyz, lightDir, godRayDensity);

    // --- Point light scatter framework ---
    // (Uncomment when point light system is wired)

    // Volumetric: completely off at night — no sun means no volumetric shafts.
    // During day, biome fog density modulates scatter visibility.
    float scatterNightDim = 1.0 - nightFade;
    float scatterBiomeMod = clamp(env.fogDensityMul * 0.8, 0.5, 1.5);
    float volFade = 1.0 - smoothstep(tp.fogParams.y * 0.35, tp.fogParams.y * 0.75, dist);
    litColor += volumetric * volFade * scatterNightDim * scatterBiomeMod;

    // ── Curvature micro-AO (screen-space derivative darkening) ───
    {
        float curvature = length(fwidth(normal));
        litColor *= 1.0 - clamp(curvature * 5.0, 0.0, 0.25);
    }

    // ── Biome-aware exposure compensation ────────────────────────
    // Apply per-biome exposure bias before tone mapping. This allows
    // bright open biomes (plains, snow, beach) to feel dazzling and
    // dark biomes (forest, cave, deep water) to feel properly dim
    // and atmospheric, rather than everything being the same brightness.
    {
        float ev = env.exposureBias;
        float exposureMul = pow(2.0, ev);
        litColor *= exposureMul;
    }

    // ── Output linear HDR ─────────────────────────────────────────
    // Tone mapping and gamma are applied once in the composite pass
    // (rt_composite.frag) to avoid double tone mapping when rendering
    // to the offscreen HDR (R16G16B16A16_SFLOAT) target.
    outColor = vec4(litColor, 1.0);
}
