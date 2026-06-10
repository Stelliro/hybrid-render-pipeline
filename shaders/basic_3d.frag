/// @file basic_3d.frag
/// @brief 3D fragment shader — environment-aware directional lighting with
///        per-vertex color, flat-shaded normals from screen-space derivatives,
///        hemisphere ambient, shadow receiving, volumetric scatter, and biome-tinted fog.
///
/// Companion to basic_3d.vert. Uses dFdx/dFdy on world position to compute
/// flat-shaded face normals without requiring per-vertex normal data.
///
/// Shadow receiving:
///   - Samples the terrain shadow map via a comparison sampler at set 0 binding 0.
///   - Receives fragLightSpacePos from the vertex shader for shadow UV lookup.
///   - Uses rotated Poisson-disk PCF for soft shadow edges.
///
/// Environment awareness:
///   - envParams push constant carries per-biome exposure, ambient, fog, and
///     roughness parameters blended on the CPU based on the player's current
///     biome and the day/night cycle. This gives objects and vegetation unique
///     atmosphere in each biome without per-vertex biome data.
///
/// Volumetric scattering is approximated without shadow segmentation:
///   - Henyey-Greenstein phase function for directional sun scatter
///   - Height-based aerial density for valley haze
///   - Consistent with the terrain shader's camera-ray approach visually

#version 450

layout(location = 0) in vec3 fragColor;
layout(location = 1) in vec3 fragWorldPos;
layout(location = 2) in vec4 fragLightSpacePos;

layout(location = 0) out vec4 outColor;

// ── Shadow map (depth comparison sampler, same layout as terrain set 1) ──
layout(set = 0, binding = 0) uniform sampler2DShadow shadowMap;

// ── Push constants (vertex stage uses bytes 0–127 for MVP + lightViewProj) ──
// Fragment stage uses bytes 128–207 for lighting + environment parameters
layout(push_constant) uniform FragParams {
    layout(offset = 128) vec4 lightDir;    // xyz = sun direction, w = ambient strength
    vec4 cameraPos;                        // xyz = camera position, w = shadow bias
    vec4 fogParams;                        // x = fogStart, y = fogEnd, z = unused, w = shadowMapSize
    vec4 fogColor;                         // rgb = fog color, a = shadow strength
    vec4 envParams;                        // x = exposureBias, y = ambientScale,
                                           // z = fogDensityMul, w = roughnessBase
} fp;

const float PI = 3.14159265359;

// ── Shadow calculation (PCF with rotated Poisson disk) ───────────────
/// Percentage-Closer Filtering for soft shadow edges.
/// Returns 0.0 = full shadow, 1.0 = full light.
float calculateShadow(vec4 lightSpacePos, vec3 normal, vec3 lightDir)
{
    vec3 projCoords = lightSpacePos.xyz / lightSpacePos.w;

    // NDC xy [-1,1] → UV [0,1] before bounds check
    vec2 shadowUV = projCoords.xy * 0.5 + 0.5;
    float currentDepth = projCoords.z;

    // Outside shadow map → fully lit
    if (shadowUV.x < 0.0 || shadowUV.x > 1.0 ||
        shadowUV.y < 0.0 || shadowUV.y > 1.0 ||
        currentDepth > 1.0 || currentDepth < 0.0)
        return 1.0;

    // Slope-scaled bias
    float bias = fp.cameraPos.w;
    float slopeBias = max(bias * (1.0 - dot(normal, lightDir)), bias * 0.5);
    currentDepth -= slopeBias;

    // 5-sample rotated Poisson disk (lighter than terrain's 13-sample)
    float shadowMapSize = fp.fogParams.w;
    float texelSize = 1.0 / max(shadowMapSize, 1.0);

    const vec2 poissonDisk[5] = vec2[5](
        vec2( 0.0,       0.0),
        vec2(-0.9406,   -0.2836),
        vec2( 0.9452,   -0.7684),
        vec2( 0.3432,    0.8458),
        vec2(-0.6714,    0.6393)
    );

    float angle = fract(sin(dot(fragWorldPos.xz, vec2(12.9898, 78.233))) * 43758.5453) * PI * 2.0;
    float cosA = cos(angle);
    float sinA = sin(angle);

    float shadow = 0.0;
    float pcfRadius = 1.5;

    for (int i = 0; i < 5; ++i) {
        vec2 offset = poissonDisk[i] * texelSize * pcfRadius;
        vec2 rotated = vec2(
            offset.x * cosA - offset.y * sinA,
            offset.x * sinA + offset.y * cosA
        );
        shadow += texture(shadowMap, vec3(shadowUV + rotated, currentDepth));
    }
    shadow /= 5.0;

    // Edge fade
    float fadeRange = 0.05;
    float fadeFactor = 1.0;
    fadeFactor *= smoothstep(0.0, fadeRange, shadowUV.x);
    fadeFactor *= smoothstep(0.0, fadeRange, 1.0 - shadowUV.x);
    fadeFactor *= smoothstep(0.0, fadeRange, shadowUV.y);
    fadeFactor *= smoothstep(0.0, fadeRange, 1.0 - shadowUV.y);
    shadow = mix(1.0, shadow, fadeFactor);

    float strength = fp.fogColor.a;
    return mix(1.0, shadow, strength);
}

// ── Approximate Henyey-Greenstein phase function ─────────────────────
float henyeyGreenstein(float cosTheta, float g)
{
    float g2 = g * g;
    float denom = 1.0 + g2 - 2.0 * g * cosTheta;
    return (1.0 - g2) / (4.0 * PI * denom * sqrt(max(denom, 0.0001)));
}

/// Approximate sun in-scatter without shadow map.
/// Uses Henyey-Greenstein phase function and distance-based integration
/// to match the terrain shader's volumetric look without shadow segmentation.
/// The approximation assumes partial shadowing from the density of the atmosphere.
vec3 approximateSunScatter(vec3 fragPos, vec3 camPos, vec3 lightDir)
{
    const float SUN_ANISO   = 0.72;    // Forward-scatter (tight when looking sunward)
    const float DENSITY     = 0.003;   // Lower than terrain (no shadow segmentation)
    const float HEIGHT_FADE = 0.01;    // Thicker in valleys

    vec3  rayVec   = fragPos - camPos;
    float rayLen   = length(rayVec);
    if (rayLen < 0.1) return vec3(0.0);

    vec3  rayDir   = rayVec / rayLen;
    float marchDist = min(rayLen, 80.0);

    // Phase function
    float cosTheta = dot(rayDir, lightDir);
    float phase    = henyeyGreenstein(cosTheta, SUN_ANISO);

    // Integrated density along the ray (analytical approximation)
    // Average height along the ray for height-dependent density
    float avgHeight = (camPos.y + fragPos.y) * 0.5;
    float heightDensity = exp(-max(avgHeight, 0.0) * HEIGHT_FADE);
    float scatter = marchDist * DENSITY * (0.4 + 0.6 * heightDensity) * phase;

    // Sun beam color (matches terrain shader)
    float sunElev = max(lightDir.y, 0.0);
    vec3 beamColor = mix(
        vec3(1.0, 0.82, 0.50),
        vec3(1.0, 0.96, 0.90),
        smoothstep(0.0, 0.6, sunElev)
    );

    return beamColor * clamp(scatter, 0.0, 0.5);
}

void main()
{
    // ── Derive flat-shaded normal from screen-space derivatives ───
    vec3 dpdx = dFdx(fragWorldPos);
    vec3 dpdy = dFdy(fragWorldPos);
    vec3 normal = normalize(cross(dpdx, dpdy));

    // Ensure normal faces toward camera
    vec3 viewDir = normalize(fp.cameraPos.xyz - fragWorldPos);
    if (dot(normal, viewDir) < 0.0) normal = -normal;

    // ── Missing-texture checkerboard fallback ────────────────────
    // If vertex color is near-black (uninitialized or missing data),
    // replace with a magenta/black checkerboard debug pattern so
    // missing textures are immediately obvious.
    vec3 baseColor = fragColor;
    float lum = dot(baseColor, vec3(0.2126, 0.7152, 0.0722));
    if (lum < 0.005) {
        // World-space 1-unit checkerboard
        float checker = step(0.0,
            sin(fragWorldPos.x * PI) * sin(fragWorldPos.y * PI) * sin(fragWorldPos.z * PI));
        baseColor = mix(vec3(0.0), vec3(1.0, 0.0, 1.0), checker);
    }

    // ── Unpack environment parameters ────────────────────────────
    float envExposureBias  = fp.envParams.x;   // per-biome EV shift
    float envAmbientScale  = fp.envParams.y;   // ambient brightness multiplier
    float envFogDensityMul = fp.envParams.z;   // fog density multiplier
    float envRoughnessBase = fp.envParams.w;   // biome-appropriate roughness

    // ── Lighting ─────────────────────────────────────────────────
    vec3  lightDir = normalize(fp.lightDir.xyz);
    float ambient  = fp.lightDir.w;
    float nightFade = smoothstep(0.04, 0.005, ambient);

    // Wrap lighting — wrap factor shrinks toward zero at night so the
    // planet-light direction only contributes a whisper of direct wrap.
    // lightDir is blended by CPU from sun → planet direction at night.
    float NdotL   = dot(normal, lightDir);
    float sunHeightFactor = clamp(lightDir.y * 6.0 + 0.5, 0.0, 1.0);
    float wrapAmount = 0.3 * mix(sunHeightFactor, 0.0, nightFade);
    float wrapLit = (NdotL + wrapAmount) / (1.0 + wrapAmount);
    wrapLit       = clamp(wrapLit, 0.0, 1.0);

    // Hemisphere ambient — biome-modulated brightness via envAmbientScale.
    // Shifts to cool blue planetlight at night (low ambient).
    vec3 skyAmb  = mix(vec3(0.50, 0.58, 0.70), vec3(0.06, 0.08, 0.18), nightFade)
                 * ambient * envAmbientScale;
    vec3 gndAmb  = mix(vec3(0.30, 0.25, 0.18), vec3(0.02, 0.02, 0.04), nightFade)
                 * ambient * envAmbientScale * 0.5;
    vec3 hemiAmb = mix(gndAmb, skyAmb, normal.y * 0.5 + 0.5);

    // Direct-light intensity — at night direct light is negligible.
    // Planet-reflected light is orders of magnitude weaker than sunlight.
    float directIntensity = mix(1.0 - ambient, max(ambient * 0.5, 0.001), nightFade);

    // Sun colour (warm by day, cool blue planetlight at night)
    float sunElevation = max(lightDir.y, 0.0);
    vec3 daySunColor = mix(vec3(1.0, 0.75, 0.45),
                           vec3(1.0, 0.97, 0.92),
                           smoothstep(0.0, 0.6, sunElevation));
    vec3 nightPlanetColor = vec3(0.15, 0.18, 0.35);
    vec3 sunColor = mix(daySunColor, nightPlanetColor, nightFade);

    // Cook-Torrance GGX specular (per-biome roughness base)
    vec3  halfVec = normalize(lightDir + viewDir);
    float NdotH   = max(dot(normal, halfVec), 0.0);
    float NdotV   = max(dot(normal, viewDir), 0.001);
    float HdotV   = max(dot(halfVec, viewDir), 0.0);
    // True geometric NdotL for microfacet BRDF — wrap lighting is an
    // artistic diffuse concept and must not enter the geometry term or
    // normalization denominator.
    float NdotL_geom = max(dot(normal, lightDir), 0.0);

    float roughness = envRoughnessBase;
    float a   = roughness * roughness;
    float a2  = a * a;
    float den = NdotH * NdotH * (a2 - 1.0) + 1.0;
    float D   = a2 / (PI * den * den);

    // Fresnel-Schlick
    vec3 F0 = vec3(0.04);
    vec3 F  = F0 + (1.0 - F0) * pow(clamp(1.0 - HdotV, 0.0, 1.0), 5.0);

    // Smith geometry (uses true NdotL, not wrap-lit)
    float k = (roughness + 1.0) * (roughness + 1.0) / 8.0;
    float Gv = NdotV     / (NdotV     * (1.0 - k) + k);
    float Gl = NdotL_geom / (NdotL_geom * (1.0 - k) + k);

    vec3 specular = (D * Gv * Gl * F)
                  / max(4.0 * NdotV * NdotL_geom, 0.001);

    // Combined lighting (energy-conserving) with shadow.
    // Diffuse uses wrap-lit for artistic softening; specular uses
    // physical NdotL to preserve highlight placement.
    float shadow = calculateShadow(fragLightSpacePos, normal, lightDir);
    vec3 kD      = vec3(1.0) - F;
    vec3 diffuse = baseColor * kD / PI;
    vec3 litColor = baseColor * kD * hemiAmb
                  + diffuse * sunColor * directIntensity * wrapLit * shadow
                  + specular * sunColor * directIntensity * NdotL_geom * shadow;

    // Rim lighting (sky-coloured edge highlight)
    float rim = pow(1.0 - max(dot(normal, viewDir), 0.0), 4.0) * 0.10;
    litColor += skyAmb * rim;

    // ── Atmospheric fog (biome fog density modulated) ──────────────
    // The envFogDensityMul scales the fog range per biome — thick haze
    // in wetlands, thin clear air in highlands, moderate in forests.
    float biomeFogStart = fp.fogParams.x / envFogDensityMul;
    float biomeFogEnd   = fp.fogParams.y / envFogDensityMul;

    float dist      = length(fragWorldPos - fp.cameraPos.xyz);
    float fogFactor = smoothstep(biomeFogStart, biomeFogEnd, dist);

    // Height-based fog (thicker in valleys, biome-amplified)
    float heightFog = 1.0 - smoothstep(-15.0, 25.0, fragWorldPos.y);
    fogFactor       = max(fogFactor, heightFog * 0.25 * envFogDensityMul);

    // Exponential horizon dissolution — objects gently fade into the
    // skybox at the fog-of-war boundary, matching the terrain shader.
    // Factor 1.5 preserves texture detail at medium distances.
    float distNorm = dist / max(fp.fogParams.y, 1.0);
    float expFog   = 1.0 - exp(-distNorm * distNorm * 1.5);
    fogFactor      = max(fogFactor, expFog);

    // Hard edge guarantee at the fog boundary
    float hardEdge = smoothstep(fp.fogParams.y * 0.85, fp.fogParams.y, dist);
    fogFactor      = max(fogFactor, hardEdge);

    // Mie scattering — fog is brighter looking toward the sun
    // Completely off at night — no solar disc means no forward scatter
    float sunViewDot = max(dot(normalize(fragWorldPos - fp.cameraPos.xyz), lightDir), 0.0);
    float mieScatter = pow(sunViewDot, 8.0) * 0.3 * (1.0 - nightFade);
    vec3 fogWithMie  = mix(fp.fogColor.rgb, sunColor * 1.2, mieScatter);

    fogFactor       = clamp(fogFactor, 0.0, 1.0);
    litColor        = mix(litColor, fogWithMie, fogFactor);

    // ── Approximate volumetric sun scatter ────────────────────────
    // Completely off at night — no sun means no volumetric scatter.
    float scatterNightDim = 1.0 - nightFade;
    float scatterBiomeMod = clamp(envFogDensityMul * 0.8, 0.5, 1.5);
    vec3 volumetric = approximateSunScatter(
        fragWorldPos, fp.cameraPos.xyz, lightDir) * scatterNightDim * scatterBiomeMod;
    float volFade = 1.0 - smoothstep(fp.fogParams.x * 0.8, fp.fogParams.y * 0.6, dist);
    litColor += volumetric * volFade;

    // ── Curvature micro-AO (screen-space derivative darkening) ───
    {
        float curvature = length(fwidth(normal));
        litColor *= 1.0 - clamp(curvature * 5.0, 0.0, 0.25);
    }

    // ── Biome-aware exposure compensation ────────────────────────
    // Per-biome exposure bias from envParams — brighter in open snow/beach,
    // darker in forest canopy, giving each environment unique atmosphere.
    {
        float exposureMul = pow(2.0, envExposureBias);
        litColor *= exposureMul;
    }

    // ── Output linear HDR ─────────────────────────────────────────
    // Tone mapping and gamma are applied once in the composite pass
    // (rt_composite.frag) to avoid double tone mapping when rendering
    // to the offscreen HDR (R16G16B16A16_SFLOAT) target.
    outColor = vec4(litColor, 1.0);
}
