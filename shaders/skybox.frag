/// @file skybox.frag
/// @brief Procedural alien-planet skybox — cinematic-quality LIVING sky.
///
/// Premium procedural sky system with zero textures, organised into 11 layers:
///
///   LAYER 1 — ATMOSPHERE
///     Multi-scatter atmospheric model with rich sunrise/sunset colour bands,
///     Belt of Venus, extended twilight, and zodiacal light
///
///   LAYER 2 — SUN DISK + SOLAR PHENOMENA
///     Limb-darkened sun with 6-point diffraction spikes, 4-layer corona,
///     sun dogs (parhelia), light pillars, crepuscular rays
///
///   LAYER 3 — NEARBY PLANET
///     Massive procedural planet (replaces moon) — oceans, continents,
///     biomes, ice caps, cloud layer, city lights, atmospheric limb;
///     horizon-hugging orbit for navigational phase tracking
///
///   LAYER 4 — STARS + VARIABLE STARS
///     6-layer star field (giant→ultra-faint) with spectral colours,
///     atmospheric scintillation, seasonal rotation for celestial
///     navigation; 5 prominent pulsating "heartbeat" variable stars
///
///   LAYER 5 — DEEP SPACE
///     Volumetric galaxy band with spiral arm structure, dust lanes,
///     HII emission regions; 6 distinct nebula regions
///
///   LAYER 6 — PLANETARY RING
///     3 bands (A/B/C) + Cassini/Encke gaps + fine ringlets,
///     forward-scattering, planet shadow, compositional colour
///
///   LAYER 7 — TRANSIENT EVENTS
///     Shooting stars with 16-point trails and afterglow;
///     Spectacular meteor showers with fireballs;
///     Comets with dual tails (ion blue + dust yellow), once per year;
///     Orbiting satellites / Titan debris with Iridium-style flares;
///     Alien craft sightings (3 types: arc, formation, erratic);
///     Cosmic pulse events (distant explosion flash + expanding ring)
///
///   LAYER 8 — ATMOSPHERIC PHENOMENA
///     Rare aurora curtains (green→purple→red);
///     Noctilucent clouds, iridescent clouds;
///     Mesospheric airglow emission bands
///
///   LAYER 9 — DIM PHENOMENA
///     Zodiacal light, gegenschein
///
///   LAYER 10 — STORM EFFECTS
///     Procedural lightning (double-flash, directional)
///
///   LAYER 11 — POST-PROCESSING
///     Horizon fog, ACES filmic tone mapping, true darkness at midnight

#version 450

layout(location = 0) in vec2 inUV;
layout(location = 1) in vec3 inViewRay;

layout(location = 0) out vec4 outColor;

layout(push_constant) uniform PushConstants {
    mat4  invViewProj;
    vec4  sunDir;        // xyz = toward-sun direction, w = intensity
    vec4  cameraPos;     // xyz = world pos, w = time (total elapsed seconds)
    vec4  skyParams;     // x = dayProgress [0,1], y = cloudCover [0,1], z = starIntensity, w = ringRotation
    vec4  skyParams2;    // x = meteorTimer, y = meteorActive (0 or 1), z = floor=seed / fract=yearProgress, w = planetOrbitalPhase [0,1]
    vec4  skyParams3;    // x = alienCraftTimer, y = alienCraftType (0=off, 1=arc, 2=formation, 3=erratic), z = alienCraftSeed, w = windDirAngle
    vec4  skyParams4;    // x = cloudType [0..5], y = cloudDensity, z = windStrength, w = cloudBaseAlt (meters)
} pc;

// ═══════════════════════════════════════════════════════════════════════
// Constants
// ═══════════════════════════════════════════════════════════════════════

const float PI      = 3.14159265358979;
const float TAU     = 6.28318530717959;
const float INV_PI  = 0.31830988618379;
const float PHI     = 1.61803398874989;  // Golden ratio — decorrelation constant

// Atmospheric scattering
const float k_MieAnisotropy = 0.76;

// Ring geometry
const vec3  k_RingNormal = normalize(vec3(0.15, 0.92, 0.36));

// ═══════════════════════════════════════════════════════════════════════
// Hash Functions — high-quality procedural randomness
// ═══════════════════════════════════════════════════════════════════════

float hash11(float p)
{
    p = fract(p * 0.1031);
    p *= p + 33.33;
    p *= p + p;
    return fract(p);
}

float hash21(vec2 p)
{
    vec3 p3 = fract(vec3(p.xyx) * 0.1031);
    p3 += dot(p3, p3.yzx + 33.33);
    return fract((p3.x + p3.y) * p3.z);
}

vec2 hash22(vec2 p)
{
    vec3 p3 = fract(vec3(p.xyx) * vec3(0.1031, 0.1030, 0.0973));
    p3 += dot(p3, p3.yzx + 33.33);
    return fract((p3.xx + p3.yz) * p3.zy);
}

vec3 hash32(vec2 p)
{
    vec3 p3 = fract(vec3(p.xyx) * vec3(0.1031, 0.1030, 0.0973));
    p3 += dot(p3, p3.yxz + 33.33);
    return fract((p3.xxy + p3.yzz) * p3.zyx);
}

float hash31(vec3 p)
{
    p = fract(p * 0.1031);
    p += dot(p, p.zyx + 31.32);
    return fract((p.x + p.y) * p.z);
}

// ═══════════════════════════════════════════════════════════════════════
// Noise Functions — quintic-interpolated for smooth gradients
// ═══════════════════════════════════════════════════════════════════════

float noise2D(vec2 p)
{
    vec2 i = floor(p);
    vec2 f = fract(p);
    // Quintic interpolation: C2-continuous, avoids grid artifacts
    f = f * f * f * (f * (f * 6.0 - 15.0) + 10.0);

    float a = hash21(i);
    float b = hash21(i + vec2(1.0, 0.0));
    float c = hash21(i + vec2(0.0, 1.0));
    float d = hash21(i + vec2(1.0, 1.0));

    return mix(mix(a, b, f.x), mix(c, d, f.x), f.y);
}

float noise3D(vec3 p)
{
    vec3 i = floor(p);
    vec3 f = fract(p);
    f = f * f * f * (f * (f * 6.0 - 15.0) + 10.0);

    float n = dot(i, vec3(1.0, 57.0, 113.0));
    return mix(
        mix(mix(hash11(n +   0.0), hash11(n +   1.0), f.x),
            mix(hash11(n +  57.0), hash11(n +  58.0), f.x), f.y),
        mix(mix(hash11(n + 113.0), hash11(n + 114.0), f.x),
            mix(hash11(n + 170.0), hash11(n + 171.0), f.x), f.y),
        f.z);
}

// FBM with inter-octave rotation to break grid alignment
float fbm(vec2 p, int octaves)
{
    float value     = 0.0;
    float amplitude = 0.5;
    float frequency = 1.0;
    // Rotation matrix between octaves — decorrelates axis-aligned patterns
    const mat2 rot = mat2(0.8, 0.6, -0.6, 0.8);
    for (int i = 0; i < octaves; i++) {
        value     += amplitude * noise2D(p * frequency);
        amplitude *= 0.5;
        p          = rot * p;
        frequency *= 2.0;
    }
    return value;
}

float fbm3D(vec3 p, int octaves)
{
    float value     = 0.0;
    float amplitude = 0.5;
    float frequency = 1.0;
    for (int i = 0; i < octaves; i++) {
        value     += amplitude * noise3D(p * frequency);
        amplitude *= 0.5;
        frequency *= 2.0;
    }
    return value;
}

// Domain-warped FBM — warps input coordinates for organic shapes
float warpedFBM(vec2 p, int octaves) {
    vec2 q = vec2(fbm(p + vec2(0.0, 0.0), 4),
                  fbm(p + vec2(5.2, 1.3), 4));
    return fbm(p + 4.0 * q, octaves);
}

// ═══════════════════════════════════════════════════════════════════════
// Atmospheric Scattering — cinematic day/night transitions
// ═══════════════════════════════════════════════════════════════════════

// Henyey-Greenstein phase function
float phaseHG(float cosTheta, float g)
{
    float g2 = g * g;
    return (1.0 - g2) / (4.0 * PI * pow(1.0 + g2 - 2.0 * g * cosTheta, 1.5));
}

// Rayleigh phase function
float phaseRayleigh(float cosTheta)
{
    return 3.0 / (16.0 * PI) * (1.0 + cosTheta * cosTheta);
}

vec3 atmosphericScattering(vec3 dir, vec3 sunDir, float dayProgress)
{
    float sunElev   = sunDir.y;
    float cosTheta  = dot(dir, sunDir);

    // Phase functions
    float rayleigh  = phaseRayleigh(cosTheta);
    float mie       = phaseHG(cosTheta, k_MieAnisotropy);

    // Sun-driven intensity with multi-scatter approximation
    float sunIntensity = max(sunElev, 0.0) * 22.0;
    // Longer atmospheric path near horizon enriches scattering
    float pathMulti = 1.0 + 0.8 * pow(1.0 - max(sunElev, 0.0), 4.0);
    sunIntensity *= pathMulti;

    // ── Zenith colour ────────────────────────────────────────────────
    // Daytime: alien deep-blue zenith with slight purple due to atmosphere composition
    vec3 zenithDeep = vec3(0.06, 0.015, 0.20);    // Low sun: purple-blue
    vec3 zenithHigh = vec3(0.10, 0.20, 0.60);     // High sun: rich blue
    vec3 zenithDay  = mix(zenithDeep, zenithHigh, smoothstep(0.0, 0.55, sunElev));

    // Ozone-like layer: absorbs warm tones at zenith → deeper blue
    float ozone = smoothstep(0.3, 1.0, max(dir.y, 0.0)) * 0.12;
    zenithDay  -= vec3(ozone * 0.4, ozone * 0.08, 0.0);

    // Night zenith: near-black, modulated by planet illumination
    float planetPhase = pc.skyParams2.w;
    // Planet illumination derived from orbital geometry (computed in CPU)
    // Used here only for ambient sky brightness contribution
    float planetOrbAngle = planetPhase * TAU;
    float planetElev = 0.10 * sin(planetOrbAngle);
    float planetVisibility = clamp((planetElev + 0.18) / 0.36, 0.0, 1.0);
    // Approximate planet illumination from sun-planet alignment
    vec3 approxPlanetDir = normalize(vec3(cos(planetOrbAngle), planetElev, sin(planetOrbAngle)));
    float planetSunDot = dot(approxPlanetDir, sunDir);
    float planetIllum = (planetSunDot + 1.0) * 0.5 * planetVisibility;
    vec3 nightAmbient = vec3(0.001, 0.001, 0.003) + vec3(0.006, 0.008, 0.018) * planetIllum;

    float nightFactor = smoothstep(0.05, -0.12, sunElev);
    vec3  zenith      = mix(zenithDay, nightAmbient, nightFactor);

    // Scattering contribution
    vec3 rayleighScatter = vec3(5.8e-6, 13.5e-6, 33.1e-6) * rayleigh;
    vec3 mieScatter      = vec3(21e-6) * mie;
    vec3 sky             = zenith + (rayleighScatter + mieScatter) * sunIntensity;

    // ── Horizon glow ─────────────────────────────────────────────────
    float horizPow = pow(1.0 - max(dir.y, 0.0), 5.0);

    // Proximity to the sun azimuthally (for dawn/dusk directional glow)
    float sunAzimuthDot = dot(normalize(vec3(dir.x, 0.0, dir.z)),
                              normalize(vec3(sunDir.x, 0.0, sunDir.z)));
    float nearSun = smoothstep(-0.3, 1.0, sunAzimuthDot);

    // Dawn/dusk colours — near-sun vs. far-from-sun
    vec3 dawnNear   = vec3(1.0, 0.38, 0.12);   // Deep orange at the sun
    vec3 dawnMid    = vec3(1.0, 0.62, 0.22);   // Gold spreading outward
    vec3 dawnFar    = vec3(0.85, 0.40, 0.55);  // Pink/magenta — Belt of Venus
    vec3 duskGlow   = mix(dawnFar, mix(dawnMid, dawnNear, nearSun), nearSun);

    float duskFactor = smoothstep(0.20, 0.0, sunElev) * smoothstep(-0.18, 0.0, sunElev);

    // Day horizon: hazy blue-white
    vec3  dayHorizon  = vec3(0.55, 0.60, 0.72);
    float dayFactor   = smoothstep(0.0, 0.30, sunElev);

    vec3 horizonColor = mix(duskGlow, dayHorizon, dayFactor);
    horizonColor      = mix(nightAmbient * 2.0, horizonColor, smoothstep(-0.12, 0.05, sunElev));
    sky = mix(sky, horizonColor, horizPow);

    // ── Extended twilight glow ───────────────────────────────────────
    // After sunset, a warm glow lingers near where the sun went down
    float twilight = smoothstep(-0.22, -0.02, sunElev) * (1.0 - smoothstep(-0.02, 0.05, sunElev));
    float sunProx  = pow(max(sunAzimuthDot, 0.0), 3.0) * (1.0 - max(dir.y, 0.0));
    vec3  twilightCol = vec3(0.30, 0.12, 0.04) * twilight * sunProx;
    sky += twilightCol;

    // ── Belt of Venus ────────────────────────────────────────────────
    // Pinkish band opposite the sun during twilight
    float antiSunDot = max(-sunAzimuthDot, 0.0);
    float belt       = smoothstep(0.0, 0.6, antiSunDot)
                     * smoothstep(0.0, 0.2, dir.y) * (1.0 - smoothstep(0.2, 0.5, dir.y));
    vec3  beltColor  = vec3(0.35, 0.18, 0.22) * belt * duskFactor;
    sky += beltColor;

    return max(sky, vec3(0.0));
}

// ═══════════════════════════════════════════════════════════════════════
// Sun Disk — limb-darkened with corona, bloom, and diffraction hint
// ═══════════════════════════════════════════════════════════════════════

vec3 computeSunColor(float dayProgress)
{
    if (dayProgress >= 0.25 && dayProgress < 0.35) {
        float t = (dayProgress - 0.25) / 0.10;
        return mix(vec3(1.0, 0.55, 0.15), vec3(1.0, 0.95, 0.88), t);
    } else if (dayProgress >= 0.35 && dayProgress < 0.65) {
        return vec3(1.0, 0.98, 0.95);
    } else if (dayProgress >= 0.65 && dayProgress < 0.80) {
        float t = (dayProgress - 0.65) / 0.15;
        return mix(vec3(1.0, 0.95, 0.88), vec3(1.0, 0.35, 0.08), t);
    }
    return vec3(0.15, 0.18, 0.30);
}

vec3 sunDisk(vec3 dir, vec3 sunDir)
{
    float cosAngle   = dot(dir, sunDir);
    float dayProgress = pc.skyParams.x;

    // Hard disk with anti-aliased edge
    float sunRadius    = 0.9997;
    float edgeSoftness = 0.0004;
    float sun = smoothstep(sunRadius - edgeSoftness, sunRadius, cosAngle);

    // Limb darkening — empirical 3-term
    float limbMu = clamp((cosAngle - sunRadius) / (1.0 - sunRadius), 0.0, 1.0);
    float limb   = 0.28 + 0.52 * limbMu + 0.20 * limbMu * limbMu;

    vec3 baseColor = computeSunColor(dayProgress);
    vec3 discColor = baseColor * 55.0 * sun * limb;

    // Multi-layer corona
    float innerCorona = pow(max(cosAngle, 0.0), 600.0) * 14.0;
    float midCorona   = pow(max(cosAngle, 0.0), 140.0) * 4.0;
    float outerBloom  = pow(max(cosAngle, 0.0), 25.0)  * 0.30;
    float wideHaze    = pow(max(cosAngle, 0.0), 8.0)   * 0.05;

    vec3 coronaTint = mix(vec3(1.0, 0.60, 0.25), vec3(1.0, 0.90, 0.72),
                          smoothstep(0.0, 0.4, pc.sunDir.y));
    vec3 coronaColor = coronaTint * (innerCorona + midCorona + outerBloom + wideHaze);

    // Subtle 6-point diffraction spikes (only on brightest pixels near disk)
    float spike = 0.0;
    if (cosAngle > 0.998) {
        vec2 sunScreen = normalize(dir.xz - sunDir.xz);
        float angle = atan(sunScreen.y, sunScreen.x);
        spike = pow(abs(cos(angle * 3.0)), 64.0) * 2.0;
        spike *= smoothstep(0.998, 0.9995, cosAngle);
    }
    coronaColor += coronaTint * spike;

    float intensity = max(pc.sunDir.w, 0.0);
    return (discColor + coronaColor) * intensity;
}

// ═══════════════════════════════════════════════════════════════════════
// Star Field — 6 layers, spectral colours, atmospheric scintillation
// ═══════════════════════════════════════════════════════════════════════

float starLayer(vec3 dir, float scale, float threshold, float seed)
{
    vec2 uv = vec2(atan(dir.z, dir.x) / TAU + 0.5,
                   asin(clamp(dir.y, -1.0, 1.0)) / PI + 0.5);
    uv *= scale;

    vec2  cell    = floor(uv);
    vec2  localUV = fract(uv);
    float brightness = 0.0;

    for (int x = -1; x <= 1; x++) {
        for (int y = -1; y <= 1; y++) {
            vec2 neighbor = cell + vec2(float(x), float(y));
            vec2 starPos  = hash22(neighbor + seed);
            vec2 diff     = localUV - (starPos + vec2(float(x), float(y)));
            float dist    = length(diff);

            float starHash = hash21(neighbor * 127.1 + seed);
            if (starHash > threshold) {
                // Star angular size. Generous so the bright layers (low scale =
                // big cells) read as clearly visible points rather than the
                // sub-pixel specks that were effectively invisible before. The
                // faint/dust layers use high `scale` (tiny cells), so the same
                // value still yields small stars there — the look stays layered.
                float starSize = 0.045 + 0.090 * hash21(neighbor * 31.7 + seed);
                float star = 1.0 - smoothstep(0.0, starSize, dist);
                star *= star;

                // Atmospheric scintillation — stronger near horizon
                float altitude   = max(dir.y, 0.0);
                float scintRate  = 2.0 + 5.0 * (1.0 - altitude);
                float scintDepth = mix(0.40, 0.10, altitude);  // Horizon twinkles more
                float twinkle    = 1.0 - scintDepth
                                 + scintDepth * sin(pc.cameraPos.w * scintRate * (1.0 + 2.0 * starHash)
                                                    + starHash * 47.0);
                // Occasional strong flicker (atmospheric cells)
                float flicker = sin(pc.cameraPos.w * 13.7 * starHash + starHash * 100.0);
                twinkle *= 1.0 - 0.12 * max(flicker, 0.0) * (1.0 - altitude);

                brightness += star * max(twinkle, 0.0) * (0.5 + 0.5 * starHash);
            }
        }
    }
    return brightness;
}

// Spectral colour from stellar temperature class
vec3 starColor(vec3 dir, float seed)
{
    float h = hash21(vec2(dir.x * 1000.0 + seed, dir.z * 1000.0));
    // Realistic distribution biased toward white/blue (hot main sequence)
    vec3 classO = vec3(0.55, 0.65, 1.0);     // Hot blue
    vec3 classB = vec3(0.70, 0.80, 1.0);     // Blue-white
    vec3 classA = vec3(0.90, 0.92, 1.0);     // White
    vec3 classF = vec3(1.0, 0.96, 0.88);     // Yellow-white
    vec3 classG = vec3(1.0, 0.90, 0.70);     // Yellow (Sun-like)
    vec3 classK = vec3(1.0, 0.72, 0.45);     // Orange
    vec3 classM = vec3(1.0, 0.45, 0.30);     // Red dwarf / giant

    if (h < 0.08) return classO;
    if (h < 0.25) return classB;
    if (h < 0.48) return classA;
    if (h < 0.65) return classF;
    if (h < 0.78) return classG;
    if (h < 0.90) return classK;
    return classM;
}

vec3 renderStars(vec3 dir, float starVis, float deepNightVis)
{
    if (starVis < 0.001) return vec3(0.0);

    float seed  = floor(pc.skyParams2.z);
    float sInt  = pc.skyParams.z;

    // ── Seasonal star rotation for celestial navigation ──────────────
    // Stars rotate over the 40-day year so players can determine the date
    // from star positions. yearProgress [0,1] maps to a full rotation.
    float yearProgress = fract(pc.skyParams2.z);
    float seasonAngle = yearProgress * TAU;
    // Rotate the view direction around the vertical axis by the season angle
    float csa = cos(seasonAngle), ssa = sin(seasonAngle);
    vec3 rotDir = vec3(dir.x * csa - dir.z * ssa, dir.y, dir.x * ssa + dir.z * csa);

    vec3  stars = vec3(0.0);

    // Layer 1: Giant / navigation stars (very sparse, VERY bright)
    // These are the key stars players learn to navigate by — visible from dusk
    float giants = starLayer(rotDir, 50.0, 0.94, seed);
    stars += giants * starColor(rotDir, seed) * 14.0;

    // Layer 2: Bright primary stars
    float bright = starLayer(rotDir, 90.0, 0.91, seed + 13.0);
    stars += bright * starColor(rotDir, seed + 13.0) * 7.0;

    // Layer 3: Medium stars
    float medium = starLayer(rotDir, 220.0, 0.87, seed + 37.0);
    stars += medium * starColor(rotDir, seed + 37.0) * 3.5;

    // Layer 4: Faint stars (appear later in twilight)
    float faint = starLayer(rotDir, 500.0, 0.82, seed + 61.0);
    stars += faint * starColor(rotDir, seed + 61.0) * 1.4 * deepNightVis;

    // Layer 5: Very faint star dust
    float dust1 = starLayer(rotDir, 1000.0, 0.79, seed + 89.0);
    stars += dust1 * vec3(0.80, 0.85, 1.0) * 0.45 * deepNightVis;

    // Layer 6: Ultra-faint background (visual texture)
    float dust2 = starLayer(rotDir, 1800.0, 0.76, seed + 127.0);
    stars += dust2 * vec3(0.75, 0.80, 0.95) * 0.24 * deepNightVis;

    // First 3 layers use starVis, last 3 use deepNightVis (applied above)
    return stars * starVis * sInt;
}

// ═══════════════════════════════════════════════════════════════════════
// Galaxy Band — spiral structure, dust lanes, emission regions
// ═══════════════════════════════════════════════════════════════════════

vec3 renderGalaxy(vec3 dir, float nightVisibility)
{
    if (nightVisibility < 0.001) return vec3(0.0);

    float seed = pc.skyParams2.z;

    // Galaxy plane normal — tilted great circle across the sky
    vec3   galaxyAxis = normalize(vec3(0.30, 0.10, 0.95));
    float  galacticLat = abs(dot(dir, galaxyAxis));

    // Band intensity — narrow Gaussian
    float bandWidth     = 0.13;
    float bandIntensity = exp(-galacticLat * galacticLat / (2.0 * bandWidth * bandWidth));
    if (bandIntensity < 0.005) return vec3(0.0);

    // Galactic coordinate system
    vec3  galRight   = normalize(cross(galaxyAxis, vec3(0.0, 1.0, 0.0)));
    vec3  galForward = cross(galRight, galaxyAxis);
    float galLon     = atan(dot(dir, galForward), dot(dir, galRight));

    // UV within the band
    vec2 galUV = vec2(galLon * 3.0, galacticLat * 15.0);

    // ── Spiral arm hints via domain warping ──────────────────────────
    float spiralWarp = sin(galLon * 2.0 + galacticLat * 20.0 + seed) * 0.3;
    galUV.y += spiralWarp;

    // Multi-frequency cloud structure
    float cloud1 = fbm(galUV * 2.0 + seed, 5);
    float cloud2 = fbm(galUV * 5.5 + vec2(seed * 0.7, 0.0), 4);
    float cloud3 = fbm(galUV * 14.0 + vec2(0.0, seed * 1.3), 3);

    float galaxyDensity = cloud1 * 0.55 + cloud2 * 0.30 + cloud3 * 0.15;
    galaxyDensity = smoothstep(0.18, 0.80, galaxyDensity) * bandIntensity;

    // ── Multiple dark dust lanes ─────────────────────────────────────
    float dustLane1 = fbm(galUV * 9.0 + vec2(seed * 2.1, seed * 0.5), 4);
    float dustLane2 = fbm(galUV * 6.0 + vec2(seed * 1.1, seed * 2.3), 3);
    float dustAbsorption = smoothstep(0.32, 0.62, dustLane1) * 0.7
                         + smoothstep(0.40, 0.65, dustLane2) * 0.3;
    galaxyDensity *= mix(0.08, 1.0, dustAbsorption);

    // ── Galactic core (warm, bright, centrally concentrated) ─────────
    float coreDist   = length(vec2(galLon, galacticLat * 5.0) - vec2(0.5, 0.0));
    float coreBright = exp(-coreDist * coreDist * 3.5);
    galaxyDensity   += coreBright * 0.45;

    // ── Colour ───────────────────────────────────────────────────────
    vec3 coreColor   = vec3(1.0, 0.82, 0.55);   // Warm old-star core
    vec3 armColor    = vec3(0.55, 0.65, 1.0);    // Blue young-star arms
    vec3 nebulaColor = vec3(0.90, 0.28, 0.45);   // Pink HII emission

    float nebulaRegion = fbm(galUV * 7.0 + vec2(seed * 3.0, 0.0), 3);
    nebulaRegion = smoothstep(0.52, 0.78, nebulaRegion) * bandIntensity;

    vec3 galaxyColor = mix(armColor, coreColor, coreBright);
    galaxyColor      = mix(galaxyColor, nebulaColor, nebulaRegion * 0.55);

    // Embedded bright stars within the band
    float galaxyStars = starLayer(dir, 350.0, 0.86, seed + 200.0);
    galaxyColor += vec3(1.0, 0.95, 0.90) * galaxyStars * 1.8 * bandIntensity;

    // Second population: blue hot stars in the arms
    float blueStars = starLayer(dir, 280.0, 0.92, seed + 300.0);
    galaxyColor += vec3(0.6, 0.7, 1.2) * blueStars * 1.2 * bandIntensity * (1.0 - coreBright);

    return galaxyColor * galaxyDensity * nightVisibility * 0.40 * pc.skyParams.z;
}

// ═══════════════════════════════════════════════════════════════════════
// Nebulae — 6 distinct regions with volumetric depth
// ═══════════════════════════════════════════════════════════════════════

vec3 singleNebula(vec3 dir, vec3 center, float size, vec3 color1, vec3 color2,
                  float detailScale, float seed, float nightVis)
{
    float dist = 1.0 - dot(dir, center);
    if (dist > size * 5.0) return vec3(0.0);  // Early out

    float intensity = exp(-dist * dist / (2.0 * size * size));

    // Spherical UV for detail noise
    vec2 nuv  = vec2(atan(dir.z, dir.x), asin(clamp(dir.y, -1.0, 1.0)));

    // Domain-warped noise for organic, volumetric shapes
    float warp1 = fbm(nuv * detailScale * 0.5 + seed, 3);
    float warp2 = fbm(nuv * detailScale * 0.5 + seed + 5.0, 3);
    vec2  warped = nuv * detailScale + vec2(warp1, warp2) * 2.0;

    float detail = fbm(warped + seed, 4);
    intensity *= smoothstep(0.22, 0.72, detail);

    // Internal structure: filaments and bright knots
    float filaments = fbm(warped * 2.0 + seed * 1.7, 3);
    filaments = smoothstep(0.4, 0.8, filaments);

    // Colour varies between two tones based on internal detail
    vec3 nebulaCol = mix(color1, color2, filaments * 0.6);

    // Bright embedded stars (tiny hot stars illuminating the gas)
    float embeddedStars = starLayer(dir, 600.0, 0.97, seed + 500.0);
    nebulaCol += vec3(1.0, 0.95, 0.90) * embeddedStars * 2.5 * intensity;

    return nebulaCol * intensity * nightVis * 0.40;
}

vec3 renderNebulae(vec3 dir, float nightVisibility)
{
    if (nightVisibility < 0.001) return vec3(0.0);

    float seed = pc.skyParams2.z;
    float sInt = pc.skyParams.z;
    vec3  nebula = vec3(0.0);

    // Nebula 1: Large purple/magenta emission nebula (Orion-like)
    nebula += singleNebula(dir, normalize(vec3(0.60, 0.38, -0.70)),
                           0.16,
                           vec3(0.50, 0.12, 0.70), vec3(0.75, 0.30, 0.85),
                           18.0, seed, nightVisibility);

    // Nebula 2: Red supernova remnant (Crab-like)
    nebula += singleNebula(dir, normalize(vec3(-0.50, 0.60, 0.62)),
                           0.18,
                           vec3(0.82, 0.18, 0.12), vec3(1.0, 0.50, 0.20),
                           15.0, seed * 1.7, nightVisibility);

    // Nebula 3: Teal/green planetary nebula (Ring-like)
    nebula += singleNebula(dir, normalize(vec3(0.28, 0.72, 0.63)),
                           0.10,
                           vec3(0.08, 0.65, 0.60), vec3(0.20, 0.80, 0.55),
                           22.0, seed * 2.3, nightVisibility);

    // Nebula 4: Blue reflection nebula (Pleiades-like)
    nebula += singleNebula(dir, normalize(vec3(-0.70, 0.30, -0.65)),
                           0.14,
                           vec3(0.30, 0.45, 0.90), vec3(0.50, 0.60, 1.0),
                           16.0, seed * 3.1, nightVisibility);

    // Nebula 5: Golden/amber star-forming pillar (Pillars of Creation-like)
    nebula += singleNebula(dir, normalize(vec3(0.80, 0.50, 0.30)),
                           0.12,
                           vec3(0.85, 0.55, 0.15), vec3(0.70, 0.35, 0.10),
                           20.0, seed * 4.0, nightVisibility);

    // Nebula 6: Dark absorption nebula with faint red rim (Horsehead-like)
    // This one absorbs light rather than emitting it
    {
        vec3  center = normalize(vec3(-0.20, 0.55, -0.80));
        float dist   = 1.0 - dot(dir, center);
        if (dist < 0.10 * 5.0) {
            float intensity = exp(-dist * dist / (2.0 * 0.10 * 0.10));
            vec2  nuv    = vec2(atan(dir.z, dir.x), asin(clamp(dir.y, -1.0, 1.0)));
            float detail = warpedFBM(nuv * 18.0 + seed * 5.5, 4);
            float shape  = smoothstep(0.25, 0.65, detail);

            // Dark absorption: subtract light, add faint red edge glow
            float absorption = intensity * shape * nightVisibility;
            nebula -= nebula * absorption * 0.5;
            float edge = smoothstep(0.3, 0.5, detail) * (1.0 - smoothstep(0.5, 0.7, detail));
            nebula += vec3(0.60, 0.10, 0.05) * edge * intensity * nightVisibility * 0.20;
        }
    }

    return nebula * sInt;
}

// ═══════════════════════════════════════════════════════════════════════
// Planetary Ring — forward-scatter, planet shadow, fine ringlets
// ═══════════════════════════════════════════════════════════════════════

vec3 renderPlanetaryRing(vec3 dir, float nightVisibility)
{
    float time         = pc.cameraPos.w;
    float ringRotation = pc.skyParams.w;

    // Slowly rotating ring normal
    float rotAngle = ringRotation;
    vec3 ringNormal = normalize(vec3(
        k_RingNormal.x * cos(rotAngle) - k_RingNormal.z * sin(rotAngle),
        k_RingNormal.y,
        k_RingNormal.x * sin(rotAngle) + k_RingNormal.z * cos(rotAngle)
    ));

    float denom = dot(dir, ringNormal);
    if (abs(denom) < 0.001) return vec3(0.0);

    float angleFromPlane = asin(clamp(abs(denom), 0.0, 1.0));
    float ringWidth = 0.065;
    float ringBandIntensity = exp(-angleFromPlane * angleFromPlane / (2.0 * ringWidth * ringWidth));
    if (ringBandIntensity < 0.002) return vec3(0.0);

    // Azimuthal and radial coordinates
    vec3  ringRight   = normalize(cross(ringNormal, vec3(0.0, 1.0, 0.01)));
    vec3  ringForward = cross(ringRight, ringNormal);
    float ringAzimuth = atan(dot(dir, ringForward), dot(dir, ringRight));
    float ringRadial  = angleFromPlane / ringWidth;
    float ringUV      = ringAzimuth / TAU + 0.5;

    // ── Fine radial structure — thousands of ringlets ────────────────
    float fineRinglets = 0.7 + 0.3 * noise2D(vec2(ringUV * 50.0, ringRadial * 300.0));
    fineRinglets *= 0.8 + 0.2 * noise2D(vec2(ringUV * 120.0, ringRadial * 600.0 + time * 0.08));

    // ── Band structure with Cassini-like divisions ───────────────────
    float ringPattern = 0.0;

    // Band A (outer): diffuse, wider
    float bandA = smoothstep(0.0, 0.12, ringRadial) * (1.0 - smoothstep(0.30, 0.38, ringRadial));
    float bandANoise = fineRinglets * (0.75 + 0.25 * noise2D(vec2(ringUV * 200.0 + time * 0.15, ringRadial * 50.0)));
    ringPattern += bandA * bandANoise;

    // Cassini Division (dark gap)
    float cassiniGap = smoothstep(0.36, 0.40, ringRadial) * (1.0 - smoothstep(0.48, 0.52, ringRadial));
    ringPattern *= (1.0 - cassiniGap * 0.88);

    // Band B (main, brightest, densest)
    float bandB = smoothstep(0.46, 0.54, ringRadial) * (1.0 - smoothstep(0.76, 0.84, ringRadial));
    float bandBNoise = fineRinglets * (0.65 + 0.35 * noise2D(vec2(ringUV * 400.0 + time * 0.12, ringRadial * 80.0)));
    ringPattern += bandB * bandBNoise * 1.4;

    // Encke Gap (narrow division in B ring)
    float enckeGap = smoothstep(0.64, 0.65, ringRadial) * (1.0 - smoothstep(0.66, 0.67, ringRadial));
    ringPattern *= (1.0 - enckeGap * 0.7);

    // Band C (inner, translucent)
    float bandC = smoothstep(0.82, 0.87, ringRadial) * (1.0 - smoothstep(0.95, 1.0, ringRadial));
    float bandCNoise = fineRinglets * (0.55 + 0.45 * noise2D(vec2(ringUV * 150.0 + time * 0.08, ringRadial * 30.0)));
    ringPattern += bandC * bandCNoise * 0.35;

    // ── Ring colour — compositional variation ────────────────────────
    vec3 dustGold = vec3(0.88, 0.75, 0.52);   // Silicate dust
    vec3 iceBlue  = vec3(0.62, 0.68, 0.85);   // Water ice particles
    vec3 darkGap  = vec3(0.30, 0.25, 0.20);   // Gap material
    vec3 ringColor = mix(dustGold, iceBlue, smoothstep(0.30, 0.85, ringRadial));
    ringColor      = mix(ringColor, darkGap, cassiniGap * 0.5);

    // ── Ice particle sparkle ─────────────────────────────────────────
    float sparkle = starLayer(dir, 900.0, 0.96, pc.skyParams2.z + 500.0);
    ringColor += vec3(1.0, 0.96, 0.92) * sparkle * 1.0 * ringBandIntensity;

    // ── Forward scattering (bright when backlit by sun) ──────────────
    // When looking through the ring toward the sun, thin parts scatter light forward
    float sunAngle       = dot(dir, pc.sunDir.xyz);
    float forwardScatter = pow(max(sunAngle, 0.0), 16.0) * 0.6;
    // Thin regions (C ring) forward-scatter more
    float thinness = (1.0 - bandB * 0.7);
    forwardScatter *= thinness;
    ringColor += vec3(1.0, 0.92, 0.75) * forwardScatter;

    // ── Planet shadow across the ring ────────────────────────────────
    // Where the planet blocks sunlight, the ring is in shadow
    // Project sun direction onto the ring plane to find shadow region
    vec3  sunOnPlane = pc.sunDir.xyz - ringNormal * dot(pc.sunDir.xyz, ringNormal);
    float shadowAz   = atan(dot(normalize(sunOnPlane), ringForward),
                            dot(normalize(sunOnPlane), ringRight));
    // Shadow is opposite to sun direction on the ring, in a cone
    float shadowAngle = ringAzimuth - shadowAz + PI;
    float planetShadow = 1.0 - 0.75 * exp(-shadowAngle * shadowAngle / 0.06)
                              * smoothstep(0.2, 0.6, ringRadial); // Only inner parts shadowed
    ringColor *= planetShadow;

    // ── Sun/night lit factor ─────────────────────────────────────────
    // The ring is sunlit, so its brightness should track how much the sun
    // actually faces it. Deep at night the sun is far below the horizon and the
    // visible ring is mostly in the planet's shadow / only faint earthshine, so
    // fade it well down instead of leaving it glaring against the dark sky.
    float sunLit    = dot(ringNormal, pc.sunDir.xyz);
    float litFactor = 0.18 + 0.82 * max(sunLit, 0.0);
    // Sun-elevation falloff: full by day, ~0.22 at deep night.
    float ringSunIllum = clamp(0.22 + 0.78 * smoothstep(-0.35, 0.05, pc.sunDir.y), 0.0, 1.0);
    litFactor *= ringSunIllum;

    // Stars faintly visible through thin ring parts (C ring)
    float transparency = bandC * 0.25;

    // Day dimming (atmosphere overpowers ring during day). Kept very low so the
    // ring is a subtle night/twilight feature rather than bands across the day sky.
    float dayDim     = mix(1.0, 0.03, smoothstep(-0.05, 0.20, pc.sunDir.y));
    float visibility = mix(dayDim, 1.0, nightVisibility);

    return ringColor * ringPattern * ringBandIntensity * litFactor * visibility * 0.55;
}

// ═══════════════════════════════════════════════════════════════════════
// Nearby Planet — massive, horizon-hugging, with oceans, continents, clouds
// Replaces the moon as the primary navigational celestial body.
// The planet orbits 360° around the world but never appears fully above
// the horizon — players see a partial disk rising and setting, providing
// a phase cycle for date tracking alongside the seasonal star rotation.
// ═══════════════════════════════════════════════════════════════════════

vec3 renderPlanet(vec3 dir, float nightVisibility)
{
    float dayProgress     = pc.skyParams.x;
    float planetOrbPhase  = pc.skyParams2.w;   // [0,1] position in orbit
    float orbitalAngle    = planetOrbPhase * TAU;
    float seed            = floor(pc.skyParams2.z);

    // ── Orbital parameters — massive nearby planet near the horizon ──
    // Angular radius: ~10.3° — a HUGE nearby world dominating the sky
    const float k_PlanetAngRadius = 0.18;
    // Center elevation oscillates: -0.28 (below horizon) to +0.08 (just above)
    // This ensures the bottom of the disk is ALWAYS at or below the horizon.
    const float k_MaxCenterElev   = 0.08;
    const float k_MinCenterElev   = -0.28;
    // Sinusoidal elevation: peak at π/2 (quarter orbit), trough at 3π/2
    float centerElev = mix(k_MinCenterElev, k_MaxCenterElev,
                           sin(orbitalAngle) * 0.5 + 0.5);

    // Planet center direction (orbits around the horizon)
    vec3 planetDir = normalize(vec3(
        cos(orbitalAngle),
        centerElev,
        sin(orbitalAngle)
    ));

    // ── Angular distance check ───────────────────────────────────────
    float cosAngle = dot(dir, planetDir);
    float angDist  = acos(clamp(cosAngle, -1.0, 1.0));

    // ── Atmospheric glow visible slightly beyond the disk edge ───────
    if (angDist > k_PlanetAngRadius && angDist < k_PlanetAngRadius + 0.05) {
        float glowDist = angDist - k_PlanetAngRadius;
        float atmoGlow = exp(-glowDist * glowDist / 0.00025);
        float phaseDot = dot(planetDir, pc.sunDir.xyz);
        float phaseIll = clamp((phaseDot + 1.0) * 0.5, 0.0, 1.0);
        // Blue atmospheric rim visible whenever planet is above horizon
        vec3 atmoColor = vec3(0.10, 0.20, 0.48) * atmoGlow * phaseIll;
        float horizClip = smoothstep(-0.005, 0.015, dir.y);
        return atmoColor * horizClip * 0.35 * max(nightVisibility * 0.7 + 0.3, 0.0);
    }

    if (angDist > k_PlanetAngRadius) return vec3(0.0);

    // ── Disk mask with anti-aliased edge ─────────────────────────────
    float diskMask = smoothstep(k_PlanetAngRadius, k_PlanetAngRadius - 0.003, angDist);

    // ── Horizon clipping — sharp cut at y=0 (planet occluded by world) ──
    float horizClip = smoothstep(-0.003, 0.008, dir.y);
    diskMask *= horizClip;
    if (diskMask < 0.001) return vec3(0.0);

    // ── Surface coordinate mapping (hemisphere facing viewer) ────────
    vec3  toCenter = dir - planetDir * cosAngle;
    float toCenterLen = length(toCenter);
    vec3  localDir = (toCenterLen > 0.0001) ? toCenter / toCenterLen : vec3(0.0, 1.0, 0.0);

    // Planet's local coordinate frame (stable up axis)
    vec3  pRight = normalize(cross(vec3(0.0, 1.0, 0.001), planetDir));
    vec3  pUp    = normalize(cross(planetDir, pRight));

    float px = dot(localDir, pRight) * (angDist / k_PlanetAngRadius);
    float py = dot(localDir, pUp)    * (angDist / k_PlanetAngRadius);

    // Map to 3D point on a visible hemisphere (z = depth into sphere)
    float r2 = clamp(px * px + py * py, 0.0, 1.0);
    float sz = sqrt(max(0.0, 1.0 - r2));

    vec3 surfPt = vec3(px, py, sz);

    // Slow rotation — the planet rotates on its own axis
    float planetRot = pc.cameraPos.w * 0.006;
    float cr = cos(planetRot), sr = sin(planetRot);
    surfPt = vec3(surfPt.x * cr - surfPt.z * sr, surfPt.y, surfPt.x * sr + surfPt.z * cr);

    // Seed offset for unique planet per world
    vec3  seedOff = vec3(seed * 0.13, seed * 0.07, seed * 0.19);

    // ── Continental terrain (multi-octave noise) ─────────────────────
    float continent = fbm((surfPt + seedOff).xy * 2.5 + (surfPt.z + seedOff.z) * 1.8, 5);
    continent = smoothstep(0.36, 0.56, continent);

    // Sub-continental mountain detail
    float mountains = fbm((surfPt + seedOff).xy * 10.0 + surfPt.z * 7.0, 4);
    mountains = smoothstep(0.52, 0.72, mountains) * continent;

    // ── Ocean ────────────────────────────────────────────────────────
    vec3 deepOcean    = vec3(0.01, 0.04, 0.16);
    vec3 shallowOcean = vec3(0.04, 0.12, 0.28);
    float coastal = smoothstep(0.32, 0.40, continent);
    vec3 oceanColor = mix(deepOcean, shallowOcean, coastal);
    // Subtle ocean surface variation
    oceanColor += vec3(0.005, 0.01, 0.02) * noise2D((surfPt.xy + seedOff.xy) * 8.0);

    // ── Land biomes by latitude ──────────────────────────────────────
    float latitude = abs(py);
    vec3 tropical  = vec3(0.06, 0.20, 0.04);    // Deep green (equator)
    vec3 temperate = vec3(0.12, 0.18, 0.06);    // Cooler green
    vec3 arid      = vec3(0.30, 0.24, 0.12);    // Brown desert
    vec3 tundra    = vec3(0.28, 0.30, 0.28);    // Grey-green

    vec3 landColor = mix(tropical, temperate, smoothstep(0.10, 0.30, latitude));
    landColor = mix(landColor, arid, smoothstep(0.22, 0.50, latitude) * (1.0 - mountains * 0.6));
    landColor = mix(landColor, tundra, smoothstep(0.55, 0.75, latitude));
    // Mountains are lighter (grey-brown rock)
    landColor = mix(landColor, vec3(0.22, 0.20, 0.17), mountains * 0.6);

    // ── Ice caps (poles) ─────────────────────────────────────────────
    float iceCap = smoothstep(0.72, 0.88, latitude);
    // Slight noise on ice cap boundary for natural look
    iceCap *= smoothstep(0.45, 0.55, noise2D((surfPt.xy + seedOff.xy) * 6.0) + latitude * 0.2);
    vec3 iceColor = vec3(0.82, 0.86, 0.94);
    landColor = mix(landColor, iceColor, iceCap);

    // ── Combine ocean and land ───────────────────────────────────────
    vec3 surfaceColor = mix(oceanColor, landColor, continent);
    // Sea ice near poles
    surfaceColor = mix(surfaceColor, iceColor * 0.9, iceCap * 0.4 * (1.0 - continent));

    // ── Cloud layer (animated, swirling) ─────────────────────────────
    vec3  cloudPt  = surfPt * 1.8 + seedOff
                   + vec3(pc.cameraPos.w * 0.002, 0.0, pc.cameraPos.w * 0.0015);
    float clouds   = fbm(cloudPt.xy * 3.5 + cloudPt.z * 2.5, 5);
    clouds = smoothstep(0.32, 0.62, clouds);

    // Cyclonic swirl patterns
    float swirl = noise2D(surfPt.xy * 1.2 + vec2(pc.cameraPos.w * 0.004, seedOff.x));
    clouds *= (0.65 + 0.35 * swirl);

    // Thicker cloud bands near equator, thinner at poles
    clouds *= (1.0 - latitude * 0.35);

    // Cloud shadow on surface below (subtle darkening where clouds are)
    surfaceColor *= (1.0 - clouds * 0.15);

    vec3 cloudWhite = vec3(0.88, 0.90, 0.95);
    surfaceColor = mix(surfaceColor, cloudWhite, clouds * 0.7);

    // ── Limb darkening + atmospheric rim ─────────────────────────────
    float limbFactor = 1.0 - (angDist / k_PlanetAngRadius);
    limbFactor = pow(max(limbFactor, 0.0), 0.35);

    // Atmospheric blue-purple rim at the limb
    float limbGlow = smoothstep(0.35, 0.95, 1.0 - limbFactor);
    vec3  atmosphereRim = vec3(0.12, 0.22, 0.55) * limbGlow;
    surfaceColor = surfaceColor * limbFactor + atmosphereRim * 0.45;

    // ── Phase illumination (day/night terminator from sun direction) ──
    // Physically-based: the terminator depends on the sun's position
    // relative to the planet, exactly like real planet phases.
    float phaseDot = dot(planetDir, pc.sunDir.xyz);
    // Project sun direction onto the planet's disk plane for terminator
    vec3  sunOnDisk = pc.sunDir.xyz - planetDir * phaseDot;
    float sunOnLen  = length(sunOnDisk);
    vec3  sunProj   = (sunOnLen > 0.001) ? sunOnDisk / sunOnLen : pRight;
    // Terminator position: how far across the disk the shadow falls
    float terminatorPos = phaseDot;  // -1 = sun behind planet (dark), +1 = sun behind viewer (full)
    // Each fragment's position relative to the terminator
    float fragSunSide = dot(vec3(px, py, sz), sunProj) * sign(terminatorPos);
    float rawIllum = fragSunSide - abs(terminatorPos) + 1.0;
    float illumination = smoothstep(-0.15, 0.15, rawIllum);

    // Terminator detail — cloud edges catch light at the shadow boundary
    float terminatorDist = abs(rawIllum);
    float terminatorGlow = clouds * 0.35 * smoothstep(0.20, 0.0, terminatorDist);

    // Dark side: very faint ambient (reflected starlight + ring light)
    float darkSideAmbient = 0.015;

    surfaceColor *= (illumination + darkSideAmbient + terminatorGlow);

    // ── Specular ocean reflection (sun glint on water) ───────────────
    if (continent < 0.3 && illumination > 0.3) {
        vec3  halfVec = normalize(sunProj + vec3(0.0, 0.0, 1.0));
        float specular = pow(max(dot(vec3(px, py, sz), halfVec), 0.0), 64.0);
        surfaceColor += vec3(0.2, 0.18, 0.15) * specular * (1.0 - continent) * illumination * 0.3;
    }

    // ── City lights on the dark side (inhabited world) ───────────────
    {
        float darkness = 1.0 - smoothstep(0.0, 0.20, illumination);
        if (darkness > 0.01 && continent > 0.35)
        {
            // Population density: coastal plains preferred, mountains and ice sparse
            float popDensity = smoothstep(0.35, 0.50, continent)
                             * (1.0 - mountains * 0.85)
                             * (1.0 - iceCap);

            // Coastal megacities (brighter near coastline)
            float coastProx = smoothstep(0.32, 0.42, continent)
                            * (1.0 - smoothstep(0.42, 0.58, continent));
            popDensity += coastProx * 0.8;

            // Multi-scale city clusters for realism
            float cities = noise2D((surfPt.xy + seedOff.xy) * 65.0);
            cities = smoothstep(0.58, 0.72, cities);

            float metro = noise2D((surfPt.xy + seedOff.xy) * 30.0);
            metro = smoothstep(0.50, 0.70, metro);

            float points = noise2D((surfPt.xy + seedOff.xy) * 140.0);
            points = smoothstep(0.70, 0.82, points);

            float lightIntensity = (cities * 0.5 + metro * 0.3 + points * 0.2)
                                 * popDensity;

            // Regional colour variation: warm sodium vs cool LED
            float colorVar = noise2D((surfPt.xy + seedOff.xy) * 15.0);
            vec3 warmLight = vec3(1.0, 0.78, 0.42);
            vec3 coolLight = vec3(0.85, 0.90, 1.0);
            vec3 cityColor = mix(warmLight, coolLight, colorVar * 0.25);

            surfaceColor += cityColor * lightIntensity * darkness * 0.04;
        }
    }

    // ── Overall brightness and visibility ────────────────────────────
    float brightness = 2.8;  // Planet is a bright reflective body

    // Day dimming (planet visible at dusk/dawn and night, faint during day)
    float dayDim = mix(1.0, 0.08, smoothstep(-0.05, 0.25, pc.sunDir.y));
    float vis = mix(dayDim, 1.0, nightVisibility);

    return surfaceColor * diskMask * brightness * vis;
}

// ═══════════════════════════════════════════════════════════════════════
// Shooting Stars & Meteor Showers — long trails, fireballs, afterglow
// ═══════════════════════════════════════════════════════════════════════

vec3 renderShootingStar(vec3 dir, float startTime, float duration,
                        vec3 startPos, vec3 endPos, vec3 color, float brightness)
{
    float time = pc.cameraPos.w;
    float t = (time - startTime) / duration;

    // Active trail (visible during flight)
    if (t >= 0.0 && t <= 1.0) {
        vec3 meteorPos = mix(startPos, endPos, t);

        // Long glowing trail with 16 sample points
        float streak = 0.0;
        for (int i = 0; i < 16; i++) {
            float trailT = t - float(i) * 0.006;
            if (trailT < 0.0) break;
            vec3  trailPos  = mix(startPos, endPos, trailT);
            float d         = 1.0 - dot(dir, normalize(trailPos));
            float pointSize = 0.000025 * (1.0 + float(i) * 0.6);
            float fade      = 1.0 - float(i) / 16.0;
            fade *= fade;  // Quadratic falloff for natural trail shape
            streak += exp(-d / pointSize) * fade;
        }

        // Bright head
        float headDist = 1.0 - dot(dir, normalize(meteorPos));
        float head     = exp(-headDist / 0.000015) * 2.0;
        streak += head;

        // Fade in/out
        float fadeFactor = smoothstep(0.0, 0.08, t) * (1.0 - smoothstep(0.75, 1.0, t));

        return color * streak * fadeFactor * brightness;
    }

    // Afterglow (lingers 0.5–1.5s after meteor passes)
    float afterT = (time - (startTime + duration));
    if (afterT > 0.0 && afterT < 1.5) {
        float afterFade = 1.0 - afterT / 1.5;
        afterFade *= afterFade;

        // Ghost trail along the full path
        float afterStreak = 0.0;
        for (int i = 0; i < 6; i++) {
            float trailT = 0.3 + float(i) * 0.10;
            vec3  pt     = mix(startPos, endPos, trailT);
            float d      = 1.0 - dot(dir, normalize(pt));
            afterStreak += exp(-d / 0.00008) * (1.0 - float(i) / 6.0);
        }

        return color * afterStreak * afterFade * brightness * 0.12;
    }

    return vec3(0.0);
}

vec3 renderMeteors(vec3 dir, float nightVisibility)
{
    if (nightVisibility < 0.1) return vec3(0.0);

    float time         = pc.cameraPos.w;
    float seed         = pc.skyParams2.z;
    float meteorTimer  = pc.skyParams2.x;
    float meteorActive = pc.skyParams2.y;
    vec3  meteors      = vec3(0.0);

    // ── Occasional shooting stars ────────────────────────────────────
    float starCycle = 12.0;
    for (int i = 0; i < 3; i++) {
        float offset    = float(i) * starCycle * 0.33;
        float spawnHash = hash11(floor((meteorTimer + offset) / starCycle) * 7.31 + seed);

        if (spawnHash > 0.68) {
            float startT  = floor((meteorTimer + offset) / starCycle) * starCycle - offset;
            float duration = 0.5 + spawnHash * 0.8;

            vec3 startDir = normalize(vec3(
                hash11(spawnHash * 100.0 + 1.0) * 2.0 - 1.0,
                0.25 + hash11(spawnHash * 100.0 + 2.0) * 0.65,
                hash11(spawnHash * 100.0 + 3.0) * 2.0 - 1.0
            ));
            vec3 velocity = normalize(vec3(
                hash11(spawnHash * 100.0 + 4.0) * 0.4 - 0.2,
                -0.12 - hash11(spawnHash * 100.0 + 5.0) * 0.15,
                hash11(spawnHash * 100.0 + 6.0) * 0.4 - 0.2
            ));
            vec3 endDir = startDir + velocity * 0.18;

            vec3 meteorColor = mix(vec3(1.0, 0.95, 0.80), vec3(0.80, 0.88, 1.0), hash11(spawnHash * 200.0));
            meteors += renderShootingStar(dir, startT, duration, startDir, endDir, meteorColor, 3.5);
        }
    }

    // ── Meteor shower (CPU-triggered) ────────────────────────────────
    if (meteorActive > 0.5) {
        // Atmospheric glow during intense showers
        float showerGlow = 0.005 * nightVisibility;
        meteors += vec3(0.02, 0.03, 0.05) * showerGlow;

        for (int i = 0; i < 12; i++) {
            float showerCycle = 2.5;
            float offset      = float(i) * showerCycle / 12.0;
            float spawnHash   = hash11(floor((meteorTimer + offset) / showerCycle) * 13.7 + seed + 1000.0);

            if (spawnHash > 0.35) {
                float startT  = floor((meteorTimer + offset) / showerCycle) * showerCycle - offset;
                float duration = 0.3 + spawnHash * 0.6;

                // Radiant point — all meteors diverge from a common sky direction
                vec3 radiant = normalize(vec3(0.40, 0.80, 0.30));
                vec3 spread  = vec3(
                    (hash11(spawnHash * 100.0 + 10.0) - 0.5) * 0.45,
                    (hash11(spawnHash * 100.0 + 11.0) - 0.5) * 0.35,
                    (hash11(spawnHash * 100.0 + 12.0) - 0.5) * 0.45
                );
                vec3 startDir = normalize(radiant + spread);
                vec3 vel      = normalize(startDir - radiant * 0.5 + vec3(0.0, -0.3, 0.0));
                vec3 endDir   = startDir + vel * 0.25;

                // Colour from composition
                float colorHash = hash11(spawnHash * 300.0);
                vec3  meteorColor;
                if (colorHash < 0.25) meteorColor = vec3(0.30, 1.0, 0.40);        // Green (magnesium)
                else if (colorHash < 0.50) meteorColor = vec3(1.0, 0.58, 0.18);   // Orange (iron)
                else if (colorHash < 0.75) meteorColor = vec3(1.0, 0.95, 0.85);   // White (silicate)
                else meteorColor = vec3(0.70, 0.80, 1.0);                          // Blue (nitrogen)

                // Every 4th meteor is a fireball (3x brighter, larger)
                float fireball = (mod(float(i), 4.0) < 0.5) ? 3.5 : 1.0;
                if (fireball > 2.0) duration *= 0.7;  // Fireballs burn faster

                meteors += renderShootingStar(dir, startT, duration, startDir, endDir,
                                              meteorColor, 3.0 * fireball);
            }
        }
    }

    return meteors * nightVisibility;
}

// ═══════════════════════════════════════════════════════════════════════
// Lightning — procedural flashes during storms
// ═══════════════════════════════════════════════════════════════════════

vec3 renderLightning(vec3 dir, float cloudCover)
{
    if (cloudCover < 0.65) return vec3(0.0);

    float time = pc.cameraPos.w;
    float seed = pc.skyParams2.z;

    // Lightning probability increases with cloud cover
    float stormIntensity = smoothstep(0.65, 0.95, cloudCover);

    // Procedural timing: rapid hash checks at ~3 Hz, with rare triggers
    float flashBin    = floor(time * 3.0);
    float flashRoll   = hash11(flashBin * 17.3 + seed * 3.7);
    float shouldFlash = step(1.0 - stormIntensity * 0.08, flashRoll);

    if (shouldFlash < 0.5) return vec3(0.0);

    // Flash timing within the bin (~0.33 second window)
    float flashTime   = fract(time * 3.0);
    float flashBright = exp(-flashTime * 12.0);  // Very rapid flash decay

    // Double flash (common in real lightning)
    float secondFlash = exp(-(flashTime - 0.15) * (flashTime - 0.15) * 200.0) * 0.6;
    flashBright += secondFlash;

    // Flash direction — illuminates a patch of sky
    vec2  flashDir    = hash22(vec2(flashBin, seed)) * 2.0 - 1.0;
    vec3  flashCenter = normalize(vec3(flashDir.x, 0.1 + hash11(flashBin + 7.0) * 0.3, flashDir.y));
    float flashDist   = max(1.0 - dot(dir, flashCenter), 0.0);
    float flashSpread = exp(-flashDist * 8.0);

    // Lightning colour: blue-white
    vec3 lightningColor = vec3(0.75, 0.80, 1.0);

    return lightningColor * flashBright * flashSpread * stormIntensity * 0.4;
}

// ═══════════════════════════════════════════════════════════════════════
// Aurora — rare animated curtains on clear magnetic nights
// ═══════════════════════════════════════════════════════════════════════

vec3 renderAurora(vec3 dir, float nightVisibility)
{
    if (nightVisibility < 0.3 || dir.y < 0.01) return vec3(0.0);

    float cloudCover = pc.skyParams.y;
    if (cloudCover > 0.40) return vec3(0.0);  // Clouds hide aurora

    float time = pc.cameraPos.w;
    float seed = pc.skyParams2.z;

    // Aurora activity: appears and fades over ~4 minute cycles
    // Active roughly 20% of clear nights — rare but spectacular
    float activity = noise2D(vec2(time * 0.004, seed * 0.1));
    activity = smoothstep(0.58, 0.78, activity);
    if (activity < 0.01) return vec3(0.0);

    // Aurora visible toward the "magnetic pole"
    vec3  magneticPole   = normalize(vec3(0.65, 0.0, 0.76));
    float poleDir        = dot(normalize(vec3(dir.x, 0.0, dir.z)), magneticPole);
    float poleProximity  = smoothstep(0.15, 0.75, poleDir);
    if (poleProximity < 0.01) return vec3(0.0);

    // Curtain shape: vertical ribbons with flowing horizontal motion
    float azimuth   = atan(dir.z, dir.x);
    float elevation = dir.y;

    // Layered curtain waves at different speeds
    float curtain = 0.0;
    curtain += (sin(azimuth * 10.0 + time * 0.55 + sin(time * 0.3) * 2.0) * 0.5 + 0.5) * 0.6;
    curtain *= (sin(azimuth * 4.0  - time * 0.25) * 0.3 + 0.7);
    curtain *= noise2D(vec2(azimuth * 6.0 + time * 0.35, elevation * 4.0 - time * 0.8));

    // Altitude distribution: peaks 10–30 degrees above horizon
    float altWeight = smoothstep(0.02, 0.12, elevation) * (1.0 - smoothstep(0.22, 0.55, elevation));

    // Colour: green at lower altitude, purple/magenta higher
    vec3  auroraGreen  = vec3(0.12, 0.85, 0.30);
    vec3  auroraPurple = vec3(0.55, 0.10, 0.80);
    vec3  auroraRed    = vec3(0.80, 0.15, 0.20);  // Rare red at top
    float colorT       = smoothstep(0.08, 0.30, elevation);
    float colorT2      = smoothstep(0.30, 0.50, elevation);
    vec3  auroraColor  = mix(auroraGreen, auroraPurple, colorT);
    auroraColor        = mix(auroraColor, auroraRed, colorT2 * 0.4);

    float intensity = curtain * altWeight * poleProximity * activity;
    intensity *= nightVisibility * (1.0 - cloudCover);

    return auroraColor * intensity * 0.18;
}

// ═══════════════════════════════════════════════════════════════════════
// LAYER: Alien Craft — mysterious lights traversing the night sky
// CPU-scheduled rare events with three distinct behaviour types.
// ═══════════════════════════════════════════════════════════════════════

vec3 renderAlienCraft(vec3 dir, float nightVisibility)
{
    if (nightVisibility < 0.2) return vec3(0.0);

    float craftType = pc.skyParams3.y;
    if (craftType < 0.5) return vec3(0.0);  // No event active

    float craftTimer = pc.skyParams3.x;
    float craftSeed  = pc.skyParams3.z;
    float time       = pc.cameraPos.w;
    int   iType      = int(craftType + 0.5);

    float eventDuration = 30.0;
    float t    = clamp(craftTimer / eventDuration, 0.0, 1.0);
    float fade = smoothstep(0.0, 0.06, t) * (1.0 - smoothstep(0.88, 1.0, t));
    if (fade < 0.001) return vec3(0.0);

    vec3 result = vec3(0.0);

    if (iType == 1) {
        // ── Type 1: Slow arc — single light glides across the sky ────
        vec3 startDir = normalize(vec3(
            hash11(craftSeed * 100.0 + 1.0) * 2.0 - 1.0,
            0.12 + hash11(craftSeed * 100.0 + 2.0) * 0.60,
            hash11(craftSeed * 100.0 + 3.0) * 2.0 - 1.0));
        vec3 endDir = normalize(vec3(
            hash11(craftSeed * 100.0 + 4.0) * 2.0 - 1.0,
            0.08 + hash11(craftSeed * 100.0 + 5.0) * 0.55,
            hash11(craftSeed * 100.0 + 6.0) * 2.0 - 1.0));
        vec3 craftDir = normalize(mix(startDir, endDir, smoothstep(0.0, 1.0, t)));

        float angDist = 1.0 - dot(dir, craftDir);
        float core = exp(-angDist / 0.000025) * 3.5;
        float glow = exp(-angDist / 0.00045)  * 0.55;

        float colorPulse = sin(time * 2.5 + craftSeed) * 0.5 + 0.5;
        vec3 craftColor  = mix(vec3(0.15, 1.0, 0.35), vec3(0.55, 0.80, 1.0), colorPulse);
        result = craftColor * (core + glow) * fade;
    }
    else if (iType == 2) {
        // ── Type 2: Formation — three lights in a rotating triangle ──
        vec3 baseDir = normalize(vec3(
            hash11(craftSeed * 100.0 + 1.0) * 2.0 - 1.0,
            0.18 + hash11(craftSeed * 100.0 + 2.0) * 0.50,
            hash11(craftSeed * 100.0 + 3.0) * 2.0 - 1.0));
        vec3 moveDir = normalize(vec3(
            hash11(craftSeed * 100.0 + 4.0) - 0.5,
            0.0,
            hash11(craftSeed * 100.0 + 5.0) - 0.5));
        vec3 center = normalize(baseDir + moveDir * t * 0.3);
        vec3 right  = normalize(cross(center, vec3(0.0, 1.0, 0.0)));
        vec3 up     = normalize(cross(right, center));

        for (int i = 0; i < 3; i++) {
            float angle  = float(i) * TAU / 3.0 + time * 0.35;
            float spread = 0.018 + 0.004 * sin(time * 1.8);
            vec3 offset  = right * cos(angle) * spread + up * sin(angle) * spread;
            vec3 craftDir = normalize(center + offset);

            float angDist = 1.0 - dot(dir, craftDir);
            float core = exp(-angDist / 0.000018) * 2.8;
            float glow = exp(-angDist / 0.00030)  * 0.35;

            result += vec3(0.25, 0.50, 1.0) * (core + glow) * fade;
        }
    }
    else {
        // ── Type 3: Erratic — approach, hover with jitter, zip away ──
        float p1 = 0.30, p2 = 0.62;
        vec3 startDir = normalize(vec3(
            hash11(craftSeed * 100.0 + 1.0) * 2.0 - 1.0,
            0.15 + hash11(craftSeed * 100.0 + 2.0) * 0.55,
            hash11(craftSeed * 100.0 + 3.0) * 2.0 - 1.0));
        vec3 hoverPos = normalize(vec3(
            hash11(craftSeed * 100.0 + 7.0) - 0.5,
            0.22 + hash11(craftSeed * 100.0 + 8.0) * 0.40,
            hash11(craftSeed * 100.0 + 9.0) - 0.5));
        vec3 escapeDir = normalize(vec3(
            hash11(craftSeed * 100.0 + 10.0) - 0.5,
            0.30 + hash11(craftSeed * 100.0 + 11.0) * 0.5,
            hash11(craftSeed * 100.0 + 12.0) - 0.5));

        vec3 craftDir;
        if (t < p1) {
            craftDir = normalize(mix(startDir, hoverPos, t / p1));
        } else if (t < p2) {
            vec3 jitter = vec3(
                sin(time * 7.3 + craftSeed) * 0.007,
                sin(time * 5.1 + craftSeed) * 0.004,
                sin(time * 8.7 + craftSeed) * 0.007);
            craftDir = normalize(hoverPos + jitter);
        } else {
            float zt = (t - p2) / (1.0 - p2);
            craftDir = normalize(hoverPos + escapeDir * zt * zt * zt * 0.6);
        }

        float angDist = 1.0 - dot(dir, craftDir);
        float core = exp(-angDist / 0.000025) * 4.0;
        float glow = exp(-angDist / 0.00055)  * 0.65;

        float pulseSpeed = (t > p1 && t < p2) ? 9.0 : 3.5;
        float colorPulse = sin(time * pulseSpeed + craftSeed) * 0.5 + 0.5;
        vec3 craftColor  = mix(vec3(1.0, 0.50, 0.08), vec3(1.0, 0.18, 0.08), colorPulse);
        result = craftColor * (core + glow) * fade;
    }

    return result * nightVisibility;
}

// ═══════════════════════════════════════════════════════════════════════
// LAYER: Comet — bright head with coma, dual tails (ion + dust)
// Appears once per year for ~4 in-game days. Procedural from seed.
// ═══════════════════════════════════════════════════════════════════════

vec3 renderComet(vec3 dir, float nightVisibility)
{
    if (nightVisibility < 0.08) return vec3(0.0);

    float seed         = floor(pc.skyParams2.z);
    float yearProgress = fract(pc.skyParams2.z);

    // One comet per year — visible for ~4 in-game days (10 % of 40-day year)
    float cometSeed   = hash11(seed * 3.14 + 42.0);
    float cometCenter = fract(cometSeed * 7.13);
    float cometHalf   = 0.05;
    float dist        = abs(yearProgress - cometCenter);
    if (dist > 0.5) dist = 1.0 - dist;
    if (dist > cometHalf) return vec3(0.0);

    float cometPhase = 1.0 - dist / cometHalf;
    float brightness = cometPhase * cometPhase * 3.5 + 0.4;

    // Sky position (high in sky, drifts slowly day to day)
    float cometAz = hash11(cometSeed * 13.7) * TAU;
    float cometEl = 0.28 + hash11(cometSeed * 17.3) * 0.42;
    cometAz += (yearProgress - cometCenter) * 2.5;

    vec3 cometDir = normalize(vec3(
        cos(cometAz) * cos(cometEl),
        sin(cometEl),
        sin(cometAz) * cos(cometEl)));

    float angDist = acos(clamp(dot(dir, cometDir), -1.0, 1.0));
    vec3  result  = vec3(0.0);

    // Bright nucleus
    float core = exp(-angDist * angDist / 0.0000001);
    result += vec3(1.0, 1.0, 0.95) * core * brightness * 14.0;

    // Coma (fuzzy envelope)
    float comaR = 0.005 + 0.003 * cometPhase;
    float coma  = exp(-angDist * angDist / (2.0 * comaR * comaR));
    result += vec3(0.92, 0.90, 0.78) * coma * brightness * 2.5;

    // Tail direction: away from the sun
    vec3 tailDir = normalize(cometDir - pc.sunDir.xyz);

    // Ion tail — straight, narrow, blue
    for (int i = 1; i <= 12; i++) {
        float tt   = float(i) * 0.012;
        vec3  pt   = normalize(cometDir + tailDir * tt);
        float d    = 1.0 - dot(dir, pt);
        float w    = 0.00006 * (1.0 + float(i) * 0.55);
        float tfade = 1.0 - float(i) / 12.0;
        result += vec3(0.22, 0.42, 1.0) * exp(-d / w) * tfade * tfade * brightness * 0.55;
    }

    // Dust tail — wider, yellowish, slightly curved from orbital motion
    vec3 dustCurve = normalize(cross(cometDir, vec3(0.0, 1.0, 0.0)));
    for (int i = 1; i <= 8; i++) {
        float tt     = float(i) * 0.010;
        vec3  curved = normalize(tailDir + dustCurve * tt * 0.45);
        vec3  pt     = normalize(cometDir + curved * tt);
        float d      = 1.0 - dot(dir, pt);
        float w      = 0.00018 * (1.0 + float(i) * 0.85);
        float tfade  = 1.0 - float(i) / 8.0;
        result += vec3(0.88, 0.72, 0.32) * exp(-d / w) * tfade * tfade * brightness * 0.30;
    }

    return result * nightVisibility;
}

// ═══════════════════════════════════════════════════════════════════════
// LAYER: Satellites — Titan debris orbiting the planet
// Several per night, fast-moving dots with occasional "Iridium" flares.
// ═══════════════════════════════════════════════════════════════════════

vec3 renderSatellites(vec3 dir, float nightVisibility)
{
    if (nightVisibility < 0.15) return vec3(0.0);

    float time = pc.cameraPos.w;
    float seed = floor(pc.skyParams2.z);
    vec3  result = vec3(0.0);

    // Up to 5 satellite tracks active at any given time
    for (int i = 0; i < 5; i++) {
        float cycle  = 75.0 + float(i) * 12.0;
        float offset = float(i) * cycle * 0.2;
        float binT   = time + offset;
        float bin    = floor(binT / cycle);
        float t      = fract(binT / cycle);

        float roll = hash11(bin * 13.7 + seed + float(i) * 73.0);
        if (roll < 0.45) continue;  // ~55 % chance per slot

        // Orbital arc across the sky
        float incl    = 0.20 + hash11(roll * 100.0 + 1.0) * 0.55;
        float startAz = hash11(roll * 100.0 + 2.0) * TAU;
        float trackAngle = t * PI;
        float az  = startAz + t * PI * 0.75;
        float el  = sin(trackAngle) * incl;

        vec3 satDir = normalize(vec3(cos(az), max(el, 0.01), sin(az)));
        if (satDir.y < 0.02) continue;

        float angDist = 1.0 - dot(dir, satDir);
        float point   = exp(-angDist / 0.000006) * 2.5;

        // Iridium-style flare (solar panel glint, very bright spike)
        float flareT    = hash11(bin * 77.0 + float(i) * 31.0);
        float flareDist = abs(t - flareT * 0.7 - 0.15);
        float flare     = exp(-flareDist * flareDist / 0.00025) * 8.0;

        float horizFade = smoothstep(0.02, 0.10, satDir.y);
        float entryFade = smoothstep(0.0, 0.05, t) * (1.0 - smoothstep(0.92, 1.0, t));

        vec3 col = vec3(0.92, 0.88, 0.78) * point
                 + vec3(1.0, 0.96, 0.92)  * flare;
        result += col * horizFade * entryFade;
    }

    return result * nightVisibility * 0.65;
}

// ═══════════════════════════════════════════════════════════════════════
// LAYER: Heartbeat Stars — prominent variable stars that pulse visibly
// Five distinct pulsating stars placed across the sky dome.
// ═══════════════════════════════════════════════════════════════════════

vec3 renderHeartbeatStars(vec3 dir, float starVis)
{
    if (starVis < 0.01) return vec3(0.0);

    float time = pc.cameraPos.w;
    float seed = floor(pc.skyParams2.z);
    vec3  result = vec3(0.0);

    for (int i = 0; i < 5; i++) {
        float az = hash11(seed * 0.71 + float(i) * 17.3) * TAU;
        float el = 0.15 + hash11(seed * 0.33 + float(i) * 23.7) * 0.60;
        vec3  starDir = normalize(vec3(cos(az), el, sin(az)));

        float angDist = 1.0 - dot(dir, starDir);
        if (angDist > 0.0008) continue;  // Early out

        float period = 2.2 + hash11(seed * 0.91 + float(i) * 37.0) * 5.5;
        float phase  = sin(time * TAU / period + float(i) * 1.73);
        float bright = 0.55 + 0.55 * phase;
        float sz     = 0.000020 + 0.000012 * max(phase, 0.0);
        float star   = exp(-angDist / sz) * 4.0;

        vec3 col;
        if      (i == 0) col = vec3(1.0,  0.38, 0.28);  // Red giant (Mira-type)
        else if (i == 1) col = vec3(0.65, 0.78, 1.0);   // Blue Cepheid
        else if (i == 2) col = vec3(1.0,  0.92, 0.38);  // Yellow eclipsing binary
        else if (i == 3) col = vec3(0.45, 0.95, 0.50);  // Green exotic pulsar
        else             col = vec3(0.88, 0.58, 1.0);   // Purple exotic

        result += col * star * bright;
    }

    return result * starVis * pc.skyParams.z;
}

// ═══════════════════════════════════════════════════════════════════════
// LAYER: Airglow — mesospheric oxygen emission bands at night
// Subtle green/red banded glow low in the sky with gravity-wave ripples.
// ═══════════════════════════════════════════════════════════════════════

vec3 renderAirglow(vec3 dir, float nightVisibility)
{
    if (nightVisibility < 0.25) return vec3(0.0);
    if (dir.y < 0.04 || dir.y > 0.45) return vec3(0.0);

    float time     = pc.cameraPos.w;
    float altitude = dir.y;
    float band     = smoothstep(0.04, 0.14, altitude) * (1.0 - smoothstep(0.22, 0.42, altitude));
    if (band < 0.01) return vec3(0.0);

    // Gravity-wave ripple pattern (slowly drifting)
    float azimuth = atan(dir.z, dir.x);
    float wave    = (sin(azimuth * 9.0 + time * 0.028) * 0.5 + 0.5)
                  * (sin(azimuth * 3.5 - time * 0.018 + 1.4) * 0.5 + 0.5);

    // Green oxygen 557.7 nm emission
    vec3 color = vec3(0.12, 0.22, 0.06);
    // High-altitude red 630 nm component
    color += vec3(0.06, 0.015, 0.015) * smoothstep(0.18, 0.38, altitude);

    return color * band * wave * nightVisibility * 0.035;
}

// ═══════════════════════════════════════════════════════════════════════
// LAYER: Cosmic Pulse — distant explosion / energy burst (very rare)
// Brief flash + expanding ring. ~12% chance per 4-minute window.
// ═══════════════════════════════════════════════════════════════════════

vec3 renderCosmicPulse(vec3 dir, float nightVisibility)
{
    if (nightVisibility < 0.3) return vec3(0.0);

    float time = pc.cameraPos.w;
    float seed = floor(pc.skyParams2.z);

    float cycle  = 240.0;
    float bin    = floor(time / cycle);
    float localT = fract(time / cycle) * cycle;

    float roll = hash11(bin * 23.7 + seed * 1.3);
    if (roll > 0.12) return vec3(0.0);

    float eventStart = roll * cycle * 0.5;
    float eventT     = localT - eventStart;
    if (eventT < 0.0 || eventT > 4.0) return vec3(0.0);

    // Event direction
    vec3 burstDir = normalize(vec3(
        hash11(bin * 11.0 + seed) * 2.0 - 1.0,
        0.15 + hash11(bin * 13.0 + seed) * 0.65,
        hash11(bin * 17.0 + seed) * 2.0 - 1.0));

    float angDist = acos(clamp(dot(dir, burstDir), -1.0, 1.0));

    // Phase 1: brilliant flash (0–0.3 s)
    float flash     = exp(-eventT * 10.0) * 6.0;
    float flashGlow = exp(-angDist * angDist / 0.002) * flash;

    // Phase 2: expanding ring of light (0.3–4 s)
    float ringRadius = eventT * 0.08;
    float ringWidth  = 0.006 + eventT * 0.003;
    float ringDist   = abs(angDist - ringRadius);
    float ring       = exp(-ringDist * ringDist / (ringWidth * ringWidth));
    ring *= smoothstep(0.3, 0.8, eventT) * (1.0 - smoothstep(2.5, 4.0, eventT));

    // Phase 3: residual warm glow at centre
    float residual = exp(-angDist * angDist / 0.0008) * exp(-eventT * 0.8);

    vec3 flashCol    = vec3(0.7, 0.8, 1.0)  * flashGlow;
    vec3 ringCol     = vec3(0.6, 0.75, 1.0) * ring * 1.5;
    vec3 residualCol = vec3(1.0, 0.6, 0.3)  * residual * 0.8;

    return (flashCol + ringCol + residualCol) * nightVisibility;
}

// ═══════════════════════════════════════════════════════════════════════
// Zodiacal Light — faint pyramid of light after sunset
// ═══════════════════════════════════════════════════════════════════════

vec3 renderZodiacalLight(vec3 dir, float sunElev)
{
    // Only visible during early twilight, near ecliptic plane
    float twilight = smoothstep(-0.15, -0.02, sunElev) * (1.0 - smoothstep(-0.02, 0.02, sunElev));
    if (twilight < 0.01) return vec3(0.0);

    // Elongated cone of light along ecliptic toward sun
    float sunAzDot = dot(normalize(vec3(dir.x, 0.0, dir.z)),
                         normalize(vec3(pc.sunDir.x, 0.0, pc.sunDir.z)));
    float nearSun  = max(sunAzDot, 0.0);
    nearSun        = pow(nearSun, 3.0);

    // Narrow in elevation, concentrated near ecliptic
    float ecliptic = exp(-dir.y * dir.y / 0.02) * smoothstep(0.0, 0.05, dir.y);

    vec3 zodColor = vec3(0.25, 0.18, 0.10);
    return zodColor * nearSun * ecliptic * twilight * 0.20;
}

// ═══════════════════════════════════════════════════════════════════════
// Sun Dogs (Parhelia) — ice-crystal refraction halos ±22° from the sun
// ═══════════════════════════════════════════════════════════════════════

vec3 renderSunDogs(vec3 dir, vec3 sunDir, float sunElev)
{
    // Parhelia only when sun is low (horizontally-oriented plate crystals)
    if (sunElev > 0.42 || sunElev < -0.02) return vec3(0.0);

    float lowSunFactor = smoothstep(0.42, 0.10, sunElev)
                       * smoothstep(-0.02, 0.02, sunElev);

    // Parhelion positions: 22° left and right of the sun in the horizontal plane
    vec3 sunHoriz = normalize(vec3(sunDir.x, 0.0, sunDir.z));
    vec3 sunRight = normalize(cross(sunHoriz, vec3(0.0, 1.0, 0.0)));

    vec3 rightParh = normalize(sunDir + sunRight * 0.404); // tan(22°)
    vec3 leftParh  = normalize(sunDir - sunRight * 0.404);

    float rightGlow = exp(-(1.0 - dot(dir, rightParh)) / 0.0008);
    float leftGlow  = exp(-(1.0 - dot(dir, leftParh))  / 0.0008);
    float parhelia  = rightGlow + leftGlow;

    // Chromatic dispersion — red inner edge, blue outer
    float cosAngle  = dot(dir, sunDir);
    float angFromSun = acos(clamp(cosAngle, -1.0, 1.0));
    float dispersion = (angFromSun - 0.384) / 0.04;

    vec3 innerColor = vec3(1.0, 0.75, 0.45);
    vec3 outerColor = vec3(0.65, 0.80, 1.0);
    vec3 dogColor   = mix(innerColor, outerColor, clamp(dispersion, 0.0, 1.0));

    // Stronger near the horizon where crystal alignment is best
    float horizWeight = exp(-dir.y * dir.y / 0.015);

    dogColor *= computeSunColor(pc.skyParams.x);

    return dogColor * parhelia * lowSunFactor * horizWeight * 0.8;
}

// ═══════════════════════════════════════════════════════════════════════
// Light Pillars — vertical ice-crystal reflection columns
// ═══════════════════════════════════════════════════════════════════════

vec3 renderLightPillars(vec3 dir, vec3 sunDir, float sunElev)
{
    if (sunElev > 0.35 || sunElev < -0.10) return vec3(0.0);

    float visStrength = smoothstep(0.35, 0.05, sunElev)
                      * smoothstep(-0.10, -0.02, sunElev);

    // Horizontal alignment with sun azimuth
    vec3 sunHoriz = normalize(vec3(sunDir.x, 0.0, sunDir.z));
    vec3 dirHoriz = normalize(vec3(dir.x, 0.0, dir.z));
    float horizAlign = dot(dirHoriz, sunHoriz);

    // Vertical extent above (and slightly below) the sun
    float aboveSun = dir.y - sunElev;
    float pillarHeight = exp(-aboveSun * aboveSun / 0.04)
                       * step(0.0, aboveSun + 0.05);

    // Pillar widens with height (spreading crystal column)
    float widthFalloff = max(1.0 + aboveSun * 2.0, 0.5);
    float pillarWidth  = exp(-(1.0 - horizAlign) / (0.0015 * widthFalloff));

    // Subtle vertical structure from ice-crystal variation
    float structure = 0.7 + 0.3 * noise2D(vec2(
        atan(dir.z, dir.x) * 50.0,
        dir.y * 20.0 + pc.cameraPos.w * 0.1
    ));

    vec3 pillarColor = computeSunColor(pc.skyParams.x) * vec3(1.0, 0.92, 0.85);

    // Toned down: a soft glow above the sun, not a bright column that reads as a
    // separate object (it was being mistaken for the galaxy at dusk).
    return pillarColor * pillarWidth * pillarHeight * structure * visStrength * 0.10;
}

// ═══════════════════════════════════════════════════════════════════════
// Noctilucent Clouds — high-altitude silvery twilight clouds (~80 km)
// ═══════════════════════════════════════════════════════════════════════

vec3 renderNoctilucentClouds(vec3 dir, float sunElev)
{
    // Visible only during deep twilight (sun 6–16° below horizon)
    float twilightBand = smoothstep(-0.28, -0.18, sunElev)
                       * (1.0 - smoothstep(-0.10, -0.04, sunElev));
    if (twilightBand < 0.01) return vec3(0.0);

    // Near-horizon strip where the long light path illuminates them
    float horizStrip = smoothstep(-0.01, 0.03, dir.y)
                     * (1.0 - smoothstep(0.08, 0.25, dir.y));
    if (horizStrip < 0.01) return vec3(0.0);

    // Preferentially visible toward the twilight glow (sun side)
    float sunAzDot = dot(normalize(vec3(dir.x, 0.0, dir.z)),
                         normalize(vec3(pc.sunDir.x, 0.0, pc.sunDir.z)));
    float sunSide  = smoothstep(-0.2, 0.6, sunAzDot);

    // Distinctive ripple / wave pattern of noctilucent clouds
    float time = pc.cameraPos.w;
    vec2 cloudUV = vec2(atan(dir.z, dir.x) * 12.0, dir.y * 80.0);

    float waves = (sin(cloudUV.x * 3.0 + time * 0.05) * 0.5 + 0.5)
                * (sin(cloudUV.y * 2.0 - time * 0.03 + cloudUV.x * 0.5) * 0.5 + 0.5);

    float ripples = noise2D(cloudUV * 4.0 + vec2(time * 0.02, 0.0));
    ripples = smoothstep(0.3, 0.6, ripples);

    float density = waves * ripples;

    // Silvery-blue with electric-blue edge closer to horizon
    vec3 nlcColor  = vec3(0.55, 0.70, 0.95);
    vec3 edgeColor = vec3(0.40, 0.55, 1.0);
    vec3 color     = mix(nlcColor, edgeColor, smoothstep(0.05, 0.02, dir.y));

    return color * density * horizStrip * sunSide * twilightBand * 0.15;
}

// ═══════════════════════════════════════════════════════════════════════
// Gegenschein — anti-solar interplanetary dust backscatter
// ═══════════════════════════════════════════════════════════════════════

vec3 renderGegenschein(vec3 dir, vec3 sunDir, float nightVis)
{
    if (nightVis < 0.5) return vec3(0.0);

    vec3  antiSun  = -sunDir;
    float cosAngle = dot(dir, antiSun);
    float glow     = exp(-(1.0 - cosAngle) / 0.015);
    float horizMask = smoothstep(0.0, 0.15, dir.y);

    return vec3(0.20, 0.16, 0.10) * glow * horizMask * nightVis * 0.06;
}

// ═══════════════════════════════════════════════════════════════════════
// Crepuscular Rays — volumetric light shafts through cloud gaps
// ═══════════════════════════════════════════════════════════════════════

vec3 renderCrepuscularRays(vec3 dir, vec3 sunDir, float cloudCover, float sunElev)
{
    if (cloudCover < 0.15 || cloudCover > 0.85) return vec3(0.0);
    if (sunElev > 0.25 || sunElev < -0.08) return vec3(0.0);

    float dawnDusk     = smoothstep(0.25, 0.08, sunElev)
                       * smoothstep(-0.08, 0.02, sunElev);
    float cloudFactor  = smoothstep(0.15, 0.35, cloudCover)
                       * (1.0 - smoothstep(0.65, 0.85, cloudCover));

    float cosAngle = dot(dir, sunDir);
    float angDist  = acos(clamp(cosAngle, -1.0, 1.0));
    if (angDist > 0.8) return vec3(0.0);

    float rayAtten = exp(-angDist * 2.5);

    // Radial cloud-gap bands emanating from the sun
    float azimuth    = atan(dir.z - sunDir.z, dir.x - sunDir.x);
    float seed       = pc.skyParams2.z;
    float rayPattern = 0.0;
    for (int i = 0; i < 5; i++) {
        float freq  = 4.0 + float(i) * 3.0;
        float phase = hash11(float(i) + seed) * TAU;
        float amp   = 1.0 / (1.0 + float(i));
        rayPattern += sin(azimuth * freq + phase + pc.cameraPos.w * 0.01) * amp;
    }
    rayPattern = smoothstep(0.3, 0.7, rayPattern * 0.5 + 0.5);

    vec3 rayColor = computeSunColor(pc.skyParams.x) * vec3(1.0, 0.90, 0.75);

    return rayColor * rayPattern * rayAtten * dawnDusk * cloudFactor * 0.12;
}

// ═══════════════════════════════════════════════════════════════════════
// Iridescent Clouds — thin-film interference colours near the sun
// ═══════════════════════════════════════════════════════════════════════

vec3 renderIridescentClouds(vec3 dir, vec3 sunDir, float cloudCover)
{
    if (cloudCover < 0.10 || cloudCover > 0.60) return vec3(0.0);
    if (pc.sunDir.y < 0.05) return vec3(0.0);

    float cosAngle = dot(dir, sunDir);
    float angDist  = acos(clamp(cosAngle, -1.0, 1.0));

    // Iridescence ring 3–15° from the sun
    float irisZone = smoothstep(0.05, 0.08, angDist)
                   * (1.0 - smoothstep(0.20, 0.30, angDist));
    if (irisZone < 0.01) return vec3(0.0);

    // Thin cloud patches (only partial density creates diffraction)
    vec2  cloudUV  = vec2(atan(dir.z, dir.x) * 8.0, dir.y * 30.0);
    float thinCloud = noise2D(cloudUV * 3.0
                     + vec2(pc.cameraPos.w * 0.008, pc.cameraPos.w * 0.005));
    thinCloud = smoothstep(0.35, 0.55, thinCloud)
              * (1.0 - smoothstep(0.55, 0.75, thinCloud));

    // Rainbow interference pattern (angle-dependent thin-film optics)
    float phase = angDist * 25.0 + noise2D(cloudUV * 5.0) * 2.0;
    vec3 irisColor = vec3(
        sin(phase)         * 0.5 + 0.5,
        sin(phase + 2.094) * 0.5 + 0.5,   // +120°
        sin(phase + 4.189) * 0.5 + 0.5    // +240°
    );
    irisColor = mix(vec3(0.8), irisColor, 0.4); // Pastel saturation

    return irisColor * thinCloud * irisZone * cloudCover * 0.08;
}

// ═══════════════════════════════════════════════════════════════════════
// Horizon Fog — smooth sky-to-terrain blending
// ═══════════════════════════════════════════════════════════════════════

vec3 horizonFog(vec3 skyColor, vec3 dir, float dayProgress, float sunElev)
{
    float horizFactor = 1.0 - max(dir.y, 0.0);
    horizFactor = pow(horizFactor, 8.0);

    // Fog colour depends on time of day and planet illumination
    vec3  dayFog  = vec3(0.62, 0.67, 0.77);
    vec3  duskFog = vec3(0.52, 0.30, 0.18);
    float planetPhaseH = pc.skyParams2.w;
    float planetOrbAngleH = planetPhaseH * TAU;
    float planetElevH = mix(-0.28, 0.08, sin(planetOrbAngleH) * 0.5 + 0.5);
    float planetVisH = clamp((planetElevH + 0.18) / 0.36, 0.0, 1.0);
    vec3  approxPDirH = normalize(vec3(cos(planetOrbAngleH), planetElevH, sin(planetOrbAngleH)));
    float planetIllumH = clamp((dot(approxPDirH, pc.sunDir.xyz) + 1.0) * 0.5, 0.0, 1.0) * planetVisH;
    vec3  nightFog  = vec3(0.002, 0.002, 0.005) + vec3(0.005, 0.005, 0.012) * planetIllumH;

    float duskAmt = smoothstep(0.0, 0.15, sunElev) * (1.0 - smoothstep(0.15, 0.4, sunElev));
    vec3  fogColor = mix(nightFog, mix(dayFog, duskFog, duskAmt), smoothstep(-0.10, 0.20, sunElev));

    return mix(skyColor, fogColor, horizFactor * 0.55);
}

// ═══════════════════════════════════════════════════════════════════════
// ACES Filmic Tone Mapping — cinematic colour
// ═══════════════════════════════════════════════════════════════════════

vec3 ACESFilm(vec3 x)
{
    float a = 2.51;
    float b = 0.03;
    float c = 2.43;
    float d = 0.59;
    float e = 0.14;
    return clamp((x * (a * x + b)) / (x * (c * x + d) + e), 0.0, 1.0);
}

// ═══════════════════════════════════════════════════════════════════════
// Weather Clouds — procedural cloud dome layer driven by weather system
// ═══════════════════════════════════════════════════════════════════════

/// Render weather-driven procedural clouds on a spherical dome.
///
/// Cloud appearance varies by type:
///   0 = Cumulus      — billowy, medium coverage
///   1 = Stratus      — flat, layered, wide
///   2 = Cirrus       — wispy, high altitude, fine detail
///   3 = Cumulonimbus — towering, dramatic, dense
///   4 = Nimbostratus — thick, dark, rain-bearing
///   5 = Fog          — low, diffuse (handled elsewhere)
///
/// Parameters arrive via skyParams4 (cloudType, cloudDensity, windStrength, cloudBaseAlt).
/// Wind direction comes from skyParams3.w.
///
/// The cloud color shifts with sun elevation, weather type, and time of day:
///   - Dawn/dusk: warm orange and pink tones
///   - Midday: bright white tops, darker bases
///   - Storm: dark grey/blue undersides
///   - Night: very dark, subtly lit by planet light
vec3 renderWeatherClouds(vec3 viewDir, vec3 sunDir, float dayProgress,
                         float cloudCover, float cloudType, float cloudDensity,
                         float windStrength, float windDirAngle, float cloudBaseAlt,
                         float time, float planetLight)
{
    // Skip if no clouds or looking below horizon
    if (cloudCover < 0.02 || cloudDensity < 0.01 || viewDir.y < 0.0)
        return vec3(0.0);

    // Cloud dome: ray-sphere intersection at normalized altitude
    // Map cloudBaseAlt to a dome elevation band
    float domeRadius = 15000.0;       // dome shell radius
    float baseNorm   = cloudBaseAlt / domeRadius;
    float layerThickness = 0.12;      // cloud layer vertical extent (normalized)

    // Intersection: viewDir hits the dome at this elevation
    float t = (baseNorm + 0.1) / max(viewDir.y, 0.001);
    vec3  hitPos = viewDir * t;

    // Wind-driven coordinate offset
    float windAngle = windDirAngle;
    vec2 windDir = vec2(cos(windAngle), sin(windAngle));
    float driftSpeed = 8.0 + windStrength * 40.0;
    vec2 drift = windDir * time * driftSpeed;

    // Sample coordinates on the dome. hitPos.xz is in normalized dome units
    // (~±1 across the visible sky), so the spatial scale must be ~O(1) to get
    // a handful of distinct clouds — the old 0.0004 sampled the noise at a
    // near-constant point, giving a flat featureless overcast. Wind drift is
    // kept slow and separate so clouds advect across the sky over ~tens of sec.
    vec2 uv = hitPos.xz * 0.7 + drift * 0.0004;

    // ── Shape noise — varies by cloud type ──────────────────────────

    float shape = 0.0;
    float detail = 0.0;
    // Fade clouds in just above the horizon (avoids a hard seam at the skyline)
    // and keep them at full strength all the way up to the zenith. The old
    // upper term smoothstep(0.85,0.4,...) zeroed clouds above ~24° elevation,
    // which left them only in a thin band around the horizon ("ring, none
    // above"). A cloud dome should be densest directly overhead.
    float verticalFade = smoothstep(0.03, 0.16, viewDir.y);

    if (cloudType < 0.5) {
        // Cumulus — billowy, rounded
        shape  = fbm(uv * 3.0, 5);
        detail = noise2D(uv * 24.0) * 0.15;
        shape  = shape * 0.8 + detail;
        // Cumulus has a rounded coverage ramp
        shape = smoothstep(0.45 - cloudCover * 0.35, 0.65, shape);
    }
    else if (cloudType < 1.5) {
        // Stratus — flat, layered
        shape  = fbm(uv * 2.0, 4);
        float layer = noise2D(uv * 8.0 + vec2(37.0, 19.0)) * 0.3;
        shape  = shape * 0.7 + layer * 0.3;
        // Stratus covers wider area with softer edges
        shape = smoothstep(0.30 - cloudCover * 0.30, 0.55, shape);
    }
    else if (cloudType < 2.5) {
        // Cirrus — wispy, stretched by wind
        vec2 stretchUV = uv * vec2(1.5, 4.0); // elongated along one axis
        shape  = fbm(stretchUV * 3.5, 4);
        float wisps = noise2D(stretchUV * 20.0) * 0.2;
        shape  = shape * 0.6 + wisps * 0.4;
        shape = smoothstep(0.50 - cloudCover * 0.25, 0.70, shape);
        // Cirrus is thinner
        shape *= 0.6;
    }
    else if (cloudType < 3.5) {
        // Cumulonimbus — towering, dramatic, very dense
        shape  = warpedFBM(uv * 2.5, 6);
        detail = fbm(uv * 12.0, 3) * 0.2;
        shape  = shape * 0.85 + detail * 0.15;
        shape = smoothstep(0.30 - cloudCover * 0.30, 0.55, shape);
        // Dense, darker base
        shape = min(shape * 1.3, 1.0);
    }
    else if (cloudType < 4.5) {
        // Nimbostratus — thick, uniform, dark
        shape  = fbm(uv * 1.8, 5);
        shape = smoothstep(0.25 - cloudCover * 0.25, 0.50, shape);
        shape = min(shape * 1.2, 1.0);
    }
    else {
        // Fog-layer clouds — very low, diffuse
        shape  = fbm(uv * 1.5, 3);
        shape = smoothstep(0.35 - cloudCover * 0.30, 0.55, shape);
        shape *= 0.5;
    }

    // Apply vertical fade (clouds thin toward horizon and zenith)
    shape *= verticalFade;

    // Apply overall density multiplier
    float alpha = clamp(shape * cloudDensity * 1.5, 0.0, 1.0);
    if (alpha < 0.005) return vec3(0.0);

    // ── Cloud lighting / coloring ───────────────────────────────────

    float sunElev = sunDir.y;

    // Base cloud color from sun lighting
    // Bright side: lit by sun; dark side: ambient sky light
    float sunDot = dot(viewDir, sunDir) * 0.5 + 0.5;
    float lightSide = smoothstep(0.3, 0.8, sunDot);

    // Daytime cloud color
    vec3 cloudBright = vec3(0.95, 0.95, 0.93);   // sunlit tops
    vec3 cloudDark   = vec3(0.45, 0.48, 0.55);   // shadowed bases

    // Dawn/dusk warm tinting
    float dawnDusk = smoothstep(0.0, 0.12, sunElev) * smoothstep(0.30, 0.12, sunElev);
    vec3 dawnColor  = vec3(1.0, 0.55, 0.25);     // warm orange
    vec3 duskColor  = vec3(0.95, 0.40, 0.35);     // pinkish red
    // Use dayProgress to distinguish dawn from dusk
    float isDawn = smoothstep(0.20, 0.30, dayProgress) * smoothstep(0.40, 0.30, dayProgress);
    float isDusk = smoothstep(0.65, 0.75, dayProgress) * smoothstep(0.85, 0.75, dayProgress);
    vec3 warmTint = mix(duskColor, dawnColor, isDawn / max(isDawn + isDusk, 0.001));
    cloudBright = mix(cloudBright, warmTint, dawnDusk * 0.7);
    cloudDark   = mix(cloudDark, warmTint * 0.5, dawnDusk * 0.5);

    // Storm darkening for Cumulonimbus and Nimbostratus
    float stormDarken = 0.0;
    if (cloudType > 2.5 && cloudType < 5.0) {
        stormDarken = smoothstep(2.5, 3.5, cloudType) * 0.5;
        cloudBright = mix(cloudBright, vec3(0.5, 0.5, 0.55), stormDarken);
        cloudDark   = mix(cloudDark, vec3(0.20, 0.22, 0.28), stormDarken);
    }

    // Night: clouds lit dimly by planetlight and ambient
    // Values aligned with sacred night pipeline (ambient 0.001–0.008)
    float nightFade = smoothstep(0.05, -0.10, sunElev);
    vec3 nightCloudColor = vec3(0.003, 0.004, 0.008) + vec3(0.008, 0.010, 0.018) * planetLight;
    cloudBright = mix(cloudBright, nightCloudColor, nightFade);
    cloudDark   = mix(cloudDark, nightCloudColor * 0.5, nightFade);

    // Composite lit and shaded portions
    vec3 cloudColor = mix(cloudDark, cloudBright, lightSide);

    // Silver lining — bright edge when looking toward the sun
    float silverLining = pow(max(dot(viewDir, sunDir), 0.0), 8.0) * (1.0 - nightFade);
    cloudColor += vec3(0.15, 0.12, 0.08) * silverLining * alpha;

    return cloudColor * alpha;
}

// ═══════════════════════════════════════════════════════════════════════
// Main
// ═══════════════════════════════════════════════════════════════════════

void main()
{
    vec3 viewDir = normalize(inViewRay);

    float dayProgress = pc.skyParams.x;
    float cloudCover  = pc.skyParams.y;
    float sunElev     = pc.sunDir.y;

    // ── Night visibility (layered transitions) ───────────────────────
    float planetPhaseM = pc.skyParams2.w;
    float planetOrbAngleM = planetPhaseM * TAU;
    float planetElevM = mix(-0.28, 0.08, sin(planetOrbAngleM) * 0.5 + 0.5);
    float planetVisM = clamp((planetElevM + 0.18) / 0.36, 0.0, 1.0);
    vec3  approxPDirM = normalize(vec3(cos(planetOrbAngleM), planetElevM, sin(planetOrbAngleM)));
    float planetIllumM = clamp((dot(approxPDirM, pc.sunDir.xyz) + 1.0) * 0.5, 0.0, 1.0) * planetVisM;

    // Bright stars first, then faint features — stars appear from early dusk
    float starVis      = smoothstep(0.10, -0.08, sunElev);
    float deepNightVis = smoothstep(0.0,  -0.15, sunElev);

    // Cloud dimming
    float cloudDim = 1.0 - cloudCover * 0.75;
    starVis      *= cloudDim;
    deepNightVis *= cloudDim;

    // Full planet washout of faintest stars (bright nearby planet dims faint stars)
    float planetWashout = 1.0 - planetIllumM * 0.35;
    starVis      *= planetWashout;
    deepNightVis *= planetWashout;

    // Composite nightVisibility for features that use the single float
    float nightVisibility = deepNightVis;

    // ═══ LAYER 1: Atmosphere ═════════════════════════════════════════
    vec3 sky = atmosphericScattering(viewDir, pc.sunDir.xyz, dayProgress);

    // ═══ LAYER 2: Sun Disk + Solar Phenomena ══════════════════════════
    // sunDir.w carries sun intensity pre-multiplied by terrain occlusion,
    // so all solar phenomena naturally fade when terrain blocks the sun.
    float sunVis = max(pc.sunDir.w, 0.0);
    if (sunElev > -0.05 && sunVis > 0.001) {
        sky += sunDisk(viewDir, pc.sunDir.xyz);
    }
    sky += renderSunDogs(viewDir, pc.sunDir.xyz, sunElev) * sunVis;
    sky += renderLightPillars(viewDir, pc.sunDir.xyz, sunElev) * sunVis;
    sky += renderCrepuscularRays(viewDir, pc.sunDir.xyz, cloudCover, sunElev) * sunVis;

    // ═══ LAYER 3: Nearby Planet (horizon-hugging, navigational) ═══════
    sky += renderPlanet(viewDir, nightVisibility);

    // ═══ LAYER 4: Stars + Variable Stars ══════════════════════════════
    sky += renderStars(viewDir, starVis, deepNightVis);
    sky += renderHeartbeatStars(viewDir, starVis);

    // ═══ LAYER 5: Deep Space (Galaxy + Nebulae) ═══════════════════════
    sky += renderGalaxy(viewDir, deepNightVis);
    sky += renderNebulae(viewDir, deepNightVis);

    // ═══ LAYER 6: Planetary Ring ══════════════════════════════════════
    sky += renderPlanetaryRing(viewDir, nightVisibility);

    // ═══ LAYER 7: Transient Events (Meteors, Comets, Craft, Satellites)
    sky += renderMeteors(viewDir, nightVisibility);
    sky += renderComet(viewDir, nightVisibility);
    sky += renderSatellites(viewDir, nightVisibility);
    sky += renderAlienCraft(viewDir, nightVisibility);
    sky += renderCosmicPulse(viewDir, nightVisibility);

    // ═══ LAYER 8: Atmospheric Phenomena ═══════════════════════════════
    sky += renderAurora(viewDir, nightVisibility);
    sky += renderNoctilucentClouds(viewDir, sunElev);
    sky += renderIridescentClouds(viewDir, pc.sunDir.xyz, cloudCover);
    sky += renderAirglow(viewDir, nightVisibility);

    // ═══ LAYER 8.5: Weather Clouds ═══════════════════════════════════
    // Procedural cloud dome driven by WeatherSystem + VolumetricClouds.
    // Renders after atmospheric phenomena so clouds occlude aurora/airglow,
    // but before dim phenomena so zodiacal light shines through thin patches.
    {
        float wCloudType    = pc.skyParams4.x;
        float wCloudDensity = pc.skyParams4.y;
        float wWindStrength = pc.skyParams4.z;
        float wCloudBaseAlt = pc.skyParams4.w;
        float wWindDirAngle = pc.skyParams3.w;
        float wTime         = pc.cameraPos.w;
        sky += renderWeatherClouds(viewDir, pc.sunDir.xyz, dayProgress,
                                   cloudCover, wCloudType, wCloudDensity,
                                   wWindStrength, wWindDirAngle, wCloudBaseAlt,
                                   wTime, planetIllumM);
    }

    // ═══ LAYER 9: Dim Phenomena ═══════════════════════════════════════
    sky += renderZodiacalLight(viewDir, sunElev);
    sky += renderGegenschein(viewDir, pc.sunDir.xyz, nightVisibility);

    // ═══ LAYER 10: Storm Effects ══════════════════════════════════════
    sky += renderLightning(viewDir, cloudCover);

    // ═══ LAYER 11: Horizon + Post-processing ═════════════════════════
    sky = horizonFog(sky, viewDir, dayProgress, sunElev);

    // ── Below horizon ────────────────────────────────────────────────
    if (viewDir.y < -0.01) {
        float fade = smoothstep(-0.01, -0.10, viewDir.y);
        vec3  belowHorizon = vec3(0.0005, 0.0005, 0.002) + vec3(0.003, 0.003, 0.005) * planetIllumM;
        sky = mix(sky, belowHorizon, fade);
    }

    // ── Output linear HDR ────────────────────────────────────────────
    // Tone mapping and gamma are applied once in the composite pass
    // (rt_composite.frag) to avoid double tone mapping in the HDR pipeline.
    // Night exposure adjustment is kept as a pre-tonemap intensity scale.
    float nightExposure = mix(1.1, 0.85, smoothstep(0.05, -0.12, sunElev));
    sky *= nightExposure;

    outColor = vec4(sky, 1.0);
}
