# Hybrid Forward+Compute Render Pipeline

> A novel rendering architecture combining forward geometry passes with compute-based screen-space lighting, designed for cinematic open-world games running on mid-range GPUs.

**Author:** [Stelliro](https://github.com/Stelliro)  
**Engine:** Custom C++20 / Vulkan 1.3  
**Status:** Prototype — proven in production on *Stelliferrum Forge* engine  
**License:** CC BY-SA 4.0 (see [LICENSE](LICENSE))

---

## Table of Contents

- [Why This Exists](#why-this-exists)
- [Architecture Overview](#architecture-overview)
- [Phase 1 — Forward Geometry](#phase-1--forward-geometry)
- [Phase 2 — Compute Post-Processing](#phase-2--compute-post-processing)
  - [Hi-Z Pyramid Build](#1-hi-z-pyramid-build)
  - [GTAO+ (Dual-Horizon Multi-Scale AO)](#2-gtao-dual-horizon-multi-scale-ao)
  - [SVRM (Sparse Voxel Radiance Marching)](#3-svrm-sparse-voxel-radiance-marching)
  - [Hi-Z Screen-Space Reflections](#4-hi-z-screen-space-reflections)
  - [Temporal Denoise](#5-temporal-denoise)
  - [Composite](#6-composite--tone-mapping)
- [World-Anchored Volumetric God Rays](#world-anchored-volumetric-god-rays)
- [Biome-Aware Per-Pixel Lighting](#biome-aware-per-pixel-lighting)
- [Procedural Skybox Architecture](#procedural-skybox-architecture)
- [Orientation Anchor System (6-Point Surface Capture)](#orientation-anchor-system-6-point-surface-capture)
- [Shadow Mapping — Light-Frustum Culling](#shadow-mapping--light-frustum-culling)
- [Single Tone-Mapping Point](#single-tone-mapping-point)
- [Performance Budget](#performance-budget)
- [Implementation Guide](#implementation-guide)
- [Pitfalls & Lessons Learned](#pitfalls--lessons-learned)
- [Citation](#citation)

---

## Why This Exists

Traditional deferred rendering is great for many lights but wastes bandwidth writing G-buffers for outdoor scenes dominated by a single directional light. Pure forward rendering is simple but lacks screen-space effects. Ray tracing requires hardware most players don't have.

This pipeline takes a different approach: **render geometry forward into HDR offscreen buffers, then run sophisticated compute shaders for AO, GI, reflections, and volumetrics on those buffers**. You get the simplicity and cache coherency of forward rendering with the visual quality of deferred/compute lighting — all on a GTX 1060 class GPU at 60 FPS.

Built on the **Vulkan 1.3** graphics API.

---

## Architecture Overview

```
┌─────────────────── PHASE 1: FORWARD GEOMETRY ───────────────────┐
│                                                                   │
│   Shadow Pass (D32F)                                              │
│       │                                                           │
│       ▼                                                           │
│   Scene Pass ──► Offscreen HDR (RGBA16F)                         │
│     ├─ Skybox (fullscreen triangle)      ──► Color Buffer         │
│     ├─ Terrain (triplanar, biome-lit)    ──► Depth Buffer (D32F)  │
│     └─ Objects (vertex-colored, shadowed)──► Normal Buffer (RGB16F)│
│                                                                   │
├─────────────────── PHASE 2: COMPUTE LIGHTING ───────────────────┤
│                                                                   │
│   Hi-Z Pyramid Build (max-reduction, 6 mips)                     │
│       │                                                           │
│       ├──► GTAO+ (dual-horizon AO + bent normals)                │
│       ├──► SVRM  (sparse probe-grid GI)                          │
│       ├──► Hi-Z Reflections (specular ray march)                 │
│       │                                                           │
│       ▼                                                           │
│   Temporal Denoise (motion-vector reprojection)                   │
│       │                                                           │
│       ▼                                                           │
│   Composite (energy-conserving blend + ACES tone map + post-FX)  │
│       │                                                           │
│       ▼                                                           │
│   Swapchain (LDR sRGB)                                           │
└───────────────────────────────────────────────────────────────────┘
```

### Key Design Decisions

| Decision | Rationale |
|---|---|
| Forward geometry, not deferred | Single directional sun means G-buffer bandwidth is wasted. Forward is simpler, fewer render targets. |
| Offscreen HDR intermediate | Full-precision data for compute passes. Tone mapping applied once at the end. |
| Compute for screen-space effects | Async compute overlap with next frame's CPU work. Each effect is an independent dispatch. |
| Single tone-mapping point | Prevents double-mapping artifacts (a real bug we shipped and fixed). |
| Sparse probe grid for GI | 64 probes instead of per-pixel traces. 0.06 rays/pixel/frame = real-time on low-end. |

---

## Phase 1 — Forward Geometry

All geometry shaders output **linear HDR** color. No tone mapping, no gamma correction in geometry passes. This is critical — the composite pass handles all color space conversion.

### Render Targets

| Buffer | Format | Purpose |
|---|---|---|
| Color | RGBA16F | Scene radiance in linear HDR |
| Depth | D32_SFLOAT | Depth for compute ray marching |
| Normal | RGB16F | World-space normals for AO/GI |
| Motion | RG16F | Per-pixel motion vectors for temporal |

### Shadow Pass

- Orthographic projection from sun direction
- Depth-comparison sampler enables hardware PCF
- Front-face culling reduces self-shadowing artifacts
- **Light-frustum culling**: Test world geometry (e.g. chunk AABBs) against the light's view-projection frustum. Only objects visible from the light's POV render to the shadow map. This typically reduces shadow draw calls by **85–95%** vs. rendering everything.

### Skybox

Rendered as a fullscreen triangle (not a cube — single draw call, no vertex buffer). The fragment shader computes everything procedurally. See [Procedural Skybox Architecture](#procedural-skybox-architecture).

---

## Phase 2 — Compute Post-Processing

Each effect is an independent compute dispatch. All operate on the offscreen HDR buffers from Phase 1.

### 1. Hi-Z Pyramid Build

**Purpose:** Accelerate all ray marching (AO, GI, reflections) from O(n) to O(log n).

**Algorithm:**
1. Start from full-resolution depth buffer (mip 0)
2. Each mip level: sample 2×2 texels from previous mip, store **maximum** depth
3. Build 6 mip levels (e.g. 1920×1080 → 30×17)
4. Ray marching starts at coarse mip, refines to fine mip on hit

**Dispatch:** `ceil(width / 8) × ceil(height / 8)` per mip level, chained with barriers.

### 2. GTAO+ (Dual-Horizon Multi-Scale AO)

**Novel contribution: simultaneous dual-radius AO with bent normal output.**

Traditional GTAO traces horizon angles in screen space. GTAO+ extends this with three innovations:

#### Dual-Radius Simultaneous Tracing

```
For each pixel:
    For each angular slice (e.g. 8 slices):
        Trace SHORT-range horizon (4 texels)  → crisp contact shadows
        Trace LONG-range horizon  (24 texels) → room-scale occlusion
        
    shortAO = integrateHorizons(shortHorizons, normal)
    longAO  = integrateHorizons(longHorizons, normal)
    finalAO = shortAO * longAO   // multiplicative combination
```

**Why multiplicative?** Additive loses contact shadows when long-range is fully occluded. Multiplicative preserves both scales independently.

#### Bent Normal Computation

While tracing horizon angles, accumulate the average **unoccluded direction**:

```
bentNormal = vec3(0)
For each slice:
    unoccludedArc = PI - (horizon_left + horizon_right)
    midAngle = (horizon_left - horizon_right) / 2
    bentNormal += sliceDirection * sin(midAngle) * unoccludedArc
bentNormal = normalize(bentNormal)
```

The bent normal replaces the geometric normal when sampling indirect lighting (GI). Objects receive light from the direction that's actually open, not the surface facing direction. This dramatically improves light leaking in crevices.

#### Slope-Adaptive Angular Sampling

Instead of uniform angular distribution for horizon traces, concentrate samples near the **horizon angle** where occlusion changes most rapidly:

```
For i in [0, numSteps):
    // Non-linear: more samples near horizon, fewer overhead
    t = (i + 0.5) / numSteps
    sampleAngle = t * t * maxAngle  // quadratic concentration
```

This yields approximately **30% quality improvement** at the same sample count vs. uniform stepping.

**Output:** RGBA16F — R = AO factor, GBA = bent normal (world space).

### 3. SVRM (Sparse Voxel Radiance Marching)

**Novel contribution: probe-grid screen-space GI with depth-gap bridging.**

Full per-pixel screen-space GI is too expensive for real-time (hundreds of rays per pixel). SVRM solves this with a three-phase approach that achieves plausible indirect lighting from **0.06 rays per pixel per frame**.

#### Phase A — Radiance Probe Tracing

Divide the screen into an **N×N probe grid** (e.g. 8×8 = 64 probes at 1080p, each covering ~135×135 pixels):

```
For each probe:
    centerPixel = probeGridPosition * (screenSize / gridSize)
    
    For each ray (e.g. 4 rays per probe):
        direction = cosineWeightedHemisphere(normal, blueNoiseJitter)
        
        // March through Hi-Z pyramid
        hit = hierarchicalRayMarch(centerPixel, direction, hiZBuffer)
        
        if hit:
            radiance = sampleSceneColor(hitPixel)
            probeResult += radiance * cosTheta / pdf
```

**Key details:**
- **Cosine-weighted importance sampling**: More rays near the normal, matching the BRDF
- **Blue-noise temporal jitter**: Each frame rotates ray directions by the golden angle (~137.5°). Over 8 frames, covers the hemisphere uniformly. Eliminates structured aliasing.
- **Hierarchical Z-march**: Start at coarse mip for fast traversal, refine on intersection

#### Phase B — Depth-Gap Bridging (DGB)

The **#1 limitation** of screen-space methods: when a ray hits a depth discontinuity (edge of an object), there's nothing behind it — the data simply isn't on screen.

DGB addresses this:

```
When ray hits depth discontinuity (|depthDelta| > threshold):
    // Don't discard — extrapolate
    neighborColors = sampleNearbyPixels(hitPosition, 3x3 kernel)
    neighborDepths = sampleDepths(hitPosition, 3x3 kernel)
    
    // Weight by depth similarity to the FAR side of the gap
    weights = exp(-abs(neighborDepths - expectedDepth) * depthSensitivity)
    
    extrapolatedColor = weightedAverage(neighborColors, weights)
    confidence *= 0.5  // Lower confidence for extrapolated data
```

This doesn't perfectly reconstruct occluded surfaces, but it produces plausible color estimates that are **far better than discarding the ray entirely** (which causes dark halos around objects).

#### Phase C — Radiance Propagation

Scatter probe results to all pixels within the probe's region:

```
For each pixel:
    // Find 4 nearest probes (bilinear in probe grid)
    probes = findNearestProbes(pixelPosition, probeGrid)
    
    indirect = vec3(0)
    totalWeight = 0
    
    For each probe:
        // Depth-aware bilateral weight
        depthWeight = exp(-abs(probeDepth - pixelDepth) * depthSigma)
        normalWeight = max(0, dot(probeNormal, pixelNormal))
        distWeight = bilinearWeight(pixelPosition, probePosition)
        
        w = depthWeight * normalWeight * distWeight
        indirect += probe.radiance * w
        totalWeight += w
    
    indirect /= max(totalWeight, 0.001)
    
    // Temporal accumulation
    prevIndirect = reproject(motionVector, previousGI)
    finalGI = mix(indirect, prevIndirect, 0.9)  // Heavy temporal = stability
```

**Output:** RGBA16F — RGB = indirect illumination, A = confidence.

**Why "Sparse Voxel" in the name?** The probe grid conceptually voxelizes screen space into a sparse grid, and radiance is marched (not traced) through the depth hierarchy. The name distinguishes it from dense per-pixel approaches.

### 4. Hi-Z Screen-Space Reflections

- **Input:** Scene color, depth, normals, Hi-Z pyramid
- **Algorithm:** Per-pixel specular ray march through Hi-Z mip chain
- **Roughness modulation:** Rough surfaces march shorter distances (fewer artifacts from screen-space limitations)
- **Conservative hit test:** Compare against max-depth (from Hi-Z) for safe underestimate — prevents false reflections through thin geometry
- **Fallback:** Rays that exit screen space or exceed max march distance return zero (environment map fallback is an option)

### 5. Temporal Denoise

All screen-space effects produce noisy 1-sample-per-pixel results. Temporal accumulation stabilizes them:

```
currentSample = currentFrameEffect(pixel)
previousSample = samplePreviousFrame(pixel + motionVector)

// Neighborhood clamp to prevent ghosting
neighborhoodMin, neighborhoodMax = computeNeighborhoodBounds(pixel, 3x3)
previousSample = clamp(previousSample, neighborhoodMin, neighborhoodMax)

confidence = computeConfidence(motionVector, depthDelta)
blendFactor = mix(0.05, 0.2, 1.0 - confidence)  // High confidence = keep history

output = mix(previousSample, currentSample, blendFactor)
```

**Motion vectors** are computed per-pixel from the difference between current and previous frame's world-space positions projected to screen space.

### 6. Composite & Tone Mapping

The final fullscreen pass blends all effects with energy conservation:

```
albedo       = sampleSceneColor(pixel)
ao           = sampleAO(pixel).r
bentNormal   = sampleAO(pixel).gba
gi           = sampleGI(pixel).rgb
giConfidence = sampleGI(pixel).a
reflections  = sampleReflections(pixel).rgb

// Energy-conserving combination
ambientGI = gi * giConfidence * max(0, dot(bentNormal, lightDir))
finalColor = albedo * (ambientGI + directLight) * ao + reflections

// Post-processing (applied in this order)
finalColor *= exposure
finalColor = ACESFilmicToneMap(finalColor)   // [0,∞) → [0,1]
finalColor = linearToSRGB(finalColor)         // Gamma correction
finalColor = applyVignette(finalColor, uv)
finalColor = applyChromaticAberration(finalColor, uv)
finalColor = applyFilmGrain(finalColor, time) // MUST be post-tone-map
```

**Critical:** Film grain must be applied **after** tone mapping. Applying noise in linear HDR space produces non-perceptual noise distribution (bright areas get disproportionately noisy).

---

## World-Anchored Volumetric God Rays

Traditional screen-space god rays slide across the scene when the camera rotates because they're computed in screen space relative to the sun's projected position. This pipeline takes a different approach.

### Concept: Camera-Ray Shadow Segmentation

For each pixel's view ray, divide it into **lit and unlit segments** using the shadow map:

```
For each pixel:
    rayOrigin = cameraPosition
    rayEnd    = worldPosition(pixel)
    
    totalScattering = vec3(0)
    
    For step in [0, numSteps):
        samplePos = mix(rayOrigin, rayEnd, step / numSteps)
        
        // Transform to shadow map space
        shadowUV = lightViewProj * vec4(samplePos, 1.0)
        
        // Is this point in light or shadow?
        inLight = sampleShadowMap(shadowUV) > shadowUV.z
        
        if inLight:
            // Accumulate in-scattered light
            phase = HenyeyGreenstein(dot(viewDir, lightDir), g=0.72)
            density = heightExponentialFog(samplePos.y)
            totalScattering += sunColor * phase * density * stepSize
    
    // Blend with scene color
    finalColor = sceneColor * extinction + totalScattering
```

**Why this works:** The shadow map is in **world space**. Shadowed regions (behind trees, buildings, mountains) contribute zero scattering. Lit regions contribute positive scattering. The resulting shafts of light are anchored to the world geometry that casts the shadows — they **do not move when the camera rotates**.

### Implementation Notes

- **Froxel volume** (e.g. 160×90×64) is more efficient than per-pixel marching
- **Henyey-Greenstein phase function** with forward-scattering bias (g ≈ 0.7) makes god rays brightest when looking toward the sun
- **Height-exponential density** prevents fog from being uniform — valleys are foggier than mountaintops
- Local volumetric sources (campfires, explosions) inject density into nearby froxels

---

## Biome-Aware Per-Pixel Lighting

Instead of a global ambient/fog setting, each biome defines **day and night environment profiles**:

| Parameter | Description | Example Range |
|---|---|---|
| `ambientTint` | RGB color shift to ambient light | Warm for grasslands, green for forests |
| `ambientScale` | Brightness multiplier | 0.5 (dark forest) → 1.0 (open plains) |
| `fogDensityMul` | Fog thickness modifier | 0.5 (clear desert) → 2.0 (swamp) |
| `exposureBias` | EV adjustment for tone mapping | -1.0 (bright snow) → +0.5 (dark cave) |
| `shadowDepth` | Shadow darkness multiplier | 1.0 (normal) → 1.5 (deep forest shadows) |

**Per-pixel blending:** At each fragment, the biome profile is interpolated based on the world-space position. Biome boundaries blend smoothly over a transition zone. Day/night profiles cross-fade based on sun elevation:

```
nightBlend = smoothstep(0.05, -0.12, sunElevation)  // [0,1]
profile    = mix(biome.dayProfile, biome.nightProfile, nightBlend)
```

This means a forest at sunset gradually darkens with deep green-tinted shadows, while an adjacent grassland retains warm amber light — all computed **per fragment**, not per chunk or per biome zone.

---

## Procedural Skybox Architecture

The skybox is rendered as a single fullscreen triangle with zero texture lookups. Everything is computed from noise functions and analytic models. The layering architecture enables complex atmospheric scenes:

| Layer | Contents | Key Technique |
|---|---|---|
| 1 | Atmosphere (Rayleigh/Mie scattering) | Multi-scatter approximation, horizon color bands |
| 2 | Solar phenomena (sun disk, corona, sun dogs) | Airy disk diffraction, parhelia at ±22° |
| 3 | Celestial body (planet/moon with surface detail) | Procedural continents, clouds, city lights at night |
| 4 | Star field (spectral colors, scintillation) | Hash-based placement, atmospheric twinkling |
| 5 | Deep space (galaxy band, nebulae, dust lanes) | FBM noise, emission regions |
| 6 | Planetary ring system | Phase-dependent forward scattering, gap modeling |
| 7 | Transient events (meteors, comets, satellites) | Timer-driven spawning, trail rendering |
| 8 | Rare atmospheric (aurora, noctilucent clouds) | Curtain wave functions, mesospheric modeling |

**Design principle:** Each layer composites onto previous layers with proper alpha handling. Layers are evaluated bottom-to-top (atmosphere first, transients last). Horizon fog at the bottom seamlessly blends skybox into terrain.

---

## Orientation Anchor System (6-Point Surface Capture)

**Novel contribution: motion-capture-style marker points that tell the renderer exactly which surfaces are camera-visible and how much of each object to render.**

In a large open world with thousands of objects (trees, rocks, enemies, structures), rendering full geometry for everything is impossible. LOD systems replace distant objects with lower-poly meshes or flat billboard impostors. But traditional impostors are **orientation-blind** — they assume every object is upright. A fallen tree, a tumbling rock, or a ragdolling enemy renders incorrectly because the billboard doesn't know the object's actual 3D orientation.

The Orientation Anchor System (OAS) solves this by embedding a small number of **"motion capture markers"** — fixed reference points — into every renderable object. By knowing where these markers are in world space, the renderer reconstructs the object's full orientation and determines **exactly which surfaces face the camera**, rendering only what's needed.

### Core Concept: 3–6 Anchor Points = Full Orientation

Every object defines **3 mandatory anchor points** (minimum) in its local coordinate space:

| Anchor | Role | Typical Placement |
|---|---|---|
| **Origin (A₀)** | Object base / ground contact | Bottom center (feet, trunk base, rock bottom) |
| **Up (A₁)** | Defines the object's vertical axis | Top of object (treetop, head, rock peak) |
| **Forward (A₂)** | Defines the object's front facing | Front face offset (thickest branch side, chest, largest face) |

From these 3 world-space positions, the system reconstructs the **full rotation matrix**:

```
up      = normalize(A₁_world - A₀_world)                    // Object's up axis
forward = normalize(A₂_world - midpoint(A₀_world, A₁_world)) // Object's forward
right   = normalize(cross(up, forward))                      // Derived right axis
forward = cross(right, up)                                    // Re-orthogonalize
R       = [right | up | forward]                              // Object → world rotation
```

For articulated or complex objects, **up to 6 additional anchors** can track independent parts:

| Anchor | Example Purpose |
|---|---|
| **A₃** | Canopy center (independent orientation after tree felling) |
| **A₄** | Heaviest branch tip (canopy tilt from wind) |
| **A₅** | Root center (exposure during terrain destruction) |

Maximum: **8 anchors per object** (128 bytes per entity in GPU storage).

### How It Saves Performance

Once the renderer knows an object's full orientation relative to the camera, it can make intelligent decisions:

#### 1. Impostor Atlas: Render Only the Visible Face

Instead of a simple 8-direction billboard (N/NE/E/SE/S/SW/W/NW), the system uses an **octahedron-mapped impostor atlas** — pre-rendered views from directions distributed uniformly on a sphere:

```
// Convert camera-to-object direction into the object's local frame
D_local = transpose(R) * normalize(objectPos - cameraPos)

// Map to octahedron UV (uniform angular coverage, no polar distortion)
vec2 octahedronUV(vec3 d) {
    d /= (abs(d.x) + abs(d.y) + abs(d.z));       // Project onto octahedron
    if (d.y < 0.0)
        d.xz = (1.0 - abs(d.zx)) * sign(d.xz);  // Fold lower hemisphere
    return d.xz * 0.5 + 0.5;                      // [0,1] UV range
}
```

The octahedron UV tells the renderer **exactly which pre-rendered face to display**. Only the 4 atlas tiles needed for bilinear interpolation are sampled — the other 12–32 tiles are never touched.

#### 2. Roll Correction: Fallen Objects Render Correctly

When an object is tilted or lying on its side (felled tree, rolling boulder, ragdolled enemy), the billboard quad itself rotates in screen space:

```
rollAngle = atan2(R[0][1], R[1][1])  // Extract roll from orientation matrix
// Billboard vertex positions rotated by rollAngle in screen space
```

A fallen tree's impostor actually appears **sideways** — not standing upright with a wrongly-rotated texture.

#### 3. Normal Correction: Lighting Stays Accurate

Each impostor atlas entry bakes per-pixel normals from its rendered viewing angle. When the object's actual orientation differs from the baked direction (due to tilt/roll), the normals are corrected:

```
// In impostor fragment shader:
bakedNormal = texture(normalAtlas, uv).xyz * 2.0 - 1.0
correction  = mat3(R_actual) * inverse(mat3(R_baked))
worldNormal = correction * bakedNormal
// Use worldNormal for standard lighting — a sideways tree lights correctly
```

#### 4. Proactive Texture Streaming: Load Only What's Visible

The visible hemisphere of each object (determined by camera direction relative to the orientation matrix) identifies which atlas tiles are needed. The system maintains a **virtual texture tile pool** with LRU eviction:

- Each frame: compute the 4 tiles needed per visible LOD 2+ object
- Check residency — if tiles are in the pool, render immediately
- Missing tiles are loaded with priority based on: distance to camera, screen coverage, angular velocity
- **Angular velocity pre-fetch**: for rotating/moving objects, predict which tiles will be needed next frame and load them one frame ahead, preventing tile pops

```
predictedUV = octahedronUV(D_local + angularVelocity * dt)
```

### Performance Impact

The key insight is that **anchors are only processed for LOD 2+ objects** (distant impostors). LOD 0–1 objects have full geometry and skip the anchor system entirely.

| Operation | Cost | When |
|---|---|---|
| Anchor transform (3× mat4×vec4) | ~30 ns/object | LOD 2–3 only |
| Orientation reconstruction | ~20 ns/object | LOD 2–3 only |
| Octahedron UV lookup | ~5 ns/object | LOD 2–3 only |
| Normal correction matrix | ~10 ns/object | LOD 2–3 only |
| **Total per object** | **~65 ns** | **Per frame** |
| 750 distant objects (trees + shrubs + enemies) | **~49 μs** | **Per frame total** |
| Tile residency + streaming | ~100 μs/frame | Amortized |

**Total CPU cost: ~0.15 ms/frame** — negligible. GPU cost is **zero additional draw calls** — the atlas UV calculation and normal rotation happen in the existing impostor shader.

**Net savings**: By only loading/rendering the visible face of each impostor (instead of the full atlas), VRAM usage for distant objects drops by ~75%, and the impostor shader reads 4 tiles instead of blending 8+ directions. Combined with the orientation-correct lighting, **distant objects look better and cost less**.

### Implementation Checklist

1. **Define anchors per object type** — 3 mandatory points (origin, up, forward) placed at meaningful positions in local space
2. **Auto-infer for existing assets** — origin = bbox min-Y center, up = bbox max-Y center, forward = bbox center offset toward +Z
3. **Per-frame anchor transform** — multiply local anchors by model matrix for LOD 2+ objects only
4. **Orientation reconstruction** — cross-product based, re-orthogonalized
5. **Octahedron atlas generation** — pre-render 16–36 views per object type (offline bake)
6. **Fragment shader** — octahedron UV lookup, bilinear blend, normal correction
7. **Roll correction** — rotate billboard quad vertices by extracted roll angle
8. **Tile pool** — virtual texture with LRU eviction, angular velocity pre-fetch

### What This Enables

- **Physics-felled trees** render correctly as impostors at any angle, tumbling during fall
- **Enemy ragdolls** transition to impostor at distance with correct twisted-body orientation  
- **Tumbling debris** from destruction events render accurately mid-flight
- **Slope-conforming objects** (trees on hillsides, leaning structures) display the correct tilt
- **Proactive streaming** prevents tile pop-in during camera orbits

The combination of surface-aware rendering + orientation-correct lighting + selective texture streaming makes this the **performance backbone** of the hybrid pipeline — it's how thousands of objects stay on screen at 60 FPS without sacrificing visual quality at distance.

---

## Shadow Mapping — Light-Frustum Culling

Standard shadow mapping renders all geometry from the light's perspective. For large open worlds with many terrain chunks, this is wasteful — most chunks are behind the light or outside its projection.

### Technique

1. Compute the light's orthographic view-projection matrix (covers player area ± some extent)
2. **Invert** the light VP to get the light's frustum planes
3. Before rendering each chunk to the shadow map, test its AABB against the light frustum
4. Skip chunks that fail the frustum test

```
lightVP = ortho(-extent, extent, -extent, extent, near, far) * lookAt(lightPos, lightDir)
frustumPlanes = extractFrustumPlanes(inverse(lightVP))

For each chunk:
    if !testAABBFrustum(chunk.bounds, frustumPlanes):
        continue  // Skip — not visible from light's perspective
    renderToShadowMap(chunk)
```

**Result:** Typically **85–95% fewer draw calls** in the shadow pass. With 200+ terrain chunks loaded, only 5–15 are visible from the sun's perspective.

### Additional Optimization

Force all shadow-pass geometry to use the lowest LOD level. Shadow map resolution is low enough that high-detail meshes don't improve quality but cost significantly more in vertex processing.

---

## Single Tone-Mapping Point

**Critical architectural rule:** All shaders output linear HDR values. Tone mapping and gamma correction happen **exactly once**, in the composite pass.

### Why This Matters

If geometry shaders apply their own tone mapping (a common mistake), you get:
1. **Double tone mapping** — composite applies ACES again on already-mapped values → washed out, low contrast
2. **Inconsistent sky/scene** — skybox in different color space than terrain
3. **Effect artifacts** — screen-space effects operating on tone-mapped (non-linear) data produce incorrect results

### Implementation

```
// WRONG — do NOT do this in geometry shaders:
fragColor = ACESToneMap(lighting);  // ← No! This is linear HDR space

// CORRECT — geometry shaders output linear HDR:
fragColor = lighting;  // Linear HDR, may exceed 1.0, that's fine

// Composite pass (and ONLY the composite pass):
finalColor = ACESFilmicToneMap(linearHDR * exposure);
finalColor = pow(finalColor, vec3(1.0/2.2));  // Linear → sRGB
```

---

## Performance Budget

Target: **60 FPS (16.7ms frame time)** on GTX 1060 / RX 580 class hardware at 1080p.

| Pass | Target | Notes |
|---|---|---|
| Shadow | 0.5 ms | Light-frustum culled, lowest LOD |
| Skybox | 0.2 ms | Single fullscreen triangle, no textures |
| Terrain | 2.0 ms | 20–40 chunks, triplanar texturing |
| Objects | 0.5 ms | Player, vegetation, world objects |
| Hi-Z Build | 0.2 ms | 6-level max-reduction |
| **GTAO+** | **0.8 ms** | Dual-radius, 8 slices, bent normals |
| **SVRM** | **2.0 ms** | 64 probes × 4 rays + propagation |
| Reflections | 0.5 ms | Hi-Z ray march, roughness-limited |
| Denoise | 0.3 ms | Temporal accumulation |
| Composite | 0.2 ms | Tone map + post-FX |
| **Total** | **~7.2 ms** | **43% of frame budget** — headroom for gameplay |

---

## Implementation Guide

This section provides enough to build a compatible pipeline in any Vulkan or DirectX 12 engine.

### Prerequisites

- **Graphics API:** Vulkan 1.3+ or D3D12 (compute shaders required)
- **Offscreen render targets:** RGBA16F color, D32F depth, RGB16F normals, RG16F motion vectors
- **Compute dispatch capability:** Independent compute queue preferred for async overlap

### Step-by-Step

1. **Set up offscreen HDR rendering**  
   Create framebuffer with RGBA16F color + D32F depth. Render all geometry here instead of the swapchain.

2. **Implement shadow pass**  
   Orthographic depth-only pass from sun direction. Bind shadow map as texture in scene shaders.

3. **Output linear HDR from all geometry shaders**  
   Compute lighting normally but do NOT apply tone mapping. Output values can exceed 1.0.

4. **Build Hi-Z pyramid**  
   Compute shader that downsamples depth buffer using max operation. 6 mip levels.

5. **Implement GTAO+**  
   Start with standard GTAO, then add: second radius, bent normal accumulation, quadratic angular stepping.

6. **Implement SVRM**  
   Start with a regular probe grid. Add cosine-weighted hemisphere sampling. Add temporal jitter. Add DGB as an enhancement. Add radiance propagation.

7. **Implement reflections**  
   Hi-Z ray march along reflected view vector. Modulate march distance by roughness.

8. **Implement temporal denoise**  
   Reproject previous frame using motion vectors. Neighborhood clamp to prevent ghosting.

9. **Composite pass**  
   Fullscreen triangle that reads all effect buffers, blends with energy conservation, applies ACES tone mapping + gamma + post-FX. Write to swapchain.

### Buffer Neutral Values

When effects are disabled, their buffers must contain **neutral values** (not garbage):

| Buffer | Neutral Value | Why |
|---|---|---|
| AO | (1.0, 0.5, 0.5, 0.5) | AO=1 (no occlusion), bent normal = up |
| GI | (0, 0, 0, 0) | Zero indirect light, zero confidence |
| Reflections | (0, 0, 0, 0) | No reflections |

**Failure to clear these causes black-screen bugs** — the composite multiplies scene color by AO, and uninitialized AO ≈ 0 produces pure black.

---

## Pitfalls & Lessons Learned

These are real bugs encountered during development. Learn from our pain:

| # | Bug | Cause | Fix |
|---|---|---|---|
| 1 | Entire scene washed out white | Geometry shaders applied ACES tone map, then composite applied it again (double mapping) | Remove all tone mapping from geometry shaders. Single ACES in composite only. |
| 2 | Sky appears purple/wrong color | Skybox wrote LDR to HDR buffer, composite early-return bypassed tone mapping for sky pixels | Skybox outputs linear HDR. Composite detects sky by depth ≥ 0.9999 and applies tone mapping to sky too. |
| 3 | Black screen when effects enabled | Disabled effects left buffers uninitialized. AO buffer had garbage ≈ 0. Scene × AO = black. | Clear all effect buffers to neutral values. Zero out intensity uniforms for disabled effects. |
| 4 | Blown-out specular highlights | Sun color multiplied twice in specular calculation (once in specular term, once in final combine) | Multiply sun contribution exactly once. |
| 5 | Film grain looks wrong | Grain applied in linear HDR space before tone mapping — noise distribution is non-perceptual | Apply film grain after tone mapping (in sRGB / display space). |
| 6 | Shadows too dark in some biomes | Unclamped biome shadow-depth parameter produced negative mix factors | Clamp all biome interpolation factors to [0, 1]. |

---

## Citation

If you use this rendering architecture or any of the novel techniques (SVRM, GTAO+, world-anchored volumetrics, biome-aware lighting) in your project, please credit:

```
Hybrid Forward+Compute Render Pipeline
Author: Stelliro (https://github.com/Stelliro)
Origin: Stelliferrum Forge Engine
Year: 2025–2026
License: CC BY-SA 4.0
```

The novel techniques documented here — particularly **Sparse Voxel Radiance Marching (SVRM)** with Depth-Gap Bridging, **GTAO+ dual-horizon multi-scale AO** with bent normals and slope-adaptive sampling, **camera-ray shadow segmentation** for world-anchored volumetric god rays, and the **Orientation Anchor System (6-Point Surface Capture)** for motion-capture-style impostor rendering — are original work by the author.

---

## License

This work is licensed under the [Creative Commons Attribution-ShareAlike 4.0 International License](https://creativecommons.org/licenses/by-sa/4.0/).

You are free to:
- **Use** — including commercially
- **Adapt** — remix, transform, build upon

Under these terms:
- **Attribution** — You must give appropriate credit to Stelliro, provide a link to the license, and indicate if changes were made
- **ShareAlike** — If you remix or build upon this material, you must distribute your contributions under the same license

Built with the [Vulkan](https://www.vulkan.org/) graphics API by Khronos Group.
