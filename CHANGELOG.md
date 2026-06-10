# Changelog

All notable changes to this repository will be documented in this file.

## [1.2.0] - 2026-06-10

### Added
- Reference implementation in `src/` and `shaders/` (C++20 / Vulkan 1.3)
  - `Renderer`, `ScreenSpaceEffects`, `Camera`, `Input`, `Log`, `Platform`, `Types`, `Frustum`, `ErrorCodes`
  - 25 GLSL shaders covering the full forward+compute pipeline
  - `main.cpp` demo: procedural FBM terrain + sky dome + orbit camera, driving the full screen-space stack
  - `CMakeLists.txt` with vcpkg manifest mode; auto-compiles shaders to `build/shaders/*.spv`

## [1.1.0] - 2026-04-15

### Added
- Orientation Anchor System (6-Point Surface Capture) documentation
  - Motion-capture-style marker points for camera-visible surface determination
  - Octahedron-mapped impostor atlas with orientation-correct UV lookup
  - Roll correction for physics-driven objects (fallen trees, ragdolls, debris)
  - Per-pixel normal correction for accurate impostor lighting at any orientation
  - Proactive texture streaming with angular velocity pre-fetch
  - Implementation checklist and performance budget (~0.15 ms/frame for 750 objects)

## [1.0.0] - 2026-04-15

### Added
- Initial public release of the Hybrid Forward+Compute Render Pipeline architecture
- SVRM (Sparse Voxel Radiance Marching) algorithm documentation with Depth-Gap Bridging
- GTAO+ (Dual-Horizon Multi-Scale AO) with bent normals and slope-adaptive sampling
- World-anchored volumetric god rays via camera-ray shadow segmentation
- Biome-aware per-pixel lighting profile system
- Procedural skybox (11-layer) architecture overview
- Light-frustum culled shadow mapping technique
- Single tone-mapping point architectural pattern
- Performance budget table and implementation guide
- Pitfalls & lessons learned from production use
- CC BY-SA 4.0 license
