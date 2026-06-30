# General Performance Benchmark — Where Giants Rust (Hybrid Render Pipeline)
**Date recorded:** 2026-07-01
**Engine phase:** Phase 0 (pre-optimization baseline)
**Source data:** `general_performance_benchmark.csv`

---

> **NOTE: The engine is entirely unoptimized as of Phase 0. These figures represent raw baseline performance and are not target metrics.**

---

## What this benchmark measures

This is the headless terrain-streaming benchmark (`--benchmark` mode). It launches with a hidden window, generates a world, and teleports through multiple positions to stress-test chunk streaming and rendering across various distances and biomes.

**This benchmark does NOT cover:**
- NPC / AI load (no NPCs spawned)
- Settlement mesh rendering
- Per-subsystem bucket timings (Render / Terrain / Vegetation / Physics / AI / UI)

It serves as a proxy for raw terrain and rendering throughput, not a full simulation stress test.

---

## Run summary

| Metric              | Value         |
|---------------------|---------------|
| Total frames        | 11,558        |
| Total duration      | 60.0 s        |
| Steady-state frames | 11,160 (excludes first 5s) |
| Reference hardware  | [TBD — record your GPU/CPU here]              |
| Resolution          | [TBD]                                         |

---

## Frame time — steady state (t > 5 s)

The first 5 seconds are excluded from steady-state figures to filter out the initial world-gen and chunk upload spike (worst single frame: **250.0 ms at t = 0.8 s**).

| Statistic       | Frame time (ms) | Equivalent FPS |
|-----------------|-----------------|----------------|
| **Average**     | 4.93 ms         | ~203 FPS       |
| **Median (p50)**| 4.48 ms         | ~223 FPS       |
| **p95**         | 6.47 ms         | ~155 FPS       |
| **p99**         | 7.64 ms         | ~131 FPS       |
| **Max (spike)** | 75.32 ms        | ~13 FPS        |
| **Min**         | 3.66 ms         | ~273 FPS       |

### FPS distribution

| Statistic     | FPS   |
|---------------|-------|
| Average       | 212.8 |
| p1 low        | 130.7 |
| p5 low        | 154.5 |
| Peak          | 273.1 |

---

## Stutter analysis — steady state

| Threshold                   | Count | % of frames |
|-----------------------------|-------|-------------|
| > 33.3 ms  (below 30 FPS)   | 10    | 0.1%        |
| > 100 ms   (visible hitch)  | 0     | 0.0%        |

10 frames breached the 33.3 ms threshold during steady state. The worst of those was **75.3 ms**, likely a single-frame GPU upload stall during a chunk LOD transition. 

---

## Per-10-second window breakdown

This table tracks performance evolution as chunks stream in and out while teleporting.

| Window   | Avg FPS | p95 ms | Peak Loaded | Peak Drawn | Peak Pending | Peak Vertices | Peak Trees |
|----------|---------|--------|-------------|------------|--------------|---------------|------------|
| 0–10 s   | 149.3   | 11.12  | 289         | 114        | 21           | 1,758,336     | 1,157      |
| 10–20 s  | 159.1   | 6.83   | 289         | 115        | 0            | 1,758,336     | 1,157      |
| 20–30 s  | 184.0   | 6.83   | 84          | 56         | 22           | 1,545,216     | 292        |
| 30–40 s  | 234.1   | 5.59   | 47          | 31         | 23           | 947,328       | 191        |
| 40–50 s  | 239.6   | 4.96   | 47          | 31         | 23           | 947,328       | 191        |
| 50–60 s  | 241.1   | 4.78   | 19          | 16         | 22           | 281,664       | 133        |
| 60–70 s  | 101.7   | 9.84   | 19          | 16         | 17           | 281,664       | 20         |

The 0–20 s window handles the heaviest load (289 chunks loaded, ~1.76M vertices) while sustaining ~150 FPS. The dip to ~102 FPS at 60–70 s is benchmark teardown/chunk unloading.

---

## Peak terrain counters — steady state

| Counter         | Average | Peak    |
|-----------------|---------|---------|
| Loaded chunks   | 72      | 289     |
| Drawn chunks    | 33      | 115     |
| Culled chunks   | 39      | 175     |
| Pending chunks  | 14      | 23      |
| Total vertices  | 542,458 | 1,758,336 |
| Veg chunks      | 8       | 30      |
| Trees           | 268     | 1,157   |

Frustum culling handles ~60% of chunks at peak load (175 of 289). Drawn chunks hover consistently at ~40% of loaded chunks.

---

## Assessment

| Area                    | Status                                                     |
|-------------------------|------------------------------------------------------------|
| Frame rate floor        | ✅ Well above 60 FPS target at all terrain loads tested    |
| p99 frame time          | ✅ 7.64 ms — comfortably within 16.67 ms budget            |
| Stutter rate            | ✅ 0.1% of frames                                          |
| Zero 100 ms+ hitches    | ✅ No hard hitches in steady state                         |
| Initial load spike      | ⚠️  250 ms at t = 0.8 s                                    |
| 75 ms spike (steady)    | ⚠️  One frame; likely a chunk upload stall                 |
| NPC / AI load           | ❓ Not measured yet                                        |
| Subsystem buckets       | ❓ Not available in this run                               |

**Overall:** the engine handles terrain streaming well in this unoptimized Phase 0 state.
The main unknowns are NPC/AI cost and per-subsystem breakdown under the full stress scenario.
Those will be the next measurement priority.
