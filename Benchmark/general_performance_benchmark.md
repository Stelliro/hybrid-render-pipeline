# General Performance Benchmark — Where Giants Rust
**Date recorded:** 2026-07-01
**Engine phase:** Phase 0 (pre-optimization baseline)
**Source data:** `general_performance_benchmark.csv`

---

> **NOTE: This game is currently unoptimized. No performance passes have been done as of
> Phase 0. These figures represent baseline/pre-optimization state and should not be taken
> as target performance.**

---

## What this benchmark measures

This is the **headless terrain-streaming benchmark** (`--benchmark` mode). It launches the
engine with a hidden window, generates a world, then teleports through multiple positions to
stress-test chunk streaming and rendering at various distances and biome types.

**This benchmark does NOT cover:**
- NPC / AI load (no NPCs are spawned)
- Settlement mesh rendering
- Per-subsystem bucket timings (Render / Terrain / Vegetation / Physics / AI / UI) — those
  require the debug-server build and the stress scenario in `docs/PERF_BASELINE.md`

It is a good proxy for **raw terrain and rendering throughput** but is not the full
stress scenario.

---

## Run summary

| Metric              | Value         |
|---------------------|---------------|
| Total frames        | 11,558        |
| Total duration      | 60.0 s        |
| Steady-state frames | 11,160 (after first 5 s loading spike excluded) |
| Reference hardware  | [TBD — record your GPU/CPU here]              |
| Resolution          | [TBD]                                         |

---

## Frame time — steady state (t > 5 s)

The first ~5 seconds are excluded from these figures; that window contains the initial
world-generation and chunk-upload spike (worst single frame: **250.0 ms at t = 0.8 s**,
which is normal engine startup behaviour, not a gameplay stutter).

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

The **p1 low of 130.7 FPS** means the bottom 1% of frames still clear 60 FPS comfortably,
which is a healthy sign even in this unoptimized state.

---

## Stutter analysis — steady state

| Threshold                   | Count | % of frames |
|-----------------------------|-------|-------------|
| > 33.3 ms  (below 30 FPS)  | 10    | 0.1%        |
| > 100 ms   (visible hitch) | 0     | 0.0%        |

10 frames breached the 30 FPS threshold during steady state. The worst of those was
**75.3 ms**, which is likely a single-frame GPU upload stall during a chunk LOD transition.
Zero hitches above 100 ms in steady state is a good result at this stage.

---

## Per-10-second window breakdown

The benchmark teleports through multiple world positions; the window table shows how
performance evolves as chunks stream in and out.

| Window   | Avg FPS | p95 ms | Peak Loaded | Peak Drawn | Peak Pending | Peak Vertices | Peak Trees |
|----------|---------|--------|-------------|------------|--------------|---------------|------------|
| 0–10 s   | 149.3   | 11.12  | 289         | 114        | 21           | 1,758,336     | 1,157      |
| 10–20 s  | 159.1   | 6.83   | 289         | 115        | 0            | 1,758,336     | 1,157      |
| 20–30 s  | 184.0   | 6.83   | 84          | 56         | 22           | 1,545,216     | 292        |
| 30–40 s  | 234.1   | 5.59   | 47          | 31         | 23           | 947,328       | 191        |
| 40–50 s  | 239.6   | 4.96   | 47          | 31         | 23           | 947,328       | 191        |
| 50–60 s  | 241.1   | 4.78   | 19          | 16         | 22           | 281,664       | 133        |
| 60–70 s  | 101.7   | 9.84   | 19          | 16         | 17           | 281,664       | 20         |

**Reading the table:**

The 0–20 s window is the heaviest load: 289 chunks loaded, 115 drawn, ~1.76 M vertices,
1,157 trees — and the engine still sustains ~150–160 FPS. FPS climbs steadily as the
benchmark moves to lower-density positions and fewer chunks need to be held in memory.
The 60–70 s dip to ~102 FPS is the benchmark winding down and unloading chunks.

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

**Cull rate:** at peak load, 175 of 289 chunks (60%) are frustum-culled. That's expected —
the culler is working. Drawn chunks stay at roughly 40% of loaded at all times.

---

## Assessment

| Area                    | Status                                                     |
|-------------------------|------------------------------------------------------------|
| Frame rate floor        | ✅ Well above 60 FPS target at all terrain loads tested     |
| p99 frame time          | ✅ 7.64 ms — comfortably within 16.67 ms budget             |
| Stutter rate            | ✅ 0.1% of frames — very low                               |
| Zero 100 ms+ hitches    | ✅ No hard hitches in steady state                         |
| Initial load spike      | ⚠️  250 ms at t = 0.8 s — normal but worth monitoring      |
| 75 ms spike (steady)    | ⚠️  One frame; likely a chunk upload stall — investigate if it recurs |
| NPC / AI load           | ❓ Not measured here — see `docs/PERF_BASELINE.md` stress scenario |
| Subsystem buckets       | ❓ Not available in this run — requires debug-server build  |

**Overall:** the engine handles terrain streaming well in this unoptimized Phase 0 state.
The main unknowns are NPC/AI cost and per-subsystem breakdown under the full stress scenario.
Those should be the next measurement priority.

---

## Next steps

1. Record hardware used for this run (GPU, CPU, RAM, resolution) and fill in the
   [TBD] fields above.
2. Run the full stress scenario from `docs/PERF_BASELINE.md` (48 NPCs, settlement plaza,
   midday) with the debug-server build to get per-subsystem bucket numbers.
3. Fill in the baseline tables in `docs/PERF_BASELINE.md` once both datasets exist.
4. Re-run this benchmark after any major Phase (1, 2, 3 …) and diff the window table
   to catch terrain regressions early.
