# Mojo 1.2 nightly trial (2026-09-26)

Question: what does moving mojolearn from Mojo 1.0.0 (ed45d567, max-core 26.5.0)
to the newest nightly do? Status: **DONE** (Mac, CPU, one NVIDIA box; AMD not tested).

## Verdict

**The upgrade moves 30 of 627 release cells (10 lanes x 3 fixtures), all in
the k-means family (kmeans, kmeans-sqrt, kmeans-weighted) and the lanes that
consume a k-means fit (gmm, gmm-sample, ivf, ivf-euclidean, ivf-extend,
metrics, metrics-classification). GPU cross-vendor agreement HOLDS on the
nightly (Metal == CUDA on 627/627 cells, 984/984 infer/model), but CPU vs GPU
agreement BREAKS on exactly those 30 cells: the nightly CPU host column keeps
the 0.8.19 bits (627/627 equal to the 0.8.19 Metal column) while both
nightly GPU columns moved to the same new bits. Incompatibilities: six
mechanical API changes (below), all fixed; nothing non-trivial.** The Mojo
1.0.0 FAST svm compile deadlock is gone on the nightly.

Under the project rule (any cross-column difference is a defect) the
nightly is NOT upgradeable yet: the GPU k-means path must be made to agree
with the CPU host oracle again (or both moved together) first.

## Recommended upgrade plan

1. Root-cause the k-means GPU move (next section) and restore CPU == GPU on
   the nightly; re-run these three columns until all read 627/627.
2. Land the source migration on its own branch as one commit
   (`migrate.py` + `fixups.py`, 6 changes), plus the pixi pin in a
   separate commit; the renamed APIs do not exist in 1.0.0, so the source
   commit and the toolchain commit must land together.
3. Wait for a STABLE Mojo 1.2 / MAX 26.7 (or pin one nightly by exact
   build) before a release; a moving nightly is not a release toolchain.
4. Release run on the new toolchain: all four columns (CPU, Metal, NVIDIA,
   AMD) at full release selection, 0 DIVERGENT, plus the cross-compile gate;
   then drop 147725fe4's svm workaround (item 4 shows it is unneeded).
5. Separately, a second mechanical pass for the new deprecation warnings
   (pointer indexing `unsafe_offset=`, `unsafe_bitcast`, `Pointer`,
   `unsafe_alloc`/`Layout`, `@__parameter`) before they become errors.

## Toolchain

| | current (main, 0.8.19) | trial |
|---|---|---|
| mojo | 1.0.0 (ed45d567) | 1.2.0.dev2026092505 (8b86047f) |
| mojo-compiler | 1.0.0 | 1.2.0.dev2026092505 |
| max / max-core | 26.5.0 | 26.7.0.dev2026092505 |
| channel | conda.modular.com/max | conda.modular.com/max-nightly (first), then max, conda-forge |
| python (default env) | 3.14 | 3.14.7 |

Everything below was run at the **0.8.19 commit (69a519c1522d)** plus the
nightly pin and the source renames, on local branch
`trial/mojo-1-2-at-0819` (commits: ad0c44375 pixi pin, a4dd5d70a mechanical
renames, e85d3b7ae trial driver files, e8d302a46 lane_id import). Running at
the release commit means every moved cell is the toolchain's doing, not a
source change since 0.8.19.

## 1. Source incompatibilities

With the pin alone, **76 of 76 bindings fail to parse**. All failures are
Mojo 1.1 removals/renames (mojolang.org/releases/v1.1.0):

| change | fix | occurrences |
|---|---|---|
| `std.gpu` is private (`std._gpu`); the package moved to `max.gpu` | `std.gpu` -> `max.gpu` | 477 in the tree |
| `memcpy`, `memset`, `memset_zero` removed from `std.memory` | `unsafe_memcpy`, `unsafe_memset`, `unsafe_memset_zero` | 112 |
| `InlineArray` alias removed | `Array` | 214 |
| `@parameter if` / `@parameter for` removed | `comptime if` / `comptime for` | 255 |
| `lane_id` no longer re-exported by `max.gpu.primitives.warp` | import from `max.gpu` | 2 files |
| `nn.topk.top_k` gained a `KEngine` parameter that cannot be inferred when `k` is defaulted | pass `KEngine=type_of(input_tile).Engine` | 2 call sites |

All six are mechanical. The first four are applied by `migrate.py` (in this
directory) over every tracked `.mojo` file (425 files at 0.8.19); the last
two by `fixups.py`. After them, **every binding builds** on the Mac
(76/76: identical 23 GPU-tier + 32 host, fast 18, deterministic 3; see `compile_times.md`)
and **every CUDA binding builds** on an RTX 4090 (23 GPU + 32 host, 0
failures).

New WARNINGS (not errors, nothing fixed): positional `ptr[i]` on pointers
(`use unsafe_offset=`, ~440 sites), `ptr + n` (`use unsafe_offset`),
`bitcast` -> `unsafe_bitcast`, `UnsafePointer` -> `Pointer`, `alloc` without a
`Layout` (use `unsafe_alloc` for now), `@parameter` on closures ->
`@__parameter` (2 sites). These will become errors in a later release; they
are a second, larger mechanical pass.

## 2. Identity (0.8.19 release selection: 209 lanes x base,denormal,odd = 627 cells, one fit, core probes)

### NVIDIA RTX 4090 (sm_89), nightly, vs the recorded 0.8.19 NVIDIA column
RunPod pod 6rhqfhehr5o96m, cuda set built on the box with the nightly
(1097 s for 23 GPU families + 439 s for 32 host families, 16 jobs), then
`verify_lanes.py --gpu-pass cuda --selection selection-cuda.json`
(the recorded 0.8.19 selection): COMPLETE, 627 cells, 159 s.
The gemm device check and the gemm card also ran on the nightly: device
gates GREEN, card IDENTICAL to an Apple card (60/60 stages).

`identity_break.py --diff 0.8.19-cuda nightly-cuda`: **IDENTICAL=597, DIVERGENT=30**.
The 30 cells are 10 lanes x 3 fixtures, all in one family tree:

| lane | parts that moved |
|---|---|
| kmeans, kmeans-weighted | centers, labels |
| kmeans-sqrt | centers, labels, inertia (scales agree) |
| gmm, gmm-sample | every fitted part (init is k-means) |
| ivf, ivf-euclidean, ivf-extend | dist, idx, cand (coarse quantizer is k-means) |
| metrics | accuracy, ari, vmeasure, silhouette (r2 agrees) |
| metrics-classification | entropy, v_measure, rand, mutual_info, homogeneity, completeness, silhouette_samples (29 other parts agree) |

### Apple M4 Metal, nightly, vs the recorded 0.8.19 Metal column
(first pass, 567 of 627 cells: 60 cells refused because the worktree lacked
`python/mojolearn/.dylibs/libMojolearnMath.dylib`, an environment gap, not
the toolchain; rerun pending): **IDENTICAL=540, DIVERGENT=27**, the same
lanes as NVIDIA (metrics-classification was among the refused).

### Cross-vendor agreement on the nightly
nightly Metal vs nightly CUDA: **IDENTICAL=567 of 567 compared, 0 DIVERGENT**
(infer/model 876 IDENTICAL). The moved cells moved to the SAME bits on
both vendors.

### CPU host column, nightly
`verify_lanes.py --cpu-pass --selection selection-cpu.json` (209 lanes, 627
cells, 2 shards, 663 s): COMPLETE.
- nightly CPU vs 0.8.19 Metal: **IDENTICAL=627** (infer/model 978 IDENTICAL).
- nightly CPU vs nightly Metal vs nightly CUDA: **DIVERGENT=30** (the same 30
  cells; CPU carries the old hash, Metal and CUDA share the new one), e.g.
  `kmeans/base` model 355598f4 (CPU) vs d4eb8bc4 (Metal, CUDA).

### Where the move comes from (not yet root-caused)
Every moved lane is a k-means fit or consumes one (`metrics-classification`
scores `ml.KMeans(...).labels_`; gmm initializes from k-means; ivf's coarse
quantizer is k-means). The CPU host route (`cluster/host/kmeans_oracle.mojo`,
compiled by the SAME nightly) did not move, so the change is on the GPU
k-means path and is vendor-independent. First suspect checked and CLEARED:
`max.gpu.primitives.block.prefix_sum` (k-means++ sampling scan), whose block
and warp source in modular/modular main is the same algorithm as
max/v26.5.0. Next step: an identity trace (`core/identity_trace`) of one
`kmeans/base` fit on both toolchains to find the first stage whose hash
differs (candidates: other MAX GPU primitives, `core.gemm.gemm_nt` in the
unfused arm, or a codegen change such as contraction).

AMD: not tested (no AMD box in this trial; DigitalOcean and Hot Aisle are
reserved for the GPT-3 run).

## 3. Speed (observations, no claims)

Compile, Mac (one binding at a time, -j 1, cold private cache, machine load
average 10 to 16 from other lanes): see `compile_times.md`. Matched total
3509 s nightly vs 2436 s for the Sep 21 1.0.0 cache builds (ratio 1.44), but
the baseline is a different day, load and source, so this is not a
measurement. On the 4090 the 23 GPU families took 1097 s at 16 jobs; the Sep
19 1.0.0 gap leg recorded 1204 s for the same 23 on a 4090.

Runtime (cell timings in the columns, tiny fixtures, one run each): NVIDIA
sum of cell seconds 104.3 (0.8.19) vs 93.9 (nightly), nothing beyond noise
except `knn` 0.05 s -> 0.52 s. Metal sum 325.8 s (0.8.19) vs 535.8 s
(nightly; pass wall 376 s vs 594 s), with the largest ratios in gbdt adapter
/ CTR lanes (about 2.7x) and cross-val (3.3x); this Mac was running other
lanes' CPU work (load 10 to 16), so it is a flag to re-measure on a quiet
machine, not a finding.

## 4. FAST svm deadlock (147725fe4's workaround reverted)

Copy of main (lane/mojo-1-2-trial base, 5e33d2db3-era) + migrate.py + fixups.py,
with `svm/impl/distance/kernel_matrices.mojo:366` reverted to the plain
`_kernel_rows(...)` call (the 0.8.19 form that deadlocked Mojo 1.0.0 for
NVPTX/AMDGPU), FAST tier, compiled on the Mac through
`tools/cross_compile_check.py`'s own `compile_cmd`/`run_one` (STALL detector on):

| target | Mojo 1.0.0 (147725fe4 message) | 1.2 nightly |
|---|---|---|
| sm_89 | STALLED (>600 s, deadlock) | **PASS, 38.2 s** |
| gfx942 | STALLED (>600 s, deadlock) | **PASS, 46.3 s** |

The deadlock is gone in real code on the nightly.

## Left to do

1. Root-cause the GPU k-means move (identity trace diff of `kmeans/base`,
   1.0.0 vs nightly, on Metal) and fix so CPU == GPU on the nightly.
2. AMD column (needs an MI300X box; DO/Hot Aisle reserved for GPT-3).

## Files

- `migrate.py`, `fixups.py`: the source migration (run from a checkout root).
- `compile_times.md`: per-binding compile seconds (Mac).
- `diffs/`: the identity_break --diff outputs quoted above.
- Evidence (outside the repo): NVIDIA leg `~/mojolearn-evidence/e1g/2026-09-26_022110-nvidia/`
  (nightly CUDA column at `remote/trial/cuda/column.json`); nightly Metal and CPU
  and CUDA columns gzipped in `columns/` here; `diffs/` keeps each diff's summary and non-IDENTICAL rows.
- Source used: branch `trial/mojo-1-2-at-0819` (pushed).

Cost so far: one RTX 4090 pod, 31 min at $0.74/h = about $0.40; terminated
and verified gone (HTTP 404). An earlier create attempt got HTTP 500 and
made no pod.
