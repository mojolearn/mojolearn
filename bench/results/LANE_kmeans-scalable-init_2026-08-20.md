# LANE_kmeans-scalable-init — 2026-08-20

**Goal:** port `initScalableKMeansPlusPlus` (cuVS k-means||) so the DEFAULT
initialization (`oversampling_factor = 2.0`) stops raising.
**Result: done.** The default config runs end to end, both arms of the
selection are checked, all 12 cluster checks green. No timings were run.

## Commits

    65327c5 parent f9f09c7   kmeans|| scalable init ported (kmeans.cuh:568-785)

Files: `cluster/mojo_only/scalable_init.mojo` (new),
`cluster/ported/cluster/detail/kmeans.mojo`,
`cluster/mojo_only/kmeans_check.mojo`, `cluster/kmeans_main.mojo`,
`cluster/README.md`, `cluster/PORTED_MAP.tsv`, `cluster/UNPORTED.tsv`,
`UNWIRED.md`, `PORTING.md` (deviations 47, 48).

## Upstream citations (cuVS `94c2819`, the repo's recorded pin; paths under `cpp/src/cluster/detail/`)

| piece | where |
|---|---|
| function | `kmeans.cuh:568-785` |
| step 1, host mt19937 seed point + flag fill | `:585-609` (flag fill `:593-601`) |
| step 2, psi = phi_X(C), `identity_op` sum, host readback | `:629-650` |
| round count `min(8, (int)ceil(log(psi)))` | `:656` |
| sampling rounds (full recompute per round, psi re-read) | `:660-720`; per-round cost `:664-685` |
| per-sample uniforms | `raft::random::uniform`, `:689-690` |
| selection predicate | `SamplingOp::operator()`, `kmeans_common.cuh:73-81` — `prob_x = (l * k * value) / cost`, strict `>`, `!flag` exclusion |
| compaction + count readback + flag update | `sampleCentroids`, `kmeans_common.cuh:228-289` (`cub::DeviceSelect::If` `:241-266`, count sync `:267-269`, `thrust::for_each_n` `:270-276`) |
| append / resize | `kmeans.cuh:713-719` |
| selection tail, three arms | `> k` `:726-753`; `< k` supplement-with-random `:755-777`; `== k` copy `:778-784` |
| step 7 weights | `countSamplesInCluster`, `kmeans_common.cuh:615-668`; `countLabels` = `cub::DeviceHistogram::HistogramEven` with FLOAT counters, `:95-135` |
| step 8 reduction to k | UNWEIGHTED `kmeansPlusPlus` under the OUTER params (`kmeans.cuh:738-739`), then `kmeans_fit_main` over the weighted candidates under FRESH defaults with only `n_clusters` copied (`:743-753`) |
| algorithm switch | `:910-915` — `oversampling_factor == 0` picks classic k-means++, anything else picks this |

Two upstream facts a from-the-paper reimplementation gets wrong, copied
faithfully and documented in the port's docstring: the candidate-set
k-means++ is unweighted (weights enter only in the following Lloyd pass),
and that Lloyd pass runs under fresh default params (L2Expanded, default
stopping rule) even when the outer fit's differ. The `< k` arm fills the
FIRST `k - |C|` rows with random init and puts the candidates AFTER them.

## Primitives: reused vs added

Reused (zero duplication of existing machinery):

- `min_cluster_and_distance_compute` + `compute_centroid_norms` for every
  per-round `minClusterDistanceCompute` and for step 7's assignment (their
  `countSamplesInCluster` assignment is the same fused pass; ours also
  yields the label, which step 7 consumes directly).
- `_sum_device` (`SUM_MODE_PLAIN`) for `computeClusterCost`/psi.
- The three-stage inclusive scan from `mojo_only/plus_plus.mojo`
  (`chunk_sums` / `scan_chunk_offsets` / `write_inclusive_scan`) as the
  ranking half of `DeviceSelect::If`.
- `gather_rows_kernel` for the candidate append, `copy_f32_kernel`,
  `zero_i32_kernel`, `init_random` (supplement arm), `kmeans_plus_plus`
  (step 8), `kmeans_fit_main` recursively for the step-8 Lloyd
  (`INIT_ARRAY` = their "no init, start from centroidsRawData").
- `choose_scale` for the inner Lloyd's fixed-point scales.

Added (`cluster/mojo_only/scalable_init.mojo`, each naming the vendor call
it replaces): `scalable_uniform` + `scalable_keep` (raft::random::uniform +
SamplingOp, host-callable on purpose for the replay check),
`sample_flags_kernel`, `select_scatter_kernel` (DeviceSelect::If's write +
the thrust flag update), `count_labels_kernel` (cub::DeviceHistogram, float
counters as theirs), `zero_f32_kernel`, `set_flag_kernel`.

## Oracle seed-to-seed spread, and the tolerance chosen from it

sklearn 1.9.0 (`.pixi/envs/bench/bin/python`),
`KMeans(n_clusters=4, init='k-means++', n_init=1, max_iter=50, tol=1e-4)`
on the exact check fixture (512 x 4, planted centers `100*(c+1)+f`, hashed
jitter, round-robin membership), n_init pinned to 1 on both sides:

| seed | inertia |
|---|---|
| 0..9 (all ten) | 171.193649 |

    min 171.193649  max 171.193649  spread 0  rel_spread 0

The oracle's seed spread is ZERO — the fixture is separable enough that
every draw reaches the same optimum — so the check tolerance cannot come
from draw variance and covers only our arithmetic (float32 GEMM path +
fixed-point accumulate). That arithmetic error is measured at 0.0029
relative on the classic-init path (170.703125 vs analytic 171.19362); the
scalable check keeps the same 0.02 relative ceiling the existing
`check_kmeans_fit` uses, ~7x above the measured error, 0 draw allowance.

## Check output (all 12 green, `cluster/kmeans_main.mojo` binary)

    check_reach_by_sabotage OK: centroid_norm moved 384/512 labels; x_norm moved 512 distances and 0 labels, which is the predicted shape
    check_kmeans_fit OK: 4/4 centroids matched as a permutation, 0/512 rows misassigned, inertia 170.703125 vs expected 171.19361649473004 (rel 0.0028651272446548032), 2 iterations
    check_device_inclusive_scan OK: 20000 entries, worst relative error 0.0, past one block's worth
    check_kmeans_plus_plus_init OK: 4/4 centroids recovered as a permutation through the k-means++ path, inertia 170.703125, 2 iterations
    check_scalable_sampling_selection OK: 6/4000 drawn exactly as the host replay predicts (order, flags, count); zeroing 1/3 of the costs moved the selection to exactly baseline-minus-zeroed (4), 2 rows vanished as predicted
    check_scalable_kmeans_plus_plus_init OK: DEFAULT config (oversampling_factor=2.0) ran end to end, 4/4 centroids as a permutation, 0/512 rows misassigned, inertia 170.703125 vs expected 171.19361649473004 (rel 0.0028651272446548032), run-twice bitwise equal, 2 iterations
    check_scalable_supplement_branch OK: oversampling_factor=1e-9 starves every round, the < k supplement arm ran and landed in a worse basin as predicted (inertia 1280083.125 vs default-arm 157.7734375), run-twice bitwise equal, labels in range
    check_fused_reduction_across_lanes OK: 512 rows x 40 clusters match a host argmin, winners spread over 16 owner lanes
    check_assignment_arm_dispatch OK: fused arm proved (0/32768 tile cells written; unfused sabotage overwrote 32768); arms agree on all labels, min_dist worst rel 0.0
    check_fused_policy_dispatch OK: selection pinned (4/2/1 by k and alignment, skinny at k<32), bench alignment k=32 takes veclen 4 on real buffers, scalar and 2-wide arms correct through the launcher with the m grid-stride exercised
    check_privatized_accumulate OK: 2048 sum cells + 64 weight cells bit-identical direct vs privatized, run-twice bitwise equal, dropped flush moved 352 cells, veclen=4 read arm pinned
    check_accumulate_veclen_dispatch OK: scalar read arm reached-and-correct at d=33, 2-wide arm at d=34, 4-wide arm pinned in check_privatized_accumulate

Reach evidence is sabotage/replay, not digests:

- `check_scalable_sampling_selection` holds one sampling round to an EXACT
  host replay of the same predicate (`scalable_uniform`/`scalable_keep` are
  host-callable): same count, same indices, same order, flags exactly
  preflagged-union-selected. Then it ZEROES every third planted cost and
  reruns with the same seed and psi: the strict `>` makes a zero-cost
  sample undrawable, so the predicted movement is exactly
  baseline-minus-zeroed — and it is (6 -> 4, both vanished rows predicted).
- `check_scalable_kmeans_plus_plus_init` runs the previously-raising
  DEFAULT end to end, pins BOTH arms of the `:910-915` predicate, recovers
  4/4 as a permutation, and reproduces the inertia bitwise on a second run.
- `check_scalable_supplement_branch` proves the oversampling VALUE reaches
  `prob_x` at the fit level: `1e-9` starves every round (~1e-12
  probabilities vs 24-bit uniforms, strict compare), forcing the `< k`
  random-supplement arm, which lands in a measurably worse basin
  (1280083.125 vs 157.77 at 8 planted clusters). First version used the
  4-cluster fixture and the two arms LEGITIMATELY tied bitwise (four random
  supplements covered four basins); at k=8 coverage probability is
  7!/7^7 ~ 0.6% and the seed is pinned. Classic-arm reach
  (`oversampling_factor = 0`) was already held by
  `check_kmeans_plus_plus_init`.

## Deviations, priced (full text in PORTING.md 47/48)

- **47 — randomness:** per-sample uniforms are a splitmix64 counter hash of
  `(round_seed, index)` (host contributes one u64 per round; O(rows) host
  randomness is forbidden and no Mojo Philox exists), and `prob_x` is
  formed in f32 (no device f64 on Apple) with `l * k` pre-multiplied in
  f64 on the host. Price: a different draw stream than cuVS from the same
  seed (already the rule under deviation 17 — compare inertia across seeds,
  never draws), and at most one-ulp probability-threshold flips vs their
  double arithmetic, which the replay check sidesteps by replaying the same
  f32 expression.
- **48 — selection:** `DeviceSelect::If` = flags + the existing f32
  three-stage scan + rank scatter (stable, like theirs); the thrust flag
  update is folded into the scatter; `countLabels` is a float-counter
  atomic histogram (float counters are upstream's own choice there). Price:
  counts are exact only below 2^24 rows, so the driver RAISES at
  `n_samples >= 2^24` instead of silently mis-scattering; lifting that
  bound means an integer scan, not a tolerance. Also inside 48: the step-8
  Lloyd's fixed-point scales come from an O(candidates) host readback
  (`choose_scale` over count-weighted candidates), a mechanism upstream
  does not need because their accumulate never quantizes.
- Unnumbered mechanism notes in the code: the step-1 flag fill is a device
  zero + one-thread write where theirs host-fills and copies O(rows)
  (`:593-601`) — same values; per-round buffer churn mirrors their
  `rmm::device_uvector::resize`; our per-round assignment also produces
  labels (their init-path `minClusterDistanceCompute` does not), consumed
  by step 7 at no extra traffic.

## Known limits / open

- The `== k` copy-out arm of the selection tail (`:778-784`) is ported but
  UNREACHED — no deterministic fixture pins the candidate count to exactly
  k. Recorded in UNWIRED.md's cluster table.
- `init = Random` as a user-facing entry path still has no dedicated check
  (init_random itself is now reached through the supplement arm).
- The fit passes its `dist_buf` (sized for k centroids) into the init's
  candidate assignments; the fused arm — the only shipped arm for the
  L2 metrics `validate()` admits — never touches it. If the unfused arm is
  ever wired for those metrics, the init needs its own candidate-sized
  tile buffer.
- `kmeans_common.mojo::sampling_probability` (Float64) remains a
  documentation row: the shipped predicate is the f32 `scalable_keep`
  (no device f64); UNWIRED.md row updated accordingly.

## Suggested SCOREBOARD sentence (orchestrator's to place)

k-means: cuVS's DEFAULT init (scalable k-means||, oversampling_factor=2.0)
is now ported and checked — the default no longer raises; sampling proven by
exact host replay + cost sabotage, default config run-twice bitwise, <k
supplement arm reached; correctness-only lane, no timings run.
