# AI and classical ML IDENTICAL handoff — September 10, 2026

## Scope and starting point

Continue GPU AI training/inference and non-tree classical ML, exclusively in
IDENTICAL. Preserve the existing arithmetic, reduction order, rounding, tie
rules and numerical admission. Trees, forests, isolation forest, GBDT and WP8
are another lane. FAST and DETERMINISTIC product changes are outside this plan;
recent FAST KDE/SVM results are not IDENTICAL performance evidence.

Main at publication includes `92928a2c`, which makes every non-tree binding
IDENTICAL-only. The performance/source review used `1484dd87`, including the completed
WP6/WP7 work published through `9b868c42`. It introduces no implementation or
new timing result. Read current source before executing an older handoff.
Commit passing, scoped changes and push to main; preserve concurrent lane work.

## What is already done

- Runtime-shaped LM configuration, layer traversal and parameter registry are
  implemented. Device gradients pass between layers. Use this generalized
  trainer for new work; retain small configurations as correctness fixtures.
- Fused attention and lazy quadratic-stage allocation are wired into training.
  Do not schedule their initial integration again. Eager/fallback allocation
  still needs to be accounted for when assessing memory fit.
- Owned resident LM sessions retain context, parameters, Adam state and
  workspaces. Metal's real-text capture runner uses them by default; the
  public constructor and CUDA/HIP defaults remain unchanged. Host snapshots,
  mirror readbacks and full state/gradient outputs remain.
- Non-tree WP6/WP7 share host-copy helpers, bulk-copy binding data and remove
  the documented duplicate immutable uploads. kNN classification retains
  device search indices, uploading a corrected order only when host sorting
  changes it. Weighted distance/weight transfers remain.

Sources: [session handoff](HANDOFF_lm_session_2026-09-10.md),
[lean attention](HANDOFF_lm_lean_attention_2026-09-10.md),
[device gradients](HANDOFF_lm_device_gradients_2026-09-10.md), and
[WP6/WP7](HANDOFF_wp6_wp7_2026-09-10.md).
The WP6/WP7 full IDENTICAL suites passed on RTX 4090; integrated validation
passed 88 host tests and 13 output-capture comparisons. The broader 956-array
count spans the modes supported at qualification time and must not be labeled
IDENTICAL-only. Do not rebuild those withdrawn lower tiers.
Apple's 54,944 host-helper bit checks do not qualify estimator GPU execution
on Apple or AMD. Those transfer-sweep vendor checks remain owed.

## Performance we can actually claim

These are historical, scoped measurements, not a rebaseline of today's main.
Exact fixtures and provenance are in the linked records.

| Area | Latest recorded evidence | Interpretation |
|---|---|---|
| H100 dense GEMM, Llama t512 | 4.07–4.52× cached cuBLAS FP32; 9.73–11.28 TFLOP/s | Qualified component gap; not whole-training throughput |
| L40S attention, original HD64 fixture | 3.91× forward; 4.32× forward+backward versus cached FP32 SDPA | Component gap; exact opponent tuple required |
| NVIDIA bare kNN, 400k/4k/d32, k10/15 | 2.61–2.88× cached cuML | Aligned-load default already qualified; selection is next |
| RTX 4090 kNN classification, same sizes | 2.2–4.8% less own native request time after WP7 | Does not revise the bare-search opponent ratio |
| RTX 4090 GP native mean prediction | 4.073 → 1.514 s median after WP6 | Synthetic 20k-row identity factor, 1.6 GB, four queries; excludes fit and Python preparation |
| Apple resident LM, 20.45M parameters, L2048, V8192 | 8.115 → 6.769 s complete-step median; 62-second pilot | 16.6% lower on synthetic batches; not 125M/full-vocabulary qualification |
| Transformer | Large numerical admission still fails | No qualified opponent ratio |
| Mamba3 | Historical slow regime did not recur in 192 admitted diagnostic calls | No newly qualified ordinary-request opponent ratio |
| UMAP | Historical 100k 5.2× headline compares different GPUs | No usable current same-GPU ratio |

See [performance status](PERFORMANCE_STATUS_2026-09-10.md),
[transfer evidence](../../bench/results/wp6_wp7_2026-09-10/README.md), and
[opponent cache](../../bench/OPPONENT_REFERENCE.md). No opponent was rerun for
WP6/WP7. Other classical estimators need a qualified baseline before describing
them as above or below 2×.

## Execution order and acceptance gates

### 1. Resolve NVIDIA trainer lifetime and reduce whole-model transfers

Start in `python/mojolearn/_byte_lm_impl.py`,
`bindings/_mojolearn_byte_lm.mojo` and `training/byte_lm.mojo`.
The unchanged WP6 baseline hung inside a second native call after stateless
training, including stateless→stateless and stateless→resident sequences.
Resident-only two-step training plus evaluation passed. Diagnose context and
buffer teardown/recreation with per-case deadlines; the cause is unresolved.
Do not call it a transfer-sweep regression or hide the failed sequence.

Gate: repeated stateless calls, repeated resident calls, mixed lifetime calls,
close/reopen, restore and failure recovery complete without hangs and preserve
all state/loss/gradient bits. Keep each diagnostic short with retained stacks
on timeout. Do this before timing an in-process stateless/resident comparison.

Then design a device-owned step API with a lean result and explicit state,
gradient and checkpoint exports. Current residency still downloads mirror
state and returns complete model/optimizer/gradient arrays every step. Specify
ownership, mutation rejection, failed-update recovery and export semantics
before removing copies. Preserve the existing API's failure atomicity; do not
silently weaken host-state validation. Gate with multi-step output equality,
checkpoint continuation, deliberate mutation and failed-update controls.

### 2. Bound vocabulary storage and measure complete training steps

Inventory the five `B*L*V` FP32 loss/logit buffers and all duplicate state,
activation and workspace allocations. Investigate lifetime reuse and tiled
loss/head processing only where the exact existing folds can be retained.
Compare loss, dlogits, gradients and updated optimizer/model bits; measure peak
memory separately from process RSS and label each memory boundary.

Choose an explicit runtime configuration and report its actual parameter
count. The existing 12-layer/DM768/FF2048/V50257 untied configuration is
162,147,840 parameters, not 125M, and uses RMSNorm/RoPE/SwiGLU rather than the
exact GPT-3 architecture. A GPT-3-Small-scale goal needs an agreed configuration;
changing shape is not a same-workload speed improvement.

Use the 20.45M/L2048/V8192 pilot as a control, then attempt full-vocabulary,
L2048, B1 target-scale steps only after capacity admission. A few steps at
large shape are sufficient; no corpus training run is needed. Report complete
step latency, tokens/s, memory and time fractions for GEMM, attention, loss,
optimizer and host transfers. Include synchronization needed for completion.
If setup or one step exceeds the experiment budget, record that limitation;
do not substitute a small model and label it target-scale qualification.

### 3. Attack the measured neural kernel costs

GEMM: inspect registers, spills, achieved occupancy and load traffic on the
actual large training shapes. Test one order-preserving change at a time:
register footprint, tile/K-step choice, vector loads, shared-memory swizzle,
then buffering where resources permit. The previous scalar-load trial failed;
do not repeat it without a new mechanism. Track useful TFLOP/s against the
hardware's FP32 non-tensor peak. The historical H100 30–40 TFLOP/s band is an
aspiration, not a gate or established achievable rate. Gate every dispatch
configuration with BITS MATCH and full-call timing, including ragged controls.
See [GEMM handoff](HANDOFF_speed_gemm_2026-09-10.md).

Attention: profile backward and repeated operand traffic; reuse/fuse work only
with identical arithmetic and reachable fallback checks. Measure forward and
forward+backward separately and then the full training step. Transformer
admission proceeds alongside this: isolate attention, QKV and RoPE errors
against the existing oracle, preserve tolerances, and require the original
large-shape gate to pass before publishing a torch ratio. Before/after bit
identity alone does not establish correctness against that oracle.

Mamba3 follows the training bottlenecks unless a bounded current profile makes
it a higher-value target. Measure ordinary calls on the original large grid
with explicit synchronization/timing boundaries. Keep latency diagnostics
separate from request prices. Do not reuse the old 72–85× headline.

### 4. Continue classical ML with kNN selection first

Keep the admitted aligned-load default. Profile/tune selection on
400k index rows / 4k queries / d32, k10 and k15; selection was nearly as costly
as distance at k15. Require complete distance/index equality, ties, exact
matches, ragged partitions, explicit candidate/default reach and sabotage.
Price ordinary search requests; classify uniform/distance separately. A tile
win alone cannot promote a request default. Expand dimensions, k or vendor
scope only after additional representative large-request gates.

Next inspect residual weighted-classification transfers and their host fold
order. Any device weight/probability route must preserve zero-distance rules,
ordering and output bits; WP7 did not already remove that work.

For GP, kernel methods, mixture, SVM, preprocessing and metrics, profile the
post-WP6/WP7 IDENTICAL public surface before choosing more changes. Distinguish
Python preparation, host copy, upload, kernel and download costs. GP's synthetic
factor win is not proof of a fitted-estimator speedup. NumPy/Array improvements
matter where those profiles show wrapper cost; do not assume they accelerate
native GEMM. For UMAP, measure current same-GPU large fit/transform boundaries
and their kNN/graph/optimizer shares before selecting work. Avoid an unbounded
million-row fit merely to refresh an old headline.

### 5. Finish training readiness after step throughput is measurable

Audit existing schedule, accumulation, clipping and weight-decay-group wiring
before implementing missing pieces. Define accumulation order under IDENTICAL.
Audit the checkpoint size limit and codec; qualify target-size save/load and
continued training with a bounded round trip. Wire token streaming/prefetch to
the generalized trainer while preserving token order. Cross-vendor checks must
cover generalized shapes and the optimized paths; unavailable hardware remains
RUN OWED. Estimate training duration from measured complete-step tokens/s with
stated checkpoint/data overhead, not GEMM-only extrapolation.

## Experiment and publication protocol

1. Pin source, binary, dependency versions, GPU/driver, fixture, shape, mode,
   dispatch and timing boundary, including opponent threading and BLAS. First rebaseline our current IDENTICAL surface
   after intervening Array/NumPy changes; retain historical numbers as history.
2. Small fixtures establish correctness and branch reach. Production-sized
   requests decide speed defaults. Use a hard 300-second measurement-process
   deadline, initially one warmup and three alternating timed pairs; use five
   pairs where affordable. Budget builds separately and bound rental lifetime.
   Interleave old/new arms in one process; retain both run orders, every sample
   and failed/drifting windows. Fewer samples mean a more limited claim.
3. Compare full outputs outside the timed region. Preserve original identity
   and oracle gates; deliberately corrupt each candidate path to prove reach.
   Promote a scoped winner in the same session only after correctness and
   representative whole-request improvement pass. Reject or retain as
   provisional when drift or incomplete coverage prevents that conclusion.
4. Reuse an opponent only for its matching hardware/software/shape/fixture and
   boundary. If a necessary tuple is absent, measure it once and add provenance
   to `bench/OPPONENT_REFERENCE.md`. Our before/after result is not an opponent
   row. Failed numerical admission means an unqualified opponent ratio.
5. Commit source, checks and evidence with corrected labels; push to main
   without overwriting concurrent changes. Keep archive/binary artifacts out
   of `bench/results` per repository hooks; publish hashes and durable locations.
   Terminate scoped rentals and record cleanup.

Aim below 2×, with below 3× an intermediate target. No proven IDENTICAL
slowdown floor above either threshold has been established. Estimate bounds
from required work and traffic (`max(FLOPs/allowed_peak, bytes/bandwidth)`),
then account for dependencies and launches. Use measured step fractions to
estimate whole-training payoff. There is still concrete work to do; reasonable
full-training time has not yet been demonstrated.
