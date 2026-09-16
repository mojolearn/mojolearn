# lane/rf-score-weighted-nondeterminism

**Status 2026-09-16: REPRODUCED and LOCALIZED to a stage. NOT root-caused. NO FIX LANDED.**
The 0.8.6 blocker stands. Nothing here changes any shipped code; this branch adds the probe
tooling and this note only.

## What the defect is, corrected

`rf-score-weighted` moves between two fits on one MI300X. Two claims in the original framing
are now measurably wrong, and the correction matters for where to look:

1. **The clf/reg asymmetry is NOT an output-shape artifact.** It was argued that the classifier
   returns an argmax whose strict comparison swallows a last-bit difference, so a perturbation
   anywhere upstream would produce this pattern. Measured: over 80 fits the classifier's
   **model itself** is bit-stable in all five exported arrays. Its trees never moved at all.
   The defect is specific to the regressor.
2. **The weighted metric path is exonerated.** `reg_unweighted` passes no weights and moves with
   the other three reg parts (leg 3 `/wide` `5950fe9897edc56c`, leg 4 `/base` `a0adfc1ac03e495d`).

## Reproduction (one Hot Aisle MI300X, gfx942, ~10 s of compute)

80 fits of the lane's exact configuration from the installed 0.8.6 release wheel:

| fixture | stable cell | stable | moved |
|---|---|---|---|
| wide | `49be8ea935a47640` | 39/40 | 1 (`5f7a3bd31486fb53`) |
| base | `d744878e7c0e31ee` | 38/40 | 2 (`6debf9a1b42c0d43`, `1b5a80605dd90a43`) |

About 4% of fits. Every odd value is **distinct** and none equals leg 3's `50ce4a9f62cddf8e`
or leg 4's `eb475cefa0f32408`, so this is a race, not two deterministic code paths.
The stable values match NVIDIA and the CPU column exactly.

## First differing stage

Hashing the five exported model arrays per fit, and per tree:

- **Never move:** `_offsets`, `_left_child`. Node counts are identical every run
  (wide 7592, base 6718). **The tree SHAPE is invariant.**
- **Move:** `_colid` (the chosen split feature) and `_quesval` (its threshold), usually
  dragging `_leaves`.
- **Exactly ONE tree differs per occurrence** (observed: tree 0, 0, 7, 3).
- wide repeat 4 moved `_colid`+`_quesval` but left `_leaves`, predict and the cell hash
  untouched: an *equivalent* split, i.e. DEVIATION 105's equal-gain tie class.

So: same topology, one node in one tree picks a different feature at equal gain.

## Ruled out, with the reason

- **Weighted metrics** — `reg_unweighted` moves (above).
- **Label scale** (the one regressor-only fit-wide scalar, `bindings/_mojolearn_rf.mojo:594-600`):
  the `mag` loop reads `yp[i]`, the caller's own numpy buffer, after `ctx.synchronize()`, so it is
  a serial Float64 fold over stable host memory. `choose_scale` also snaps to a power of two and
  every fixture's Sigma|y| sits far from a boundary (fractional log2 0.14 to 0.92), so even a
  perturbed `mag` could not flip the scale.
- **Float reduction ordering in the histogram** — `RegressionBin` is `label_sum: Int32` fixed
  point plus `count: UInt32`, both relaxed **integer** atomics; order-free. No dither on this
  path (`split.mojo:37`); `hist2_quantize`'s dither is the pointwise/GBDT path.
- **Wavefront-64 grouping (DEVIATION 104/105)** — DEVIATION 404 pins the reduce to 32 lanes
  under IDENTICAL and the shipped wheel is IDENTICAL mode, yet it still moves.
- **Device-property-derived launch shapes** — there is no `get_attribute`, occupancy or CU-count
  query anywhere on the fit path; the one SM-shaped number (`4 * 108` in `core/philox.mojo`) is a
  frozen stride, and the bootstrap and feature draws are pure functions of
  `(seed, tree_id, node index, element index, n)`.
- **Cross-device peer copy (the Sep 15 MI300X hazard)** — single-device box, no peer copy on
  this path.

## NOT ruled out, in priority order

1. **The pipelined K=4 shared `SplitStaging`.** `_enqueue_splits_download` only records a byte
   count under shared staging; `flush_splits_downloads` sends one prefix copy spanning slots,
   then one `ctx.synchronize()`, then each slot reads its own host bytes. "Exactly one tree per
   occurrence" fits a per-slot staging race. `builder.mojo` ~1690-1775, `ensemble/randomforest.mojo`
   2727-2810.
2. **Shared-memory histogram zeroing extent.** Zeroed only over the live prefix, and
   `SMEM_BIN_SLOTS = 16 KiB / size_of[BinT]` differs between `RegressionBin` (8 B) and
   `ClassificationBin` (4 B) — a regressor-only stale-shared-cell read would look exactly like this.
3. **The compare-then-skip H2D caches** (`DeviceArgs.staged`, `ops_staged`), which
   `reset_for_tree` deliberately does not reset.

## THE NEXT EXPERIMENT, and it is cheap

`n_streams` is a public constructor parameter, so serializing the pipeline needs no rebuild:

    RandomForestRegressor(n_estimators=16, max_depth=8, random_state=7, n_streams=1)

Run the probe with `n_streams=1` beside the default 4, 40 repeats each. **If K=1 is stable and
K=4 moves, the race is the shared staging/pipeline (candidate 1) and the fix is a per-slot sync
or an unshared staging. If both move, it is inside one tree's kernels (candidates 2 and 3).**
One leg, about 10 s of GPU time, roughly $0.05.

## Evidence

Outside the repo, per the no-blobs-in-git rule:

    ~/mojolearn-evidence/rf-score-weighted-blocker/leg-1/
      remote/identity/rf_probe.json     80 fits + 30 n-sweep, per-array and per-tree hashes
      remote/identity/record.txt        wheel sha256, import line, timings
      leg.txt, teardown.txt, vm_states.txt

The n-sweep (n = 500..8000, 3 repeats each) read `distinct=1` everywhere. **That is not evidence
of n-independence**: at a 4% per-fit rate, 3 repeats expects 0.12 moves per cell. It is
underpowered and should not be cited either way.

## Resume

    WT=$(mktemp -d)/wt && git worktree add --detach $WT origin/main
    S=$WT/tools/rf_nondeterminism
    bash $S/make_rf_probe_body.sh /tmp/rf_probe_body.sh 40 1500      # mints presigned GET+PUT
    cd $WT && MOJOLEARN_HOTAISLE_SPEC=8core MOJOLEARN_GPU_ARCHS=gfx942 \
      MOJOLEARN_GEMM_LEG_EXTRA=/tmp/rf_probe_body.sh \
      MOJOLEARN_GEMM_LEG_OUT=$HOME/mojolearn-evidence/rf-score-weighted-blocker/leg-2 \
      MOJOLEARN_HOTAISLE_LANE=rf-score-weighted-probe \
      bash tools/hotaisle_leg.sh amd --rent --minutes 40 --skip-gates   # drop --rent to dry run

Edit `rf_probe_body.template.sh`'s `one_repeat` to add the `n_streams=1` arm before renting.

## Box and cost

Hot Aisle 1x MI300X 8core, VM `8796a71c-a8c2-4620-9116-4def52b1baa5`, 2026-09-16 09:56 to 09:59 UTC.
DELETE returned HTTP 204, then GET 404 and absent from the listing (`verified_gone ... yes
get=404 list=200 listed=no`). Team balance $32.78 to $32.73, so about **$0.05** observed
(3 minutes at $2.99/h is about $0.15; provider billing may lag). One box, one at a time, deleted.
