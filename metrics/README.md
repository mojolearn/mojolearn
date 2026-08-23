# metrics: cuML `cpp/src/metrics/` and the RAFT `raft/stats/` headers it calls

Seventh section. **COPY, DO NOT IMPROVE.** cuML's metrics are thin wrappers
(`metrics/ported/metrics/`, one file per `.cu`) over RAFT's `raft::stats`
(`metrics/ported/stats/detail/`, one file per `.cuh`). Pinned: cuML 265b9da,
RAFT ebf9268 (the v26.08.00 checkouts; `PORTED_MAP.tsv`). sklearn is read
for SEMANTICS only and its formula stands beside each of ours in a comment.

## Status

CONSTRUCTION plus one Apple device's gates; no second vendor has run this.

Group A (the label metrics) is ported, gated in both modes, and sabotaged.
`r2_score` (Group B) is WRITTEN in `scores.mojo` and smoke-run, NOT YET
GATED: its checks land with the Group B commit. KL, Groups C (silhouette)
and D (trustworthiness) are recorded below as they land; a metric absent
from the per-metric table has no gate yet.

## Commands

No pixi task (pixi.toml is not this lane's). Every device run goes
through the build lock:

    tools/with_build_lock.sh     pixi run mojo run -I . metrics/mojo_only/label_metrics_check.mojo
    tools/with_identical_mode.sh pixi run mojo run -I . metrics/mojo_only/label_metrics_check.mojo

Every printed line carries the mode the binary COMPILED in.

## The one idea, stated once: integer atomics ARE identity-safe, float atomics are not

RAFT's label metrics all start from an INTEGER contingency matrix or
histogram built with `atomicAdd` on `int`. Integer addition is associative
and commutative, so whatever interleaving the hardware produces, the
counter holds the same bits at the end -- on Metal, PTX and AMDGPU, run
after run, in BOTH numeric modes. The device half of every Group A metric
is therefore exact and needs no IDENTICAL arm, and the checks compare it
EXACTLY, per cell.

The float half of those metrics is small -- a few dozen `-p log p` or
`c (log(nc) - log(ab))` terms -- and RAFT folds it on the device in double
with `atomicAdd`, an arrival order. Under IDENTICAL we do not replace the
accumulator with a fixed tree here; we move it. DEVIATION 650: the host
reads the integer matrix back and performs the float ops serially,
ascending. DEVIATION 651: under IDENTICAL those ops are Float32 through
`identical_log` / `identical_mul_add` / `ftz` (one arithmetic on every
vendor and every host; no portable Float64 log exists in
`mojo_only/numerics.mojo`), and under FAST they are RAFT's Float64 through
the host's `log`. IDENTICAL's value therefore differs from FAST's in the
seventh digit by design, and is the same bits everywhere.

The float metrics (Groups B, C) are different: a reduction over `n` of
Float32 terms, where the fold shape IS the number. There the accumulator is
replaced (DEVIATION 653, `metrics/mojo_only/pinned_sum.mojo`): a
`PINNED_SUM_W = 256` slab tree whose width is a constant of the source and
not the launch's block size, per-chunk partials folded ascending on the
host, `ftz` at every stored seam, gated bit for bit against a host model
of the same additions and launch-invariant across two block sizes and two
grid shapes.

## Per-metric table

| metric | cuML file | RAFT file | ported | refused | gated (mode) | sabotaged |
|---|---|---|---|---|---|---|
| accuracy_score | accuracy_score.cu | scores.cuh | yes | -- | count EXACT, value bitwise (both) | via (a) below, the kernel family |
| rand_index | rand_index.cu | rand_index.cuh | yes (DEV 652) | -- | (a, b) EXACT vs O(n^2) host loop AND sklearn pair-confusion; value bitwise (both) | (a) |
| adjusted_rand_index | adjusted_rand_index.cu | adjusted_rand_index.cuh | yes | int64 overload | bitwise vs host formula (both); 4 early returns | (a) |
| entropy | entropy.cu | entropy.cuh | yes (DEV 650, 651) | -- | IDENTICAL bitwise vs oracle; both vs Float64 ref 1e-6; constant -> 0, size 0 -> 1 | (c) null, recorded |
| mutual_info_score | mutual_info_score.cu | mutual_info_score.cuh | yes (DEV 650, 651) | -- | IDENTICAL bitwise; both vs ref; MI == 0 exactly on a product-structured labeling | (b) |
| homogeneity / completeness / v_measure | homogeneity_score.cu, completeness_score.cu, v_measure.cu | homogeneity_score.cuh, v_measure.cuh | yes | -- | IDENTICAL bitwise; both vs ref; constant-truth (1,0,0), constant-pred (0,1,0), both-constant (1,1,1), independent (0,0,0), singleton | (b) |

## Group A: what the gates printed (Apple M4, 2026-08-23)

FAST and IDENTICAL, `label_metrics_check.mojo`:

    check_contingency_matrix_exact    k=7 SMEM arm, minLabel=3: 49 cells EXACT, sum 4099
                                      k=31 SMEM arm ... 961 cells EXACT
                                      k=40 GLOBAL arm ... 1600 cells EXACT
    check_accuracy_exact              count 3315 of 4099 EXACT; value 0.8087338 0x3f4f092e bitwise
    check_rand_index_exact            a=1942300 b=4572497 EXACT (two host tallies agree)
                                      rand_index 0x3fe8d258ef472429 bitwise
    check_adjusted_rand_index         0x3fe01ca6e2aee9c2 bitwise; singleton 0x3fe0192ed2b0394e bitwise
    check_entropy_epilogue            IDENTICAL: 0x3ff7ecef00000000 bitwise vs oracle, rel 4.1e-09 vs Float64
                                      FAST: 0x3ff7eceefe5c3322, rel 3.7e-13 vs Float64 (bitwise is a REPORT)
    check_mutual_info_epilogue        IDENTICAL: 0x3fdadbbb00000000 bitwise, rel 1.5e-07; FAST rel 1.1e-10
    check_homogeneity_completeness_v  IDENTICAL: h/c/v bitwise; perfect -> h 1.0 exactly;
                                      FAST: perfect -> h 0.9999999999919665 (MI and H are two
                                      spellings of one quantity; RAFT does not promise equality)
    check_label_epilogue_order_is_visible
                                      MI ascending 0x3ed6ddd8, descending 0x3ed6ddd9: the
                                      fixture separates a reversed fold by one ulp, so the
                                      bitwise gates have teeth

**Completeness is the transposed fold.** `completeness_score.cu` calls
`homogeneity_score(y_hat, y)`, so RAFT computes a SECOND mutual information
over the transposed contingency matrix. A transposed fold is a different
Float32 sum: on the singleton fixture `MI(truth, pred) = 0x3fe0589f00000000`
and `MI(pred, truth) = 0x3fe0589ee0000000`. sklearn computes one MI and
uses it for both h and c. Ours mirrors RAFT (two folds), and so does the
oracle; the first draft of the oracle used one MI and the IDENTICAL gate
caught it on the singleton fixture (`v_measure(singleton) got
0x3fd8213c1ffcaab2 oracle 0x3fd8213c36e1d2cb`), which is the gate doing
its job.

**sklearn semantics that differ from RAFT and are mirrored as RAFT:** the
0/0 case of ARI returns 0 (sklearn warns and returns NaN); entropy's
`log(pi) - log(pi_sum)` grouping versus RAFT's divide-then-log (same
quantity, last bits differ, both within 1e-6 of each other in the gates).

**One integer bug of theirs not ported:** `mutual_info_kernel`'s
`a[i] * b[j]` is an `int` product (overflows past 46,340 samples in one
class pair); ours is Int64. Recorded in `mutual_info_score.mojo`.

## Sabotages (each reverted in the same session; a check that cannot fail is not a check)

| # | what was broken | check | failing output |
|---|---|---|---|
| (a) | contingency index written `(pd - off) * width + gt - off` (transposed) | check_contingency_matrix_exact | `cell 0,1 device 160 host 185 ... contingency matrix k=7 (SMEM arm): 38 cells differ` |
| (b) | the ported MI epilogue's (i, j) loop run DESCENDING | check_mutual_info_epilogue (IDENTICAL) | `BITWISE MISMATCH mutual_info_score got 0.4196613132953644 (0x3fdadbbb20000000) oracle 0.419661283493042 (0x3fdadbbb00000000)` |
| (c) | `ftz` dropped from the entropy epilogue's stored term | check_entropy_epilogue (IDENTICAL) | NO CHANGE on Apple: no term is subnormal on the label fixtures and the host flushes nothing either; recorded as the expected null. The FTZ seam that CAN be reached is Group B's (planted subnormal squares), where the sabotage must fail |

## Deviations spent

| n | where | what |
|---|---|---|
| 650 | entropy.mojo (banner), mutual_info_score.mojo, adjusted_rand_index.mojo | the label metrics' float epilogue runs on the HOST, serially, from the INTEGER device product; readback is k or k^2 ints instead of one scalar |
| 651 | entropy.mojo (banner), mutual_info_score.mojo | IDENTICAL: Float32 through identical_log / identical_mul_add / ftz; FAST: Float64 host log (RAFT's precision) |
| 652 | rand_index.mojo | 64-bit atomic replaced by per-block Int32 partials summed on the host in Int64 (Apple has no 64-bit atomic; a 32-bit total overflows past 65,536 samples) |
| 653 | mojo_only/pinned_sum.mojo, scores.mojo (r2) | the float sums over n as one fixed slab tree + ascending host fold, launch-invariant by construction; FAST arm is block.sum |

## ROW TEXT FOR THE IDENTITY LANE

| n | path | what is vendor-dependent in their spelling | what we did | status |
|---|---|---|---|---|
| (next) | metrics: label metrics (accuracy, rand, ARI, entropy, MI, h/c/v) | contingency matrix and histograms are INTEGER atomics (order-free, identity-safe on every vendor); the float epilogue is a device double `atomicAdd` fold (arrival order) with the vendor's `log` | device product stays integer and is gated EXACTLY per cell; the float epilogue moves to the host, serial ascending, Float32 through identical_log/identical_mul_add/ftz under IDENTICAL (DEVIATIONS 650, 651); completeness is the transposed fold as theirs; gated bitwise vs an oracle and 1e-6 vs a Float64 sklearn-spelled reference; two sabotages fail, one null recorded | Apple FAST+IDENTICAL green; no second vendor |

## HAND-OFF (Python surface, for the bindings lane)

Names, mirroring `cuml.metrics`: `mojolearn.metrics.accuracy_score(y_true,
y_pred)`, `rand_score`, `adjusted_rand_score`, `mutual_info_score`,
`homogeneity_score`, `completeness_score`, `v_measure_score(beta=1.0)`,
`entropy(labels)` (cuML exposes `cuml.metrics.cluster.entropy`). Each takes
int32 labels; the Python side computes `lower_class_range = min(y, y_hat)`
and `upper_class_range = max(...)` exactly as cuML's `.pyx` files do and
passes them to the Mojo entry; a label outside the range is a host-side
raise by name (RAFT writes out of bounds; we do not port that). Float64
inputs to the float metrics are refused by name (no Float64 on device).
pixi lines to register when pixi.toml is next edited by its owner:

    check-metrics = "mojo run -I . metrics/mojo_only/label_metrics_check.mojo"

## HAND-OFF TO THE IDENTITY LANE

Nothing outside `metrics/` was edited. Two asks, neither blocking:
1. `mojo_only/numerics.mojo` has `identical_exp64` but no `identical_log64`;
   a portable Float64 log (Cephes, the `portable_exp64` shape) would let
   DEVIATION 651 keep RAFT's double precision under IDENTICAL. Until then
   IDENTICAL is Float32 at these seams.
2. IDENTITY_PATHS.md: the row above.
