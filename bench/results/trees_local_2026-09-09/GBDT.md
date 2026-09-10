# GBDT fusion and partition candidates, Apple M4, 2026-09-09

Fusion remains opt-in. The larger fit experiment does not justify enabling
it by default: FAST's aggregate improvement reverses with run order, and
IDENTICAL regresses in the aggregate. No NVIDIA hardware was used.

## Validation

`checks/gbdt_fused_move_check.mojo`, built FAST and IDENTICAL, compares the
split and fused kernels bitwise for all 12 single-target objectives, both
estimation/search layouts, weighted/unweighted inputs, repeated moves,
ragged tails, and scattered bin assignments. All 48 cells plus five
1,000,003-row Logloss cells pass in each mode. It compares cursor,
derivative planes, function-value partials and magnitude partials.
Logs: `gbdt-fused-{fast,identical}.log`; each prints its compiled mode.
These kernel microbenchmarks are not end-to-end speed claims.

`tools/gbdt_fused_ab.sh` builds isolated native binaries without touching
installed Python bindings. It compares splits, leaf values, predictions
and all Float64 loss bits. The baseline/fused fingerprints agree for all
96 fits (including warmups), within each mode and dataset size. Compiled
numeric mode and fusion flag are verified from each binary's output.

Model: 8 synthetic numeric features, symmetric Logloss, 20 trees, depth 6,
32 borders, learning rate 0.2, 10 leaf estimation iterations. Each arm has
two passes, each with one warmup and five timed fits. Arm order reverses
on the second pass. Data generation and prediction are outside fit timing.
Compilation and timing both hold the build lock; timing also holds the
benchmark lock.

| Mode | Rows | Baseline median ms | Fused median ms | Baseline / fused |
|---|---:|---:|---:|---:|
| FAST | 65,537 | 201.460 | 191.752 | 1.051 |
| FAST | 1,000,003 | 488.581 | 467.386 | 1.045 |
| IDENTICAL | 65,537 | 267.664 | 274.131 | 0.976 |
| IDENTICAL | 1,000,003 | 568.751 | 609.684 | 0.933 |

The FAST 1M pass medians change direction: baseline/fused approximately
472/496 ms in pass 1 and 547/464 ms in pass 2. Thermal/order effects are
large enough that the aggregate 4.5% is not convincing evidence for a
shipped default. Detailed samples: `gbdt-fit-v2-summary.log` and
`gbdt-fit-v2/*.log`.

The v2 driver completed all 16 run logs, then its summary shell step failed
because the running script was edited. The exact summary was rerun against
those complete logs, including mode, flag, sample-count and fingerprint
assertions. The driver failure is preserved in `gbdt-fit-v2-driver.log`.
The driver now parses its complete body before execution. The earlier
`gbdt-fit/` run hashed losses after Float32 conversion and is superseded by
v2; do not use it as evidence of full-width loss identity.

## Controls

- Fusion opt-in: `-D MOJOLEARN_2030_FUSED_EST_MOVE=1`.
- Fusion force-off: `-D MOJOLEARN_2030_NO_FUSED_EST_MOVE=1`, taking
  precedence over opt-in. A host check importing the production constant
  passed with both defines present (`gbdt-force-off.log`).
- New NVIDIA IDENTICAL partition opt-in:
  `-D MOJOLEARN_IDENTICAL_SINGLE_PASS_PARTITION=1`. The existing
  `MOJOLEARN_2042_FAST_NO_LOOKBACK` kill switch takes precedence.
  Apple and AMD routing is unchanged, and IDENTICAL remains off by default.

`checks/gbdt_partition_route_check.mojo` passes default, opt-in, and
opt-in-plus-kill-switch builds (`gbdt-partition-routes.log`). This validates
routing only. A NVIDIA device test above 500,000 rows per leaf is still
required before making the IDENTICAL single-pass partition default.

Reproduce fit A/B:

```sh
tools/gbdt_fused_ab.sh bench/results/gbdt_fused_ab
```
