# 136 lanes, three columns, at 4048e1b51 (2026-09-14): the CPU gate's GPU columns

The record `python/mojolearn/host_surface.py` names as TRAINING_GPU_COLUMNS from
this commit on. Commit 4048e1b51 is main with the DEVIATION 2712/2713 fix
(m2_ydiag_kernel's X_d read bounded, every Mamba device buffer filled) and the
sixteen multi-GPU driver lanes (`par-forest`, `par-forest-et`, `par-boosting`,
`par-kmeans`, `par-gram`, `par-logistic`, `par-cd`, `par-svm`, `par-gp`,
`par-dbscan`, `par-scaler`, `par-arima`, `par-mlp`, `par-samba`, `par-byte-lm`,
`par-iforest`), each run with one device (`MOJOLEARN_PAR_DEVICES` at its
default "0") and the smallest sharding that exercises the split; thirteen hold
equality to the plain fit through `_same_bytes` and would read REFUSED with the
pair named if a byte differed, the three neural trainers hash state after
ordered-shard steps. Nine hostile fixtures, two fits per cell, train, infer and
model parts; every binding rebuilt from this commit on each box.

| column | box | how | cells |
|---|---|---|---|
| apple-m4 | Apple M4 (Metal), this Mac, one core | every binding rebuilt in the worktree (27 builds), `tools/identity_break.py --vendor apple-m4` | 1224 stable, 0 moved, 0 refused |
| nvidia-h100-sm_90a | RunPod H100 | `tools/identity_three_columns_leg.sh` through tools/gemm_remote_leg.sh, leg `bench/results/e1g/2026-09-14_123934-nvidia` | 1224 stable, 0 moved, 0 refused; against apple-m4 IDENTICAL=1224 and (infer/model) IDENTICAL=1665, N/A=783 (`diff.apple-h100.txt`) |
| amd-mi300x-gfx942 | Hot Aisle MI300X 8core, 22.04 ROCm container | the same body through tools/hotaisle_leg.sh | (filled when the leg lands) |

`diff.apple-vs-120-lanes-2712fix.txt`: this Apple column against the 120-lane
record's at 9ade3e2cd reads IDENTICAL on all 1080 shared training cells and
1476 inference and model cells; the 144 ONE-COLUMN cells are the sixteen new
lanes. The two-device column of the par-* lanes, `nvidia-2xh100-sm_90a.par-devices-0-1.json`
(RunPod pod with two H100s, `MOJOLEARN_PAR_DEVICES=0,1`, recorded in its `package.par_devices`,
leg `bench/results/e1g/2026-09-14_131123-nvidia`, every binding rebuilt from this commit): all
144 training cells and 144 inference cells of the sixteen lanes STABLE, and against apple-m4 and
the one-device nvidia-h100-sm_90a column `--require-columns 3` over the sixteen lanes reads
IDENTICAL x3 on every cell (`diff.par-two-devices.txt`). What that says, exactly: the equality is
of BITS (the cell hashes), the sixteen cells are the drivers' one-device replay contract exercised
with the smallest sharding that splits the work, and it is a SAME-VENDOR two-device result (two
H100s against one H100 and an Apple M4 running one device) until the Hot Aisle 2gpu column lands;
"IDENTICAL x3" here is not a three-vendor two-device claim. No throughput number is part of this
record.
