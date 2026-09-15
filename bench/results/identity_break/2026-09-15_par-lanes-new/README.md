# Eight par lanes on two H100s and two MI300X, one device against two

Two groups of `tools/identity_break.py` lanes, each run twice on one RunPod
box with two GPUs: once with `MOJOLEARN_PAR_DEVICES=0` (`one.json`) and once
with `MOJOLEARN_PAR_DEVICES=0,1` (`two.json`), all nine fixtures, then
`--diff one.json two.json` (`diff.txt`). `body.sh` is the leg body, `pod_id.txt`
the pod, `gate.txt` the box's own summary.

## The four new lanes (`*-new8`)

`par-cholesky` (a 600 x 600 Cauchy SPD factor, 1.44 MiB, and three
right-hand sides), `par-kernel-ridge` (600 rows, two targets),
`par-nystroem` (32 components of 600 rows) and `par-rbf-sampler` (1024
features over 1000 rows in shards of 300), each held by `_same_bytes` to the
plain one-device path, beside their plain lanes `cholesky`, `kernel-ridge`,
`nystroem` and `rbf-sampler`.

| directory | GPUs | pod | source | one vs two |
| --- | --- | --- | --- | --- |
| `nvidia-2xh100-new8` | 2x NVIDIA H100 80GB HBM3, sm_90a | `egfauhq0zuq2jl` | `e520c8f49` | IDENTICAL=72; infer/model IDENTICAL=72, N/A=72; batch IDENTICAL=72 |
| `amd-2xmi300x-new8` | 2x AMD Instinct MI300X, gfx942 | `bz2o7gsyaryg2g` | `8977e1816` | IDENTICAL=72; infer/model IDENTICAL=72, N/A=72; batch IDENTICAL=72 |

`8977e1816` differs from `e520c8f49` only in
`training/checks/peer_copy_check.mojo`, which no lane builds (the AMD body's
`commit.txt` names `e520c8f49` because the body was written at `e520c8f49` and the leg shipped the next commit;
`leg.txt` in the raw leg directory records the shipped commit).

`diff.new8-four-columns.txt` puts the four JSONs side by side with
`--require-columns 4`: `summary: IDENTICAL=72`, infer/model `IDENTICAL=72,
N/A=72`, batch `IDENTICAL=72`. No cell reads REFUSED, so every `_same_bytes`
hold inside the par lanes passed on both vendors and both device counts, and
the H100 and MI300X cells are equal. The N/A cells are the lanes' declared
`n/a:no-save` model parts.

## The four lanes the multigpu lane added (`amd-2xmi300x-old4`)

`par-forest-pool`, `par-gmm`, `par-resample` and `par-hdbscan` beside
`rf-clf`, `gmm`, `bootstrap`, `hdbscan` and `par-forest`, on two MI300X at
`e520c8f49` (pod `34svay8axa1gpk`): IDENTICAL=81; infer/model IDENTICAL=72,
N/A=90; batch IDENTICAL=45, N/A=36, the same counts as the two-H100 run at
`f067bbbc0` (`bench/results/multi_gpu/2026-09-14/par-lanes-h100/`).
`diff.old4-four-columns.txt` diffs the MI300X one and two against the H100
one and two with `--require-columns 4`: IDENTICAL=81, 72 and 45. The two
vendors' columns are at different commits. Between `f067bbbc0` and
`e520c8f49`, in the drivers these lanes reach, the Python files only gained
the Cholesky and kernel-method entries (no line removed), the mixture,
resample and HDBSCAN bindings lost the stale `*_parallel_available` flags
that no driver read, and `mixture/estimator.mojo` changed only its
docstring.

## What this is not

Same-box one-device against two-device equality, plus equal hashes across
the two vendors. No throughput is measured, and no Apple column of these
lanes is recorded here.
