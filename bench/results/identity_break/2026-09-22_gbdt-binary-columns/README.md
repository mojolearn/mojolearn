# gbdt-binary-columns, 2026-09-22

Source: commit `602000d1c` (perf/gbdt-host-one-border-sep22). The lane is new:
five columns of every fixture become 0/1 flags (the top mantissa bit of the
value), so each quantizes to exactly one border, the BinaryFeatures histogram
policy. No other lane's fixture has such a column, so no existing cell moved.

| file | column | build |
|---|---|---|
| `apple-m4.json` | Apple M4, Metal, device route pinned by the harness | round 2 device binding (device code unchanged since main 536ca4048) |
| `cpu-apple-m4.json` | CPU host binding on the M4 | clean host build of `602000d1c` |
| `cpu-apple-m4.sabotage-binary.json` | CPU host binding on the M4 | the same source with `-D MOJOLEARN_GBDT_BINARY_SABOTAGE=1` |

All nine fixtures, all rows, one fit per cell. Apple and CPU agree on every
part: train 9 of 9, infer and model 18 of 18, batch 9 of 9 IDENTICAL. The
sabotage build (nibble value 0 left out of a binary feature's sum, on the
Plain and the Ordered writebacks alike) reads DIVERGENT on the train part of
all nine fixtures; on `base` it moved the Plain RMSE and every Ordered RMSE
fit, on `ties` the Plain and Ordered Logloss fits, so both host restatements
are reached. Summing the set side instead is not a control: a one-fold
feature's cosine score is symmetric in its two sides, and that build read
IDENTICAL.

## NVIDIA, AMD and x86-64 CPU columns (admitted 2026-09-22)

| file | column |
|---|---|
| `nvidia-h100.json` | NVIDIA H100, CUDA, commit 3e1b87ccf (RunPod) |
| `nvidia-rtx4090.json` | NVIDIA RTX 4090, CUDA, commit 91f713c06 (RunPod) |
| `amd-mi325x.json` | AMD MI325X, HIP, commit 3e1b87ccf (DigitalOcean) |
| `cpu-amd-epyc-9965-x86_64.json` | x86-64 CPU host binding, commit 3e1b87ccf (RunPod CPU) |

Recorded with `--require-backend cuda|hip --repeats 1 --batch-grad
--batch-scale --ragged --step-full`. `diff.five-columns.txt`, all six columns:
train 9, infer and model 18, batch 9 IDENTICAL, exit 0. Admitted into
`verify_reference/table.json`, and `ensemble._HOST_ONE_BORDER_VENDORS` is now
metal, cuda and hip.
