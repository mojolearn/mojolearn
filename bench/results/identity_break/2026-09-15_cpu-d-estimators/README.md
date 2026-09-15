# CPU training for the workstream D estimators, the M4 host column (2026-09-15)

Branch `lane/cpu-training-d-estimators`. The eight lanes the 166-lane record
gave their first GPU columns, trained on the CPU through four host families,
and diffed with `tools/identity_break.py --diff --require-columns 4` against
the record's three GPU columns (`bench/results/identity_break/2026-09-14_166-lanes/`,
Apple M4, NVIDIA H100 sm_90a, AMD MI325X gfx942), exactly as the CPU identity
gate does.

| lane | host family | host restatement |
|---|---|---|
| cholesky | gp | `cholesky/host/chol_oracle.mojo` (the door already on the gp host binding) |
| rbf-sampler, kernel-ridge, nystroem | kernel_methods (new) | `kernel_methods/host/km_host_oracle.mojo` |
| gmm, gmm-random-init | mixture (new) | `mixture/host/gmm_host_oracle.mojo` |
| hdbscan, hdbscan-leaf | hdbscan (new) | `hdbscan/host/hdbscan_host_oracle.mojo` |

The four gp lanes are in the same run because `chol_oracle.mojo` gained
`chol_host_factor_lower` (the factorization without the door's validation, which
the kernel ridge solve and the mixture's precision Cholesky reach on the device);
`chol_host_potrf` now calls it after its two refusals.

Where it ran: the Apple M4, one core, shared machine, host bindings built from
the branch's working tree (the column JSON records commit 9b7ee8960, the branch
start, because the tree was uncommitted when it ran), fresh output directories
`host-v2` and `host-sab-v2`.

| file | what |
|---|---|
| `cpu-apple-m4.json` | the CPU column, 12 lanes, 9 fixtures, 2 repeats |
| `diff.four-columns.txt` | `summary: IDENTICAL=108`, `summary (infer/model): IDENTICAL=90, N/A=126`, `summary (batch): IDENTICAL=90, N/A=18`, `require-columns 4 ... : OK` |
| `cpu-apple-m4.sabotage.json` | the same lanes through `-D MOJOLEARN_HOST_SABOTAGE=1` builds |
| `diff.four-columns.sabotage.txt` | `summary: DIVERGENT=108`, `summary (infer/model): DIVERGENT=86, IDENTICAL=4, N/A=126` |

The four infer cells (and the four batch cells) that stay IDENTICAL under the
sabotage are `cholesky/denormal`, `cholesky/denormal_ftz`,
`gmm-random-init/denormal` and `gmm-random-init/denormal_ftz`, the held-out
evaluations on the denormal fixture pair, which the moved folds do not reach
in their last bit; the train cells of all four are DIVERGENT.

The sabotage arms: kernel_methods and mixture move through
`gemm/host/gemm_oracle.mojo`'s descending leaf (every kernel matrix, the
Cholesky trailing update, the normalization, the feature map, the means and
covariances); hdbscan reads every core distance one neighbor early; gp walks the
scaled distance's feature axis descending, and the Cholesky door moves through
the same gemm leaf.

Owed: the seven-runner CPU identity gate on the branch (the bit claim on
Linux ARM64, x86-64 and hosted macOS is the gate's, not this directory's).
