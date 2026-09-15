# CPU training for the Mamba block lanes, the M4 host column (2026-09-15)

Branch `lane/cpu-training-mamba`. The `mamba2`, `mamba2-dtlimit`, `mamba1` and
`mamba3` lanes of `tools/identity_break.py` (the three blocks at d_model 32;
`mamba2-dtlimit` with `dt_limit=(0.01, 0.1)`), run on the CPU through the new
mamba host family and diffed with `tools/identity_break.py --diff
--require-columns 4` against the 166-lane record's three GPU columns
(`bench/results/identity_break/2026-09-14_166-lanes/`, Apple M4, NVIDIA H100
sm_90a, AMD MI325X gfx942), as the CPU identity gate does.

Each train cell hashes four parts: the stateless forward, the carried-state
prefill, one decode step after it, and the zero-state prefill backward (the
input gradient and every weight gradient). The infer cell is the forward on a
held-out slab; the batch cell is each sequence alone against eight and every
prefix length against 16. There is no model cell (the blocks have no save).

| piece | where |
|---|---|
| binding | `bindings/_mojolearn_mamba_host.mojo` (every entry of the GPU binding, same address and params contract) |
| forward, prefill, decode step | `mamba/checks/mamba_oracle.mojo`, `mamba2_oracle.mojo`, `mamba3_oracle.mojo` (the host Float32 oracles) |
| backward | `mamba/host/gen/`, the device prefill VJPs and their forward stages written out for the host by `tools/mamba_host_gen.py`: each kernel a serial loop over its launch grid, the device GEMM through `gemm_oracle` (`mamba/host/device_shim.mojo`), the shared-memory tiled Mamba-3 kernels replaced by their unshared arms |
| sabotage | `gemm/host/gemm_oracle.mojo::GEMM_ORACLE_HOST_SABOTAGE`, every GEMM leaf walked descending |

Why the backward is generated and not the Mamba-1 host backward oracle: a
first build over `mamba_backward_oracle.mojo` read the `mamba1/base` backward
part `6e6f78707ccbe4d7` against the record's `f1511e64f8fa8249`; the Metal
binding at 1eea14f80, called with the same buffers, reproduced the record,
and the oracle's gradients differed from it in the low bits of 8 of 11
tensors (for example `x` at 5 of 1024 cells, `dt_proj.bias` at 49 of 64). The
generated device pass matched on the first build.

Where it ran: the Apple M4, one core, shared machine, bindings built from the
working tree at df617c699 plus this branch's changes (the column JSONs record
the base commit) into fresh output directories.

| file | what |
|---|---|
| `cpu-apple-m4.json` | the CPU column, 4 lanes, 9 fixtures, 2 repeats (349 s) |
| `diff.four-columns.txt` | `summary: IDENTICAL=36`, `summary (infer/model): IDENTICAL=36, N/A=36`, `summary (batch): IDENTICAL=36`, `require-columns 4 over ['mamba1', 'mamba2', 'mamba2-dtlimit', 'mamba3']: OK` |
| `cpu-apple-m4.sabotage.json` | the same lanes through a `-D MOJOLEARN_HOST_SABOTAGE=1` build |
| `diff.four-columns.sabotage.txt` | `summary: DIVERGENT=36`, `summary (infer/model): DIVERGENT=35, IDENTICAL=1, N/A=36` (mamba1/negative infer keeps its hash), `summary (batch): DIVERGENT=36`; the backward part differs in every train cell |

Two throwaway controls on the generated backward (working tree only, never
committed), `mamba2/base` through a per-part probe:

- the S15 cstate decay adjoint (`mamba2_cstate_ddecay_kernel`) with its
  product regrouped read every part SAME: at L = 16 < Q = 256 there is one
  chunk, the final-state cotangent is zero, so `d_cstate` is zero and that
  kernel's output is zero whatever its order. Inert at this shape, reported
  as inert.
- `mamba2_ydiag_matrix_backward_kernel` with its head-dim fold walked
  descending read `backward: cpu=bc95c5f0499ff6d2 gpu=f252b3fc19f5f9f5 DIFF`
  and forward, prefill, step and infer SAME: the backward part reaches the
  generated SSD backward.

`python3 tools/mamba_host_gen.py --check` fails (`STALE
mamba/host/gen/mamba2_ssd_backward.mojo`, exit 1) when a device kernel is
edited and not regenerated; the CPU identity gate's manifest step runs it.

Owed: the seven-runner CPU identity gate on the branch.
