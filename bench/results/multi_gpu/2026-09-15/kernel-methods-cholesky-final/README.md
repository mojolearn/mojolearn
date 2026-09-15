# Cholesky, KernelRidge, Nystroem and RBFSampler — final receipts on two H100s and two MI300X

One commit, `4e1008500` (`*/commit.txt`), one body (`body.sh`), two boxes:
RunPod pod `ceayy80nhw97bg` (2x NVIDIA H100 80GB HBM3, sm_90a) and pod
`72l6rzxextnvqd` (2x AMD Instinct MI300X, gfx942), 2026-09-15 00:18-00:27Z.
The commit includes the host-staged Cholesky column solve
(`../../2026-09-14/cholesky-mi300x-diag/`). Box and Mac source SHA256 agree on
both. Design: `docs/multi_gpu/cholesky.md`, `docs/multi_gpu/kernel_methods.md`.

Both `*/gate.txt` read:

- `tools/parallel_kernel_methods_check.py`: `PASS 32 kernel method
  configurations and 8 refusals` (KernelRidge duals and predictions for
  linear, rbf, poly and sigmoid up to 515 rows and 3 targets, one equal
  sigmoid refusal; Nystroem state and transforms; RBFSampler transforms over
  ragged row shards; refusals and failed-fit publication).
- Against the `-D MOJOLEARN_CHOLESKY_PARALLEL_SABOTAGE=1` binding the check
  FAILS (`KernelRidge outcome differs, 37, linear`).
- `training/checks/cholesky_parallel_check.mojo`: 40 PASS lines, twice
  (`PASS cholesky parallel gate`), with identical trace digests in both runs.
- The factor-only sabotage run FAILS at `chol.panel000.trailing` (n=65).
- `kernel_methods/checks/km_check.mojo`: `18 checks OK [IDENTICAL]`.

## Cross-vendor

`h100/public.json` equals `mi300x/public.json` as JSON, and
`h100/chol_trace_digests.txt` equals `mi300x/chol_trace_digests.txt` after
sorting (152 Cholesky trace files: every factor and solve stage of every case,
one and two devices).

No speed or capacity claim.
