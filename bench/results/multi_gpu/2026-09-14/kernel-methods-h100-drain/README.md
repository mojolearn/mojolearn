# Kernel methods and Cholesky rows after the both-ends drain — two H100s

RunPod pod `puorkzf92fnm9d`, two NVIDIA H100 80GB HBM3 (sm_90a), 2026-09-14
23:28-23:34Z, commit `d24e39765` (`commit.txt`), which adds the Cholesky
copy drains on top of `../kernel-methods-h100/`. Box and Mac source SHA256
agree. Body `body.sh`.

## Result (`out/gate.txt`)

- `tools/parallel_kernel_methods_check.py`: `PASS 32 kernel method
  configurations and 8 refusals`; `out/public.json` equals the earlier H100
  report as JSON.
- The Cholesky sabotage binding fails the check (`KernelRidge outcome
  differs, 37, linear`).
- `training/checks/cholesky_parallel_check.mojo`: 40 PASS lines, and 40 again
  in a second run in the same box; `out/chol_trace_digests.txt` and
  `out/chol_trace_digests_repeat.txt` are identical, and for every case the
  one-device and two-device trace files have the same SHA256.
- The factor-only sabotage run fails at `chol.panel000.trailing` (n=65).
- `kernel_methods/checks/km_check.mojo`: `18 checks OK [IDENTICAL]`.

## Naming the lone column of the MI300X failure

`out/n513-r2/` keeps the solve traces of the case that failed on two MI300X
(`../kernel-methods-mi300x-failed/`). Here both read
`0 chol.solve.forward f32 1026 9e249d211a9cf9d4`. The MI300X failure printed
`9e249d211a9cf9d4` (one device) VS `f7c6a7c436c780fe` (two devices). So the
MI300X one-device forward equals both H100 columns, and the MI300X
two-device forward is the column that stands alone: the defect was in the
distributed solve's transport on that box, not in the arithmetic of either
vendor.
