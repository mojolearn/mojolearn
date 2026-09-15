# Kernel methods and Cholesky rows — two MI300X, FAILED Cholesky gate

RunPod pod `1s2ahbzykxtlmv`, two AMD Instinct MI300X (gfx942), 2026-09-14
23:04-23:13Z, commit `f229b8b3c` (`commit.txt`), body `body.sh`.

- `tools/parallel_kernel_methods_check.py`: `PASS 32 kernel method
  configurations and 8 refusals`; `public.json` equals the two-H100
  `../kernel-methods-h100/out/public.json` as JSON (every KernelRidge,
  Nystroem and RBFSampler digest and the sigmoid refusal).
- The sabotage binding FAILS the check (`KernelRidge outcome differs, 37,
  linear`), as on the H100s.
- `training/checks/cholesky_parallel_check.mojo` FAILED after 33 passing
  cases: `solve trace differs n513-r2: 0 chol.solve.forward f32 1026
  9e249d211a9cf9d4 VS 0 chol.solve.forward f32 1026 f7c6a7c436c780fe`. The
  factor traces of that case were equal; the forward substitution over two
  right-hand sides differed between one device and two. The same gate passed
  all 40 cases on two H100s (twice, at `00037ee37` and `d12a1597f`). The
  traces were deleted on the box, so the one-device hash could not be put
  beside the H100's; which column stands alone is NOT known from this run.
- The factor-only sabotage run fails at `chol.panel000.trailing` (n=65), as
  on the H100s.

Follow-up: the solve's column gather released its packed source right after
the copy had drained only its source stream. The next commit drains both ends
before release and before any destination read, and the next legs keep every
trace digest and run the native gate twice.
