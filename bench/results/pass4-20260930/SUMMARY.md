# Neural pass 4 (PR #8, lane/neural-pass4 cfa3f9347) on NVIDIA L4 (nvc1), 2026-09-30

Job `tools/pass4_job.sh` (nvc1-0005); the optimizer check rerun after one fix to the check tool.

**Check-tool fix.** `tools/optimizer_resident_check.py` handed both runs the SAME gradient arrays
(`np.ascontiguousarray` returns its argument when it is already contiguous float32), and the clip scales the
caller's gradients in place, so the second run's clipped step read already-clipped gradients: FAIL from the
clipped step on, for adamw, adam and sgd alike. Each step now gets a copy. Not a bug in the resident path.

| check | result |
|---|---|
| optimizer_resident_check adamw / adam / sgd (5.66 M floats, 32 steps, round trip at 10, clip at 16) | PASS, every byte equal; per-call 49.9 / 46.2 / 50.7 ms -> resident 31.9 / 30.3 / 32.6 ms (1.56x / 1.53x / 1.56x) |
| samba-train-step and lm-train-step, 10 step losses, resident vs `MOJOLEARN_OPTIMIZER_RESIDENT=0` vs the S16-pass L4 run | identical float losses in all three |
| samba-train-step | 148.7 ms (S16 pass, L4) -> 103.4 ms (resident; 110.5 ms with the optimizer on the host) |
| lm-train-step | 109.1 -> 108.5 ms |
| Samba step profile | optimizer.step 13.5 ms of 103.0 (was 55.7 of 147.6) |
| Mamba-3 y + 10 gradients, default arm (now regs) and regs2, both shapes | equal to the strides/S16 digests |
| S16 qk+s15 board shape: regs (new default) vs regs2 | 9.16 vs 10.80 ms |
| small-MLP GPU gate (26 tests), fused and per-operation; mlp_step_check resident on and off | 26/26 each; PASS each |
| matmul digests (21) | equal to the PR #7 run |
| fixed15 table vs H100 choice | 0 of 422 digests differ; mlp_down bwd_dw now 0.99x (was 0.86x); lm_head 1.41x, mlp_up 1.51x, mlp_down bwd_dx 1.47x |

AMD and Apple: not run.
