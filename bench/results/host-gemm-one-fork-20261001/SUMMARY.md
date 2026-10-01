# Host GEMM one fork a call (PR #19, lane/neural-pass13), 2026-10-01: NOT MERGED (slower at 64 threads)

AMD EPYC 9575F, 64 cores. Bits: check-gemm-host-rows PASS, host ab mamba3/transformer PASS (same sha), train gate 640/640.
Host GEMM bench at the policy, PR #18 (merged) vs PR #19: lm_head 512x8192x384 1261 -> 877 GFLOP/s; mamba3_in_proj
2048x1728x384 1548 -> 734; head_scores 449 -> 465. One thread unchanged (~100-107). Packing the right operand once on
the calling thread costs more than the second fork saves at 64 threads. Code stays on lane/neural-pass13.
