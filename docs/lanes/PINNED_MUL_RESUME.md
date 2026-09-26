# lane/pinned-mul-contract-free: resume file

Goal: IDENTICAL arithmetic independent of the compiler's contraction choice by
construction. Evidence: `~/mojolearn-evidence/pinned-mul-contract-free/`.

## Proven so far (2026-09-26 ~07:50Z)

- The pin mechanism (`checks/numerics.mojo` `pinned_mul_f32` / `pinned_mul_f64`,
  used by `identical_mul`, new `identical_mul64`, gpc `_mul64`, the nine mamba
  `pinned_mul` helpers). Per-backend proof in `probes/RESULTS.txt`:
  arm64 `fmul`+`fadd`, x86-64-v3 `vmulss`+`vaddss`, sm_80/89/90 `mul.rn`+`add`,
  gfx942 `v_mul_f32`+`v_add_f32`, Apple M4 runtime 0 of 65536 fused (the old
  `fma(a, b, -0.0)` pin: 65506 of 65536 fused on the M4, `fmadd` on arm64).
- `tools/contraction_census.py` (build / compare / locate, `--self-test` OK).

## Census before the site rewrites (census/*_round1.txt, *_locate1.txt)

- host families (arm64): 14 functions; located and rewritten (commit acae29a3e).
- device bindings, NVIDIA PTX (sm_90a): ~340 kernels, 5099 fused ops that
  depend on the mode, plus 3 PTX plain mul->add pair sites (pinned).
  Line-level locate from the `-g` builds pending (census/nvidia/*-g).

## Remaining

1. Locate the PTX sites by source line (`contraction_census.py locate
   census/nvidia/default-g census/nvidia/off-g`), rewrite each (explicit fma
   where the default fused, pinned where it did not).
2. ORIGINAL-tree census (git archive of db66ff952 at
   ~/mojolearn-wt/pmcf-orig, queued): new-default fused counts must equal the
   original default per function on host, NVIDIA and AMD (bit-neutral on AMD
   without an AMD column).
3. Final census new default vs new off: host, nvidia, amd (CLEAN).
4. Columns: default build Metal + CPU vs 0.8.19; contract=off build Metal +
   CPU vs 0.8.19; NVIDIA leg (nvidia_wrap.sh in the evidence dir, both variants,
   RTX 4090, cap $6) vs the 0.8.19 CUDA column.
5. check_no_recursion_around_kernels.py, lane_select, IDENTITY_PATHS row 9 text.

## Spend

$0 so far (no rental yet).
