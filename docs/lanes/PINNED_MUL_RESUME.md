# lane/pinned-mul-contract-free: resume file

Goal: IDENTICAL arithmetic independent of the compiler's contraction choice by
construction. Evidence: `~/mojolearn-evidence/pinned-mul-contract-free/`
(`probes/`, `census/`, `scripts/`). The rule is written down in
`IDENTITY_PATHS.md`, "Contraction independence (row 9)".

## Proven so far (2026-09-26 ~09:05Z)

- The pin (`checks/numerics.mojo` `pinned_mul_f32` / `pinned_mul_f64`, used by
  `identical_mul`, new `identical_mul64`, gpc `_mul64`, the nine mamba
  `pinned_mul` helpers). `probes/RESULTS.txt`: arm64 `fmul`+`fadd`, x86-64-v3
  `vmulss`+`vaddss`, sm_80/89/90 `mul.rn`+`add`, gfx942 `v_mul_f32`+`v_add_f32`,
  Apple M4 runtime 0 of 65536 fused (old `fma(a, b, -0.0)`: 65506 fused).
- `tools/contraction_census.py` (build / compare / locate, `--self-test` OK);
  opt-in rehearsal step `--with contraction-census`.
- tools/check_no_recursion_around_kernels.py exit 0; lane_select: no
  UNATTRIBUTED path (269 of 274 lanes).

## Sites rewritten (commits acae29a3e, c46fe29fb and the numerics pins)

Host census (arm64, all 32 families) and NVIDIA line-table census (all 23
device bindings) located them; each is the default build's fused op (explicit
fma / identical_mul_add) or, for an exact product or a product the reference
spells, pinned: see the commit messages. Shared helpers pinned:
portable_sqrtf halvings, portable_expf's last scaling (every `exp + 1`),
portable_erff's `x*p`.

## Result (2026-09-26 ~14:00Z): merged to main

Merged tree `b3bde9d7e` (lane + origin/main, clean), columns against the 0.8.19
references (Metal `release-check/69a519c1522d/metal`, CUDA and HIP
`release/0.8.19/69a519c1522d`), 209 lanes x base/denormal/odd, one fit each:

| build | Metal | CPU | vs 0.8.19 Metal / CUDA / HIP |
|---|---|---|---|
| default | 627/627 COMPLETE | 627/627 COMPLETE | 0 differ |
| `--fp-mode contract=off` | 627/627 COMPLETE | 627/627 COMPLETE | 0 differ |

NVIDIA L40S (lane tip c5713df7d, both builds): 621/621 equal to the 0.8.19 CUDA
column, 6 byte-lm cells REFUSED (binding not built on that leg).
Evidence: `~/mojolearn-evidence/pinned-mul-contract-free/m-*` and `nvidia2/`.

The first local run (fe91bde48) read INCOMPLETE for two reasons, neither the pin:

1. **The AIR blob floor undercounted.** `air_blobs()` in every
   `bindings/build*.sh` matched `<name>_<16 hex>air`, i.e. only a kernel whose
   next string-table entry is an `air.*` intrinsic. With the Apple pin
   (`llvm.fma.f32`) the next entry of many gp, kernel_methods and mamba kernels
   became `llvm.fma.f32` or the bare version tag `32023.883air64-...`, and their
   floors failed (gp 0 of 3) with every kernel present. The regex now accepts
   `air`, `llvm.` or `<version>air64` after the hash; checked equal to the MTLB
   header count on every IDENTICAL binding main ships (gp 53, mamba 131,
   kernel_methods 59, gbdt 221), old names a subset of new on all 22.
2. **`libMojolearnMath.dylib` was never built in the worktree** (gitignored;
   `packaging/portable_math/stage.py` build()): 114 cells refused on import.

## AMD column: PASSED (2026-09-26 ~14:36Z)

Hot Aisle 1x MI300X (gfx942), merged tree `b3bde9d7e`, default build, the same
209 lanes x base/denormal/odd: 621/621 equal to the 0.8.19 Metal, CUDA and HIP
references; 6 byte-lm cells REFUSED (that binding is not built by this leg
body, as on the NVIDIA leg). Evidence `~/mojolearn-evidence/pinned-mul-contract-free/amd2/`;
VM e6fb84aa verified gone, $1.50. The 14 gfx942 kernel families whose fused-op
counts differ from 0.8.19 (see the census notes) move no bit in any cell.

## Original-vs-new census findings (census/orig_vs_round1_nvidia.txt, host fusedtext)

- 0.8.19 FUSED some identical_mul / pinned_mul products into the add they fed
  (the old pin folded): mamba3 `m3_mod_2pi`, the dynamic-boosting cursor
  (kernel + host oracle), now explicit fma (bits kept).
- Box-Muller `identical_mul_add(ftz(identical_mul(r, c)), 1.0, 0.0)` (gp
  sample_y, RBF sampler, GMM sample): 10 fusions per function lost; bit-neutral
  (fma(r, c, +0) == round(r*c) + 0 for every r, c, including zero signs and the
  ftz arm). The columns confirm.
- COST: on the host the fence blocks vectorization where a pinned product sits
  in a loop (mamba host: fmla 226 -> 79, fmul.4s 71 -> 2). A CPU speed cost on
  IDENTICAL host paths that use identical_mul in hot loops; not measured yet.

## Risks noted

- ordered model length and resample quantile position: the host oracle
  computed the product unfused, the device binding's inlined copy fused it;
  now pinned in both (reference semantics). A CPU or GPU cell could move;
  the columns decide.

## Spend

- 12:53Z RunPod MI300X create refused (no instances): $0. Hot Aisle 12:41Z to
  ~14:00Z no stock on 13core, 8core or 2gpu: $0 so far.

- 09:02Z RTX 4090 create refused (no instances): $0.
- 09:04Z L40S pod bdmyb7no2e9uzk, lease 150 min at $1.09/h (at most $2.73),
  leg `scripts/nvidia_leg.sh rent` at c5713df7d, out
  ~/mojolearn-evidence/pinned-mul-contract-free/nvidia2 (dead-man armed,
  leg tears down; confirm `DELETE ... VERIFIED gone` in nvidia2.out).
