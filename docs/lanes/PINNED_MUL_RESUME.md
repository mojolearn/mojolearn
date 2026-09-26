# lane/pinned-mul-contract-free: resume file

Goal: IDENTICAL arithmetic independent of the compiler's contraction choice by
construction. Evidence: `~/mojolearn-evidence/pinned-mul-contract-free/`
(`probes/`, `census/`, `scripts/`). The rule is written down in
`IDENTITY_PATHS.md`, "Contraction independence (row 9)".

## Proven so far (2026-09-26 ~07:45Z)

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

## Running / next

1. ORIGINAL-tree census (git archive db66ff952, ~/mojolearn-wt/pmcf-orig):
   host/nvidia/amd default -> census/orig. Queue log census/queue_orig.log.
2. Snapshot census r2 (git archive c46fe29fb, ~/mojolearn-wt/pmcf-snap):
   host/nvidia/amd default+off -> census/r2 (queue_r2.log). Then
   `contraction_census.py compare census/r2/T/default census/r2/T/off` per T;
   residual bindings get `build --debug --only <binding>` + `locate`.
   Bit-neutrality on AMD (no AMD column): compare census/orig/amd/default with
   census/r2/amd/default per kernel: fused counts must be equal except where a
   site was deliberately pinned (exact products).
3. Columns: all bindings default -> Metal + CPU vs 0.8.19; contract=off ->
   Metal + CPU; NVIDIA leg (`nvidia_wrap.sh` in the evidence dir: both
   variants, 209 lanes, RTX 4090, cap $6) vs the 0.8.19 CUDA column
   (~/mojolearn-evidence/release/0.8.19/69a519c1522d/smoke-linux/column-cuda.json;
   Metal reference ~/mojolearn-evidence/release-check/69a519c1522d/metal/column.json).
4. Merge to main only when all of the above pass.

## Risks noted

- ordered model length and resample quantile position: the host oracle
  computed the product unfused, the device binding's inlined copy fused it;
  now pinned in both (reference semantics). A CPU or GPU cell could move;
  the columns decide.

## Spend

$0 so far (no rental yet).
