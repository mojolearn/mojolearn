# LANE STATUS: apple-seam-repair (branch `lane/apple-seam-repair`)

Written for a session with no context. Brief (binding, verbatim):
`~/mojolearn-evidence/relaunch-sep18/BRIEF_apple-seam-repair_original.txt`.
Andrew's directive: Apple MUST compute the contract's round-then-flush (`rtf`)
seam bitwise identically to NVIDIA and AMD. No exception, no narrowing.

## Registered prediction (before any run, 2026-09-18)

`fbr` (Apple's native flush-before-round) and `rtf` agree on every input
except products landing in `[2^-126 - 2^-150, 2^-126)`. No identity card or LM
witness has landed there. So the repair is predicted **INERT on every
existing recorded Apple cell**. Falsifier: any existing Apple cell's hash
moves -> the repair changed something besides the boundary; find out what
before shipping.

Affirmative proof: seam probe on Apple must hash `rtf` = `62a6b5621e27c707`
over 262,144 triples (pre-repair: `fbr` = `f269fc70e5625987`).

## Progress log

- 2026-09-18 session 1 killed ~10 min in (no edits, no runs).
- 2026-09-18 session 2 (this): resumed. Metal slot was held by
  release_runner (pid 92580, release-087-final) at start; queued behind it.
- Repair committed (unmeasured): `checks/rtf_seam.mojo` (`rtf_mul_add`,
  `rtf_fix`, `rtf_repair_zero` = the kNN integer repair), kernel-matrix row
  `lib_zero_fma_repair_for` (Apple only; `-D MOJOLEARN_NO_ZERO_FMA_REPAIR` is
  the never-shipped price arm). Wired into `_tuned_step` (software branch) and
  the flat / tiled / leaf GEMM seams in gemm/checks/gemm_identical.mojo,
  core/gemm.mojo pinned nt/gemv, core/gram_multi_gpu.mojo, gemm_lowbit flat.
  Seam probe gained lane 6 `nativefix` (native FMA + repair, no software ftz:
  the candidate cheaper Apple spelling; must hash rtf to count).
- Evidence dir (outside repo): ~/mojolearn-evidence/apple-seam-repair-2026-09-18/
  probe/ (repaired + norepair probe binaries, run.sh queued on `mac_slot.sh metal`),
  price/ (bench/gemm_step_price_main.mojo, both arms same dir),
  xasm/xcheck.sh (sm_90a / gfx942 / apple-m4 asm of five GEMM programs at
  origin/main files vs branch files, same path; NVIDIA/AMD must be byte-equal,
  Apple must differ = the check can fail).
- Scope audit delegated read-only; results to be recorded below.

## Scope (static, read-only audit, 2026-09-18; approximate, classified by script)

Per-call table: ~/mojolearn-evidence/apple-seam-repair-2026-09-18/scope_audit_calls.tsv
(family, class, file, line, function, source). 255 non-bench, non-archive
files, ~1,148 call sites of `identical_mul_add`/`identical_mul` in code bodies.

| class | calls | files | Apple differs from NVIDIA/AMD on |
|---|---:|---:|---|
| 1 rtf-spelled, device (`ftz(fma)`) | ~401 | 94 | the boundary window only (the 315-type triples) |
| 2 no-flush, device | ~20 real (30 raw) | ~10 | the window AND every subnormal result (NVIDIA/AMD keep it) |
| 3 host-only / CPU | ~652 | 110 | nothing (not Metal) |
| undetermined | ~50 | 17 | mostly host scalar math, unconfirmed |
| probes / sabotage | 15 | 3 | n/a |

Plus a PLAIN-OP class not counted in the table: `ftz(x * y)` / `ftz(x / y)`
appear ~230 times in 77 non-host files (glm 44, arima 37, holtwinters 34,
gbdt 27, metrics 20, decomposition 16, core 10). Whether Metal's plain
multiply/divide also flush before rounding is UNTESTED; if they do, each
carries the class-1 window.

Class 1 by family: mamba 104 (backward heavy), training 48 (optimizer
kernels 38), arima 29, transformer 26, holtwinters 20, gaussian_process 19,
decomposition 19, neighbors 16 (kNN already repaired), resample 16, hdbscan 10,
umap 10, kernel_methods 11, cholesky 9, gbdt 8, glm 6, others small.

Class 2 (window + every subnormal): core/gram_splitk.mojo:695,705,723,739;
core/column_stats.mojo:235; glm/impl/qn/simple_mat/dense.mojo:129,149;
glm/impl/qn/glm_softmax.mojo:241; gbdt/methods/dynamic_boosting.mojo:100-101;
gbdt/methods/kernel_add_model_value.mojo:100; ensemble/decisiontree/
batched_levelalgo/objectives.mojo:601,612,617,675,690; spectral/impl/
spectral_predict.mojo:77; umap/optimizer_identical_device.mojo:165,210.
Class 2 cannot be repaired by the zero-result repair: Apple would have to
PRESERVE subnormals, which its FMA does not do natively.

## In flight (2026-09-18 ~10:10 ET)
- Metal slot held by release_runner (release-087-final, pid 92580, 7200 s
  timeout from 09:25). Queued behind it: probe/run.sh (ticket 95), then
  price/run.sh (Apple GEMM sum, norepair/repair/repair/norepair, 11 rounds).
- FINDING before any run: the repair's Apple price binary embeds a 33.8 MB
  metallib against 3.1 MB for the no-repair arm (`__const` 0x204b4ad vs
  0x2f210d): the integer repair is inlined into every unrolled step. Expect a
  real price; a deferred (per-block flag, recompute) spelling is the candidate
  if it is large.
- xasm/xcheck.sh running on one CPU core: it SWAPS the five changed source
  files to origin/main's bytes and back (restores from a byte copy on exit).
  If the session dies mid-run, restore with `git checkout HEAD -- <file>` for
  checks/kernel_matrix.mojo core/gemm.mojo core/gram_multi_gpu.mojo
  gemm/checks/gemm_identical.mojo gemm/checks/gemm_lowbit.mojo (all committed).

## RESULT 1: the seam probe, Apple M4 (2026-09-18, one Metal slot, 0.82 s of GPU)

Evidence `bench/results/e1g/2026-09-18_apple-m4-seam-repair-probe/`. Both arms
built from the same source in the same directory; the control arm adds
`-D MOJOLEARN_NO_ZERO_FMA_REPAIR=1`. Host reference `tools/gemm_seam_probe_reference.py`.

| arm | shipped lane | nativefix lane | boundary a=3f7fffff b=00800000 acc=0 |
|---|---|---|---|
| no repair (control) | `f269fc70e5625987` -> fbr | fbr | shipped=00000000 |
| repair | **`62a6b5621e27c707` -> rtf** | **`62a6b5621e27c707` -> rtf** | shipped=00800000 |

All 262,144 triples. `shipped/swrtf` mismatches 315 in the repaired arm (the
unrepaired software lane is still fbr in the same binary), 0 in the control.
The check fails when it should: the control arm reads fbr. `nativefix` (the
native Apple FMA, NO software ftz, plus the zero repair) ALSO hashes rtf: on
Apple the software `ftz` after the FMA is redundant over these triples, a
candidate cheaper spelling.

## RESULT 2: price of the PER-STEP inline repair on Apple (REJECTED as the spelling)

Evidence `bench/results/e1g/2026-09-18_apple-m4-seam-repair-price-inline/`.
`bench/gemm_step_price_main.mojo`, arm `shipped`, the twelve LM GEMM calls
at target shape (all TUNED 128x128 reg8x8 KS=16 on Apple), 11 rounds, runs
interleaved norepair / repair / repair / norepair, one Metal slot. A GEMM
SUM weighted by per-step counts, not a step time.

| run | arm | GEMM sum ms |
|---|---|---:|
| 1 | no repair | 12,476.965 |
| 2 | inline repair | 29,954.621 |
| 3 | inline repair | 30,421.764 |
| 4 | no repair | 13,041.771 |

The per-step inline repair costs **2.36x on the Apple GEMM sum (+136%)**,
every call about equally. Metallib 33.8 MB vs 3.1 MB: the integer repair is
inlined into all 64 cells of the reg8x8 tile per step. This is NOT the
shipped spelling; it survives only as the test arm
`-D MOJOLEARN_GEMM_INLINE_ZERO_FMA_REPAIR=1`. The kNN precedent's 39% is not
comparable (different kernel, different tile).

Replacement (committed, being measured): EXACT BLOCK ADMISSION in the tuned
kernel (`TUNED_BLOCK_ADMIT`): fast step; per window each thread folds the
minimum nonzero exponent of the operand words it staged; one block
reduction after the last window; `minA + minB >= 151` proves every exact
step result is a multiple of 2^-149, hence outside the window, hence the fast
bits ARE rtf; otherwise the block recomputes its cells with `rtf_mul_add`
in the contract order and fold (`_rtf_cell`, `_rtf_leaf_partial`). Test arm
`-D MOJOLEARN_GEMM_ADMIT_NEVER=1` forces every block onto the exact path.
New check `gemm/checks/gemm_rtf_boundary_check.mojo` compares whole device
GEMMs on adversarial-word matrices with the host oracle (rtf), five plans.
