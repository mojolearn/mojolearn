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
