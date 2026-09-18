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

## RESULT 3: whole-GEMM boundary check on Apple (2026-09-18, one Metal slot, 35 s)

Evidence `bench/results/e1g/2026-09-18_apple-m4-gemm-rtf-boundary-check/`.
`gemm/checks/gemm_rtf_boundary_check.mojo`, four builds of the same source in
the same directory, five plans each (dispatcher, FLAT, TUNED 64 reg4x4, TUNED
128 reg8x8, SPLIT 128 reg8x8), seven fixtures, device bits vs host oracle:

| build | failing plan/case pairs |
|---|---:|
| no repair (`-D MOJOLEARN_NO_ZERO_FMA_REPAIR=1`) | **15** (pairs 8 cells, draw0 21, draw1 11, on every plan) |
| block admission (the default) | 0 |
| admission forced to fail (`-D MOJOLEARN_GEMM_ADMIT_NEVER=1`) | 0 |
| inline per-step repair (`-D MOJOLEARN_GEMM_INLINE_ZERO_FMA_REPAIR=1`) | 0 |

The check fails without the repair (first: cell (6,16) device 00000000,
oracle 00800000). The default build passes on the same fixtures, so its
exact block path is REACHED (its fast step alone is fbr). Fixtures draw2-4
and `admitted` do not separate the arms (OK even unrepaired): they are
coverage, not evidence. Device FNV hashes per case are printed for cross-column
comparison (e.g. pairs `a62658ede5aff32e`, draw0 `941b9586f3f46787`).

## RESULT 4 (UNDERPOWERED): price of the block admission on the Apple GEMM sum

Evidence `bench/results/e1g/2026-09-18_apple-m4-seam-repair-price-admit/`,
same harness as RESULT 2, runs admit / norepair / norepair / admit:
admit 14,518 and 12,155 ms; norepair 13,116 and 11,910 ms (and 12,477 /
13,042 in RESULT 2). Pairwise +10.7% (5 vs 6) and +2.1% (8 vs 7). The
norepair arm alone spans 11.9 to 13.1 s across four runs (~10%), so these
two pairs do NOT resolve the admission's price; no conclusion drawn. More
alternations queued (price/run3.sh). What IS resolved: the admission is not
the inline repair's 2.36x.

## RESULT 5: price of the block admission on the Apple GEMM sum (six interleaved pairs)

Same evidence dir, `summary.txt`. Eight more runs (7 rounds each) added to
RESULT 4's four. Adjacent admit/norepair pairs: 1.107, 1.021, 1.043, 1.043,
0.994, 1.057; **geomean 1.043, 5 of 6 pairs positive**. The admission costs
about **4% of the Apple GEMM sum** (the inline repair cost 136%). The
norepair arm alone ranges 11.46 to 13.12 s over eight runs, so +/- 2 points
on that 4% is honest. Still a GEMM sum, not a step: the step share is owed
(binding builds queued, see bind/ in the evidence dir).

## In flight (2026-09-18 ~12:00 ET) and how to resume
- xasm: `~/mojolearn-evidence/apple-seam-repair-2026-09-18/xasm/xcheck.sh` in the
  scratch worktree `~/mojolearn-wt/apple-seam-xasm` (detached; remove with
  `git worktree remove` when done). Emits digests.txt: main vs branch asm of
  five GEMM programs for sm_90a and gfx942 (must be equal) and one for
  apple-m4 (must differ). ~6 min per compile at one core.
- cpu_queue.sh (after xcheck): builds bench/rtf_pinned_price_main.mojo (pinned
  kernels' price, 2 arms), bench/gemm_card_main.mojo (Apple GEMM identity card,
  2 arms; run with MOJOLEARN_GEMM_CARD_ARM=device MOJOLEARN_IDENTITY_TRACE=<card>),
  then the byte-LM binding in both arms (bind/build.sh, pixi shim) for the step
  share on enwik8 + Pile GitHub (corpus/ pulled from R2, sha256 verified).
- Then Metal: pinned price, the two cards (compare with each other and with
  the retained main card bench/results/e1g/2026-09-18_013251-nvidia-h100-gemm-proj-phase/local/apple.card),
  LM lean steps (tools/lm_step_memory_probe.py --target --resident-lean
  --witness-every-step) with each binding: witnesses give the LM-lane
  prediction test, medians give the step price.
- Then NVIDIA + AMD legs: tools/gemm_remote_leg.sh with
  MOJOLEARN_GEMM_LEG_EXTRA=tools/gemm_rtf_leg.sh --local-card <branch apple card>.
- POD (mine): 9rts6f6fpgw7fm mojolearn-gemm-nvidia-2026-09-18_110519-93300 (RTX 4090, 60 min lease, dead-man armed) — NVIDIA leg: GEMM card vs Apple card + tools/gemm_rtf_leg.sh. Console ~/mojolearn-evidence/apple-seam-repair-2026-09-18/nvidia_leg.console. Pods d8klbzo3aga6d6 / vn4vonca6du36q are OTHER lanes'.

## RESULT 6: NVIDIA H100 leg (RunPod, 2026-09-18 15:16Z, pod 2knce7572vc9k1 terminated, verified 404)

Evidence `bench/results/e1g/2026-09-18_151559-nvidia-h100-apple-seam-repair/`
(branch commit 431cf44c6). A first attempt on an RTX 4090 (pod
9rts6f6fpgw7fm) never got ssh in 600 s and was terminated and verified 404
(infra, not evidence; console kept as nvidia_leg_4090_readytimeout.console in
the evidence dir).
- GEMM identity card, NVIDIA (this branch) vs the retained Apple card from
  main: 60 matched stages, **RESULT: IDENTICAL**.
- Seam probe on NVIDIA: shipped lane **`62a6b5621e27c707` -> rtf** (unchanged
  from 2026-09-13); `nativefix` = `none` (rtf_fix is the identity off Apple),
  as designed.
- Whole-GEMM boundary check: pairs / draw0 / draw1 / draw2 / draw3 /
  admitted device_fnv values EQUAL the Apple column's, cell for cell hash:
  a62658ede5aff32e, 941b9586f3f46787, f17d5419b7613ab8, 3be7a64bdda61cab,
  4667eea0d0c3eb34, b89e27b4dd7768ca. So Apple (repaired) and NVIDIA now agree
  on the fixtures that separate fbr from rtf.
- **FINDING, NOT THIS LANE'S DEFECT: draw4 (64x64x300) differs across columns
  in NaN payloads.** NVIDIA device fnv bb1a4c2c7aba843d, first cell device
  7fffffff vs the pod's x86 host oracle ffc00000; Apple device = Apple (ARM)
  host oracle = fa49a9f1815ee8a5 (in BOTH the repaired and the unrepaired
  Apple builds, so the repair did not cause it). The fixture overflows to
  inf and forms inf - inf: each backend writes its own default NaN (NVIDIA
  0x7fffffff, ARM 0x7fc00000, x86 0xffc00000). Only the FIRST mismatch was
  printed; that all 3,078 are NaN-payload cells is inferred, not listed.
  Recorded as a CANDIDATE (NaN canonicalization in the GEMM contract), not
  opened.
- Queued on Metal (metal2.sh): Apple GEMM cards both arms; pinned price x4; LM lean steps (target shape, 5 steps, enwik8 + Pile GitHub, norepair/admit/admit/norepair) with private package copies ~/mojolearn-evidence/apple-seam-repair-2026-09-18/pkg-{norepair,admit} (byte_lm .so sha256 3c879da6863b8c82 / 3e882e32bf0b0f14). Hot Aisle MI300X AMD leg launched (console amd_hotaisle.console).

## RESULT 7: Apple GEMM identity card and the pinned kernels' price (2026-09-18 11:30 ET)

Evidence `bench/results/e1g/2026-09-18_apple-m4-seam-repair-card-pinned/`.
- Apple GEMM identity card (`bench/gemm_card_main.mojo`, device arm, 60
  stages) built without and with the repair in one directory: the two cards
  are byte-identical, and equal (stage hashes) to the retained main Apple
  card `bench/results/e1g/2026-09-18_013251-nvidia-h100-gemm-proj-phase/local/apple.card`.
  **INERT on every GEMM card cell**, as predicted. The H100 card of this
  branch matched the same Apple card (RESULT 6).
- Pinned one-cell-per-thread kernels (core/gemm.mojo, the classical
  estimators' `gemm_nt` / `gemv_n` / `gemm_nt_gram`; they carry the INLINE
  per-step repair, no block admission), runs norepair/admit/admit/norepair,
  7 rounds each, output bits equal across arms:

| call | no repair ms | repair ms | cost |
|---|---:|---:|---:|
| gemm_nt 1024x1024x512 | 12.60, 12.31 | 15.08, 15.09 | **+21%** |
| gram 512x4096 | 24.36, 24.07 | 29.88, 29.64 | **+23%** |
| gemv_n 1M x 64 | 3.52, 3.40 | 3.49, 3.66 | ~+2% (memory-bound) |

  A real price on the classical GEMM paths, measured in isolation. Not yet
  expressed as a share of any estimator's fit time (unmeasured).

## RESULT 8: NVIDIA and AMD device code, main vs branch (cross-compiled on the M4, no rental)

Evidence `bench/results/e1g/2026-09-18_apple-m4-seam-repair-xasm/`. Five GEMM
programs (gemm_step_price_main, gemm_device_check, lanes_price_main,
gemm_lowbit_check, gram_outputs_parallel_check) built `--emit asm` for sm_90a
and gfx942 with origin/main's five changed files and with the branch's, in
ONE worktree path (`devcmp.txt`):
- **NVIDIA sm_90a: every embedded PTX module is byte-identical** (44 to 421
  modules per program); host asm differs only in integer immediates (source
  line numbers baked into error paths).
- **AMD gfx942: NOT byte-identical.** 5 to 8 code objects per program differ;
  in gemm_device_check the 5 are the TUNED kernels (LDS 9,216 to 40,960 B).
  Per-kernel GCN sidecars (`gcn-diff-*.txt`, `fpcmp.txt`): the floating-point
  opcode multiset is equal in every differing kernel; 3 of 5 differ only in
  register assignment/order, 2 in integer address arithmetic
  (v_lshlrev_b64 / v_subb / v_lshl_add_u64 / s_mov / s_nop counts). So the
  AMD arithmetic is unchanged by construction and by opcode count, but the
  byte-level claim does NOT hold on AMD; the AMD runtime proof (card + probe
  + boundary check on an MI300X) is required and is pending Hot Aisle stock
  (RunPod AMD create returned HTTP 500). Cause under test (gcn3.sh): the
  tuned kernel's always-declared admission registers or the
  `_tuned_step_admitted` wrapper.
- Apple: the gemm_device_check metallib differs (59 of 66 modules), as it
  must: the check can fail.

## RESULT 9: AMD MI300X leg (Hot Aisle 8core, VM 4875e63c deleted, verified 404, $0.30)

Evidence `bench/results/e1g/2026-09-18_153549-amd-mi300x-hotaisle-apple-seam-repair/`
(RunPod AMD create had returned HTTP 500; Hot Aisle 13core had no stock).
- Seam probe on AMD: shipped lane **`62a6b5621e27c707` -> rtf** (unchanged).
- GEMM identity card of this branch on AMD vs the Apple card: 60 matched
  stages, **RESULT: IDENTICAL** (`diff_apple_vs_amd.txt`).
- Whole-GEMM boundary check: all seven fixtures OK against the host oracle;
  pairs/draw0-3/admitted device_fnv EQUAL Apple's and NVIDIA's.
- draw4 (the NaN fixture): AMD device fnv 476d0179e49c6ea5 = the x86 oracle's
  (x86 default NaN 0xffc00000). **All three columns write a different NaN
  there** (Apple fa49a9f1815ee8a5, NVIDIA bb1a4c2c7aba843d, AMD
  476d0179e49c6ea5). Pre-existing, independent of this repair; CANDIDATE.

So after the repair, Apple, NVIDIA and AMD agree bit for bit on every
fixture that separates flush-before-round from round-then-flush, on the GEMM
identity card, and (seam probe) on all 262,144 triples of the shipped seam.
