# LANE STATUS: `lane/gemm-next` (2026-09-17)

**2026-09-18 AMD update:** the class spelling now has device proof and full
700-step comparisons on both corpora and columns. AMD whole-step reduction
is 8.43%; see [the measured follow-up](LANE_STATUS_amd-gemm-class.md). The
8-slot shipped / 5-slot candidate wording below records the earlier state.

Written for a reader with NO context. Branch `lane/gemm-next`, worktree
`~/mojolearn-wt/gemm-next`, branched from `origin/main` at 712eedd16.

**NO PODS ARE OUT.** All four legs this lane ran finished, and every box was
terminated and verified gone by its own runner (RunPod `e8rs6eq63pwejl` and
`20usq9mvqsq2u5`, Hot Aisle VMs `7a1b7c34` and `7c0350cd`, all HTTP 404
confirmed). Nothing is billing.

The lane owns `gemm/` and `core/gemm.mojo`. It does NOT own
`transformer/impl/llama/` or the fused attention (`lane/attention-speed`),
`python/mojolearn/verify_reference/table.json` or `docs/VERIFY_EXTERNALLY.md`
(`lane/reference-regen`), `bench/results/identity_break/` (`sabotage-sweep`), or
the `--compare` path in `_verify_all.py` (`compare-commit-reveal`).

---

## 1. THE CORRECTNESS ITEM, AND IT IS STILL LIVE

**The shipped Apple GEMM seam does not implement the identity contract.** The
contract's seam is ROUND-THEN-FLUSH. Apple computes FLUSH-BEFORE-ROUND.

Rechecked on 2026-09-17 at `origin/main` (712eedd16) on the M4, one Metal slot,
0.294 s of GPU, no dataset. Evidence
`bench/results/e1g/2026-09-17_162800-apple-m4-seam-probe-recheck/`, commit
b2f4eea71, brief section 19.

```
lane shipped  fnv1a64=f269fc70e5625987 -> fbr      (what Apple computes)
reference rtf fnv1a64=62a6b5621e27c707             (what the contract requires)
boundary i=400 a=3f7fffff b=00800000 acc=00000000  rtf=00800000 fbr=00000000
```

All 262,144 adversarial triples, 0 mismatches between lanes, 315 triples
separating the two semantics. The check CAN fail the other way and does: the
2026-09-13 NVIDIA and AMD legs read `rtf` on their shipped lanes from this same
host reference script, which never sees the device.

**It is not a GEMM defect.** `_tuned_step` off NVIDIA is
`ftz(identical_mul_add(a, b, acc))`, and `identical_mul_add` is `std.math.fma`,
the native instruction. The outer `ftz` cannot restore a value the hardware
already flushed to zero INSIDE the FMA. **245 Mojo files call
`identical_mul_add`**; GEMM is one consumer. The 2026-09-09 kNN audit repaired
its own consumer only.

**What a fix would take.** The repair already exists and is generic:
`neighbors/checks/zero_fma_boundary.mojo::repair_zero_fma(a, b, c, zero)`,
integer-only (no floating-point operation in the decision), `@no_inline`, and
called ONLY on the branch where the flushed FMA returned zero. A GEMM fix is one
line in `_tuned_step` on the columns whose native FMA is `fbr`. The cost is the
per-step zero-compare and branch, not the 39 percent the kNN inner loop paid for
its own repair. **It has not been priced.**

**What it would move.** In principle every Apple reference cell. In practice
probably nothing: cross-vendor identity on Apple is conditional on no product
landing in `[2^-126 - 2^-150, 2^-126)`, and no card and no LM witness has ever
produced one. That is checkable only by a run of the Apple column.

**THIS LANDS ON `lane/reference-regen`.** A fix changes Apple's bits at those
315 patterns and therefore the Apple column of the reference table. Do not fix
it silently. It is Andrew's call whether GEMM takes the kNN repair or the
contract records the exception.

---

## 2. THE ITEMIZATION, MEASURED FRESH

Evidence `bench/results/e1g/2026-09-17_203224-nvidia-h100-step-breakdown-after-flips/remote/step-breakdown/`.
RunPod H100 80GB HBM3, **commit b2f4eea71**, one pod, one commit. No new code:
`tools/step_breakdown_leg.sh` (DEVIATION 2630) re-run. Every phase in
`status.tsv` exit 0; `witnesses.tsv` reads `all bits_identical True`; corpora
staged from R2 in 18 s. Brief section 21.

| | enwik8 | Pile GitHub |
|---|---:|---:|
| timed envelope | 214.42 ms | 213.67 ms |
| **REAL step (untimed, shipped build)** | **207.21 ms** | **207.87 ms** |
| instrumentation | 7.21 ms (**3.48%**) | 5.80 ms (2.79%) |

Shares **of the real step**, enwik8 (Pile GitHub agrees to a tenth of a point):

| component | ms | share |
|---|---:|---:|
| **GEMM, every call** | **119.50** | **57.7%** |
| attention kernels, launchers, regime scans | 60.12 | 29.0% |
| everything else together | 27.6 | 13.3% |

Nothing in "everything else" is above 2.1 percent. Against 2026-09-11 (GEMM
144.90 / 48.3 percent, attention 116.23 / 38.8 percent of a 299.78 ms envelope),
attention's flips have been the larger half of the step's improvement and
**GEMM's share ROSE from 48.3 to 57.7 percent**.

**CAVEAT, and it is the only one.** This leg's commit b2f4eea71 predates
DEVIATION 2900 (the attention causal block-index map), which landed on main at
16:37:50, three minutes AFTER this payload started, and took the step to about
0.1979 s. 2900 only shrinks ATTENTION, so GEMM's share can only be HIGHER than
57.7 percent at current main, never lower. The ranking is robust; the exact
percentages are one attention flip old. Do NOT patch these numbers with another
leg's figure.

## 3. THE CEILING SENTENCE

GEMM does 1.5180 TFLOP per step in 118.18 ms = **12.84 TFLOP/s, 38.3 percent of
the 33.5 TFLOP/s contract ceiling** (it was 10.68 and 31.9 percent on
2026-09-11, so `kpack_hg` and the wait removal moved the rate by a fifth).

> **Taking GEMM to its contract ceiling would make it 45.31 ms and buy 35.2
> percent of the step, and nothing more than that is available under the
> contract.** A clean SIMT rewrite, which section 3.2 bounds at 60 to 70 percent
> of the ceiling, buys **20.6 to 25.8 percent of the step**.

For contrast: attention halved buys 14.5 percent; everything that is neither
GEMM nor attention, halved, buys 6.7 percent. **GEMM is both the largest
component and the one with the largest headroom. THE LANE DOES NOT STAND DOWN.**

The 33.5 ceiling is PERMANENT and is not open for rediscovery: the identity seam
on NVIDIA is two issued instructions (`fma.rn` then `mul.rn.ftz`), the H100
issues one warp instruction per cycle per scheduler whatever the pipe, and the
2026-09-13 seam probe closed the only lever on it. Do not propose changing the
contract.

## 4. A RATE GAP INSIDE GEMM THAT IS NEW AT THIS COMMIT

The three `proj_*` kinds are **35.05 ms at about 9.9 TFLOP/s while the other
nine run 12.8 to 15.8** (`gateup` 12.9 to 15.0, `down` 12.8 to 15.1, `head` 14.0
to 15.8). They are 144 of the step's GEMM calls at 2048 x 768 x 768. At 13.0
TFLOP/s they would be 26.8 ms and at 15.0 they would be 23.2: **8.3 to 11.9 ms
of the step, 4.0 to 5.7 percent**, with no change to the kernel's arithmetic.

**THE CAUSE IS NOT ISOLATED AND MUST NOT BE QUOTED AS ONE.** Brief section 2
recorded these calls as running the `ksplit` one-leaf path rather than the TUNED
128 plan the kernel-body row serves, which would explain why `kpack_hg` did not
move them. But this leg's own launch and sync counters cannot check it
(`gemm.down_dA` reads 0.0 launches for 12 calls, which cannot be literally
true). It is a HYPOTHESIS. See section 6 for the command that settles it.

## 5. AMD: ONE FINDING, ONE DEAD LEVER, ONE LIVE ONE

**AMD's kernel-body row is already 1.** It shipped on 2026-09-13 (brief 18.4,
gather staging, measured and gated on a Hot Aisle MI300X, GEMM sum 592 -> 559
ms). Any handoff saying `kpack_hg` shipped NVIDIA-only, or that the AMD row is
0, is STALE. **Apple's row is the only one still 0.**

**AMD's seam is EIGHT issued instructions per product step, not "about six".**
Counted off the emitted gfx942 GCN on the M4, no GPU and no rental
(`bench/results/e1g/2026-09-17_163900-apple-m4-amd-seam-instruction-count/`,
brief 20.1). The loop body is `v_fmac_f32` plus `v_and`, `v_and`, `v_cmp_ne`,
`v_cmp_eq`, `s_and`, `v_and`, `v_cndmask` -- `ftz()` from
`checks/numerics.mojo` spelled out once per product step. NVIDIA's seam is two.
About half the AMD step is GEMM (559 ms of ~1.15 s), so this is the largest
counted structural difference between the two columns.

**The wave-mode lever is DEAD.** Setting `MODE.FP_DENORM` with
`llvm.amdgcn.s.setreg` leaves a loop body of `v_fmac_f32` alone -- eight
instructions to one -- but it does NOT compute the contract. Measured on a Hot
Aisle MI300X, 2026-09-17, build and run in 7 s
(`bench/results/e1g/2026-09-17_205022-amd-mi300x-hotaisle-seam-mode-probe-b/`,
brief 21.4): the `modeftz` lane hashes `eb76eb53d65e0007`, which is NONE of the
three semantics, and the boundary triple reads `00000000` where the contract
requires `00800000`. **That question is closed and it cost one minute of
MI300X.** A dead lever is a result: it is the difference between not building
the arm and building a defect.

**The negative found something larger.** `eb76eb53d65e0007` is EXACTLY the hash
NVIDIA's `fma.rn.ftz` lane produced on 2026-09-13
(`bench/results/e1g/2026-09-13_161417-nvidia-h100-seam-probe/remote/seam-probe/seam_probe.log`
line 7). **AMD's wave-mode flush and NVIDIA's hardware ftz FMA agree BIT FOR BIT
over all 262,144 triples.** Brief 14.5 said the three native FMAs disagree,
which is true of the DEFAULTS; two of the three converge on one semantics once
AMD's mode register is set, and the odd column is APPLE. A one-instruction seam
exists on NVIDIA and AMD and computes the same bits on both; what stands between
it and a doubled ceiling is Apple and those 315 triples. **That is a contract
observation for Andrew. No contract change is proposed by this lane.**

**The live AMD lever, now COMPILED and counted (2026-09-17, free, no rental).**
`bench/results/e1g/2026-09-17_171500-apple-m4-amd-ftz-class-spelling/`. The
eight instructions CAN be reduced without leaving the contract, because the
flush stays POST-ROUND on the rounded FMA result and only the spelling changes.
Per product step on gfx942, one compile, all kernels launched:

| arm | issue slots | body |
|---|---:|---|
| SHIPPED `ftz(fma)` | **8** | `v_fmac_f32`, `v_and`, `v_and`, `v_cmp_ne`, `v_cmp_eq`, `s_and`, `v_and`, `v_cndmask` |
| CLASS spelling | **5** | `v_fmac_f32`, `v_and`, `v_cmp_class_f32`, `s_nop 1`, `v_cndmask` |
| BARE `fma` (control) | 1 | `v_fmac_f32` |

Four instructions plus a gfx9 hazard `s_nop` the scheduler can usually fill from
another cell's chain. NVIDIA's seam is 2, so this closes a little over half the
seam gap. The sameness is an ISA reading (class mask `0x90` is exactly "exponent
0, mantissa non-zero"; signed zero, NaN and infinity are separate classes and are
left alone as `ftz` leaves them), **not a measurement at that date**. The device proof and prices below were
subsequently completed on 2026-09-18; see the measured follow-up above.
The historical proof requirement was a
device run of this spelling as a probe lane on an MI300X hashing to
`62a6b5621e27c707` with mismatch count 0 against `shipped`. Until that exists it
is a candidate and no arm is built on it. It would live as a kernel-matrix
capability row beside `lib_hardware_ftz_fma_for`, never an inline vendor branch.

## 6. WHAT IS OWED, AND THE EXACT NEXT COMMANDS

In priority order.

**(a) DONE 2026-09-18, and it is now the top BUILD item (brief 22.1, commit
437f01fcc).** The `PHASE` lines isolated it: the fold is a constant 0.05 to 0.07
ms per call from one leaf to sixty-four, so it is a fixed LAUNCH cost. Every
`proj` call has `group_leaves = 1`, and contract section 7.3 is titled "`P == 1`
performs NO fold addition". At 144 proj calls that fold is **8.3 ms, 4.0 percent
of the step, performing no arithmetic.**

THE CHANGE, stated precisely: the `P == 1` fold is not a no-op, it applies seam
5g and copies (`c.unsafe_store(cell, ftz(v))`). So it is NOT "skip the launch";
it is "when `leaves == 1`, have the GROUP kernel apply 5g and write `c` directly
instead of writing a partial to the workspace for a second kernel to flush and
copy". Bit-preserving because `ftz` lands on the same binary32 word, same order,
same address. NOT a scheduling arm: no plan, geometry, group rule or leaf
boundary moves. **Unbuilt. This is where to start.**

The superseded command, kept because it is how any future PHASE question is
asked:

```sh
cd ~/mojolearn-wt/gemm-next
MOJOLEARN_RUNPOD_KEY_FILE=$HOME/.mojolearn_runpod_key \
MOJOLEARN_GPU_ARCHS=sm_90a \
MOJOLEARN_STAGE_STRICT=1 \
MOJOLEARN_GEMM_LEG_EXTRA=tools/gemm_kernel_leg.sh \
MOJOLEARN_GEMM_LEG_OUT=bench/results/e1g/$(date -u +%Y-%m-%d_%H%M%S)-nvidia-h100-gemm-proj-phase \
sh tools/gemm_remote_leg.sh nvidia --payload gemm --rent --minutes 60 \
    --gpu "NVIDIA H100 80GB HBM3" --allow-concurrent \
    --local-card bench/results/e1g/2026-09-11_190725-nvidia-h100-80gb-hbm3-step-breakdown/local/apple.card
```
Then read `remote/gemm-kernel/price_tables.txt`, the `PHASE` lines for
`proj_fwd`, `proj_dA`, `proj_dB`.

**(b) CLOSED 2026-09-18:** device proof, twelve-kind prices, and both
700-step corpus comparisons passed; see `LANE_STATUS_amd-gemm-class.md`.
The prior 2026-09-17 instructions are retained for provenance: add
the `_ftz_class` spelling as a lane of `gemm/checks/gemm_seam_probe.mojo` (the
harness already takes extra lanes, the `modeftz` lane was added the same day)
and run it on an MI300X; it must hash `62a6b5621e27c707` and read mismatch 0
against `shipped`. That rides along with (c) in ONE lease, so do not buy a box
for it alone.

The compile method, for reuse: write the kernels into a scratch `.mojo`, LAUNCH
all of them (an unlaunched kernel is dead-stripped and its check "passes"), then
```sh
mojo build --emit asm --target-accelerator gfx942 -I . \
    -D MOJOLEARN_NUMERIC_IDENTICAL=1 <file>.mojo -o /tmp/x.s
```
and read the `*.amdgcn` sidecars. **`MOJOLEARN_GPU_ARCHS=gfx942` does NOT
retarget `mojo build`** -- it silently emits `air64-apple-macosx` and would
"prove" the result on Apple's backend. Check the
`.amdgcn_target "amdgcn-amd-amdhsa-unknown-gfx942"` line first, every time.

**(c) DONE 2026-09-18 (brief 22.2, commit 437f01fcc): AMD's step is 81 percent
GEMM.** Real step 687.97 ms, instrumentation 1.01 percent, GEMM 559.22 ms =
81.3 percent against 57.7 on the H100, attention 91.57 ms = 13.3 percent against
29.0. Same 1.5180 TFLOP per step at **2.72 TFLOP/s where the H100 does 12.84**,
a 4.7x gap on the same kernel at the same shapes. The 33.5 TFLOP/s ceiling is an
H100 figure and MUST NOT be applied to AMD; no AMD ceiling has been derived.

The discriminating fact is FLATNESS: all twelve AMD kinds sit in 2.51 to 2.85
TFLOP/s (1.13x spread) where NVIDIA's twelve span 9.80 to 15.79 (1.61x). A
shape-independent rate is what a fixed cost PER PRODUCT STEP looks like, and
AMD's seam is 8 instructions where NVIDIA's is 2. **That is a contrast, not an
isolation** -- clocks, occupancy and memory are not excluded, and no speedup is
claimed from it.

**(d) The NVIDIA leftovers, LAST and smallest.** The barrier skew and the
staging head's scalar global loads. Brief 15.5 says neither can be sized from
static text and Nsight Compute is refused in RunPod containers
(`ERR_NVGPUCTRPERM`), so sizing them means DIAG variants RE-BASED on the shipped
`kpack_hg` body (`bench/gemm_step_diag_main.mojo`'s base is still `kpack_padv`,
the body kpack_hg replaced). Measure before building.

## 7. STANDING CONSTRAINTS FOR WHOEVER PICKS THIS UP

- **No ninth scheduling arm.** Eight were tried, two flipped, both shipped. And
  there is no shape worth moving: all thirteen `gemm.<kind>` leaves are ONE
  kernel at different shapes (`identical_gemm_backward_a_into` and `_b_into`
  forward to a single `identical_gemm_into`), so a one percent KERNEL win is
  worth about 1.2 ms of the step where one percent of the largest single cell is
  worth 0.12 ms. Broad-and-shallow ARGUES FOR kernel-body work.
- Bit equality is absolute. IDENTICAL FP32, no tensor cores, no TF32, the seam,
  leaf boundaries and fold topology kept. An arm that changes one output bit is
  a defect. Report INERT, never "bits unchanged".
- Opponents are read from `bench/OPPONENT_REFERENCE.md`, measured once per
  tuple, never re-run to refresh a row.
- Datasets always stage from R2 (`tools/stage_from_r2.sh`), with
  `MOJOLEARN_STAGE_STRICT=1`.
- Commit to the branch BEFORE any long run. Six other lanes move HEAD; run
  `git rev-parse --abbrev-ref HEAD` immediately before every commit. Never
  `git stash`, never `git add -A`.
- Two traps this lane walked into, both now recorded in the evidence: an
  unlaunched kernel is dead-stripped and its check "passes"; and a leg body is
  copied to `/root/gemm_leg_extra.sh`, NOT into the source tree, so
  `dirname "$0"` is `/root` (that one cost a VM and three minutes).

## 8. COMMITS ON THIS BRANCH

```
e752f833f Brief section 21: the itemization re-run, the proj rate gap, the AMD wave-mode negative
9b2b1cee3 The fresh step itemization, and the AMD wave-mode answer
c8e642e65 The wrapper resolved its body against the wrong directory; a Hot Aisle MI300X found it
02ae39cad Brief section 20: what is left on each column
059d92c03 Name the column before the seam probe's build
e0b6582b0 Seam probe: a fifth lane, the AMD wave-mode question
445148a4c AMD's identity seam is EIGHT issued instructions per product step
b2f4eea71 The Apple GEMM seam still does not implement the contract
```
Nothing here changes a shipped line and nothing moves a bit.
