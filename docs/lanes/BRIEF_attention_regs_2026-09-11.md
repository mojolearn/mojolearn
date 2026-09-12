# The register pressure lens on the other attention kernels (DEVIATIONS 2653 to 2656)

Lane `lane/attention-regs-h100`, branched from origin/main at 8dc33f00
(2026-09-11). NVIDIA H100, IDENTICAL mode. Source only. NOTHING HERE WAS
COMPILED OR RUN (no build on the Mac, by rule); the orchestrator's M4
commands in section 7 are the first compile.

STATUS AND VERDICT, up front. The lane was asked to take section 20.2's lens
(blocks per SM = `floor(256 / pad8(regs))` on a 256-thread block) to the two
attention kernels nobody has looked at through it, the backward dq kernel
(12.5 ms per step) and the forward kernel (20.7 ms per step). The reading
closes one of them and finds the other unmeasurable from this desk.

1. dq is ALREADY at two blocks per SM (118 registers read back on the H100),
   and section 3 proves by arithmetic that NO mechanism on this lane's list
   can take it to three. Its whole counted per-thread float state is 21
   registers and three blocks per SM needs a cut of 38. NO ARM WRITTEN.
2. The forward kernel has NEVER had a register readback, on any column. The
   `RESOURCES` section of the price harness covered eleven BACKWARD kernels
   and no forward kernel, and no filed leg log carries a forward row. Its
   blocks per SM today is therefore unknown, and an arm written now would be
   written blind. Section 4 gives the window (129 to 147 registers) in which
   an arm can pay at all, and section 6 names the two arms (DEVIATIONS 2655
   and 2656) that become worth a pod inside that window.
3. What this lane builds instead is the missing number. DEVIATION 2653 adds
   six forward `RESOURCES` rows to the price harness (six instantiations a
   trial build already compiles, each a compile-time readback that launches
   nothing), and DEVIATION 2654 is the leg body that brings them home beside
   the step timers. A wrong arm costs a 30 minute pod; this readback costs
   about four minutes of one that is already renting.

## 1. The rule, spelled out

sm_90 has 65,536 registers per SM and allocates per thread in units of 8
(`tools/knn_selector_kernel_stats.py`, `SM90_REGS_PER_SM`). A 256-thread
block therefore takes `256 * pad8(regs)` registers and an SM holds
`floor(65,536 / (256 * pad8(regs))) = floor(256 / pad8(regs))` of them.
The boundaries, exactly:

| registers per thread | pad8 | 256 / pad8 | blocks per SM |
|---:|---:|---:|---:|
| 64 or fewer | 64 | 4.00 | 4 |
| 65 to 80 | 72 or 80 | 3.55 or 3.20 | 3 |
| 81 to 128 | 88 to 128 | 2.91 to 2.00 | 2 |
| 129 or more | 136 or more | 1.88 or less | 1 |

The three block boundary is 80 registers and not 84, since `pad8(84)` is 88
and `256 / 88` is 2.91. Every arm below is measured against 80 and 128.

## 2. The measured rows this lane starts from

H100 80GB HBM3, commit 6d4bd867, the zdot leg's `resources.txt` as section
20.2 of `docs/lanes/BRIEF_attention_step_2026-09-11.md` prints it. The step
is the NVIDIA default `stash_tiled_fgrid_r32_qres_pf_kvgrid_r32` (word
52327), 290.3 ms, attention 116 ms of it (zdot 66.4, forward 20.7, dq 12.5,
dk/dv 9.9).

| kernel | phase ms per step | regs | pad8 | blocks per SM |
|---|---:|---:|---:|---:|
| `zdot_stash_pf` | 66.37 | 134 | 136 | 1 |
| `fwd_r2` at 32 rows, qres, pf | 20.70 | NOT READ BACK, EVER | | unknown |
| `dq_tiled_pf` | 12.49 | 118 | 120 | 2 |
| `kvgrid_r32` (dk/dv) | 9.93 | 63 | 64 | 4 |

The gap in row two is the finding of this lane. It was checked against every
`resources.txt` and every `attention-step` log under `bench/results`, and
against `run_resources` in `bench/attention_step_price_main.mojo`, whose own
docstring says "eleven backward kernels". No forward kernel of any round has
ever been compiled for its attributes.

The AMD column reads differently and is not this lane's column (the
DigitalOcean MI325X leg,
`bench/results/e1g/2026-09-11_180903-amd-mi325x-do-attention-dkdv/remote/attention-step/resources.txt`,
reads `dq_tiled_pf regs=128 blocks_per_sm_256=4` under a different rule).

## 3. dq, counted, and why it gets no arm

The kernel is `fused_bwd_dq_tiled_pf_kernel[64]`, one block per 64 query
rows of one head, 256 threads as 16 x 16, 16 key tile per iteration. Grid
`b * nh * ceil(L / 64)` is 12 x 32 = 384 blocks per layer at the target
shape, 4,608 per step.

Today. 118 registers, `pad8` 120, `floor(256 / 120)` is 2.13, so TWO blocks
per SM, and the readback's `blocks_per_sm_256` agrees. It is not the kernel
section 20.2 was hunting.

The next boundary. Three blocks per SM needs `pad8(regs)` at 80 or below, so
118 has to fall to 80, a cut of 38 registers.

What can be cut. The harness's own source row for this kernel
(`RESOURCES_BEGIN label=dq_tiled_pf ... accumulators_per_thread=16
operand_registers_per_thread=5 thread_local_floats=0`) counts the whole
per-thread float state at 21 registers, and the source agrees: 16 dq
accumulators (`acc`, RPT x CPT with RPT = CPT = 4), the `ka` operand vector
(4) and the staged `dcell` (1). There is no thread-local array to move (the
zdot kernels' 64-float `vec` has no counterpart here; every operand comes
from the shared page). So:

- moving EVERY accumulator and EVERY staged operand into shared memory
  leaves `118 - 21 = 97` registers, still above 80, and each move adds an
  address register rather than removing one;
- recomputing a value from one already live removes at most one register per
  site and none of them is a row of 38;
- halving the output tile (64 x 64 to 64 x 32 or 32 x 64, 8 accumulators per
  thread) removes 8 of the 38 and doubles two things the source counts. The
  block count per layer goes 384 to 768 and the K staging traffic goes 3.2 GB
  to 6.4 GB per layer (brief section 3.2's count, every block stages its
  head's whole causal key range). Its ceiling, if the occupancy model held
  perfectly, is `12.49 * (1 - 2/3) = 4.16` ms per step, 1.4 percent of 290
  ms, bought with a doubling of the kernel's only large global read.

So 97 registers of dq's 118 are addresses, loop indices, `_row_range`
results and the shared loads the backend keeps in flight across the unrolled
fold, and nothing on this lane's list reaches them. NO ARM FOR dq. A lane
that wants those 97 registers is attacking the backend's unrolling, not the
kernel's arithmetic, and it cannot price a variant without a readback per
variant, which is exactly what DEVIATION 2653 makes cheap.

## 4. The forward, counted from the source, and the window where an arm pays

The kernel the NVIDIA default runs is
`fused_attn_forward_r2_kernel[64, 32, True, True, False]` (32 query rows per
block, Q residency, preflushed seams), launched once per layer by
`_launch_fwd_r2` and reading 20.70 / 20.74 ms per step. Grid
`b * nh * ceil(L / 32)` is 12 x 64 = 768 blocks per layer, 9,216 per step,
on about 132 SMs, so the SM count and not the grid bounds how many run at
once at one or two blocks per SM.

Per-thread float state from the source, at TQ 32, RPT 2, CPT 4.

| value | registers | live |
|---|---:|---|
| `mpart`, the row maximum parts | 2 | the whole kernel |
| `dacc`, the denominator | 1 | the whole kernel |
| `cacc`, the context accumulators | 8 | the whole kernel |
| `dots`, the score accumulators | 4 | pass 1, inside the key block |
| `qa`, `ka`, the staged dot operands | 2 and 2 | pass 1, inside the p window |
| `va`, the staged V operands | 4 | pass 3, inside the key block |
| thread-local arrays | 0 | there are none |

At most 19 floats are live at one time (11 persistent plus 8 in pass 1),
against the shipped zdot copy's 70 (2 accumulators, 4 operands, the 64-float
`vec`) and dq's 21.

Why that does NOT settle the kernel's register count. dq holds 21 counted
floats and reads 118 registers. A reading that took the forward's 19 floats
to mean four blocks per SM is the same reading that would have put dq at
four, which the readback refutes. The forward has three passes, four
comptime-unrolled staging loops per key block, two `_row_range` calls per
row per pass and a `[32][16]` K page at stride 20, so its addressing
pressure is at least dq's order. Its count could plausibly land anywhere
from the seventies to the one fifties, which spans three of the four
occupancy classes in section 1.

The window. Writing the decision out before the number exists, so the next
lane reads it and does not re-argue it.

| readback | blocks per SM | what it means | action |
|---|---:|---|---|
| 128 or fewer | 2 or more | The forward is not the zdot case. The next boundary (80) needs a cut of at least 48 and at most 19 floats can move, so the lens is CLOSED for the forward as well. | No arm. File the number and stop. |
| 129 to 147 | 1 | Moving the 11 persistent floats (and, at the top of the range, the 4 pass-1 score accumulators) into the shared page lands the kernel at or below 128, which is TWO blocks per SM. | DEVIATION 2655 or 2656, section 6. |
| 148 or more | 1 | Out of reach of the float moves; what is left is addressing and staging, a different lane (fewer staging round trips, or a geometry that stages once). | Name it, do not build it here. |

## 5. Expected saving, with the arithmetic

Only for the middle row of section 4, since the other two rows buy nothing.

The forward at 20.70 ms per step, moving from one block per SM to two.

- The section 20.2 model (a kernel's time scales with block iterations over
  blocks per SM) gives `20.70 / 2 = 10.35` ms, a saving of 10.35 ms, 3.6
  percent of the 290 ms step.
- The one measured precedent for an occupancy-only change on this file is
  `_kvgrid_r32`, which took dk/dv from 106 registers and 2 blocks per SM to
  63 and 4 and moved that kernel from 1.0 to 0.8 ms per call (section 19), a
  ratio of 0.8 for a DOUBLING of the blocks per SM, not 0.5. On that
  precedent the forward prices at `20.70 * 0.8 = 16.56` ms, a saving of 4.14
  ms, 1.4 percent.
- Both clear the 1 percent bar section 20.1 set for this final pass (2.9 ms
  of 290 ms), the first comfortably. Neither is measured; both are what the
  readback would license a lane to go after.

The cost of the readback itself is zero launches and, on the leg, one build
of the price harness and one run of it at L 512 with timing off, about four
minutes (section 8).

## 6. The two arms, named and NOT built (DEVIATIONS 2655 and 2656, REFUSED BY THE READBACK 2026-09-12)

**NEITHER ARM WILL BE BUILT. THE NUMBER CAME BACK AND CLOSED THE WINDOW.**
The forward reads 100 registers at TWO blocks per SM (section 8), which is
the first row of section 4's table: "No arm. File the number and stop." The
arithmetic is tighter than that row assumed and worth writing down, because
it is what forecloses a later lane re-arguing it. Three blocks per SM needs
`pad8(regs) <= 85`, so 80 registers or fewer, a cut of 20 from 100. At most
19 floats can move (the 11 persistent plus the 4 pass-1 score accumulators
plus the 4 staged operands), so moving EVERY movable float lands at 81, pads
to 88, and still reads two blocks per SM. The forward cannot reach the next
occupancy class by moving floats at all, which is exactly the property the
zdot kernel did not have (134 to 64, four blocks).

Both arms are kept described below, unbuilt, because the description is what
makes the refusal checkable. Neither is written on this branch. Each
inherits an identity argument that is already filed, which is the reason
each would have been cheap to build had the number said so.

DEVIATION 2655, `_fres` (the context accumulators in the shared page). The 8
`cacc` floats per thread become a `[32][64]` page written by the same
`_step` or `_step_preflushed` per key, one shared load and one shared store
per term where the register version has neither. This is exactly the trade
`_estash_dres` (DEVIATION 2651) makes for dctx in the zdot kernel, and its
identity argument is 2651's item 5 plus 14.3's item 5 (each accumulator is
still one output cell's chain from +0.0 over key blocks ascending and keys
within a block ascending, and which storage holds the running value is not a
term of the chain). The page goes from 15,232 to 23,424 bytes, which still
fits every column's limit. Removes 8 of the 11 persistent floats and adds
one page address. Worth building when the readback lands at 129 to 136, and
only then, because the shared traffic it adds is the forward context chain's
whole term count.

DEVIATION 2656, `_fgrid_r16` (16 query rows per block). RPT falls from 2 to
1, so `cacc` goes 8 to 4, `mpart` 2 to 1, `dots` 4 to 2 and `qa` 2 to 1, a
cut of 11 floats without a single shared access added. Its identity argument
is 14.3 verbatim (rows per block is a partition of consecutive rows, every
row lies in exactly one block, the loop bounds are block-uniform, every
thread reaches every barrier, and the row maximum stays an `identical_fmax`
fold whose grouping is free). The page goes from 15,232 to 13,184 bytes.
The cost is the one `_fgrid_r32` already paid once against `_fgrid_r64` and
won on the H100 (lean step 0.3845 to 0.3346 s, section 15), doubled again.
Blocks per layer go 768 to 1,536 and the K staging per layer doubles, since
each block stages the same 32-key range for half as many rows. Worth
building when the readback lands at 129 to 140, and it is the arm to prefer
over 2655 if both fit, because it adds no shared traffic.

A lane that builds either one owes what section 20.6 owed: a bit in
`ATTN_ARM_NEW_BITS` and in `ATTN_ARM_DEFAULT_REFUSED_BITS`, the parser and
name function as inverses, `step_count_launch` before every launch and
`step_count_sync` before every synchronize, the arm in
`transformer/checks/transformer_attention_arms_check.mojo` with its reach
and sabotage (the forward flips of 14.5 already say what must move and what
must hold), and a `RESOURCES` row of its own beside the six this lane adds.
None of that is on this branch, because none of it is warranted until the
number exists.

## 7. What was changed, per file

| file | deviation | change |
|---|---:|---|
| `bench/attention_step_price_main.mojo` | 2653 | `_res_fwd_r2[TQ, QRES, PF]` and six calls in `run_resources`, each in its own `try`, all inside `comptime if ATTN_ARM_TRIAL` |
| `tools/attention_regs_leg.sh` | 2654 | the leg body (section 8) |
| `docs/lanes/BRIEF_attention_regs_2026-09-11.md` | | this file |

No kernel was added, no arm bit was added, and no file under
`transformer/`, `checks/`, `gemm/`, `core/` or `training/` was touched. The
parser, the name function, `ATTN_ARM_NEW_BITS`,
`ATTN_ARM_DEFAULT_REFUSED_BITS`, `attn_default_arm_for` and
`transformer_attention_arms_check.mojo` are unchanged, so the arms check's
counts and the fused check's lines read exactly as they did on origin/main.
A shipped build compiles nothing new, and a build of the harness WITHOUT
`-D MOJOLEARN_ATTN_ARM_TRIAL=1` compiles no new pipeline either (the six
rows are behind the trial guard, the pattern the `_estash` rows use).

The six rows, each an instantiation a trial build already compiles for the
DEVIATION 2530, 2531 and 2533 forward arms, so no new pipeline on any build.

| label | instantiation | what it prices |
|---|---|---|
| `fwd_r2_r32_qres_pf` | `[64, 32, True, True, False]` | THE NVIDIA DEFAULT's forward, the 20.7 ms row |
| `fwd_r2_r32_qres` | `[64, 32, True, False, False]` | the preflush knob's register cost |
| `fwd_r2_r32_pf` | `[64, 32, False, True, False]` | the Q residency knob's register cost |
| `fwd_r2_r32` | `[64, 32, False, False, False]` | the 32-row copy with both knobs off |
| `fwd_r2_r64_pf` | `[64, 64, False, True, False]` | the 64-row geometry, preflushed |
| `fwd_r2_r64` | `[64, 64, False, False, False]` | the 64-row copy, the shipped sstash arithmetic |

Together they say what each round 3 knob did to the register count, which
is the reading round 3 could not do, and the r32 to r64 pair says whether
the rows knob crosses a boundary by itself.

## 8. The H100 leg

### WHAT IT FOUND (2026-09-12, measured): the forward register count

RunPod pod 4pmnz0eju0iptm, NVIDIA H100 80GB HBM3, driver 580.126.09, commit
5b7e1e41, phase one only (`MOJOLEARN_ATTN_REGS_FULL=0`), 58 seconds of
build and 2 seconds of readback. Evidence
`bench/results/e1g/2026-09-12_124715-nvidia-h100-owed/remote/attention-regs/`
(`resources.txt`, `gate.txt`, `status.tsv`). Both stages exit 0, the run
ends `attention_step_price: PASS`, and there are ZERO `RESOURCES_ERROR`
rows: this is the first box on which the forward's count has ever been read,
because Metal refuses the attribute outright.

| forward instantiation | regs | pad8 | blocks_per_sm_256 |
|---|---:|---:|---:|
| `fwd_r2_r32_qres_pf` (the shipped NVIDIA default) | 100 | 104 | 2 |
| `fwd_r2_r32_qres` | 102 | 104 | 2 |
| `fwd_r2_r32_pf` | 96 | 96 | 2 |
| `fwd_r2_r32` | 96 | 96 | 2 |
| `fwd_r2_r64_pf` | 166 | 168 | 1 |
| `fwd_r2_r64` | 166 | 168 | 1 |

Three readings, in order of what they settle.

1. THE ARMS ARE REFUSED. 100 registers is section 4's first row, so
   DEVIATIONS 2655 and 2656 are not built (section 6 carries the
   arithmetic). Section 4's own guess is falsified in the useful direction:
   it said the count "could plausibly land anywhere from the seventies to
   the one fifties", and the answer is at the low end, so the forward was
   never the register-bound kernel the zdot was.
2. THE 32-ROW GEOMETRY IS EXPLAINED, not just measured. `_fgrid_r64` reads
   166 registers at ONE block per SM against `_fgrid_r32`'s 100 at two.
   Section 15's H100 result (lean step 0.3845 to 0.3346 s when 64 rows
   became 32) had no mechanism attached to it until now; the occupancy
   class is the mechanism.
3. THE LENS REPRODUCES ACROSS PODS. The same readback re-measured the zdot
   rows on a second pod and a later commit: `zdot_stash_pf` 134 at one
   block, `zdot_estash_pf` 125 at two, `zdot_estash_dres_pf` 64 at four,
   identical to the numbers section 20.11 of the attention step brief took
   from pod 43v3euoz80r9zy. Also filed: `dq_tiled_pf` 118 at two (section 3
   argued no mechanism takes it to three, and the count agrees),
   `dkdv_tiled_pf` 106 at two, `kvgrid_r32` 63 at four, `kvsplit_r32_fold`
   48 at five.

Phase two was not run (`attention_step_leg=NOT RUN`), so this leg carries no
timing and no arm, which is what section 8 always said it was for.

### The body

Body `tools/attention_regs_leg.sh`, two phases.

Phase one, the readback, standalone and first, so the one number this lane
needs comes home even if the lease dies later. It builds
`bench/attention_step_price_main.mojo` under IDENTICAL with the trial define
and runs it once at L 512 with `MOJOLEARN_ATTN_RESOURCES=1`,
`MOJOLEARN_ATTN_TIMING=0`, `MOJOLEARN_ATTN_ORACLE=0` and
`MOJOLEARN_ATTN_REACH=0`, candidate and baseline both the NVIDIA default by
name, then greps the `RESOURCES` lines into `resources.txt`. Nothing is
timed, no corpus is fetched, no binding is built, about four minutes.

Phase two, the full body (`MOJOLEARN_ATTN_REGS_FULL=1`, the default), is
`tools/attention_step_leg.sh` with `MOJOLEARN_ATTN_BASELINE` and
`MOJOLEARN_ATTN_LEG_ARMS` and `MOJOLEARN_ATTN_LEG_LM_ARMS` all the NVIDIA
default `stash_tiled_fgrid_r32_qres_pf_kvgrid_r32`, TIMERS ON
(`MOJOLEARN_ATTN_LEG_SKIP_TIMERS=0`) and both corpora. It brings home the
per-kernel breakdown (`timers_summary.tsv` and the `lmtiming-*` component
lines) beside the register rows, so `attn.fwd_r2_kernel` at 20.7 ms and
`attn.bwd_dq_tiled_pf` at 12.5 ms are read on the same pod as the counts
that explain them. There is no candidate arm on this branch, so the price
compares the default with itself and its ratio is 1.0 by construction; the
`RESOURCES`, `timers` and `lmtiming` lines are what the leg is for.

From a `git worktree add --detach` checkout at the lane's merge commit:

    MOJOLEARN_RUNPOD_KEY_FILE=$HOME/.mojolearn_runpod_key \
    MOJOLEARN_GPU_ARCHS=sm_90a \
    MOJOLEARN_GEMM_LEG_EXTRA=tools/attention_regs_leg.sh \
    MOJOLEARN_GEMM_LEG_OUT=bench/results/e1g/$(date -u +%Y-%m-%d_%H%M%S)-nvidia-h100-attention-regs \
    sh tools/gemm_remote_leg.sh nvidia --payload gemm --rent --minutes 60 \
        --gpu "NVIDIA H100 80GB HBM3" \
        --local-card bench/results/e1/2026-08-28_131651-runpod-nvidia/lanes/gemm.identical.card

Set `MOJOLEARN_ATTN_REGS_FULL=0` in the body's environment to run phase one
alone, which fits inside any other NVIDIA lease that is already renting.

Gates. Phase one exits 0, `resources.txt` carries
`RESOURCES_BEGIN label=fwd_r2_r32_qres_pf` and a `RESOURCES label=...
regs=` and `blocks_per_sm_256=` line for each of the six labels, and
`attention_step_price: PASS`. Phase two is section 6 of the attention step
brief with the NVIDIA default in place of `baseline` (arms check exit 0,
smokes exit 0, every `BITS` MATCH on both corpora's activations, `REACH`
with `clean_restored=True`, lean steps `limited: false`).

There is no flip. This lane changes no kernel and no default, so
ENGINEERING_RULES 9 has nothing to decide; the leg's product is the six
register rows and what section 4's table says to do with them.

## 9. Risks only a build or a box can settle

1. Nothing was compiled. The likeliest fault is in the new rows themselves,
   where a comptime kernel alias is bound inside a generic `def` with two
   Bool parameters and a String label (the `_res_zdot_estash[DRES]` and
   `_res_dkdv_r2[BJ]` rows are the precedents for exactly that shape), and
   the call to `_fwd_r2_page_bytes`, a module-private `def` imported across
   modules the way `_download` and `_upload` already are.
2. `ctx.compile_function` on a forward kernel may answer some attributes and
   raise on another. Each row is its own `try` and prints a
   `RESOURCES_ERROR` line, so a vendor that refuses one keeps the rest; on
   Metal the whole section may answer `RESOURCES_ERROR`, which is what the
   M4 gate expects and not a failure.
3. `blocks_per_sm_256` is the runtime's own occupancy answer and accounts
   for static shared memory as well as registers. The forward's page is
   15,232 bytes at 32 rows with Q residency, so at 228 KB of shared memory
   per SM the page allows 15 blocks and cannot be what bounds the number; if
   the readback disagrees with `floor(256 / pad8(regs))` the page is the
   first thing to check, and the row prints it.
4. Section 4's window is a reading, not a measurement. If the readback lands
   at 129 or above, DEVIATION 2655 or 2656 is worth a pod and its saving is
   section 5's; if it lands at 128 or below, this lane's answer is that the
   register lens is closed on the forward too, and the next 20 ms has to be
   attacked as staging and barriers, not as occupancy.
5. The dq argument in section 3 rests on one readback taken on one pod at
   one commit (134 for the zdot kernel and 118 for dq, section 20.2). If a
   later build changes dq's count the arithmetic has to be redone, and the
   row that says so is printed by every price run.
</content>
</invoke>
