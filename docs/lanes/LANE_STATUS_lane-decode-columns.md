# lane/decode-columns, 2026-09-16

Evidence: `/Users/andrewhendel/mojolearn-evidence/decode-columns/`.
Leg artifacts: `bench/results/e1g/2026-09-16_134311-nvidia/` and
`bench/results/e1g/2026-09-16_amd-mi325x-decode-columns/`.

## What was owed

`lane/stateful-cpu-decoding` merged as `3708d7596` and shipped the claim that
a sequence decoded ONE TOKEN AT A TIME with a carried state is BITWISE the
same sequence run as ONE fresh-state forward pass, at every position. It
proved that on TWO columns, the CPU host column `cpu-apple-m4` and the
Apple/Metal column, and its own report closed by recording that the two
remaining vendor columns for the `stepfull` part were still outstanding and
that it had rented no box, because it was asking first.

This lane is the answer to that sentence, and nothing else. (The sentence is
deliberately NOT quoted here. A correction that quotes the text it
supersedes makes "is the old claim gone?" return a match forever, which is
how a probe was killed on 2026-09-13; the operative words now live only in
git history.) No new lanes, no
new cells, no sweep.

## THE RESULT: four columns, eight lanes, one hash each

`tools/identity_break.py --step-full`, base fixture, two repeats per GPU
column. The `stepfull` hash is the whole-sequence pass's bytes, so a column
that holds the equality lands on the same sixteen hex digits as every other
column that holds it.

| lane | CPU `cpu-apple-m4` | Apple Metal `arm64` | NVIDIA RTX 3090 `sm_86` | AMD MI325X `gfx942` | verdict |
|---|---|---|---|---|---|
| transformer | 99e9fe5ec967e1dd | same | same | same | IDENTICAL x4 |
| transformer-window | a05e05cf5055c79f | same | same | same | IDENTICAL x4 |
| mamba1 | f582474b00117f8e | same | same | same | IDENTICAL x4 |
| mamba2 | bfd516aa93fe1b12 | same | same | same | IDENTICAL x4 |
| mamba2-dtlimit | 44421178c1c5b188 | same | same | same | IDENTICAL x4 |
| mamba3 | 6a8f4924575a931d | same | same | same | IDENTICAL x4 |
| samba | e9c89afd1eb7f273 | same | same | same | IDENTICAL x4 |
| samba-untied-dropout-accum | dc215181275f4d1f | same | same | same | IDENTICAL x4 |

`30_diff_four_columns.log`: `summary (stepfull): IDENTICAL=8`, every cell
`IDENTICAL x4`. The `infer` cells the same run carries read `IDENTICAL=10`
across the same four columns, and nothing moved or refused anywhere.

THE GPU COLUMNS ARE NOT THE CPU RUN AGAIN. `identity_break`'s `_public_est`
returns the `*Inference` wrapper on a CPU column only and says so in its
docstring ("GPU columns are unchanged"). `ml.vendor()` read back `cuda` on
the RunPod box and `hip` on the DigitalOcean box, so the part asked
`TransformerBlock`, `Mamba1/2/3Block` and `SambaStack` themselves, their own
`forward`, `allocate_state` and `step`, on that silicon.

The Apple column was extended from the five lanes
lane/stateful-cpu-decoding ran to all eight, one Metal job through
`mac_slot.sh metal`, this branch's harness over the shared checkout's Metal
bindings (`06_apple_metal_stepfull_all8.json`).

## The sabotage arms, on each box, not only on the Mac

TWO arms, because the first arm this part ever had could not fail: it
perturbed `k_cache[0]` and `h[0]` by one ULP and reported BITWISE EQUAL,
which is indistinguishable from a pass. The transformer's `k_cache` absorbs
one ULP at all of the first thirty-two cells, mamba1's `conv_window` absorbs
it at cells 0 and 1, and mamba2's `h` is all zeros inside the first chunk so
+1 ULP there is a denormal that `ftz` flushes back to zero.

| arm | Mac rehearsal | NVIDIA RTX 3090 | AMD MI325X |
|---|---|---|---|
| `MOJOLEARN_IDENTITY_BATCH_SABOTAGE=1`, every stepfull cell must read BATCH_MOVED | 8 of 8 | 8 of 8 | 8 of 8 |
| ONE ULP on a carried cache cell, GPU classes, must fire | 6 of 6 | 6 of 6 | 6 of 6 |
| ONE ULP, `*Inference` wrappers over the neural host binding | 6 of 6 | 6 of 6 | REFUSED, see below |

Every BATCH_MOVED cell names the position and both bit patterns, for example
`mamba1/base stepfull: FIRST DIFFERING POSITION 0 of 16: element 0: full
0x3eb0f4dc vs 0x3eb0f4dd`. Every one-ULP arm names the cell it moved, its
bits before and after, and the position that moved, for example
`transformer(window=0): piece 1 cell 3 0x3ceb9b3c -> 0x3ceb9b3d (+1 ULP) at
step 8` then `FIRST DIFFERING POSITION t=12 ... 0x3fae086e vs 0x3fae086f`.
The scan is what makes that arm evidence: it FAILS the run when no cell
fires, so a silent equal cannot pass for a result.

The one-ULP arm was rehearsed on this Mac against the GPU BLOCK CLASSES,
not only the wrappers, before either box was rented, because the wrappers
bind `_mojolearn_neural_host` by name and would have run the CPU host route
on the box while looking like a GPU arm.

## Everything the boxes did, in order

| phase | NVIDIA RTX 3090 sm_86 | AMD MI325X gfx942 |
|---|---|---|
| bindings: base, training, mamba, transformer | 0 / 0 / 0 / 0, 232 s | 0 / 0 / 0 / 0, 119 s |
| `ml.vendor()` read back | `cuda` | `hip` |
| stepfull column, 8 lanes, x2 | exit 0, 8 STABLE | exit 0, 8 STABLE |
| stepfull BATCH_MOVED arm | 8 of 8 | 8 of 8 |
| on-box diff, column vs its own sabotage | `stepfull: BATCH_MOVED=8` | `stepfull: BATCH_MOVED=8` |
| GPU classes, step vs full | EQUAL, 6 of 6 | EQUAL, 6 of 6 |
| GPU classes, ONE ULP | fires, 6 of 6 | fires, 6 of 6 |
| `bindings/build_neural_host.sh` | exit 0 | **exit 2, refused** |
| host route, step vs full | EQUAL, 6 of 6 | not reached |
| host route, ONE ULP | fires, 6 of 6 | not reached |

## The one thing that failed, and why

`build_neural_host` exited 2 in zero seconds on the MI325X:

    neural host compiles the CPU column only; MOJOLEARN_TARGET_COLUMN=amd is refused

`tools/do_extra_leg.sh` EXPORTS `MOJOLEARN_TARGET_COLUMN=amd` (the RunPod
runner exports nothing, which is why the same body's host build passed on
the 3090), and `bindings/build_neural_host.sh` refuses any column but `cpu`
BY NAME. Unsetting `MOJOLEARN_GPU_ARCHS` was only half the fix;
`tools/identity_three_columns_leg.sh` has carried both halves since
2026-09-14 and this body did not. `3e5cde69d` adds
`MOJOLEARN_TARGET_COLUMN=cpu`.

REPORT IT AS WHAT IT IS. It cost the MI325X leg a THIRD, free column (the
x86 CPU host route beside the GPU one), which is a bonus and was never the
owed work. The AMD deliverable, the `stepfull` GPU column and both of its
sabotage arms, is complete and was taken before that phase ran, which is why
the body puts the deliverable first.

## The boxes, and the spend

| leg | provider | box | created | gone | rate | spend |
|---|---|---|---|---|---|---|
| NVIDIA | RunPod | RTX 3090, sm_86 | 13:43:27 | 13:53:40, HTTP 404 | $0.22/h | about $0.04 |
| AMD | DigitalOcean | MI325X, gfx942, tor1 | 13:44:18 | 13:50:06, HTTP 404 | about $3.80/h | about $0.38 |

TOTAL about $0.42, and at worst $3.84 if DigitalOcean charges the MI325X a
full hour rather than by the second. One RTX A4000 create returned no stock
and made no pod, so it cost nothing.

NO SPEC WAS PINNED on either side. The NVIDIA leg walked a cheapest-first
list of sixteen specs and stopped at the second; a Hot Aisle leg starved on
a pinned spec for thirty minutes this morning and created nothing. AMD went
straight to DigitalOcean: RunPod lists `AMD Instinct MI300X OAM` but its
`uninterruptablePrice` was null, which is no stock.

Both boxes were asked for by the API after the delete and both answered
HTTP 404. A listing taken afterwards shows zero live RunPod pods and zero
live DigitalOcean droplets.

## What this does NOT say

A matching hash means the two runs' output buffers held the same bits, not
that the two computations were identical, and nothing is known about what
the hash does not cover. The shapes are the part's own: B = 2, L = 16, one
fixture. That is the same scope the CPU and Apple columns were taken at, and
it is what makes the four columns comparable; it is not a statement about
long sequences or serving batch sizes, which the `batchscale` and `ragged`
parts carry separately.

The RunPod leg's own gemm gate ran beside this work and read
`RESULT: IDENTICAL` against the Apple card supplied to it. That card came
from commit `9af299bc7`, not this one; nothing can check that for you, and
the gemm gate is not this lane's deliverable.

## Files

* `tools/decode_columns_leg.sh` -- the on-box body, vendor agnostic, no GPU
  pinned, deliverable first, both sabotage arms.
* `docs/lanes/LANE_STATUS_lane-decode-columns.md` -- this file.
