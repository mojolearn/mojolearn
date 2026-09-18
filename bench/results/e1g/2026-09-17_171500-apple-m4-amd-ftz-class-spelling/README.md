# The AMD seam can go from 8 issue slots to 5 WITHOUT leaving the contract

2026-09-17, lane `lane/gemm-next`, on the M4, one core, `nice 19`, NO GPU and NO
RENTAL. A COMPILE result read out of the emitted gfx942 GCN, not a timing.

## Why this, and why it is not the dead lever

The wave-mode arm (`bench/results/e1g/2026-09-17_205022-amd-mi300x-hotaisle-seam-mode-probe-b`)
took the seam from 8 instructions to 1 and was MEASURED on an MI300X to compute
`eb76eb53d65e0007`, none of the three semantics, returning `00000000` at the
boundary triple where the contract requires `00800000`. It is a defect. Dead.

This is a different thing and the difference is the whole point: the flush stays
POST-ROUND, applied to the already-rounded FMA result, so it computes the SAME
FUNCTION as `checks/numerics.mojo::ftz`. Only the SPELLING changes.
`v_cmp_class_f32` tests the subnormal class in one instruction where the current
spelling uses two `v_and` and two `v_cmp`.

## The three loop bodies, one compile, all three kernels launched

An unlaunched kernel is dead-stripped and its check "passes"; all three are
enqueued. Every sidecar here begins
`.amdgcn_target "amdgcn-amd-amdhsa-unknown-gfx942"` -- read that line first,
because `MOJOLEARN_GPU_ARCHS=gfx942` does NOT retarget `mojo build` and will
silently emit `air64-apple-macosx`. `--target-accelerator gfx942` does.

Per product step, excluding the two loads, the address increment, the loop
counter and the branch, which are identical in all three:

| arm | issue slots | body |
|---|---:|---|
| **SHIPPED** `ftz(fma)` | **8** | `v_fmac_f32`, `v_and`, `v_and`, `v_cmp_ne`, `v_cmp_eq`, `s_and`, `v_and`, `v_cndmask` |
| **CLASS** this spelling | **5** | `v_fmac_f32`, `v_and`, `v_cmp_class_f32`, `s_nop 1`, `v_cndmask` |
| BARE `fma` alone (control) | 1 | `v_fmac_f32` |

Four real instructions plus one `s_nop 1`, which is the gfx9 hazard wait between
a VALU write of VCC and the `v_cndmask` that reads it. Counted honestly the seam
is 8 slots today and 5 with this spelling, or 8 instructions against 4 if the
scheduler can fill the hazard from another cell's chain -- and in the real
kernel, where many cells are in flight, it usually can.

NVIDIA's seam is 2. So this closes a little over half the AMD/NVIDIA seam gap.

## Why it is the same function, and why that is an ARGUMENT and not yet a proof

`ftz` is: if the exponent field is 0 AND the mantissa is non-zero, return the
sign bit alone; otherwise return the value. AMD's class mask `0x90` is bit 4
(negative subnormal) and bit 7 (positive subnormal), and "subnormal" in that
encoding is exactly "exponent 0, mantissa non-zero". Signed zero is a separate
class (bits 5 and 6) and is therefore left alone, as `ftz` leaves it. NaN and
infinity are separate classes and are left alone, as `ftz` leaves them. The
predicate is the same predicate and the selected value is the same value.

**That is a reading of the ISA, not a measurement, and bit equality is
absolute.** The proof is a device run: the same spelling as a lane of
`gemm/checks/gemm_seam_probe.mojo`, on an MI300X, hashing to
`62a6b5621e27c707` (`rtf`, the contract) over all 262,144 triples and matching
the `shipped` lane with mismatch count 0. The harness already takes extra lanes
(the `modeftz` lane was added the same day). Until that run exists this is a
candidate, not a result, and no arm should be built on it.

## Where it would live

NOT an inline vendor branch. `[[always-GPU-agnostic]]`: this is a kernel-matrix
CAPABILITY row beside `lib_hardware_ftz_fma_for` (which is how NVIDIA's
`mul.rn.ftz` is selected today), read by `ftz` or by a seam helper, answering
true only on columns whose ISA documents the class test. Every other column
keeps the software spelling it compiles now.

## What it is worth, and what nobody knows

About half the AMD step is GEMM (559 ms of roughly 1.15 s, brief 18.4), and every
product step of it carries these instructions. But **whether the MI300X's
accumulate loop is ISSUE-BOUND is UNMEASURED**, and on NVIDIA it is not: brief
15.5 measured 11,960 cycles per block window against 4,120 issue cycles for the
loop, and 16.3's bare FMA chain read 0.31 of the body. So NO SPEEDUP IS CLAIMED
HERE. The AMD step has never been itemized at all; that measurement
(`tools/step_breakdown_leg.sh` on an MI300X, which derives its own column) should
come before this arm, because it is what says whether the loop is where AMD's
time is.
