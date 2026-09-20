# GPT-3-small exact GEMM fold-storage trial

## Finding

The shipped kpack kernel instantiates a profile-wide 16-level local fold
stack for every 8x8 register tile. That is `16 * 64 * 4 = 4,096` bytes per
thread even when the call's fixed arithmetic tree is much smaller:

- GPT-3-small `k=768`: `P=6`, covered by FS4 (1,024 bytes/thread).
- GPT-3-small `k=3072`: `P=24`, covered by FS8 (2,048 bytes/thread).

This aligns with the prior H100 resource capture in
`bench/results/gemm_resources_2026-09-10`: 255 physical registers/thread,
4,144 local bytes/thread, one block/SM, and 12.5% theoretical occupancy for
the predecessor kernel. Local storage/spill pressure is therefore a more
credible target than another tile search.

## Exact trial

An opt-in trial entry selects FS4 when the existing launch folds at most 8
leaves, FS8 at most 128, and otherwise retains FS16. It derives the bound
only from `contract_partition(k)` and the existing group rule. Leaves,
windows, per-cell products/additions, merge order, and output storage are
unchanged. Smaller stacks cannot compile without both
`MOJOLEARN_GEMM_ARM_TRIAL` and `MOJOLEARN_GEMM_FOLD_SPECIALIZE_TRIAL`.

The exhaustive host-static check covered every `P` in 1..1024 and group
class 0/1/2/4/8/16/32/64/128/256; every returned FS class covered the maximum
leaves folded by its launch. The standard non-trial IDENTICAL device suite
passed all eight gates after the launch-helper refactor.

The Apple M4 trial build completed and the nine weighted GPT forward/input-
gradient cases all had zero full-output mismatches and stable hashes. Apple
does not select the kpack body, so the entry intentionally fell through to
the shipped path; its noisy same-path timings are not candidate performance
evidence.

## Static NVIDIA resource evidence

A local `--emit asm --target-accelerator sm_89` build emitted all three
specializations. Their PTX local depots are exactly:

| specialization | local depot/thread |
|---|---:|
| shipped FS16 | 4,096 bytes |
| trial FS8 | 2,048 bytes |
| trial FS4 | 1,024 bytes |

This is source/static evidence, not physical occupancy. ptxas assembly and a
timed exact NVIDIA A/B remain required before production promotion. Two
externally owned NVIDIA pods were active during this lane, so none was
touched and no new rental was created. The fastest next step is to copy this
already compiling trial to the first cleared guarded NVIDIA pod, capture
ptxas register/local/spill counts for the three entry points, then run the
existing weighted interleaved matrix. AMD qualification is warranted only if
that physical resource change produces a repeatable timing win.

## Disposition

Promising but pending vendor qualification. The commit contains only the
explicit opt-in trial, its static bound gate, and benchmark routing. The
production selector remains unchanged.
