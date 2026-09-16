# codex/metal-block-copy-fusion — isolated candidate, 2026-09-16

The complete Problem 2 brief was omitted from the user paste (97 hidden lines)
and could not be located in the scratchpad or evidence notes. This branch is a
reviewable candidate for the launch overhead described in
`docs/lanes/LANE_STATUS_lane-neural-shape-shrink.md`, not a claim that Problem 2
is fixed. No production dispatch, current identity fixture, reference table,
release branch, or other session's worktree was changed. Nothing was merged
or pushed.

## Candidate

`training/byte_lm_block_copy.mojo` implements a single-launch copy of the nine
disjoint tensors in one decoder block, in either direction. The y grid selects
a tensor; each thread copies one FP32 value, without floating point arithmetic.
It validates absolute offsets and per-tensor capacity before enqueueing and
leaves synchronization and buffer lifetime with the caller.

The current `_unpack_block` and `_pack_block` in `training/byte_lm.mojo` launch
nine independent `_copy_into` kernels apiece. Wiring this candidate would
replace 18 launches with 2 per block. That is a source count, **not a measured
training speedup**. The helper is reached only by its new check on this branch.
No dispatch switch was added.

## Validation

Built with `mojo build -j 1 -D MOJOLEARN_NUMERIC_IDENTICAL=1 -I .` using the
existing main worktree's activated Pixi environment. Compilation used
`mac_slot.sh run`; execution used `mac_slot.sh metal`, which sets nice 19 and
single-thread/job environment knobs. One process at a time in this lane.

The independent oracle is the existing nine `_copy_into` calls, not a
roundtrip through the new implementation. Pack inputs are independently
initialized, so inverse mapping errors cannot cancel each other.

The final control passed seven cases: all-empty ranges; uneven ranges with
empty, one-element, 255/256/257-element and multiple-block tails; both blocks
of the default profile; and all three blocks of an alternate GQA profile.
All comparisons use UInt32 bit patterns, including signed zeros, subnormals,
infinities and NaN payloads. Checks include unchanged pack sources, untouched
flat-buffer prefixes/tails and tensor-buffer tail canaries. Five invalid
range/offset cases are rejected by name in both directions for every fixture
(70 refusal checks).

The sabotage shifts every pack destination by one element while preserving
range lengths and staying within the allocation. With
`MOJOLEARN_BLOCK_COPY_SABOTAGE=1`, the final binary exited **1** with:

    uneven pack including prefix and tail: bits differ at 3

The final control exited **0**. Raw logs, source hashes and the binary hash are
in `bench/results/byte_lm_block_copy/2026-09-16-apple-m4/`.

An earlier prototype with a SIMD[int32,16] offset argument and dynamically
selected pointer compiled but Metal rejected compute-pipeline creation with
an internal compiler error. The current scalar-offset, explicit-branch
spelling passed. Both details changed together; this is not a diagnosis of
which earlier construct triggered the compiler failure.

## Remaining work

Recover the full Problem 2 brief and reconcile the active Codex/Fable sessions
before deciding whether this candidate addresses the requested scope. If it
does, wire `_unpack_block` and `_pack_block` with the existing synchronization
and lifetime guarantees and update launch accounting. Then measure the
complete training step and gate its full state against the existing dispatch,
including other vendor columns. This branch provides neither an end-to-end
performance result nor AMD/NVIDIA validation. Do not present it as a shipped
fix or merge it without the user's express authorization.
