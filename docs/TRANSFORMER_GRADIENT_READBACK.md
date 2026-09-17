# Transformer gradient readback

The Python GPU binding now submits its ten backward-result copies together,
then waits once before returning. Previously it waited after every copy.
This removes nine explicit completion waits per backward call without changing
arithmetic, output shapes, fixture sizes, or the public Python API.

This is shared Metal/CUDA/HIP code. CPU execution is unchanged: it has no device
readback queue. The focused verification tool supports all four backends and
refuses a binding whose reported vendor or numeric mode differs from the request.
The existing scoped test runners remain the way to reduce routine test work on
all backends; this optimization does not replace their coverage controls.

## Ownership and correctness

Deferred copies are permitted only for full buffers. `LlamaBackwardStages`
owns all ten sources until after the final wait, and the calling Python frame
retains all destination arrays. Partial-buffer views synchronize before their
local owners die. The diagnostic legacy pinned-staging path remains synchronous.
Other callers retain the helper's synchronous default. Do not move the final
wait past owner destruction or return to Python.

No weights are cached by pointer: modifying an existing Python weight array
must still affect the next call. This change does not introduce persistent
contexts, resident model state, or reuse between Python calls. Those require
explicit ownership and invalidation contracts and separate verification.

## Bounded verification

Run one case at a time under the shared scheduler, using an immutable copy of
an IDENTICAL binding. The three selectable cases are `small` (B1/L5/d32),
`ring` (B2/L9/d128, window 5), and `wide` (B1/L7/d256). Each checks every
input/weight gradient for finite, fully written, repeatable results and records
a second set after changing a weight in place.

```sh
bash tools/mac_slot.sh --timeout 60 --wait-timeout 10 metal \
  pixi run -e test python tools/transformer_readback_check.py \
  --binding /path/to/candidate/_mojolearn_transformer.so \
  --backend metal --case small --out /tmp/candidate-small.npz

pixi run -e test python tools/transformer_readback_check.py \
  --compare /tmp/baseline-small.npz /tmp/candidate-small.npz
```

Use the matching backend and scheduler resource on other machines. CPU uses
`run` with the host binding; `cuda`/`hip` reserve the shared GPU lease. Each comparison checks all 20 complete arrays byte for byte.
These narrow checks supplement the existing public-surface and release gates.

## Measured limits, 2026-09-17

On the local Apple M4, all three candidate cases matched both the same-source
baseline and the existing CPU host oracle: six comparisons, 20 arrays each.
Each GPU diagnostic finished in seconds, below its 60-second cap. Baseline and
candidate compilation finished in 55.1 and 52.4 seconds after one initial
baseline build timed out. The build script skips its bundled gate; the explicit
native execution and byte comparisons above are the validation performed here.

The repeated small case's median readback time was 0.299 ms before and 0.181 ms
after. The first small run went the other way (0.306 to 0.811 ms), so this is
not evidence of a stable end-to-end speedup. The first ring/wide medians were
1.423 to 1.389 ms and 1.531 to 1.164 ms. Entire calls still took hundreds of
milliseconds. Readback is a small part of the remaining cost; removing these
waits does not solve the original Metal slowdown or predict suite runtime.

CUDA and HIP were not executed in this round. No GPU-wide speedup or CPU
speedup is claimed. The host oracle is an existing binary, identified by its
hash, rather than a fresh build of this checkout. See the committed
[measurements and comparison results](evidence/transformer_gradient_readback_2026-09-17.json).
Full local logs, binaries, and NPZ arrays are under
`~/mojolearn-evidence/neural-transfer-batching/`.
