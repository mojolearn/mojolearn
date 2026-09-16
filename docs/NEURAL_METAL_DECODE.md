# Small neural decode calls on Metal

A healthy command queue does not make our current host-array API efficient.
`TransformerBlock.step` and `Mamba1Block.step` cross the native boundary once
per token. The binding constructs a DeviceContext, uploads weights and carried
state, allocates stages, executes the block, then downloads output and state.
The next token repeats this setup. Python model reuse does not imply resident
GPU weights or GPU state.

The reported 225.71 ms Transformer and 211.67 ms Mamba-1 measurements used
B=1, L=16, d_model=32, and timed 16 separate step calls. They are tiny-call
latencies, not evidence that Apple arithmetic is hundreds of times slower.
They also do not measure how much of a nine-hour test suite can be removed.
The source contains avoidable synchronization at these sizes.

* LlamaDeviceStages allocated and zeroed 29 buffers with 29 separate waits.
* MambaDeviceStages did the same for 19 buffers.
* Each Transformer weight validation called a nonfinite scan with four waits.
* Mamba's list transfers allocated pinned staging and waited twice per upload
  or download. The first block call downloads every weight for validation,
  and the Python binding constructs a fresh weight object each token.

Stage construction now submits all production zero fills before one final
wait. Every field belongs to the stage object until that wait completes.
The zero fills themselves must remain. Partially written scratch arrays once
exposed allocator-dependent NaNs (deviations 2712 and 2713).
`_zeros` and `mamba_zeros` retain synchronous defaults for other callers.
Mamba guard-band builds retain their per-buffer wait because their temporary
sub-buffer views have a shorter lifetime.

The nonfinite scan now writes its partials into an owning host List with one
completion wait. The device kernel, first-index reduction, and refusal ordering
are unchanged. Mamba list uploads and downloads likewise use ordinary host
pointers with one completion wait, retaining the old padded empty-upload path.
The list owner or borrow survives until the copy completes.

Do not remove waits across a host read or buffer destruction. A pinned staging
buffer must remain live until its copy completes. A borrowed caller pointer
must remain valid until the native call returns. Replacing buffers after an
asynchronous fill needs particular care. Keep both the allocation owner and
any view alive through the completion fence.

Do not cache caller weights or state by pointer alone. The public API accepts
mutable arrays; subsequent calls must observe edits. A resident decode API
would need explicit weight/state ownership and refresh semantics. It is a
separate improvement from batching initialization, and the current changes do
not make device state persistent.

## Regression checks

`tools/bench_neural_decode.py` imports the named native GPU binaries directly,
checks their numeric mode and reports vendor and binary SHA-256. It cannot
silently benchmark the CPU fallback. Run baseline and candidate in separate
processes from immutable binary directories under the Metal scheduler.

```
python tools/bench_neural_decode.py --bindings BASELINE --out baseline.npz
python tools/bench_neural_decode.py --bindings CANDIDATE --out candidate.npz
python tools/bench_neural_decode.py --compare baseline.npz candidate.npz
```

The probe covers two model widths, batch sizes one and two, and Transformer
sliding-window decoding. It checks every prefill/step output and both carried
state arrays bytewise on every repetition. Compare the baseline and candidate
artifacts too. Timings are warmed per-token samples, not a shared-machine
wall-clock pass/fail threshold. Repeat the arms in reverse order before making
a latency claim.

`training/checks/neural_stage_init.mojo` requires
`-D MOJOLEARN_STEP_PHASE_TIMERS=1`. It checks the executed Transformer stage
initialization wait count equals one, checks Mamba also uses one production
wait (20 with temporary guard views), and reads every scratch field back to
require positive-zero bits. The pre-fix implementation must fail with 29 waits.
Also compile with `-D MOJOLEARN_MAMBA_POISON=1` to exercise guarded allocations.
These are narrow Metal runtime checks, not full Apple identity columns.

Run the narrow gates with `pixi run check-neural-stage-init`,
`pixi run check-neural-stage-init-poison`, and `pixi run check-device-scan`.
On this shared Mac, wrap each in `bash tools/mac_slot.sh metal ...`.
The transfer gate includes signed zero, subnormal, NaN-payload and infinity
bits at empty, singleton, odd and multi-block lengths. The scan gate plants
nonfinite values at multiple indices and checks the first-index result against
a host oracle across block and grid boundaries.

## Validation status

The change has compiled as native Metal Transformer and Mamba bindings.
Runtime byte comparisons and timings are required before merging it. The local
A/B artifacts and bounded verification script are under
`~/mojolearn-evidence/neural-metal-setup/`. Until those checks run, no measured
latency improvement or full-suite speedup is claimed.
