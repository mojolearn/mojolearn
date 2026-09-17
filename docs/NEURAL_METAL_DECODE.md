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

The nonfinite scan uses one completion wait. Integration retains the newer
main implementation and its pinned host buffer; the measurements below used
an owning host List. The device kernel, first-index reduction, and refusal
ordering are unchanged. Mamba list uploads and downloads likewise use ordinary host
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

## Measured on Apple M4, 2026-09-16

Both native bindings and regression probes compiled. The pre-fix stage probe
failed with 29 waits. Production and guarded candidate probes passed, including
all scratch fields, transfer bits, and wait budgets. `device_scan_check` passed
176 cases. Two A/B rounds compared all 30 named output/state arrays bytewise;
all matched. Arm order was baseline, candidate, candidate, baseline under one
exclusive Metal lease. Each row below gives the two run medians, with three
warmed samples per run, in milliseconds per decode token.

| Case | Baseline medians | Candidate medians |
| --- | --- | --- |
| Transformer B1 L16 D32 | 226.31 / 205.18 | 158.54 / 172.30 |
| Transformer B2 L5 D64 | 243.61 / 268.54 | 226.42 / 193.66 |
| Transformer B1 L5 D32 window 3 | 213.59 / 201.46 | 170.07 / 169.49 |
| Mamba-1 B1 L16 D32 | 168.54 / 172.06 | 60.82 / 60.53 |
| Mamba-1 B2 L5 D64 | 171.86 / 179.05 | 67.42 / 58.11 |

These are small native-call measurements. They do not predict a nine-hour
suite's speedup or establish large-model throughput. Per-call context creation,
weight validation, and many compute-path synchronizations remain. The earlier
211.67 ms Mamba report came from a different run; it is not substituted for this
experiment's same-session baseline.

Full logs, binary hashes, NPZ byte comparisons and the bounded verification
script are under `~/mojolearn-evidence/neural-metal-setup/`. The committed
`tools/neural_runtime_leg.sh` reproduces the narrow GPU checks and exports a
native decode NPZ for comparison with Apple on a rented CUDA or HIP machine.

## AMD follow-up

The same narrow runtime leg passed on a dedicated DigitalOcean MI325X
(gfx942), including production/guarded initialization, the 176-case scan gate,
and the current mutex primitive check. All 30 decode output/state arrays
matched the Apple candidate bytewise. The tiny Mamba-1 B1 case took 1.31 ms per
token in that AMD run. This is targeted coverage, not a full AMD release or a
proof that the intermittent random-forest defect is resolved. Droplet
601175617 was deleted and its subsequent GET returned HTTP 404.

Apple routine work is bypassed while the remaining latency is investigated.
The standalone scan experiment showed variable launch/wait latency and no
consistent win from replacing the reduction with a single-thread scan, so that
experimental kernel was not shipped. No further Apple matrix was launched.

## Integration with the later wait-removal merge

Main advanced during this work. Its qualified pinned-buffer, one-wait scan
implementation was retained, together with its trace-adjacent wait removals.
The merged stage/transfer/scan-budget probe compiled in 5.04 seconds and passed
in 0.51 seconds. The merged Transformer binding built in 48.27 seconds; its
native decode probe, alongside the previously validated unchanged Mamba binding,
finished in 14.57 seconds. All 30 output/state arrays matched the earlier
candidate bytewise. Each integration command had a 60-second execution limit.
The timings in the table above remain measurements of the earlier candidate,
not of the final merged implementation.

## CUDA and CPU, 2026-09-17: the resident decode session

The resident decode API named above now exists for two blocks
(lane/infer-speed-neural, DEVIATIONS 2940 and 2941):
`TransformerBlock.decode_session(state)` and `Mamba1Block.decode_session(state)`
return a session that holds the device context, the weights, the carried state,
the rotary table and the L = 1 stages across decode steps, so a token costs one
input upload, the certified block call and one output download. Ownership is
explicit: the session copies the weights and the state at open, the per-call
`forward` and `step` refuse a state while it is resident, `sync_state` and
`close` write the state back, `load_state` re-uploads it, and a weight refresh
is a new session. The per-call entries are unchanged, and a session's outputs
are bytewise the per-call entries' on the same block and state.

Measured on one RunPod RTX 4090 (IDENTICAL, `sm_89`) with
`tools/bench_neural_decode.py --resident-ab`, a fresh state prefilled to 1024
positions and 64 decode tokens at d_model 1024, five interleaved rounds after a
warmup, every output and state piece bytewise equal to the fresh full forward:

| Case | Per-call ms per token | Session ms per token | Paired ratio |
| --- | --- | --- | --- |
| Transformer B1 (16 heads, 4 kv, head_dim 64, ff 2816) | 9.440 | 0.817 | 11.54 |
| Transformer B8 | 12.528 | 0.987 | 12.68 |
| Mamba-1 B1 (d_inner 2048) | 31.801 | 0.946 | 33.61 |
| Mamba-1 B8 | 26.930 | 1.234 | 21.82 |

These are CUDA numbers. The Apple and AMD columns of the session paths are
owed at the next release record. The CPU host route exports no session
(`TransformerBlockInference.decode_session` refuses by name); its per-call
`step` is the path there. Records, commands and the identity evidence:
`docs/lanes/LANE_STATUS_lane-infer-speed-neural.md`.
