# Owned IDENTICAL LM sessions — September 10, 2026

`LanguageModelTrainer(..., resident=True)` now retains an owned native
DeviceContext and ByteTrainer across Python calls. This retains model weights,
Adam state and workspaces. The default remains reconstruction per call pending
target-model and target-GPU timing qualification. No numerical kernel, fold,
tolerance or attention dispatch changed. Trees untouched.

Upstream LlamaModel owns its layers in
`upstream/transformers/src/transformers/models/llama/modeling_llama.py:348-359`
and reuses them in its forward loop at 402-412. The native object uses Mojo's
[Python type binding and owned-value support](https://mojolang.org/docs/manual/python/mojo-from-python/).
There is no integer-handle registry or retained borrowed host pointer.

## Lifetime and failure behavior

Sessions are lazy: construction and metadata reads do not create a GPU context.
`close()` releases device state, is idempotent, and permits a later call to
resume from retained host state. The native owner destroys buffers before its
context, including when Python drops the object. Concurrent native reuse and
close are refused while the session is busy; the public trainer retains its
existing lock.

`load_state_dict()` validates a replacement before releasing device state.
Checkpoint restoration accepts `resident=True` on both file and bytes APIs;
device ownership is a runtime preference, not a serialized checkpoint field.
The existing JSON size limit is unchanged. A failed native call or failed
Python validation discards the session, so a device update cannot silently
survive a rejected host-state commit. Subsequent calls recreate from the last
committed host state.

Reused native state is checked against supplied parameters, moments, flags,
step, optimizer and shape. Optimizer Float32 values match raw bits, including
signed zero; a native negative control verifies this. Full snapshots, returned gradients, validation
readbacks and per-forward flat-weight unpacking remain. In particular, reuse
currently adds three state readbacks for host/device mirror admission; this is
not yet a minimal-transfer trainer.

## Loader correction

Package selection and per-call mode selection previously initialized the same
extension under two Python names. That aborts when registering an owned Mojo
Python type twice. The mode-set loader now shares an existing canonical module
only when its resolved binary path is exactly the requested file. Vendor and
called-module numeric-mode checks remain; distinct binaries load separately.
The stale “fixed two-block” loader label now says runtime-shaped decoder LM.

## Validation and measurements

Evidence lives in `bench/results/lm_session_2026-09-10/`. The host suite covers
session reuse, close/reopen, restore, checkpoints, missing capabilities,
failure after native writes, Python-side rejection and exact-file loader reuse.
Native gates explicitly exercise resident fused/eager attention and stateless
execution. The lifecycle probe detects a sabotaged host/device mirror and
injects a Python validation failure after a real GPU update; both recover from
retained state. It also checks shape-changing restore and evaluation.
All 112 host tests pass. Final native fused-resident, eager-resident and
fused-stateless gates each pass eight training steps and eight evaluation
checks (24 of each total), with unchanged FP64 oracle tolerances and existing
wrong-derivative/negated-gradient controls. Three 88-array bitwise comparisons
pass: prior versus resident, resident fused versus eager, and resident versus
stateless. Native lifecycle controls pass separately. CUDA/HIP session builds
and execution remain unqualified.

The pilot uses B1/L2048/DM256/H=KV4/HD64/FF768, six layers, V8192:
9,309,184 parameters. Each process interleaves resident and reconstructed
complete training calls, with one warmup pair and alternating timed order.
Every pair compares all parameter, moment, flag, gradient and loss bytes, plus
step counters. Timing includes the public train_step call and its gradient
copies; external state exports/comparisons are outside the timed region.
Process maximum RSS covers both arms; it is not GPU peak memory.

All processes have a 300-second deadline. The initial and intermediate pilot
captures are retained with their native source snapshots. Those versions
introduced a redundant stateless teardown synchronization; they are superseded
by the release pilot, which restores the original single-drain teardown.
They are not current timing evidence. The release pilot below precedes the
final bitwise optimizer-configuration admission tightening; its source is
retained. `pilot-admitted` measures the final compiled implementation.

The release pilot completed in 35.08 seconds on Apple M4. Four timed samples
per arm give medians of 3.60453 seconds reconstructed and 3.21008 seconds
resident: 10.94% less complete-call time (1.123x throughput). Individual samples
span 3.471–3.859 seconds reconstructed and 3.115–3.255 seconds resident; all
four paired resident samples are faster. Both arms share a process; timed
order alternates. All five full-output comparisons, including warmup, pass.
Process maximum RSS for both arms was 4,219,633,664 bytes (3.93 GiB).
This is a short pilot, not a confidence interval or sustained training result.

The final admitted build completed its pilot in 45.96 seconds. Medians were
4.92817 seconds reconstructed and 4.13652 seconds resident (16.06% lower),
but samples varied substantially: 3.471–5.182 seconds reconstructed and
3.289–4.920 seconds resident. One of the four paired resident calls was 2.8%
slower; the other three were faster. All five full-output comparisons passed.
Process maximum RSS was 4,249,681,920 bytes. Retain this drift and do not infer
its cause from these timings. The speed gain remains provisional; neither
the larger median saving nor the earlier steadier window establishes a
sustained target-model gain. No further timing retries were made to select
a cleaner result.

Production context length does not make this the target 125M-scale workload.
No scoped default is promoted from this smaller-model pilot. No opponent was
rerun or added; cached opponent ratios remain unchanged, and every future new
opponent tuple must still be recorded in `bench/OPPONENT_REFERENCE.md`.

## Next work

Reduce full host captures and redundant validation readbacks while preserving
explicit export and failure semantics. Bound the five token-by-vocabulary loss
arrays before attempting full-vocabulary target-model training. Then measure
complete target-model steps under the same five-minute experiment deadline.
GEMM occupancy and attention cost remain the largest qualified component gaps;
Transformer numerical admission remains unresolved. No full-training wall-time
forecast follows from this pilot.
