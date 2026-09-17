# Reusing Transformer setup

The Python Transformer previously constructed a device context, rotary table,
KV buffers and scratch on every stateful call. The retained-session path keeps
one context and one compatible workspace per model. It uses the same forward
kernels and preserves the caller's weights and cache as the authoritative data.
This is a setup optimization, not a reduction in numerical verification.

## Ownership and refresh rules

A workspace matches only when batch, sequence length, model dimension, head
counts, head dimension, intermediate dimension, cache capacity, window and lean
attention choice all match. Changing any of these releases the old workspace
before constructing the new one. There is no dictionary of old shapes.

All nine mutable weight arrays are uploaded and checked on every call. Input
and both complete caller cache buffers are copied into the workspace every
call, including after same-address edits and resets. No caller pointer survives
the call. Only immutable rotary data is reused without refreshing. All 29
scratch buffers retain their existing zero-initialization contract; resetting
them batches completion behind one wait. GEMM scratch retains its existing
producer/consumer contract.

Completed workspaces larger than 64 MiB are released after the call. This limit
counts owned device-buffer bytes, including cache, rotary, input, stages and
GEMM scratch. It does not bound driver memory, the context or runtime allocator
caches, and does not prevent a legal large call from using more memory while
running. Per-call weights are released and their frees drained before return.
The context remains available for the next call.

A per-model lock covers cache-counter reads, the native call and cache-counter
updates. Native busy/closed checks protect the private session entry points.
Close is idempotent; buffers are released and their frees drained before the
context is destroyed. Refused calls clear scratch before recovery. Pickle,
shallow copy and deep copy each create fresh runtime ownership and a new lock.
The ordinary Python weights/state serialization remains unchanged.

CPU and older extensions without the private session exports keep their
existing route. NVIDIA's existing stateless prefill fast path and all backward
entry points remain unchanged. `MOJOLEARN_TRANSFORMER_LEGACY_SETUP=1` is a
process-environment diagnostic switch: the next Python call closes any retained
session and uses the old route. It changes setup, not the numerical tier.

## Focused verification

These checks load one explicit native binding and require its reported vendor
and IDENTICAL tier. They do not import unrelated algorithm families.

```sh
bash tools/mac_slot.sh --timeout 50 --wait-timeout 10 metal \
  pixi run -e test python tools/transformer_session_check.py \
  --binding /absolute/path/_mojolearn_transformer.so --backend metal \
  --group reuse --out /tmp/transformer-reuse.npz
```

Run one group per bounded invocation:

| Tool | Group | What it checks |
| --- | --- | --- |
| `transformer_session_check.py` | `reuse` | Five shapes, carried decoding, prefill, full/ring caches, mutable weights/cache, reset; all 75 output/cache arrays against the old entry point |
| same | `refusals` | Bad input, weight, cache and capacity; matching errors, unchanged caller caches, valid recovery |
| same | `lifetime` | Repeated create/call/close, double close and closed-call refusal |
| same | `budget` | Two calls above 64 MiB; full caches compared in memory and no workspace retained |
| `transformer_session_surface_check.py` | `state` | Public full/ring state, mutation/reset, fresh prefill and resumed decode, legacy switch |
| same | `serialization` | Pickle, copy and deepcopy after use |
| same | `threads` | Shared-model state progression and independent models on two threads, including process exit |

`--reverse-order` reverses the old/new call order in the native check. Compare
all NPZ arrays bytewise across vendors, including the existing CPU host oracle
for the public state and serialization groups. The CPU route does not acquire
the new GPU session lock; the threads group tests GPU session ownership.

`tools/transformer_session_leg.sh` runs these groups separately with 60-second
limits. Compilation has a separate 120-second limit and two compiler workers.
Set `MOJOLEARN_STAGE_KEYS=''` for this synthetic leg. A printed result before a
shutdown crash is a failed check: require process exit zero as well as matching
arrays. Broad release identity fixtures remain available and unchanged.

## Measurements and qualification

The 2026-09-17 Apple M4 five-shape paired check measured warm call medians of
101.29 ms with old setup and 83.33 ms with retained setup. Reversing call order
measured 85.22 ms and 70.58 ms. Both runs matched every output/cache byte. These
are about 17% lower observed latency on a shared Mac, not a stable throughput
benchmark or a claim about full-suite duration.

A separate timing-enabled three-call probe attributed 7.79–9.72 ms of warmed
old-call time to cache/stage/input setup, versus 0.63–0.94 ms when retained.
Phase timing adds waits, so these numbers explain the setup reduction rather
than establish an end-to-end speed claim. Weight uploads, validation, zero-fill
launches, computation and readback still run on every call.

Qualification status and cross-vendor evidence are recorded alongside this
change. Failed/time-limited attempts are retained as failures, never passes.

The NVIDIA H100 same-process paired group measured 0.730 ms with old setup and
0.520 ms with retained setup (about 29% lower). MI300X measured 2.054 ms and
0.949 ms (about 54% lower). These are small synthetic-call observations, with
both arms on the same GPU; do not compare vendor milliseconds as library
speedup ratios. All 130 recorded arrays matched Metal across both GPU vendors,
and 26 public state/serialization arrays matched the existing CPU host oracle.
NVIDIA passed all seven groups with clean process exit.

### AMD first-call thread limitation

On the tested MI300X/ROCm 6.4.1 system, making the **first GPU call on a worker
thread** produced correct arrays, then crashed at process exit. Three retained
setup reproductions and three legacy-setup controls each exited 139. The native
backtrace locates SIGSEGV in `libamdhip64.so`, called by `__run_exit_handlers`.
This control establishes that the retained workspace did not introduce the
failure; it does not establish a complete driver-level root cause.

Performing one call with a separate fresh state on the main thread before
starting the workers eliminated the failure in three repetitions. The final
checker, with no extra caller-side lock around retained-session calls, also
passed and exited zero. This is the supported threaded configuration measured
for AMD here. Applications on this runtime should make their initial GPU call
on the main thread before dispatching worker calls. Discard the warm-up state
and allocate the real sequence state separately.

The checker keeps both cases explicit:

```sh
# Normal AMD ownership check; wrap each command in a 60-second timeout.
pixi run python tools/transformer_session_surface_check.py \
  --binding python/mojolearn/identical/_mojolearn_transformer.so --backend hip \
  --group threads --thread-init main --out /tmp/amd-main.npz

# Cold-worker regression probe: known exit-139 failure on the measured runtime.
pixi run python tools/transformer_session_surface_check.py \
  --binding python/mojolearn/identical/_mojolearn_transformer.so --backend hip \
  --group threads --thread-init worker --out /tmp/amd-worker.npz
```

Run the second command with `MOJOLEARN_TRANSFORMER_LEGACY_SETUP=1` for the old
setup control. That control supplies a caller-side lock only for shared-state
updates, which the old route did not serialize; independent models still run
on separate threads. The production retained path must supply its own lock.
`transformer_session_leg.sh` selects `main` for HIP and `worker` for CUDA;
the local checker also defaults to `worker` for Metal;
`MOJOLEARN_SESSION_THREAD_INIT=worker` explicitly requests the failing AMD
regression probe. It is never reclassified as a passing test because it printed
matching arrays. No driver fix or cold-worker AMD support is claimed here.

The machine-readable [qualification record](evidence/transformer_session_2026-09-17.json)
includes hashes, selected groups, timings, failures, controls and rental cleanup.
