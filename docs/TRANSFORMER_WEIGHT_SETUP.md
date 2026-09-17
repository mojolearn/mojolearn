# Batching Transformer weight validation

Tiny Transformer calls were paying nine separate GPU scan allocations,
readbacks, and completion waits before computing anything. Every Python call
uploads its mutable weights and constructs `LlamaDeviceWeights`, so this cost
repeats during token-by-token decode. The warmed Apple M4 diagnostic attributed
31–63 ms to weight upload and validation in a B1/L1/d32 call, compared with
4–11 ms for cache, stage, and input setup. These are noisy stage measurements;
phase timing itself adds compute-path waits.

Both weight constructors now validate through `DeviceNonfiniteBatch`:

| Finite-weight validation work | Before | After |
| --- | ---: | ---: |
| Device scratch allocations | 9 | 1 |
| Pinned host scratch allocations | 9 | 1 |
| Result copies | 9 | 1 |
| Explicit completion waits | 9 | 1 |
| Scan kernels | 9 | 9 |

The existing scan kernel, thread/grid geometry, first-index reduction, and
nonfinite bit test are unchanged. Packed scratch holds exactly the sum of the
nine partial-result lengths. All nine scans enqueue before one copy and wait;
the host then examines results in the original tensor order. Only an invalid
weight needs the existing scalar download to distinguish NaN from infinity.
The table describes valid inputs, not refusal-path work or runtime-internal
synchronization.

## Ownership and scope

`LlamaDeviceWeights` retains every source through the completion wait. The
batch retains device scratch and its pinned host mirror until that wait and
the host fold finish. Callers of the reusable helper must keep sources and
batch on the same live context, enqueue every slot, and finish before releasing
sources. A second completed batch uses the same storage but rereads all data;
duplicate, missing, and out-of-range slots are errors.

No persistent pointer cache or trusted-validation flag was added. New Python
calls still observe in-place weight edits and still refuse bad weights before
mutating output/cache state. This is shared Metal/CUDA/HIP implementation code;
the CPU oracle and public Python API are unchanged. No fixtures or numerical
checks were removed or reduced. Persistent model contexts and resident Python
model state remain separate work requiring explicit lifetime and refresh rules.

## Small checks

```sh
bash tools/mac_slot.sh --timeout 50 --wait-timeout 10 metal \
  pixi run check-device-scan-batch
bash tools/mac_slot.sh --timeout 50 --wait-timeout 10 metal \
  pixi run check-transformer-weight-validation
```

The first checks 45 host-oracle cases across empty, singleton, block-edge, and
grid-stride lengths, NaN payloads, both infinities, signed zero and subnormals.
It reuses scratch across changing inputs, requires zero additional allocations,
one copy and one wait per completed batch, and refuses an incomplete batch.
The second exercises the host-list weight constructor and requires nine scans,
one copy/wait and two allocations. It plants a bad device weight, checks the
exact refusal, restores the weight, and requires successful revalidation.

`tools/transformer_setup_check.py` accepts exactly one group: `outputs` or
`refusals`, plus an explicit native binding and expected backend. `outputs`
records all 54 arrays from full and ring caches, carried calls, in-place weight
mutation, and reset. `refusals` checks 63 cases, including all nine names,
first/last indices, NaN/positive/negative infinity, competing bad tensors,
first-index precedence, unchanged state on refusal, and recovery afterward.
Each is a separate bounded job, not a broad identity matrix. Existing
`transformer_readback_check.py` compares complete NPZ arrays bytewise and checks
a selected backward case. The remote body `tools/transformer_setup_leg.sh`
runs these narrow candidate checks with individual 60-second execution limits.
Its cold full binding compile has a separate 120-second cap and two workers;
all timeouts escalate to a kill after five seconds. Set `MOJOLEARN_STAGE_KEYS=""`
when launching this synthetic leg so unrelated corpora are not downloaded.

## Apple result, 2026-09-17

The candidate matched all 54 baseline forward/cache arrays and all 20 backward
arrays byte for byte. The forward/cache arrays also matched the existing CPU
host oracle. All 63 refusal messages matched exactly. The focused scan and
host-list-constructor gates passed.

Forward-call medians in baseline/candidate/candidate/baseline order were
148.72 / 116.57 / 108.56 / 152.45 ms for the small group above. This is about
22–29% lower observed latency, not a stable general benchmark, a large-model
throughput claim, or a full-suite speedup. The warmed candidate weight-setup
profile was 15.56–28.44 ms. Cache construction, weight uploads, context setup,
and many computation-path costs remain.

The first binding build timed out at 60 seconds; a cached build completed in
29.37 seconds. A broader retained-stage diagnostic also hit its compile cap
and was not run. It was replaced with the focused constructor gate, which
compiled in 5.62 seconds and ran in under one second. Timeouts are recorded
as incomplete work, not passes. The bundled binding-build gate is skipped by
its existing script; explicit runtime and bytewise checks supply this evidence.

Initial AMD MI300X and NVIDIA H100 legs both passed the 45-case scan check,
but timed out building the full binding at 60 seconds. Neither initial leg
completed its Transformer integration checks. Both machines were deleted,
with HTTP 404 confirmation. A final confirmation uses two compiler workers
and a 120-second cold-binding build cap, retaining 60-second runtime caps.
