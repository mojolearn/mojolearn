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
checks were removed or reduced. The follow-up [retained Transformer setup](TRANSFORMER_SESSION_REUSE.md)
documents context/workspace ownership and refresh rules; it still reloads and
validates mutable Python weights on every call.

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


## Completed cloud confirmation

The final candidate passed on AMD Instinct MI300X (HIP/gfx942) and NVIDIA
H100 80GB HBM3 (CUDA/sm_90). Each passed the 45-case scan gate, the host-list
constructor/counter gate, both setup groups, and the selected backward case.
All 54 forward/cache arrays and all 20 gradient arrays matched Metal byte for
byte. All 63 refusal messages also matched Metal exactly. No cross-vendor
performance ratio is inferred: these cloud legs ran the candidate only.
They are focused checks, not complete installed-wheel release qualification.

The final AMD VM `46ec4ae5-06b7-4d83-914b-d83aa2ff6d9d` and NVIDIA droplet
`601401995` were deleted and verified absent with HTTP 404. The initial timed-out
legs were likewise deleted. Every nested final check returned zero; the initial
legs' `extra_exit=124` remains a failure even though their orchestration wrapper
returned zero after successfully collecting logs and cleaning up.

[Committed measurements and comparison results](evidence/transformer_weight_setup_2026-09-17.json)
include source commits, binary hashes, raw timing samples, check counts, and
teardown receipts. Full local build logs, NPZ arrays, and cloud logs remain in
`~/mojolearn-evidence/transformer-setup-profile/`. The CPU reference is the
existing host binding identified by its hash, not a fresh rebuild this round.

To reproduce the narrow remote leg, follow the provider-selection rules and
set `MOJOLEARN_STAGE_KEYS=""` **on the local runner invocation**, before renting:

```sh
MOJOLEARN_STAGE_KEYS="" \
MOJOLEARN_GEMM_LEG_EXTRA=tools/transformer_setup_leg.sh \
MOJOLEARN_GEMM_LEG_OUT=/absolute/path/to/evidence \
MOJOLEARN_HOTAISLE_SLOT_WAIT_MINUTES=0 \
bash tools/hotaisle_leg.sh amd --rent --minutes 20 --skip-gates
```

The body needs no corpora. Setting the variable inside the remote body is too
late: the generic runner stages datasets before launching it. This removed
41–120 seconds of unrelated downloads in the confirmation attempts. The
`--skip-gates` option skips the generic card/device gates; the explicit narrow
checks above still run, and this leg does not replace release qualification.
