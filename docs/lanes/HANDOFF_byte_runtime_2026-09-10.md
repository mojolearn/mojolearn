# Runtime byte-LM shapes and Transformer backward wiring

The initial byte model fixed batch 2, length 32, model width 32 and a 20-tensor
registry to define the first training/checkpoint profile. The underlying Llama
arithmetic accepts runtime dimensions. This pass removes that wrapper limit
without changing the two-block architecture or 256-byte alphabet.

## Implemented

- Host-only `training.byte_lm_config.ByteConfig` owns batch, length, d_model,
  n_heads, n_kv, head_dim and intermediate. Positive/integer/divisibility/RoPE
  constraints and checked indexing products precede allocations.
- `ByteTrainer(..., config=ByteConfig())` uses the runtime configuration for
  every tensor allocation, parameter slice, block copy, token layout,
  forward/backward dimension, gradient capture and evaluation readback.
  Native checkpoint construction accepts the same trailing config; the
  explicit profile must match the capture. The unchanged checkpoint codec
  still needs its external profile/data-schedule sidecar for resume admission.
- Default counts, offsets, profile and old `byte_lm_run(addresses,params)` ABI
  remain compatible. Configured entry points negotiate 7 integer dimensions
  and an explicit v2 profile before reading borrowed addresses. No native
  pointer/context survives a Python call.
- Public `ByteLanguageModelConfig` is immutable. Pass it as `shape=` to
  `SmallByteLanguageModelTrainer`; use `parameter_registry(shape)` before
  preparing actual FP32 parameters. State snapshots/checkpoints preserve
  shape metadata, and loading state atomically replaces its shape too.
- JSON checkpoints retain the existing 2 MiB bound. Guaranteed-oversize saves
  now refuse before copying/hex-encoding state. Larger models can export
  `state_dict()` arrays; a scalable file codec is still future work.
- `TransformerBlock.backward` was already bound and invoked by SambaStack.
  Its IDENTICAL zero-state prefill VJP was not missing. Host checks now cover
  its 21-address/8-scalar boundary, owned/noncontiguous buffers and mixed
  attention→Mamba3→attention reverse cotangent routing. The existing numerical
  hybrid gate now requires backward rather than treating absence as success.

Example (host configuration only; caller supplies actual parameter values):

```python
from mojolearn import ByteLanguageModelConfig, SmallByteLanguageModelTrainer
shape = ByteLanguageModelConfig(batch=1, length=256, d_model=512,
    n_heads=8, n_kv=4, head_dim=64, intermediate=1024)
registry = SmallByteLanguageModelTrainer.parameter_registry(shape)
# shape.n_total == 4_982_784; no device execution is implied.
# trainer = SmallByteLanguageModelTrainer(parameters, shape=shape,
#     data_schedule=actual_corpus_and_schedule_metadata)
```

## Validation completed, no model/GPU run

Evidence: `bench/results/byte_runtime_2026-09-10/`.

- 64 host tests plus 4 subtests pass: old byte-wrapper compatibility tests,
  configured state/ABI/checkpoint/gradient-routing tests, and Transformer/Samba
  wiring tests. Native arithmetic is mocked in these tests.
- `pixi run check-byte-lm-config` passes host-only Mojo registry/profile,
  alternate GQA counts, copy isolation, malformed shapes and overflow gates.
- Final byte-LM shared library compiles for Apple IDENTICAL, two jobs, nice 19,
  shared build lock. Existing compiler/linker warnings are retained in the log.
- Actual compiled binding agrees with Python on 4 profiles (including a
  context 256/model 512 configuration), rejects 7 invalid shapes, and rejects
  null spans on both ABIs before context creation. The freshly built optional
  extension was installed at the previously absent ignored local IDENTICAL
  byte-LM path. Public metadata negotiation passes without model execution.
- Relabeled GEMM probe compiles. Its parser returns the same three timings
  from the historical forced-plan log and a copy with corrected labels.

The first attempted host run used the Pixi interpreter, which has no pytest;
that setup failure is retained. Root used the existing validation environment
for the successful host tests. No dependencies or opponent timings changed.

New runtime shapes are **not numerically or cross-vendor qualified** by these
checks. Default ABI/registry compatibility is checked; a device comparison of
the old/new default trajectory is still owed. No rented GPU was provisioned,
no model execution or M4 throughput measurement was performed, and trees and
foreign `checks/fixed_point.mojo` changes were excluded.

## Next work, in order

1. Numerically compare the old/default trajectory and a small alternate GQA
   shape, with full loss/gradient/update/evaluation-state captures. Then run
   the existing mixed-stack numerical gate and cross-vendor checks.
2. GEMM current-stage occupancy/register/spill inspection and block-swizzle
   experiments, preserving contraction/fold order. The supposed live 64×64
   regression was a rejected experiment with misleading labels; do not repeat
   it blindly. Corrected speed handoff has the evidence and exact scope.
3. Bounded, uncontended M4 achieved-FP32 measurement before model budgets.
4. Diagnose Mamba3 same-binary latency regimes; further kNN selector/launch
   work; resolve Transformer Torch numerical admission. Cached opponent
   measurements stay in `bench/OPPONENT_REFERENCE.md`; rerun only when scope
   or provenance needs it, and append any newly measured opponent there.
5. A scalable checkpoint codec/data pipeline before long multi-million-
   parameter quality runs. No long training run is authorized by these checks.
