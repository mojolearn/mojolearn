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

## Continuation results, September 10

The earlier “no model/GPU run” section describes the previous pass. This pass
ran bounded native Metal checks and measurements, all by root under the shared
build lock at nice 19 with two build/host workers. Trees and foreign edits were
excluded. No GPU rental or new opponent timing was made.

- **Runtime numerical checks pass:** default B2/L32/D32 and alternate
  B3/L7/D24/H3/KV1/HD8/FF40, two updates per shape. All 20 parameter gradients,
  losses, actual-gradient AdamW parameters/m/v and four evaluation-state
  invariance checks pass the existing preset tolerances. Negated-gradient and
  wrong-SiLU-derivative controls are detected. Native arithmetic ran on Metal;
  independent FP64 autograd ran on CPU. Full captures and binding provenance:
  `bench/results/byte_runtime_numerical_2026-09-10/`. This does not compare an
  old binary trajectory or qualify cross-vendor bits. 85 host tests plus
  11 subtests also pass. Torch was installed only in the temporary validation
  environment for the FP64 oracle; project dependencies are unchanged.
- **GEMM swizzle experiment:** forced plan 19 changes only tile visitation on
  the existing 128×128 KS16 kernel. Dispatch remains unchanged. All three
  Llama t512 shapes match digests, and all seven stronger GEMM device gates
  pass with all 20 plans. One 9.49-second M4 measurement window records four
  alternating samples per arm/shape. Baseline median throughput is
  0.156–0.162 TFLOP/s; transpose swizzle is 0.159–0.163. These small differences
  do not establish a default-worthy gain. Evidence, raw timing samples and
  drift: `bench/results/gemm_swizzle_2026-09-10/`.
- **GEMM occupancy evidence:** H100 cross-compilation emits PTX with the
  current 128×128 KS16 kernel's 40 KiB shared allocation and 4 KiB local stack
  per thread, including local loads/stores and the required RN-FMA followed
  by FTZ multiplication. PTX virtual registers are not physical allocation;
  physical register count, additional spills and achieved occupancy still
  need ptxas/profiler evidence. The Metal assembly invocation emitted no
  kernel sidecars. No NVIDIA throughput was measured in this pass.
- **Mamba diagnosis:** the new controlled shape-order probe records binary,
  input/output hashes, per-call wall/CPU/resource counters and optional
  continuous GPU telemetry. Two tiny Metal calls pass the retained H100
  output hash. This is a diagnostic smoke test, not an explanation of the
  large-shape H100 latency regimes. See `MAMBA3_REGIME_DIAGNOSTIC_2026-09-10.md`.
- **kNN:** current preflight versus exact-repair/no-preflight produced equal
  full outputs in both execution orders on the small 65537×129×17, k10
  Metal fixture. Phase times drifted substantially between passes, and the
  arm ordering reversed; no speedup or default change is admitted. The next
  candidate computes exponent minima once per input vector. The old 39%
  Apple repair penalty predates accepted whole-chain preflight. Evidence:
  `bench/results/knn_phase_2026-09-10/`.
- **Transformer:** stage-local versus propagated FP64 error diagnostics are
  implemented; reporting tests and a small CPU GQA stage smoke pass. Original
  large GPU fixtures remain numerically unadmitted. No tolerance changed and
  no opponent ratio is qualified. See `TRANSFORMER_ADMISSION_STAGE_FOLLOWUP_2026-09-10.md`.

Remaining: physical H100 register/spill/occupancy inspection and paired swizzle
measurement; old/new default trajectory and cross-vendor runtime-shape checks;
large-shape same-binary Mamba regime reproduction with phase/clock evidence;
kNN metadata reuse with stable target-shape timings; original Transformer GPU
stage attribution. Scalable checkpoints remain separate future work.
