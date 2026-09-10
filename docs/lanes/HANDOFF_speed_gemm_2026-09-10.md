# IDENTICAL FP32 GEMM throughput — corrected 2026-09-10

Measure useful achieved throughput as `2*m*n*k/(milliseconds*1e9)` TFLOP/s
and compare it with the exact device's FP32 non-tensor peak. An internal
baseline/candidate timing ratio is a tuning diagnostic, not an opponent claim.

## Arithmetic boundary

This lane preserves the existing IDENTICAL FP32 contract. No tensor/matrix
units, TF32, FP16/BF16, FP64 accumulation, compensated accumulation, changed
leaf boundaries, or changed fold topology. Do not investigate tensor-core
configurations in this lane. Keep the corrected RN-FMA followed by FTZ
multiplication: removing that rounding seam already failed numerical admission.
The extra instruction means nominal FMA peak is not all available for useful
`2*m*n*k` work. A 30–40 TFLOP/s H100 target is an aspiration, not a demonstrated
or guaranteed ceiling for this contract.

## Current measured source and hardware

The previous draft used an older rejected-candidate log as current performance.
The accepted staging improvement is recorded in
`bench/results/staging_performance_2026-09-10/gemm-stage/summary.json`, with
raw `price-staged-1.log` and `price-staged-2.log`. Source is the accepted
staging implementation retained in `c803148a`; GPU was H100 80GB HBM3,
UUID `GPU-504d7226-23e4-42fe-6ed9-64586e4da2e2`, driver 580.126.09.
These are two runs, each with seven timed rounds after warmup; the intervals
below span the two run medians, not confidence intervals.

| Shape | Actual m,n,k | Median ms range | Achieved TFLOP/s range |
|---|---|---:|---:|
| llama8b.qkv.t512 | 512,4096,4096 | 1.522974–1.523785 | 11.274–11.280 |
| llama8b.mlp_up.t512 | 512,14336,4096 | 6.171459–6.177937 | 9.733–9.743 |
| llama8b.mlp_down.t512 | 512,4096,14336 | 5.355679–5.356788 | 11.225–11.227 |

NVIDIA lists H100 SXM at 67 TFLOP/s FP32 (non-tensor), checked 2026-09-10:
[NVIDIA H100 specifications](https://www.nvidia.com/en-us/data-center/h100/).
The retained telemetry reports 80GB HBM3 and a 700W default/max power limit,
consistent with SXM. Against that published nominal reference, achieved useful
throughput is 14.5–16.8%; this is not a measurement of the rented board's
sustained FP32 peak. The 30–40 target is 44.8–59.7% of that nominal reference.
Use the exact device variant when applying this comparison elsewhere.
Cached cuBLAS prices remain
in `bench/OPPONENT_REFERENCE.md`; no opponent rerun is needed for these source
comparisons. No new opponent measurement was made by the runtime-shape pass.

## Corrected item 1: no demonstrated live dispatch regression

`performance_residual_2026-09-10/gemm-residual/64tile-price.log` compares the
current 128×128 plan with a **forced 64×64 candidate**, not an untuned plan
against current dispatch. `MOJOLEARN_GEMM_BASELINE_PLAN=-2` selects current
dispatch; `MOJOLEARN_GEMM_PLAN` overrides the candidate. The old probe printed
`untuned` and `dispatch` regardless of those overrides. The 0.747–0.841 ratios
are evidence for rejecting the forced candidate, as already documented in
`performance_residual_2026-09-10/gemm-fold/README.md`.

The probe now prints `baseline`/`candidate`, both selectors, both plan names,
actual dimensions, and each arm's achieved TFLOP/s. Its table parser accepts
both historical and corrected labels. Original raw evidence is unchanged.

## Next justified work

1. Inspect register count, spills and shared-memory use on the current staged
   128×128 configuration before choosing an occupancy change.
2. Compare the existing transpose block swizzle with no swizzle on that same
   configuration. Swizzle changes tile visitation order; retain each output's
   exact contraction and fold order. Use both run orders and retained samples.
3. Explore further operand staging only after examining generated loads.
   Guarded float4 loads already exist for the K-contiguous mapping; the
   outer-contiguous path uses scalar accesses across neighboring lanes.
4. K-step changes remain possible scheduling experiments, but current dense
   dispatch already uses KS=16. The rejected 64×64 candidate used KS=32.
   Avoid repeating rejected configurations without a new reason.
5. `fold=16` is local stack capacity, not arithmetic fold arity. Capacity or
   storage changes must retain sufficient depth and the pinned push/drain
   tree. Capacity specialization was already measured and rejected; changing
   the tree is outside this lane.

## Apple measurement completed in the continuation

The continuation measured the same actual Llama dimensions on the M4. See
`bench/results/gemm_swizzle_2026-09-10/`: baseline medians 0.156–0.162 TFLOP/s,
forced transpose-swizzle medians 0.159–0.163, all bits checks green. No default
changed. For subsequent measurements, use the same harness and report
achieved TFLOP/s and start/end drift separately. Root must use one
bounded thermal window, two build/host workers, `nice 19`, and the shared build
lock. Check that the device is otherwise idle; a concurrent lane invalidates
an uncontended performance claim. This measurement informs model-run budgets.

## Gates and ownership

`BITS MATCH` is required for every configuration and shape; the probe's check
is full-output FNV-1a64 plus poison detection. Retain the stronger raw-output
and adversarial correctness gates before adopting a default. Three-vendor
identity fixtures must qualify a cross-vendor default claim; no new physical
vendor qualification follows from a host-only compile.

Root runs all builds/tests/measurements. Lanes may author source and record
exact commands. GPU rentals require a one-hour lease; no rental is active for
this pass. Trees belong to another lane.

## H100 compile-only resource inspection

The continuation cross-compiled for x86_64 Linux and sm_90 (emitted PTX target
sm_90a). The current 128×128 KS16 specialization has 40960 shared bytes and a
4096-byte per-thread local depot. The retained PTX includes actual local loads
and stores plus `fma.rn.f32`/`mul.rn.ftz.f32`. These are static intermediate-code
facts, not SASS register/spill counts or achieved occupancy. Physical NVIDIA
measurement is still owed. The transpose plan shares that same specialization.

## Resource-capture continuation

`tools/gemm_cuda_resources.py` now captures offline assembler statistics,
cuobjdump resource usage and SASS, tool versions/hashes, exact commands and
artifact hashes. It uses the PTX's target rather than substituting another
architecture, refuses an existing output directory and preserves incomplete
failure evidence. Run on a compatible CUDA toolkit host (no GPU required):

```sh
python3 tools/gemm_cuda_resources.py \
  bench/results/gemm_swizzle_2026-09-10/h100-current-128.ptx.gz \
  /tmp/gemm-h100-resources
```

Tool flags follow NVIDIA's [binary utilities documentation](https://docs.nvidia.com/cuda/cuda-binary-utilities/)
and [assembler options](https://docs.nvidia.com/cuda/archive/13.0.0/cuda-compiler-driver-nvcc/contents.html).
Offline assembly is toolkit-specific; equivalence to the runtime driver's JIT
is not established. Resource counts do not measure achieved occupancy.

Validation: retained PTX target/hash/entry checks; malformed PTX, missing tool,
and output overwrite rejection; mocked successful subprocess orchestration
and timeout partial-log preservation; Python compilation. Actual CUDA assembly
has **not run**. This Mac has no ptxas, and the RunPod REST pod-list request
returned HTTP 403 during this continuation. No rental was created. H100
register/spill results, profiler evidence and large paired timings remain owed.
No kernel arithmetic, dispatch gate, or opponent measurement changed.


## H100 resource inspection and rejected scalar-load trial

The follow-up obtained a dedicated H100 80GB HBM3, driver 580.126.09, and
assembled the retained PTX with ptxas 12.6.85. CUDA 12.4 first refused PTX
8.5; that failure is retained too. The selected 128x128 KS16 specialization
uses **255 registers/thread, 4144 bytes stack/thread, 44 bytes spill stores
and 44 bytes spill loads**, and 40960 bytes static shared memory. The CUDA
driver reports one active 256-thread block per SM for this offline cubin:
**12.5% theoretical thread occupancy**. This is a static occupancy query,
not achieved occupancy or proof that the runtime JIT emits the same cubin.
The ptxas spill byte counts describe static compiler output, not dynamic
traffic. cuobjdump's 41984 shared resource figure includes more than the
40960-byte user allocation reported by ptxas and the driver; raw reports
are preserved without conflating those figures.

A forced experiment routes full windows through the existing ascending
scalar shared-read loop, preserving the arithmetic and fold. Two seven-round
captures per executable in baseline/candidate/candidate/baseline order cover
the actual three large Llama shapes. Every full-output digest agrees across
executables. QKV is effectively flat, up improves about 0.6%, down is slightly
slower. Its offline assembly still uses 255 registers and one block/SM,
with 48-byte spill counts; the intended occupancy improvement did not happen.
This is insufficient benefit to retain: **candidate rejected, default
unchanged**. The patch is archived only; the probe now prints successful
output digests so separate executable captures can be checked explicitly.

Evidence: `bench/results/gemm_resources_2026-09-10/`. No opponent was rerun.
The next occupancy experiment must reduce live storage enough to change the
resident-block limit while preserving the numerical fold. More work on fold
storage or operand lifetimes needs new evidence, not another blind replay of
previously rejected tile or stack-capacity choices.
