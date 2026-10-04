# Resident-input catalog GEMM contract, all10 variants

This is a distinct contract from the completed cold66-call screen. Its
context creation, input transfer and buffer allocation dominated23–30ms
cold results; that screen did not select a universal geometry. Old evidence
and failures are retained. No old cold call is repeated by this contract.

Base1f33e9764. Catalog kernels and exact G0..G10 dispatch copied from compiled
9f1a1657c6e054573a495425e223fd73e0c174db, originally sourced at
9ab2d3d3fb770498ef025db08f595a0149792bb7. Core NT and generic SDK source match
that compiled source. No production entry imports this diagnostic binding.
Default-OFF MOJOLEARN_APPLE_GEMM_RESIDENT_PROBE requires FAST Apple GPU,
excluding CPU column. Binding resident_gemm_probe, ABI1.

## Predeclared measured boundary

Each scored shape/arm gets a fresh process. `prepare` creates the process-
lifetime context, allocates A/B/C, uploads inputs, poisons device C, and
synchronizes. Aliased Gram inputs share one device buffer. No GEMM executes
in preparation. Host output allocation/input hashes also precede the timer.

Timer starts immediately before `run_read(arm,C)`. It includes one GEMM call,
full output download, synchronization and Python call overhead. The binding
keeps buffers/context alive when it returns. Then the complete first host
read (`C.sum(dtype=float64)`, same read as cold contract) is timed separately
and included in total. `release` and hashes/quality checks run after the
measured interval. `run_read` consumes a one-shot preparation token before
launch, preventing a second scored launch from those prepared buffers.

No warmup or pipeline precompile: first GEMM pipeline work can remain in the
measured call. This is **resident-input call+completion+first-read**, not
pure kernel time, steady-state throughput or a peak-TFLOPS measurement.
Output transport remains included. No hidden timing around setup.

Exactly66 scored calls, fixed shape-major then G0..G10 order:

| Shape | M,N,K | Layout |
|---|---|---|
| dense | 2048,512,512 | NN |
| square | 1024,1024,1024 | NN |
| tall projection | 32768,64,220 | NN |
| low-width kmeans | 32768,8,220 | NT |
| ragged | 4097,71,221 | NT |
| alias Gram | 1024,1024,220 | NT, A aliases B |

One call per arm/shape; no repetitions, retries of completed arms, opponent
calls, or adaptive shape selection. Changed output words flag review and
remain recorded; never silently discard a score and rerun it.

## Quality lifecycle and commands

Manager first compiles M2 resident_gemm_probe Aempty/B
MOJOLEARN_APPLE_GEMM_RESIDENT_PROBE. Custom builder only compiles. The M3
quality helper directly imports verified B.so; no installed custom binary
or backup required. Exact source, manifest, binary hash and M3 guards apply.

`python3 tools/catalog_resident_quality.py SOURCE_SHA resident-catalog-q-v1 PRIOR_ALL10_QUALITY_REPORT`

PRIOR_ALL10_QUALITY_REPORT must hash to
f2bcde131ffea5ff54cea85920c3a49ec7888b5d4b2824e2077189b50aa973a0. That original
report remains FAIL for n=1. This new resident contract predeclares n>=2;
it explicitly refuses n=1 and checks the refusal. Eleven original matrix
fixtures retain independent FP64 scaled error<=5e-6 and zero degradation
against G0, G1/G5 exact equality, plus exact word equality with all arms in
the reviewed prior report. K0 uses explicit GPU-zero output contract. The
quality phase checks one-shot replay refusal without a second GEMM launch.
New quality is unscored and runs separately from timing children.

Only after manager review of PASS, authorize:

`python3 tools/catalog_resident_timing.py SOURCE_SHA resident-catalog-t-v1 NEW_QUALITY_REPORT NEW_REPORT_SHA256`

The timing helper revalidates per-case/arm quality and provenance, then
creates an exclusive output directory and all66 fresh children. It cannot
reuse the cold quality receipt in place of this new resident ABI gate.
No runtime has been performed while preparing this branch.

All variants remain experiments. No geometry default or caller promotion
follows from this screen. Retain the separate decomposition/PCA adapter
plan: they need strides/split contracts and actual downstream quality.
