# Performance benchmarks

This directory measures the execution mode users actually receive. Historical tables and tuning
diaries are available in Git history; raw accepted runs belong under `bench/results/`.

## Comparison policy

Every method is measured with three interleaved arms on one machine and in one session:

1. mojolearn FAST;
2. mojolearn IDENTICAL;
3. one appropriate external implementation.

Use cuBLAS or PyTorch for dense neural primitives, cuML/cuVS for comparable classical methods,
CatBoost for GBDT, and `mamba-ssm` for Mamba. A CPU library is acceptable only when no equivalent
GPU implementation exists. Do not combine timings from different rentals.

FAST is intended to be the fastest mojolearn mode. A repeatable `IDENTICAL / FAST < 1.0` result is
a performance defect or a measurement defect and must be investigated. Do not weaken IDENTICAL to
make the ratio green.

## Admission rules

A publishable row records:

- commit, device, driver, runtime, dependency versions, and exact fixture;
- one warm-up excluded from statistics;
- at least five interleaved rounds (seven preferred);
- explicit device synchronization around the timed region;
- median, IQR, and paired-ratio range;
- output hashes and an accuracy/agreement result;
- identical inputs and weights for every arm.

Reject a row if outputs are not comparable, weights differ, the mode witness is wrong, the box was
contended, or timing bands cannot distinguish the arms. A tiny smoke fixture proves only that the
program builds and runs.

TF32 and other reduced-precision paths must be named. Strict FP32 compares with strict FP32; a FAST
path that deliberately uses TF32 compares with a correspondingly labelled TF32 arm.

## Drivers

| Family | mojolearn driver | external driver |
|---|---|---|
| GEMM | `gemm_speed_main.mojo` | `../../tools/speed_gemm_arm.py` |
| Transformer/Mamba | `seq_speed_main.mojo` | `../../tools/speed_torch_seq.py` |
| Classical ML | `classical_speed_main.mojo` | `../../tools/speed_cuml_arm.py` |
| Forests/boosting | `forest_speed_arm.py` | `../../tools/speed_gbdt_arm.py` |

The parsers consume the `FSPEED-*` record format emitted by these drivers. Preserve existing field
names when extending it; unknown records should remain visible as notes rather than disappearing.

## NVIDIA execution

Guarded remote legs are launched through `tools/gemm_remote_leg.sh` with a bounded rental:

```bash
tools/gemm_remote_leg.sh nvidia --payload speed --family gemmseq --rent
tools/gemm_remote_leg.sh nvidia --payload speed --family classical --rent
tools/gemm_remote_leg.sh nvidia --payload speed --family forest --rent
```

Run only after clean-wheel and artifact-identity gates pass. Vendor Python arms may need the image's
system Python because RAPIDS and the Mojo-built bindings can support different Python versions.

## Known investigation targets

- Profile RMSNorm and other one-token sequence kernels that underfill large NVIDIA GPUs.
- Avoid materializing transposes for TN GEMM where the vendor API supports transpose flags.
- Use a compatible prebuilt `mamba-ssm` wheel so the comparator is its CUDA scan, not a Python loop.
- Refuse or reroute GEMV shapes that exceed backend grid-dimension limits.

Treat these as hypotheses until a current, admitted run demonstrates them.
