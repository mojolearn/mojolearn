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
| Classical ML (generator fixtures: correctness and smoke only) | `classical_speed_main.mojo` | `../../tools/speed_cuml_arm.py` |
| Classical ML on taxi and Istella-S (ENGINEERING_RULES.md section 9, the timing path) | `../../tools/classical_two_datasets.py` (public binding, IDENTICAL) | same file: scikit-learn CPU, torch GPU |
| Forests/boosting | `forest_speed_arm.py` | `../../tools/speed_gbdt_arm.py` |

The parsers consume the `FSPEED-*` record format emitted by these drivers. Preserve existing field
names when extending it; unknown records should remain visible as notes rather than disappearing.

The forest/boosting driver also emits, after every timer, what each arm actually BUILT:
`FSPEED-FIT` (trees, nodes, leaves, max depth, and the library API that answered),
`FSPEED-FIT-NOTE`, and `FSPEED-FIT-VERDICT` (`COMPARABLE`, `NOT-COMPARABLE` or `UNKNOWN`). Holding a
config equal is not the same as fitting comparable models: an arm that silently builds fewer or
shallower trees just looks faster. `UNKNOWN` means the shapes could not be read and is **not** a
pass. `tools/test_fit_equivalence.py` gates the extraction and the verdict on fixtures, with no GPU
and no vendor library.

A dataset that is missing on the box is a **refusal**, not a substitution. The loader used to fall
back to a synthetic fixture and let the run succeed while every line described a different dataset;
it now names the missing key and the manifest it was pinned in and stops. Generated fixtures are
still reachable by name (`--dataset synthclf`, `synth`, `anomaly`).

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

## Checks that cannot fail

Ask of every check here: **what would make this fail?** If the answer is
"nothing", it is decoration, and it will read as a pass forever. This question
found three real defects on 2026-09-12, all of which looked green:

- a verification grep matching a phrase that is split across two source lines,
  so it returned 0 on the broken side too;
- a `git restore` that errored on an unsplit zsh variable, whose verification
  errored the same way and printed nothing, which read as a clean bill of health;
- fixtures built as hand-written dicts, which could not have noticed if the
  declaration they were testing were deleted from the source.

The fix in each case was a control: run the probe against the UNFIXED side and
watch it fail, or sabotage the thing under test and confirm a named failure.
`tools/test_ctd_span.py::test_source_declares_the_flag` is the pattern to copy --
it reads the source file, because a fixture cannot see a deleted declaration.

Two known instances of this shape are still OPEN. Neither is a defect on its own;
both are places where a green result currently proves less than it appears to.

- **`race()`'s own loop is exercised only by a live run with workers.** The
  judgement inside it is reachable by test; the plumbing around it is not, so a
  passing test suite does not mean a race would run. Closing this means a
  worker-free harness for the loop itself.
- **cuML's forest JSON schema is unsettled.** `_shape_cuml` rejects a parse
  unless `nodes >= leaves >= trees >= 1`, which catches impossible counts but
  CANNOT catch a wrong-but-self-consistent schema -- a flat per-tree node list
  gives `leaves == nodes` and passes. `tools/cuml_forest_json_probe.py` settles
  it in about a minute on any box with cuML, needs no dataset, and flags that
  case as SUSPICIOUS by name. Until someone runs it, rf and et cells read
  `verdict=UNKNOWN`. **Leave them UNKNOWN**; never backfill a fitted shape from
  `n_estimators`, which would turn a missing number into a wrong one.
