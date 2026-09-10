# GPU StandardScaler implementation smoke — 2026-09-10

Apple M4, Mojo 1.0.0 (`ed45d567`), Python 3.13.15. Focused local
implementation evidence; no throughput or cross-vendor qualification.

All three preprocessing extension builds passed their pre-install native GPU
StandardScaler and MinMaxScaler fit/transform/inverse checks and numeric-mode
readback: FAST 0, DETERMINISTIC 2, IDENTICAL 1. Linker deployment-floor
warnings remain in `binding-{mode}.build.log`.

`public.smoke.log`: **48 StandardScaler fit configurations passed**, covering
all four centering/scaling flag combinations and all three modes. Fixtures
include ordinary data, constants, a single row, 513-row data and strided input
with a 513-row constant Float32 0.1 column. Independent NumPy Float64 moments
are approximate numerical oracles, not a CPU product implementation or a
claim of sklearn numerical equivalence. Checks also cover owned Float32
outputs, inverse transforms, pickle output, failed-refit clearing, fitted-mode
and flag capture, and exact both-flags-disabled copying of signed zero,
subnormal and maximum-finite bits.

Three small direct GPU StandardScaler → RandomForestRegressor → GPU R²
compositions passed, one per mode, with R² 1.0 on their simple training
fixture. This is interoperability evidence, not a generalization result or
arbitrary sklearn Pipeline qualification. `public.initial.smoke.log` retains
the earlier successful 36-configuration run before adding the strided fixture.

`minmax.public.smoke.log`: **309 existing MinMaxScaler checks passed** after
extracting the shared Python scaler protocol and rebuilding its extension.
`protocol.smoke.log` and `minmax.protocol.smoke.log`: optional sklearn 1.8.0
clone/tag/parameter smokes passed; these checks launch no GPU work.

The final StandardScaler log contains three `Context leak detected,
CoreAnalytics returned false` diagnostics; the initial run contains two and
the MinMaxScaler run one. All processes exited zero and assertions passed.
These diagnostics also appeared in earlier metric/scaler work. Their origin
and memory impact remain unresolved; no memory-stability claim is made.

Commands from the repository root:

```sh
# Repeat for fast, deterministic, identical; builder acquires shared lock.
MOJOLEARN_PYTHON="$PWD/checks/pipeline_python.sh" \
MOJOLEARN_PIPELINE_PYTHON="$PWD/.pixi/envs/default/bin/python" \
MOJOLEARN_NUMERIC_MODE=fast sh bindings/build_preprocessing.sh

PYTHONPATH=python \
MOJOLEARN_PIPELINE_PYTHON="$PWD/.pixi/envs/default/bin/python" \
tools/with_build_lock.sh checks/pipeline_python.sh checks/standard_scaler_smoke.py
# Same command with checks/minmax_scaler_smoke.py for shared-code regression.

PYTHONPATH=python /tmp/mojolearn-b1-sklearn/bin/python \
checks/standard_scaler_protocol_smoke.py
# Same interpreter with checks/minmax_scaler_protocol_smoke.py.
```

`pixi run check-standard-scaler` rebuilds the three preprocessing artifacts
and runs the public smoke. Python compilation, shell syntax and
`git diff --check` passed. Source/artifact hashes are in `provenance.json`;
compiled artifacts are untracked. Native sources were unchanged after builds.

Pending: NVIDIA/AMD execution, cross-vendor intermediate-bit comparisons,
large-scale memory/performance measurements and fully resident preprocessing.
The wheel launch smoke was updated; no full release wheel was built.
No remote jobs or mutations were performed. See
[the numerical contract](../../../docs/lanes/GPU_STANDARD_SCALER.md) and
[the next GBDT adapter work](../../../docs/lanes/GBDT_SKLEARN_ADAPTER_PLAN.md).
