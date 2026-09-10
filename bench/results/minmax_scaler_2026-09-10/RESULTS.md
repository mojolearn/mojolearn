# GPU MinMaxScaler implementation smoke — 2026-09-10

Apple M4, Mojo 1.0.0 (`ed45d567`), Python 3.13.15. This is a focused
implementation check, not a throughput or cross-vendor qualification record.

All three preprocessing binding builds passed their pre-install native GPU
fit/transform/inverse smoke and numeric-mode readback: FAST 0, DETERMINISTIC 2,
IDENTICAL 1. See `binding-{mode}.build.log`. Linker deployment-floor warnings
are preserved in those logs.

`public.smoke.log`: **309 checks passed** across the three modes. Fixtures
cover ordinary data, constants, a single row, 513 rows, strided input,
near-constant ranges, custom bounds, clipping, inverse transforms, copied
outputs, fitted attributes, pickle restoration, and exact signed-zero and
subnormal extrema bits. The independent NumPy arithmetic is a check oracle;
the public scaler computes statistics and transformations on GPU.

The public log also contains one `Context leak detected, CoreAnalytics returned
false` diagnostic. Similar diagnostics appeared in earlier metric checks.
The process exited zero and all assertions passed; origin and memory impact
remain unresolved. No memory-stability claim is made.

`protocol.smoke.log`: sklearn 1.8.0 clone, tags, unfitted-state checks and
parameter validation passed using the isolated sklearn environment. This
protocol check launches no GPU work and does not qualify arbitrary pipelines.

Commands (repository root):

```sh
# Run for fast, deterministic and identical; each builder uses the shared lock.
MOJOLEARN_PYTHON="$PWD/checks/pipeline_python.sh" \
MOJOLEARN_PIPELINE_PYTHON="$PWD/.pixi/envs/default/bin/python" \
MOJOLEARN_NUMERIC_MODE=fast sh bindings/build_preprocessing.sh

PYTHONPATH=python \
MOJOLEARN_PIPELINE_PYTHON="$PWD/.pixi/envs/default/bin/python" \
tools/with_build_lock.sh checks/pipeline_python.sh checks/minmax_scaler_smoke.py

PYTHONPATH=python /tmp/mojolearn-b1-sklearn/bin/python \
checks/minmax_scaler_protocol_smoke.py
```

The repeatable build/public-smoke driver is `pixi run check-minmax-scaler`.
Python compilation, shell syntax and `git diff --check` also passed.
Native sources were unchanged after compilation. `provenance.json` records
source and artifact hashes; compiled binaries are intentionally untracked.

Pending: NVIDIA/AMD execution, cross-vendor intermediate bit comparisons,
large-scale memory and performance measurements, broad adversarial inputs,
and fully resident preprocessing. No remote jobs or mutations were performed.
Wheel build/extension lists were updated, but a full release wheel was not
built in this slice. StandardScaler remains the next planned implementation.
