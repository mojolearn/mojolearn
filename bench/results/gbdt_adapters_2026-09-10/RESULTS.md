# Bounded GBDT adapters: local implementation evidence

2026-09-10, Apple M4 / Metal, Mojo 1.0.0 ed45d567. Base commit
`ef9ee060fbbbc273f10dfb44ca06b891b7dcd849`. No remote jobs or mutations.

## Results

- FAST, DETERMINISTIC and IDENTICAL GBDT bindings built successfully. The
  pre-existing broad build gate was explicitly skipped; separate focused GPU
  checks below exercised the new entry points and existing learner. Build logs
  retain compiler and macOS deployment-floor warnings.
- Direct GPU postprocessing: 265 margins per mode, independent Float64 sigmoid
  oracle within the fixture tolerance, exact class codes, finite normalized
  probabilities. Includes signed zero, both signs of the smallest subnormal,
  ordinary margins and extreme finite margins. No cross-device identity claim.
- Public adapter smoke: 18 classifier/regressor × growth-policy × mode cells.
  Each matches a separately fitted legacy learner's model text and raw Float32
  prediction bits. GPU scores, probability shape/dtype, pickle results,
  large integer labels, fitted-mode capture and unknown-evaluation-label failed
  refit are checked. This is 45 small actual fits including reference fits.
  The new probabilities also feed the GPU log-loss metric.
- Optional sklearn 1.8.0: clone/tags/parameter/fitted-state checks and six serial
  StandardScaler → adapter Pipeline/GridSearchCV searches, two depths and two
  folds each, with refit (30 small fits). This verifies bounded interoperability,
  not general sklearn conformance or out-of-sample model quality.
- Python compilation, shell syntax and `git diff --check` pass. Wheel launch
  smoke is wired to the adapter; a full release wheel was not built.

The public smoke emitted three `Context leak detected, CoreAnalytics returned
false` lines; the sklearn smoke emitted two. Both completed with assertions
passing. These unresolved diagnostics do not establish their origin, resource
impact or long-running stability. Direct native checks emitted none.

## Reproduction

All GPU work used `tools/with_build_lock.sh`. The compilation sequence used
`MOJOLEARN_SKIP_BUILD_GATE=1 MOJOLEARN_NUMERIC_MODE=<mode> sh bindings/build_gbdt.sh`
inside that lock, once per mode. The direct fixture runs as:

```sh
MOJOLEARN_PIPELINE_PYTHON="$PWD/.pixi/envs/default/bin/python" \
  tools/with_build_lock.sh checks/pipeline_python.sh \
  checks/gbdt_binary_prediction_smoke.py \
  python/mojolearn/identical/_mojolearn_gbdt.so 1
```

Use the root extension and code 0 for FAST, `deterministic/` and code 2 for
DETERMINISTIC. `native_smoke.py` preserves the original fixture used for these
logs; `checks/gbdt_binary_prediction_smoke.py` is its reusable copy.

```sh
PYTHONPATH=python MOJOLEARN_PIPELINE_PYTHON="$PWD/.pixi/envs/default/bin/python" \
  tools/with_build_lock.sh checks/pipeline_python.sh checks/gbdt_adapters_smoke.py
PYTHONPATH=python tools/with_build_lock.sh \
  /tmp/mojolearn-b1-sklearn/bin/python checks/gbdt_adapters_protocol_smoke.py --gpu
```

The isolated sklearn Python uses NumPy 2.5.2 and Python 3.13.15. Do not apply
the pipeline helper's DYLD override to that environment. For future local
reproduction `pixi run check-gbdt-adapters` builds all modes and runs direct and
public smoke; the optional sklearn check remains separate.

No timings, CUDA/HIP runs, broad learner regression suite, metadata routing,
multiclass adapter qualification or fully resident pipeline are claimed.
