# RF and ET inference algorithms

Random Forest and Extra Trees retain their distinct training algorithms. Each
trained forest can now be evaluated with either inference algorithm:

| `inference_engine` | Execution and arithmetic |
| --- | --- |
| `sequential` (default) | Existing host traversal. Each row accumulates trees in increasing tree order, then divides by the tree count. |
| `parallel_groves` (experimental, opt-in) | Shared GPU traversal for RF and ET. Rows, output components and 32 fixed tree groups run in parallel. Each group accumulates trees g, g+32, ...; a fixed 16/8/4/2/1 reduction combines the groups before division. |

```python
from mojolearn import RandomForestClassifier

model = RandomForestClassifier(
    numeric_mode="identical",
    inference_engine="parallel_groves",
)
model.fit(X_train, y_train)
probabilities = model.predict_proba(X_test)

# Evaluate the same fitted forest using the other inference algorithm.
model.inference_engine = "sequential"
reference_probabilities = model.predict_proba(X_test)
```

The option also applies to `RandomForestRegressor`, `ExtraTreesClassifier` and
`ExtraTreesRegressor`. It changes prediction, not fitting, split selection,
bootstrap or ET's random thresholds. Directly changing the inference engine is
supported for a fitted forest; `set_params` retains its established behavior of
clearing fitted state. Invalid engine names are refused at construction and
prediction. A binding without the requested GPU entrypoint raises a rebuild
error rather than silently selecting the sequential algorithm.

## Numerical contract

`parallel_groves` uses a fixed graph across GPU vendors, rather than choosing
the number of groups from hardware occupancy. In IDENTICAL mode it flushes
subnormal arithmetic operands/results and uses the existing portable Float32
division. RF retains its input-feature FTZ rule; ET preserves finite subnormal
comparisons. Equality, including signed zero, routes left in both engines.

Changing addition order can change probabilities/regression predictions and,
near ties, class predictions. Cross-GPU identity of one inference algorithm is
a different contract from matching the other algorithm's bits. Do not expect
`parallel_groves` to reproduce every sequential result. The existing engine
remains available and is still the default. No training mode is deprecated.

The bounded new path accepts finite Float32 features, thresholds and leaves;
non-finite inputs/results and malformed tree graphs are refused. Host code
validates and stages arrays; traversal and prediction arithmetic run on the GPU.
On the first `parallel_groves` prediction, the estimator validates and uploads
an owned model snapshot. Later predictions reuse its device buffers and context;
input upload and output readback still occur each call. The five private model
arrays become immutable host snapshots at preparation, so previously retained
mutable aliases cannot silently change device predictions. Replacing a private
model array causes preparation of a new snapshot. Refitting or `set_params`
releases the old cache. Pickling excludes device handles and rebuilds on demand.
Native registry operations currently retain the Python GIL, serializing calls
through that boundary; concurrent calls and borrowed GPU inputs remain work.

A compile-time `MOJOLEARN_FOREST_VECTOR_GROVES` candidate reuses each tree
traversal across 2–8 output components, following nvForest's vector-leaf loop.
It preserves every per-output addition and uses the same fixed reduction graph.
`MOJOLEARN_FOREST_SCALAR_GROVES` forces the scalar-output reference. Outputs
above eight retain that reference. Selection remains experimental pending
large-data timing; this is not another public inference algorithm.

GPU-engine archives use a separate `*-parallel-groves-1` format and retain the
numeric mode. Current loaders restore the selected engine; older loaders reject
the unfamiliar format. Sequential archives retain their original format and
bytes. Pickle retains the estimator state as before.

## Source and qualification

The [source audit and implementation plan](lanes/GPU_FOREST_INFERENCE_NEXT.md)
pins nvForest v26.08.00 and its actual row/tree/grove GPU dispatch. The shared
Mojo implementation is `core/forest_inference.mojo`; declared deviations cover
flat-array layout, fixed group topology and mode-specific arithmetic. It is not
a claim of identical numerics or full layout/cache parity with nvForest.

Handcrafted IDENTICAL GPU checks have passed on H100 CUDA and Apple M4 Metal,
including tree counts 1/31/32/33, scalar/vector outputs, equality, signed zeros,
subnormals, cancellation and malformed graphs. Public H100 checks cover all
four estimators, an independent fixed-graph oracle, repeat calls, both inference
algorithms and versioned save/load. These are bounded checks: HIP, broader
objectives/weights/feature distributions and large-model cross-vendor output
coverage remain pending. The option stays experimental and opt-in.

Training comparisons are independent of inference timing. The campaign's
[raw evidence](../bench/results/tree_tuning_h100_2026-09-10/README.md) records
large-data training and prediction results separately.
