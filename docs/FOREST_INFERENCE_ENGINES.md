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
input upload and output readback still occur each call. The native boundary
borrows the contiguous host input and fresh output arrays for that synchronous
call, avoiding intermediate input/output Lists and the extra pinned output
staging buffer. Input finiteness is checked before launch; output finiteness is
checked after readback. Failed public calls never return the output array. The five private model
arrays become immutable host snapshots at preparation, so previously retained
mutable aliases cannot silently change device predictions. Replacing a private
model array causes preparation of a new snapshot. Refitting or `set_params`
releases the old cache. Pickling excludes device handles and rebuilds on demand.
Native registry operations currently retain the Python GIL, serializing calls
through that boundary; concurrent calls and borrowed GPU inputs remain work.

Within `parallel_groves`, vector-leaf traversal reuse is now the default for
2–8 outputs, following nvForest's vector-leaf loop. It preserves every
per-output addition and the same fixed reduction graph. The compile-time
`MOJOLEARN_FOREST_SCALAR_GROVES` switch forces the scalar-output reference;
outputs one and above eight retain that fallback. No enable flag is required.
This is an implementation choice inside the existing opt-in GPU engine, not
another public inference algorithm or a change to the `sequential` default.

The promotion follows exact scalar/vector output checks and stable large
resident-kernel measurements: Apple M4 FAST improved the two-output fixture;
H100 IDENTICAL improved both two- and seven-output fixtures. All 1,440
handcrafted IDENTICAL output-bit records match across CUDA scalar/vector and
Metal vector routes. These are bounded synthetic resident-kernel results;
the [vector-kernel report](lanes/FOREST_VECTOR_GROVES.md) gives their limits.
Separate eight-call throughput blocks on H100 validated the full public ET
path on HIGGS and Year, with identical outputs across transient, resident and
borrowed-buffer paths. RF and several single-call measurements remained noisy;
there is no qualified RF/cuML parity claim. See the
[full campaign evidence](../bench/results/forest_groves_2026-09-10/README.md).

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


### Device I/O reuse candidate (September 10 follow-up)

The shared resident RF/ET owner now supports one retained exact-size device
input/output pair through `forest_predict_resident_reuse_gpu`. Public predictions
still use the existing borrowed-host allocation path. The performance harness
`bench/speed/forest_inference_ab.py --reuse-io` adds both the borrowed baseline and
reuse candidate to its interleaved, exact-output comparison on the same forest.
This is a benchmark option, not another inference algorithm or numeric mode.

Equal nonempty batch sizes reuse both allocations; a changed size releases the
old pair before allocating its replacement. Empty calls retain the pair without
copying or launching. Memory retained is `4 * rows * (features + outputs)` bytes
for the most recent nonempty batch, released with the model. Inputs still upload
and outputs still download on every call. The binding holds the GIL and each call
synchronizes before returning, so this workspace is not concurrently shared.

Source basis: nvForest `cef3a50d`, `forest_model.hpp:284–308`, borrows device I/O
from its caller. Our NumPy interface needs owned device staging; retaining that
staging is declared `FOREST-IO-REUSE-1`, an unmeasured candidate rather than a
literal upstream allocation policy. The fixed reduction and threshold policies
are unchanged. Native FAST/IDENTICAL Metal checks compare both arms for RF/ET,
changed inputs, reuse, resizing, zero rows, and cleanup. These are correctness
fixtures, not speed evidence or CUDA/HIP qualification.

Next measurement: NVIDIA IDENTICAL on HIGGS (1M fit / 500k prediction rows),
Year and Covtype, recording cold calls, interleaved warm single-call timings,
throughput blocks, and retained memory. Compare reuse directly with borrowed
allocation; separate this from cuML's independently trained RF. No speedup or
new default is justified until these large-workload results are stable.

Validation of this follow-up: native lifecycle checks passed on Metal in FAST
and IDENTICAL; public FAST Metal checks passed for all four estimators with
both entrypoints, matching complete prediction hashes, independent graph
oracles, pickle and archive checks. The same-process harness selector was
exercised for both RF and ET. The existing 67 forest host tests passed.
`bindings/build_trees.sh` now checks GPU repeats and explicit CPU refusal;
its stale smoke test attempted to fit the removed CPU learner. Both binding
build gates passed after that correction. No NVIDIA rental or performance
measurement was run for this follow-up.
