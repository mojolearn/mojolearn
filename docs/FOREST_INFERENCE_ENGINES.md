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


### Device I/O workspace reuse

Public `parallel_groves` predictions now select
`forest_predict_resident_reuse_gpu`, retaining one exact-size device input/output
pair. `bench/speed/forest_inference_ab.py --reuse-io` explicitly compares it with
`forest_predict_resident_into_gpu`, which still allocates per call. This changes
storage lifetime, not the inference algorithm, arithmetic or archive format.

Equal nonempty batch sizes reuse both allocations; a changed size releases the
old pair before allocating its replacement. Empty calls retain the pair without
copying or launching. Retained workspace is `4 * rows * (features + outputs)`
bytes for the most recent nonempty batch, released with the model cache. Inputs
still upload and outputs still download every call. The binding holds the GIL
and synchronizes before returning, so this workspace is not concurrently shared.

Source basis: nvForest `cef3a50d`, `forest_model.hpp:284–308`, borrows device I/O
from its caller. Our NumPy interface requires owned device staging; retaining
that staging is declared `FOREST-IO-REUSE-1`, not a literal copy of upstream's
allocation policy. RF/ET share ownership, copying, validation and GPU dispatch.

The [large-data follow-up](../bench/results/forest_io_reuse_2026-09-10/README.md)
qualified the reuse/baseline pairs on NVIDIA IDENTICAL ET single calls (HIGGS,
Year, Covtype), RF/HIGGS throughput, and Metal FAST RF/HIGGS single calls.
Gains were modest: roughly 1–2% in most qualified cells and 5.5% for ET/HIGGS
single calls. No ET throughput gain is certified; those pairs were noisy. The
Metal Year cell and NVIDIA RF single calls also failed stability. Those noisy
cells do not justify a speed claim or the default selection.

RF/HIGGS throughput is now a qualified competitor comparison for this one cell:
21.09 ms/call versus cuML's 10.70 ms/call, averaged within eight-call blocks.
MojoLearn remains about 1.97 times slower. The forests are independently trained;
this is not same-model inference or evidence of a cost caused by IDENTICAL.

Native lifecycle checks passed on CUDA IDENTICAL and Metal FAST/IDENTICAL,
including reuse, resize, empty input, changed values, error paths and cleanup.
All four public estimators passed independent graph, repeat, pickle and archive
checks with baseline and reuse paths. The promoted default was checked on CUDA
IDENTICAL and Metal FAST, and 67 host tests passed. HIP and broader large-model
cross-vendor qualification remain open.
