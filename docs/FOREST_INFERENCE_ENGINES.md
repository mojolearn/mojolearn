# RF and ET inference algorithms

Random Forest and Extra Trees retain their distinct training algorithms. Each
trained forest can now be evaluated with either inference algorithm:

| `inference_engine` | Execution and arithmetic |
| --- | --- |
| `auto` (default) | `parallel_groves` in FAST; `sequential` in DETERMINISTIC and IDENTICAL. |
| `sequential` | Existing host traversal. Each row accumulates trees in increasing tree order, then divides by the tree count. |
| `parallel_groves` | Shared GPU traversal for RF and ET. Rows, output components and 32 fixed tree groups run in parallel. Each group accumulates trees g, g+32, ...; a fixed 16/8/4/2/1 reduction combines the groups before division. |

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
`parallel_groves` to reproduce every sequential result. Explicitly selecting
either engine overrides `auto`. The reproducibility tiers keep sequential as
their automatic selection; FAST makes the throughput choice. No training mode
is deprecated.

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
This is an implementation choice inside the GPU engine, not another public
inference algorithm.

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
coverage remain pending. For that reason the reproducibility tiers do not
select it automatically; FAST does, and explicit selection remains available.

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
that staging is declared `FOREST-IO-REUSE-1`, not a literal copy of the reference's
allocation policy. RF/ET share ownership, copying, validation and GPU dispatch.

The [large-data follow-up](../bench/results/forest_io_reuse_2026-09-10/README.md)
qualified the reuse/baseline pairs on NVIDIA IDENTICAL ET single calls (HIGGS,
Year, Covtype), RF/HIGGS throughput, and Metal FAST RF/HIGGS single calls.
Gains were modest: roughly 1–2% in most qualified cells and 5.5% for ET/HIGGS
single calls. No ET throughput gain is certified; those pairs were noisy. The
Metal Year cell and NVIDIA RF single calls also failed stability. Those noisy
cells did not by themselves justify the later FAST-only automatic selection;
the million-row end-to-end qualification does.

RF/HIGGS throughput was a qualified competitor comparison for this one cell
on the H100 (2026-09-10): 21.09 ms/call versus cuML's 10.70 ms/call, averaged
within eight-call blocks, the forests independently trained. The 2026-09-17
L40S measurement with the packed default and cuML FIL on OUR OWN trees is in
the lane status named below; it is a different box and not comparable to the
H100 numbers by ratio.

Native lifecycle checks passed on CUDA IDENTICAL and Metal FAST/IDENTICAL,
including reuse, resize, empty input, changed values, error paths and cleanup.
All four public estimators passed independent graph, repeat, pickle and archive
checks with baseline and reuse paths. The promoted default was checked on CUDA
IDENTICAL and Metal FAST, and 67 host tests passed. HIP and broader large-model
cross-vendor qualification remain open.

### Packed resident layout, the default since 2026-09-17

Shared RF/ET inference changes apply to both Metal FAST and NVIDIA IDENTICAL;
each platform still needs its own performance measurement. The resident
snapshot packs each node's four words (threshold bits or leaf id, local left
child, feature, padding) once at preparation and stores only leaf vectors,
preserving the grove arithmetic; the kernel reads a node with one 16-byte
load (DEVIATION 2963). This layout is the default since
lane/forest-groves-cpu-and-speed; `-D MOJOLEARN_FOREST_SEPARATE_NODES=1`
builds the separate-arrays arm (the old opt-in `MOJOLEARN_FOREST_PACKED_NODES`
is accepted and inert). Both layouts pass the small correctness matrix on
Metal (FAST and IDENTICAL, 2026-09-10) and on NVIDIA IDENTICAL (L40S,
2026-09-17), and the L40S A/B on HIGGS, Covtype and Year at 100 and 500
trees read the same output hashes from both layouts with the packed one
faster on every model (RF/HIGGS 100 trees 45.9 to 24.8 ms per call in
eight-call blocks, 500 trees 224 to 128 ms; the table is in
[LANE_STATUS_lane-forest-groves-cpu-and-speed.md](lanes/LANE_STATUS_lane-forest-groves-cpu-and-speed.md)).
Metal speed under the packed default is not measured. See the
[layout experiment](lanes/GPU_FOREST_INFERENCE_NEXT.md#next-layout-experiment-after-io-measurement)
for the source basis.

## Host inference with no GPU

A forest saved by `save` predicts on a CPU with no GPU through
`mojolearn.HostForest`, the `sequential` algorithm compiled with no accelerator
target (`bindings/build_forest_host.sh`, output
`python/mojolearn/host/_mojolearn_forest_host.so`, IDENTICAL only). The GPU
path and its defaults are unchanged; this is a second door onto a saved file.

```python
import mojolearn

model = mojolearn.HostForest.from_file("forest.npz")   # any of the four estimators
labels = model.predict(X_test)                          # what the GPU class returns
probabilities = model.predict_proba(X_test)             # classifiers only

mojolearn.host_predict("forest.npz", X_test)            # the one-call form
```

`predict` and `predict_proba` return the dtypes the GPU classes return for the
same file. RF probabilities float32, RF regression float32, ET probabilities
and regression float64, labels through `classes_`. A `-parallel-groves-1`
archive predicts through the host grove engine (the next section); until
lane/forest-groves-cpu-and-speed (2026-09-17) it was refused by name. On a box
with no GPU binary set the package imports as a CPU-only install when this
binding is built (DEVIATION 2615 widened), `mojolearn.vendor()` answers
`cpu`, and every GPU estimator raises by name on use.

### The `parallel_groves` engine on the CPU (lane/forest-groves-cpu-and-speed, 2026-09-17)

Both engines are features of a saved forest, and each has a public CPU door.
`HostForest.from_file` reads the engine from the archive's format tag and
`host.inference_engine` answers it: a `mojolearn-randomforest-1` or
`mojolearn-extratrees-1` archive runs the sequential walk above, and a
`-parallel-groves-1` archive runs `core/forest_host_groves.mojo`, the host
restatement of the GPU grove kernels (`core/forest_inference.mojo`): the same
32 fixed tree groups (lane `g` adds trees `g, g+32, ...` in increasing order,
both operands and the result flushed), the same 16/8/4/2/1 fold, the same
RF input flush, equality routing left, `identical_div` and the final flush.
The shipped forest host binding exports it as `forest_host_groves_prepare`,
`forest_host_groves_predict` and `forest_host_groves_release` (a validated
snapshot, prepared on the first prediction and released with the model).
A groves archive saved under a numeric mode other than `identical` is
refused by name, because the host binding computes IDENTICAL bits only. A
GPU class loaded from a groves archive on a CPU-only install predicts
through the rf and trees host families' resident entries over the same
module, as it has since 2026-09-15.

DEVIATION 2960: the host grove walk fans its rows out to host threads the way
DEVIATION 2900 fans the sequential walk out (`MOJOLEARN_CPU_THREADS`, the same
reading); a thread owns whole rows and its own 32-lane scratch, so no output
bit depends on the count. Two negative controls, both default off:
`MOJOLEARN_FOREST_HOST_SABOTAGE` (the forest host gate's existing control)
divides the grove fold by `trees + 1` as it does the sequential vote, and
`MOJOLEARN_FOREST_GROVES_SABOTAGE` (DEVIATION 2961) replaces the 16/8/4/2/1
fold with a left fold over the 32 lanes in lane order, the same additions in
another association and nothing else, so the comparison against the GPU
engine is watched to fail on association alone. `tools/forest_groves_identity.py`
is the comparison: every rf and et lane's fixtures fitted, saved as a groves
archive, reloaded through `host_model` and diffed against the GPU groves
predictions bit for bit, then the HIGGS and Covtype sized models. The numbers
are in [LANE_STATUS_lane-forest-groves-cpu-and-speed.md](lanes/LANE_STATUS_lane-forest-groves-cpu-and-speed.md).

What this promises is only what has been measured. `tools/forest_host_gate.py
record` runs on a GPU box and writes the SHA-256 of that box's predictions for
a saved model and a regenerable fixture; `tools/forest_host_gate.py check`
runs on the CPU box and exits 0 only when the host predictions hash the same.
`.github/workflows/forest-host-gate.yml` runs the check on seven hosted CPUs
against the fixtures under `bench/results/forest_host/`, and the brief
[BRIEF_forest_host_inference_2026-09-13.md](lanes/BRIEF_forest_host_inference_2026-09-13.md)
records which recordings exist and which are still owed. A CPU or a vendor
not in that record is not certified.

### Host threads (lane/infer-speed-trees, 2026-09-17)

The sequential walk fans its rows out to host threads (DEVIATION 2900 in
`core/forest_host_predict.mojo`, DEVIATION 2901 in `core/gbdt_host_predict.mojo`).
`MOJOLEARN_CPU_THREADS` sets the count; absent or `0` is one thread per
physical core, `1` is the calling thread and no pool. Each thread owns a
contiguous row range and every row's arithmetic is the reference loop
unchanged (zero, add every tree's leaf in increasing tree order, divide by
the tree count; for GBDT one float32 add per tree in tree order into a
cursor seeded with `Float32(bias)`), so no output bit depends on the count.
The same functions now serve the GPU classes' `sequential` engine
(`bindings/_mojolearn_rf.mojo`, `bindings/_mojolearn_trees.mojo`), the CPU
training column and `HostForest`, so the columns share the walk by
construction. GBDT host quantization bisects a non-decreasing border list
for the count the linear scan produced; a border list that is not
non-decreasing keeps the linear count. Logloss and CrossEntropy
`predict_proba` columns come from the binding in one pass
(`gbdt_sigmoid_pair`, `forest_host_gbdt_sigmoid_pair`, DEVIATION 2902), the
same `p` and the same one double subtraction per row the Python
comprehension of DEVIATION 2333 computed. The numbers, the identity
evidence and what is owed are in
[LANE_STATUS_lane-infer-speed-trees.md](lanes/LANE_STATUS_lane-infer-speed-trees.md).
