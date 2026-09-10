Implementation update: an opt-in shared GPU engine is now wired for all four
RF/ET estimators. Public names are `sequential` and `parallel_groves`; see the
[engine contract](../FOREST_INFERENCE_ENGINES.md). The audit below records the
original source analysis and explains the fixed-graph design and remaining work.

# Shared GPU forest inference: next bounded slice

The active H100 run reported a 26.43 s public RF probability call for 500,000
held-out rows and 100 trees, versus a 7.65 s full-fit warmup. These are preliminary
single-call observations, not stable performance results. `prediction_ms` includes
boundary copying, reconstruction, traversal and output copying; it does not isolate
traversal. GPU training does not imply GPU prediction. The new opt-in engine and bounded qualification are described in the linked
engine contract; training and inference measurements remain separate.

## Sequential paths retained alongside the GPU engine

RF `bindings/_mojolearn_rf.mojo:541` rebuilds trees from flat NumPy arrays, copies
row-major X into a Mojo host List, invokes host `RandomForest.predict_proba`, and
copies probabilities back. Regression follows the same pattern at `:605`.
`ensemble/randomforest.mojo:969,1073` iterate rows and trees on the host, add leaf
vectors in increasing tree order, then divide once by tree count. The GIL release
around this work does not make it GPU execution.

ET `bindings/_mojolearn_trees.mojo:289` independently reconstructs the same logical
forest representation and calls `forest_vote` per host row (`:368`).
`extratrees/impl/randomforest/randomforest.mojo:510` sums all tree leaf vectors
in tree order and divides once. Both Python classifiers take first-index argmax
and map through `classes_` (`randomforest.py:635`, `extratrees.py:430`). Regression
returns the averaged scalar. Training differences—RF bootstrap/quantile splits,
ET randomized thresholds—are already represented in their saved trees and do
not require distinct inference algorithms.

Both bindings store Int32 tree offsets, column IDs, and left-child IDs; Float32
thresholds and leaf vectors. Child IDs are local to each tree; right child is
left+1; left=-1 identifies a leaf. For tree t, global node lookup is
`offsets[t] + local_node`; the leaf vector begins at global_node*num_outputs.
Use those arrays directly, avoiding reconstruction into host node Lists.

## What the pinned incumbent actually dispatches

Read source: `upstream/cuml-v26.08.00`, pin `265b9da6`.

* Public classifier `python/cuml/cuml/ensemble/randomforestclassifier.py:396–409`
  gets an nvForest model, requests C-order **device** input, then calls
  `nvforest_model.predict_proba(X)`. Class prediction uses nvForest at `:336–349`.
* `randomforest_common.pyx:675–693` caches the default inference model;
  `:424–456` builds it with `nvforest.load_from_treelite_model(..., device="gpu")`.
  Training also initializes the inference model at `:665–666`. nvForest layout,
  chunk size and alignment are real public controls.
* The lower-level `cpp/src/randomforest/randomforest.cuh:382–436` instead copies
  input to host, walks rows/trees, adds sequentially and divides. This is the
  path our host code follows. It is **not** the current cuML Python production
  GPU inference path. Do not label a GPU translation of this fallback as a
  literal nvForest port or infer identical association to nvForest.

## Resolved nvForest source and actual GPU algorithm

Public source acquisition succeeded on 2026-09-10:
`git ls-remote https://github.com/rapidsai/nvforest.git refs/tags/v26.08.00*`,
then a shallow clone of `v26.08.00` into
`/Users/andrewhendel/CascadeProjects/upstream/nvforest-v26.08.00`.
The annotated tag object is `cb8704b7baa5a3a9b1137e7233af9067c38ff4fc`;
the peeled source commit is
`cef3a50da0f74b0015876b9d6d424c86141898dc`.
This release corresponds to the installed `nvforest-cu12 26.8.0` and
`libnvforest 26.8.0` versions reported by the active leg. cuML's
`dependencies.yaml:990–1006,1086–1102` specifies **26.8.*** package ranges, not
an exact source SHA. The release tag is therefore an explicit source pin matching
the package version, not proof that wheel bytes were built from that SHA;
retain wheel/build metadata separately if exact artifact provenance is required.

The following paths are relative to that pinned nvForest checkout:

* `python/nvforest/nvforest/_forest_inference.py:316–323` returns
  `self.forest.predict` for classifier probabilities. The detailed wrapper
  `detail/forest_inference.pyx:363–375` validates shape and resolves chunk size;
  its implementation at `:222–237` converts to C-order CuPy input and allocates
  device output before entering native prediction. This is the actual GPU path.
* `cpp/include/nvforest/treelite_importer.hpp:151–164,307–323` imports vector
  leaves, average factors and postprocessing. For ordinary averaged RF vector
  leaves the average factor is number of trees. Import selects node-layout
  specialization; the Python default is depth-first (`_forest_inference.py:264`),
  while breadth-first and layered layouts are supported. This is a packed node
  representation with root/child offsets and separate vector-leaf storage,
  not a direct use of our current struct-of-arrays format.
* `detail/integration/treelite.hpp:56–60,104–121` maps inclusive comparisons and
  child orientation. `detail/decision_forest_builder.hpp:162–176` converts an
  inclusive threshold with `nextafter(threshold,+infinity)`;
  `detail/evaluate_tree.hpp:35–65` then uses `<` and child offsets, with missing
  value/default-child handling. Copying `<` onto our unconverted thresholds
  would break equality routing. An initial direct-flat-array implementation
  must retain our `<=`, or port conversion and prove its edge cases.
* `detail/infer.hpp:52–158` specializes vector-leaf and categorical variants.
  `detail/infer/gpu.cuh:103–179` chooses block size and chunking from device
  shared-memory/occupancy limits. It initially considers one row and selects
  32 rows when the residency heuristic permits, reducing chunk size as needed.
  Input/output workspaces can fall back from shared to global memory.
* `detail/infer_kernel/gpu.cuh:97–175` parallelizes **row × tree** tasks inside
  each block: `row=task_index % chunk_size`, `tree=task_index / chunk_size`.
  Each thread accumulates its strided tree subset into a per-row/output
  **grove** workspace. Vector leaves are accumulated component by component.
  `:178–204` reduces groves using warp shuffles with offsets 16,8,4,2,1, then
  applies postprocessing. `detail/postprocessor.hpp:52–74` divides by the
  average factor before bias/other configured transformations.

Consequently nvForest's production summation is not the sequential host left-fold
through trees. Its association depends on task assignment, grove count and
chunk/block choices. A literal copy can be a legitimate FAST candidate, but
cannot be assumed to preserve our current IDENTICAL tree accumulation or
cross-vendor bits. Fixed chunking alone does not recover the original fold.
This is now a source-supported design constraint, not a missing-source blocker.

## Legacy-bit reference option: shared seam and arithmetic constraints

The shared module `core/forest_inference.mojo` takes the existing
flat buffers plus dimensions and a small feature-comparison policy. Both bindings
upload those buffers, launch the same kernel, and return the existing ABI
outputs. No new model format is needed. Keep a reference route until qualification.

First use one independent GPU worker per row/output component, walking trees in
increasing order and maintaining one Float32 accumulator. Rewalking a path for
multiple output components is acceptable for the first bounded binary/regression
slice; a row-owned vector can remove that duplication later. Do not introduce a
tree-parallel reduction, atomics, pairwise tree sum, or per-tree division: each
changes the existing association. Public class-code output must preserve the
first-maximum tie rule. Arbitrary class labels remain host metadata.

There is an important RF/ET seam that must not be erased by deduplication:
RF `ensemble/decisiontree/decisiontree.mojo:152,650` calls `_ftz_feature` before
`<=` comparison (IDENTICAL policy); ET
`extratrees/impl/decisiontree/flatnode.mojo:377–424` compares the stored input
value directly. Both route equality left. Preserve these policies explicitly;
ET subnormal inputs require bit-based ordering or a proven denormal-preserving
comparison if a GPU's ordinary comparison flushes them.

The present host sums and final divisions are ordinary Float32 operations, not
explicit portable arithmetic. Merely imposing portable GPU division or FTZ on
intermediate sums can change existing IDENTICAL predictions. Qualification must
resolve that arithmetic seam explicitly: either preserve host-reference bits for
the supported domain or record a versioned numerical correction with independent
oracles and user-visible scope. Do not silently drop signed-zero, subnormal or
cancellation cases. Training-mode contracts are not evidence for inference bits.

## Ordered-fold reference sequence (candidate A, not the final performance target)

1. The shared GPU flat-forest implementation now follows nvForest's row/tree
   task decomposition, vector-leaf traversal and fixed-grove reduction. The
   fixed 32-group graph is the declared IDENTICAL deviation; it does not promise
   the previous sequential tree-fold bits.
2. RF and ET share graph/finite validation, kernel dispatch, native ownership
   and Python cache invalidation. Both classifier and regressor paths are wired.
3. Handcrafted graph oracles, threshold/subnormal cases, output specializations,
   lifecycle checks and public archive/pickle checks have passed on CUDA and
   Metal in the scopes recorded below. No CPU learner was added.
4. Large real-data public inference measurements now separate transient model
   upload, resident-model staging and borrowed host input/output. Single-call
   timings and batched throughput are recorded separately from fitting and from
   the private resident-kernel experiment.
5. Model buffers/context persist across calls. Immutable host snapshots prevent
   stale device state; replacement/refit invalidates the cache, and GPU handles
   are excluded from pickle. Borrowed host input/output avoids intermediate
   Lists. The default now retains an exact-size device input/output pair after
   qualified large-data measurements; the benchmark `--reuse-io` arm explicitly
   compares reuse with the original per-call allocation path. GPU-array inputs and nvForest-style
   packed node layout remain next work.

The shared engine and new native entrypoints are implemented. Bounded CUDA and
Metal kernel checks and public CUDA RF/ET checks passed; broader cross-vendor
large-model coverage and HIP remain open. See the engine contract for results.

## Updated target: GPU parallel inference with a fixed IDENTICAL topology

The user's target is GPU-parallel inference and cross-GPU identity, not retaining
the old serial tree fold indefinitely. The sequential fold is an oracle/reference
option, not a permanent restriction on the production design. Two concrete
implementations should be distinguished:

| Candidate | Parallel work | Association | Role |
|---|---|---|---|
| A: row-parallel | Rows (and optionally output components); trees visited serially | Existing tree-order fold, subject to explicit device arithmetic seams | Simple reference, migration diagnostic |
| B: fixed-grove GPU | Rows × 32 logical tree groups, each group visits a strided tree subset | Fixed group-local folds plus fixed 16,8,4,2,1 reduction | Implemented opt-in `parallel_groves` engine |

Candidate B follows the production nvForest decomposition read at
`cpp/include/nvforest/detail/infer_kernel/gpu.cuh:110–175` (row/tree task mapping,
per-grove vector-leaf accumulation) and `:178–204` (grove reduction/postprocessing).
Its explicit departure is to fix the logical geometry across architectures,
rather than use `detail/infer/gpu.cuh:103–179` device-dependent chunk/block sizing.
This permits parallel tree traversal while keeping the reduction graph invariant.
It can change prediction bits relative to the previous host traversal. Report
that numerical migration directly, compare quality, and require new cross-device
exact-output gates; do not hide it behind a sequential-bit equivalence claim.

### Implemented GPU engine

* The shared RF/ET buffers retain offsets, column IDs, thresholds, local left
  children, contiguous Float32 node/output leaves and row-major Float32 X.
  Positive output dimensions are supported within explicit Int32 element bounds;
  no arbitrary 32-class cap is imposed.
* A 128-thread block handles four rows for 2–8 outputs, reusing each tree
  traversal across output components. Logical grove `thread_id % 32` visits
  trees g, g+32, ... and keeps one accumulator per output. Scalar regression
  and outputs above eight use four scalar `(row, output)` items per block.
* The shared reduction workspace is 512 bytes per output capacity (up to
  4 KiB). Steps 16/8/4/2/1 retain explicit indices and whole-block barriers,
  independent of hardware warp/wave width. Inactive tails hit all barriers.
  Each output divides once; there are no floating atomics or scheduling-based
  changes in association.
* Both bindings share a validated resident model and synchronous borrowed host
  I/O. Model graph/finite validation occurs at preparation, input finite
  validation before launch, and output finite validation after readback.
* The public names are `sequential` (existing default) and `parallel_groves`
  (opt-in GPU). A test-only ordered-fold GPU kernel is retained as a graph
  reference; it is not an additional public inference-engine setting.

### Qualification for this numerical migration

Before public-default selection, compare B with A and the sequential host route on
identical stored RF/ET forests, checking quality and explaining any changed bits.
Independently construct a CPU test oracle for B's exact group mapping and reduction
graph (test-only arithmetic, not a product CPU backend). Check complete Float32
outputs against B on NVIDIA, AMD and Apple under IDENTICAL; within-device repeats
alone are not cross-GPU qualification. Include 1,31,32,33 and non-power-of-two tree
counts, four-row tails, classification ties, threshold equality, subnormal features,
regression cancellation, and every accepted output specialization. Retain old model
archives as inputs: this is an inference arithmetic change, not a retraining demand.

Then measure large real-data public predict calls, including uploads/downloads,
against both the old route and cuML's actual cached nvForest prediction, reporting
model/quality differences and warm/cold residency separately. No speed estimate is
inferred from the algorithm. The [September 10 campaign](../../bench/results/forest_groves_2026-09-10/README.md) records implemented paths, exact-output checks, stable ET throughput cells and remaining noisy RF comparisons.

### Next layout experiment after I/O measurement

The pinned nvForest `detail/node.hpp:81–175` stores threshold/output-or-index,
child offset and feature metadata together; `detail/evaluate_tree.hpp:44–65`
loads a node and advances through relative child offsets. The vector-leaf
builder (`detail/decision_forest_builder.hpp:135–149`) stores only actual leaf
vectors in its external output array. Our current flat model stores separate
column/threshold/child arrays and allocates output slots for internal nodes too.
These are concrete remaining layout differences, not evidence of a measured
bottleneck by themselves.

A bounded candidate should prepare the packed representation once per immutable
resident snapshot, preserving tree IDs and the fixed grove reduction. Reuse the
existing graph/finite validation and RF/ET comparison policy. Explicitly handle
root-only trees and retain raw `<=` threshold routing; copying upstream `<`
without its threshold conversion is incorrect. If nodes are reordered for
upstream's depth-first layout, rewrite child/root indices and compact leaf
vectors without reordering trees or changing any leaf bits. Keep archives in
their existing representation; device packing is a derived cache.

Measure preparation cost, resident bytes, traversal and complete public calls
separately on large HIGGS/Year/Covtype forests. Compare complete outputs against
the direct-layout kernel, with threshold equality, subnormal RF/ET policy,
root leaves, ragged trees and every output specialization. A default decision
requires the same large NVIDIA IDENTICAL and separate Metal FAST evidence as
other inference candidates. The first candidate now specializes the shared node reader with a packed sibling
layout and compact leaf vectors. It is enabled only in diagnostic builds with
`-D MOJOLEARN_FOREST_PACKED_NODES=1`; ordinary builds retain separate arrays.
Resident preparation packs once, while the transient path stays a direct-layout
reference. The fixed grove kernels and comparison arithmetic are shared.

This bounded candidate retains sibling order rather than implementing upstream's
default depth-first ordering. Each node uses four Int32 words (threshold bits or
leaf ID, local left child, feature ID, padding); field loads avoid the recorded
Metal whole-struct load issue. Only leaves occupy the output buffer. Device model
bytes are `4*(trees+1) + 16*nodes + 4*leaf_count*outputs + 8` (the last eight bytes
are unused ABI operands), versus `4*(trees+1) + 12*nodes + 4*nodes*outputs` before.
This can increase regression storage and is not a demonstrated speedup.

`pixi run check-forest-resident-layouts` explicitly checks both layouts under
FAST and IDENTICAL. The initial Metal matrix passed full-bit comparisons with
separate-array GPU inference for RF/ET, outputs 1/2/3/5/8/9, ragged 33-tree
forests, root leaves, equality/subnormal routing and five-row tails. A poisoned
compact leaf buffer must alter predictions. Lifecycle, validation and both I/O
allocation policies also pass. These are small correctness fixtures, not a
large-model or cross-vendor identity qualification.

Next: run the same checks on NVIDIA, then build a same-process **resident versus
resident** layout A/B retaining identical I/O and grove dispatch, and measure
large HIGGS/Year/Covtype preparation, memory and complete prediction calls. The
existing transient-versus-resident harness now reports native resident layout,
but cannot isolate layout cost because its transient arm also reuploads models.
Do not use that comparison alone to promote this candidate. Depth-first node
reordering remains a separate upstream layout slice.
