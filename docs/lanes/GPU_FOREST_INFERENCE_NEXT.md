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

1. Use the resolved nvForest source above for packed traversal, shared-input
   caching and vector-leaf layout decisions. Explicitly choose an IDENTICAL
   arithmetic deviation: retain the sequential tree fold rather than
   copying nvForest's grove/shuffle reduction. The first bounded GPU slice can
   parallelize rows while preserving that fold. A later per-tree-output kernel
   plus ordered fold is possible but adds rows×trees×outputs scratch; do not
   silently introduce that memory cost. FAST may separately port the production
   grove reduction after its own quality and timing gates. The updated target
   below also permits fixed-grove parallelism in IDENTICAL with an explicit
   numerical migration and cross-device gates.
2. Add one shared GPU flat-forest kernel plus a synchronous upload/download helper,
   with RF/ET thin wrappers. Start Float32 binary probabilities and scalar
   regression, finite input, valid acyclic flat forests. Validate offsets, feature
   and child bounds and output sizes before GPU access, including loaded models.
3. Compare full Float32 output bits against the current host route for both RF/ET,
   all numeric modes, single/multiple trees, equality thresholds, local-child
   offsets, one-node trees, non-power-of-two counts, fractional/regression leaves,
   cancellation, signed zero and subnormals. Include independent small hand-built
   forests so two routes cannot share a hidden tree-layout error. No CPU learner
   is introduced by retaining the existing prediction oracle.
4. Run large real-data NVIDIA IDENTICAL public inference A/B separately from fit
   timing: identical stored forest and HIGGS held-out rows, warmups, balanced order,
   raw latency, quality and complete probability hashes. Compare cuML's actual
   nvForest public prediction with clear model/quality differences, not its host
   fallback. Mac FAST tree inference is a separate requested performance leg.
5. Only after that baseline, consider persistent GPU model buffers and chunked X
   uploads. Existing Python arrays can be mutated and models can be refitted,
   unpickled or loaded, so caching needs explicit invalidation/version ownership.
   The first implementation may upload per call and must include that cost in
   public timing. Do not hide setup cost by timing a private resident-buffer API.

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

### Implemented first GPU engine

* The shared RF/ET buffers retain offsets, column IDs, thresholds, local left
  children, contiguous Float32 node/output leaves and row-major Float32 X.
  Positive output dimensions are supported within explicit Int32 element bounds;
  this scalar-output implementation introduces no arbitrary 32-class cap.
* A 128-thread block handles four scalar `(row, output)` items. Logical grove
  `thread_id % 32` visits trees g, g+32, ... for its item. `thread_id // 32`
  selects the item, and each thread keeps one Float32 accumulator. This
  parallelizes rows, components and tree groups, but repeats traversal for each
  component; nvForest's vector-leaf traversal reuse remains a next optimization.
* The shared reduction workspace is 128 Float32 values (512 bytes). Steps
  16/8/4/2/1 use whole-block barriers and explicit indices, independent of
  hardware warp/wave width. Inactive tail items still execute all barriers.
  Grove zero divides once by the number of trees. No floating atomics or
  scheduling-dependent association are used.
* Both bindings upload and download through the shared helper. Host graph and
  finite-value validation prevent cyclic/out-of-bounds traversal before launch.
  Persistent device models and borrowed input buffers are not implemented yet.
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
inferred from the algorithm. Product code is still unchanged by this document;
root must coordinate shared kernel and binding ownership before implementation.
