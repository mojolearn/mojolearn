# Shared GPU forest inference: next bounded slice

The active H100 run reported a 26.43 s public RF probability call for 500,000
held-out rows and 100 trees, versus a 7.65 s full-fit warmup. These are preliminary
single-call observations, not stable performance results. `prediction_ms` includes
boundary copying, reconstruction, traversal and output copying; it does not isolate
traversal. GPU training does not imply GPU prediction. No GPU inference speedup
has been measured or implemented by this audit.

## Actual current paths

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

The local upstream inventory contains cuML but no standalone nvForest checkout.
Before claiming incumbent GPU-dispatch parity, acquire/read the exact nvForest
version resolved by cuML26.08, including Treelite import, node layout, traversal,
leaf-vector handling and reduction kernels. This audit did not read those absent
kernels. A standalone flat-array GPU translation preserving our existing output
contract is a separately documented implementation choice, not a vendor primitive
substitution justified by missing source.

## Shared implementation seam and arithmetic constraints

A proposed shared module such as `core/forest_inference.mojo` can take the existing
flat buffers plus dimensions and a small feature-comparison policy. Both bindings
would upload those buffers, launch the same kernel, and return the existing ABI
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

## Bounded implementation and evidence sequence

1. Read pinned nvForest implementation first, then document whether the initial
   flat-array route is a translation of that dispatch or an explicit bounded
   alternative preserving our legacy association.
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

No product code or builds changed for this document. The RF/ET shared inference
opportunity is concrete; its speed and cross-vendor numerical qualification
remain open work.
