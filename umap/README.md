# UMAP

The Python API is `mojolearn.UMAP(...).fit(X)` / `.fit_transform(X)`.
It uses the existing metrics/spectral extension in all three numeric modes.
`embedding_`, `n_features_in_`, and `input_copied_` describe a completed fit.
The supported slice is dense Euclidean input, spectral initialization,
2D/3D output, and `local_connectivity=1`. The published 0.5.0 wheel uses a
dense graph and has no `transform`. Current source enables CSR fit and
`transform`; their qualification status is described below. Supervised
fitting is unsupported.
The installed-wheel gate exercises the public surface in each shipped mode
and compares IDENTICAL layout bits to the named source fixture below.

The host-list estimator composes exact k-NN, fuzzy graph construction,
spectral initialization, curve fitting and 2D/3D layout optimization.
IDENTICAL uses serial layout updates; FAST uses the GPU Jacobi trajectory
at 1,024 samples and above, with the existing serial crossover below that.
The public slice requires `local_connectivity=1` and enough samples for
spectral initialization. Current source stores graph edges in CSR.

## Identity evidence

Apple M4 and NVIDIA RTX 4090 matched all 186 captured cells at source
commit `718495cd`. The
[comparison record](../bench/results/umap/2026-09-05_718495cd-apple/apple-nvidia-comparison.json)
links the retained captures and names the hardware.

The recovered AMD Instinct MI300X capture at the same `718495cd` source also
matches all 186 cells; the
[three-vendor record](../bench/results/e1g/2026-09-05_042552-amd-mamba/cross-device.json)
links the AMD capture, comparison output and verified teardown.

The named `umap.identical.8x1.2d.e4.seed19.v1` fixture exports 186 Float32 bit
patterns across input, rho, sigma, directed memberships, fuzzy weights,
curve parameters, spectral initialization and final layout. It also checks
that the composed pipeline and public estimator agree bit for bit.

Run on each device from matching source and build settings:

```sh
tools/with_build_lock.sh pixi run mojo run \
  -D MOJOLEARN_NUMERIC_IDENTICAL=1 -I . \
  umap/checks/identity_check.mojo > umap.identity.log
python3 tools/umap_identity_compare.py first.identity.log second.identity.log
```

Keep each capture with its source revision/dirty patch, compiler, numeric
mode, device and driver information. The comparator requires every stage
cell and completion marker, rejects non-finite data, and compares uint32
patterns without a tolerance. Repeated local captures establish local
repeatability. A cross-vendor claim needs matching captures from the named
hardware; it does not transfer to other shapes or installed wheels.

The curve check separately compares default and custom fits with numerical
references. These tolerances validate parameter accuracy, not bit identity.
`checks/finite_params_check.mojo` verifies rejection of NaN and infinities
at the parameter, curve and fuzzy-graph surfaces.

Both optimizer entries also reject non-finite learning rates, repulsion,
curve parameters, graph weights and initial coordinates. The public
`fuzzy_graph_from_data` and `fit_transform` entries reject non-finite input
coordinates before uploading data or running k-NN. The direct FAST check
bypasses the small-layout serial fallback so the GPU entry's validation is
covered independently.

Run the refusal checks in the main testing lane:

```sh
tools/with_build_lock.sh pixi run check-umap-finite-params
tools/with_build_lock.sh pixi run check-umap-finite-optimizer
tools/with_build_lock.sh pixi run check-umap-finite-optimizer-identical
```

The latter two tasks each cover 42 cases: NaN, positive infinity and negative
infinity across six arguments in each optimizer and the two public data
entries. `check-umap-stage-identity` captures the named IDENTICAL fixture for
comparison with the command above.

## Broader local API coverage (2026-09-05)

The installed macOS 0.5.0 wheel also passes three 16x3 input profiles varying
neighbors, seed, output dimension, min_dist, spread and graph mixing. All
three modes produce finite noncollapsed output without changing the input;
DETERMINISTIC and IDENTICAL repeat by raw layout bits. The seven API test
groups pass in each mode. See the
[local results](../bench/results/umap/2026-09-05-api-broader/results.json).
These checks do not establish embedding quality or add remote identity
coverage. NVIDIA and DigitalOcean AMD runs remain pending.

## Quality and broader stage capture

`tools/umap_quality_check.py` evaluates two exact-input 64x3 synthetic
fixtures against scikit-learn's independent trustworthiness metric at k=5.
The predefined gate requires a score of at least 0.90 and a margin of at
least 0.20 over a fixed row-permutation control. On the installed macOS
0.5.0 wheel, all six fixture/mode cases passed (0.993–0.998); the scrambled
controls scored 0.425–0.458. See the
[quality record](../bench/results/umap/2026-09-05-quality/summary.json).
This is a small synthetic quality check, not upstream coordinate parity,
a general quality benchmark, or a performance result.

Run in an environment with the wheel, NumPy and scikit-learn installed:

```sh
tools/with_build_lock.sh python tools/umap_quality_check.py \
  --mode identical --device 'actual device name' --output quality.json
```

The second native identity profile,
`umap.identical.16x3.3d.e12.seed7.mix05.v1`, varies neighbors, seed,
min_dist, spread and graph mixing as well as input/output dimensionality.
All 690 cells across eight stages repeated exactly on Apple M4, with
composed/public agreement. Its 48 final layout cells are also pinned in the
installed Python IDENTICAL gate. See the
[stage record](../bench/results/umap/2026-09-05-broader-stages/metadata.json).
NVIDIA and DigitalOcean AMD remain pending for this profile.

```sh
tools/with_build_lock.sh pixi run check-umap-stage-identity-broader > broader.log
python3 tools/umap_identity_compare.py broader.log other-broader.log
```

The comparator accepts only registered profiles and complete finite captures,
and refuses comparisons between the original and broader fixtures. Its
negative controls cover a changed bit, signed zero, missing/duplicate cells,
unknown stages, non-finite values and incomplete runs.

## Source transform and sparse candidates (2026-09-05)

The source Python API now implements `transform(X)` against a frozen fitted
embedding. It retains private training/model copies and requires refitting
after parameter or mode changes. Query batching may change results. This is
not yet available in the published 0.5.0 wheel.

Initial Apple IDENTICAL qualification passed the native transform gate, five
transform API groups, all seven existing fit API groups with unchanged pinned
fit layouts, and two held-out quality fixtures. The quality record retains
exact source hashes and inputs; its source matches this implementation.
These initial checks do not establish cross-vendor transform identity.

Public source `fit_transform` now delegates to the sparse estimator, which
builds CSR directly from neighbors and shares the existing spectral solver.
The dense graph helper remains available for stage certificates and the
independent dense comparison. Apple dense/sparse graph and fit comparisons
passed initial fixtures before this public switch. The integrated public
path still requires fresh all-mode and remote qualification; these source
edits are not part of the published 0.5.0 wheel or the running older-source
NVIDIA campaign.

Graph storage is O(n_samples*n_neighbors), linear when neighbors are bounded.
For 2D/3D output the Lanczos basis has at most 20 vectors. Exact neighbor
computation still compares quadratically many pairs, using bounded query-tile
workspace. Transform additionally retains copies of training data and the
fitted embedding, using O(n_samples*n_features + n_samples*n_components)
storage. These are storage-complexity statements, not measured peak-memory
or performance claims.

See [retained Apple evidence](../bench/results/umap/2026-09-05-sparse-transform/).
Broader modes, remote qualification and installed-artifact checks remain open.

The integrated CSR source API subsequently passed Apple metrics builds, all
fit/transform API groups, and both held-out quality cases in each of the three
modes. The retained embeddings match the prior dense-fit records within each
mode. See [public CSR integration evidence](../bench/results/umap/2026-09-05-public-sparse/).
Remote public integration and installed-wheel qualification remain open.

## Optimizer controls (source, qualification pending)

`learning_rate` (positive, default 1), `repulsion_strength` (nonnegative,
default 1), and `negative_sample_rate` (nonnegative integer, default 5)
now propagate through the Python binding and native parameters to both
serial and GPU fit optimizers. Transform uses one quarter of the learning
rate and the same repulsion and negative-sampling settings. Changing these
settings after fitting requires a refit. The default parameter lists retain
the legacy binding ABI; extended lists require the updated metrics extension.

Existing identity fixtures continue to gate the default trajectory. New
custom-control API checks cover finite output, repeatability and parameter
mutation; these are authored source checks, pending main-thread execution.
They do not establish cuML coordinate identity or qualify untested profiles.
The public slice still excludes supervised targets, alternative metrics and
initialization, arbitrary output dimensions, and local_connectivity != 1.

## Why our embedding scored below cuML's, and DEVIATION 2668

2026-09-11 (lane/knn-finish, H200 pod `zwmta1li2twxx2`, NYC taxi 100,000
rows x 11 numeric columns, n_neighbors 15, n_epochs 200, sampled
trustworthiness and 10-neighbor retention on a 4,000-row stride sample):
our IDENTICAL UMAP scored trustworthiness 0.9062 (retention 0.3736) where
cuML 26.8 scored 0.9657 (0.4804). The cause is the OPTIMIZER'S UPDATE ORDER,
not the graph, the init or any parameter:

- the two inits agree. Ours with the optimizer effectively off scored 0.9097
  and cuML's the same way 0.9104;
- our own serial host optimizer (`-D MOJOLEARN_UMAP_IDENTICAL_HOST_OPTIMIZER=1`),
  on the same graph and the same init, scored 0.9796 (retention 0.4987),
  ABOVE cuML, in 48.6 s against the device optimizer's 2.6 s;
- cuML's own edge-parallel kernel (`force_serial_epochs=False`) drops them
  to 0.9429, and a random init to 0.8027. `umap.pyx:562-570` selects their
  per-vertex SERIAL kernel for a spectral fit, which updates a vertex's
  position after every edge and every negative sample
  (`optimize_batch_kernel.cuh:569-577, 608-616`).

The device optimizer (2026-09-09) summed every move of an epoch from one
snapshot, so a vertex with about twenty eligible edges applied twenty
undamped moves computed from the same point and overshot; lowering the
learning rate, which damps exactly that, recovered a third of the gap
(0.9457 at 0.25, 0.9348 at 0.05).

DEVIATION 2668 is the smallest change in cuML's direction that keeps the
identity contract: within a vertex's fold its own attractive and repulsive
moves land on its running position as the fold visits them, and only the
mirror edge's tail move is still deferred to the epilogue. It is still a
pure function of the epoch snapshot with one writer per vertex, so the bits
stay independent of launch width and vendor; they differ from the snapshot
fold, which is a re-baseline of the UMAP cards. Kernel-matrix row
`umap_device_optimizer_live_row_for`;
`-D MOJOLEARN_UMAP_IDENTICAL_SNAPSHOT_FOLD=1` restores the snapshot fold.
Measured on taxi: trustworthiness 0.9062 to 0.9323, retention 0.3736 to
0.3627. The trial arm that also applies the mirror move live
(`-D MOJOLEARN_UMAP_LIVE_BOTH_ARM=1`) is worse on both (0.8907, 0.2826).
What remains between 0.9323 and the host loop's 0.9796 is the order ACROSS
vertices: the host loop is a Gauss-Seidel sweep in which a vertex sees every
earlier vertex's move of the same epoch, and that order has no parallel form
whose bits are a function of the inputs alone.
