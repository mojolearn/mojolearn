# UMAP

The Python API is `mojolearn.UMAP(...).fit(X)` / `.fit_transform(X)`.
It uses the existing metrics/spectral extension in all three numeric modes.
`embedding_`, `n_features_in_`, and `input_copied_` describe a completed fit.
The supported slice is dense Euclidean input, spectral initialization,
2D/3D output, and `local_connectivity=1`; graph memory is quadratic in the
sample count. `transform` of new data and supervised fitting are not implemented.
The installed-wheel gate exercises the public surface in each shipped mode
and compares IDENTICAL layout bits to the named source fixture below.

The host-list estimator composes exact k-NN, fuzzy graph construction,
spectral initialization, curve fitting and 2D/3D layout optimization.
IDENTICAL uses serial layout updates; FAST uses the GPU Jacobi trajectory.
The public slice requires `local_connectivity=1` and enough samples for
spectral initialization. Graph storage is currently dense.

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
