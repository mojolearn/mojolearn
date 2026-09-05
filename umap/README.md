# UMAP

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
