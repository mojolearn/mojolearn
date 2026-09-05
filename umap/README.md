# UMAP

The host-list estimator composes exact k-NN, fuzzy graph construction,
spectral initialization, curve fitting and 2D/3D layout optimization.
IDENTICAL uses serial layout updates; FAST uses the GPU Jacobi trajectory.
The public slice requires `local_connectivity=1` and enough samples for
spectral initialization. Graph storage is currently dense.

## Identity evidence

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
