# cluster: a port of cuVS k-means

**Same strategy as `ported/`, different upstream.** This section mirrors
[rapidsai/cuvs](https://github.com/rapidsai/cuvs) at commit `2140532c`, file
for file, so that "did we port this?" is answered by `ls` rather than by
reading.

## The upstream is cuVS, not RAFT, and that was not always true

k-means and brute-force k-NN used to live in RAFT. They do not any more.
RAFT 26.10 (`9aa17e5`) ships `cpp/include/raft/` with `linalg`, `matrix`,
`random`, `sparse`, `stats`, `solver`, `spectral` and `util`, and **no
`cluster/` and no `neighbors/`**; `fusedL2NN` survives there only in a test
shim. The algorithms moved to cuVS and the primitives they call stayed in
RAFT, which is a clean two-layer split and is the split this section
mirrors:

| layer | upstream | where it lands here |
|---|---|---|
| algorithm | cuVS `cpp/src/cluster/`, `cpp/src/distance/` | `ported/` |
| primitive | RAFT `linalg`, `matrix`, and cuBLAS | `mojo_only/` |

A RAFT call is NOT a `ported/` file. RAFT is a general library this tree does
not mirror, so what gets copied from a `raft::linalg::norm` call is the CALL
SITE and its semantics, and the kernel underneath is ours. Every such file
says so in its first paragraph and names the call it replaces.

## Path mirroring

cuVS has two roots and both are kept, with their constant prefixes dropped
the same way `catboost/cuda/` is dropped on the boosting side:

    cpp/src/            ->  ported/
    cpp/include/cuvs/   ->  ported/
    python/cuvs/cuvs/   ->  ported/python/

So `cpp/src/cluster/detail/kmeans.cuh` is `ported/cluster/detail/kmeans.mojo`
and can be diffed against it side by side.

## The one rule still applies here

**COPY. DO NOT IMPROVE.** Same rule, same reason: if this comes out slow we
have to be able to say it is cuVS's design that is slow on this hardware and
not our interpretation of it. Every deviation is numbered in the root
`PORTING.md` (items 14 and up belong to this section) and every unported file
is named in `UNPORTED.tsv`.

## What runs and what does not

`ported/cluster/kmeans.mojo::fit` is wired end to end: norms, tiled GEMM,
fused distance-and-argmin reduction, fixed-point cluster accumulation,
centroid finalize with the empty-cluster rule, centroid shift, and both
convergence tests.

**It refuses cuVS's DEFAULT initialization.** `oversampling_factor = 2.0`
selects scalable k-means|| and that is not ported, so `kmeans_fit_main`
raises instead of quietly running classic k-means++ and reporting the
inertia. Set `oversampling_factor = 0` for classic k-means++, which is
ported, or `init = Random`.

**Nothing here has been launched yet**, and by this tree's rule a kernel is
not ported until it has been enqueued (`PORTING.md 9`). Every kernel in this
section is therefore UNREACHED, not merely untested, and `UNWIRED.md` says
so. The first job is `cluster/mojo_only/kmeans_check.mojo`, and the reach
evidence has to be a SABOTAGE: corrupt the thing that should matter and watch
the answer move.

## Validation

Against scikit-learn, on the same data, never against our own previous
output. `cluster/tools/sklearn_reference.py` writes the fixture and the
expected values.

**Compare inertia, not centroid identity.** cuVS keeps the old centroid for a
cluster that goes empty; scikit-learn relocates it onto the furthest point.
Two implementations that disagree there diverge permanently from the same
seed while both remaining correct.

## The finding worth reading before anything else here

cuVS's float32 distance GEMM runs in **TF32** by default
(`CUBLAS_COMPUTE_32F_FAST_TF32`, `unfused_distance_nn.cuh:196`): 10 mantissa
bits where float32 has 23. Their shipped float32 k-means does not compute
float32 distances on NVIDIA, and no non-NVIDIA implementation can reproduce
its answer bit for bit. See `mojo_only/gemm.mojo`, and the `bitwise-gbdt`
tree, where this is a second incumbent with a device-dependent number system.
