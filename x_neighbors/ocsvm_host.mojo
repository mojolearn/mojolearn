# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The host column of x_neighbors/ocsvm_dev.mojo: OneClassSVM's SMO as the
one sequential item `ocsvm_smo_item`. CPU-only installs and the
verification digests only. The device's grid scans pick the item's index
(a total order) and its rho is the same blocked fold, so the two agree."""
from x_neighbors.items import FP, IP, ocsvm_smo_item
from x_neighbors.host_ops import X_NEIGHBORS_HOST_SABOTAGE


def op_ocsvm(q: Int, cv: Int, alpha: Int, info: Int, iters: Int, n: Int, eps: Float32, max_iter: Int) raises:
    var s_g = List[Float32](length=n if n > 0 else 1, fill=Float32(0))
    ocsvm_smo_item(
        0, FP(unsafe_from_address=q), FP(unsafe_from_address=cv), FP(unsafe_from_address=alpha),
        FP(unsafe_from_address=Int(s_g.unsafe_ptr())), FP(unsafe_from_address=info),
        IP(unsafe_from_address=iters), n, eps, max_iter,
    )
    comptime if X_NEIGHBORS_HOST_SABOTAGE:
        if n > 0:
            var pa = FP(unsafe_from_address=alpha)
            pa.unsafe_store(0, pa.unsafe_load(0) + Float32(1e-3))
    _ = s_g^
