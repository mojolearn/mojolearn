# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""What `density.py` computed in Python after a DBSCAN fit (lane apple-fast-py2mojo-cluster, 2026-10-03), exported by both
`_mojolearn_estimators` and `_mojolearn_estimators_host` with one contract:

  dbscan_core_arrays(x_addr, labels_addr, core_addr, [n, d])
      -> [core_sample_indices array('i'), components array('f') (n_core x d,
         row-major), core labels array('i')], ascending training index, from
         the fit's own uint8 core mask (a byte other than 0 or 1 raises).

It acts on the host arrays the fit already returned (its
outputs), so the two columns run the same loops."""
from core.py2mojo_cluster_est_impl import dbscan_core_count, dbscan_core_gather
from std.python import Python, PythonObject
from bindings.native_loaded_module_door import native_loaded_module
from bindings.py_balanced import py_call

# TOMBSTONE: estimators_py2mojo_cluster, pca_rank_finish, pca_fast_pool_on (dead native entries: constant probes nothing reads; pca_rank_finish runs inside the PCA owners) deleted 2026-10-26 by 65f812332; code recoverable at a88745022.
# Restore: git apply experiments/removed/dead_entries_estimators.patch


def _array_addr(a: PythonObject) raises -> Int:
    return Int(py=a.buffer_info()[0])
