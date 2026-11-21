# SPDX-License-Identifier: Apache-2.0
# PACK part and operator of the x_cluster fit door: the compiled owner of the
# fits of MiniBatchKMeans (fit, partial_fit), BisectingKMeans, MeanShift,
# OPTICS and AffinityPropagation (python/mojolearn/_expansion_cluster.py, with
# the centers' fit_predict and fit_transform), one native call per public
# method, reached from bindings/family_native_door.mojo for every
# `family_cfa_*` operation. Each fit is UNPACK
# (bindings/cluster_fit_public_native_unpack.mojo: the Python method's reads
# and refusals in its order, into one typed stage, and the typed provider
# call), RUN (cluster/cluster_fit_door_run.mojo: the number refusals and the
# request) and the publication here: the attributes the Python set, with the
# same names, dtypes and shapes, each result `array.array` becoming its
# Array's store. Same entries, same operands, same order: the bits do not
# move.
from std.python import Python, PythonObject
from core.cluster_centroid_fit_api import CentroidFitResultSlot
from bindings.cluster_centroid_fit_boundary_door import CentroidPackedResult
from bindings.cluster_public_native_pack import _xc_array
from bindings.cluster_fit_public_native_unpack import (
    _set,
    meanshift_fit_stage, meanshift_fit_call,
)


# ---------------------------------------------------------------- MeanShift
def meanshift_fit(model: PythonObject, binding: PythonObject, source: PythonObject,
                  context: PythonObject) raises:
    """`MeanShift.fit(X)`."""
    var stage = meanshift_fit_stage(model, source, context)
    var result = CentroidFitResultSlot()
    meanshift_fit_call(binding, stage, result)
    var n = stage.n
    var d = stage.d
    var packed = CentroidPackedResult(result, False)
    var kc = result.storage[0].clusters
    _set(model, "cluster_centers_", _xc_array(packed.centers, Python.tuple(kc, d), "<f4", "f", context))
    _set(model, "labels_", _xc_array(packed.labels, Python.tuple(n), "<i4", "i", context))
    _set(model, "bandwidth_", PythonObject(result.storage[0].bandwidth))
    _set(model, "n_iter_", PythonObject(result.storage[0].iterations))
    _set(model, "n_features_in_", PythonObject(d))
