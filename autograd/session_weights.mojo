# SPDX-License-Identifier: Apache-2.0
# Session.load_weights, typed: the records of autograd/weight_import.mojo
# (the target handle in word 0) are all checked, then each source tensor is
# cast into its target session tensor in place. The Python door
# (providers/python/autograd_pyapi.mojo ops_load_weights_py) converts the records and names.
from sequence.exec_trait import Exec
from core.cast_in import CastIn
from autograd.ops_api import ops_session
from autograd.rankn_api import rn_dims
from autograd.weight_import import session_load_check, wi_record_starts, wi_source


def session_load_weights[E: Exec & Movable & Deinitable, KIND: Int](
    sid: Int, r_: List[Int], nm: List[String],
) raises -> Int:
    """Returns the number of tensors written."""
    var n = len(nm)
    if n < 1:
        raise Error("load_weights: the import takes at least one source tensor")
    var starts = wi_record_starts(r_, n, "load_weights")
    var p = ops_session[E, KIND](sid)
    var views = List[CastIn]()
    for r in range(n):  # small-loop(n: one per source weight tensor): checks metadata, no data
        var h = r_[starts[r]]
        var src = wi_source(r_, starts[r], nm[r])
        views.append(session_load_check(src, r_[starts[r] + 2], rn_dims(p[].shape_nd(h))))
    for r in range(n):  # small-loop(n: one per source weight tensor): one cast launch each
        p[].load_into(r_[starts[r]], views[r])
    return n
