# SPDX-License-Identifier: Apache-2.0
# Session.load_weights, typed: the records of autograd/weight_import.mojo
# (the target handle in word 0) are all checked, then each source tensor is
# cast into its target session tensor in place. The Python door
# (autograd/pyapi.mojo ops_load_weights_py) converts the records and names.
from sequence.exec_trait import Exec
from core.cast_in import CastIn
from autograd.ops_api import ops_session
from autograd.rankn_api import rn_dims
from autograd.weight_import import WI_RECORD, session_load_check, wi_source


def session_load_weights[E: Exec & Movable & Deinitable, KIND: Int](
    sid: Int, r_: List[Int], nm: List[String],
) raises -> Int:
    """Returns the number of tensors written."""
    var n = len(nm)
    if n < 1 or len(r_) != n * WI_RECORD:
        raise Error("load_weights: the import takes " + String(WI_RECORD) + " integers per source tensor")
    var p = ops_session[E, KIND](sid)
    var views = List[CastIn]()
    for r in range(n):  # small-loop(n: one per source weight tensor): checks metadata, no data
        var h = r_[r * WI_RECORD]
        var src = wi_source(r_, r, nm[r])
        views.append(session_load_check(src, r_[r * WI_RECORD + 2], rn_dims(p[].shape_nd(h))))
    for r in range(n):  # small-loop(n: one per source weight tensor): one cast launch each
        p[].load_into(r_[r * WI_RECORD], views[r])
    return n
