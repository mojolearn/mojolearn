# SPDX-License-Identifier: Apache-2.0
# Typed pieces of the Session operation door (ops_frontend_pyapi.mojo):
# the door converts arguments and calls these; every session call that
# runs more than once per entry runs here.
from sequence.exec_trait import Exec
from autograd.ops_api import ops_session


def session_split[E: Exec & Movable & Deinitable, KIND: Int](
    sid: Int, hs: List[Int], axis: Int, tail: List[Int], count: Int, fs: List[Float32],
) raises -> List[Int]:
    """Session.split: one view per section (operation 40 with [axis, section,
    then `tail`]: [1, count] for equal sections, [0, sizes...] for sizes)."""
    var s = ops_session[E, KIND](sid)
    var out = List[Int]()
    for k in range(count):  # driver-loop: over requested split views
        var spec = List[Int]()
        spec.append(axis)
        spec.append(k)
        for j in range(len(tail)):  # driver-loop: over operation metadata
            spec.append(tail[j])
        out.append(s[].op(40, hs, spec, fs))
    return out^


def clip_low_unbounded() -> Float32:
    """Session.clip's lower bound when none is given: float32 -inf."""
    return Float32(-1.0) / Float32(0.0)


def clip_high_unbounded() -> Float32:
    """Session.clip's upper bound when none is given: float32 +inf."""
    return Float32(1.0) / Float32(0.0)
