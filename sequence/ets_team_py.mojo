# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The GPU binding's ETS entry over one block per series
(sequence/ets_team.mojo; Apple FAST, `-D MOJOLEARN_ETS_TEAM`): `ets_py`'s
contract (sequence/pyapi.mojo) for ETS(A, A|Ad, N). The fit is one group
launch (sliced on Apple only where the command-buffer bound needs it,
`sequence/fit_team_py.mojo::_run_team`) and one wait; the forecast and the
info rows come back together. Every other model, and every other build,
keeps `ets_py`."""
from std.python import PythonObject

from sequence.exec_device import DeviceExec
from sequence.ets_team_impl import ets_team_impl
from sequence.pyapi import ival, py_f64s, py_ints


def ets_team_applies(ip: PythonObject) raises -> Bool:
    """Whether ets_py's integer parameters name the model the team fits:
    additive errors (ip[3] 0), an additive trend (ip[4] 1), no season
    (ip[7] 0)."""
    if len(ip) != 9 and len(ip) != 10:
        return False
    return ival(ip, 3) == 0 and ival(ip, 4) == 1 and ival(ip, 7) == 0


def ets_team_py(mut ex: DeviceExec, addrs: PythonObject, ip: PythonObject, fp: PythonObject) raises -> PythonObject:
    """The Python door of `ets_team_impl` (sequence/ets_team_impl.mojo)."""
    return PythonObject(ets_team_impl(ex, py_ints(addrs), py_ints(ip), py_f64s(fp)))
