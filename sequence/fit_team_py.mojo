# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The GPU binding's GARCH and Prophet fit entries (lane neural-pass143): one
series per block (`sequence/fit_team.mojo`), the whole fit in one launch on
NVIDIA and AMD, and on Apple sliced over launches of bounded work (macOS
aborts a long command buffer silently: memory MACOS ABORTS LONG METAL
LAUNCHES). Same arguments and outputs as `garch_py` / `prophet_fit_py`.

Every slice ends each block with the completion witness
(x_linear/witness.mojo). On Apple a slice is one idempotent unit: the
group's private rows (each thread's phase, optimizer state and scratch) and
shared rows are copied aside before it, and a slice whose words do not all
read the nonce (cut by macOS, or its copies dropped) gets them back and runs
again, the same iterations from the same state, up to WITNESS_TRIES times;
then the fit raises. NVIDIA and AMD run the fit in one launch with no check
(witness_end compiles to nothing there).
`-D MOJOLEARN_WITNESS_SABOTAGE=1` makes every Apple GARCH / Prophet fit
raise (the test that the check is wired)."""
from std.python import PythonObject

from sequence.exec_device import DeviceExec
from sequence.fit_team_impl import garch_team_impl, prophet_fit_team_impl
from sequence.pyapi import py_f64s, py_ints


def garch_team_py(mut ex: DeviceExec, addrs: PythonObject, ip: PythonObject) raises -> PythonObject:
    """The Python door of `garch_team_impl` (sequence/fit_team_impl.mojo)."""
    return PythonObject(garch_team_impl(ex, py_ints(addrs), py_ints(ip)))


def prophet_fit_team_py(mut ex: DeviceExec, addrs: PythonObject, ip: PythonObject, fp: PythonObject) raises -> PythonObject:
    """The Python door of `prophet_fit_team_impl` (sequence/fit_team_impl.mojo)."""
    return PythonObject(prophet_fit_team_impl(ex, py_ints(addrs), py_ints(ip), py_f64s(fp)))
