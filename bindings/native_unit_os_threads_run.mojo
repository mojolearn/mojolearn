# SPDX-License-Identifier: Apache-2.0
# RUN part of bindings/native_unit_threads_unpack.mojo: typed Mojo only (no
# Python imports): the host's core count.
#
# REMOVED (lane mf-r9b, 2026-11-24): the unit threads' OS-thread start and
# join (`native_unit_threads_start` / `_join`, `NativeUnitEntry`, the stack
# and attribute words) and the unit gate (`native_unit_gate`, `_try`,
# `_leave`, `native_unit_yield`). Their one user, the Python cv units
# (bindings/msel_cv_door.mojo `msel_cv_units`), is deleted: every search
# and cv tool is the search kind's typed RUN, whose units run on the host
# pool (`host_parallelize`). Recoverable at ea2e0c0139.
from std.sys.info import num_physical_cores


def native_unit_cores() -> Int:
    """Physical cores of this host (at least 1)."""
    return max(num_physical_cores(), 1)
