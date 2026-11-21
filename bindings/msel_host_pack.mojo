# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The model-selection helpers of the core host binding
(`bindings/_mojolearn_core_host.mojo`): the host column of
`bindings/msel_device.mojo` (lane cpu2-l4-modelsel, 2026-10-04), same names,
signatures and refusals, over `core/msel_host.mojo`."""

from std.python import Python, PythonObject


# lane calltable-msel: the store slot and the scatter, column, split-table
# and group-fold bodies are the runs the model-selection helper table
# publishes (core/msel_helpers_host.mojo); these exports unpack and call them.
from core.msel_helpers_host import (
    MSEL_HOST_STORE, msel_scatter_rows_host_run, msel_proba_column_host_run, msel_split_table_host_run,
    msel_group_fold_perm_host_run,
)


# `msel_put` and `msel_take_rows` deleted (arena unit E, 2026-11-21): no
# caller; the resident owner calls the helper table as typed RUN.
def msel_free_binding(id: PythonObject) raises -> PythonObject:
    MSEL_HOST_STORE.get_or_create_ptr()[].free(Int(py=id))
    return PythonObject(0)


def msel_live_binding() raises -> PythonObject:
    return PythonObject(MSEL_HOST_STORE.get_or_create_ptr()[].live)


