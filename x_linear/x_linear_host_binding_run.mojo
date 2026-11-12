# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""RUN half of `bindings/_mojolearn_x_linear_host.mojo` (wave 8 run split,
2026-11-12): the native publication of the CPU binding's linear entry tables,
called once from the module builder. Typed Mojo only, no Python import
(tools/python_free_dirs.sh). Moved word for word."""
from core.linear_entry_table import LinearEntryKey, linear_table_publish, LINEAR_TABLE_X_LINEAR
from core.linear_fit_types import RIDGE_FIT_ABI
from core.linear_entry_table_ext import xl_ops_publish
from x_linear.ridge_descriptor_host import ridge_fit_host_v1, RIDGE_FIT_HOST_API
from x_linear.ops_entry_host import x_linear_ops_host_v1, XLINEAR_OPS_HOST_API
from checks.numerics import GLOBAL_NUMERIC_MODE


def _publish_linear_entry_table() raises:
    # Native publication only; the existing capsule ABI remains for clients
    # not yet converted. The original descriptor owner has image lifetime.
    var api = RIDGE_FIT_HOST_API.get_or_create_ptr()
    var key = LinearEntryKey(GLOBAL_NUMERIC_MODE, 0)
    linear_table_publish[LINEAR_TABLE_X_LINEAR](key,
        Int(api[].entries.unsafe_ptr()), [String(RIDGE_FIT_ABI)], [1], [ridge_fit_host_v1])
    # The x_linear operations table (core/linear_entry_table_ext.mojo): the
    # native owner of the linear expansion estimators calls these entries.
    var ops = XLINEAR_OPS_HOST_API.get_or_create_ptr()
    xl_ops_publish(key, Int(ops[].entries.unsafe_ptr()), x_linear_ops_host_v1)
