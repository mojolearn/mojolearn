# SPDX-License-Identifier: Apache-2.0
"""RUN piece of the x_decomp host binding (bindings/_mojolearn_x_decomp_host.mojo):
the native publication of its operations table in the linear entry table.
Typed Mojo only, no Python import (tools/python_free_dirs.sh); moved
unchanged (lane rs-decomposition, 2026-11-12)."""
from core.linear_entry_table import LinearEntryKey, linear_table_publish, LINEAR_TABLE_DECOMP
from checks.numerics import GLOBAL_NUMERIC_MODE
from x_decomp.descriptor_schema import XD_ABI_NAME, XD_OPERATION_COUNT
from x_decomp.descriptor_host import X_DECOMP_HOST_API, x_decomp_host_v1


def _publish_linear_entry_table() raises:
    # Native publication only; the existing capsule ABI remains for clients
    # not yet converted. The original descriptor owner has image lifetime.
    var api = X_DECOMP_HOST_API.get_or_create_ptr()
    var key = LinearEntryKey(GLOBAL_NUMERIC_MODE, 0)
    linear_table_publish[LINEAR_TABLE_DECOMP](key,
        Int(api[].entries.unsafe_ptr()), [String(XD_ABI_NAME)], [XD_OPERATION_COUNT], [x_decomp_host_v1])
