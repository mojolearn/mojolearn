# SPDX-License-Identifier: Apache-2.0
"""RUN piece of the x_decomp device binding (bindings/_mojolearn_x_decomp.mojo):
the native publication of its operations table in the linear entry table.
Typed Mojo only, no Python import (tools/python_free_dirs.sh); moved
unchanged (lane rs-decomposition, 2026-11-12)."""
from core.linear_entry_table import LinearEntryKey, linear_table_publish, LINEAR_TABLE_DECOMP
from checks.vendor import COMPILED_VENDOR
from checks.numerics import GLOBAL_NUMERIC_MODE
from x_decomp.descriptor_schema import XD_ABI_NAME, XD_OPERATION_COUNT
from x_decomp.descriptor_device import X_DECOMP_DEVICE_API, x_decomp_device_v1


def _publish_linear_entry_table() raises:
    # Native publication only; the existing capsule ABI remains for clients
    # not yet converted. The original descriptor owner has image lifetime.
    var api = X_DECOMP_DEVICE_API.get_or_create_ptr()
    var key = LinearEntryKey(GLOBAL_NUMERIC_MODE, (1 if COMPILED_VENDOR == "metal" else 2 if COMPILED_VENDOR == "cuda" else 3 if COMPILED_VENDOR == "hip" else -1))
    linear_table_publish[LINEAR_TABLE_DECOMP](key,
        Int(api[].entries.unsafe_ptr()), [String(XD_ABI_NAME)], [XD_OPERATION_COUNT], [x_decomp_device_v1])
