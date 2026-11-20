# SPDX-License-Identifier: Apache-2.0
"""Publication of `_mojolearn_neural_host`'s primitive entry (embedding, RMS
norm and linear forward of the checkpoint CPU inference model) in the Samba
table (`core/samba_entry_table.mojo`, family SAMBA_TABLE_NEURAL_HOST,
vendor 0). Called once at module init; a repeated init publishes the same
pointer (core/entry_table.mojo). Metadata only; nothing to split.
"""
from checks.numerics import GLOBAL_NUMERIC_MODE
from core.native_descriptor_abi import NativeDescriptorCall
from core.samba_entry_table import SambaEntryKey, samba_table_publish, SAMBA_TABLE_NEURAL_HOST
from core.neural_host_dispatch import NATIVE_HOST_PRIMITIVE_API, neural_host_dispatch


def publish_samba_neural_host_primitives() raises:
    var owner = NATIVE_HOST_PRIMITIVE_API.get_or_create_ptr()
    var entry: NativeDescriptorCall = neural_host_dispatch[0]
    samba_table_publish[SAMBA_TABLE_NEURAL_HOST](SambaEntryKey(Int(GLOBAL_NUMERIC_MODE), 0), Int(owner), entry)
