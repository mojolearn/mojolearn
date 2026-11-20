# SPDX-License-Identifier: Apache-2.0
"""Publication of `_mojolearn_training_host`'s neural primitive entry in the
Samba table (`core/samba_entry_table.mojo`, family SAMBA_TABLE_TRAINING,
vendor 0). Called once at module init; a repeated init publishes the same
pointer (core/entry_table.mojo). Metadata only; nothing to split.
"""
from checks.numerics import GLOBAL_NUMERIC_MODE
from core.native_descriptor_abi import NativeDescriptorCall
from core.samba_entry_table import SambaEntryKey, samba_table_publish, SAMBA_TABLE_TRAINING
from training.host.native_primitive_dispatch import NATIVE_HOST_TRAINING_PRIMITIVE_API, host_primitive_v1


def publish_samba_training_host_primitives() raises:
    var entry: NativeDescriptorCall = host_primitive_v1
    var owner = NATIVE_HOST_TRAINING_PRIMITIVE_API.get_or_create_ptr()
    samba_table_publish[SAMBA_TABLE_TRAINING](SambaEntryKey(Int(GLOBAL_NUMERIC_MODE), 0), Int(owner), entry)
