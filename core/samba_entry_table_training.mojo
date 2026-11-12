# SPDX-License-Identifier: Apache-2.0
"""Publication of `_mojolearn_training`'s neural primitive entry in the Samba
table (`core/samba_entry_table.mojo`, family SAMBA_TABLE_TRAINING). Called
once at module init. A build with no GPU vendor ("none") publishes nothing:
its CPU install routes to `_mojolearn_training_host`, which publishes the
vendor-0 slot. The same init publishes the Samba program entry
(`core/samba_entry_table_programs.mojo`). A repeated init publishes the same pointer
(core/entry_table.mojo). Metadata only; nothing to split.
"""
from checks.numerics import GLOBAL_NUMERIC_MODE
from checks.vendor import COMPILED_VENDOR
from core.native_descriptor_abi import NativeDescriptorCall
from core.samba_entry_table import SambaEntryKey, samba_table_publish, samba_vendor_word, SAMBA_TABLE_TRAINING
from training.native_primitive_dispatch import NATIVE_PRIMITIVE_API, native_primitive_v1
from core.samba_entry_table_programs import samba_programs_publish
from training.samba_program_entry import NATIVE_SAMBA_PROGRAMS_API


def publish_samba_training_primitives() raises:
    var word = samba_vendor_word(String(COMPILED_VENDOR))
    if word <= 0:
        return
    var entry: NativeDescriptorCall = native_primitive_v1
    var owner = NATIVE_PRIMITIVE_API.get_or_create_ptr()
    samba_table_publish[SAMBA_TABLE_TRAINING](SambaEntryKey(Int(GLOBAL_NUMERIC_MODE), word), Int(owner), entry)
    # The Samba stack program behind native handles
    # (core/samba_entry_table_programs.mojo, lane samba-sessions-native).
    var programs = NATIVE_SAMBA_PROGRAMS_API.get_or_create_ptr()
    samba_programs_publish(SambaEntryKey(Int(GLOBAL_NUMERIC_MODE), word), Int(programs), programs[].entries[0])
