# SPDX-License-Identifier: Apache-2.0
"""The Samba program table: the Samba stack program behind native registry
handles (lane samba-sessions-native, 2026-11-12).

`SambaStack.forward` / `step` on a device build and `SambaDecodeSession` run
one `SambaStackProgram` (training/samba_program.mojo). The program lives in
the GPU training image's own state registry (`core/state_registry.mojo`,
generation-checked handles, Arc leases); a caller holds a handle word only
(`core/samba_program_handle.mojo`) and reaches every program operation through
the ONE descriptor entry `_mojolearn_training` publishes here at module init
(`training/samba_program_entry.mojo::samba_programs_v1`), keyed by the
stack's (tier, vendor). GPU keys only (vendor 1 metal, 2 cuda, 3 hip): a
host build has no device program, so a CPU key refuses here and the caller
runs its host route. This replaces the nine Python entries of
`_mojolearn_training` (`samba_program_open/shape/forward`,
`samba_decode_open/step/tokens/export/load`) and their Python program
objects.

A handle word is `generation << 32 | index` (never 0: generations start at
1); 0 means "no handle". Publication and lookup are small metadata
operations with no independent numerical work to split.
"""
from core.entry_table import EntryRef, entry_ref_call, entry_ref_word, entry_table_publish, entry_table_read
from core.native_descriptor_abi import NativeDescriptorCall
from core.samba_entry_table import SambaEntryKey
from core.state_registry import StateHandle

comptime SAMBA_PROGRAMS_SCHEMA = "mojolearn.samba.programs.v1"
comptime SAMBA_PROGRAMS_OPERATIONS = 8

# Operation codes. `p` (integer words) starts with the handle word except for
# OPEN. The caller owns every buffer an address names until the call returns.
comptime SAMBA_PROGRAM_OPEN = 0
"""a = [registry (n_total f32, read once), state pieces...];
p = [n_layers, kinds (n_layers: 0 mamba3, 1 attention), vocab, d_model,
n_heads, n_kv_heads, head_dim, intermediate, tie (0/1), n_total,
batch_size, max_tokens, cursors (2 per layer when pieces are given)...];
f = [norm_eps]. With state pieces the state is loaded (a decode session).
Returns the handle word."""
comptime SAMBA_PROGRAM_STATUS = 1
"""p = [h] -> 1 when the program is usable, 0 after a failed call."""
comptime SAMBA_PROGRAM_FORWARD = 2
"""a = [registry, ids (b*l i32), logits (b*l*vocab f32, written), cursors_out
(2 Int words per layer, written; 0 when stateless), reports (4 f32 per
Mamba-3 layer in layer order, written)..., state pieces...];
p = [h, b, l, n_reports, cursors in (2 per layer when pieces are given)...].
Stateless (no pieces): a zero state, discarded. Returns 0."""
comptime SAMBA_PROGRAM_DECODE_STEP = 3
"""a = [ids (b i32), logits (b*vocab f32, written)]; p = [h, b] -> the first
attention layer's cached tokens after the step."""
comptime SAMBA_PROGRAM_DECODE_TOKENS = 4
"""a = [ids (b*n i32, (B, N) order), logits (b*n*vocab f32, written)];
p = [h, b, n] -> as DECODE_STEP. Bytes equal n DECODE_STEP calls."""
comptime SAMBA_PROGRAM_DECODE_EXPORT = 5
"""a = [cursors_out (capacity Int words, written), state pieces...];
p = [h, capacity] -> the number of cursor words written."""
comptime SAMBA_PROGRAM_DECODE_LOAD = 6
"""a = [state pieces...]; p = [h, cursors...]. A successful load makes a lost
program usable."""
comptime SAMBA_PROGRAM_RELEASE = 7
"""p = [h]: release the handle; the program is freed with its last lease."""


def samba_program_handle_word(handle: StateHandle) raises -> Int:
    """One plain word for a registry handle (index low, generation high)."""
    if handle.index < 0 or handle.index > 0x7FFFFFFF or handle.generation == 0 or handle.generation > UInt64(0x7FFFFFFF):
        raise Error("native Samba programs: handle out of range")
    return (Int(handle.generation) << 32) | handle.index


def samba_program_handle_of(word: Int) raises -> StateHandle:
    if word <= 0:
        raise Error("native Samba programs: no program handle")
    return StateHandle(word & 0xFFFFFFFF, UInt64(word >> 32))


def _programs_key_check(key: SambaEntryKey) raises:
    if key.mode < 0 or key.mode > 1 or key.vendor < 1 or key.vendor > 3:
        raise Error("native Samba programs: invalid tier/vendor key (GPU images only)")


def _programs_publish_named[name: StaticString](key: SambaEntryKey, publisher: Int, entry: NativeDescriptorCall) raises:
    var schemas: List[String] = [String(SAMBA_PROGRAMS_SCHEMA)]
    var operations: List[Int] = [SAMBA_PROGRAMS_OPERATIONS]
    var entries = List[NativeDescriptorCall]()
    entries.append(entry)
    entry_table_publish[name](key.mode, key.vendor, publisher, schemas, operations, entries)


def _programs_read_named[name: StaticString](key: SambaEntryKey) raises -> NativeDescriptorCall:
    return entry_table_read[name](key.mode, key.vendor, 0, String(SAMBA_PROGRAMS_SCHEMA), SAMBA_PROGRAMS_OPERATIONS)


def samba_programs_publish(key: SambaEntryKey, publisher: Int, entry: NativeDescriptorCall) raises:
    """Publish the GPU training image's program entry under `key` (module
    init). A repeated init publishes the same pointer."""
    _programs_key_check(key)
    if key.mode == 1 and key.vendor == 1:
        _programs_publish_named["MojolearnEntriesSambaProgramsIdenticalMetalV1"](key, publisher, entry)
    elif key.mode == 0 and key.vendor == 1:
        _programs_publish_named["MojolearnEntriesSambaProgramsFastMetalV1"](key, publisher, entry)
    elif key.mode == 1 and key.vendor == 2:
        _programs_publish_named["MojolearnEntriesSambaProgramsIdenticalCUDAV1"](key, publisher, entry)
    elif key.mode == 0 and key.vendor == 2:
        _programs_publish_named["MojolearnEntriesSambaProgramsFastCUDAV1"](key, publisher, entry)
    elif key.mode == 1 and key.vendor == 3:
        _programs_publish_named["MojolearnEntriesSambaProgramsIdenticalHIPV1"](key, publisher, entry)
    else:
        _programs_publish_named["MojolearnEntriesSambaProgramsFastHIPV1"](key, publisher, entry)


def samba_programs_entry(key: SambaEntryKey) raises -> NativeDescriptorCall:
    """The published program entry for `key`; a missing or mismatched
    publication refuses with the table's error."""
    _programs_key_check(key)
    if key.mode == 1 and key.vendor == 1:
        return _programs_read_named["MojolearnEntriesSambaProgramsIdenticalMetalV1"](key)
    if key.mode == 0 and key.vendor == 1:
        return _programs_read_named["MojolearnEntriesSambaProgramsFastMetalV1"](key)
    if key.mode == 1 and key.vendor == 2:
        return _programs_read_named["MojolearnEntriesSambaProgramsIdenticalCUDAV1"](key)
    if key.mode == 0 and key.vendor == 2:
        return _programs_read_named["MojolearnEntriesSambaProgramsFastCUDAV1"](key)
    if key.mode == 1 and key.vendor == 3:
        return _programs_read_named["MojolearnEntriesSambaProgramsIdenticalHIPV1"](key)
    return _programs_read_named["MojolearnEntriesSambaProgramsFastHIPV1"](key)


def samba_programs_ref(key: SambaEntryKey) raises -> EntryRef:
    """The program entry of the image `key` names, resolved once (lane
    entry-table-zero-alloc); the table refusals raise here."""
    return EntryRef(samba_programs_entry(key), entry_ref_word(key.mode, key.vendor, SAMBA_PROGRAMS_OPERATIONS))


def samba_programs_available(key: SambaEntryKey) -> Bool:
    """Whether the stack's key has a device program: a GPU key whose
    training image published its program entry. Replaces the former
    `hasattr(entries, "samba_program_open")` probe of a Python module."""
    if key.vendor < 1 or key.vendor > 3:
        return False
    try:
        _ = samba_programs_entry(key)
        return True
    except:
        return False


def samba_programs_call(entry: EntryRef, operation: Int, addresses: List[Int],
    words: List[Int], floats: List[Float64]) raises -> Int:
    """One synchronous program call through a resolved entry: one indirect
    call. The caller owns every buffer the addresses name until this returns
    and releases the GIL around it."""
    return entry_ref_call(entry, operation, addresses, words, floats)
