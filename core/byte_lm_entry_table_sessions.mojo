# SPDX-License-Identifier: Apache-2.0
"""Resident byte LM trainer sessions as native registry handles (lane
byte-lm-sessions-native, 2026-11-11).

Family `BYTE_LM_TABLE_SESSIONS` of the byte LM tables
(`core/byte_lm_entry_table.mojo` holds the trainer and sequence optimizer
families). Each byte LM image publishes one session entry beside its trainer
table: `_mojolearn_byte_lm` (vendor COMPILED_VENDOR, publisher
`training/byte_lm_session_entry.mojo`) and `_mojolearn_byte_lm_host` (vendor
cpu, `training/byte_lm_host_session_entry.mojo`).

The entry owns the image's resident sessions in a `core/state_registry.mojo`
registry (generation-checked handles, Arc leases). It replaces the Python
session object of the resident trainer (`byte_lm_session_create`, the
`byte_lm_native_session_handle_v1` handle factory and the provider object
of `byte_lm_trainer_api_v1`): the trainer door holds a `StateHandle` only.

Operations (the descriptor shape of `core/native_descriptor_abi.mojo`):
- `BS_CREATE`: a = [out], writes out[0] = handle index, out[1] = handle
  generation. A new session is closed and owns no device memory (creation is
  lazy, as `byte_lm_session_create` was).
- `BS_RUN`: one trainer operation on the session: p = [index, generation,
  trainer operation, trainer words...], a = the trainer descriptors after
  the session, f = the trainer scalars. The entry holds the session's lease
  for the call, so a concurrent release cannot free it under the call.
- `BS_CLOSE`: p = [index, generation]; releases the handle. The session's
  native destructor (device teardown) runs when the last lease drops.
A stale or foreign handle refuses ("state is not fitted or was released").

Table names carry the tier and vendor as the trainer table's do. Extending
the layout requires a new V suffix. Publication and lookup are small
metadata operations with no independent numerical work to split.
"""
from core.entry_table import entry_table_publish, entry_table_read
from core.native_descriptor_abi import NativeDescriptorCall
from core.native_descriptor_call import native_descriptor_call
from core.state_registry import StateHandle
from core.byte_lm_entry_table import ByteLmEntryKey

comptime BYTE_LM_SESSIONS_ABI = "mojolearn.training.byte_lm.sessions.v1"
comptime BS_CREATE = 0
comptime BS_RUN = 1
comptime BS_CLOSE = 2
comptime BS_OP_COUNT = 3


def _byte_lm_sessions_key_check(key: ByteLmEntryKey) raises:
    if key.mode < 0 or key.mode > 1 or key.vendor < 0 or key.vendor > 3:
        raise Error("native byte LM sessions: invalid tier/vendor key")


def byte_lm_sessions_publish(key: ByteLmEntryKey, publisher: Int, entry: NativeDescriptorCall) raises:
    _byte_lm_sessions_key_check(key)
    var schemas: List[String] = [String(BYTE_LM_SESSIONS_ABI)]
    var operations: List[Int] = [BS_OP_COUNT]
    var entries = List[NativeDescriptorCall]()
    entries.append(entry)
    if key.vendor == 0 and key.mode == 1:
        comptime NAME = "MojolearnEntriesByteLmSessionsIdenticalCPUV1"
        entry_table_publish[NAME](key.mode, key.vendor, publisher, schemas, operations, entries)
        return
    if key.vendor == 0 and key.mode == 0:
        comptime NAME = "MojolearnEntriesByteLmSessionsFastCPUV1"
        entry_table_publish[NAME](key.mode, key.vendor, publisher, schemas, operations, entries)
        return
    if key.vendor == 1 and key.mode == 1:
        comptime NAME = "MojolearnEntriesByteLmSessionsIdenticalMetalV1"
        entry_table_publish[NAME](key.mode, key.vendor, publisher, schemas, operations, entries)
        return
    if key.vendor == 1 and key.mode == 0:
        comptime NAME = "MojolearnEntriesByteLmSessionsFastMetalV1"
        entry_table_publish[NAME](key.mode, key.vendor, publisher, schemas, operations, entries)
        return
    if key.vendor == 2 and key.mode == 1:
        comptime NAME = "MojolearnEntriesByteLmSessionsIdenticalCUDAV1"
        entry_table_publish[NAME](key.mode, key.vendor, publisher, schemas, operations, entries)
        return
    if key.vendor == 2 and key.mode == 0:
        comptime NAME = "MojolearnEntriesByteLmSessionsFastCUDAV1"
        entry_table_publish[NAME](key.mode, key.vendor, publisher, schemas, operations, entries)
        return
    if key.vendor == 3 and key.mode == 1:
        comptime NAME = "MojolearnEntriesByteLmSessionsIdenticalHIPV1"
        entry_table_publish[NAME](key.mode, key.vendor, publisher, schemas, operations, entries)
        return
    if key.vendor == 3 and key.mode == 0:
        comptime NAME = "MojolearnEntriesByteLmSessionsFastHIPV1"
        entry_table_publish[NAME](key.mode, key.vendor, publisher, schemas, operations, entries)
        return
    raise Error("native byte LM sessions: unsupported execution key")


def byte_lm_sessions_entry(key: ByteLmEntryKey) raises -> NativeDescriptorCall:
    _byte_lm_sessions_key_check(key)
    comptime SCHEMA = BYTE_LM_SESSIONS_ABI
    if key.vendor == 0 and key.mode == 1:
        comptime NAME = "MojolearnEntriesByteLmSessionsIdenticalCPUV1"
        return entry_table_read[NAME](key.mode, key.vendor, 0, String(SCHEMA), BS_OP_COUNT)
    if key.vendor == 0 and key.mode == 0:
        comptime NAME = "MojolearnEntriesByteLmSessionsFastCPUV1"
        return entry_table_read[NAME](key.mode, key.vendor, 0, String(SCHEMA), BS_OP_COUNT)
    if key.vendor == 1 and key.mode == 1:
        comptime NAME = "MojolearnEntriesByteLmSessionsIdenticalMetalV1"
        return entry_table_read[NAME](key.mode, key.vendor, 0, String(SCHEMA), BS_OP_COUNT)
    if key.vendor == 1 and key.mode == 0:
        comptime NAME = "MojolearnEntriesByteLmSessionsFastMetalV1"
        return entry_table_read[NAME](key.mode, key.vendor, 0, String(SCHEMA), BS_OP_COUNT)
    if key.vendor == 2 and key.mode == 1:
        comptime NAME = "MojolearnEntriesByteLmSessionsIdenticalCUDAV1"
        return entry_table_read[NAME](key.mode, key.vendor, 0, String(SCHEMA), BS_OP_COUNT)
    if key.vendor == 2 and key.mode == 0:
        comptime NAME = "MojolearnEntriesByteLmSessionsFastCUDAV1"
        return entry_table_read[NAME](key.mode, key.vendor, 0, String(SCHEMA), BS_OP_COUNT)
    if key.vendor == 3 and key.mode == 1:
        comptime NAME = "MojolearnEntriesByteLmSessionsIdenticalHIPV1"
        return entry_table_read[NAME](key.mode, key.vendor, 0, String(SCHEMA), BS_OP_COUNT)
    if key.vendor == 3 and key.mode == 0:
        comptime NAME = "MojolearnEntriesByteLmSessionsFastHIPV1"
        return entry_table_read[NAME](key.mode, key.vendor, 0, String(SCHEMA), BS_OP_COUNT)
    raise Error("native byte LM sessions: unsupported execution key")


def byte_lm_sessions_published(key: ByteLmEntryKey) -> Bool:
    """True when the image of `key` publishes resident sessions: the trainer
    door's resident/stateless choice (it replaced the probe of the six
    Python session entry names)."""
    try:
        _ = byte_lm_sessions_entry(key)
        return True
    except:
        return False


# ---------------------------------------------------------------- caller side

def byte_lm_resident_open(key: ByteLmEntryKey) raises -> StateHandle:
    """A new closed resident session of the image `key` names."""
    var out = List[Int](length=2, fill=0)
    var a: List[Int] = [Int(out.unsafe_ptr())]
    _ = native_descriptor_call(byte_lm_sessions_entry(key), BS_CREATE, a, List[Int](), List[Float64]())
    var index = out[0]
    var generation = out[1]
    _ = out^  # keep the handle output alive past the synchronous call
    if index < 0 or generation <= 0:
        raise Error("native byte LM sessions: creation returned an invalid handle")
    return StateHandle(index, UInt64(generation))


def byte_lm_resident_run(key: ByteLmEntryKey, handle: StateHandle, operation: Int, addresses: List[Int],
                         ints: List[Int], floats: List[Float64]) raises -> Int:
    """One trainer operation (the `byte_lm_trainer_api_v1` numbering) on the
    resident session `handle`. `addresses` are the descriptors after the
    session word."""
    var words: List[Int] = [handle.index, Int(handle.generation), operation]
    for i in range(len(ints)):  # small-loop(the trainer plan words)
        words.append(ints[i])
    return native_descriptor_call(byte_lm_sessions_entry(key), BS_RUN, addresses, words, floats)


def byte_lm_resident_close(key: ByteLmEntryKey, handle: StateHandle) raises:
    """Release the handle; a stale handle refuses."""
    var words: List[Int] = [handle.index, Int(handle.generation)]
    _ = native_descriptor_call(byte_lm_sessions_entry(key), BS_CLOSE, List[Int](), words, List[Float64]())


# ---------------------------------------------------------------- publisher side

def byte_lm_sessions_handle(p: List[Int]) raises -> StateHandle:
    """The handle words p[0:2] of a run or close request."""
    if len(p) < 2 or p[0] < 0 or p[1] <= 0:
        raise Error("native byte LM sessions: invalid session handle")
    return StateHandle(p[0], UInt64(p[1]))


def byte_lm_sessions_write_handle(a: List[Int], handle: StateHandle) raises -> Int:
    if len(a) != 1 or a[0] == 0:
        raise Error("native byte LM sessions: create takes one handle output")
    var out = MutPointer[Int, MutUntrackedOrigin](unsafe_from_address=a[0])
    out[0] = handle.index
    out[1] = Int(handle.generation)
    return 0


@fieldwise_init
struct ByteLmSessionRequest(Movable):
    """A run request rewritten for the image's trainer entry: the session
    address first, then the caller's descriptors and trainer words."""
    var operation: Int
    var addresses: List[Int]
    var ints: List[Int]


def byte_lm_sessions_request(session: Int, a: List[Int], p: List[Int]) raises -> ByteLmSessionRequest:
    if len(p) < 3 or session == 0:
        raise Error("native byte LM sessions: invalid run request")
    var addresses: List[Int] = [session]
    for i in range(len(a)):  # small-loop(the trainer descriptors of one call)
        addresses.append(a[i])
    var ints = List[Int]()
    for i in range(3, len(p)):  # small-loop(the trainer plan words)
        ints.append(p[i])
    return ByteLmSessionRequest(p[2], addresses^, ints^)


def byte_lm_sessions_floats(ff: Int, nf: Int) raises -> List[Float64]:
    if nf < 0 or (nf > 0 and ff == 0):
        raise Error("native byte LM sessions: null scalar span")
    var f = List[Float64]()
    var fp = MutPointer[Float64, MutUntrackedOrigin](unsafe_from_address=ff)
    for i in range(nf):  # small-loop(the eight optimizer scalars)
        f.append(fp[i])
    return f^
