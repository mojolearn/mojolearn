# SPDX-License-Identifier: Apache-2.0
"""THE MEMBER ABI: the one native interface through which a meta-estimator
runs a member estimator (model selection, the parallel drivers, bagging,
AdaBoost, stacking and voting, the multi-output and calibration wrappers).

There is exactly one, defined here. A family PUBLISHES one entry per image
(tier, vendor) in its member table; a caller RESOLVES that entry once, when it
unpacks the member, and afterwards calls it with typed values only. No call
of this interface holds, reads or returns a Python object, and none calls the
member's Python door (docs/plans/PYTHON_AT_THE_BOUNDARY.md sections 2.5, 3).

WHAT CROSSES (all plain words; the caller owns every addressed buffer until
the call returns):
  * `MemberMatrix`: a typed matrix view (address, rows, cols, row stride,
    element type, home). X, y, the sample weights and the output are views.
  * the STATE HANDLE: an integer the family's own state store minted
    (`MEMBER_OP_NEW`) and releases (`MEMBER_OP_FREE`). A handle that was
    released, or that another family minted, is refused in words by the
    store's generation check; it is never a dangling pointer.
  * the member's `DeviceRoute` (core/device_route.mojo), resolved ONCE by
    the caller's UNPACK from the member's `device` and numeric mode with the
    family's own resolver. The member entry reads its kernels' entries from
    the route it is handed: it asks no process-wide device question and
    reads no table per call.

THE CALL. The entry has the descriptor signature every native table of ours
uses (`NativeDescriptorCall`, core/entry_table.mojo): operation = one of the
`MEMBER_OP_*` codes, ONE address = the caller's `MemberCall`, no integer or
float spans. Results come back in the `MemberCall` (`state`, `value`,
`out_cols`) and in the buffer `dst` views; the result word is 0, or the
capability mask for `MEMBER_OP_CAPABILITIES`. A refusal is the entry's error
text, raised by `member_call` in the caller.

SERIAL BY NATURE: publication and resolution are metadata operations with no
independent units. The member operations themselves are the family's RUN and
parallel inside as that family is; a caller runs independent members (folds,
candidates, bagged members) on its own workers, one `MemberCall` per worker.
"""
from core.device_route import DeviceRoute
from core.entry_table import EntryRef, entry_table_publish, entry_table_resolve, entry_ref_call
from core.native_descriptor_abi import NativeDescriptorCall

#: The schema of a member entry. Bump with `MEMBER_CALL_LAYOUT` on any change
#: of `MemberCall`, `MemberMatrix`, `DeviceRoute` or the operation codes.
comptime MEMBER_ABI = "mojolearn.member.v1"
comptime MEMBER_CALL_LAYOUT = 1

comptime MEMBER_OP_CAPABILITIES = 0  # result word: the mask of MEMBER_CAN_* this family's member answers
comptime MEMBER_OP_NEW = 1           # params -> call.state (an unfitted member state)
comptime MEMBER_OP_FREE = 2          # call.state is released; a later use is refused
comptime MEMBER_OP_FIT = 3           # x, y, weights (rows 0: none) -> the state is fitted
comptime MEMBER_OP_PREDICT = 4       # x -> dst
comptime MEMBER_OP_PREDICT_PROBA = 5 # x -> dst (rows x classes)
comptime MEMBER_OP_DECISION = 6      # x -> dst
comptime MEMBER_OP_SCORE = 7         # x, y, weights -> call.value (the family's default score)
comptime MEMBER_OP_OUTPUTS = 8       # call.flags = MEMBER_OP_* asked -> call.out_cols (columns of dst per row)
comptime MEMBER_OPERATION_COUNT = 9

comptime MEMBER_CAN_FIT = 1
comptime MEMBER_CAN_PREDICT = 2
comptime MEMBER_CAN_PREDICT_PROBA = 4
comptime MEMBER_CAN_DECISION = 8
comptime MEMBER_CAN_SCORE = 16

#: Element types of a `MemberMatrix`.
comptime MEMBER_F32 = 0
comptime MEMBER_F64 = 1
comptime MEMBER_I32 = 2
comptime MEMBER_I64 = 3

#: Where a view's words live: host memory, or a device-rows handle of the
#: route's device (bindings/device_rows_door.mojo; `data` is then the handle).
comptime MEMBER_HOME_HOST = 0
comptime MEMBER_HOME_DEVICE = 1

#: Families with a member table. A new family takes the next number and adds
#: its eight slot names to `member_publish` and `member_resolve` below: the
#: tier and the vendor are part of the slot name (plan section 2.5).
comptime MEMBER_FAMILY_LINEAR = 0
comptime MEMBER_FAMILY_COUNT = 1


struct MemberMatrix(ImplicitlyCopyable, Movable):
    """A typed matrix view: `rows` x `cols` elements of `dtype`, row `i`
    starting `i * stride` elements after `data` (`stride == cols` for a dense
    row-major matrix; a vector is `cols == 1`). `rows == 0` is "not given"."""
    var data: Int
    var rows: Int
    var cols: Int
    var stride: Int
    var dtype: Int
    var home: Int

    def __init__(out self):
        self.data = 0
        self.rows = 0
        self.cols = 0
        self.stride = 0
        self.dtype = MEMBER_F32
        self.home = MEMBER_HOME_HOST

    def __init__(out self, data: Int, rows: Int, cols: Int, dtype: Int = MEMBER_F32):
        self.data = data
        self.rows = rows
        self.cols = cols
        self.stride = cols
        self.dtype = dtype
        self.home = MEMBER_HOME_HOST

    def given(self) -> Bool:
        return self.rows > 0 and self.data != 0

    def dense_host(self, dtype: Int) -> Bool:
        """A dense row-major host view of `dtype` (what a host kernel reads
        by address without a gather)."""
        return self.given() and self.home == MEMBER_HOME_HOST and self.dtype == dtype and self.stride == self.cols


struct MemberCall(Movable):
    """One member operation. The caller builds it on its own frame (no heap),
    fills the views the operation reads and calls `member_call`."""
    var layout: Int
    var state: Int
    var route: DeviceRoute
    var x: MemberMatrix
    var y: MemberMatrix
    var weights: MemberMatrix
    var dst: MemberMatrix
    # MEMBER_OP_NEW: the member's hyperparameters as Float64 words, in the
    # order the family documents beside its entry (address, count).
    var params: Int
    var param_count: Int
    var flags: Int
    # Results.
    var value: Float64
    var out_cols: Int

    def __init__(out self, route: DeviceRoute, state: Int = 0):
        self.layout = MEMBER_CALL_LAYOUT
        self.state = state
        self.route = route
        self.x = MemberMatrix()
        self.y = MemberMatrix()
        self.weights = MemberMatrix()
        self.dst = MemberMatrix()
        self.params = 0
        self.param_count = 0
        self.flags = 0
        self.value = 0.0
        self.out_cols = 0


@fieldwise_init
struct MemberRef(ImplicitlyCopyable, Movable):
    """A family's member entry of one image, resolved once (two words) plus
    its capability mask. The caller's member owner holds it beside the
    member's `DeviceRoute` and state handle."""
    var entry: EntryRef
    var capabilities: Int

    def can(self, capability: Int) -> Bool:
        return (self.capabilities & capability) != 0


def _member_key_check(mode: Int, vendor: Int) raises:
    if mode < 0 or mode > 1 or vendor < 0 or vendor > 3:
        raise Error("member table: invalid tier/vendor key")


def member_publish[family: Int](mode: Int, vendor: Int, publisher: Int, entry: NativeDescriptorCall) raises:
    """Publish a family's member entry for the image (`mode` 1 identical /
    0 fast, `vendor` 0 CPU, 1 Metal, 2 CUDA, 3 HIP). Called once from the
    image's module initialization, after its kernel tables are published;
    `publisher` is any stable nonzero address of the image."""
    comptime assert family >= 0 and family < MEMBER_FAMILY_COUNT, "unknown member family"
    _member_key_check(mode, vendor)
    var schemas: List[String] = [String(MEMBER_ABI)]
    var operations: List[Int] = [MEMBER_OPERATION_COUNT]
    var entries: List[NativeDescriptorCall] = [entry]
    if vendor == 0 and mode == 1:
        comptime NAME = ("MojolearnMemberLinearIdenticalCPUV1")
        entry_table_publish[NAME](mode, vendor, publisher, schemas, operations, entries)
    elif vendor == 0:
        comptime NAME = ("MojolearnMemberLinearFastCPUV1")
        entry_table_publish[NAME](mode, vendor, publisher, schemas, operations, entries)
    elif vendor == 1 and mode == 1:
        comptime NAME = ("MojolearnMemberLinearIdenticalMetalV1")
        entry_table_publish[NAME](mode, vendor, publisher, schemas, operations, entries)
    elif vendor == 1:
        comptime NAME = ("MojolearnMemberLinearFastMetalV1")
        entry_table_publish[NAME](mode, vendor, publisher, schemas, operations, entries)
    elif vendor == 2 and mode == 1:
        comptime NAME = ("MojolearnMemberLinearIdenticalCUDAV1")
        entry_table_publish[NAME](mode, vendor, publisher, schemas, operations, entries)
    elif vendor == 2:
        comptime NAME = ("MojolearnMemberLinearFastCUDAV1")
        entry_table_publish[NAME](mode, vendor, publisher, schemas, operations, entries)
    elif mode == 1:
        comptime NAME = ("MojolearnMemberLinearIdenticalHIPV1")
        entry_table_publish[NAME](mode, vendor, publisher, schemas, operations, entries)
    else:
        comptime NAME = ("MojolearnMemberLinearFastHIPV1")
        entry_table_publish[NAME](mode, vendor, publisher, schemas, operations, entries)


def _member_entry[family: Int](mode: Int, vendor: Int) raises -> EntryRef:
    comptime assert family >= 0 and family < MEMBER_FAMILY_COUNT, "unknown member family"
    _member_key_check(mode, vendor)
    var schema = String(MEMBER_ABI)
    if vendor == 0 and mode == 1:
        comptime NAME = ("MojolearnMemberLinearIdenticalCPUV1")
        return entry_table_resolve[NAME](mode, vendor, 0, schema, MEMBER_OPERATION_COUNT)
    if vendor == 0:
        comptime NAME = ("MojolearnMemberLinearFastCPUV1")
        return entry_table_resolve[NAME](mode, vendor, 0, schema, MEMBER_OPERATION_COUNT)
    if vendor == 1 and mode == 1:
        comptime NAME = ("MojolearnMemberLinearIdenticalMetalV1")
        return entry_table_resolve[NAME](mode, vendor, 0, schema, MEMBER_OPERATION_COUNT)
    if vendor == 1:
        comptime NAME = ("MojolearnMemberLinearFastMetalV1")
        return entry_table_resolve[NAME](mode, vendor, 0, schema, MEMBER_OPERATION_COUNT)
    if vendor == 2 and mode == 1:
        comptime NAME = ("MojolearnMemberLinearIdenticalCUDAV1")
        return entry_table_resolve[NAME](mode, vendor, 0, schema, MEMBER_OPERATION_COUNT)
    if vendor == 2:
        comptime NAME = ("MojolearnMemberLinearFastCUDAV1")
        return entry_table_resolve[NAME](mode, vendor, 0, schema, MEMBER_OPERATION_COUNT)
    if mode == 1:
        comptime NAME = ("MojolearnMemberLinearIdenticalHIPV1")
        return entry_table_resolve[NAME](mode, vendor, 0, schema, MEMBER_OPERATION_COUNT)
    comptime NAME = ("MojolearnMemberLinearFastHIPV1")
    return entry_table_resolve[NAME](mode, vendor, 0, schema, MEMBER_OPERATION_COUNT)


def member_call(member: MemberRef, operation: Int, mut call: MemberCall) raises -> Int:
    """One member operation: one indirect call, the `MemberCall` passed by
    address from the caller's frame. No heap allocation, no table read. A
    refusal of the family is raised here with the family's words."""
    var address: Array[Int, 1] = [Int(MutPointer(to=call))]
    var result = entry_ref_call(member.entry, operation, Int(address.unsafe_ptr()), 1, 0, 0, 0, 0)
    # Integer addresses carry no lifetime: keep the envelope past the call.
    _ = address
    return result


def member_resolve[family: Int](route: DeviceRoute) raises -> MemberRef:
    """The member entry of the image `route` resolved to (its `mode` and
    `vendor` words), read ONCE per member by the caller's UNPACK, with its
    capability mask. Every refusal of the table (the family's image is not
    loaded, the ABI differs) happens here, never at call time."""
    var entry = _member_entry[family](route.mode, route.vendor)
    var probe = MemberCall(route)
    var member = MemberRef(entry, 0)
    member.capabilities = member_call(member, MEMBER_OP_CAPABILITIES, probe)
    return member


def member_call_from_spans(operation: Int, aa: Int, na: Int, np: Int, nf: Int) raises -> Int:
    """PROVIDER side: admit a member entry's spans and return the address of
    the caller's `MemberCall` (read it through
    `MutPointer[MemberCall, MutUntrackedOrigin](unsafe_from_address=...)`)."""
    if operation < 0 or operation >= MEMBER_OPERATION_COUNT or na != 1 or np != 0 or nf != 0 or not aa:
        raise Error("member ABI: invalid request spans")
    var address = MutPointer[Int, MutUntrackedOrigin](unsafe_from_address=aa)[0]
    if not address:
        raise Error("member ABI: null call address")
    if MutPointer[MemberCall, MutUntrackedOrigin](unsafe_from_address=address)[].layout != MEMBER_CALL_LAYOUT:
        raise Error("member ABI: the caller's call layout differs (mixed binary versions)")
    return address


def member_unsupported(family: String, operation: Int) raises:
    """The words of an operation a family's member does not answer."""
    var name = String("fit") if operation == MEMBER_OP_FIT else (
        String("predict") if operation == MEMBER_OP_PREDICT else (
        String("predict_proba") if operation == MEMBER_OP_PREDICT_PROBA else (
        String("decision_function") if operation == MEMBER_OP_DECISION else (
        String("score") if operation == MEMBER_OP_SCORE else String("operation ") + String(operation)))))
    raise Error("AttributeError: a " + family + " member has no " + name)
