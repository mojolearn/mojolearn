# SPDX-License-Identifier: Apache-2.0
"""The member ABI (core/member_abi.mojo) and the lifetime of the linear
member state store (core/linear_member.mojo), without a kernel: three
independent fixtures, one host task each. A fit or a prediction with numbers
needs the family's kernel tables and belongs to the installed-wheel tests."""
from core.device_route import DeviceRoute
from core.entry_table import EntryRef
from core.host_parallel import host_parallelize
from core.linear_member import linear_member_v1
from core.member_abi import (
    MemberCall, MemberMatrix, MemberRef, member_call, MEMBER_OP_CAPABILITIES, MEMBER_OP_NEW, MEMBER_OP_FREE,
    MEMBER_OP_PREDICT, MEMBER_OP_PREDICT_PROBA, MEMBER_OP_OUTPUTS, MEMBER_CAN_FIT, MEMBER_CAN_PREDICT,
    MEMBER_CAN_SCORE, MEMBER_CAN_PREDICT_PROBA,
)

comptime MEMBER_ABI_FIXTURES = 3


def require(ok: Bool, message: String) raises:
    if not ok:
        raise Error(message)


def _member() -> MemberRef:
    return MemberRef(EntryRef(linear_member_v1, 0), 0)


def _new(kind: Float64, alpha: Float64) raises -> Int:
    var params: List[Float64] = [kind, 1.0, alpha]
    var call = MemberCall(DeviceRoute(0))
    call.params = Int(params.unsafe_ptr())
    call.param_count = 3
    _ = member_call(_member(), MEMBER_OP_NEW, call)
    _ = len(params)
    return call.state


def _refused(operation: Int, mut call: MemberCall, words: String) raises:
    var refused = False
    try:
        _ = member_call(_member(), operation, call)
    except error:
        refused = String(error).find(words) >= 0
    require(refused, "member ABI: expected a refusal holding '" + words + "'")


def _fixture(unit: Int) raises:
    if unit == 0:
        # capabilities and the output width
        var call = MemberCall(DeviceRoute(0))
        var mask = member_call(_member(), MEMBER_OP_CAPABILITIES, call)
        require(mask == (MEMBER_CAN_FIT | MEMBER_CAN_PREDICT | MEMBER_CAN_SCORE), "member ABI: linear capability mask")
        require((mask & MEMBER_CAN_PREDICT_PROBA) == 0, "member ABI: a regressor has no predict_proba")
        _ = member_call(_member(), MEMBER_OP_OUTPUTS, call)
        require(call.out_cols == 1, "member ABI: one output column")
    elif unit == 1:
        # two live states are two handles; a released handle is refused, never read
        var first = _new(1.0, 0.5)
        var second = _new(0.0, 0.0)
        require(first != 0 and second != 0 and first != second, "member ABI: two states, two handles")
        var call = MemberCall(DeviceRoute(0), first)
        _ = member_call(_member(), MEMBER_OP_FREE, call)
        require(call.state == 0, "member ABI: free clears the caller's handle")
        var stale = MemberCall(DeviceRoute(0), first)
        _refused(MEMBER_OP_FREE, stale, "released or never created")
        var third = _new(1.0, 1.0)
        require(third != first, "member ABI: a reused slot is a new generation")
        var other = MemberCall(DeviceRoute(0), second)
        _ = member_call(_member(), MEMBER_OP_FREE, other)
        var last = MemberCall(DeviceRoute(0), third)
        _ = member_call(_member(), MEMBER_OP_FREE, last)
    else:
        # refusals in words: bad parameters, an unfitted predict, an operation the family lacks
        var bad = MemberCall(DeviceRoute(0))
        _refused(MEMBER_OP_NEW, bad, "three parameters")
        var handle = _new(1.0, 0.0)
        var storage: List[Float32] = [1.0, 2.0, 0.0]
        var call = MemberCall(DeviceRoute(0), handle)
        call.x = MemberMatrix(Int(storage.unsafe_ptr()), 1, 2)
        call.dst = MemberMatrix(Int(storage.unsafe_ptr()) + 8, 1, 1)
        _refused(MEMBER_OP_PREDICT, call, "call fit before predict")
        _refused(MEMBER_OP_PREDICT_PROBA, call, "has no predict_proba")
        _ = len(storage)
        _ = member_call(_member(), MEMBER_OP_FREE, call)


def run_checks() raises:
    var flags = List[UInt8](length=MEMBER_ABI_FIXTURES, fill=UInt8(0))
    var flag_ptr = rebind[MutPointer[UInt8, MutUntrackedOrigin]](flags.unsafe_ptr())

    def _one(unit: Int) {imm flag_ptr}:
        try:
            _fixture(unit)
        except:
            flag_ptr.unsafe_store(unit, UInt8(1))

    host_parallelize(_one, MEMBER_ABI_FIXTURES)
    for unit in range(MEMBER_ABI_FIXTURES):  # driver-loop: over the fixtures' flags
        if flags[unit] != UInt8(0):
            # Raise the failing fixture's own words.
            _fixture(unit)
            raise Error("member ABI: fixture " + String(unit) + " failed on its host task and passed alone")
    _ = len(flags)
