# SPDX-License-Identifier: Apache-2.0
"""Regression contracts for admitted optimizer moment/checkpoint storage.

Plans inspect metadata only. Every successful span must preserve its exact
borrowed address; every malformed span must refuse before any upload/read.
Independent contracts run in parallel, with no Python objects or imports.
"""
from core.host_parallel import host_parallelize
from bindings.tensor_view import TensorView
from training.optimizer_public_common import PublicStep
from training.optimizer_state_plan import (
    OptimizerStateSnapshot, optimizer_state_count, optimizer_state_span,
    optimizer_checkpoint_spans, optimizer_state_download_needed,
    optimizer_host_moment_plan, optimizer_checkpoint_input,
)

comptime STATE_PLAN_FIXTURES = 32


def require(ok: Bool, message: String) raises:
    if not ok:
        raise Error(message)


def view(address: Int, count: Int, integer: Bool = False) -> TensorView:
    var result = TensorView()
    result.addr = address
    result.nbytes = count * 4
    result.itemsize = 4
    result.fmt = "i" if integer else "f"
    result.dtype = "int32" if integer else "float32"
    result.shape = [count]
    result.strides = [4]
    result.has_strides = True
    return result^


def span_fixture(fixture_id: Int) raises:
    var owner: List[UInt32] = [0xAABBCCDD, 0x80000000, 0x3F800000, 0x7FC00042, 0, 0xAABBCCDD]
    var address = Int(owner.unsafe_ptr()) + 4
    var source = view(address, 4)
    var count = 4
    var integer = False
    var expected = String("")
    if fixture_id < 4:
        var formats: List[String] = ["f", "@f", "=f", "<f"]
        source.fmt = formats[fixture_id]
    elif fixture_id == 4:
        source.fmt = "i"
        integer = True
    elif fixture_id == 5:
        source = view(0, 0)
        source.shape = [0, 9223372036854775807, 9223372036854775807]
        count = 0
    elif fixture_id == 6:
        source = view(address, 1)
        source.shape = []
        source.strides = []
        count = 1
    elif fixture_id == 7:
        source.shape = [2, 2]
        source.strides = [8, 4]
    elif fixture_id == 8:
        source.fmt = "d"
        source.itemsize = 8
        expected = "TypeError: optimizer exp_avg requires a native float32 buffer"
    elif fixture_id == 9:
        integer = True
        expected = "TypeError: optimizer exp_avg requires a native int32 buffer"
    elif fixture_id == 10:
        source.nbytes = 12
        expected = "ValueError: optimizer exp_avg does not match the optimizer registry"
    elif fixture_id == 11:
        source.shape = [3]
        expected = "ValueError: optimizer exp_avg does not match the optimizer registry"
    elif fixture_id == 12:
        source.readonly = True
        expected = "ValueError: optimizer exp_avg requires contiguous writable native storage"
    elif fixture_id == 13:
        source.strides = [-4]
        source.c_contig = False
        expected = "ValueError: optimizer exp_avg requires contiguous writable native storage"
    elif fixture_id == 14:
        source.addr = 0
        expected = "ValueError: optimizer exp_avg requires contiguous writable native storage"
    elif fixture_id == 15 or fixture_id == 16:
        count = -1 if fixture_id == 15 else 2305843009213693952
        expected = "ValueError: optimizer state size is out of range"
    elif fixture_id == 17 or fixture_id == 18:
        source.shape = [-1] if fixture_id == 17 else [9223372036854775807, 2]
        expected = "ValueError: optimizer state shape is out of range"
    else:
        source.itemsize = 8
        expected = "TypeError: optimizer exp_avg requires a native float32 buffer"
    if expected.byte_length() == 0:
        require(optimizer_state_span(source, count, "exp_avg", integer) == source.addr,
            "accepted moment span changed borrowed address")
        require(optimizer_state_count(source) == count, "accepted state shape product differs")
    else:
        var refused = False
        try:
            _ = optimizer_state_span(source, count, "exp_avg", integer)
        except error:
            refused = True
            require(String(error) == expected, "native state span refusal differs: " + String(error))
        require(refused, "unsafe optimizer state span was admitted")
    require(owner[0] == 0xAABBCCDD and owner[5] == 0xAABBCCDD and owner[1] == 0x80000000 and
        owner[2] == 0x3F800000 and owner[3] == 0x7FC00042 and owner[4] == 0,
        "metadata-only state plan touched source or guard words")
    _ = owner


def checkpoint_fixture(fixture_id: Int) raises:
    var first_owner: List[Float32] = [1, 2, 3, 4]
    var second_owner: List[Float32] = [5, 6, 7, 8]
    var initialized_owner: List[Int32] = [1, 0]
    var first = view(Int(first_owner.unsafe_ptr()), 4)
    var second = view(Int(second_owner.unsafe_ptr()), 4)
    var initialized = view(Int(initialized_owner.unsafe_ptr()), 1, True)
    var expected = String("")
    if fixture_id == 1:
        first.shape = [3]
        first.nbytes = 12
        expected = "ValueError: mojolearn.AdamW.load_state_dict: exp_avg holds 3 floats, this optimizer's registry is 4"
    elif fixture_id == 2:
        second.shape = [3]
        second.nbytes = 12
        expected = "ValueError: mojolearn.AdamW.load_state_dict: exp_avg_sq holds 3 floats, this optimizer's registry is 4"
    elif fixture_id == 3:
        initialized.shape = [2]
        initialized.nbytes = 8
        expected = "ValueError: buf_initialized does not match the optimizer parameter registry"
    elif fixture_id == 4:
        initialized.fmt = "f"
        expected = "TypeError: optimizer buf_initialized requires a native int32 buffer"
    elif fixture_id == 5:
        second.readonly = True
        expected = "ValueError: optimizer exp_avg_sq requires contiguous writable native storage"
    var refused = False
    try:
        optimizer_checkpoint_spans(first, second, initialized, 4, 1, "AdamW")
    except error:
        refused = True
        require(String(error) == expected, "checkpoint metadata refusal differs")
    require(refused == (expected.byte_length() != 0), "checkpoint unsafe span acceptance differs")
    require(first_owner[0] == 1 and first_owner[3] == 4 and second_owner[0] == 5 and second_owner[3] == 8 and
        initialized_owner[0] == 1 and initialized_owner[1] == 0, "checkpoint plan changed source state")
    _ = first_owner
    _ = second_owner
    _ = initialized_owner


def ownership_fixture(fixture_id: Int) raises:
    var first: List[Float32] = [3, 4]
    var second: List[Float32] = [5, 6]
    var handle = -1 if fixture_id < 2 else 7
    var fresh = fixture_id % 2 != 0
    var resident = -1 if fixture_id == 0 else (0 if fixture_id == 1 else 1)
    var snapshot = OptimizerStateSnapshot(handle, resident, fresh, Int(first.unsafe_ptr()), Int(second.unsafe_ptr()))
    var request = PublicStep()
    request.j = 9
    request.t = 17
    request.n_total = 2
    snapshot.apply(request)
    require(request.handle == handle and request.resident_wanted == resident and request.pristine == fresh and
        request.m_addr == Int(first.unsafe_ptr()) and request.v_addr == Int(second.unsafe_ptr()), "typed state apply lost owner fields")
    require(request.j == 9 and request.t == 17 and request.n_total == 2, "typed state apply overwrote unrelated request fields")
    require(optimizer_state_download_needed(handle, fresh) == (handle >= 0 and not fresh), "state freshness decision differs")
    var plan = optimizer_host_moment_plan(2, fixture_id >= 2, fresh)
    require(plan[0] == (fixture_id < 2) and plan[1] == (not fresh), "missing native moment allocation plan differs")
    optimizer_checkpoint_input(1, 2)
    _ = first
    _ = second


def run_fixture(fixture_id: Int) raises:
    if fixture_id < 20:
        span_fixture(fixture_id)
    elif fixture_id < 26:
        checkpoint_fixture(fixture_id - 20)
    elif fixture_id < 30:
        ownership_fixture(fixture_id - 26)
    else:
        var refused = False
        try:
            optimizer_checkpoint_input(2 if fixture_id == 30 else 1, 2 if fixture_id == 30 else 0)
        except error:
            refused = True
            require(String(error) == "ValueError: optimizer checkpoint arrays must be nonempty one-dimensional buffers",
                "checkpoint input rank/count refusal differs")
        require(refused, "invalid checkpoint input shape accepted")


def main() raises:
    var failures = List[Int](length=STATE_PLAN_FIXTURES, fill=0)
    var fp = failures.unsafe_ptr()
    def task(fixture_id: Int) {imm fp}:
        try:
            run_fixture(fixture_id)
        except error:
            fp.unsafe_store(fixture_id, 1)
            print("FAIL optimizer_state_plan fixture", fixture_id, String(error))
    host_parallelize(task, STATE_PLAN_FIXTURES)
    var failed = 0
    for value in failures:
        failed += value
    require(failed == 0, "optimizer_state_plan failed " + String(failed) + " fixtures")
    print("PASS optimizer_state_plan:", STATE_PLAN_FIXTURES, "parallel span/owner fixtures")
