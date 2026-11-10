# SPDX-License-Identifier: Apache-2.0
"""Raw-bit optimizer gathers and native ownership; independent address oracles.

Small fixtures state their expected physical element order literally. Large
padded/reversed matrices use an affine row/column oracle, independent of the
producer's coordinate carry. Checks do not time kernels or establish identity.
"""
from std.memory import bitcast
from core.host_parallel import host_parallelize
from core.host_predict_threads import host_predict_task_count_work, host_predict_chunk
from bindings.tensor_view import TensorView
from training.optimizer_input_plan import optimizer_contiguous_address

comptime INPUT_COPY_FIXTURES = 16
comptime BYTE_GUARD = UInt8(0xA5)
comptime WORD_GUARD = UInt32(0xAABBCCDD)
comptime MUTATED = UInt32(0x3F123456)


def require(ok: Bool, message: String) raises:
    if not ok:
        raise Error(message)


def fixture_word(index: Int) -> UInt32:
    # Fixture data only: includes both zero signs and positive/negative quiet
    # NaN payloads. No floating arithmetic may participate in the copy.
    if index % 31 == 0:
        return 0x80000000
    if index % 31 == 1:
        return 0
    if index % 31 == 2:
        return 0x7FC00042
    if index % 31 == 3:
        return 0xFFC12345
    return UInt32(0x3F000000) | UInt32((index * 8191 + 17) & 0x007FFFFF)


def raw_word(address: Int) -> UInt32:
    var p = Pointer[UInt8, ImmUntrackedOrigin](unsafe_from_address=address)
    return UInt32(p[0]) | (UInt32(p[1]) << 8) | (UInt32(p[2]) << 16) | (UInt32(p[3]) << 24)


def output_owner(keep: List[List[Float32]], address: Int, count: Int, expected_owners: Int) raises:
    require(len(keep) == expected_owners, "gather allocated the wrong number of native owners")
    require(len(keep[0]) == 3 and bitcast[DType.uint32](keep[0][0]) == WORD_GUARD and
        bitcast[DType.uint32](keep[0][1]) == 0x80000000 and bitcast[DType.uint32](keep[0][2]) == WORD_GUARD,
        "gather changed the pre-existing guarded output owner")
    if expected_owners > 1:
        require(Int(keep[1].unsafe_ptr()) == address and len(keep[1]) == max(count, 1),
            "gather pointer does not belong to its sole retained output allocation")


def guarded_owners() -> List[List[Float32]]:
    var keep = List[List[Float32]]()
    var guard: List[Float32] = [bitcast[DType.float32](WORD_GUARD), bitcast[DType.float32](UInt32(0x80000000)), bitcast[DType.float32](WORD_GUARD)]
    keep.append(guard^)
    return keep^


def small_fixture(fixture_id: Int) raises:
    var source = List[UInt8](length=72, fill=BYTE_GUARD)
    var start = 5 if fixture_id >= 12 else 4
    var sp = source.unsafe_ptr()
    def prepare(index: Int) {imm sp, imm start}:
        var word = fixture_word(index)
        for byte in range(4):
            sp[start + 4 * index + byte] = UInt8((word >> UInt32(8 * byte)) & 255)
    host_parallelize(prepare, 16)
    var original = source.copy()
    var view = TensorView()
    view.addr = Int(source.unsafe_ptr()) + start
    view.itemsize = 4
    view.fmt = "f"
    view.dtype = "float32"
    view.shape = [2, 3]
    view.strides = [12, 4]
    view.has_strides = True
    view.c_contig = fixture_id == 0 or fixture_id == 8 or fixture_id == 10
    var order: List[Int] = [0, 1, 2, 3, 4, 5]
    if fixture_id == 1 or fixture_id == 13:
        view.strides = [4, 8]
        order = [0, 2, 4, 1, 3, 5]
    elif fixture_id == 2:
        view.addr += 8
        view.strides = [12, -4]
        order = [2, 1, 0, 5, 4, 3]
    elif fixture_id == 3:
        view.addr += 12
        view.strides = [-12, 4]
        order = [3, 4, 5, 0, 1, 2]
    elif fixture_id == 4:
        view.strides = [4, 0]
        order = [0, 0, 0, 1, 1, 1]
    elif fixture_id == 5:
        view.strides = [0, 4]
        order = [0, 1, 2, 0, 1, 2]
    elif fixture_id == 6:
        view.shape = [2, 2, 3]
        view.strides = [4, 24, 8]
        order = [0, 2, 4, 6, 8, 10, 1, 3, 5, 7, 9, 11]
    elif fixture_id == 7:
        view.shape = [2, 1, 3]
        view.strides = [20, 0, 4]
        order = [0, 1, 2, 5, 6, 7]
    elif fixture_id == 8 or fixture_id == 9:
        view.shape = []
        view.strides = []
        order = [0]
    elif fixture_id == 10 or fixture_id == 11:
        view.shape = [2, 0, 3]
        view.strides = [12, 12, 4]
        view.addr = 0
        order = []
    elif fixture_id == 12:
        view.strides = [16, 4]
        order = [0, 1, 2, 4, 5, 6]
    view.nbytes = 4 * len(order)
    var keep = guarded_owners()
    var address = optimizer_contiguous_address(view, keep)
    if view.c_contig:
        require(address == view.addr, "contiguous gather did not borrow its exact input")
        output_owner(keep, address, len(order), 1)
    else:
        require(address != 0 and address != view.addr, "strided gather did not acquire disjoint native storage")
        output_owner(keep, address, len(order), 2)
    for i in range(len(order)):  # Small literal-fixture cells; fixtures run in parallel.
        require(raw_word(address + 4 * i) == fixture_word(order[i]), "small gather changed physical order or raw bits")
    for i in range(len(source)):
        require(source[i] == original[i], "gather mutated its source or source guards")
    if len(order):
        # Change source bytes after the synchronous gather. Copied output must
        # remain independent, while the contiguous borrowed path must alias.
        def mutate(index: Int) {imm sp, imm start}:
            for byte in range(4):
                sp[start + 4 * index + byte] = UInt8((MUTATED >> UInt32(8 * byte)) & 255)
        host_parallelize(mutate, 16)
        for i in range(len(order)):
            var wanted = MUTATED if view.c_contig else fixture_word(order[i])
            require(raw_word(address + 4 * i) == wanted, "borrowed/copied ownership semantics differ")
        if not view.c_contig:
            var next_address = optimizer_contiguous_address(view, keep)
            require(len(keep) == 3 and next_address != address, "second gather reused a live output owner")
            output_owner(keep, address, len(order), 3)
            for i in range(len(order)):
                require(raw_word(address + 4 * i) == fixture_word(order[i]), "growing keep invalidated an earlier result")
                require(raw_word(next_address + 4 * i) == MUTATED, "second retained output has stale source bits")
    _ = original
    _ = source
    _ = keep


def large_fixture(reverse: Bool) raises:
    comptime ROWS = 509
    comptime COLS = 521
    comptime PITCH = COLS + 3
    comptime COUNT = ROWS * COLS  # 265189 cells, over the requested 262k boundary.
    comptime STORAGE = ROWS * PITCH
    var source = List[UInt32](length=STORAGE + 4, fill=WORD_GUARD)
    var sp = source.unsafe_ptr()
    var tasks = host_predict_task_count_work(COUNT, COUNT * 4)
    var chunk = host_predict_chunk(COUNT, tasks)
    if tasks > 1:
        require(chunk % COLS != 0, "large fixture did not cut a row at the worker boundary")
    var source_tasks = host_predict_task_count_work(STORAGE, STORAGE * 4)
    var source_chunk = host_predict_chunk(STORAGE, source_tasks)
    def prepare(unit: Int) {imm sp, imm source_chunk}:
        for index in range(unit * source_chunk, min(STORAGE, (unit + 1) * source_chunk)):
            if index % PITCH < COLS:
                sp[index + 2] = fixture_word(index)
    host_parallelize(prepare, source_tasks)
    var view = TensorView()
    view.addr = Int(source.unsafe_ptr()) + 8
    if reverse:
        view.addr += 4 * ((ROWS - 1) * PITCH + COLS - 1)
    view.nbytes = COUNT * 4
    view.itemsize = 4
    view.fmt = "f"
    view.dtype = "float32"
    view.shape = [ROWS, COLS]
    view.strides = [-4 * PITCH, -4] if reverse else [4 * PITCH, 4]
    view.has_strides = True
    view.c_contig = False
    var keep = guarded_owners()
    var address = optimizer_contiguous_address(view, keep)
    require(address != view.addr, "large strided output aliases source")
    output_owner(keep, address, COUNT, 2)
    var rp = Pointer[UInt32, ImmUntrackedOrigin](unsafe_from_address=address)
    var failures = List[Int](length=tasks + source_tasks, fill=0)
    var fp = failures.unsafe_ptr()
    def check_output(unit: Int) {imm rp, imm fp, imm chunk, imm reverse}:
        var bad = 0
        for index in range(unit * chunk, min(COUNT, (unit + 1) * chunk)):
            var row = index // COLS
            var col = index % COLS
            # Independent affine addressing; no producer coordinate carry.
            var physical = (ROWS - 1 - row) * PITCH + COLS - 1 - col if reverse else row * PITCH + col
            if rp[index] != fixture_word(physical):
                bad = 1
        fp.unsafe_store(unit, bad)
    def check_source(unit: Int) {imm sp, imm fp, imm source_chunk, imm tasks}:
        var bad = 0
        for index in range(unit * source_chunk, min(STORAGE, (unit + 1) * source_chunk)):
            var wanted = fixture_word(index) if index % PITCH < COLS else WORD_GUARD
            if sp[index + 2] != wanted:
                bad = 1
        fp.unsafe_store(tasks + unit, bad)
    host_parallelize(check_output, tasks)
    host_parallelize(check_source, source_tasks)
    require(source[0] == WORD_GUARD and source[1] == WORD_GUARD and
        source[STORAGE + 2] == WORD_GUARD and source[STORAGE + 3] == WORD_GUARD, "large source outer guards")
    var failed = 0
    for value in failures:
        failed += value
    require(failed == 0, "large gather affine bits/source padding differs")
    # Overwrite all source data after copying, including raw NaN payloads.
    def mutate(unit: Int) {imm sp, imm source_chunk}:
        for index in range(unit * source_chunk, min(STORAGE, (unit + 1) * source_chunk)):
            if index % PITCH < COLS:
                sp[index + 2] = MUTATED
    host_parallelize(mutate, source_tasks)
    host_parallelize(check_output, tasks)
    for unit in range(tasks):
        require(failures[unit] == 0, "large retained output changed after source overwrite")
    output_owner(keep, address, COUNT, 2)
    print("optimizer_input_copy large fixture workers", tasks, "cells", COUNT, "reverse", reverse)
    _ = source
    _ = keep


def run_fixture(fixture_id: Int) raises:
    if fixture_id < 14:
        small_fixture(fixture_id)
    else:
        large_fixture(fixture_id == 15)


def run_checks() raises:
    var failures = List[Int](length=INPUT_COPY_FIXTURES, fill=0)
    var fp = failures.unsafe_ptr()
    def task(fixture_id: Int) {imm fp}:
        try:
            run_fixture(fixture_id)
        except error:
            fp.unsafe_store(fixture_id, 1)
            print("FAIL optimizer_input_copy fixture", fixture_id, String(error))
    host_parallelize(task, INPUT_COPY_FIXTURES)
    var failed = 0
    for value in failures:
        failed += value
    require(failed == 0, "optimizer_input_copy failed " + String(failed) + " fixtures")
    print("PASS optimizer_input_copy:", INPUT_COPY_FIXTURES, "native raw-bit/owner fixtures")
