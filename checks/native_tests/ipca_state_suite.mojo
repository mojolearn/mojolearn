# SPDX-License-Identifier: Apache-2.0
"""Typed IncrementalPCA storage, valid-zero handles and single-release state.

Mock handles are plain values here and never reach a real resident pool. The
state itself has no cleanup callback; only take_handle transfers ownership.
"""
from core.host_parallel import host_parallelize
from core.native_descriptor_abi import NativeDescriptorError
from core.native_descriptor_call import native_descriptor_report
from core.ipca_native_api import IPCANativeDriver
from core.ipca_native_state import IPCAOutputRoom, IPCAResidentState, ipca_output_shape

comptime IPCA_STATE_FIXTURES = 26


def require(ok: Bool, message: String) raises:
    if not ok:
        raise Error(message)


def ipca_state_entry(operation: Int, aa: Int, na: Int, pp: Int, np: Int, ff: Int, nf: Int,
    result: Int, context: Int, sink: NativeDescriptorError) abi("C") -> Int:
    if result == 0 or operation < 0 or operation > 2 or na != 0 or np != 0 or nf != 0:
        var problem = Error("IPCA state fixture only permits native metadata")
        native_descriptor_report(problem, context, sink)
        return 1
    MutPointer[Int, MutUntrackedOrigin](unsafe_from_address=result)[] = 1 if operation == 0 else (4 if operation == 1 else 15)
    return 0


def state(handle: Int = 73, generation: Int = 9) -> IPCAResidentState:
    return IPCAResidentState(101, generation, 211, 7, 11, handle, IPCANativeDriver(ipca_state_entry))


def move_state(var owner: IPCAResidentState) -> IPCAResidentState:
    return owner^


def move_room(var room: IPCAOutputRoom) -> IPCAOutputRoom:
    return room^


def room_addresses(room: IPCAOutputRoom) -> List[Int]:
    var outputs = room.outputs()
    var addresses: List[Int] = [outputs.components, outputs.singular, outputs.mean,
        outputs.population_variance, outputs.explained_variance, outputs.ratio]
    return addresses^


def check_room(room: IPCAOutputRoom, features: Int, written: Bool) raises:
    var square = max(features * features, 1)
    var vector = max(features, 1)
    var count = square + 5 * vector
    var addresses = room_addresses(room)
    require(len(room.values) == 6, "IPCA owns six output buffers")
    for part in range(6):  # Fixed output descriptor metadata, not data work.
        require(len(room.values[part]) == (square if part == 0 else vector), "IPCA output room capacity")
        require(addresses[part] != 0 and addresses[part] % 4 == 0
            and addresses[part] == Int(room.values[part].unsafe_ptr()), "IPCA output descriptor matches its live native owner")
        for previous in range(part):
            require(addresses[part] != addresses[previous], "IPCA output buffers do not alias")
    var failures = List[Int](length=count, fill=0)
    var fp = failures.unsafe_ptr()
    var ap = addresses.unsafe_ptr()
    def cell(index: Int) {imm fp, imm ap, imm square, imm vector, imm written}:
        var part = 0 if index < square else 1 + (index - square) // vector
        var offset = index if index < square else (index - square) % vector
        var p = Pointer[Float32, ImmUntrackedOrigin](unsafe_from_address=ap[part])
        var expected = Float32(part * 1000 + offset + 1) if written else Float32(0)
        fp.unsafe_store(index, Int(p[offset] != expected))
    host_parallelize(cell, count)
    for failed in failures:
        require(failed == 0, "IPCA owned output words remain initialized and independent")
    _ = addresses^
    _ = room


def fill_room(mut room: IPCAOutputRoom, features: Int):
    var square = max(features * features, 1)
    var vector = max(features, 1)
    var addresses = room_addresses(room)
    var ap = addresses.unsafe_ptr()
    def cell(index: Int) {imm ap, imm square, imm vector}:
        var part = 0 if index < square else 1 + (index - square) // vector
        var offset = index if index < square else (index - square) % vector
        var p = MutPointer[Float32, MutUntrackedOrigin](unsafe_from_address=ap[part])
        p.unsafe_store(offset, Float32(part * 1000 + offset + 1))
    host_parallelize(cell, square + 5 * vector)
    _ = addresses^
    _ = room


def fixture(unit: Int) raises:
    if unit < 4:
        var rows = 0 if unit == 3 else 3
        var cols = 0 if unit == 2 else 5
        var vector = unit == 1 or unit == 3
        var shape = ipca_output_shape(rows, cols, vector)
        require(len(shape) == (1 if vector else 2) and shape[0] == rows,
                "IPCA vector/matrix rank and row extent")
        if not vector:
            require(shape[1] == cols, "IPCA zero matrix columns remain a dimension")
        return
    if unit < 10:
        var features = 0 if unit == 4 else (1 if unit == 5 else (17 if unit == 7 else 3))
        var room = IPCAOutputRoom(features)
        check_room(room, features, False)
        if unit >= 8:
            fill_room(room, features)
            if unit == 9:
                var address = room.outputs().components
                var moved = move_room(room^)
                require(moved.outputs().components == address, "IPCA move retains output allocation")
                check_room(moved, features, True)
            else:
                check_room(room, features, True)
        return
    var owner = state(0 if unit == 10 or unit == 17 or unit == 25 else (-1 if unit == 12 or unit == 19 else 73),
        -1 if unit == 15 or unit == 24 or unit == 25 else 9)
    if unit == 10 or unit == 11 or unit == 12 or unit == 25:
        require(owner.matches(101, 211) == (unit != 12), "IPCA zero and positive handles are live; only minus one is empty")
        if unit == 25:
            require(owner.generation == -1, "IPCA match does not adopt a pending generation")
    elif unit == 13:
        require(not owner.matches(103, 211) and owner.handle == 73, "IPCA provider mismatch does not consume handle")
    elif unit == 14:
        require(not owner.matches(101, 223) and owner.handle == 73, "IPCA source mismatch does not consume handle")
    elif unit == 15:
        require(owner.accept_generation(45) and owner.generation == 45, "IPCA pending generation adopts first query")
        require(owner.accept_generation(45) and not owner.accept_generation(46)
            and owner.generation == 45, "IPCA adopted generation rejects later mutation without replacement")
    elif unit == 16:
        require(owner.accept_generation(9) and not owner.accept_generation(10)
            and owner.generation == 9, "IPCA fixed generation matches exactly")
    elif unit == 17 or unit == 18 or unit == 19:
        var expected = 0 if unit == 17 else (-1 if unit == 19 else 73)
        require(owner.take_handle() == expected and owner.take_handle() == -1
            and owner.handle == -1, "IPCA valid-zero/positive/empty handles transfer at most once")
    elif unit == 20:
        _ = owner.take_handle()
        require(not owner.matches(101, 211), "IPCA consumed owner cannot admit resident state")
    elif unit == 21:
        var moved = move_state(owner^)
        require(moved.matches(101, 211) and moved.generation == 9 and moved.rows == 7
            and moved.cols == 11, "IPCA move preserves native owner identity and shape")
        require(moved.driver.metadata(1) == 4 and moved.driver.metadata(2) == 15,
                "IPCA move preserves native descriptor driver")
        require(moved.take_handle() == 73 and moved.take_handle() == -1, "IPCA moved owner transfers once")
    elif unit == 22:
        var replacement = state(79, -1)
        _ = owner.take_handle()
        require(replacement.accept_generation(17) and replacement.matches(101, 211)
            and replacement.handle == 79 and owner.take_handle() == -1,
            "IPCA old cleanup cannot consume replacement owner")
    elif unit == 23:
        _ = owner.take_handle()
        require(owner.connection == 101 and owner.generation == 9 and owner.source == 211
            and owner.rows == 7 and owner.cols == 11 and owner.driver.metadata(0) == 1,
            "IPCA handle transfer changes no other native field")
    else:
        require(owner.accept_generation(0) and owner.generation == 0 and owner.accept_generation(0)
            and not owner.accept_generation(1), "IPCA zero generation is adopted as a real value")


def run_checks() raises:
    var failures = List[Int](length=IPCA_STATE_FIXTURES, fill=0)
    var fp = failures.unsafe_ptr()
    def task(unit: Int) {imm fp}:
        try:
            fixture(unit)
        except error:
            fp.unsafe_store(unit, 1)
            print("FAIL ipca_state fixture", unit, String(error))
    host_parallelize(task, IPCA_STATE_FIXTURES)
    var failed = 0
    for value in failures:
        failed += value
    require(failed == 0, "ipca_state failed " + String(failed) + " fixtures")
    print("PASS ipca_state:", IPCA_STATE_FIXTURES, "native storage and single-release state fixtures")
