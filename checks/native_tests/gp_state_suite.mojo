# SPDX-License-Identifier: Apache-2.0
"""Owned GP storage contracts, with independent fixtures across host cores.

These checks exercise the typed metadata and scratch owners, not GP numerical
kernels. Raw spans are borrowed only while their named List owners remain live.
No pointer to a borrowed struct field is captured by a parallel closure.
"""
from std.memory import bitcast
from core.host_parallel import host_parallelize
from core.gp_native_api import GPKernelView, GPQueryResult
from core.gp_native_storage import gp_native_ints, gp_native_f64, GPKernelStorage, GPRFitScratch, GPRGradientScratch, GPOptimizerScratch, GPCMemberTables

comptime GP_STATE_FIXTURES = 32


def require(ok: Bool, message: String) raises:
    if not ok:
        raise Error(message)


def kernel_storage() -> GPKernelStorage:
    var value = GPKernelStorage()
    value.kinds = [0, 2, 5, 1, 4]
    value.parameters = [1.5, 0.0, 0.0, 0.2, 0.0]
    value.lengths = [0, 3, 0, 0, 0]
    value.scales = [0.8, 1.2, 2.0]
    value.finish()
    return value^


def check_kernel(value: GPKernelStorage) raises:
    var view = value.view()
    require(view.nodes == 5 and view.scale_count == 3, "GP kernel node and ARD extents")
    require(view.kinds == Int(value.kinds.unsafe_ptr()) and view.parameters == Int(value.parameters.unsafe_ptr())
        and view.lengths == Int(value.lengths.unsafe_ptr()) and view.scales == Int(value.scales.unsafe_ptr()),
        "GP kernel view borrows all four live owners")
    require(value.kinds[0] == Int32(0) and value.kinds[4] == Int32(4)
        and value.parameters[0] == Float32(1.5) and value.parameters[3] == Float32(0.2)
        and value.lengths[1] == Int32(3) and value.scales[2] == Float32(2.0), "GP kernel retained metadata")
    _ = value


def kernel_case(unit: Int) raises:
    if unit == 0 or unit == 1:
        var value = GPKernelStorage()
        if unit == 1:
            value.kinds = [0]
            value.parameters = [2.0]
            value.lengths = [0]
        value.finish()
        var view = value.view()
        require(view.nodes == unit and view.scale_count == 0 and len(value.scales) == 1
            and value.scales[0] == Float32(1.0) and view.scales == Int(value.scales.unsafe_ptr()),
            "GP no-length-scale view has one owned unused word and zero logical scales")
        _ = value^
        return
    var value = kernel_storage()
    if unit == 2:
        check_kernel(value)
    elif unit == 3:
        var copied = value.copy()
        var before = value.view()
        var after = copied.view()
        require(before.kinds != after.kinds and before.parameters != after.parameters
            and before.lengths != after.lengths and before.scales != after.scales,
            "GP kernel copy owns independent metadata allocations")
        value.kinds[0] = 1
        value.parameters[0] = 7.0
        value.lengths[1] = 1
        value.scales[2] = 11.0
        check_kernel(copied)
        _ = copied^
    else:
        var before = value.view()
        var moved = value^
        var after = moved.view()
        require(before.kinds == after.kinds and before.parameters == after.parameters
            and before.lengths == after.lengths and before.scales == after.scales,
            "GP kernel move retains all allocation identities")
        check_kernel(moved)
        _ = moved^


def fit_scratch_case() raises:
    var scratch = GPRFitScratch()
    require(len(scratch.scalars) == 5, "GP fit scalar extent")
    var zero = scratch.result()
    require(zero.info == 0 and zero.panel_width == 0 and zero.logdet == 0.0
        and zero.y_dot_alpha == 0.0 and zero.likelihood == 0.0, "GP fit scalar initialization")
    var address = scratch.address()
    var words = MutPointer[Float64, MutUntrackedOrigin](unsafe_from_address=address)
    words[0] = 3.0
    words[1] = 32.0
    words[2] = 1.0000000000000002
    words[3] = bitcast[DType.float64](UInt64(0x8000000000000000))
    words[4] = -71.125
    var result = scratch.result()
    require(result.info == 3 and result.panel_width == 32 and result.logdet == Float64(1.0000000000000002)
        and bitcast[DType.uint64](result.y_dot_alpha) == UInt64(0x8000000000000000)
        and result.likelihood == Float64(-71.125), "GP named fit result scalar order and binary64 precision")
    _ = scratch^


def gradient_case(unit: Int) raises:
    var counts: List[Int] = [0, 1, 7, 17]
    var count = counts[unit]
    var scratch = GPRGradientScratch(count)
    require(len(scratch.gradient) == max(count, 1) and len(scratch.scalars) == 2,
        "GP gradient empty/one/multiple parameter storage extents")
    require(scratch.gradient[0] == 0.0 and scratch.gradient[len(scratch.gradient) - 1] == 0.0
        and scratch.scalars[0] == 0.0 and scratch.scalars[1] == 0.0, "GP gradient initialization")
    scratch.gradient[len(scratch.gradient) - 1] = 1.0000000000000002
    require(scratch.gradient[len(scratch.gradient) - 1] == Float64(1.0000000000000002),
        "GP gradients retain binary64 storage")


def optimizer_case(unit: Int) raises:
    var counts: List[Int] = [0, 1, 7, 17]
    var starts: List[Int] = [0, 1, 3, 5]
    var count = counts[unit]
    var runs = starts[unit]
    var scratch = GPOptimizerScratch(count, runs)
    require(len(scratch.theta) == max(count, 1) and len(scratch.values) == max(count, 1)
        and len(scratch.runs) == max(4 * runs, 1), "GP optimizer parameter and four-word run extents")
    require(scratch.theta[0] == 0.0 and scratch.values[len(scratch.values) - 1] == 0.0
        and scratch.runs[len(scratch.runs) - 1] == 0.0, "GP optimizer initialization")
    scratch.theta[0] = 0.125
    scratch.values[0] = 0.25
    scratch.runs[0] = 0.5
    require(scratch.theta[0] == 0.125 and scratch.values[0] == 0.25 and scratch.runs[0] == 0.5,
        "GP optimizer scratch spans do not alias")


def member_case(unit: Int) raises:
    var counts: List[Int] = [0, 1, 3, 7]
    var count = counts[unit]
    var tables = GPCMemberTables(count)
    require(len(tables.columns) == count and len(tables.targets) == count and len(tables.factors) == count
        and len(tables.pis) == count and len(tables.wsrs) == count and len(tables.scalars) == count
        and len(tables.scalar_storage) == 3 * count, "GP OVR member metadata extents")
    tables.prepare_scalars()
    for member in range(count):  # driver-loop: metadata for each fitted OVR member
        require(tables.columns[member] == 0 and tables.targets[member] == 0 and tables.factors[member] == 0
            and tables.pis[member] == 0 and tables.wsrs[member] == 0,
            "GP OVR arrays stay unbound until caller supplies owners")
        require(tables.scalars[member] == Int(tables.scalar_storage.unsafe_ptr() + 3 * member),
            "GP OVR per-member three-word scalar addresses")
        var words = MutPointer[Float64, MutUntrackedOrigin](unsafe_from_address=tables.scalars[member])
        words[0] = Float64(100 + member)
        words[1] = Float64(200 + member)
        words[2] = Float64(300 + member)
    for member in range(count):  # driver-loop: independent metadata slices, no numerical training work
        require(tables.scalar_storage[3 * member] == Float64(100 + member)
            and tables.scalar_storage[3 * member + 1] == Float64(200 + member)
            and tables.scalar_storage[3 * member + 2] == Float64(300 + member),
            "GP OVR scalar slices remain separate")
    _ = tables^


def span_case(unit: Int) raises:
    if unit == 0:
        var copied = gp_native_ints(0, 0)
        require(len(copied) == 0, "GP empty integer metadata admits null source")
    elif unit == 1:
        var source: List[Int] = [0, -7, 9007199254740993, 9223372036854775807]
        var copied = gp_native_ints(Int(source.unsafe_ptr()), 4)
        require(Int(source.unsafe_ptr()) != Int(copied.unsafe_ptr()), "GP integer span owns copy")
        source[2] = 11
        require(copied[0] == 0 and copied[1] == -7 and copied[2] == 9007199254740993
            and copied[3] == 9223372036854775807, "GP integer metadata retains full native word width")
        _ = source^
    elif unit == 2:
        var copied = gp_native_f64(0, 0)
        require(len(copied) == 0, "GP empty binary64 metadata admits null source")
    else:
        var source: List[Float64] = [bitcast[DType.float64](UInt64(0x8000000000000000)),
            1.0000000000000002, bitcast[DType.float64](UInt64(0x7ff0000000000000)),
            bitcast[DType.float64](UInt64(0x7ff8000000000123))]
        var copied = gp_native_f64(Int(source.unsafe_ptr()), 4)
        require(Int(source.unsafe_ptr()) != Int(copied.unsafe_ptr()), "GP binary64 span owns copy")
        source[1] = 11.0
        require(bitcast[DType.uint64](copied[0]) == UInt64(0x8000000000000000)
            and copied[1] == Float64(1.0000000000000002)
            and bitcast[DType.uint64](copied[2]) == UInt64(0x7ff0000000000000)
            and bitcast[DType.uint64](copied[3]) == UInt64(0x7ff8000000000123),
            "GP binary64 metadata copies words without conversion")
        _ = source^


def refusal_case(unit: Int) raises:
    var address = 1 if unit % 3 == 0 else 0
    var count = -1 if unit % 3 == 0 else (1 if unit % 3 == 1 else 17)
    var refused = False
    var expected = "GP native metadata: invalid integer span" if unit < 3 else "GP native metadata: invalid float64 span"
    try:
        if unit < 3:
            _ = gp_native_ints(address, count)
        else:
            _ = gp_native_f64(address, count)
    except error:
        require(String(error) == expected, "GP invalid metadata span refusal")
        refused = True
    require(refused, "GP invalid span refused before dereference")


def fixture(unit: Int) raises:
    if unit < 5:
        kernel_case(unit)
    elif unit == 5:
        fit_scratch_case()
    elif unit < 10:
        gradient_case(unit - 6)
    elif unit < 14:
        optimizer_case(unit - 10)
    elif unit < 18:
        member_case(unit - 14)
    elif unit < 22:
        span_case(unit - 18)
    elif unit < 28:
        refusal_case(unit - 22)
    elif unit == 28:
        var result = GPQueryResult(9007199254740993, 1, -7)
        var copied = result.copy()
        require(copied.value == 9007199254740993 and copied.finite == 1 and copied.range_status == -7,
            "GP query result words remain exact native integers")
    elif unit == 29:
        var view = GPKernelView(9007199254740993, 37, 41, 43, 47, 53)
        var copied = view.copy()
        require(copied.kinds == 9007199254740993 and copied.parameters == 37 and copied.lengths == 41
            and copied.scales == 43 and copied.nodes == 47 and copied.scale_count == 53,
            "GP kernel view copies all pointer and extent words without conversion")
    elif unit == 30:
        var scratch = GPRFitScratch()
        scratch.scalars[4] = -71.125
        var before = scratch.address()
        var moved = scratch^
        require(moved.address() == before and moved.result().likelihood == Float64(-71.125),
            "GP fit scratch move keeps address and named result")
        _ = moved^
    else:
        var tables = GPCMemberTables(3)
        tables.prepare_scalars()
        var before = tables.scalars[2]
        tables.scalar_storage[8] = 29.0
        var moved = tables^
        require(moved.scalars[2] == before and moved.scalars[2] == Int(moved.scalar_storage.unsafe_ptr() + 6)
            and moved.scalar_storage[8] == 29.0, "GP member-table move retains interior scalar addresses")
        _ = moved^


def run_checks() raises:
    var failures = List[Int](length=GP_STATE_FIXTURES, fill=0)
    var fp = failures.unsafe_ptr()
    def task(unit: Int) {imm fp}:
        try:
            fixture(unit)
        except error:
            fp.unsafe_store(unit, 1)
            print("FAIL gp_state fixture", unit, String(error))
    host_parallelize(task, GP_STATE_FIXTURES)
    var failed = 0
    for value in failures:
        failed += value
    require(failed == 0, "gp_state failed " + String(failed) + " fixtures")
    print("PASS gp_state:", GP_STATE_FIXTURES, "native ownership/storage/span fixtures")
