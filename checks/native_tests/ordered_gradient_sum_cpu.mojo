# SPDX-License-Identifier: Apache-2.0
# native-check: cpu
"""THE CPU SIDE OF `ordered_sum_gradients` (lane ogs-cpu; core/ordered_gradient_sum_host.mojo), in a process with no
Python.

The CPU fold's bytes equal the host oracle of the GPU side's fold: per element, the total starts at shard 0 and
takes shards 1 .. S-1 in ascending order, each step `ftz(identical_mul_add(1, ftz(total), ftz(shard)))`, the
spelling of the GPU tree level (training/samba_ops.mojo `samba_tree_level_kernel`) written here from
checks/numerics, independently of the fold under test (core/hotpath_impl_rows.mojo `ordered_fold_step`).

1. four shards, 37 elements: 2**100 cancellations (a balanced reduction returns 0 where the left fold returns 3),
   subnormals of both signs (flushed), signed zeros, mixed magnitudes;
2. two shards (the single fold step, no running-total scan);
3. three shards of 131,077 elements (the elements split into tasks across the host's cores);
4. a non-finite input is refused in the GPU side's words;
5. a running total that overflows before the last step is refused (the GPU scans it when it uploads the last pair);
   an overflow in the last step is not (the GPU never scans the final total): the total is the infinity;
6. only slot 0 is written: the other slots keep their bytes, and the returned slot is 0.

FAIL-PROOF EDITS (each must make the named case fail): fold the shards in descending order (cases 1, 3); drop the
`ftz` of the sum in `ordered_fold_step` (case 1); skip the last `ordered_fold_f32` call (cases 1 to 3); drop the
running-total scan (case 5).
"""
from std.memory import bitcast
from std.testing import TestSuite
from checks.native_tests.testsuite_cases import assert_same_words, check_cases, require
from checks.numerics import ftz, identical_mul_add
from core.ordered_gradient_sum_host import ordered_gradient_sum_cpu

comptime CASES = 6
comptime REFUSAL = "mojolearn samba ops: non-finite accumulate parts at flat index "


def _value(shard: Int, e: Int) -> Float32:
    """A deterministic fixture word: cancellations at element 0, subnormals and signed zeros, mixed magnitudes."""
    if e == 0:
        if shard == 0:
            return bitcast[DType.float32](UInt32(0x71800000))  # 2**100
        if shard == 1:
            return Float32(1.0)
        if shard == 2:
            return bitcast[DType.float32](UInt32(0xF1800000))  # -(2**100)
        return Float32(3.0)
    if e == 1:
        return bitcast[DType.float32](UInt32(0x00000001 + shard))  # positive subnormals
    if e == 2:
        return bitcast[DType.float32](UInt32(0x80400000 + shard))  # negative subnormals
    if e == 3:
        return Float32(-0.0) if shard % 2 == 0 else Float32(0.0)
    var sign = Float32(1.0) if (e + shard) % 3 != 0 else Float32(-1.0)
    var scale = Float32(1 << ((e * 7 + shard * 3) % 24))
    return sign * scale * (Float32(1.0) + Float32((e * 13 + shard * 29) % 97) / Float32(97.0))


def _stack(shards: Int, count: Int) -> List[Float32]:
    var stack = List[Float32](capacity=shards * count)
    for shard in range(shards):  # small-loop(fixture)
        for e in range(count):  # small-loop(fixture)
            stack.append(_value(shard, e))
    return stack^


def _oracle(stack: List[Float32], shards: Int, count: Int) -> List[Float32]:
    """The GPU side's fold, serial, in its spelling."""
    var total = List[Float32](capacity=count)
    for e in range(count):  # small-loop(fixture oracle)
        var acc = stack[e]
        for shard in range(1, shards):  # small-loop(fixture oracle: ascending shard order)
            acc = ftz(identical_mul_add(Float32(1.0), ftz(acc), ftz(stack[shard * count + e])))
        total.append(acc)
    return total^


def _fold(mut stack: List[Float32], shards: Int, count: Int) raises -> List[Float32]:
    """The CPU fold of `stack` in place; slot 0 returned as a copy."""
    var slot = ordered_gradient_sum_cpu(Int(stack.unsafe_ptr()), count, shards)
    require(slot == 0, "the CPU fold returned slot " + String(slot) + ", not 0")
    var total = List[Float32](capacity=count)
    for e in range(count):  # small-loop(fixture)
        total.append(stack[e])
    return total^


def _refusal(mut stack: List[Float32], shards: Int, count: Int) -> String:
    try:
        _ = ordered_gradient_sum_cpu(Int(stack.unsafe_ptr()), count, shards)
    except problem:
        return String(problem)
    return String("")


def _against_oracle(shards: Int, count: Int, what: String) raises:
    var stack = _stack(shards, count)
    var want = _oracle(stack, shards, count)
    var got = _fold(stack, shards, count)
    assert_same_words[DType.float32](got, want, what)


def run_case(which: Int) raises:
    if which == 0:
        _against_oracle(4, 37, "four shards against the GPU fold's oracle")
        var stack = _stack(4, 37)
        var got = _fold(stack, 4, 37)
        require(got[0] == Float32(3.0), "the 2**100 cancellation: the left fold gives 3, got " + String(got[0]))
        require(bitcast[DType.uint32](got[1]) == UInt32(0), "positive subnormals flush to +0")
        require(bitcast[DType.uint32](got[2]) == UInt32(0x80000000), "negative subnormals flush to -0")
    elif which == 1:
        _against_oracle(2, 19, "two shards against the GPU fold's oracle")
    elif which == 2:
        _against_oracle(3, (1 << 17) + 5, "three shards of 131077 elements against the GPU fold's oracle")
    elif which == 3:
        var stack = _stack(3, 11)
        stack[2 * 11 + 4] = bitcast[DType.float32](UInt32(0x7FC00000))  # a NaN in shard 2, element 4
        var got = _refusal(stack, 3, 11)
        # the GPU's last pair is (running total, shard 2): the input sits in its second half
        require(got == String(REFUSAL) + String(11 + 4), "a NaN input: " + got)
    elif which == 4:
        var stack = List[Float32](length=3 * 5, fill=Float32(0.0))
        stack[1] = Float32(3.0e38)
        stack[5 + 1] = Float32(3.0e38)  # the running total of element 1 overflows at step 1
        var got = _refusal(stack, 3, 5)
        require(got == String(REFUSAL) + String(1), "an overflowed running total: " + got)
        var last = List[Float32](length=2 * 5, fill=Float32(0.0))
        last[3] = Float32(3.0e38)
        last[5 + 3] = Float32(3.0e38)  # the overflow is the last step's
        var total = _fold(last, 2, 5)
        require(bitcast[DType.uint32](total[3]) == UInt32(0x7F800000), "an overflow in the last step is the infinity")
    else:
        var stack = _stack(4, 23)
        var before = stack.copy()
        _ = _fold(stack, 4, 23)
        for i in range(23, 4 * 23):  # small-loop(fixture)
            require(bitcast[DType.uint32](stack[i]) == bitcast[DType.uint32](before[i]),
                "the CPU fold wrote slot " + String(i // 23) + " element " + String(i % 23))


def test_ordered_gradient_sum_cpu() raises:
    check_cases("ordered_gradient_sum_cpu", CASES, run_case)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
