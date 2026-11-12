# SPDX-License-Identifier: Apache-2.0
# Native-only invocation; Python ownership belongs in providers/python.
from std.sys import is_defined
from core.native_string import unsafe_string_from_utf8_cstr
from core.native_descriptor_abi import NativeDescriptorCall, NativeDescriptorError
from core.native_call_output import NativeCallOutput, NativeCallWorkspaceLease


def native_descriptor_error(context: Int, address: Int, size: Int) abi("C") -> NoneType:
    var out = MutPointer[List[UInt8], MutUntrackedOrigin](unsafe_from_address=context)
    var src = MutPointer[UInt8, MutUntrackedOrigin](unsafe_from_address=address)
    for i in range(size):  # driver-loop: copy complete native refusal text
        out[].append(src[i])
    return None


def native_descriptor_call(
    entry: NativeDescriptorCall, operation: Int, addresses: List[Int], ints: List[Int], floats: List[Float64],
) raises -> Int:
    # Lane entry-table-zero-alloc (2026-11-11): the output words live on this
    # frame and reach the foreign call through the out-of-line span call's
    # `mut` reference, so no per-call heap List. The former one-element heap
    # List (Mojo 1.0.0/Apple root reproduction: inline integer-address outputs
    # did not survive the ABI roundtrip) stays behind the opt-in define
    # MOJOLEARN_ENTRY_TABLE_HEAP_OUTPUT until the stack form runs on Apple.
    comptime if is_defined["MOJOLEARN_ENTRY_TABLE_HEAP_OUTPUT"]():
        var held = List[NativeCallOutput]()
        held.append(NativeCallOutput())
        try:
            return native_descriptor_call_spans(entry, operation,
                Int(addresses.unsafe_ptr()), len(addresses), Int(ints.unsafe_ptr()), len(ints),
                Int(floats.unsafe_ptr()), len(floats), held[0])
        finally:
            _ = addresses
            _ = ints
            _ = floats
            _ = held
    else:
        var output = NativeCallOutput()
        try:
            return native_descriptor_call_spans(entry, operation,
                Int(addresses.unsafe_ptr()), len(addresses), Int(ints.unsafe_ptr()), len(ints),
                Int(floats.unsafe_ptr()), len(floats), output)
        finally:
            # Raw span addresses do not themselves retain their List owners.
            _ = addresses
            _ = ints
            _ = floats
            _ = output


@no_inline
def native_descriptor_call_spans(
    entry: NativeDescriptorCall, operation: Int, addresses: Int, address_count: Int,
    ints: Int, int_count: Int, floats: Int, float_count: Int, mut output: NativeCallOutput,
) raises -> Int:
    # The caller admits span bounds and pins their owners until this synchronous
    # call returns. output is caller storage reached by reference; this
    # function stays out of line so that storage is real memory across the
    # foreign call (see native_descriptor_call).
    # Reinitialize all writable state before every call, including cleanup calls.
    output.result = 0
    output.error.clear()
    var status = entry(operation, addresses, address_count, ints, int_count,
        floats, float_count, Int(MutPointer(to=output.result)),
        Int(MutPointer(to=output.error)), native_descriptor_error)
    if status != 0:
        output.error.append(0)
        raise Error(unsafe_string_from_utf8_cstr(output.error.unsafe_ptr()))
    return output.result


@always_inline
def native_descriptor_leased_call(
    entry: NativeDescriptorCall, workspace: Int, operation: Int, addresses: List[Int], ints: List[Int],
    floats: List[Float64],
) raises -> Int:
    """One synchronous call of `entry` under the workspace lease at
    `workspace` (a `NativeCallWorkspace` address, or 0 for a private one): the
    ABI output and its error storage are the retained scratch, reused across
    calls; a nested or concurrent call that finds it busy gets a private one.
    The one lease-and-call body of the provider route
    (providers/python/neural_descriptor_pyapi_door.mojo `neural_descriptor_execute`)
    and the Python-free door route (sequence/xseq_neural_run.mojo
    `xseq_neural_run_call`). The caller keeps every addressed buffer alive."""
    var loan = NativeCallWorkspaceLease(workspace)
    var scratch = loan.pointer()
    try:
        return native_descriptor_call_spans(entry, operation,
            Int(addresses.unsafe_ptr()), len(addresses), Int(ints.unsafe_ptr()), len(ints),
            Int(floats.unsafe_ptr()), len(floats), scratch[].output)
    finally:
        loan.release()
        # Span addresses and the workspace pointer do not retain these owners.
        _ = addresses
        _ = ints
        _ = floats


def native_descriptor_words(address: Int, count: Int) raises -> List[Int]:
    if count < 0 or (count > 0 and address == 0):
        raise Error("native descriptor: null or invalid word span")
    var source = MutPointer[Int, MutUntrackedOrigin](unsafe_from_address=address)
    var result = List[Int](capacity=count)
    for i in range(count):  # small-loop(validated native launch metadata words)
        result.append(source[i])
    return result^


def native_descriptor_report(problem: Error, context: Int, sink: NativeDescriptorError):
    var message = String(problem)
    var bytes = message.as_bytes()
    sink(context, Int(bytes.unsafe_ptr()), len(bytes))
    _ = bytes
    _ = message
