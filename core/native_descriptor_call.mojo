# SPDX-License-Identifier: Apache-2.0
# Native-only invocation; Python ownership belongs in providers/python.
from core.native_descriptor_abi import NativeDescriptorCall, NativeDescriptorError
from core.native_call_output import NativeCallOutput


def native_descriptor_error(context: Int, address: Int, size: Int) abi("C") -> NoneType:
    var out = MutPointer[List[UInt8], MutUntrackedOrigin](unsafe_from_address=context)
    var src = MutPointer[UInt8, MutUntrackedOrigin](unsafe_from_address=address)
    for i in range(size):  # driver-loop: copy complete native refusal text
        out[].append(src[i])
    return None


def native_descriptor_call(
    entry: NativeDescriptorCall, operation: Int, addresses: List[Int], ints: List[Int], floats: List[Float64],
) raises -> Int:
    # Stable writable owner: inline integer-address outputs did not survive
    # the supported ABI roundtrip on Mojo 1.0.0/Apple (root reproduction).
    var output = List[NativeCallOutput]()
    output.append(NativeCallOutput())
    try:
        return native_descriptor_call_spans(entry, operation,
            Int(addresses.unsafe_ptr()), len(addresses), Int(ints.unsafe_ptr()), len(ints),
            Int(floats.unsafe_ptr()), len(floats), output[0])
    finally:
        # Raw span addresses do not themselves retain their List owners.
        _ = addresses
        _ = ints
        _ = floats
        _ = output


def native_descriptor_call_spans(
    entry: NativeDescriptorCall, operation: Int, addresses: Int, address_count: Int,
    ints: Int, int_count: Int, floats: Int, float_count: Int, mut output: NativeCallOutput,
) raises -> Int:
    # The caller admits span bounds and pins their owners until this synchronous
    # call returns. output must be a retained heap-owned field (see above).
    # Reinitialize all writable state before every call, including cleanup calls.
    output.result = 0
    output.error.clear()
    var status = entry(operation, addresses, address_count, ints, int_count,
        floats, float_count, Int(MutPointer(to=output.result)),
        Int(MutPointer(to=output.error)), native_descriptor_error)
    if status != 0:
        output.error.append(0)
        raise Error(String(unsafe_from_utf8_ptr=output.error.unsafe_ptr()))
    return output.result


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
