# SPDX-License-Identifier: Apache-2.0
# Linalg engine operations carried by the shared native descriptor transport.
from std.ffi import _Global
from core.native_descriptor_abi import NativeDescriptorCall, NativeDescriptorError
from core.native_descriptor_call import native_descriptor_words, native_descriptor_report
from checks.numerics import GLOBAL_NUMERIC_MODE
from gemm.contract import GEMM_NUMERICAL_PROFILE_VERSION, LOWBIT_PROFILE_VERSION, INT15_PROFILE_VERSION
from gemm.host.linalg_native import gemm_native, gemm_bf16_native, gemm_int8_native, quantize_int8_native, dequantize_int8_native, gemm_int15_native, quantize_int15_native, dequantize_int15_native, to_bf16_native, from_bf16_native, from_f16_native

comptime LINALG_NATIVE_ABI_NAME = "mojolearn.linalg.v1"
comptime LINALG_NATIVE_OPERATION_COUNT = 15


def linalg_native_operation(op: Int, a: List[Int], p: List[Int]) raises -> Int:
    if op == 0:
        if len(a) != 3:
            raise Error("linalg native ABI: invalid gemm address count")
        return gemm_native(a[0], a[1], a[2], p)
    if op == 1:
        if len(a) != 3:
            raise Error("linalg native ABI: invalid gemm_bf16 address count")
        return gemm_bf16_native(a[0], a[1], a[2], p)
    if op == 2:
        if len(a) != 5:
            raise Error("linalg native ABI: invalid gemm_int8 address count")
        return gemm_int8_native(a[0], a[1], a[2], a[3], a[4], p)
    if op == 3:
        if len(a) != 3:
            raise Error("linalg native ABI: invalid quantize_int8 address count")
        return quantize_int8_native(a[0], a[1], a[2], p)
    if op == 4:
        if len(a) != 3:
            raise Error("linalg native ABI: invalid dequantize_int8 address count")
        return dequantize_int8_native(a[0], a[1], a[2], p)
    if op == 5:
        if len(a) != 7:
            raise Error("linalg native ABI: invalid gemm_int15 address count")
        return gemm_int15_native(a, p)
    if op == 6:
        if len(a) != 4:
            raise Error("linalg native ABI: invalid quantize_int15 address count")
        return quantize_int15_native(a, p)
    if op == 7:
        if len(a) != 4:
            raise Error("linalg native ABI: invalid dequantize_int15 address count")
        return dequantize_int15_native(a, p)
    if op == 8:
        if len(a) != 2:
            raise Error("linalg native ABI: invalid to_bf16 address count")
        return to_bf16_native(a[0], a[1], p)
    if op == 9:
        if len(a) != 2:
            raise Error("linalg native ABI: invalid from_bf16 address count")
        return from_bf16_native(a[0], a[1], p)
    if op == 10:
        if len(a) != 2:
            raise Error("linalg native ABI: invalid from_f16 address count")
        return from_f16_native(a[0], a[1], p)
    if op == 11:
        if len(a) != 0 or len(p) != 0:
            raise Error("linalg native ABI: metadata takes no descriptors")
        return GLOBAL_NUMERIC_MODE
    if op == 12:
        if len(a) != 0 or len(p) != 0:
            raise Error("linalg native ABI: metadata takes no descriptors")
        return GEMM_NUMERICAL_PROFILE_VERSION
    if op == 13:
        if len(a) != 0 or len(p) != 0:
            raise Error("linalg native ABI: metadata takes no descriptors")
        return LOWBIT_PROFILE_VERSION
    if op == 14:
        if len(a) != 0 or len(p) != 0:
            raise Error("linalg native ABI: metadata takes no descriptors")
        return INT15_PROFILE_VERSION
    raise Error("linalg native ABI: unknown operation")


def linalg_native_v1(op: Int, aa: Int, na: Int, pp: Int, np: Int, ff: Int, nf: Int,
                     result: Int, context: Int, sink: NativeDescriptorError) abi("C") -> Int:
    try:
        if result == 0 or nf != 0:
            raise Error("linalg native ABI: invalid result or scalar span")
        var a = native_descriptor_words(aa, na)
        var p = native_descriptor_words(pp, np)
        MutPointer[Int, MutUntrackedOrigin](unsafe_from_address=result)[0] = linalg_native_operation(op, a, p)
        return 0
    except problem:
        native_descriptor_report(problem, context, sink)
        return 1


struct LinalgNativeTable(Defaultable, Movable):
    var name: String
    var entries: List[NativeDescriptorCall]

    def __init__(out self):
        self.name = String(LINALG_NATIVE_ABI_NAME)
        self.entries = List[NativeDescriptorCall]()
        self.entries.append(linalg_native_v1)


comptime LINALG_NATIVE_TABLE = _Global[StorageType=LinalgNativeTable,
    name="MojoLinalgHostNativeTableV1", init_fn=LinalgNativeTable.__init__]
