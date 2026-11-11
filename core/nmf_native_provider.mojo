# SPDX-License-Identifier: Apache-2.0
"""NMF native requests over the existing host/device decomposition owners."""
from checks.numerics import GLOBAL_NUMERIC_MODE
from checks.vendor import COMPILED_VENDOR
from x_decomp.exec_trait import Exec
from x_decomp.cells import F32Ptr
from x_decomp.pca_randomized import RDev
from x_decomp.decomp_owner_kit import DecompDev
from x_decomp.nmf import nmf_args_checked
from x_decomp.nmf_owner import nmf_run, nmf_inverse_run
from core.pca_native_api import pca_vendor_code
from core.native_descriptor_abi import NativeDescriptorCall, NativeDescriptorError
from core.nmf_native_api import (
    NMF_NATIVE_ABI, NMF_CAP_FIT, NMF_CAP_INVERSE, NMFNativeKernels,
    NMFFitRequest, NMFFitResult, NMFInverseRequest, nmf_native_dispatch,
)


def _nmf_pointer(address: Int) raises -> F32Ptr:
    if address == 0:
        raise Error("x_decomp: null float32 buffer address")
    return F32Ptr(unsafe_from_address=address)


def _nmf_dimension(value: Int) raises:
    if value < 0 or value > 2147483647:
        raise Error("x_decomp: dimension out of range")


struct NMFProviderKernels[E: Exec, D: RDev, RES: Bool, X: DecompDev](NMFNativeKernels):
    @staticmethod
    def mode() -> Int:
        return Int(GLOBAL_NUMERIC_MODE)

    @staticmethod
    def vendor() -> Int:
        return pca_vendor_code(String(COMPILED_VENDOR)) if Self.RES else 1

    @staticmethod
    def capabilities() -> Int:
        return NMF_CAP_FIT | NMF_CAP_INVERSE

    @staticmethod
    def fit(r: NMFFitRequest) raises -> NMFFitResult:
        # Preserve the old provider's pointer/dimension/options admission order.
        var source = _nmf_pointer(r.source)
        _nmf_dimension(r.rows)
        _nmf_dimension(r.features)
        _nmf_dimension(r.components)
        var w_initial = _nmf_pointer(r.w_initial)
        var h_initial = _nmf_pointer(r.h_initial)
        var a = nmf_args_checked(r.args.beta, r.args.update_h, r.args.max_iter, r.args.tol,
            r.args.l1w, r.args.l1h, r.args.l2w, r.args.l2h, r.args.shuffle, r.args.seed)
        var w_output = _nmf_pointer(r.w_output)
        var h_output = _nmf_pointer(r.h_output)
        var word = List[Int32](length=1, fill=Int32(0))
        var result = nmf_run[Self.E, Self.D, Self.RES, Self.X](source, r.rows, r.features, r.components, r.initialization,
            w_initial, h_initial, r.args.seed, r.solver, a, w_output, h_output,
            Int(word.unsafe_ptr()) if r.test_finite else 0)
        return NMFFitResult(result[0], result[1], Int(word[0]))

    @staticmethod
    def inverse(r: NMFInverseRequest) raises:
        var source = _nmf_pointer(r.source)
        _nmf_dimension(r.rows)
        _nmf_dimension(r.count)
        var components = _nmf_pointer(r.components)
        _nmf_dimension(r.features)
        var dst = _nmf_pointer(r.dst)
        nmf_inverse_run[Self.E, Self.D, Self.RES](source, r.rows, r.count, components, r.features, dst)


def nmf_native_v1[E: Exec, D: RDev, RES: Bool, X: DecompDev](operation: Int,
    aa: Int, na: Int, pp: Int, np: Int, ff: Int, nf: Int, result: Int,
    context: Int, sink: NativeDescriptorError) abi("C") -> Int:
    return nmf_native_dispatch[NMFProviderKernels[E, D, RES, X]](operation, aa, na, pp, np, ff, nf, result, context, sink)


struct NMFProviderAPI[E: Exec, D: RDev, RES: Bool, X: DecompDev](Defaultable, Movable):
    var name: String
    var entries: List[NativeDescriptorCall]

    def __init__(out self):
        self.name = String(NMF_NATIVE_ABI)
        self.entries = List[NativeDescriptorCall]()
        self.entries.append(nmf_native_v1[Self.E, Self.D, Self.RES, Self.X])
