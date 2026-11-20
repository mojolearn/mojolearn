# SPDX-License-Identifier: Apache-2.0
"""Fixed native calls to the existing IncrementalPCA owners and resident pool."""
from checks.numerics import GLOBAL_NUMERIC_MODE
from checks.vendor import COMPILED_VENDOR
from x_decomp.cells import F32Ptr
from x_decomp.exec_trait import Exec
from x_decomp.pca_randomized import RDev
from x_decomp.ipca_owner import (
    IPCA_STATE_WORDS, IpcaDev, ipca_fit_run, ipca_partial_step, ipca_transform_run, ipca_inverse_run,
)
from core.pca_native_api import pca_vendor_code
from core.native_descriptor_abi import NativeDescriptorCall, NativeDescriptorError
from core.ipca_native_api import (
    IPCA_NATIVE_ABI, IPCA_CAP_FIT, IPCA_CAP_PARTIAL, IPCA_CAP_QUERY, IPCA_CAP_RESIDENT,
    IPCANativeKernels, IPCAFitRequest, IPCAPartialRequest, IPCAQueryRequest,
    IPCAFitResult, IPCAStateIds, ipca_native_dispatch,
)


def _ipca_pointer(address: Int) raises -> F32Ptr:
    if address == 0:
        raise Error("x_decomp: null float32 buffer address")
    return F32Ptr(unsafe_from_address=address)


def _ipca_dimension(value: Int) raises:
    if value < 0 or value > 2147483647:
        raise Error("x_decomp: dimension out of range")


def _ipca_state(ids: IPCAStateIds, resident: Bool) raises -> List[Int]:
    var values = List[Int]()
    if resident:
        for k in range(IPCA_STATE_WORDS):  # driver-loop: over five state handles
            if ids.values[k] < -1:
                raise Error("x_decomp: a resident state id must be -1 or a pool id")
            values.append(ids.values[k])
    return values^


def _ipca_result(got: Tuple[Int, Float64], kept: List[Int]) -> IPCAFitResult:
    var state = IPCAStateIds()
    var resident = len(kept) == IPCA_STATE_WORDS
    if resident:
        for k in range(IPCA_STATE_WORDS):  # driver-loop: over five state handles
            state.values[k] = kept[k]
    return IPCAFitResult(got[0], got[1], resident, state^)


struct IPCAProviderKernels[E: Exec, D: RDev, RES: Bool, X: IpcaDev](IPCANativeKernels):
    @staticmethod
    def mode() -> Int:
        return Int(GLOBAL_NUMERIC_MODE)

    @staticmethod
    def vendor() -> Int:
        return pca_vendor_code(String(COMPILED_VENDOR)) if Self.RES else 1

    @staticmethod
    def capabilities() -> Int:
        return IPCA_CAP_FIT | IPCA_CAP_PARTIAL | IPCA_CAP_QUERY | (IPCA_CAP_RESIDENT if Self.RES else 0)

    @staticmethod
    def fit(r: IPCAFitRequest) raises -> IPCAFitResult:
        # The order is the legacy provider's address/dimension/output admission.
        var source = _ipca_pointer(r.source)
        _ipca_dimension(r.rows)
        _ipca_dimension(r.features)
        var components = _ipca_pointer(r.outputs.components)
        var singular = _ipca_pointer(r.outputs.singular)
        var mean = _ipca_pointer(r.outputs.mean)
        var variance = _ipca_pointer(r.outputs.population_variance)
        var explained = _ipca_pointer(r.outputs.explained_variance)
        var ratio = _ipca_pointer(r.outputs.ratio)
        var kept = List[Int]()
        var got = ipca_fit_run[Self.E, Self.D, Self.RES, Self.X](source, r.rows, r.features,
            r.requested, r.batch_size, components, singular, mean, variance, explained, ratio, r.keep, kept)
        return _ipca_result(got, kept)

    @staticmethod
    def partial(r: IPCAPartialRequest) raises -> IPCAFitResult:
        var source = _ipca_pointer(r.source)
        _ipca_dimension(r.rows)
        _ipca_dimension(r.features)
        # Even the first batch carries the same admitted, non-null dummy addresses.
        var old_components = _ipca_pointer(r.previous.components)
        var old_singular = _ipca_pointer(r.previous.singular)
        var old_mean = _ipca_pointer(r.previous.mean)
        var old_variance = _ipca_pointer(r.previous.population_variance)
        var components = _ipca_pointer(r.outputs.components)
        var singular = _ipca_pointer(r.outputs.singular)
        var mean = _ipca_pointer(r.outputs.mean)
        var variance = _ipca_pointer(r.outputs.population_variance)
        var explained = _ipca_pointer(r.outputs.explained_variance)
        var ratio = _ipca_pointer(r.outputs.ratio)
        var previous = _ipca_state(r.state, r.resident)
        var kept = List[Int]()
        var got = ipca_partial_step[Self.E, Self.D, Self.RES, Self.X](source, r.rows, r.features,
            r.requested, r.seen, r.previous_count, r.previous_features,
            old_components, old_singular, old_mean, old_variance,
            components, singular, mean, variance, explained, ratio, previous, r.keep, kept)
        return _ipca_result(got, kept)

    @staticmethod
    def query(r: IPCAQueryRequest) raises:
        var source = _ipca_pointer(r.source)
        _ipca_dimension(r.rows)
        _ipca_dimension(r.features)
        _ipca_dimension(r.count)
        var mean = _ipca_pointer(r.mean)
        var components = _ipca_pointer(r.components)
        var variance = _ipca_pointer(r.variance)
        var dst = _ipca_pointer(r.dst)
        var state = _ipca_state(r.state, r.resident)
        if r.inverse:
            ipca_inverse_run[Self.E, Self.D, Self.RES](source, r.rows, r.features, r.count,
                mean, components, variance, r.whiten, dst, state)
        else:
            ipca_transform_run[Self.E, Self.D, Self.RES](source, r.rows, r.features, r.count,
                mean, components, variance, r.whiten, dst, state)

    @staticmethod
    def prepare(address: Int, count: Int) raises -> Int:
        if not Self.RES:
            raise Error("IncrementalPCA host provider does not keep resident state")
        var handle = Self.D.alloc(count)
        try:
            Self.D.upload(handle, _ipca_pointer(address), count)
        except error:
            try:
                Self.D.free(handle)
            except:
                pass
            raise error
        return handle

    @staticmethod
    def free(handle: Int) raises:
        Self.D.free(handle)


def ipca_native_v1[E: Exec, D: RDev, RES: Bool, X: IpcaDev](operation: Int,
    aa: Int, na: Int, pp: Int, np: Int, ff: Int, nf: Int, result: Int,
    context: Int, sink: NativeDescriptorError) abi("C") -> Int:
    return ipca_native_dispatch[IPCAProviderKernels[E, D, RES, X]](
        operation, aa, na, pp, np, ff, nf, result, context, sink)


struct IPCAProviderAPI[E: Exec, D: RDev, RES: Bool, X: IpcaDev](Defaultable, Movable):
    var name: String
    var entries: List[NativeDescriptorCall]

    def __init__(out self):
        self.name = String(IPCA_NATIVE_ABI)
        self.entries = List[NativeDescriptorCall]()
        self.entries.append(ipca_native_v1[Self.E, Self.D, Self.RES, Self.X])
