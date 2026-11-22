# SPDX-License-Identifier: Apache-2.0
"""Host-column member driver of the py2mojo batch 1 and 8 checks (lane py2mojo-A, 2026-11-22;
docs/plans/PY2MOJO_CONVERSION.md). Importable; no `main()`, no `test_*`.

`HostMember` drives a published member entry (core/member_abi.mojo) the way the public boundary's member owner does
on the host column: the StandardScaler / MinMaxScaler member (core/scaler_member.mojo `scaler_member_v2`) over the
host scaler kernels (core/scaler_native_host.mojo through `scaler_native_v1`, the entry
bindings/_mojolearn_preprocessing_host.mojo publishes) and the Prep member (core/prep_member.mojo `prep_member_v2`)
over the host x_prep transport (x_prep/transport_host.mojo `x_prep_transport_host_v1`, the entry
bindings/_mojolearn_x_prep_host.mojo publishes). The route words are the ones providers/python/member_prep_boundary_door.mojo
`_route_admit` writes for the host images: device CPU, vendor 0, the transport in `second`, no route flags (the host
transport exports XP_RUN only). The checks hand the member the same parameter words that UNPACK writes and compare
what it computes; nothing here computes a statistic.

    fit / fit_with(weights)     MEMBER_OP_FIT over a dense host float32 X (n x d)
    apply(op, x, n)             MEMBER_OP_TRANSFORM / _INVERSE_TRANSFORM, the width from MEMBER_OP_OUTPUTS
    block(index)                a copy of fitted block `index` (MEMBER_BLOCK_BY_INDEX order, float32)
    scalars()                   a copy of block 0 (MEMBER_BLOCK_SCALARS, float64)

A state is freed by `free()`; the checks call it on every path they return by. `all_close` is
numpy.testing.assert_allclose for one value (`|a - e| <= atol + rtol |e|`, NaN only against NaN).
"""
from std.math import isnan
from std.memory import bitcast
from core.entry_table import EntryRef, entry_ref_word
from core.device_route import DeviceRoute, DEVICE_CPU
from core.member_abi import (
    MemberCall, MemberMatrix, MemberRef, MEMBER_F32, MEMBER_OP_NEW, MEMBER_OP_FREE, MEMBER_OP_FIT,
    MEMBER_OP_TRANSFORM, MEMBER_OP_OUTPUTS, MEMBER_OP_EXPORT, MEMBER_BLOCK_BY_INDEX, member_call,
)
from core.prep_member import prep_member_v2
from core.scaler_member import scaler_member_v2
from core.scaler_native_api import scaler_native_v1, SCALER_OP_COUNT
from core.scaler_native_host import ScalerHostKernels
from x_prep.transport_host import x_prep_transport_host_v1
from x_prep.transport_abi import XP_OP_COUNT
from checks.numerics import GLOBAL_NUMERIC_MODE
from x_prep.blocked import IDN_STATS_BLOCKED, IDN_CLASS_ONEPASS
from x_prep.select_blocked import IDN_SELECT_BLOCKED
from x_prep.pt_blocked import IDN_PT_BLOCKED
from x_prep.host.rr_eigh_host import IDN_RR_EIGH
from checks.native_tests.testsuite_cases import require

comptime PREP_SCALERS_DIR = "checks/native_tests/fixtures/prep_scalers/"
comptime PREP_MEMBERS_DIR = "checks/native_tests/fixtures/prep_members/"
comptime MEMBER_PREP = 0
comptime MEMBER_SCALER = 1


def host_route() -> DeviceRoute:
    """The host images' route (the words `_route_admit` writes for them)."""
    var mode = Int(GLOBAL_NUMERIC_MODE)
    var route = DeviceRoute(DEVICE_CPU)
    route.mode = mode
    route.vendor = 0
    route.entry = EntryRef(scaler_native_v1[ScalerHostKernels], entry_ref_word(mode, 0, SCALER_OP_COUNT))
    route.second = EntryRef(x_prep_transport_host_v1, entry_ref_word(mode, 0, XP_OP_COUNT))
    route.has_second = True
    route.flags = 0
    return route


def host_idn_fam() -> Int:
    """The host identical image's `x_prep_idn_fam` word (bindings/_mojolearn_x_prep_host.mojo `idn_fam_binding`;
    0 when no switch is on, as an unregistered word reads)."""
    return ((1 if IDN_STATS_BLOCKED else 0) | (2 if IDN_CLASS_ONEPASS else 0) | (4 if IDN_SELECT_BLOCKED else 0)
            | (8 if IDN_PT_BLOCKED else 0) | (16 if IDN_RR_EIGH else 0))


struct HostMember:
    var member: MemberRef
    var route: DeviceRoute
    var params: List[Float64]
    var state: Int
    var features: Int

    def __init__(out self, family: Int, var params: List[Float64]) raises:
        if family == MEMBER_SCALER:
            self.member = MemberRef(EntryRef(scaler_member_v2, 0), 0)
        else:
            self.member = MemberRef(EntryRef(prep_member_v2, 0), 0)
        self.route = host_route()
        self.params = params^
        self.state = 0
        self.features = 0
        var call = MemberCall(self.route)
        call.params = Int(self.params.unsafe_ptr())
        call.param_count = len(self.params)
        _ = member_call(self.member, MEMBER_OP_NEW, call)
        self.state = call.state

    def free(mut self):
        if self.state == 0:
            return
        var call = MemberCall(self.route, self.state)
        try:
            _ = member_call(self.member, MEMBER_OP_FREE, call)
        except:
            pass
        self.state = 0

    def fit(mut self, x: List[Float32], n: Int, d: Int) raises:
        var none = List[Float32]()
        self.fit_with(x, n, d, none)

    def fit_with(mut self, x: List[Float32], n: Int, d: Int, weights: List[Float32]) raises:
        """MEMBER_OP_FIT; `weights` empty: none, else one float32 per row."""
        var call = MemberCall(self.route, self.state)
        call.x = MemberMatrix(Int(x.unsafe_ptr()), n, d)
        if len(weights) > 0:
            call.weights = MemberMatrix(Int(weights.unsafe_ptr()), n, 1)
        _ = member_call(self.member, MEMBER_OP_FIT, call)
        self.features = d
        _ = x
        _ = weights

    def width(self, operation: Int) raises -> Int:
        var call = MemberCall(self.route, self.state)
        call.flags = operation
        call.x.cols = self.features
        _ = member_call(self.member, MEMBER_OP_OUTPUTS, call)
        return call.out_cols

    def apply(self, operation: Int, x: List[Float32], n: Int) raises -> List[Float32]:
        """MEMBER_OP_TRANSFORM or MEMBER_OP_INVERSE_TRANSFORM of the (n, features) rows of `x`."""
        var cols = self.width(operation)
        var dst = List[Float32](length=n * cols, fill=0)
        var call = MemberCall(self.route, self.state)
        call.x = MemberMatrix(Int(x.unsafe_ptr()), n, len(x) // n if n > 0 else 0)
        call.dst = MemberMatrix(Int(dst.unsafe_ptr()), n, cols)
        _ = member_call(self.member, operation, call)
        _ = x
        return dst^

    def transform(self, x: List[Float32], n: Int) raises -> List[Float32]:
        return self.apply(MEMBER_OP_TRANSFORM, x, n)

    def _export(self, index: Int) raises -> MemberMatrix:
        var call = MemberCall(self.route, self.state)
        call.block = index
        call.flags = MEMBER_BLOCK_BY_INDEX
        _ = member_call(self.member, MEMBER_OP_EXPORT, call)
        return call.dst

    def block(self, index: Int) raises -> List[Float32]:
        """A copy of fitted block `index` (> 0, float32 words)."""
        var view = self._export(index)
        var count = view.rows * view.cols
        var words = List[Float32](length=count, fill=0)
        if count > 0:
            var source = MutPointer[Float32, MutUntrackedOrigin](unsafe_from_address=view.data)
            for i in range(count):  # small-loop(fixture: copy of one exported block)
                words[i] = source[i]
        return words^

    def scalars(self) raises -> List[Float64]:
        var view = self._export(0)
        var count = view.rows * view.cols
        var words = List[Float64](length=count, fill=0)
        var source = MutPointer[Float64, MutUntrackedOrigin](unsafe_from_address=view.data)
        for i in range(count):  # small-loop(fixture: copy of the scalar block)
            words[i] = source[i]
        return words^


def all_close(actual: Float32, expected: Float64, rtol: Float64, atol: Float64, label: String) raises:
    """numpy.testing.assert_allclose for one value: NaN only against NaN."""
    var a = Float64(actual)
    if isnan(expected) or isnan(a):
        require(isnan(expected) and isnan(a), label + ": " + String(actual) + " expected " + String(expected))
        return
    require(abs(a - expected) <= atol + rtol * abs(expected),
        label + ": " + String(actual) + " expected " + String(expected))


def rows_close(actual: List[Float32], expected: List[Float64], rtol: Float64, atol: Float64, label: String) raises:
    require(len(actual) == len(expected), label + ": " + String(len(actual)) + " words, expected "
        + String(len(expected)))
    for i in range(len(actual)):  # small-loop(oracle: one result block)
        all_close(actual[i], expected[i], rtol, atol, label + " word " + String(i))


def same_bits(actual: List[Float32], expected: List[Float32], label: String) raises:
    require(len(actual) == len(expected), label + ": lengths differ")
    for i in range(len(actual)):  # small-loop(oracle: one result block)
        require(bitcast[DType.uint32](actual[i]) == bitcast[DType.uint32](expected[i]),
            label + " word " + String(i) + ": " + String(actual[i]) + " expected " + String(expected[i]))


def take(values: List[Float64], start: Int, count: Int) -> List[Float64]:
    var part = List[Float64](length=count, fill=0)
    for i in range(count):  # small-loop(fixture: one slice of a reference)
        part[i] = values[start + i]
    return part^


def take32(values: List[Float32], start: Int, count: Int) -> List[Float32]:
    var part = List[Float32](length=count, fill=0)
    for i in range(count):  # small-loop(fixture: one slice of an input)
        part[i] = values[start + i]
    return part^
