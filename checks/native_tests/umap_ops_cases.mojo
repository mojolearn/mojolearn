# SPDX-License-Identifier: Apache-2.0
"""The UMAP operations table (core/neighbors_entry_table_umap.mojo), one case
per entry UMAP's owner (bindings/mani_umap_native.mojo) calls, with
expectations known by construction: the returned column count, a written
finite embedding, the same bits for the same request, the legacy layout
equal to the explicit default rates, and refusals that leave the output
untouched. No Python objects.
"""
from std.math import isfinite
from core.native_descriptor_abi import NativeDescriptorCall
from core.entry_table import entry_table_same_call
from core.native_descriptor_call import native_descriptor_call
from core.neighbors_entry_table import NeighborsEntryKey
from core.neighbors_entry_table_umap import (
    UmapConfigWords, UmapTargetWords, umap_ops_publish, umap_ops_entry, umap_fit_call, umap_fit_ex_call,
    umap_transform_call, UMAP_OP_FIT, UMAP_TAIL_LEGACY, UMAP_TAIL_RATES, UMAP_TAIL_EXTRAS,
)

comptime UMAP_SHARED_CASES = 7
comptime _N = 16
comptime _D = 2
comptime _NC = 2
comptime _SENTINEL: Float32 = -12345.0


def require(ok: Bool, message: String) raises:
    if not ok:
        raise Error(message)


def _f(values: List[Float32]) -> Int:
    return Int(values.unsafe_ptr())


def _points() -> List[Float32]:
    # Two groups of eight 2-D points, far apart: (i, 0) and (100 + i, 1).
    var x = List[Float32](capacity=_N * _D)
    for i in range(_N // 2):  # small-loop(no: a 16-point fixture): test data
        x.append(Float32(i))
        x.append(0.0)
    for i in range(_N // 2):  # small-loop(no: a 16-point fixture): test data
        x.append(Float32(100 + i))
        x.append(1.0)
    return x^


def _config() -> UmapConfigWords:
    # n_neighbors 4, 2 components, 10 epochs, the default floats, seed 3,
    # the default rates [1.0, 1.0, 5] and extras (-1, 2.0, 0.0, 0.0).
    return UmapConfigWords(4, _NC, 10, 0.1, 1.0, 1.0, 1.0, 3, 1.0, 1.0, 5, -1, 2.0, 0.0, 0.0)


def _written(values: List[Float32]) -> Bool:
    for v in values:
        if v == _SENTINEL or not isfinite(v):
            return False
    return True


def _same(a: List[Float32], b: List[Float32]) -> Bool:
    if len(a) != len(b):
        return False
    for i in range(len(a)):  # small-loop(no: a 32-word embedding): test comparison
        if a[i] != b[i]:
            return False
    return True


def case_fit_legacy(entry: NativeDescriptorCall) raises:
    # The legacy layout writes an n x 2 finite embedding and returns 2; the
    # same request gives the same bits; the explicit default rates equal the
    # legacy layout that leaves them out.
    var x = _points()
    var cfg = _config()
    var first = List[Float32](length=_N * _NC, fill=_SENTINEL)
    var again = List[Float32](length=_N * _NC, fill=_SENTINEL)
    var rated = List[Float32](length=_N * _NC, fill=_SENTINEL)
    require(umap_fit_call(entry, _f(x), _f(first), _N, _D, cfg, False) == _NC, "fit: column count")
    require(_written(first), "fit: the embedding is not written or not finite")
    _ = umap_fit_call(entry, _f(x), _f(again), _N, _D, cfg, False)
    require(_same(first, again), "fit: the same request gave other bits")
    _ = umap_fit_call(entry, _f(x), _f(rated), _N, _D, cfg, True)
    require(_same(first, rated), "fit: the explicit default rates differ from the legacy layout")
    _ = x
    _ = first
    _ = again
    _ = rated


def case_fit_ex(entry: NativeDescriptorCall) raises:
    # The option-parity fit with no initial embedding and no target.
    var x = _points()
    var dst = List[Float32](length=_N * _NC, fill=_SENTINEL)
    require(umap_fit_ex_call(entry, _f(x), _f(dst), 0, _N, _D, _config(), UmapTargetWords.none()) == _NC,
            "fit_ex: column count")
    require(_written(dst), "fit_ex: the embedding is not written or not finite")
    _ = x
    _ = dst


def case_fit_ex_initial(entry: NativeDescriptorCall) raises:
    # A given initial embedding (no spectral start) is accepted and the
    # result is written.
    var x = _points()
    var initial = List[Float32](capacity=_N * _NC)
    for i in range(_N * _NC):  # small-loop(no: a 32-word fixture): test data
        initial.append(Float32(i % 7) - 3.0)
    var dst = List[Float32](length=_N * _NC, fill=_SENTINEL)
    require(umap_fit_ex_call(entry, _f(x), _f(dst), _f(initial), _N, _D, _config(), UmapTargetWords.none())
            == _NC, "fit_ex initial: column count")
    require(_written(dst), "fit_ex initial: the embedding is not written or not finite")
    _ = x
    _ = initial
    _ = dst


def case_transform(entry: NativeDescriptorCall) raises:
    # Fit, then transform four queries in every tail layout: each returns 2
    # and writes finite words; the explicit default rates equal the legacy
    # layout, and the default extras equal both.
    var x = _points()
    var cfg = _config()
    var fitted = List[Float32](length=_N * _NC, fill=_SENTINEL)
    _ = umap_fit_call(entry, _f(x), _f(fitted), _N, _D, cfg, False)
    var queries: List[Float32] = [0.5, 0.0, 3.5, 0.0, 100.5, 1.0, 104.5, 1.0]
    var legacy = List[Float32](length=4 * _NC, fill=_SENTINEL)
    var rates = List[Float32](length=4 * _NC, fill=_SENTINEL)
    var extras = List[Float32](length=4 * _NC, fill=_SENTINEL)
    require(umap_transform_call(entry, _f(x), _f(fitted), _f(queries), _f(legacy), _N, 4, _D, cfg,
                                UMAP_TAIL_LEGACY) == _NC, "transform: column count")
    _ = umap_transform_call(entry, _f(x), _f(fitted), _f(queries), _f(rates), _N, 4, _D, cfg, UMAP_TAIL_RATES)
    _ = umap_transform_call(entry, _f(x), _f(fitted), _f(queries), _f(extras), _N, 4, _D, cfg, UMAP_TAIL_EXTRAS)
    require(_written(legacy), "transform: the embedding is not written or not finite")
    require(_same(legacy, rates), "transform: the explicit default rates differ from the legacy layout")
    require(_same(legacy, extras), "transform: the explicit default extras differ from the legacy layout")
    _ = x
    _ = fitted
    _ = queries
    _ = legacy
    _ = rates
    _ = extras


def case_refuses_negative_seed(entry: NativeDescriptorCall) raises:
    var x = _points()
    var dst = List[Float32](length=_N * _NC, fill=_SENTINEL)
    var cfg = _config()
    cfg.seed = -1
    var refused = False
    try:
        _ = umap_fit_call(entry, _f(x), _f(dst), _N, _D, cfg, False)
    except error:
        refused = "nonnegative seed" in String(error)
    require(refused, "fit: a negative seed was not refused with the export's text")
    require(dst[0] == _SENTINEL, "fit: a refused request wrote the output")
    _ = x
    _ = dst


def case_refuses_malformed_target(entry: NativeDescriptorCall) raises:
    # A categorical target kind with no target address.
    var x = _points()
    var dst = List[Float32](length=_N * _NC, fill=_SENTINEL)
    var refused = False
    try:
        _ = umap_fit_ex_call(entry, _f(x), _f(dst), 0, _N, _D, _config(), UmapTargetWords(1, 1, 0, 0.5, 0))
    except error:
        refused = "supervised target is malformed" in String(error)
    require(refused, "fit_ex: a target kind without a target was not refused")
    require(dst[0] == _SENTINEL, "fit_ex: a refused request wrote the output")
    _ = x
    _ = dst


def case_refuses_short_spans(entry: NativeDescriptorCall) raises:
    # Five Int words fit no fit layout: refused before any word is read.
    var x = _points()
    var dst = List[Float32](length=_N * _NC, fill=_SENTINEL)
    var refused = False
    try:
        var addresses: List[Int] = [_f(x), _f(dst)]
        var ints: List[Int] = [_N, _D, 4, _NC, 10]
        var floats: List[Float64] = [0.1, 1.0, 1.0, 1.0]
        _ = native_descriptor_call(entry, UMAP_OP_FIT, addresses, ints, floats)
    except error:
        refused = "request spans differ" in String(error)
    require(refused, "fit: a short request was not refused")
    require(dst[0] == _SENTINEL, "fit: a refused request wrote the output")
    _ = x
    _ = dst


def run_shared_case(entry: NativeDescriptorCall, case_id: Int) raises:
    if case_id == 0:
        case_fit_legacy(entry)
    elif case_id == 1:
        case_fit_ex(entry)
    elif case_id == 2:
        case_fit_ex_initial(entry)
    elif case_id == 3:
        case_transform(entry)
    elif case_id == 4:
        case_refuses_negative_seed(entry)
    elif case_id == 5:
        case_refuses_malformed_target(entry)
    else:
        case_refuses_short_spans(entry)


def check_umap_table(key: NeighborsEntryKey, publisher: Int, entry: NativeDescriptorCall) raises:
    """Publication then lookup returns the published pointer; a repeated
    identical publication is accepted; another tier refuses. Run first, in
    a fresh process (it fills the process slot)."""
    var missing = False
    try:
        _ = umap_ops_entry(key)
    except:
        missing = True
    require(missing, "umap table: lookup before publication did not refuse")
    umap_ops_publish(key, publisher, entry)
    umap_ops_publish(key, publisher, entry)
    require(entry_table_same_call(umap_ops_entry(key), entry), "umap table: lookup returned another pointer")
    var other = False
    try:
        _ = umap_ops_entry(NeighborsEntryKey(1 - key.mode, key.vendor))
    except:
        other = True
    require(other, "umap table: the other tier's unpublished slot did not refuse")
