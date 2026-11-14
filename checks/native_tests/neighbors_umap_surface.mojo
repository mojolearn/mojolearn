# SPDX-License-Identifier: Apache-2.0
"""UMAP fit and transform numerics through the UMAP operations table.

Ported from `python/mojolearn/tests/test_umap_surface.py`
(`test_public_layout`, `test_three_dimensions`,
`test_multidimensional_parameter_profiles`, `test_optimizer_controls`) and
`test_umap_transform.py` (`test_unseen_repeat_and_readonly_conversion`
without the float64/read-only conversion, `test_three_dimensions_and_single_query`,
`test_optimizer_controls_and_fitted_parameter_guard` without the guard). The
calls are the ones the public door makes (`bindings/mani_umap_native_door.mojo`):
`umap_fit_call` for default extras with a spectral start (the explicit rates
only when they differ from [1.0, 1.0, 5]), and `umap_transform_call` with the
legacy tail for default rates, the rates tail otherwise. The words are the
public parameters as `umap_parameters_fields` packs them (euclidean is
extras (-1, 2.0, 0.0, 0.0)).

Inputs, options and the pinned IDENTICAL layout words are the originals; the
pixi tasks build with `MOJOLEARN_NUMERIC_IDENTICAL=1`, so the IDENTICAL-only
pins and repeat checks are unconditional here. The public storage, dtype,
input conversion, snapshot, training-shortcut and refit-guard checks stay in
Python.
"""
from std.memory import bitcast
from std.math import isfinite
from core.native_descriptor_abi import NativeDescriptorCall
from core.neighbors_entry_table_umap import (
    UmapConfigWords,
    umap_fit_call,
    umap_transform_call,
    UMAP_TAIL_LEGACY,
    UMAP_TAIL_RATES,
)
from core.host_parallel import host_parallelize
from umap.ops_entry_host import umap_ops_host_v1
from umap.ops_entry_device import umap_ops_device_v1
from checks.native_tests.runner import CPU_COLUMN_BUILD, Check, Column, Report, require


comptime UMAP_SURFACE_FIXTURES = 9
comptime _SENTINEL: Float32 = -12345.0


def _f(values: List[Float32]) -> Int:
    return Int(values.unsafe_ptr())


def layout_bits() -> List[UInt32]:
    """LAYOUT_BITS of test_umap_surface.py (8 x 2)."""
    var bits: List[UInt32] = [
        3245070096, 1074042411, 3228888279, 3239233640,
        3237928427, 3203993724, 3215931591, 1091897539,
        1068472091, 1091895758, 1090606104, 3207501378,
        1092535950, 3235377189, 1091864365, 3227045188,
    ]
    return bits^


def broader_layout_bits() -> List[UInt32]:
    """BROADER_LAYOUT_BITS of test_umap_surface.py (16 x 3)."""
    var bits: List[UInt32] = [
        3226497000, 1086260798, 3227178554, 1036433928,
        1089708114, 3216624924, 1082121456, 1082849697,
        3221162157, 3238030912, 3230010965, 3201815628,
        1069619814, 1091856905, 3196772973, 3238703680,
        3227506494, 3205232299, 1062341195, 1091962546,
        3206499019, 1085703294, 3229833468, 1071493997,
        3236548473, 3229965506, 1050597661, 1086962268,
        3219069631, 3204773181, 3235819480, 3232328914,
        3168001690, 1082114376, 3219719040, 1059509553,
        1089551540, 3231064605, 1052606071, 3225822990,
        1074975858, 1093105335, 1088974209, 3224322712,
        3203895133, 3223456848, 1081193635, 1092231415,
    ]
    return bits^


def surface_x() -> List[Float32]:
    var x: List[Float32] = [0.0, 1.0, 2.2, 4.0, 6.5, 10.0, 14.5, 20.0]
    return x^


def config(n_neighbors: Int, components: Int, epochs: Int, seed: Int, min_dist: Float64 = 0.1,
           spread: Float64 = 1.0, mix: Float64 = 1.0, rate: Float64 = 1.0, repulsion: Float64 = 1.0,
           negatives: Int = 5) -> UmapConfigWords:
    return UmapConfigWords(n_neighbors, components, epochs, min_dist, spread, mix, 1.0, seed, rate,
        repulsion, negatives, -1, 2.0, 0.0, 0.0)


def fit(entry: NativeDescriptorCall, x: List[Float32], n: Int, d: Int, cfg: UmapConfigWords) raises -> List[Float32]:
    var embedding = List[Float32](length=n * cfg.n_components, fill=_SENTINEL)
    var columns = umap_fit_call(entry, _f(x), _f(embedding), n, d, cfg, not cfg.legacy_rates())
    require(columns == cfg.n_components, "UMAP fit column count")
    _ = x
    return embedding^


def transform(entry: NativeDescriptorCall, x: List[Float32], fitted: List[Float32], queries: List[Float32],
              n: Int, nq: Int, d: Int, cfg: UmapConfigWords) raises -> List[Float32]:
    var dst = List[Float32](length=nq * cfg.n_components, fill=_SENTINEL)
    var tail = UMAP_TAIL_LEGACY if cfg.legacy_rates() else UMAP_TAIL_RATES
    var columns = umap_transform_call(entry, _f(x), _f(fitted), _f(queries), _f(dst), n, nq, d, cfg, tail)
    require(columns == cfg.n_components, "UMAP transform column count")
    _ = x
    _ = fitted
    _ = queries
    return dst^


def finite(values: List[Float32], what: String) raises:
    var bad = 0
    for v in values:
        bad += Int(v == _SENTINEL or not isfinite(v))
    require(bad == 0, what + ": " + String(bad) + " values unwritten or not finite")


def same_bits(a: List[Float32], b: List[Float32]) -> Bool:
    if len(a) != len(b):
        return False
    for i in range(len(a)):
        if a[i].to_bits() != b[i].to_bits():
            return False
    return True


def case_public_layout(entry: NativeDescriptorCall) raises:
    var x = surface_x()
    var cfg = config(3, 2, 4, 19)
    var layout = fit(entry, x, 8, 1, cfg)
    finite(layout, "public layout")
    var pins = layout_bits()
    var moved = 0
    for i in range(16):
        moved += Int(bitcast[DType.uint32](layout[i]) != pins[i])
    require(moved == 0, "public layout: " + String(moved) + " words differ from LAYOUT_BITS")
    var again = fit(entry, x, 8, 1, cfg)
    require(same_bits(layout, again), "public layout: a second fit moved the bits")
    _ = x^


def case_three_dimensions(entry: NativeDescriptorCall) raises:
    var x = List[Float32](length=12, fill=0)
    for i in range(12):
        x[i] = Float32((i + 1) * (i + 1))
    var layout = fit(entry, x, 12, 1, config(3, 3, 4, 19))
    require(len(layout) == 36, "three dimensions: (12, 3) layout")
    finite(layout, "three dimensions")
    _ = x^


def case_profile(entry: NativeDescriptorCall, profile: Int) raises:
    # Exact dyadic inputs: columns i/8, ((7i) % 17)/8, ((i*i + 3) % 19)/8.
    var x = List[Float32](length=48, fill=0)
    for i in range(16):
        x[i * 3] = Float32(i) / 8.0
        x[i * 3 + 1] = Float32((i * 7) % 17) / 8.0
        x[i * 3 + 2] = Float32((i * i + 3) % 19) / 8.0
    var cfg: UmapConfigWords
    if profile == 0:
        cfg = config(4, 2, 12, 0, min_dist=0.0, spread=1.0, mix=1.0)
    elif profile == 1:
        cfg = config(5, 3, 12, 7, min_dist=0.25, spread=1.5, mix=0.5)
    else:
        cfg = config(6, 2, 12, 31, min_dist=0.5, spread=2.0, mix=0.75)
    var tag = "profile " + String(profile)
    var layout = fit(entry, x, 16, 3, cfg)
    require(len(layout) == 16 * cfg.n_components, tag + ": layout extent")
    finite(layout, tag)
    var low = layout[0]
    var high = layout[0]
    for v in layout:
        low = min(low, v)
        high = max(high, v)
    require(high - low > 0.0, tag + ": the layout spans a positive range")
    if cfg.n_components == 3:
        var pins = broader_layout_bits()
        var moved = 0
        for i in range(48):
            moved += Int(bitcast[DType.uint32](layout[i]) != pins[i])
        require(moved == 0, tag + ": " + String(moved) + " words differ from BROADER_LAYOUT_BITS")
    var again = fit(entry, x, 16, 3, cfg)
    require(same_bits(layout, again), tag + ": a second fit moved the bits")
    _ = x^


def case_optimizer_controls(entry: NativeDescriptorCall) raises:
    var x = surface_x()
    var cfg = config(3, 2, 4, 19)
    var baseline = fit(entry, x, 8, 1, cfg)
    # The explicit default rates (the extended word layout) equal the legacy one.
    var explicit = List[Float32](length=16, fill=_SENTINEL)
    _ = umap_fit_call(entry, _f(x), _f(explicit), 8, 1, cfg, True)
    require(same_bits(baseline, explicit), "optimizer controls: explicit default rates moved the layout")
    for control in range(3):
        var changed = config(3, 2, 4, 19, rate=0.5 if control == 0 else 1.0,
            repulsion=2.0 if control == 1 else 1.0, negatives=0 if control == 2 else 5)
        var tag = "optimizer control " + String(control)
        var layout = fit(entry, x, 8, 1, changed)
        finite(layout, tag)
        require(not same_bits(layout, baseline), tag + ": the control did not change the layout")
        var again = fit(entry, x, 8, 1, changed)
        require(same_bits(layout, again), tag + ": a second fit moved the bits")
    _ = x^
    _ = explicit^


def case_transform_unseen(entry: NativeDescriptorCall) raises:
    var x = surface_x()
    var cfg = config(3, 2, 12, 19)
    var fitted = fit(entry, x, 8, 1, cfg)
    var saved = fitted.copy()
    var queries: List[Float32] = [0.5, 3.0, 17.0]
    var first = transform(entry, x, fitted, queries, 8, 3, 1, cfg)
    var second = transform(entry, x, fitted, queries, 8, 3, 1, cfg)
    finite(first, "unseen transform")
    require(same_bits(first, second), "unseen transform: a repeated transform moved the bits")
    require(same_bits(fitted, saved), "unseen transform: the transform wrote the fitted embedding")
    require(abs(first[0] - first[4]) > 0 or abs(first[1] - first[5]) > 0,
        "unseen transform: the first and last queries land on the same point")
    _ = x^
    _ = queries^


def case_transform_three_dimensions(entry: NativeDescriptorCall) raises:
    var x = List[Float32](length=12, fill=0)
    for i in range(12):
        x[i] = Float32(i)
    var cfg = config(3, 3, 12, 19)
    var fitted = fit(entry, x, 12, 1, cfg)
    var query: List[Float32] = [2.5]
    var result = transform(entry, x, fitted, query, 12, 1, 1, cfg)
    require(len(result) == 3, "three-dimensional single query: (1, 3) result")
    finite(result, "three-dimensional single query")
    _ = x^
    _ = query^


def case_transform_controls(entry: NativeDescriptorCall) raises:
    var x = surface_x()
    var queries: List[Float32] = [0.5, 3.0, 17.0]
    for control in range(3):
        var cfg = config(3, 2, 12, 19, rate=0.5 if control == 0 else 1.0,
            repulsion=2.0 if control == 1 else 1.0, negatives=0 if control == 2 else 5)
        var tag = "transform control " + String(control)
        var fitted = fit(entry, x, 8, 1, cfg)
        var result = transform(entry, x, fitted, queries, 8, 3, 1, cfg)
        finite(result, tag)
        var again = transform(entry, x, fitted, queries, 8, 3, 1, cfg)
        require(same_bits(result, again), tag + ": a repeated transform moved the bits")
    _ = x^
    _ = queries^


def fixture(entry: NativeDescriptorCall, unit: Int) raises:
    if unit == 0:
        case_public_layout(entry)
    elif unit == 1:
        case_three_dimensions(entry)
    elif unit <= 4:
        case_profile(entry, unit - 2)
    elif unit == 5:
        case_optimizer_controls(entry)
    elif unit == 6:
        case_transform_unseen(entry)
    elif unit == 7:
        case_transform_three_dimensions(entry)
    else:
        case_transform_controls(entry)


# ---- The host column (was neighbors_umap_surface_cpu_suite.mojo) ----
# UMAP fit/transform numerics on the CPU column; independent fixtures run across host workers.
def _run_cpu(mut report: Report) raises:
    var cases = report.cases(UMAP_SURFACE_FIXTURES)
    def task(unit: Int) {imm cases}:
        try:
            fixture(umap_ops_host_v1, unit)
        except error:
            cases.fail(unit, error)
    report.parallel(task, UMAP_SURFACE_FIXTURES)


# ---- The device column (was neighbors_umap_surface_gpu_suite.mojo) ----
# UMAP fit/transform numerics on the device column.
#
# The entries share the metrics image's device context, so the fixtures submit
# in order (as `umap_ops_gpu_suite.mojo` does); the kernels keep their native
# parallelism.
def _run_gpu(mut report: Report) raises:
    var cases = report.cases(UMAP_SURFACE_FIXTURES)
    for unit in range(UMAP_SURFACE_FIXTURES):
        try:
            fixture(umap_ops_device_v1, unit)
        except error:
            cases.fail(unit, error)


def run_cases(column: Column, mut report: Report):
    try:
        comptime if CPU_COLUMN_BUILD:
            _run_cpu(report)
        else:
            _run_gpu(report)
    except error:
        report.abort(error)


def register() -> Check:
    return Check("neighbors_umap_surface", cpu=True, gpu=True, serial="", run=run_cases)
