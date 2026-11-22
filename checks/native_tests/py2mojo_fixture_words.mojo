# SPDX-License-Identifier: Apache-2.0
"""Fixture words of the Mojo checks that replaced numeric Python tests (lane py2mojo-pilot, 2026-11-22;
`tools/py2mojo_native_fixtures.py` writes them, docs/plans/PY2MOJO_CONVERSION.md). Importable; no `main()`, no
`test_*`. Paths are relative to the repository root, where the checks run. The binary64 words (scikit-learn
reference values) are two little-endian 32-bit words each, low first; the decode is one cell per value across the
host cores.
"""
from std.memory import bitcast
from core.host_parallel import host_parallelize
from checks.native_tests.tsa_metrics_fixture_words import fixture_words

comptime PREP_SANITY_DIR = "checks/native_tests/fixtures/prep_sanity/"
comptime METRICS_FM_DIR = "checks/native_tests/fixtures/metrics_fm/"


def fixture_f64(path: String, count: Int) raises -> List[Float64]:
    var words = fixture_words(path, 2 * count)
    var values = List[Float64](length=count, fill=Float64(0))
    var wp = words.unsafe_ptr()
    var vp = values.unsafe_ptr()
    def cell(index: Int) {imm wp, imm vp}:
        var bits = UInt64(wp[2 * index]) | (UInt64(wp[2 * index + 1]) << UInt64(32))
        vp.unsafe_store(index, bitcast[DType.float64](bits))
    host_parallelize(cell, count)
    _ = words^
    return values^
