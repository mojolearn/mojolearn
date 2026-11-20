# SPDX-License-Identifier: Apache-2.0
"""Typed NMF options, rank-two metadata and owned scratch storage contracts."""
from core.host_parallel import host_parallelize
from core.nmf_native_api import NMFOptions
from core.nmf_native_state import NMFScratch, nmf_output_shape

comptime NMF_STATE_FIXTURES = 20


def require(ok: Bool, message: String) raises:
    if not ok:
        raise Error(message)


def filled_scratch(rows: Int, cols: Int) raises -> NMFScratch:
    var scratch = NMFScratch(rows, cols)
    var p = MutPointer[Float32, MutUntrackedOrigin](unsafe_from_address=scratch.address())
    def cell(index: Int) {imm p}:
        p.unsafe_store(index, Float32(index * 3 + 1))
    host_parallelize(cell, rows * cols)
    _ = scratch.values
    return scratch^


def transfer(var scratch: NMFScratch) -> NMFScratch:
    return scratch^


def fixture(unit: Int) raises:
    if unit == 0 or unit == 1 or unit == 18:
        var options = NMFOptions(0.0 if unit == 18 else 1.5, 1)
        require(options.beta == (0.0 if unit == 18 else 1.5) and options.solver == 1,
                "NMF typed options preserve beta and solver independently")
        if unit == 1:
            var copied = options.copy()
            copied.beta = 2
            copied.solver = 0
            require(options.beta == 1.5 and options.solver == 1 and copied.beta == 2 and copied.solver == 0,
                    "NMF options copies do not alias")
    elif unit < 6:
        var rows = 0 if unit == 3 or unit == 5 else 3
        var cols = 0 if unit == 4 or unit == 5 else 5
        var shape = nmf_output_shape(rows, cols)
        require(len(shape) == 2 and shape[0] == rows and shape[1] == cols,
                "NMF output metadata preserves rank two and each zero dimension")
    elif unit < 9:
        var rows = 0 if unit == 6 or unit == 8 else 3
        var cols = 0 if unit == 7 or unit == 8 else 5
        var scratch = NMFScratch(rows, cols)
        require(len(scratch.values) == 0 and scratch.address() == 0,
                "NMF empty scratch owns no addressable values")
    elif unit < 13:
        var rows = 1
        var cols = 1
        if unit == 10:
            rows = 2
            cols = 3
        elif unit == 11:
            rows = 3
            cols = 2
        elif unit == 12:
            rows = 17
            cols = 5
        var scratch = filled_scratch(rows, cols)
        require(len(scratch.values) == rows * cols and scratch.address() != 0
                and scratch.address() % 4 == 0, "NMF scratch extent/alignment")
        for index in range(rows * cols):
            require(scratch.values[index] == Float32(index * 3 + 1), "NMF scratch owns every written float32 word")
    elif unit < 17:
        var rows = -1 if unit == 13 else 1
        var cols = -1 if unit == 14 else 1
        var expected = String("ValueError: negative dimensions are not allowed")
        if unit == 15:
            rows = Int.MAX
            cols = 2
            expected = "ValueError: array element count exceeds native storage"
        elif unit == 16:
            rows = Int.MAX // 4 + 1
            expected = "ValueError: array byte count exceeds native storage"
        var refused = False
        try:
            var scratch = NMFScratch(rows, cols)
            _ = scratch^
        except error:
            refused = String(error) == expected
        require(refused, "NMF scratch validates dimensions and overflow before allocation")
    elif unit == 17:
        var original = filled_scratch(2, 3)
        var address = original.address()
        var moved = transfer(original^)
        require(moved.address() == address and len(moved.values) == 6,
                "NMF scratch move transfers its existing native allocation")
        for index in range(6):
            require(moved.values[index] == Float32(index * 3 + 1), "NMF moved scratch retains written values")
    else:
        var first = nmf_output_shape(3, 0)
        var second = nmf_output_shape(3, 0)
        first[1] = 5
        require(len(second) == 2 and second[0] == 3 and second[1] == 0,
                "NMF output shape records own independent native lists")


def run_checks() raises:
    var failures = List[Int](length=NMF_STATE_FIXTURES, fill=0)
    var fp = failures.unsafe_ptr()
    def task(unit: Int) {imm fp}:
        try:
            fixture(unit)
        except error:
            fp.unsafe_store(unit, 1)
            print("FAIL nmf_state fixture", unit, String(error))
    host_parallelize(task, NMF_STATE_FIXTURES)
    var failed = 0
    for value in failures:
        failed += value
    require(failed == 0, "nmf_state failed " + String(failed) + " fixtures")
    print("PASS nmf_state:", NMF_STATE_FIXTURES, "native options/shape/scratch fixtures")
