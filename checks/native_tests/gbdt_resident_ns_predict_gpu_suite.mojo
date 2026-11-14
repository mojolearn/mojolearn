# SPDX-License-Identifier: Apache-2.0
"""The resident GBDT predict of a Depthwise / Lossguide model, bit for bit.

Lane fix-gbdt-prebinned-infer (2026-11-13). `ResidentGbdtModel` applies a
non-symmetric ensemble through the pack it holds from the first predict
(`IDN_NS_RESIDENT_PACK`, `NsResidentPack` / `ns_resident_apply` in
`gbdt/models/add_non_symmetric_tree_doc_parallel.mojo`); the per-call pack
(`predict`, `-D MOJOLEARN_IDN_GBDT_PREBINNED_INFER_OLD`) is the reference.
Checked per fixture (raw input, pre-binned input as the board's
`ours-prebinned` arm feeds it: every value replaced by an integer bin id
held in float32):

- a large batch, a small batch, the large batch again and the small batch
  again through ONE resident model (the held node offsets and bins buffer
  are rewritten only when the row count changes) each equal, bit for bit,
  `predict_floats` (the per-call parse and pack) on the same rows;
- the large batch handed over row-major equals the column-major result.

Fixtures run in order on one GPU (ordered GPU work on one process context);
the fixture tiling and the compares run one task per column or row across
the host cores. Build both forms: `pixi run check-native-gbdt-resident-ns`
and `pixi run check-native-gbdt-resident-ns-old`.
"""
from max.gpu.host import DeviceContext
from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_IDENTICAL
from checks.kernel_matrix import TARGET_COLUMN, COLUMN_CPU
from core.host_parallel import host_parallelize
from gbdt.models.model_text import model_text
from gbdt.resident_model import RESIDENT_RAW, ResidentGbdtModel
from gbdt.train import TrainedModel, predict_floats, train

comptime NS_FIXTURES = 3
comptime N_ROWS = 4096
comptime N_FEATURES = 8
#: a batch past one launch grid's rows and the fixture's own size, and a
#: small one, so the held offsets and bins buffer change size twice
comptime LARGE_ROWS = 300_000
comptime SMALL_ROWS = 1000


def require(ok: Bool, message: String) raises:
    if not ok:
        raise Error(message)


def _fixture_x(prebinned: Bool) -> List[Float32]:
    """`grow_policy_check`'s hashed columns (column-major). Pre-binned: each
    value replaced by its integer bin id over 40 equal-width bins."""
    var x = List[Float32](length=N_ROWS * N_FEATURES, fill=Float32(0.0))
    var xp = MutPointer[Float32, MutUntrackedOrigin](unsafe_from_address=Int(x.unsafe_ptr()))

    def column(feat: Int) {imm xp, imm prebinned}:
        for r in range(N_ROWS):  # small-loop(N_ROWS: rows of one fixture column, inside one task)
            var h = (r * 2654435761 + feat * 97003) % 10000
            var v = Float32(h) / Float32(10000.0) - Float32(0.5)
            if prebinned:
                v = Float32(Int((v + Float32(0.5)) * Float32(40.0)))
            xp.unsafe_store(feat * N_ROWS + r, v)

    host_parallelize(column, N_FEATURES)
    return x^


def _fixture_y(x: List[Float32]) -> List[Float32]:
    var y = List[Float32](length=N_ROWS, fill=Float32(0.0))
    var yp = MutPointer[Float32, MutUntrackedOrigin](unsafe_from_address=Int(y.unsafe_ptr()))
    var xp = MutPointer[Float32, MutUntrackedOrigin](unsafe_from_address=Int(x.unsafe_ptr()))

    def row(r: Int) {imm xp, imm yp}:
        var f0 = xp[0 * N_ROWS + r]
        var f3 = xp[3 * N_ROWS + r]
        var f5 = xp[5 * N_ROWS + r]
        var mid = xp[0 * N_ROWS + (N_ROWS // 2)]
        var target = Float32(3.0) * f0 - Float32(2.0) * f3 + (Float32(4.0) * f5 if f0 > mid else Float32(0.0))
        yp.unsafe_store(r, target)

    host_parallelize(row, N_ROWS)
    return y^


def _tiled(x: List[Float32], n: Int, row_major: Bool) -> List[Float32]:
    """`n` rows cycling the fixture's rows, column-major or row-major."""
    var t = List[Float32](length=n * N_FEATURES, fill=Float32(0.0))
    var tp = MutPointer[Float32, MutUntrackedOrigin](unsafe_from_address=Int(t.unsafe_ptr()))
    var xp = MutPointer[Float32, MutUntrackedOrigin](unsafe_from_address=Int(x.unsafe_ptr()))

    def column(feat: Int) {imm tp, imm xp, imm n, imm row_major}:
        for r in range(n):  # small-loop(n: rows of one tiled column, inside one task)
            var v = xp[feat * N_ROWS + r % N_ROWS]
            if row_major:
                tp.unsafe_store(r * N_FEATURES + feat, v)
            else:
                tp.unsafe_store(feat * n + r, v)

    host_parallelize(column, N_FEATURES)
    return t^


def _same_words(a: List[Float32], b: List[Float32], count: Int) raises -> Bool:
    """Bit-for-bit equality of `count` float32 words, one task per word."""
    if len(a) < count or len(b) < count:
        return False
    var failures = List[Int](length=count, fill=0)
    var fp = MutPointer[Int, MutUntrackedOrigin](unsafe_from_address=Int(failures.unsafe_ptr()))
    var ap = MutPointer[Int32, MutUntrackedOrigin](unsafe_from_address=Int(a.unsafe_ptr()))
    var bp = MutPointer[Int32, MutUntrackedOrigin](unsafe_from_address=Int(b.unsafe_ptr()))

    def cell(index: Int) {imm fp, imm ap, imm bp}:
        fp.unsafe_store(index, Int(ap[index] != bp[index]))

    host_parallelize(cell, count)
    var bad = 0
    for failure in failures:
        bad += failure
    _ = failures^
    return bad == 0


def _resident(mut model: ResidentGbdtModel, x: List[Float32], n: Int, row_major: Bool) raises -> List[Float32]:
    var out = List[Float32](length=n, fill=Float32(0.0))
    var out64 = List[Float64](length=1, fill=0.0)
    var codes = List[Int64](length=1, fill=0)
    var width = model.predict_into(
        MutPointer[Float32, MutUntrackedOrigin](unsafe_from_address=Int(x.unsafe_ptr())),
        n,
        MutPointer[Float32, MutUntrackedOrigin](unsafe_from_address=Int(out.unsafe_ptr())),
        MutPointer[Float64, MutUntrackedOrigin](unsafe_from_address=Int(out64.unsafe_ptr())),
        MutPointer[Int64, MutUntrackedOrigin](unsafe_from_address=Int(codes.unsafe_ptr())),
        RESIDENT_RAW,
        row_major,
    )
    require(width == 1, "resident raw predict wrote width " + String(width) + ", expected 1")
    _ = out64^
    _ = codes^
    _ = x
    return out^


def fixture(unit: Int) raises:
    var prebinned = unit != 0
    var policy = String("Lossguide") if unit == 2 else String("Depthwise")
    var name = policy + (String(" pre-binned") if prebinned else String(" raw"))
    var ctx = DeviceContext()
    var x = _fixture_x(prebinned)
    var y = _fixture_y(x)
    var tm = train(
        ctx, x, y, N_ROWS, N_FEATURES,
        border_count=32,
        n_estimators=12,
        max_depth=6,
        learning_rate=Float32(0.3),
        grow_policy=policy,
        max_leaves=16 if unit == 2 else -1,
    )
    var text = model_text(tm)
    var large = _tiled(x, LARGE_ROWS, False)
    var small = _tiled(x, SMALL_ROWS, False)
    var large_rm = _tiled(x, LARGE_ROWS, True)
    var ref_large = predict_floats(ctx, tm, large, LARGE_ROWS)
    var ref_small = predict_floats(ctx, tm, small, SMALL_ROWS)
    var model = ResidentGbdtModel(text)
    require(not model.oblivious, name + ": the fixture model is non-symmetric")
    # driver-loop: over the four calls of one resident model (ordered GPU work, the held pack's row count)
    for call in range(4):
        var is_large = call % 2 == 0
        var n = LARGE_ROWS if is_large else SMALL_ROWS
        var got = _resident(model, large if is_large else small, n, False)
        require(
            _same_words(got, ref_large if is_large else ref_small, n),
            name + ": resident call " + String(call) + " (" + String(n) + " rows) differs from the per-call predict",
        )
        _ = got^
    var got_rm = _resident(model, large_rm, LARGE_ROWS, True)
    require(_same_words(got_rm, ref_large, LARGE_ROWS), name + ": row-major large batch differs from column-major")
    _ = got_rm^
    _ = model^
    _ = large_rm^
    _ = large^
    _ = small^
    _ = ctx


def run_checks() raises:
    comptime assert GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL and TARGET_COLUMN != COLUMN_CPU
    var failed = 0
    for unit in range(NS_FIXTURES):  # driver-loop: over fixtures, ordered GPU work on one context
        try:
            fixture(unit)
        except error:
            failed += 1
            print("FAIL gbdt_resident_ns_predict_gpu fixture", unit, String(error))
    require(failed == 0, "gbdt_resident_ns_predict_gpu failed " + String(failed) + " fixtures")
    print("PASS gbdt_resident_ns_predict_gpu:", NS_FIXTURES, "non-symmetric fixtures (large/small/large/small, row-major)")
