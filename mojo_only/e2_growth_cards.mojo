"""E2 identity cards for the training paths Python cannot reach.

    pixi run e2-growth-cards <out_dir> [card ...]
    tools/e2_mojo_cards.sh <out_dir>          # the whole set, plus JSON

NO CATBOOST COUNTERPART: a probe, so `mojo_only/`.

WHAT IT IS FOR. E1 proved one config per family bit-identical across
Apple, AMD and NVIDIA by comparing per-stage identity cards
(`core/identity_trace.mojo`, one TAB record per stage;
`tools/identity_trace_diff.py` names the first stage that moved). E2 is the
sub-feature sweep, and `tools/e2_matrix_fit.py` drives every configuration
the CPython binding can reach. Four training paths CANNOT be reached from
Python -- there is no `grow_policy` slot on `gbdt_fit`, `MultiClassOneVsAll`
is refused by name in `python/mojolearn/ensemble.py`, and the
feature-parallel searcher has no caller but its own check (`UNWIRED.md`).
Each of those needs a MOJO-level emitter or it sits outside the sweep. This
file is that emitter: one card per path, ONE FIT PER FILE (the differ
refuses a file holding more), every tag a position in the algorithm and
never a property of the machine.

THE CARDS, and the fixture each rides:

  gbdt_depthwise         `fit_depthwise_tree` on `depthwise_check.Fixture`
                         (4096 rows, 8 binary + 4 half-byte + 4 one-byte
                         features, HASHED bins, the divergent target),
                         depth 4. The ladder `depthwise_trace_probe` walks.
  gbdt_lossguide         `fit_non_symmetric_tree` under GROW_LOSSGUIDE on
                         the SAME fixture, depth 6, max_leaves 9 -- the
                         `check-lossguide` L1 configuration, through the
                         merged non-symmetric driver's Lossguide branches.
  gbdt_multiclass_ova    `train(loss="MultiClassOneVsAll")` on
                         `multiclass_train_check`'s splitmix fixture (4096
                         rows, 5 features, 5 learnable classes), border
                         count 32, 10 trees, depth 4, lr 0.3. `train()`
                         reads `MOJOLEARN_IDENTITY_TRACE` from the
                         environment, so this probe SETS it for the one
                         call and clears it after; the env constructor
                         appends, so the file is truncated first.
  gbdt_feature_parallel  `fit_feature_parallel_oblivious_tree_structure`
                         (rung 2, `oblivious_tree_structure_searcher.mojo`)
                         on `feature_parallel_identity_check`'s fixture at
                         16,434 rows -- three compression blocks, which is
                         the case two of that gate's sabotages need. The
                         searcher has NO trace plumbing of its own, so this
                         card is OUTPUT-LEVEL: the splits it returned and
                         the per-document `docBins` it cached, hashed from
                         the returned buffers. It localizes nothing inside
                         the searcher; it says whether the searcher's
                         answer is the same answer on every vendor.

EVERY FIXTURE IS A PURE FUNCTION OF CONSTANTS. Hashed bins from a fixed
xorshift/splitmix, targets from those bins, unit weights, an explicit
`sm_count` where the searcher takes one (the feature-parallel card passes
the gate's literal 10). Nothing reads the platform RNG, the core count or
the clock, so the same rows reach every vendor and a differing hash is the
VENDOR, not the fixture. `depthwise`/`lossguide` leave `sm_count_override`
at -1 on purpose: that is the device's real core count, which is what a
user's fit would see, and `depthwise_trace_probe` already shows the ladder
agrees at this count and at 108.

THE LADDER MUST HAVE BEEN WALKED. Each card asserts a record-count floor,
per `depthwise_trace_probe`: a run that emits nothing and a run whose
stages all agree produce the same empty diff, and only the count tells them
apart (`reached-but-inert`, applied to an instrument).

NOT A MEASUREMENT. Every record drains and copies (identity_trace rule 4).

THE RUN-TO-RUN CONTROL is `tools/e2_mojo_cards.sh`'s job: it emits the set
twice and compares, because under FAST a path with a live float atomic can
legitimately disagree with itself and such a card must be read only under
IDENTICAL. This probe emits; the script judges.
"""

from max.gpu.host import DeviceBuffer, DeviceContext
from std.os import setenv, unsetenv
from std.sys import argv

from core.identity_trace import IdentityTrace, read_trace_lines
from gbdt.gpu_data.compressed_index_builder import build_layout
from gbdt.gpu_data.kernel.binarize import (
    WRITE_BLOCK_SIZE,
    write_compressed_index_kernel,
)
from gbdt.methods.greedy_subsets_searcher.greedy_search_helper import (
    TTreeWorkspace,
)
from gbdt.methods.greedy_subsets_searcher.greedy_search_helper_depthwise import (
    TDepthwiseWorkspace,
    fit_depthwise_tree,
    fit_non_symmetric_tree,
)
from gbdt.methods.oblivious_tree_structure_searcher import (
    fit_feature_parallel_oblivious_tree_structure,
)
from gbdt.options.catboost_options import SCORE_FUNCTION_COSINE
from gbdt.train import train
from mojo_only.depthwise_check import Fixture, default_options
from mojo_only.lossguide_check import lossguide_options
from mojo_only.multiclass_train_check import (
    N_FEATURES as OVA_N_FEATURES,
    N_ROWS as OVA_N_ROWS,
    build_x as ova_build_x,
    learnable_labels as ova_learnable_labels,
)
from mojo_only.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_IDENTICAL


comptime TRACE_ENV = "MOJOLEARN_IDENTITY_TRACE"

# The four names the script and the JSON use. Changing one here changes the
# card filename on every vendor at once, which is the only safe way.
comptime CARD_DEPTHWISE = "gbdt_depthwise"
comptime CARD_LOSSGUIDE = "gbdt_lossguide"
comptime CARD_OVA = "gbdt_multiclass_ova"
comptime CARD_FEATURE_PARALLEL = "gbdt_feature_parallel"

# Record-count floors. A depth-4 depthwise ladder emits ~99 records and the
# (6, 9) lossguide ladder ~110 on this fixture; the OVA fit emits borders +
# per-tree hist/pstats/winners/leaves, ~60 at 10 trees; the feature-parallel
# card is four output records by construction. The floors sit well under
# the observed counts so a legitimately shorter tree on another vendor
# (that is a STRUCTURAL divergence, and the differ's job) still produces a
# card rather than a refusal here, while an unwired trace (zero records)
# cannot pass.
comptime FLOOR_DEPTHWISE = 10
comptime FLOOR_LOSSGUIDE = 10
comptime FLOOR_OVA = 10
comptime FLOOR_FEATURE_PARALLEL = 4


def _assert_walked(name: String, n: Int, floor: Int) raises:
    if n < floor:
        raise Error(
            String("e2_growth_cards: ")
            + name
            + " emitted only "
            + String(n)
            + " records (floor "
            + String(floor)
            + "). The trace is disabled or the stages are not calling it,"
            " and an empty card reads as agreement."
        )


# ---------------------------------------------------------------------------
# gbdt_depthwise / gbdt_lossguide: the non-symmetric driver, two policies
# ---------------------------------------------------------------------------


def card_depthwise(ctx: DeviceContext, path: String) raises -> Int:
    var fx = Fixture(ctx.copy())
    fx.reset()
    var ws = List[TTreeWorkspace]()
    var dws = List[TDepthwiseWorkspace]()
    var tr = IdentityTrace.to_path(path)
    # Machine-independent provenance only: the differ skips comments, but
    # the script's run-to-run control compares whole files.
    tr.header(
        "e2 card gbdt_depthwise: depthwise_check.Fixture (4096 rows,"
        " 8 binary + 4 half-byte + 4 one-byte, hashed bins), depth 4"
    )
    var model = fit_depthwise_tree(
        fx.ctx, fx.n_rows, fx.folds, default_options(4),
        fx.cindex, fx.stats, fx.row_index,
        fx.total_weight, fx.total_gradient,
        ws, dws, tr,
    )
    _ = model^
    _ = ws^
    _ = dws^
    var n = tr.seq
    _ = tr^
    _ = fx^
    return n


def card_lossguide(ctx: DeviceContext, path: String) raises -> Int:
    var fx = Fixture(ctx.copy())
    fx.reset()
    var ws = List[TTreeWorkspace]()
    var dws = List[TDepthwiseWorkspace]()
    var tr = IdentityTrace.to_path(path)
    tr.header(
        "e2 card gbdt_lossguide: depthwise_check.Fixture (4096 rows,"
        " 8 binary + 4 half-byte + 4 one-byte, hashed bins),"
        " GROW_LOSSGUIDE depth 6 max_leaves 9"
    )
    var model = fit_non_symmetric_tree(
        fx.ctx, fx.n_rows, fx.folds, lossguide_options(6, 9),
        fx.cindex, fx.stats, fx.row_index,
        fx.total_weight, fx.total_gradient,
        ws, dws, tr,
    )
    _ = model^
    _ = ws^
    _ = dws^
    var n = tr.seq
    _ = tr^
    _ = fx^
    return n


# ---------------------------------------------------------------------------
# gbdt_multiclass_ova: train() end to end, through the env-read trace
# ---------------------------------------------------------------------------


def card_multiclass_ova(ctx: DeviceContext, path: String) raises -> Int:
    # `train()` constructs `IdentityTrace()` from the environment and that
    # constructor APPENDS (two fits in one process share a file on
    # purpose, see `to_path`). ONE fit per card means: truncate, set, fit,
    # unset -- and the unset matters, because every card after this one in
    # the same process constructs its trace by `to_path` and must not see
    # a stale env pointing at this file.
    with open(path, "w") as fh:
        fh.write("")
    if not setenv(TRACE_ENV, path, True):
        raise Error("e2_growth_cards: setenv(" + TRACE_ENV + ") failed")
    var x = ova_build_x()
    var y = ova_learnable_labels(5)
    var tm = train(
        ctx, x, y, OVA_N_ROWS, OVA_N_FEATURES,
        border_count=32, n_estimators=10, max_depth=4,
        loss="MultiClassOneVsAll", learning_rate=Float32(0.3),
    )
    _ = unsetenv(TRACE_ENV)
    if tm.model.weak_models[0].dim != 5:
        raise Error(
            "e2_growth_cards: the OVA model's dim is "
            + String(tm.model.weak_models[0].dim)
            + ", not 5 -- this is not the one-vs-all path"
        )
    _ = tm^
    return len(read_trace_lines(path))


# ---------------------------------------------------------------------------
# gbdt_feature_parallel: rung 2's searcher, OUTPUT-LEVEL
# ---------------------------------------------------------------------------

comptime FP_N_ROWS = 16434
comptime FP_MAX_DEPTH = 4
comptime FP_SM_COUNT = 10


def card_feature_parallel(ctx: DeviceContext, path: String) raises -> Int:
    # THE FIXTURE IS `feature_parallel_identity_check.run_case`'s, copied
    # rather than imported because that file builds it inline inside the
    # gate; the constants are named here so a drift between the two is a
    # one-line diff. The card hashes OUTPUTS, so a drift would change
    # this card's values on every vendor at once and never its identity.
    var folds: List[Int] = [1, 1, 12, 9, 20, 32, 48, 100, 64, 127]
    var n_features = len(folds)
    var lay = build_layout(folds)

    var cindex = ctx.enqueue_create_buffer[DType.uint32](
        FP_N_ROWS * lay.columns
    )
    ctx.enqueue_memset(cindex, UInt32(0))
    var hb = ctx.enqueue_create_host_buffer[DType.uint8](FP_N_ROWS)
    var bins8 = ctx.enqueue_create_buffer[DType.uint8](FP_N_ROWS)
    var host_bins = List[List[Int]]()
    for f in range(n_features):
        ref cf = lay.features[f]
        var col = List[Int]()
        for r in range(FP_N_ROWS):
            var x = UInt32(r * 2654435761 + f * 40503 + 0x2545F491)
            x ^= x << 13
            x ^= x >> 17
            x ^= x << 5
            var b = Int(x % UInt32(folds[f] + 1))
            col.append(b)
            hb.unsafe_ptr().unsafe_store(r, UInt8(b))
        host_bins.append(col^)
        ctx.enqueue_copy(dst_buf=bins8, src_ptr=hb.unsafe_ptr())
        ctx.enqueue_function[write_compressed_index_kernel](
            Int32(Int(cf.offset) * FP_N_ROWS),
            cf.mask,
            cf.shift,
            bins8.unsafe_ptr(),
            Int32(FP_N_ROWS),
            cindex.unsafe_ptr(),
            grid_dim=(FP_N_ROWS + WRITE_BLOCK_SIZE - 1) // WRITE_BLOCK_SIZE,
            block_dim=WRITE_BLOCK_SIZE,
        )
        ctx.synchronize()
    # `[[mojo-buffer-freed-at-last-use]]`: both are live across the
    # launches above; the keep-alive goes after the last synchronize.
    _ = bins8^
    _ = hb^

    var hs = ctx.enqueue_create_host_buffer[DType.float32](2 * FP_N_ROWS)
    for r in range(FP_N_ROWS):
        var g = Float64(0.0)
        if host_bins[9][r] > 60:
            g += 8.0
        if host_bins[2][r] > 6:
            g += 3.0
        if host_bins[8][r] > 30:
            g += 1.0
        g -= 6.0
        hs.unsafe_ptr().unsafe_store(r, Float32(1.0))
        hs.unsafe_ptr().unsafe_store(FP_N_ROWS + r, Float32(g))
    var w = ctx.enqueue_create_buffer[DType.float32](FP_N_ROWS)
    var gbuf = ctx.enqueue_create_buffer[DType.float32](FP_N_ROWS)
    ctx.enqueue_copy(dst_buf=w, src_ptr=hs.unsafe_ptr())
    ctx.enqueue_copy(
        dst_buf=gbuf, src_ptr=hs.unsafe_ptr().unsafe_offset(FP_N_ROWS)
    )
    ctx.synchronize()
    _ = hs^

    var fp = fit_feature_parallel_oblivious_tree_structure(
        ctx, lay, FP_N_ROWS, FP_MAX_DEPTH, cindex, w^, gbuf^,
        FP_SM_COUNT, Float32(1.0), SCORE_FUNCTION_COSINE,
    )
    var splits = fp[0].copy()
    var doc_bins = fp[1]
    if len(splits) == 0:
        raise Error(
            "e2_growth_cards: the feature-parallel searcher grew NO splits"
        )

    var tr = IdentityTrace.to_path(path)
    tr.header(
        "e2 card gbdt_feature_parallel: feature_parallel_identity_check"
        " fixture (16434 rows, 10 features, hashed bins), depth 4,"
        " sm_count 10. OUTPUT-LEVEL: splits + docBins, no in-searcher"
        " stages"
    )
    var sf = List[Int32]()
    var sb = List[Int32]()
    var st = List[Int32]()
    for i in range(len(splits)):
        sf.append(splits[i].feature_id)
        sb.append(splits[i].bin_idx)
        st.append(splits[i].split_type)
    tr.record_list_i32("fp.splits.feature", sf)
    tr.record_list_i32("fp.splits.bin", sb)
    tr.record_list_i32("fp.splits.type", st)
    tr.record_device(ctx, "fp.docbins", doc_bins, FP_N_ROWS)
    var n = tr.seq
    _ = tr^
    _ = doc_bins^
    _ = cindex^
    return n


# ---------------------------------------------------------------------------


def _wanted(names: List[String], name: String) -> Bool:
    if len(names) == 0:
        return True
    for i in range(len(names)):
        if names[i] == name:
            return True
    return False


def main() raises:
    var args = argv()
    if len(args) < 2:
        raise Error(
            "usage: mojo run -I . mojo_only/e2_growth_cards.mojo <out_dir>"
            " [gbdt_depthwise|gbdt_lossguide|gbdt_multiclass_ova"
            "|gbdt_feature_parallel ...]"
        )
    var out_dir = String(args[1])
    var names = List[String]()
    for i in range(2, len(args)):
        names.append(String(args[i]))

    var mode = String(
        "IDENTICAL"
    ) if GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL else String("FAST")
    print("e2 growth cards, numeric mode", mode, "->", out_dir)

    var ctx = DeviceContext()

    if _wanted(names, CARD_DEPTHWISE):
        var p = out_dir + "/" + CARD_DEPTHWISE + ".card"
        var n = card_depthwise(ctx, p)
        _assert_walked(CARD_DEPTHWISE, n, FLOOR_DEPTHWISE)
        print("  " + CARD_DEPTHWISE + ":", n, "records ->", p)

    if _wanted(names, CARD_LOSSGUIDE):
        var p = out_dir + "/" + CARD_LOSSGUIDE + ".card"
        var n = card_lossguide(ctx, p)
        _assert_walked(CARD_LOSSGUIDE, n, FLOOR_LOSSGUIDE)
        print("  " + CARD_LOSSGUIDE + ":", n, "records ->", p)

    if _wanted(names, CARD_OVA):
        var p = out_dir + "/" + CARD_OVA + ".card"
        var n = card_multiclass_ova(ctx, p)
        _assert_walked(CARD_OVA, n, FLOOR_OVA)
        print("  " + CARD_OVA + ":", n, "records ->", p)

    if _wanted(names, CARD_FEATURE_PARALLEL):
        var p = out_dir + "/" + CARD_FEATURE_PARALLEL + ".card"
        var n = card_feature_parallel(ctx, p)
        _assert_walked(CARD_FEATURE_PARALLEL, n, FLOOR_FEATURE_PARALLEL)
        print("  " + CARD_FEATURE_PARALLEL + ":", n, "records ->", p)

    print("e2 growth cards: done [" + mode + "]")
