# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Isolation Forest: the RNG unit gate, refusals, oracle sanity, device
identity, launch invariance, signed zero, the card.

DEVIATIONS 680-686's gates. The checks, in order:

    check_if_xorwow_matches_curand     the ported XORWOW (`curand_init` ->
                                       `curand` -> `curand_uniform`) against
                                       `checks/xorwow_reference.tsv`: the
                                       first 1024 draws AND their uniform
                                       bits for six (seed, tree) pairs,
                                       derived by `xorwow_reference.py` from
                                       curand_kernel.h's integer arithmetic
                                       and curand_precalc.h's own tables;
                                       the two float constants by hex
    check_if_refusals                  DEVIATION 680 (NaN / inf in X and
                                       X_query, by name and position),
                                       their asserts (n_rows, n_cols,
                                       n_estimators, max_samples), an
                                       unfitted score, a feature-count
                                       mismatch; DEVIATION 684's
                                       `ceil_log2_int` against float64
                                       `ceil(log2(n))` for n = 1..4097;
                                       `compute_global_max_nodes_per_tree`
    check_if_oracle_semantics          the float32 oracle: PLANTED OUTLIERS
                                       score above the blob median (paper
                                       scores) and are the shortest paths;
                                       duplicate rows score bit-identically;
                                       float32 scores within 2e-4 of the
                                       float64 reference over the same
                                       trees; n_rows < max_samples fits
                                       with n_sampled_rows = n_rows
    check_if_device_equals_oracle      SIX fixtures (blob+outliers+dups+
                                       constant column+signed-zero column;
                                       bootstrap; max_features 5 of 8;
                                       n_rows 100 < max_samples; a pure
                                       blob, 24 trees; max_depth 3):
                                       per tree n_nodes, max_depth, sampled
                                       rows, sampled features, and the four
                                       structure arrays PER CELL; path
                                       lengths and scores PER CELL, bit for
                                       bit under IDENTICAL; a REPORT under
                                       FAST. Plus REACH: the six fixtures'
                                       forests pairwise differ
    check_if_launch_invariance         THE HEADLINE: every structure array,
                                       path length and score byte-identical
                                       across build_tpb 32/128/256,
                                       path_tpb 256/64, pad 0/37, two
                                       poisons, run twice; and the SAME
                                       query scored alone, inside a batch
                                       of 37 and at three positions of a
                                       4096-row batch of other hashed rows
    check_if_row39_signed_zero         ADDENDUM 11: the signed-zero column
                                       with the -0.0/+0.0 positions SWAPPED
                                       grows the same forest bit for bit on
                                       device and oracle (the strict-compare
                                       fold is positional; no stored bit
                                       depends on which zero a node calls its
                                       min), and `candidate_min <
                                       candidate_max` refuses a {-0,+0} column
                                       as constant on both
    check_if_predict_thresholds        `predict` labels are `score >
                                       threshold ? 1 : -1` cell by cell;
                                       `contamination`'s percentile offset
                                       (the Python layer's `cp.percentile`,
                                       `estimator.mojo`) flags exactly the
                                       planted share
    check_if_card_is_emitted           the card's tag list (rng probe, per
                                       tree rows/meta/structure x4, pathlen,
                                       scores) and its run-to-run control

SABOTAGES (each a build define, each a no-op unless named):

    -D MOJOLEARN_IF_SABOTAGE_U64_SWAP=1   `curand_u64` takes its two draws
                                          in the other order -- the other
                                          conforming reading of cuML's
                                          unsequenced `|` (DEVIATION 750).
                                          Perturbs the RNG stream and
                                          nothing else. MUST turn
                                          `check_if_device_equals_oracle`
                                          red (the oracle keeps its own
                                          `_u64`), and MUST leave the two
                                          xorwow gates green, because they
                                          do not go through `curand_u64`
                                          at all -- that split is the
                                          per-branch reach evidence.

Run both arms:

    tools/with_build_lock.sh     pixi run mojo run -I . isolation_forest/checks/if_check.mojo
    tools/with_identical_mode.sh pixi run mojo run -I . isolation_forest/checks/if_check.mojo
"""

from std.math import ceil, log2
from std.os import getenv, makedirs, path
from std.memory import bitcast

from std.gpu import block_dim, block_idx, thread_idx
from max.gpu.host import DeviceContext

from core.identity_trace import IdentityTrace, first_divergence, read_trace_lines
from isolation_forest.estimator import (
    IsolationForestEstimator,
    percentile_linear,
)
from isolation_forest.checks.if_fixture import (
    bits_value,
    blob_fixture,
    is_planted_outlier,
    mix64,
    plant_constant_column,
    plant_duplicates,
    plant_signed_zero_column,
    to_column_major,
)
from isolation_forest.checks.if_oracle import (
    OracleForest,
    oracle_fit,
    oracle_path_lengths,
    oracle_scores,
    oracle_scores_f64,
)
from isolation_forest.impl.curand.curand_kernel import (
    CURAND_2POW32_INV,
    XORWOW_TABLE_WORDS,
    XorwowTables,
    build_xorwow_tables,
    curand,
    curand_init,
    curandStateXORWOW,
    _curand_uniform,
)
from isolation_forest.impl.isolation_forest.isolation_forest import (
    IF_params,
    IFLaunchKnobs,
    IsolationForestModel,
    ceil_log2_int,
    check_finite_by_name,
    compute_global_max_nodes_per_tree,
    fit,
    path_lengths,
    predict,
    read_f32,
    read_i32,
    read_i64,
    score_samples,
)
from isolation_forest.impl.isolation_forest.isolation_tree_builder import (
    EULER_MASCHERONI_F32,
)
from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_IDENTICAL, numeric_mode_name


comptime IDENTICAL = GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL
comptime SCRATCH = "/private/tmp/mojolearn_if_cards"
"""Where `check_if_card_is_emitted` writes its two control cards. A FIXED
path under /private/tmp, not a session scratchpad: this file used to name
`.../claude-501/-Users-.../c57d661c-.../scratchpad`, one dead session's
directory, so the gate was green only on the box where that directory
still happened to exist. `IdentityTrace.to_path` creates the file; the
directory is created by `_ensure_scratch()` below."""


def card_path() -> String:
    """`MOJOLEARN_IDENTITY_TRACE` when the caller set it, else SCRATCH.

    DEVIATION 1932, 2026-08-28. Same precedence as
    `mamba/checks/mamba_check.mojo::card_path` and
    `bench/gemm_card_main.mojo:553`, and here for the same reason mamba
    needed it (DEVIATION 970): this lane BUILT a full card and then wrote it
    to a scratch path nobody collects, so isolation forest had zero cells in
    every cross-vendor round while its own gate went green. Read at RUN time,
    never at compile time, because the harness chooses the directory.
    """
    var p = String(getenv("MOJOLEARN_IDENTITY_TRACE"))
    if p.byte_length() > 0:
        return p^
    return String(SCRATCH) + "/if_check_card_a.trace"


def _ensure_scratch() raises:
    """`mkdir -p SCRATCH`. Cheap, idempotent, and the gate cannot be green
    by luck any more."""
    if not path.exists(String(SCRATCH)):
        makedirs(String(SCRATCH))


def _mode_name() -> String:
    """The build's tier, from the ONE definition of it.

    Delegates to `numeric_mode_name()` since 2026-08-29; see the note
    on that function. A local two-way IDENTICAL-or-FAST answers "FAST"
    for a DETERMINISTIC build, which mislabels every line the driver
    prints.
    """
    return numeric_mode_name()


def _tag() -> String:
    return " [" + _mode_name() + "]"


def _hex32(v: Float32) -> String:
    comptime DIGITS = "0123456789abcdef"
    var u = bitcast[DType.uint32](v)
    var out = String("0x")
    for i in range(8):
        var nib = Int((u >> UInt32(28 - 4 * i)) & UInt32(0xF))
        out += String(DIGITS[byte=nib])
    return out


def _hex32u(u: UInt32) -> String:
    return _hex32(bitcast[DType.float32](u))


def _parse_u64(s: String) -> UInt64:
    """Decimal digits into a UInt64 (`Int(String)` refuses values past
    2^63; the reference carries a seed with the top bit set)."""
    var v: UInt64 = 0
    for cp in s.codepoints():
        var c = Int(cp)
        if c < 48 or c > 57:
            break
        v = v * 10 + UInt64(c - 48)
    return v


def _parse_hex32(s: String) raises -> UInt32:
    """Eight lowercase hex digits (the reference TSVs' `uniform_f32_bits`
    column) into a UInt32. Raises on anything else rather than stopping
    early: a silent 0 here would have made the device gate compare the
    right bits against nothing (it did, once)."""
    var v: UInt32 = 0
    var n = 0
    for cp in s.codepoints():
        var c = Int(cp)
        var d = -1
        if c >= 48 and c <= 57:
            d = c - 48
        elif c >= 97 and c <= 102:
            d = c - 87
        elif c >= 65 and c <= 70:
            d = c - 55
        else:
            raise Error("not a hex digit in '" + s + "'")
        v = v * 16 + UInt32(d)
        n += 1
    if n != 8:
        raise Error("expected 8 hex digits, got " + String(n) + " in '" + s + "'")
    return v


def _bits_equal(a: Float32, b: Float32) -> Bool:
    return bitcast[DType.uint32](a) == bitcast[DType.uint32](b)


def _expect_raise(what: String, raised: Bool, msg: String, must_contain: String) raises:
    if not raised:
        raise Error("check_if_refusals: " + what + " did NOT raise")
    if msg.find(must_contain) < 0:
        raise Error(
            "check_if_refusals: " + what + " raised without naming '" + must_contain + "': " + msg
        )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


struct Fixture(Movable):
    var name: String
    var x: List[Float32]
    var n: Int
    var d: Int
    var params: IF_params

    def __init__(out self, name: String, var x: List[Float32], n: Int, d: Int, params: IF_params):
        self.name = name
        self.x = x^
        self.n = n
        self.d = d
        self.params = params.copy()


def _params(n_estimators: Int, max_samples: Int, seed: Int) -> IF_params:
    var p = IF_params.default()
    p.n_estimators = n_estimators
    p.max_samples = max_samples
    p.seed = UInt64(seed)
    return p^


def _fixture_main(n: Int, d: Int, salt: Int, n_outliers: Int) -> List[Float32]:
    """Blob + planted outliers + 6 duplicates of row 40 (stride 97) +
    constant column 5 + signed-zero column 6."""
    var x = blob_fixture(n, d, salt, n_outliers)
    plant_constant_column(x, n, d, 5, salt)
    plant_signed_zero_column(x, n, d, 6)
    plant_duplicates(x, n, d, 40, 6, 97)  # LAST, so the copies stay copies
    return x^


def _fixtures() -> List[Fixture]:
    var out = List[Fixture]()
    out.append(Fixture("main(blob+outliers+dups+const+zeros)", _fixture_main(1024, 8, 11, 16), 1024, 8, _params(16, 256, 42)))
    var p2 = _params(12, 256, 7)
    p2.bootstrap = True
    out.append(Fixture("bootstrap", _fixture_main(1024, 8, 23, 16), 1024, 8, p2))
    var p3 = _params(12, 256, 99)
    p3.max_features = 5
    out.append(Fixture("max_features=5/8", _fixture_main(1024, 8, 31, 16), 1024, 8, p3))
    out.append(Fixture("n_rows=100<max_samples", _fixture_main(100, 8, 5, 4), 100, 8, _params(8, 256, 3)))
    out.append(Fixture("pure blob, 24 trees", blob_fixture(2048, 5, 17, 0), 2048, 5, _params(24, 256, 1234567)))
    var p6 = _params(8, 512, 77)
    p6.max_depth = 3
    out.append(Fixture("max_depth=3, max_samples=512", blob_fixture(1500, 7, 41, 8), 1500, 7, p6))
    return out^


# ---------------------------------------------------------------------------
# A device fit read back whole, for per-cell comparison
# ---------------------------------------------------------------------------


struct FitDump(Movable):
    var n_nodes: List[Int32]
    var max_depth: List[Int32]
    var offsets: List[Int32]
    var feat: List[Int32]
    var thr: List[Float32]
    var left: List[Int32]
    var right: List[Int32]
    var features: List[Int32]
    var max_nodes_per_tree: Int
    var n_features_per_tree: Int
    var n_samples_per_tree: Int
    var c_normalization: Float64
    var pathlen: List[Float32]
    var scores: List[Float32]

    def __init__(out self):
        self.n_nodes = List[Int32]()
        self.max_depth = List[Int32]()
        self.offsets = List[Int32]()
        self.feat = List[Int32]()
        self.thr = List[Float32]()
        self.left = List[Int32]()
        self.right = List[Int32]()
        self.features = List[Int32]()
        self.max_nodes_per_tree = 0
        self.n_features_per_tree = 0
        self.n_samples_per_tree = 0
        self.c_normalization = 0.0
        self.pathlen = List[Float32]()
        self.scores = List[Float32]()


def _device_fit(
    ctx: DeviceContext,
    fx: Fixture,
    query: List[Float32],
    n_query: Int,
    knobs: IFLaunchKnobs,
    trace_path: String = "",
) raises -> FitDump:
    var trace = IdentityTrace.to_path(trace_path)
    var model = IsolationForestModel(ctx)
    var x_col = to_column_major(fx.x, fx.n, fx.d)
    fit(ctx, model, x_col, fx.n, fx.d, fx.params, trace, knobs)
    var dump = FitDump()
    var n_trees = fx.params.n_estimators
    dump.n_nodes = model.tree_n_nodes_host.copy()
    dump.max_depth = model.tree_max_depth_host.copy()
    dump.offsets = read_i32(ctx, model.global_tree_offsets, n_trees)
    var total = n_trees * model.max_nodes_per_tree
    dump.feat = read_i32(ctx, model.node_feature, total)
    dump.thr = read_f32(ctx, model.node_threshold, total)
    dump.left = read_i32(ctx, model.node_left, total)
    dump.right = read_i32(ctx, model.node_right, total)
    if model.has_feature_indices:
        dump.features = read_i32(ctx, model.global_feature_indices, n_trees * model.n_features_per_tree)
    dump.max_nodes_per_tree = model.max_nodes_per_tree
    dump.n_features_per_tree = model.n_features_per_tree
    dump.n_samples_per_tree = model.n_samples_per_tree
    dump.c_normalization = model.c_normalization
    dump.pathlen = path_lengths(ctx, model, query, n_query, fx.d, knobs)
    dump.scores = score_samples(ctx, model, query, n_query, fx.d, trace, knobs)
    return dump^


def _compare_fit_to_oracle(
    fx: Fixture, dump: FitDump, orc: OracleForest, o_pl: List[Float32], o_sc: List[Float32], n_query: Int, who: String
) raises -> String:
    """Returns "" when everything matches, else the first difference."""
    var n_trees = fx.params.n_estimators
    if dump.max_nodes_per_tree != orc.max_nodes_per_tree:
        return who + " max_nodes_per_tree " + String(dump.max_nodes_per_tree) + " vs " + String(orc.max_nodes_per_tree)
    if dump.n_samples_per_tree != orc.n_samples_per_tree or dump.n_features_per_tree != orc.n_features_per_tree:
        return who + " n_samples/features_per_tree differ"
    if bitcast[DType.uint64](dump.c_normalization) != bitcast[DType.uint64](orc.c_normalization):
        return who + " c_normalization " + String(dump.c_normalization) + " vs " + String(orc.c_normalization)
    for t in range(n_trees):
        var ot = Pointer(to=orc.trees[t])
        if Int(dump.n_nodes[t]) != ot[].n_nodes:
            return who + " tree " + String(t) + " n_nodes " + String(dump.n_nodes[t]) + " vs " + String(ot[].n_nodes)
        if Int(dump.max_depth[t]) != ot[].max_depth:
            return who + " tree " + String(t) + " max_depth " + String(dump.max_depth[t]) + " vs " + String(ot[].max_depth)
        if Int(dump.offsets[t]) != t * dump.max_nodes_per_tree:
            return who + " tree " + String(t) + " offset " + String(dump.offsets[t])
        if len(dump.features) > 0:
            for i in range(dump.n_features_per_tree):
                if dump.features[t * dump.n_features_per_tree + i] != ot[].features[i]:
                    return who + " tree " + String(t) + " feature " + String(i) + " differs"
        var off = t * dump.max_nodes_per_tree
        for i in range(ot[].n_nodes):
            if dump.feat[off + i] != ot[].feat[i]:
                return who + " tree " + String(t) + " node " + String(i) + " feat " + String(dump.feat[off + i]) + " vs " + String(ot[].feat[i])
            if not _bits_equal(dump.thr[off + i], ot[].thr[i]):
                return who + " tree " + String(t) + " node " + String(i) + " thr " + _hex32(dump.thr[off + i]) + " vs " + _hex32(ot[].thr[i])
            if dump.left[off + i] != ot[].left[i] or dump.right[off + i] != ot[].right[i]:
                return who + " tree " + String(t) + " node " + String(i) + " children differ"
    for i in range(n_query):
        if not _bits_equal(dump.pathlen[i], o_pl[i]):
            return who + " pathlen[" + String(i) + "] " + _hex32(dump.pathlen[i]) + " vs " + _hex32(o_pl[i])
    for i in range(n_query):
        if not _bits_equal(dump.scores[i], o_sc[i]):
            return who + " score[" + String(i) + "] " + _hex32(dump.scores[i]) + " vs " + _hex32(o_sc[i])
    return String("")


def _compare_dumps(a: FitDump, b: FitDump, n_trees: Int, n_query: Int, who: String) -> String:
    if a.max_nodes_per_tree != b.max_nodes_per_tree:
        return who + " max_nodes_per_tree"
    for t in range(n_trees):
        if a.n_nodes[t] != b.n_nodes[t] or a.max_depth[t] != b.max_depth[t]:
            return who + " tree " + String(t) + " n_nodes/max_depth"
        var off = t * a.max_nodes_per_tree
        for i in range(Int(a.n_nodes[t])):
            if a.feat[off + i] != b.feat[off + i] or not _bits_equal(a.thr[off + i], b.thr[off + i]) or a.left[off + i] != b.left[off + i] or a.right[off + i] != b.right[off + i]:
                return who + " tree " + String(t) + " node " + String(i) + " thr " + _hex32(a.thr[off + i]) + " vs " + _hex32(b.thr[off + i])
    if len(a.features) != len(b.features):
        return who + " features length"
    for i in range(len(a.features)):
        if a.features[i] != b.features[i]:
            return who + " feature " + String(i)
    for i in range(n_query):
        if not _bits_equal(a.pathlen[i], b.pathlen[i]):
            return who + " pathlen[" + String(i) + "] " + _hex32(a.pathlen[i]) + " vs " + _hex32(b.pathlen[i])
        if not _bits_equal(a.scores[i], b.scores[i]):
            return who + " score[" + String(i) + "] " + _hex32(a.scores[i]) + " vs " + _hex32(b.scores[i])
    return String("")


def _query_rows(fx: Fixture, n_query: Int) -> List[Float32]:
    """The first `n_query` rows of the fixture (planted outliers among
    them), as the query batch."""
    var q = List[Float32]()
    for i in range(n_query * fx.d):
        q.append(fx.x[i])
    return q^


# ---------------------------------------------------------------------------
# 1. The RNG unit gate
# ---------------------------------------------------------------------------


def check_if_xorwow_matches_curand() raises:
    var tables = build_xorwow_tables()
    var seqp = tables.sequence.unsafe_ptr().unsafe_mut_cast[True]().unsafe_origin_cast[MutAnyOrigin]()
    var offp = tables.offset.unsafe_ptr().unsafe_mut_cast[True]().unsafe_origin_cast[MutAnyOrigin]()
    var text: String
    with open("isolation_forest/checks/xorwow_reference.tsv", "r") as fh:
        text = fh.read()
    var lines = text.split("\n")
    var n_checked = 0
    var n_pairs = 0
    var cur_seed: UInt64 = 0
    var cur_tree: UInt64 = 0
    var have = False
    var st = curandStateXORWOW.zero()
    var expect_idx = 0
    for li in range(len(lines)):
        var line = String(lines[li])
        if line == "" or line.startswith("#"):
            continue
        var cols = line.split("\t")
        if len(cols) != 5:
            raise Error("xorwow_reference.tsv: bad line " + line)
        var seed = _parse_u64(String(cols[0]))
        var tree = _parse_u64(String(cols[1]))
        var idx = Int(String(cols[2]))
        var u = UInt32(_parse_u64(String(cols[3])))
        var uni_hex = String(cols[4])
        if not have or seed != cur_seed or tree != cur_tree:
            cur_seed = seed
            cur_tree = tree
            have = True
            st = curandStateXORWOW.zero()
            curand_init(seed, tree, UInt64(0), st, seqp, offp)
            expect_idx = 0
            n_pairs += 1
        if idx != expect_idx:
            raise Error("xorwow_reference.tsv: index gap at " + line)
        var got = curand(st)
        if got != u:
            raise Error(
                "check_if_xorwow_matches_curand FAILED: seed " + String(seed) + " tree " + String(tree)
                + " draw " + String(idx) + " got " + String(got) + " reference " + String(u)
            )
        var uni = bitcast[DType.uint32](_curand_uniform(got))
        var uni_s = _hex32u(uni)
        if uni_s != "0x" + uni_hex:
            raise Error(
                "check_if_xorwow_matches_curand FAILED: uniform bits at seed " + String(seed) + " tree "
                + String(tree) + " draw " + String(idx) + " got " + uni_s + " reference 0x" + uni_hex
            )
        expect_idx += 1
        n_checked += 1
    if n_pairs != 6 or n_checked != 6 * 1024:
        raise Error("check_if_xorwow_matches_curand: expected 6 x 1024 reference draws, read " + String(n_checked))
    # The two float constants, by hex.
    var inv_bits = bitcast[DType.uint32](CURAND_2POW32_INV)
    if inv_bits != UInt32(0x2F800000):
        raise Error("CURAND_2POW32_INV is " + _hex32u(inv_bits) + ", not 2^-32 (0x2f800000)")
    var gamma_bits = bitcast[DType.uint32](EULER_MASCHERONI_F32)
    if gamma_bits != UInt32(0x3F13C468):
        raise Error("EULER_MASCHERONI_F32 is " + _hex32u(gamma_bits) + ", not 0x3f13c468")
    # Reach of the subsequence key: tree 1's first draw is not tree 0's.
    var s0 = curandStateXORWOW.zero()
    var s1 = curandStateXORWOW.zero()
    curand_init(42, 0, 0, s0, seqp, offp)
    curand_init(42, 1, 0, s1, seqp, offp)
    if curand(s0) == curand(s1):
        raise Error("tree 0 and tree 1 share a first draw: the subsequence skip is inert")
    print(
        "check_if_xorwow_matches_curand OK" + _tag() + ": 6 (seed, tree) pairs x 1024 draws and uniform bits equal"
        + " the curand_kernel.h reference; 2^-32f = 0x2f800000, gamma = 0x3f13c468; tree 0 != tree 1"
    )
    _ = tables^


# ---------------------------------------------------------------------------
# 1b. THE SAME REFERENCE, ON THE DEVICE
#
# `check_if_xorwow_matches_curand` above runs the port on the HOST. That is
# not the stream the forest is built from: `curand_init` walks
# `_curand_matvec_inplace` 160 bit-tests deep against a table in GLOBAL
# memory, inside a kernel, and a GPU compiler that reassociates or
# mis-widens one of those loads forks every downstream bit. The whole-forest
# gate would see it, as a structure mismatch with no localization. This gate
# sees it as the draw index it happened at.
#
# One thread per (seed, tree, offset) triple, so it also proves the stream is
# a pure function of the triple and not of the thread's position.
# ---------------------------------------------------------------------------


def xorwow_device_kernel(
    seeds: MutPointer[UInt64, MutAnyOrigin],
    trees: MutPointer[UInt64, MutAnyOrigin],
    offsets: MutPointer[UInt64, MutAnyOrigin],
    n_triples: Int32,
    n_draws: Int32,
    sequence_table: MutPointer[UInt32, MutAnyOrigin],
    offset_table: MutPointer[UInt32, MutAnyOrigin],
    out_u32: MutPointer[UInt32, MutAnyOrigin],
    out_uni: MutPointer[UInt32, MutAnyOrigin],
):
    """`curand_init(seed, tree, offset)` then `n_draws` x `curand()` and its
    `_curand_uniform`, one thread per triple, exactly the calls
    `build_isolation_trees_global_kernel` makes."""
    var i = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if i >= Int(n_triples):
        return
    var st = curandStateXORWOW.zero()
    curand_init(
        seeds.unsafe_load(i),
        trees.unsafe_load(i),
        offsets.unsafe_load(i),
        st,
        sequence_table,
        offset_table,
    )
    var n = Int(n_draws)
    for k in range(n):
        var u = curand(st)
        out_u32.unsafe_store(i * n + k, u)
        out_uni.unsafe_store(
            i * n + k, bitcast[DType.uint32](_curand_uniform(u))
        )


struct _RefStream(Movable):
    """One (seed, tree, offset) block of a reference TSV."""

    var seed: UInt64
    var tree: UInt64
    var offset: UInt64
    var u32: List[UInt32]
    var uni: List[UInt32]

    def __init__(out self, seed: UInt64, tree: UInt64, offset: UInt64):
        self.seed = seed
        self.tree = tree
        self.offset = offset
        self.u32 = List[UInt32]()
        self.uni = List[UInt32]()


def _read_reference(path: String, has_offset: Bool) raises -> List[_RefStream]:
    """Parse `xorwow_reference.tsv` (5 columns) or
    `xorwow_offset_reference.tsv` (6, with `offset` third). The `idx` column
    must run 0..n-1 with no gap inside a block."""
    var text: String
    with open(path, "r") as fh:
        text = fh.read()
    var want = 6 if has_offset else 5
    var out = List[_RefStream]()
    var expect_idx = 0
    var lines = text.split("\n")
    for li in range(len(lines)):
        var line = String(lines[li])
        if line == "" or line.startswith("#"):
            continue
        var cols = line.split("\t")
        if len(cols) != want:
            raise Error(path + ": bad line " + line)
        var seed = _parse_u64(String(cols[0]))
        var tree = _parse_u64(String(cols[1]))
        var offset: UInt64 = 0
        var c = 2
        if has_offset:
            offset = _parse_u64(String(cols[2]))
            c = 3
        var idx = Int(String(cols[c]))
        var u = UInt32(_parse_u64(String(cols[c + 1])))
        var uni = _parse_hex32(String(cols[c + 2]))
        if (
            len(out) == 0
            or out[len(out) - 1].seed != seed
            or out[len(out) - 1].tree != tree
            or out[len(out) - 1].offset != offset
        ):
            out.append(_RefStream(seed, tree, offset))
            expect_idx = 0
        if idx != expect_idx:
            raise Error(path + ": index gap at " + line)
        expect_idx += 1
        out[len(out) - 1].u32.append(u)
        out[len(out) - 1].uni.append(uni)
    return out^


def _run_xorwow_on_device(
    ctx: DeviceContext, refs: List[_RefStream], n_draws: Int, tpb: Int
) raises -> List[UInt32]:
    """Returns `n_triples * n_draws * 2` words: all u32 then all uniform
    bits, read back from the device."""
    var n_t = len(refs)
    var h_seed = ctx.enqueue_create_host_buffer[DType.uint64](n_t)
    var h_tree = ctx.enqueue_create_host_buffer[DType.uint64](n_t)
    var h_off = ctx.enqueue_create_host_buffer[DType.uint64](n_t)
    for i in range(n_t):
        h_seed.unsafe_ptr().unsafe_store(i, refs[i].seed)
        h_tree.unsafe_ptr().unsafe_store(i, refs[i].tree)
        h_off.unsafe_ptr().unsafe_store(i, refs[i].offset)
    var d_seed = ctx.enqueue_create_buffer[DType.uint64](n_t)
    var d_tree = ctx.enqueue_create_buffer[DType.uint64](n_t)
    var d_off = ctx.enqueue_create_buffer[DType.uint64](n_t)
    ctx.enqueue_copy(dst_buf=d_seed, src_ptr=h_seed.unsafe_ptr())
    ctx.enqueue_copy(dst_buf=d_tree, src_ptr=h_tree.unsafe_ptr())
    ctx.enqueue_copy(dst_buf=d_off, src_ptr=h_off.unsafe_ptr())

    var tables = build_xorwow_tables()
    var d_seq = ctx.enqueue_create_buffer[DType.uint32](XORWOW_TABLE_WORDS)
    var d_offt = ctx.enqueue_create_buffer[DType.uint32](XORWOW_TABLE_WORDS)
    var h_tbl = ctx.enqueue_create_host_buffer[DType.uint32](XORWOW_TABLE_WORDS)
    for i in range(XORWOW_TABLE_WORDS):
        h_tbl.unsafe_ptr().unsafe_store(i, tables.sequence[i])
    ctx.enqueue_copy(dst_buf=d_seq, src_ptr=h_tbl.unsafe_ptr())
    ctx.synchronize()
    for i in range(XORWOW_TABLE_WORDS):
        h_tbl.unsafe_ptr().unsafe_store(i, tables.offset[i])
    ctx.enqueue_copy(dst_buf=d_offt, src_ptr=h_tbl.unsafe_ptr())
    ctx.synchronize()

    var n_cells = n_t * n_draws
    var d_u32 = ctx.enqueue_create_buffer[DType.uint32](n_cells)
    var d_uni = ctx.enqueue_create_buffer[DType.uint32](n_cells)
    d_u32.enqueue_fill(UInt32(0xDEADBEEF))
    d_uni.enqueue_fill(UInt32(0xDEADBEEF))
    ctx.synchronize()
    var blocks = (n_t + tpb - 1) // tpb
    ctx.enqueue_function[xorwow_device_kernel](
        d_seed.unsafe_ptr(),
        d_tree.unsafe_ptr(),
        d_off.unsafe_ptr(),
        Int32(n_t),
        Int32(n_draws),
        d_seq.unsafe_ptr(),
        d_offt.unsafe_ptr(),
        d_u32.unsafe_ptr(),
        d_uni.unsafe_ptr(),
        grid_dim=(blocks, 1, 1),
        block_dim=(tpb, 1, 1),
    )
    ctx.synchronize()
    var h_out = ctx.enqueue_create_host_buffer[DType.uint32](n_cells)
    var out = List[UInt32]()
    ctx.enqueue_copy(dst_ptr=h_out.unsafe_ptr(), src_buf=d_u32)
    ctx.synchronize()
    for i in range(n_cells):
        out.append(h_out.unsafe_ptr().unsafe_load(i))
    ctx.enqueue_copy(dst_ptr=h_out.unsafe_ptr(), src_buf=d_uni)
    ctx.synchronize()
    for i in range(n_cells):
        out.append(h_out.unsafe_ptr().unsafe_load(i))
    _ = h_seed^
    _ = h_tree^
    _ = h_off^
    _ = h_tbl^
    _ = h_out^
    _ = d_seed^
    _ = d_tree^
    _ = d_off^
    _ = d_seq^
    _ = d_offt^
    _ = d_u32^
    _ = d_uni^
    _ = tables^
    return out^


def check_if_xorwow_on_device() raises:
    var ctx = DeviceContext()
    var seq_refs = _read_reference(
        "isolation_forest/checks/xorwow_reference.tsv", False
    )
    var off_refs = _read_reference(
        "isolation_forest/checks/xorwow_offset_reference.tsv", True
    )
    if len(seq_refs) != 6 or len(seq_refs[0].u32) != 1024:
        raise Error("xorwow_reference.tsv: expected 6 x 1024")
    if len(off_refs) != 6 or len(off_refs[0].u32) != 256:
        raise Error("xorwow_offset_reference.tsv: expected 6 x 256")

    var n_checked = 0
    # Two thread-block widths: the stream must not know the block it is in.
    for arm in range(2):
        var tpb = 32 if arm == 0 else 4
        var got = _run_xorwow_on_device(ctx, seq_refs, 1024, tpb)
        for t in range(len(seq_refs)):
            for k in range(1024):
                var g = got[t * 1024 + k]
                var w = seq_refs[t].u32[k]
                if g != w:
                    raise Error(
                        "check_if_xorwow_on_device FAILED (tpb "
                        + String(tpb)
                        + "): seed "
                        + String(seq_refs[t].seed)
                        + " tree "
                        + String(seq_refs[t].tree)
                        + " draw "
                        + String(k)
                        + " device "
                        + _hex32u(g)
                        + " reference "
                        + _hex32u(w)
                    )
                var gu = got[6 * 1024 + t * 1024 + k]
                var wu = seq_refs[t].uni[k]
                if gu != wu:
                    raise Error(
                        "check_if_xorwow_on_device FAILED (tpb "
                        + String(tpb)
                        + "): uniform bits at seed "
                        + String(seq_refs[t].seed)
                        + " tree "
                        + String(seq_refs[t].tree)
                        + " draw "
                        + String(k)
                        + " device "
                        + _hex32u(gu)
                        + " reference "
                        + _hex32u(wu)
                    )
                n_checked += 2

    # DEVIATION 751: the offset arm, which cuML itself never reaches.
    var got_o = _run_xorwow_on_device(ctx, off_refs, 256, 32)
    for t in range(len(off_refs)):
        for k in range(256):
            var g = got_o[t * 256 + k]
            var w = off_refs[t].u32[k]
            if g != w:
                raise Error(
                    "check_if_xorwow_on_device FAILED (offset arm): seed "
                    + String(off_refs[t].seed)
                    + " tree "
                    + String(off_refs[t].tree)
                    + " offset "
                    + String(off_refs[t].offset)
                    + " draw "
                    + String(k)
                    + " device "
                    + _hex32u(g)
                    + " reference "
                    + _hex32u(w)
                )
            var gu = got_o[6 * 256 + t * 256 + k]
            var wu = off_refs[t].uni[k]
            if gu != wu:
                raise Error(
                    "check_if_xorwow_on_device FAILED (offset arm): uniform"
                    + " bits at offset "
                    + String(off_refs[t].offset)
                    + " draw "
                    + String(k)
                )
            n_checked += 2

    # REACH, per branch, on the device:
    #   (a) the subsequence skip runs -- tree 1 is not tree 0;
    #   (b) the offset skip runs -- offset 1 is not offset 0; and it is
    #       exactly a one-draw advance, which is the single-draw perturbation
    #       the whole lane is defended against.
    var probe = List[_RefStream]()
    probe.append(_RefStream(0, 0, 0))
    probe.append(_RefStream(0, 1, 0))
    probe.append(_RefStream(0, 0, 1))
    var pg = _run_xorwow_on_device(ctx, probe, 4, 32)
    if pg[0] == pg[4]:
        raise Error("device: tree 0 and tree 1 share a first draw")
    if pg[0] == pg[8]:
        raise Error("device: offset 0 and offset 1 share a first draw")
    for k in range(3):
        if pg[8 + k] != pg[0 + k + 1]:
            raise Error(
                "device: offset 1 is not a one-draw advance of offset 0 at k "
                + String(k)
            )
    print(
        "check_if_xorwow_on_device OK"
        + _tag()
        + ": "
        + String(n_checked)
        + " device cells equal the curand_precalc.h-derived reference"
        + " (6 x 1024 sequence draws at tpb 32 and 4, 6 x 256 offset draws);"
        + " tree 1 != tree 0, offset 1 == tree 0 advanced by exactly one draw"
    )
    _ = ctx^


# ---------------------------------------------------------------------------
# 2. Refusals
# ---------------------------------------------------------------------------


def check_if_refusals() raises:
    var ctx = DeviceContext()
    var n = 64
    var d = 4
    var x = blob_fixture(n, d, 3, 2)
    var n_refused = 0

    # DEVIATION 680: NaN / inf in X, by name and position.
    var x_nan = x.copy()
    x_nan[5 * d + 2] = bitcast[DType.float32](UInt32(0x7FC00000))
    var raised = False
    var msg = String("")
    try:
        check_finite_by_name("X", x_nan, n, d)
    except e:
        raised = True
        msg = String(e)
    _expect_raise("NaN in X", raised, msg, "X contains NaN")
    _expect_raise("NaN in X (position)", raised, msg, "row 5, column 2")
    n_refused += 1
    var x_inf = x.copy()
    x_inf[7 * d + 1] = bitcast[DType.float32](UInt32(0xFF800000))
    raised = False
    try:
        check_finite_by_name("X_query", x_inf, n, d)
    except e:
        raised = True
        msg = String(e)
    _expect_raise("inf in X_query", raised, msg, "X_query contains infinity")
    n_refused += 1
    # ... and through fit / score_samples themselves.
    var model = IsolationForestModel(ctx)
    var trace = IdentityTrace.disabled()
    var params = _params(4, 32, 1)
    raised = False
    try:
        fit(ctx, model, to_column_major(x_nan, n, d), n, d, params, trace)
    except e:
        raised = True
        msg = String(e)
    _expect_raise("fit(NaN)", raised, msg, "DEVIATION 680")
    n_refused += 1
    raised = False
    try:
        _ = score_samples(ctx, model, x, n, d, trace)
    except e:
        raised = True
        msg = String(e)
    _expect_raise("score before fit", raised, msg, "not been fitted")
    n_refused += 1
    fit(ctx, model, to_column_major(x, n, d), n, d, params, trace)
    raised = False
    try:
        _ = score_samples(ctx, model, x_inf, n, d, trace)
    except e:
        raised = True
        msg = String(e)
    _expect_raise("score(inf)", raised, msg, "X_query contains infinity")
    n_refused += 1
    raised = False
    try:
        _ = score_samples(ctx, model, x, n // 2, d + 1, trace)
    except e:
        raised = True
        msg = String(e)
    _expect_raise("feature mismatch", raised, msg, "features")
    n_refused += 1
    # Their asserts.
    var p0 = _params(0, 32, 1)
    raised = False
    try:
        fit(ctx, model, to_column_major(x, n, d), n, d, p0, trace)
    except e:
        raised = True
        msg = String(e)
    _expect_raise("n_estimators=0", raised, msg, "n_estimators")
    n_refused += 1
    var pm = _params(4, 0, 1)
    raised = False
    try:
        fit(ctx, model, to_column_major(x, n, d), n, d, pm, trace)
    except e:
        raised = True
        msg = String(e)
    _expect_raise("max_samples=0", raised, msg, "max_samples")
    n_refused += 1
    raised = False
    try:
        fit(ctx, model, to_column_major(x, n, d), 0, d, params, trace)
    except e:
        raised = True
        msg = String(e)
    _expect_raise("n_rows=0", raised, msg, "n_rows")
    n_refused += 1
    # Estimator-level refusals (the Python layer's, estimator.mojo).
    var est = IsolationForestEstimator(ctx)
    est.max_samples_frac = 0.0
    est.max_samples_mode = 2
    raised = False
    try:
        est.fit(ctx, x, n, d)
    except e:
        raised = True
        msg = String(e)
    _expect_raise("float max_samples=0.0", raised, msg, "max_samples")
    n_refused += 1
    est = IsolationForestEstimator(ctx)
    est.contamination = 0.7
    est.contamination_auto = False
    raised = False
    try:
        est.fit(ctx, x, n, d)
    except e:
        raised = True
        msg = String(e)
    _expect_raise("contamination=0.7", raised, msg, "contamination")
    n_refused += 1
    est = IsolationForestEstimator(ctx)
    est.max_features_int = 9
    est.max_features_mode = 1
    raised = False
    try:
        est.fit(ctx, x, n, d)
    except e:
        raised = True
        msg = String(e)
    _expect_raise("max_features=9 of 4", raised, msg, "max_features")
    n_refused += 1
    est = IsolationForestEstimator(ctx)
    est.warm_start = True
    raised = False
    try:
        est.fit(ctx, x, n, d)
    except e:
        raised = True
        msg = String(e)
    _expect_raise("warm_start", raised, msg, "warm_start")
    n_refused += 1

    # DEVIATION 684: ceil_log2_int against float64 ceil(log2(n)).
    for nn in range(1, 4098):
        var want = Int(ceil(log2(Float64(nn))))
        if ceil_log2_int(nn) != want:
            raise Error("ceil_log2_int(" + String(nn) + ") = " + String(ceil_log2_int(nn)) + ", float64 says " + String(want))
    # compute_global_max_nodes_per_tree: both bounds.
    if compute_global_max_nodes_per_tree(8, 256) != 511:
        raise Error("max_nodes(8, 256) != 511")
    if compute_global_max_nodes_per_tree(3, 256) != 15:
        raise Error("max_nodes(3, 256) != 15")
    if compute_global_max_nodes_per_tree(40, 10) != 19:
        raise Error("max_nodes(40, 10) != 19")
    print(
        "check_if_refusals OK" + _tag() + ": " + String(n_refused)
        + " refusals by name (NaN/inf in X and X_query with row/column, unfitted score, feature mismatch,"
        + " n_estimators/max_samples/n_rows, float max_samples, contamination, max_features, warm_start);"
        + " ceil_log2_int == ceil(log2(n)) for n = 1..4097; node capacity bounds 511/15/19"
    )


# ---------------------------------------------------------------------------
# 3. Oracle semantics
# ---------------------------------------------------------------------------


def _median(values: List[Float32]) -> Float32:
    var s = values.copy()
    # insertion sort is fine at these sizes
    for i in range(1, len(s)):
        var v = s[i]
        var j = i - 1
        while j >= 0 and s[j] > v:
            s[j + 1] = s[j]
            j -= 1
        s[j + 1] = v
    return s[len(s) // 2]


def check_if_oracle_semantics() raises:
    var tables = build_xorwow_tables()
    var n = 1024
    var d = 8
    var n_out = 16
    var x = _fixture_main(n, d, 11, n_out)
    var params = _params(32, 256, 42)
    var f = oracle_fit(x, n, d, params, tables)
    var pl = oracle_path_lengths(f, x, n, d)
    var sc = oracle_scores(f, pl)
    var ref64 = oracle_scores_f64(f, x, n, d)
    # (a) planted outliers above the blob median, and the shortest paths
    var blob_scores = List[Float32]()
    var out_min = Float32(2.0)
    var out_max_pl = Float32(0.0)
    var n_out_seen = 0
    for i in range(n):
        if is_planted_outlier(i, n, n_out):
            n_out_seen += 1
            if sc[i] < out_min:
                out_min = sc[i]
            if pl[i] > out_max_pl:
                out_max_pl = pl[i]
        else:
            blob_scores.append(sc[i])
    if n_out_seen != n_out:
        raise Error("fixture: expected " + String(n_out) + " planted outliers, saw " + String(n_out_seen))
    var med = _median(blob_scores)
    if out_min <= med:
        raise Error(
            "check_if_oracle_semantics FAILED: lowest outlier score " + String(out_min)
            + " is not above the blob median " + String(med)
        )
    var n_blob_shorter = 0
    for i in range(n):
        if not is_planted_outlier(i, n, n_out) and pl[i] <= out_max_pl:
            n_blob_shorter += 1
    # (b) duplicates score bit-identically (rows 40, 137, 234, ...)
    for c in range(1, 7):
        var r = 40 + c * 97
        if not _bits_equal(sc[40], sc[r]) or not _bits_equal(pl[40], pl[r]):
            raise Error("duplicate row " + String(r) + " scores " + _hex32(sc[r]) + " vs row 40 " + _hex32(sc[40]))
    # (c) float32 vs float64 reference
    var worst = 0.0
    for i in range(n):
        var diff = Float64(sc[i]) - ref64[i]
        if diff < 0.0:
            diff = -diff
        if diff > worst:
            worst = diff
    if worst > 2e-4:
        raise Error("check_if_oracle_semantics FAILED: float32 scores drift " + String(worst) + " from the float64 reference")
    # (d) n_rows < max_samples
    var xs = blob_fixture(100, 6, 5, 4)
    var ps = _params(8, 256, 3)
    var fs = oracle_fit(xs, 100, 6, ps, tables)
    if fs.n_samples_per_tree != 100:
        raise Error("n_rows=100 < max_samples=256: n_samples_per_tree is " + String(fs.n_samples_per_tree))
    # without bootstrap, the 100 sampled rows are a permutation of 0..99
    var seen = List[Int]()
    for _ in range(100):
        seen.append(0)
    for i in range(100):
        seen[Int(fs.trees[0].rows[i])] += 1
    for i in range(100):
        if seen[i] != 1:
            raise Error("Floyd sampler over all rows is not a permutation: row " + String(i) + " seen " + String(seen[i]))
    print(
        "check_if_oracle_semantics OK" + _tag() + ": 16 planted outliers all score above the blob median ("
        + String(out_min) + " > " + String(med) + "), " + String(n_blob_shorter)
        + " of 1008 blob rows have a path as short as the longest outlier path; 6 duplicates bit-identical;"
        + " float32 vs float64 worst |diff| " + String(worst) + "; n=100<256 samples every row once"
    )
    _ = tables^


# ---------------------------------------------------------------------------
# 4. Device == oracle
# ---------------------------------------------------------------------------


def check_if_device_equals_oracle() raises:
    var ctx = DeviceContext()
    var tables = build_xorwow_tables()
    var fxs = _fixtures()
    var n_query = 64
    var n_ok = 0
    var n_cells = 0
    var reports = String("")
    var first_dump_scores = List[List[Float32]]()
    for k in range(len(fxs)):
        var fx = Pointer(to=fxs[k])
        var q = _query_rows(fx[], n_query)
        var dump = _device_fit(ctx, fx[], q, n_query, IFLaunchKnobs.default())
        var orc = oracle_fit(fx[].x, fx[].n, fx[].d, fx[].params, tables)
        var o_pl = oracle_path_lengths(orc, q, n_query, fx[].d)
        var o_sc = oracle_scores(orc, o_pl)
        var msg = _compare_fit_to_oracle(fx[], dump, orc, o_pl, o_sc, n_query, fx[].name)
        for t in range(fx[].params.n_estimators):
            n_cells += 4 * Int(dump.n_nodes[t])
        n_cells += 2 * n_query
        if msg == "":
            n_ok += 1
        else:
            comptime if IDENTICAL:
                raise Error("check_if_device_equals_oracle FAILED " + msg)
            else:
                reports += " | " + msg
        first_dump_scores.append(dump.scores.copy())
    # REACH: the six forests differ pairwise (scores of query 0..63)
    for a in range(len(fxs)):
        for b in range(a + 1, len(fxs)):
            var same = True
            var m = len(first_dump_scores[a])
            if len(first_dump_scores[b]) < m:
                m = len(first_dump_scores[b])
            for i in range(m):
                if not _bits_equal(first_dump_scores[a][i], first_dump_scores[b][i]):
                    same = False
            if same:
                raise Error("fixtures " + String(a) + " and " + String(b) + " produced identical scores: a parameter is inert")
    comptime if IDENTICAL:
        print(
            "check_if_device_equals_oracle OK" + _tag() + ": " + String(len(fxs)) + " fixtures, "
            + String(n_cells) + " structure/pathlen/score cells bit-equal to the oracle; six forests pairwise distinct"
        )
    else:
        if reports == "":
            print(
                "check_if_device_equals_oracle REPORT" + _tag() + ": " + String(len(fxs)) + " fixtures, "
                + String(n_cells) + " cells bit-equal to the oracle on this device (FAST: not asserted)"
            )
        else:
            print("check_if_device_equals_oracle REPORT" + _tag() + ": " + String(n_ok) + " of " + String(len(fxs)) + " fixtures bit-equal; FAST differences:" + reports)
    _ = tables^

    # DEVIATION 1946: the context dies LAST, after every value built on it.
    # Mojo frees at LAST USE, so without this the buffer releases above run
    # against a context that is already gone. On sm_89 the next GPU call in
    # the process then never returns (GPU idle, host threads in futex wait);
    # Apple and AMD do not show it, which is how it stayed latent here.
    _ = ctx^


# ---------------------------------------------------------------------------
# 5. Launch invariance
# ---------------------------------------------------------------------------


def check_if_launch_invariance() raises:
    var ctx = DeviceContext()
    var fxs = _fixtures()
    var n_query = 64
    var n_arms = 0
    for k in range(3):  # main, bootstrap, max_features
        var fx = Pointer(to=fxs[k])
        var q = _query_rows(fx[], n_query)
        var base = _device_fit(ctx, fx[], q, n_query, IFLaunchKnobs(128, 256, 0, Float32(0.0)))
        var arms = List[IFLaunchKnobs]()
        arms.append(IFLaunchKnobs(32, 64, 37, bitcast[DType.float32](UInt32(0xFFC0DEAD))))
        arms.append(IFLaunchKnobs(256, 256, 37, Float32(-123456.0)))
        arms.append(IFLaunchKnobs(64, 128, 0, bitcast[DType.float32](UInt32(0x7FC00000))))
        arms.append(IFLaunchKnobs(128, 256, 0, Float32(0.0)))  # run twice
        for a in range(len(arms)):
            var other = _device_fit(ctx, fx[], q, n_query, arms[a])
            var msg = _compare_dumps(base, other, fx[].params.n_estimators, n_query, fx[].name + " arm " + String(a))
            if msg != "":
                raise Error("check_if_launch_invariance FAILED " + msg)
            n_arms += 1
    # Batch composition: query i alone, inside 37, at three positions of 4096.
    var fx0 = Pointer(to=fxs[0])
    var model = IsolationForestModel(ctx)
    var trace = IdentityTrace.disabled()
    fit(ctx, model, to_column_major(fx0[].x, fx0[].n, fx0[].d), fx0[].n, fx0[].d, fx0[].params, trace)
    var d = fx0[].d
    var probes = List[Int]()
    probes.append(0)
    probes.append(13)  # a planted outlier
    probes.append(40)
    var big = blob_fixture(4096, d, 5151, 0)
    var positions = List[Int]()
    positions.append(0)
    positions.append(2047)
    positions.append(4095)
    var n_batch_cells = 0
    for pi in range(len(probes)):
        var r = probes[pi]
        var row = List[Float32]()
        for c in range(d):
            row.append(fx0[].x[r * d + c])
        var alone = score_samples(ctx, model, row, 1, d, trace)
        var in37 = List[Float32]()
        for i in range(37):
            for c in range(d):
                in37.append(fx0[].x[(i + 100) * d + c])
        var slot = (r * 7) % 37
        for c in range(d):
            in37[slot * d + c] = row[c]
        var s37 = score_samples(ctx, model, in37, 37, d, trace)
        if not _bits_equal(alone[0], s37[slot]):
            raise Error("check_if_launch_invariance FAILED batch: row " + String(r) + " alone " + _hex32(alone[0]) + " vs in 37 " + _hex32(s37[slot]))
        for pj in range(len(positions)):
            var pos = positions[pj]
            var b = big.copy()
            for c in range(d):
                b[pos * d + c] = row[c]
            var sb = score_samples(ctx, model, b, 4096, d, trace, IFLaunchKnobs(128, 64 if pj == 1 else 256, 0, Float32(0.0)))
            if not _bits_equal(alone[0], sb[pos]):
                raise Error("check_if_launch_invariance FAILED batch: row " + String(r) + " alone " + _hex32(alone[0]) + " vs at " + String(pos) + " of 4096 " + _hex32(sb[pos]))
            n_batch_cells += 1
    print(
        "check_if_launch_invariance OK" + _tag() + ": 3 fixtures x 4 arms (build_tpb 32/64/128/256, path_tpb 64/128/256,"
        + " pad 0/37, poisons 0xffc0dead/-123456/NaN, run twice) byte-identical in every structure/pathlen/score cell;"
        + " 3 probe rows scored alone == in a batch of 37 == at 3 positions of 4096 (" + String(n_batch_cells) + " cells)"
    )


# ---------------------------------------------------------------------------
# 6. Signed zero (ADDENDUM 11)
# ---------------------------------------------------------------------------


def check_if_row39_signed_zero() raises:
    var ctx = DeviceContext()
    var tables = build_xorwow_tables()
    var n = 512
    var d = 4
    var n_query = 64
    # Column 3: zeros among positives; then the SAME fixture with the -0.0
    # and +0.0 positions swapped.
    var xa = blob_fixture(n, d, 61, 0)
    plant_signed_zero_column(xa, n, d, 3)
    var xb = xa.copy()
    for i in range(n):
        if i % 4 == 1:
            xb[i * d + 3] = Float32(0.0)
        elif i % 4 == 3:
            xb[i * d + 3] = bitcast[DType.float32](UInt32(0x80000000))
    var n_neg = 0
    var n_pos = 0
    for i in range(n):
        var u = bitcast[DType.uint32](xa[i * d + 3])
        if u == 0x80000000:
            n_neg += 1
        if u == 0:
            n_pos += 1
    var params = _params(16, 256, 4242)
    var fa = Fixture("zeros A", xa^, n, d, params)
    var fb = Fixture("zeros B (swapped)", xb^, n, d, params)
    var qa = _query_rows(fa, n_query)
    var qb = _query_rows(fb, n_query)
    var da = _device_fit(ctx, fa, qa, n_query, IFLaunchKnobs.default())
    var db = _device_fit(ctx, fb, qb, n_query, IFLaunchKnobs.default())
    var oa = oracle_fit(fa.x, n, d, params, tables)
    var ob = oracle_fit(fb.x, n, d, params, tables)
    var opa = oracle_path_lengths(oa, qa, n_query, d)
    var opb = oracle_path_lengths(ob, qb, n_query, d)
    var m1 = _compare_fit_to_oracle(fa, da, oa, opa, oracle_scores(oa, opa), n_query, "zeros A")
    var m2 = _compare_fit_to_oracle(fb, db, ob, opb, oracle_scores(ob, opb), n_query, "zeros B")
    # How many nodes split on the zero column, and how many of their
    # thresholds are themselves a zero (the only place the sign could show).
    var n_zero_col_splits = 0
    var n_zero_thr = 0
    for t in range(params.n_estimators):
        var off = t * da.max_nodes_per_tree
        for i in range(Int(da.n_nodes[t])):
            if da.feat[off + i] == 3:
                n_zero_col_splits += 1
                if (bitcast[DType.uint32](da.thr[off + i]) & 0x7FFFFFFF) == 0:
                    n_zero_thr += 1
    var m3 = _compare_dumps(da, db, params.n_estimators, n_query, "A vs B")
    # A {-0.0, +0.0}-only column is refused as constant by `min < max`.
    var xc = blob_fixture(64, 3, 9, 0)
    for i in range(64):
        xc[i * 3 + 1] = bitcast[DType.float32](UInt32(0x80000000)) if i % 2 == 0 else Float32(0.0)
    var pc = _params(8, 64, 9)
    var fc = Fixture("zeros-only column", xc^, 64, 3, pc)
    var qc = _query_rows(fc, 16)
    var dc = _device_fit(ctx, fc, qc, 16, IFLaunchKnobs.default())
    var oc = oracle_fit(fc.x, 64, 3, pc, tables)
    var opc = oracle_path_lengths(oc, qc, 16, 3)
    var m4 = _compare_fit_to_oracle(fc, dc, oc, opc, oracle_scores(oc, opc), 16, "zeros-only")
    var n_col1 = 0
    for t in range(pc.n_estimators):
        var off = t * dc.max_nodes_per_tree
        for i in range(Int(dc.n_nodes[t])):
            if dc.feat[off + i] == 1:
                n_col1 += 1
    if n_col1 != 0:
        raise Error("check_if_row39_signed_zero FAILED: the {-0,+0} column was split " + String(n_col1) + " times; `min < max` must refuse it")
    comptime if IDENTICAL:
        if m1 != "" or m2 != "" or m4 != "":
            raise Error("check_if_row39_signed_zero FAILED device vs oracle: " + m1 + m2 + m4)
    else:
        if m1 != "" or m2 != "" or m4 != "":
            print("  RECORDED [FAST] zeros fixtures device vs oracle: " + m1 + m2 + m4)
    if m3 != "":
        raise Error("check_if_row39_signed_zero FAILED: swapping the -0.0/+0.0 positions moved a stored bit: " + m3)
    print(
        "check_if_row39_signed_zero OK" + _tag() + ": " + String(n_neg) + " x -0.0 and " + String(n_pos)
        + " x +0.0 planted in column 3; " + String(n_zero_col_splits) + " nodes split on it (" + String(n_zero_thr)
        + " with a zero threshold); device == oracle bit for bit on both zero orders; swapping the zeros' positions"
        + " moves NO stored bit (the positional strict-compare fold is sign-inert here); a {-0,+0}-only column is never split"
    )
    _ = tables^

    # DEVIATION 1946: the context dies LAST, after every value built on it.
    # Mojo frees at LAST USE, so without this the buffer releases above run
    # against a context that is already gone. On sm_89 the next GPU call in
    # the process then never returns (GPU idle, host threads in futex wait);
    # Apple and AMD do not show it, which is how it stayed latent here.
    _ = ctx^


# ---------------------------------------------------------------------------
# 7. predict / contamination
# ---------------------------------------------------------------------------


def check_if_predict_thresholds() raises:
    var ctx = DeviceContext()
    var n = 1024
    var d = 8
    var x = _fixture_main(n, d, 11, 16)
    var params = _params(16, 256, 42)
    var model = IsolationForestModel(ctx)
    var trace = IdentityTrace.disabled()
    fit(ctx, model, to_column_major(x, n, d), n, d, params, trace)
    var sc = score_samples(ctx, model, x, n, d, trace)
    var thr = Float32(0.5)
    var labels = predict(ctx, model, x, n, d, thr)
    var n_anom = 0
    for i in range(n):
        var want = Int32(1) if sc[i] > thr else Int32(-1)
        if labels[i] != want:
            raise Error("predict label " + String(i) + " is " + String(labels[i]) + ", score " + String(sc[i]))
        if labels[i] == 1:
            n_anom += 1
    # contamination: the estimator's percentile offset flags exactly the
    # top share (ties aside).
    var est = IsolationForestEstimator(ctx)
    est.n_estimators = 16
    est.random_state = 42
    est.contamination = 0.05
    est.contamination_auto = False
    est.fit(ctx, x, n, d)
    var pred = est.predict(ctx, x, n, d)
    var n_flag = 0
    for i in range(n):
        if pred[i] == -1:
            n_flag += 1
    var expect = Int(0.05 * Float64(n))
    if n_flag < expect - 2 or n_flag > expect + 2:
        raise Error("contamination=0.05 flagged " + String(n_flag) + " of " + String(n) + ", expected about " + String(expect))
    # every planted outlier is flagged
    var n_out_flag = 0
    for i in range(n):
        if is_planted_outlier(i, n, 16) and pred[i] == -1:
            n_out_flag += 1
    if n_out_flag != 16:
        raise Error("contamination=0.05 flagged only " + String(n_out_flag) + " of 16 planted outliers")
    # percentile_linear against a hand case
    var v: List[Float32] = [1.0, 2.0, 3.0, 4.0]
    if percentile_linear(v, 50.0) != 2.5 or percentile_linear(v, 0.0) != 1.0 or percentile_linear(v, 100.0) != 4.0 or percentile_linear(v, 25.0) != 1.75:
        raise Error("percentile_linear hand case failed")
    print(
        "check_if_predict_thresholds OK" + _tag() + ": " + String(n) + " labels == (score > 0.5 ? 1 : -1), "
        + String(n_anom) + " anomalies at 0.5; contamination=0.05 flags " + String(n_flag) + " (expected ~" + String(expect)
        + ") including all 16 planted outliers; percentile_linear hand case"
    )


# ---------------------------------------------------------------------------
# 8. The card
# ---------------------------------------------------------------------------


def check_if_card_is_emitted() raises:
    var ctx = DeviceContext()
    var fxs = _fixtures()
    var fx = Pointer(to=fxs[2])  # max_features, so the .features stage exists
    var q = _query_rows(fx[], 32)
    _ensure_scratch()
    # THE PRIMARY CARD GOES WHERE THE HARNESS ASKED. `p2` stays in scratch
    # because it is the CONTROL: a second fit at a different launch geometry
    # whose only job is to be compared against `p1` below. Emitting both to
    # the harness path would overwrite the card with the control.
    var p1 = card_path()
    var p2 = String(SCRATCH) + "/if_check_card_b.trace"
    _ = _device_fit(ctx, fx[], q, 32, IFLaunchKnobs.default(), p1)
    _ = _device_fit(ctx, fx[], q, 32, IFLaunchKnobs(32, 64, 37, Float32(-1.0)), p2)
    var div = first_divergence(p1, p2)
    if div != "":
        raise Error("check_if_card_is_emitted: two runs of one fixture diverge: " + div)
    var lines = read_trace_lines(p1)
    var n_trees = fx[].params.n_estimators
    var expect = List[String]()
    expect.append("if.rng.probe")
    for t in range(n_trees):
        var s = String(t)
        while s.byte_length() < 3:
            s = "0" + s
        expect.append("if.tree" + s + ".rows")
        expect.append("if.tree" + s + ".features")
        expect.append("if.tree" + s + ".meta")
        expect.append("if.tree" + s + ".structure.feat")
        expect.append("if.tree" + s + ".structure.thr")
        expect.append("if.tree" + s + ".structure.left")
        expect.append("if.tree" + s + ".structure.right")
        expect.append("if.tree" + s + ".split.bounds")
        expect.append("if.tree" + s + ".split.choice")
        expect.append("if.tree" + s + ".rng.final")
    expect.append("if.pathlen")
    expect.append("if.scores")
    if len(lines) != len(expect):
        raise Error("check_if_card_is_emitted: " + String(len(lines)) + " records, expected " + String(len(expect)))
    for i in range(len(expect)):
        if lines[i].find("\t" + expect[i] + "\t") < 0:
            raise Error("check_if_card_is_emitted: record " + String(i) + " is '" + lines[i] + "', expected tag " + expect[i])
    print(
        "check_if_card_is_emitted OK" + _tag() + ": " + String(len(expect)) + " stages (if.rng.probe, "
        + String(n_trees) + " x {rows, features, meta, structure.feat/thr/left/right,"
        + " split.bounds, split.choice, rng.final}, if.pathlen, if.scores),"
        + " run-to-run control identical across build_tpb 128/32, pad 0/37, two poisons"
    )


def main() raises:
    print("== isolation_forest/checks/if_check.mojo [" + _mode_name() + "] ==")
    check_if_xorwow_matches_curand()
    check_if_xorwow_on_device()
    check_if_refusals()
    check_if_oracle_semantics()
    check_if_device_equals_oracle()
    check_if_launch_invariance()
    check_if_row39_signed_zero()
    check_if_predict_thresholds()
    check_if_card_is_emitted()
