# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The typed host bodies behind `bindings/_mojolearn_gbdt_host.mojo`'s
Python doors (lane thin-trees-metrics, 2026-10-21): moved word for word out
of that shell, same operations in the same order, so a Python-free caller
reaches the same source. No Python here.

The model text reader (`_HostModelArrays`, `_parse_model` and its token
helpers), the RAW / SOFTMAX / SIGMOID predict walks, the sigmoid entries,
the binary prediction transforms, and the scans and reads `gbdt_fit`
makes of its tails (NaN in X, a constant eval target, the group sizes, the
PairLogit pairs, the categorical flags, the binary class row weights).
"""
from std.math import isfinite
from std.memory import bitcast

from core.hostptr import read_f32
from checks.numerics import ftz, identical_sigmoid
from checks.soft_f64 import (
    SF64_ONE,
    SF64_ZERO,
    sf64_add,
    sf64_div,
    sf64_exp,
    sf64_from_f32,
    sf64_ftz,
    sf64_gt,
    sf64_neg,
    sf64_sigmoid_f32,
    sf64_sigmoid_f64,
    sf64_sub,
    sf64_to_f32,
)
from core.gbdt_host_predict import GBDT_HOST_SABOTAGE, gbdt_host_predict
from gbdt.data.quantization import (
    NAN_TREATMENT_AS_FALSE,
    NAN_TREATMENT_AS_IS,
    NAN_TREATMENT_AS_TRUE,
)


# ===========================================================================
# THE MODEL TEXT, READ (the records `gbdt_host_model_text` and the GPU
# binding's `model_text` write; `gbdt/models/model_text.mojo:706` is the
# reader this follows, for the oblivious float-only shape)
# ===========================================================================


@fieldwise_init
struct _HostModelArrays(Movable):
    var fold_counts: List[Int32]
    var one_hot: List[Int32]
    var nan_treatment: List[Int32]
    var border_offsets: List[Int32]
    var borders: List[Float32]
    var tree_offsets: List[Int32]
    var split_feature: List[Int32]
    var split_bin: List[Int32]
    var split_take_bin: List[Int32]
    var leaf_offsets: List[Int32]
    var leaves: List[Float32]
    var dim: Int
    var bias: Float64
    # the NON-SYMMETRIC shape (`ntree` / `node`, `model_text.mojo:535-600`):
    # `split_*` then hold one record per pre-order node, these the node's
    # LEAF-COUNT subtrees, and `tree_offsets` the prefix scan of node counts
    var node_left: List[Int32]
    var node_right: List[Int32]
    var non_symmetric: Bool


def _fields(line: String) raises -> List[String]:
    """`_fields` (`model_text.mojo:686-695`)."""
    var out = List[String]()
    for piece in line.split(" "):
        var s = String(piece)
        if s.byte_length() > 0:
            out.append(s)
    return out^


def _hex_value(code: Int) raises -> Int:
    if code >= 48 and code <= 57:
        return code - 48
    if code >= 97 and code <= 102:
        return code - 87
    if code >= 65 and code <= 70:
        return code - 55
    raise Error("not a hex digit: byte value " + String(code))


def _token_bits(tok: String, digits: Int) raises -> UInt64:
    """The hex half of `<decimal>/<hex bits>`, the half both readers load."""
    var parts = tok.split("/")
    if len(parts) != 2:
        raise Error("malformed float token '" + tok + "'")
    var s = String(parts[1])
    if s.byte_length() != digits:
        raise Error(
            "expected " + String(digits) + " hex digits, got '" + s + "'"
        )
    var v = UInt64(0)
    for i in range(digits):
        v = (v << UInt64(4)) | UInt64(_hex_value(ord(String(s[byte=i]))))
    return v


def _need(t: List[String], count: Int, kind: String) raises:
    if len(t) < count:
        raise Error("a `" + kind + "` record is short")


def _parse_model(text: String) raises -> _HostModelArrays:
    var n_features = -1
    var n_trees = -1
    var n_losses = -1
    var header_seen = 0
    var features_seen = 0
    var trees_seen = 0
    var losses_seen = 0
    var arrays = _HostModelArrays(
        List[Int32](), List[Int32](), List[Int32](), List[Int32](),
        List[Float32](), List[Int32](), List[Int32](), List[Int32](),
        List[Int32](), List[Int32](), List[Float32](), -1, Float64(0.0),
        List[Int32](), List[Int32](), False,
    )
    arrays.border_offsets.append(Int32(0))
    arrays.tree_offsets.append(Int32(0))
    arrays.leaf_offsets.append(Int32(0))
    var cur_splits = 0
    var cur_leaves = 0
    var cur_weights = 0
    var cur_dim = 1
    # per tree, what completes it: `depth` splits and `(1 << depth) * dim`
    # leaves (`tree`), or `nodes` nodes, `(nodes + 1) * dim` leaves and, with
    # `weights 1`, `nodes + 1` weights (`ntree`)
    var want_splits = 0
    var want_leaves = 0
    var want_weights = 0
    var shape_kind = 0  # 1 oblivious, 2 non-symmetric; one file, one shape
    for raw in text.split("\n"):
        var line = String(raw)
        if line.byte_length() == 0 or line.startswith("#"):
            continue
        var t = _fields(line)
        if len(t) == 0:
            continue
        var kind = t[0]
        if kind == String("format"):
            _need(t, 3, "format")
            if header_seen != 0 or t[1] != String("mojolearn-model"):
                raise Error("not a mojolearn-model file")
            if Int(t[2]) != 2:
                raise Error("format version " + t[2] + ", this reader is version 2")
            header_seen = 1
        elif kind == String("features"):
            _need(t, 3, "features")
            if header_seen != 1:
                raise Error("`features` must follow `format`")
            n_features = Int(t[1])
            header_seen = 2
        elif kind == String("trees"):
            _need(t, 2, "trees")
            if header_seen != 2:
                raise Error("`trees` must follow `features`")
            n_trees = Int(t[1])
            header_seen = 3
        elif kind == String("losses"):
            _need(t, 2, "losses")
            if header_seen != 3:
                raise Error("`losses` must follow `trees`")
            n_losses = Int(t[1])
            header_seen = 4
        elif kind == String("bias"):
            _need(t, 2, "bias")
            arrays.bias = bitcast[DType.float64](_token_bits(t[1], 16))
        elif kind == String("feature"):
            _need(t, 12, "feature")
            if header_seen != 4 or Int(t[1]) != features_seen:
                raise Error("a `feature` record out of order")
            var folds = Int(t[3])
            var flag = Int(t[5])
            if t[6] != String("type") or (t[7] != String("float") and t[7] != String("cat")):
                _refuse_predict("a feature of type " + t[7])
            var treat: Int32
            if t[9] == String("as_is"):
                treat = Int32(NAN_TREATMENT_AS_IS)
            elif t[9] == String("as_false"):
                treat = Int32(NAN_TREATMENT_AS_FALSE)
            elif t[9] == String("as_true"):
                treat = Int32(NAN_TREATMENT_AS_TRUE)
            else:
                raise Error("unknown nan treatment '" + t[9] + "'")
            var nb = Int(t[11])
            if len(t) != 12 + nb:
                raise Error("feature " + t[1] + " declares " + t[11] + " borders")
            for b in range(nb):
                arrays.borders.append(
                    bitcast[DType.float32](UInt32(_token_bits(t[12 + b], 8)))
                )
            arrays.fold_counts.append(Int32(folds))
            arrays.one_hot.append(Int32(1) if flag != 0 else Int32(0))
            arrays.nan_treatment.append(treat)
            arrays.border_offsets.append(Int32(len(arrays.borders)))
            features_seen += 1
        elif kind == String("tree") or kind == String("ntree"):
            _need(t, 8, kind)
            if features_seen != n_features or Int(t[1]) != trees_seen:
                raise Error("a `" + kind + "` record out of order")
            if trees_seen > 0 and (
                cur_splits != want_splits or cur_leaves != want_leaves
                or cur_weights != want_weights
            ):
                raise Error("tree " + String(trees_seen - 1) + " is incomplete")
            var this_kind = 1 if kind == String("tree") else 2
            if shape_kind != 0 and shape_kind != this_kind:
                raise Error("a model text mixing oblivious and non-symmetric trees")
            shape_kind = this_kind
            arrays.non_symmetric = this_kind == 2
            var size = Int(t[3])
            cur_dim = Int(t[5])
            var n_values: Int
            if this_kind == 1:
                if Int(t[7]) != 0:
                    _refuse_predict("a tree carrying leaf weights")
                if size < 0 or size > 31 or cur_dim < 1:
                    raise Error("tree depth or dim is not sane")
                n_values = (1 << size) * cur_dim
                want_weights = 0
            else:
                if t[2] != String("nodes") or t[4] != String("dim") or t[6] != String("weights"):
                    raise Error("an `ntree` record has 8 fields")
                if size < 0 or cur_dim < 1:
                    raise Error("non-symmetric node count or dim is not sane")
                n_values = (size + 1) * cur_dim
                want_weights = (size + 1) if Int(t[7]) != 0 else 0
            if arrays.dim == -1:
                arrays.dim = cur_dim
            elif arrays.dim != cur_dim:
                raise Error("tree " + t[1] + " has a different dim")
            want_splits = size
            want_leaves = n_values
            cur_splits = 0
            cur_leaves = 0
            cur_weights = 0
            trees_seen += 1
            arrays.tree_offsets.append(
                arrays.tree_offsets[len(arrays.tree_offsets) - 1] + Int32(size)
            )
            arrays.leaf_offsets.append(
                arrays.leaf_offsets[len(arrays.leaf_offsets) - 1] + Int32(n_values)
            )
        elif kind == String("split") or kind == String("node"):
            var base = 5 if kind == String("split") else 7
            _need(t, base, kind)
            if (shape_kind == 1) != (kind == String("split")) or Int(t[1]) != trees_seen - 1 or Int(t[2]) != cur_splits:
                raise Error("a `" + kind + "` record out of order")
            arrays.split_feature.append(Int32(Int(t[3])))
            arrays.split_bin.append(Int32(Int(t[4])))
            var take_bin = len(t) >= base + 2 and t[base] == String("split_type") and t[base + 1] == String("take_bin")
            arrays.split_take_bin.append(Int32(1) if take_bin else Int32(0))
            if kind == String("node"):
                arrays.node_left.append(Int32(Int(t[5])))
                arrays.node_right.append(Int32(Int(t[6])))
            cur_splits += 1
        elif kind == String("leaf"):
            _need(t, 4, "leaf")
            if Int(t[1]) != trees_seen - 1 or Int(t[2]) != cur_leaves:
                raise Error("a `leaf` record out of order")
            arrays.leaves.append(bitcast[DType.float32](UInt32(_token_bits(t[3], 8))))
            cur_leaves += 1
        elif kind == String("weight"):
            # a non-symmetric tree's leaf weights (64-bit tokens): counted
            # for completeness, not read by the walk
            _need(t, 4, "weight")
            if shape_kind != 2 or Int(t[1]) != trees_seen - 1 or Int(t[2]) != cur_weights:
                raise Error("a `weight` record out of order")
            _ = _token_bits(t[3], 16)
            cur_weights += 1
        elif kind == String("loss"):
            losses_seen += 1
        elif (
            kind == String("ctr_columns")
            or kind == String("ctr_table") or kind == String("ctr_entry")
            or kind == String("tensor_ctr_registry")
        ):
            _refuse_predict("a model with `" + kind + "` records")
        else:
            raise Error("unknown record keyword '" + kind + "'")
    if header_seen != 4 or features_seen != n_features or trees_seen != n_trees:
        raise Error("the model text is incomplete")
    if losses_seen != n_losses:
        raise Error("the model text declares " + String(n_losses) + " losses")
    if trees_seen > 0 and (
        cur_splits != want_splits or cur_leaves != want_leaves
        or cur_weights != want_weights
    ):
        raise Error("tree " + String(trees_seen - 1) + " is incomplete")
    if arrays.dim == -1:
        arrays.dim = 1
    return arrays^


def _refuse_predict(what: String) raises:
    raise Error(
        "no CPU implementation of _mojolearn_gbdt.gbdt_predict for " + what
        + "; the gbdt host binding reads float-only models (HostGBDT"
        " in python/mojolearn/_gbdt_host.py serves the rest)"
    )


# ===========================================================================
# THE FIT'S SCANS AND TAIL READS (`gbdt_fit_binding` and its arms)
# ===========================================================================


def gbdt_host_has_nan_f32(x: List[Float32]) -> Bool:
    """True when any value of `x` is NaN (`v != v`), stopping at the first."""
    var has_nan = False
    for i in range(len(x)):
        if x[i] != x[i]:
            has_nan = True
            break
    return has_nan


def gbdt_host_has_nan_ptr(xs: MutPointer[Float32, MutUntrackedOrigin], n: Int) -> Bool:
    """True when any of the `n` values at `xs` is NaN, stopping at the first."""
    var has_nan = False
    for i in range(n):
        var v = xs.unsafe_load(i)
        if v != v:
            has_nan = True
            break
    return has_nan


def gbdt_host_eval_const(ey: List[Float32], n_eval_rows: Int) -> Bool:
    """`UpdateUseBestModel` (`options_helper.cpp:100-113`): every eval target
    equals the first."""
    var eval_const = True
    for r in range(1, n_eval_rows):
        if ey[r] != ey[0]:
            eval_const = False
            break
    return eval_const


def gbdt_host_load_u32(cp: MutPointer[UInt32, MutUntrackedOrigin], n: Int) -> List[UInt32]:
    """The first `n` uint32 words at `cp`, in order."""
    var out = List[UInt32]()
    for f in range(n):
        out.append(cp.unsafe_load(f))
    return out^


def gbdt_host_load_sources(sp: MutPointer[UInt32, MutUntrackedOrigin], n_sources: Int) -> List[Int]:
    """The two-level FeatureFreq source ids, widened to `Int`."""
    var sources = List[Int]()
    for i in range(n_sources):
        sources.append(Int(sp.unsafe_load(i)))
    return sources^


def gbdt_host_group_sizes(
    gp: MutPointer[UInt32, MutUntrackedOrigin], n_groups: Int, tail_rows: Int
) raises -> List[Int]:
    """The group tail's sizes, each refused when zero, their sum refused
    unless it covers every row (`gbdt/train.mojo::train`'s words)."""
    var host_group_sizes = List[Int]()
    var covered = 0
    for g in range(n_groups):
        var size = gp.unsafe_load(g)
        if size == UInt32(0):
            raise Error("group_id: group " + String(g) + " has no rows")
        covered += Int(size)
        host_group_sizes.append(Int(size))
    if covered != tail_rows:
        raise Error(
            "group_id: the group sizes cover " + String(covered)
            + " rows of " + String(tail_rows)
        )
    return host_group_sizes^


def gbdt_host_read_pairs(
    pp: MutPointer[UInt32, MutUntrackedOrigin],
    pw: MutPointer[Float32, MutUntrackedOrigin],
    n_pairs: Int,
    mut host_pair_winners: List[UInt32],
    mut host_pair_losers: List[UInt32],
    mut host_pair_weights: List[Float32],
):
    """The PairLogit pairs tail: (winner, loser) word pairs and one weight each."""
    for q in range(n_pairs):
        host_pair_winners.append(pp.unsafe_load(2 * q))
        host_pair_losers.append(pp.unsafe_load(2 * q + 1))
        host_pair_weights.append(pw.unsafe_load(q))


def gbdt_host_binary_class_row_weights(
    y: List[Float32], n_rows: Int, class_weights: List[Float32]
) -> List[Float32]:
    """`train`'s weight column for binary class weights: `1.0 *
    class_weights[cls]`, cls the binarized target (y > 0.5)."""
    var row_weights = List[Float32](length=n_rows, fill=Float32(0.0))
    for r in range(n_rows):
        var cls = 1 if y[r] > Float32(0.5) else 0
        row_weights[r] = Float32(1.0) * class_weights[cls]
    return row_weights^


# ===========================================================================
# THE PREDICT WALKS (`gbdt_predict_binding`, `gbdt_predict_multi_binding`)
# ===========================================================================


def gbdt_host_predict_into(
    var m: _HostModelArrays,
    x_address: Int,
    n_rows: Int,
    n_features: Int,
    n_trees: Int,
    op: MutPointer[Float32, MutUntrackedOrigin],
) raises:
    """RAW approxes of a one-dimensional model, `n_rows` float32 to `op`."""
    var x = read_f32(x_address, n_rows * n_features)
    var out = List[Float32](length=n_rows, fill=Float32(0.0))
    var node_left = m.node_left.copy()
    var node_right = m.node_right.copy()
    if len(node_left) == 0:
        node_left.append(Int32(0))
        node_right.append(Int32(0))
    var non_symmetric = m.non_symmetric
    var split_feature = m.split_feature.copy()
    var split_bin = m.split_bin.copy()
    var split_take_bin = m.split_take_bin.copy()
    if len(split_feature) == 0:
        split_feature.append(Int32(0))
        split_bin.append(Int32(0))
        split_take_bin.append(Int32(0))
    var borders = m.borders.copy()
    if len(borders) == 0:
        borders.append(Float32(0.0))
    var leaves = m.leaves.copy()
    if len(leaves) == 0:
        leaves.append(Float32(0.0))
    gbdt_host_predict(
        x, n_rows, n_features,
        rebind[MutPointer[Int32, MutUntrackedOrigin]](m.border_offsets.unsafe_ptr()),
        rebind[MutPointer[Float32, MutUntrackedOrigin]](borders.unsafe_ptr()),
        rebind[MutPointer[Int32, MutUntrackedOrigin]](m.fold_counts.unsafe_ptr()),
        rebind[MutPointer[Int32, MutUntrackedOrigin]](m.one_hot.unsafe_ptr()),
        rebind[MutPointer[Int32, MutUntrackedOrigin]](m.nan_treatment.unsafe_ptr()),
        rebind[MutPointer[Int32, MutUntrackedOrigin]](m.tree_offsets.unsafe_ptr()),
        rebind[MutPointer[Int32, MutUntrackedOrigin]](split_feature.unsafe_ptr()),
        rebind[MutPointer[Int32, MutUntrackedOrigin]](split_bin.unsafe_ptr()),
        rebind[MutPointer[Int32, MutUntrackedOrigin]](split_take_bin.unsafe_ptr()),
        rebind[MutPointer[Int32, MutUntrackedOrigin]](node_left.unsafe_ptr()),
        rebind[MutPointer[Int32, MutUntrackedOrigin]](node_right.unsafe_ptr()),
        rebind[MutPointer[Int32, MutUntrackedOrigin]](m.leaf_offsets.unsafe_ptr()),
        rebind[MutPointer[Float32, MutUntrackedOrigin]](leaves.unsafe_ptr()),
        n_trees, 1, non_symmetric, len(m.split_feature), len(m.leaves), m.bias,
        out,
    )
    # KEEP-ALIVE. Every pointer above is an untracked `unsafe_ptr()`, which
    # is not a use of its list, and Mojo frees a value at its LAST USE: the
    # last use of `m` was the `m.bias` argument and of each local copy its
    # `unsafe_ptr()`, so all of them were freed before the walk ran and
    # `gbdt_host_predict` read the allocator's free-list word through
    # `tree_offsets_p[0]`, refusing every fixture on all seven runners of
    # gate run 34889886781 ("tree_offsets must start at 0 and end at
    # n_splits"). `_parse_model` makes that check true by construction
    # (one `split` per level, `cur_splits == cur_depth` enforced per tree),
    # so only a dangling read could fail it. A use after the call keeps
    # every list alive until the walk has returned.
    _ = split_feature^
    _ = split_bin^
    _ = split_take_bin^
    _ = node_left^
    _ = node_right^
    _ = borders^
    _ = leaves^
    _ = x^
    _ = m^
    for i in range(n_rows):
        op[i] = out[i]


def gbdt_host_predict_multi_into(
    var m: _HostModelArrays,
    x_address: Int,
    n_rows: Int,
    n_features: Int,
    n_trees: Int,
    dim: Int,
    mode: Int,
    op: MutPointer[Float32, MutUntrackedOrigin],
) raises:
    """Mode 0 RAW (`dim` columns), 1 SOFTMAX (`dim + 1`), 2 SIGMOID (`dim`),
    row-major to `op`."""
    var x = read_f32(x_address, n_rows * n_features)
    var out = List[Float32](length=n_rows * dim, fill=Float32(0.0))
    var node_left = m.node_left.copy()
    var node_right = m.node_right.copy()
    if len(node_left) == 0:
        node_left.append(Int32(0))
        node_right.append(Int32(0))
    var split_feature = m.split_feature.copy()
    var split_bin = m.split_bin.copy()
    var split_take_bin = m.split_take_bin.copy()
    if len(split_feature) == 0:
        split_feature.append(Int32(0))
        split_bin.append(Int32(0))
        split_take_bin.append(Int32(0))
    var borders = m.borders.copy()
    if len(borders) == 0:
        borders.append(Float32(0.0))
    var leaves = m.leaves.copy()
    if len(leaves) == 0:
        leaves.append(Float32(0.0))
    gbdt_host_predict(
        x, n_rows, n_features,
        rebind[MutPointer[Int32, MutUntrackedOrigin]](m.border_offsets.unsafe_ptr()),
        rebind[MutPointer[Float32, MutUntrackedOrigin]](borders.unsafe_ptr()),
        rebind[MutPointer[Int32, MutUntrackedOrigin]](m.fold_counts.unsafe_ptr()),
        rebind[MutPointer[Int32, MutUntrackedOrigin]](m.one_hot.unsafe_ptr()),
        rebind[MutPointer[Int32, MutUntrackedOrigin]](m.nan_treatment.unsafe_ptr()),
        rebind[MutPointer[Int32, MutUntrackedOrigin]](m.tree_offsets.unsafe_ptr()),
        rebind[MutPointer[Int32, MutUntrackedOrigin]](split_feature.unsafe_ptr()),
        rebind[MutPointer[Int32, MutUntrackedOrigin]](split_bin.unsafe_ptr()),
        rebind[MutPointer[Int32, MutUntrackedOrigin]](split_take_bin.unsafe_ptr()),
        rebind[MutPointer[Int32, MutUntrackedOrigin]](node_left.unsafe_ptr()),
        rebind[MutPointer[Int32, MutUntrackedOrigin]](node_right.unsafe_ptr()),
        rebind[MutPointer[Int32, MutUntrackedOrigin]](m.leaf_offsets.unsafe_ptr()),
        rebind[MutPointer[Float32, MutUntrackedOrigin]](leaves.unsafe_ptr()),
        n_trees, dim, m.non_symmetric, len(m.split_feature), len(m.leaves), m.bias,
        out,
    )
    # KEEP-ALIVE past the walk (see `gbdt_predict_binding`)
    _ = split_feature^
    _ = split_bin^
    _ = split_take_bin^
    _ = node_left^
    _ = node_right^
    _ = borders^
    _ = leaves^
    _ = x^
    _ = m^
    if mode == 0:
        for i in range(n_rows * dim):
            op[i] = out[i]
    elif mode == 1:
        var eff = dim
        for r in range(n_rows):
            # lane hr2-gbdt-host: `resident_link_kernel`'s softmax row,
            # in soft binary64 (checks/soft_f64.mojo), so this column
            # and every GPU column compute the same words
            var mx = SF64_ZERO
            for k in range(eff):
                var v = sf64_from_f32(out[r * eff + k])
                if sf64_gt(v, mx):
                    mx = v
            var se = SF64_ZERO
            for k in range(eff):
                se = sf64_add(se, sf64_exp(sf64_sub(sf64_from_f32(out[r * eff + k]), mx)))
            var e_pin = sf64_exp(sf64_neg(mx))
            se = sf64_add(se, e_pin)
            for k in range(eff):
                op[r * (eff + 1) + k] = ftz(sf64_to_f32(sf64_div(sf64_exp(sf64_sub(sf64_from_f32(out[r * eff + k]), mx)), se)))
            op[r * (eff + 1) + eff] = ftz(sf64_to_f32(sf64_div(e_pin, se)))
    else:
        for i in range(n_rows * dim):
            op[i] = ftz(sf64_to_f32(sf64_sigmoid_f32(out[i])))


# ===========================================================================
# THE LINKS (`gbdt_sigmoid_pair_binding`,
# `gbdt_binary_prediction_binding`)
# ===========================================================================


# TOMBSTONE: gbdt_sigmoid (dead native entry: no live caller) deleted 2026-10-26 by 7c1201602; code recoverable at 485ba9cc1.
# Restore: git apply experiments/removed/gbdt_sigmoid.patch


def gbdt_host_sigmoid_pair_into(rp: MutPointer[Float64, MutUntrackedOrigin], op: MutPointer[Float64, MutUntrackedOrigin], count: Int):
    """`out[2 * i] = 1 - p`, `out[2 * i + 1] = p`, swapped under the sabotage define."""
    for i in range(count):
        var r = rp.unsafe_load(i)
        # lane hr2-gbdt-host: `resident_link_kernel`'s pair, soft binary64
        var pb = sf64_ftz(sf64_sigmoid_f64(bitcast[DType.uint64](r)))
        var p = bitcast[DType.float64](pb)
        var q = bitcast[DType.float64](sf64_sub(SF64_ONE, pb))
        comptime if GBDT_HOST_SABOTAGE:
            op.unsafe_store(2 * i, p)
            op.unsafe_store(2 * i + 1, q)
        else:
            op.unsafe_store(2 * i, q)
            op.unsafe_store(2 * i + 1, p)


def gbdt_host_binary_margins_check(rp: MutPointer[Float32, MutUntrackedOrigin], n: Int) raises:
    # `binary_prediction_device`'s own refusal (`gbdt/binary_prediction.mojo`)
    for i in range(n):
        if not isfinite(rp.unsafe_load(i)):
            raise Error("binary prediction: finite Float32 margins required")


def gbdt_host_binary_prediction_into[probabilities: Bool, dtype: DType](
    rp: MutPointer[Float32, MutUntrackedOrigin],
    op: MutPointer[Scalar[dtype], MutUntrackedOrigin],
    n: Int,
):
    """`binary_prediction_kernel` per element: the probability pair or the class code."""
    for i in range(n):
        var margin = rp.unsafe_load(i)
        comptime if probabilities:
            var positive = ftz(identical_sigmoid(ftz(margin)))
            op.unsafe_store(2 * i, Scalar[dtype](ftz(Float32(1) - positive)))
            op.unsafe_store(2 * i + 1, Scalar[dtype](positive))
        else:
            var bits = bitcast[DType.uint32](margin)
            var is_positive = (bits & UInt32(0x80000000)) == 0 and (bits & UInt32(0x7fffffff)) != 0
            op.unsafe_store(i, Scalar[dtype](Int32(1) if is_positive else Int32(0)))
