# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The categorical columns of the GradientBoosting fit on the host, for the
gbdt-categorical-ctr lane (lane/cpu-training-gbdt-ordered, 2026-09-15).

HOST ONLY. The imports are `dense_category_code` (the GPU-free validation
the device fit itself runs on the host) and the symmetric oracle's token
writer.

WHAT THE LANE REACHES. tools/identity_break.py `gbdt-categorical-ctr` fits
`cat_features=[0]`, `one_hot_features=[1]` and `permutation_count=2` on the
coded columns of `_coded`, whose column 0 holds TWO categories. `train`
(`gbdt/train.mojo:1244-1275`) makes a categorical column with at most
`one_hot_max_size` categories (2, their GPU default,
`gbdt/options/catboost_options.mojo:956`) a ONE-HOT column, so the lane
builds no CTR column, `has_permutation_features` stays False and the fit
runs one permutation: its model texts on all nine fixtures carry no
`ctr_columns`, `ctr_table` or `ctr_entry` record (the 166-lane Apple
column, dumped on the M4). What this lane measures on a CPU is therefore
the one-hot arm: the flags decoded as `gbdt_fit` decodes them
(`gbdt/estimator.mojo:492-499`), `train`'s categorical validation, the
one-hot grid (`gbdt/host/gbdt_oracle.mojo::gbdt_host_fit`'s `one_hot_in`),
and the model text's `type cat` records and `split_type take_bin` splits.

REFUSED BY NAME: a categorical column with more than `one_hot_max_size`
categories (a CTR column; the CTR calcers, their permutations and tables
are not restated), and categorical or one-hot columns under any arm but
SymmetricTree with the Logloss loss.
"""
from gbdt.host.gbdt_oracle import GbdtHostModel, _nan_token, gbdt_f32_token
from gbdt.models.ctr_value_table import dense_category_code

#: `one_hot_max_size`, their GPU default (`catboost_options.mojo:956`).
comptime GBDT_ONE_HOT_MAX_SIZE = 2


def gbdt_resolve_one_hot(
    flags: List[UInt32], x_colmajor: List[Float32], n_rows: Int, n_features: Int
) raises -> List[Bool]:
    """`train`'s categorical column loop (`gbdt/train.mojo:1206-1275`) for
    columns that stay one-hot: bit 0 categorical, bit 1 one-hot; both is an
    error, a categorical column must be densely coded with at least two
    categories, and one above `one_hot_max_size` refuses by name."""
    var one_hot = List[Bool](length=n_features, fill=False)
    for f in range(n_features):
        var is_cat = (flags[f] & UInt32(1)) != UInt32(0)
        var flagged_one_hot = (flags[f] & UInt32(2)) != UInt32(0)
        if is_cat and flagged_one_hot:
            raise Error(
                "feature " + String(f) + " is in both cat_features and one_hot;"
                " cat_features makes the one-hot decision itself, from"
                " one_hot_max_size"
            )
        if not is_cat:
            one_hot[f] = flagged_one_hot
            continue
        var maxc = 0
        var seen = List[Bool](length=1, fill=False)
        for r in range(n_rows):
            var c = dense_category_code(x_colmajor[f * n_rows + r], f, r)
            if c > maxc:
                maxc = c
                seen.resize(maxc + 1, False)
            seen[c] = True
        var unique_values = maxc + 1
        if unique_values <= 1:
            raise Error(
                "Error: useless catFeature found (feature " + String(f)
                + " has one category)"
            )
        for c in range(unique_values):
            if not seen[c]:
                raise Error(
                    "cat_features column " + String(f) + " is not densely coded:"
                    " category " + String(c) + " is absent from 0.." + String(maxc)
                )
        if unique_values > GBDT_ONE_HOT_MAX_SIZE:
            raise Error(
                "no CPU implementation of _mojolearn_gbdt.gbdt_fit for a"
                " cat_features column with more than one_hot_max_size ("
                + String(GBDT_ONE_HOT_MAX_SIZE) + ") categories (feature "
                + String(f) + " has " + String(unique_values) + ", a CTR"
                " column); the gbdt host binding carries one-hot categorical"
                " columns only, see gbdt/host/gbdt_oracle_onehot.mojo"
            )
        one_hot[f] = True
    return one_hot^


def gbdt_host_model_text_one_hot(m: GbdtHostModel, one_hot: List[Bool]) raises -> String:
    """`model_text` (`gbdt/models/model_text.mojo:374-670`) for an oblivious
    one-dimensional model with one-hot columns: `type cat` and `one_hot 1`
    on their feature records, the trailing `split_type take_bin` on their
    splits, otherwise `gbdt_host_model_text`'s records."""
    var n_features = len(m.fold_counts)
    var out = String("")
    out += "# mojolearn model. One record per line, keyword first.\n"
    out += "# Every float is <decimal>/<IEEE-754 bits in hex>; the BITS are\n"
    out += "# what is loaded, because this toolchain's decimal formatter\n"
    out += "# loses one ULP on ~0.46% of float32 values (measured).\n"
    out += "# Format and CTR seam: gbdt/models/model_text.mojo.\n"
    out += String("format ") + String("mojolearn-model") + " " + String(2) + "\n"
    out += String("features ") + String(n_features) + " " + String(n_features) + "\n"
    out += String("trees ") + String(m.n_trees()) + "\n"
    out += String("losses ") + String(len(m.losses)) + "\n"
    for f in range(n_features):
        var line = (
            String("feature ") + String(f) + " folds " + String(m.fold_counts[f])
            + " one_hot " + String(1 if one_hot[f] else 0)
            + " type " + (String("cat") if one_hot[f] else String("float"))
            + " nan " + _nan_token(m.nan_treatment[f])
            + " borders " + String(len(m.borders[f]))
        )
        for b in range(len(m.borders[f])):
            line += " " + gbdt_f32_token(m.borders[f][b])
        out += line + "\n"
    for t in range(m.n_trees()):
        var lo = m.tree_split_offsets[t]
        var depth = m.tree_split_offsets[t + 1] - lo
        var leaf_lo = m.tree_leaf_offsets[t]
        var n_values = 1 << depth
        if m.tree_leaf_offsets[t + 1] - leaf_lo != n_values:
            raise Error("tree " + String(t) + " has the wrong leaf count")
        out += String("tree ") + String(t) + " depth " + String(depth) + " dim 1 weights 0\n"
        for level in range(depth):
            var fid = m.split_features[lo + level]
            var line = (
                String("split ") + String(t) + " " + String(level) + " "
                + String(fid) + " " + String(m.split_bins[lo + level])
            )
            if one_hot[fid]:
                line += " split_type take_bin"
            out += line + "\n"
        for i in range(n_values):
            out += String("leaf ") + String(t) + " " + String(i) + " " + gbdt_f32_token(m.leaf_values[leaf_lo + i]) + "\n"
    for i in range(len(m.losses)):
        out += String("loss ") + String(i) + " " + _f64(m.losses[i]) + "\n"
    return out^


def _f64(v: Float64) -> String:
    from gbdt.host.gbdt_oracle import gbdt_f64_token
    return gbdt_f64_token(v)
