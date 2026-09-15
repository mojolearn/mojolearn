# SPDX-License-Identifier: Apache-2.0
"""Public CPU inference for saved GBDT models with CTR tables and tensor
CTRs (lane/inference-gbdt-ctr-tables, 2026-09-15): HostGBDT parses the
`ctr_table`, `ctr_entry`, `tensor_ctr_registry` and `feature_freq_tensor`
records and the forest host binding applies them with the same
`expand_raw_columns` and `expand_tensor_ctr_columns` the GPU predict calls.

What the source checks hold (no binding): the records parse with the Mojo
reader's checks, the raw input count is the column plan's, and what the GPU
predict refuses is refused at load by name (a CTR type with no apply-time
arithmetic, a model with both kinds, a model declaring more CTR columns than
it carries tables).

The runtime checks (skipped, and SAID to be skipped, unless the forest host
binary is built): two hand-written models whose leaves name the path each row
took, so a category the learn pool never saw, a category seen once, a
combination, a code past a source's cardinality and a NaN or non-integer
categorical value each land where `TCtrValueTable.value_for` and
`tensor_value_for_key` send them. The bit claim against the GPU column is
identity_break's gbdt-categorical-ctr-tables and gbdt-tensor-ctr-tables lanes.

    cd python && python3 -m mojolearn.tests.test_gbdt_host_ctr
"""
import os
import sys

from mojolearn._gbdt_host import parse_model_text

HEADER = "format mojolearn-model 2\n"

#: one categorical input with three categories behind one FeatureFreq column
#: (counts 6, 3, 1 over 10 learn rows, prior {0, 1}): values 6/11, 3/11, 1/11
#: and, for a code the learn pool never saw, 0/11. Borders 0.05 and 0.2 put
#: them in bins 2, 2, 1 and 0, and the depth-2 tree's leaves are the path:
#: 0.25 unseen, 0.5 the category seen once, 1.0 the frequent ones.
CTR_TEXT = HEADER + (
    "features 1 0\ntrees 1\nlosses 0\nctr_columns 1\n"
    "feature 0 folds 2 one_hot 0 type ctr nan as_is borders 2 0.05/3d4ccccd 0.2/3e4ccccd\n"
    "ctr_table 0 source 0 type FeatureFreq prior_num 0/00000000 prior_denom 1/3f800000 "
    "shift 0/00000000 scale 1/3f800000 denom 10 classes 0 target_border 0 entries 3\n"
    "ctr_entry 0 0 6\nctr_entry 0 1 3\nctr_entry 0 2 1\n"
    "tree 0 depth 2 dim 1 weights 0\nsplit 0 0 0 0\nsplit 0 1 0 1\n"
    "leaf 0 0 0.25/3e800000\nleaf 0 1 0.5/3f000000\nleaf 0 2 0.75/3f400000\nleaf 0 3 1/3f800000\n"
)

#: two categorical sources (two categories each) and one tensor FeatureFreq
#: column over them: combination counts 5, 3, 1, 1 over 10 rows, prior {0, 1}.
#: Combinations (0, x) are in bin 2, (1, x) in bin 1, and a source-0 code past
#: its cardinality takes key -1, count 0, bin 0.
TENSOR_TEXT = HEADER + (
    "features 3 3\ntrees 1\nlosses 0\ntensor_ctr_registry 2 1\n"
    "feature_freq_tensor 2 hash_hi 1 hash_lo 2 sources 2 0 1 cardinalities 2 2 2 splits 0 "
    "classes 0 target_border -1 prior_bits 0 1065353216 denominator 10 counts 4 5 3 1 1\n"
    "feature 0 folds 1 one_hot 1 type cat nan as_is borders 1 0.5/3f000000\n"
    "feature 1 folds 1 one_hot 1 type cat nan as_is borders 1 0.5/3f000000\n"
    "feature 2 folds 2 one_hot 0 type tensor_ctr nan as_is borders 2 0.05/3d4ccccd 0.2/3e4ccccd\n"
    "tree 0 depth 2 dim 1 weights 0\nsplit 0 0 2 0\nsplit 0 1 2 1\n"
    "leaf 0 0 0.25/3e800000\nleaf 0 1 0.5/3f000000\nleaf 0 2 0.75/3f400000\nleaf 0 3 1/3f800000\n"
)


def _raises(fn, needle):
    try:
        fn()
    except Exception as exc:  # the binding raises its Mojo error as a plain exception
        assert needle in str(exc), f"expected {needle!r} in {exc}"
        return
    raise AssertionError(f"no refusal naming {needle!r}")


def test_ctr_records_parse_with_the_column_plan():
    a = parse_model_text(CTR_TEXT)
    assert a["n_features"] == 1 and a["n_input_features"] == 1
    assert a["n_ctr_tables"] == 1 and a["n_tensor_tables"] == 0 and a["n_ctr_counts"] == 3
    t = parse_model_text(TENSOR_TEXT)
    assert t["n_features"] == 3 and t["n_input_features"] == 2
    assert t["n_tensor_tables"] == 1 and t["n_ctr_counts"] == 4


def test_unsupported_ctr_type_refuses_by_name():
    for name in ("Buckets", "BinarizedTargetMeanValue", "FloatTargetMeanValue"):
        _raises(lambda: parse_model_text(CTR_TEXT.replace("type FeatureFreq", f"type {name}")), name)
    _raises(lambda: parse_model_text(CTR_TEXT.replace("type FeatureFreq", "type Mystery")), "Mystery")


def test_what_predict_floats_refuses_is_refused_at_load():
    _raises(lambda: parse_model_text(CTR_TEXT.replace("ctr_columns 1", "ctr_columns 2")),
            "2 CTR columns and 1 CTR tables")
    _raises(lambda: parse_model_text(CTR_TEXT.replace("ctr_entry 0 2 1\n", "")), "declares 3 entries")
    _raises(lambda: parse_model_text(CTR_TEXT.replace("type ctr", "type float")), "says type 'float'")
    _raises(lambda: parse_model_text(TENSOR_TEXT.replace("counts 4 5 3 1 1", "counts 4 5 3 1 2")),
            "do not sum to denominator")
    _raises(lambda: parse_model_text(TENSOR_TEXT.replace("sources 2 0 1", "sources 2 1 0")), "canonical")


def _binary_built():
    from mojolearn._forest_host import binary_path
    return os.path.exists(binary_path())


def _host(text, n_inputs):
    import tempfile
    from mojolearn import _serialize
    from mojolearn._array import Array
    from mojolearn._buffer import frombytes
    from mojolearn._gbdt_host import GBDT_FORMAT, HostGBDT
    body = text.encode("utf-8")
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "m.npz")
        _serialize.write_npz(path, {
            "format": GBDT_FORMAT, "numeric_mode": "identical", "estimator": "GradientBoosting",
            "loss": "RMSE", "model": frombytes(body, "<u1", (len(body),)),
            "meta": Array.from_list([n_inputs, 1, -1, -1, 0], "<i8"),
            "bias": Array.from_list([0.0], "<f8"),
        })
        return HostGBDT.from_file(path)


def test_ctr_lookup_unseen_seen_once_and_nan_when_built():
    if not _binary_built():
        print("SKIP: the forest host binary is not built")
        return
    host = _host(CTR_TEXT, 1)
    got = list(host.predict([[0.0], [1.0], [2.0], [3.0], [40.0]]).tolist())
    assert got == [1.0, 1.0, 0.5, 0.25, 0.25], got
    _raises(lambda: host.predict([[float("nan")]]), "is not finite")
    _raises(lambda: host.predict([[1.5]]), "exact non-negative integer code")
    _raises(lambda: host.predict([[-1.0]]), "outside the UInt32 code range")


def test_tensor_combination_and_past_cardinality_when_built():
    if not _binary_built():
        print("SKIP: the forest host binary is not built")
        return
    host = _host(TENSOR_TEXT, 2)
    got = list(host.predict([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0], [2.0, 0.0]]).tolist())
    assert got == [1.0, 1.0, 0.5, 0.5, 0.25], got
    _raises(lambda: host.predict([[float("nan"), 0.0]]), "cannot quantize NaN")


if __name__ == "__main__":
    names = [n for n in sorted(globals()) if n.startswith("test_")]
    for name in names:
        globals()[name]()
        print("ok", name)
    print(f"{len(names)} passed")
    sys.exit(0)
