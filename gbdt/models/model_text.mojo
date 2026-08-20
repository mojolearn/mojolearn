"""A trained model, written to a file and read back exactly.

NOT A PORT, AND THE ONLY HONEST WAY TO SAY IT. CatBoost serializes a model
with flatbuffers (`catboost/libs/model/flatbuffers/model.fbs`, written by
`libs/model/model.cpp`) and exports JSON, CoreML, ONNX, PMML, C++ and Python
from `libs/model/model_export/`. None of that is ported here and none of it
is mirrored. This file writes a format of our own, and that is DEVIATION 49
in `PORTING.md` with the reason. What IS taken from them is the CONTENT
list: an applied model needs the trees, the leaf values in leaf order, and
the quantization borders, which is why their `TModelTrees` carries
`FloatFeatures` (borders included) beside `TreeSplits` and `LeafValues`.
Our `TrainedModel` carries the same things and so does this file.

Before this file there was NO model persistence in this repository at all --
`grep "open(" gbdt/` returned nothing -- so a trained model could not leave
the process.

## WHY TEXT, AND WHY EVERY FLOAT CARRIES ITS BITS

Text, because every fixture in this repository is text (`bench/oracle.txt`,
`bench/minentropy_oracle.txt`) and because a format you can `diff`, `grep`
and read in an error message is worth more here than a compact one. The
record-per-line, keyword-first shape is `bench/oracle.txt`'s.

Bits, because **decimal text through this toolchain is NOT exact and that was
MEASURED, not assumed.** 200,000 random bit patterns per width through
`String(x)` and back (`mojo_only/model_io_check.mojo` re-takes this
measurement every run, and REFUSES to pass if it ever comes back clean,
because then the reason for the bits is gone):

    Float32   917 of 199,223 finite patterns wrong  (0.46%)
    Float64   212 of 199,907 finite patterns wrong  (0.11%)

every one of them off by exactly one ULP, plus a whole class that loses
everything: `String(Float32(1.4e-45))` is `"0.0"`. So `%g`-style decimal at
any precision this toolchain can produce would corrupt roughly one leaf value
in two hundred, silently, and a round-trip gate written with a tolerance
would never see it. (For the record of what a correct writer looks like:
CatBoost's own JSON export sets `FloatNDigits = 9` and `DoubleNDigits = 17`
in `WriteJsonWithCatBoostPrecision`, `libs/helpers/json_helpers.h:22-32`,
which ARE the round-trip digit counts. Their writer honors them. Ours does
not.)

So every float is written as TWO halves joined by `/`:

    -0.41702934/bed5834f
     ^ what a human reads   ^ the IEEE-754 bits, which is what is loaded

The loader reads the bits and ignores the decimal for the VALUE, then checks
the decimal agrees with the bits to within one ULP, so a hand-edit of the
readable half cannot diverge from the authoritative half unnoticed. The
one-ULP slack is exactly the formatter error measured above; it is not a
tolerance on the model.

## THE RECORDS

    format mojolearn-model 1
    features <n_features> <n_one_hot_flags>
    trees <n_trees>
    losses <n_losses>
    feature <f> folds <k> one_hot <0|1> type float borders <n> <tok>...
    tree <t> depth <d> dim <k> weights <0|1>
    split <t> <level> <feature_id> <bin_idx>
    leaf <t> <leaf_index> <tok>
    weight <t> <leaf_index> <tok>          (only when weights is 1)
    loss <iteration> <tok64>

The four header records come first and in that order. Every indexed record
must arrive in ascending, gap-free index order and the counts must add up:
`features`, `trees` and `losses` are declared in the header so a section that
was never written is a load error rather than a shorter model. `depth` is
declared so a tree missing a leaf is a load error rather than a tree with a
hole in it. **An unknown record keyword is an ERROR, not a skipped line**,
which is what keeps a future writer's records from being silently dropped by
an older reader.

`n_one_hot_flags` is 0 or `n_features`, because `TrainedModel.one_hot` is
either empty or one flag per feature and `build_layout` distinguishes the
two by length. The per-feature `one_hot` token is written as 0 when the
vector is absent and read only when the header says the vector is there.

## THE LEAF ORDER IS THE MODEL AND IT IS WRITTEN DOWN

`leaf <t> <i> <tok>` carries `i` explicitly rather than relying on line
order. An oblivious tree's leaf index is `sum over levels of bit_l << l`
with level 0 the LEAST significant bit (see `oblivious_model.mojo`), and
reading the bits the other way round is a PERMUTATION of the right answer:
every total is preserved and every individual prediction is wrong. A
permutation is invisible to any check that sums, so the index is in the file
where a diff can see it, and the loader requires it to be exactly the
position it lands in.

============================ THE CTR SEAM ============================
A later lane needs to store CTR tables so a categorical model can score raw
data (`RECON_CTRS.md` step 5). This is where they go and nothing here has to
change for them to arrive.

* The `feature` record already carries `type float`. A CTR-valued feature
  writes `type ctr` and a raw categorical writes `type cat`. A reader of
  this version REFUSES an unknown type by name, so an old reader never
  half-loads a categorical model.
* The CTR tables go in their own records AFTER the `feature` block and
  before the first `tree`:

      ctr_table <feature_id> <ctr_type> <prior_num>/<hex> <prior_denom>/<hex>
      ctr_entry <feature_id> <category_hash> <value>/<hex>
      ctr_borders <feature_id> <n> <tok>...

  `ctr_table` mirrors their `TCtrFeature`'s `prior_numerator` /
  `prior_denomerator` (`json_model_helpers.cpp:104-114`), `ctr_borders`
  their `TCtrFeature::Borders` -- the CTR VALUE binarization, which is
  Uniform-15 and NOT the GreedyLogSum of a numeric feature
  (`cat_feature_options.cpp:169`) -- and `ctr_entry` is the category-to-value
  mapping their `TCtrValueTable` holds, the thing an applied model cannot do
  without.
* A model with no categorical features writes NONE of these records, so
  every file this version writes stays byte-identical when they land, and
  the `format` version stays 1 for float-only models.
* What must grow beside them, named so it is not rediscovered:
  `TBinarySplit` needs the split TYPE their `TSplitType` carries (its
  docstring in `oblivious_model.mojo` already says so), because a one-hot
  split is a different predicate and today the predicate is recovered from
  the LAYOUT rather than from the model. And `gbdt/models/cuda/evaluator.mojo`
  has no one-hot arm at all: its `XorMask` slot is documented as staying
  zero, so a categorical model cannot be scored through the device evaluator
  until that lands. Neither is built here.
======================================================================
"""

from std.memory import bitcast

from gbdt.models.oblivious_model import (
    TAdditiveModel,
    TBinarySplit,
    TObliviousTreeModel,
    TObliviousTreeStructure,
)
from gbdt.train import TrainedModel


#: The name in the `format` record. A file that does not start with this is
#: not ours and is refused rather than guessed at.
comptime MODEL_TEXT_NAME = "mojolearn-model"

#: Bumped only when a record an OLDER reader needs changes meaning. Adding
#: the CTR records above does not bump it; see THE CTR SEAM.
comptime MODEL_TEXT_VERSION = 1


# ---------------------------------------------------------------------------
# Fixed-width hex, so the columns line up in a diff and a truncated token is
# a length error rather than a different number.
# ---------------------------------------------------------------------------


def _hex_fixed(v: UInt64, digits: Int) -> String:
    var table = String("0123456789abcdef")
    var out = String("")
    for i in range(digits):
        var nib = Int((v >> UInt64((digits - 1 - i) * 4)) & UInt64(0xF))
        out += String(table[byte=nib])
    return out^


def _hex_value(code: Int) raises -> Int:
    if code >= 48 and code <= 57:
        return code - 48
    if code >= 97 and code <= 102:
        return code - 87
    if code >= 65 and code <= 70:
        return code - 55
    raise Error("not a hex digit: byte value " + String(code))


def _parse_hex(s: String, digits: Int) raises -> UInt64:
    if s.byte_length() != digits:
        raise Error(
            "expected " + String(digits) + " hex digits, got '" + s + "'"
        )
    var v = UInt64(0)
    for i in range(digits):
        v = (v << UInt64(4)) | UInt64(_hex_value(ord(String(s[byte=i]))))
    return v


# ---------------------------------------------------------------------------
# Float tokens: `<decimal>/<hex bits>`.
# ---------------------------------------------------------------------------


def f32_token(v: Float32) -> String:
    """`<decimal>/<8 hex digits>`. The decimal is for the reader's eyes."""
    return (
        String(v) + "/" + _hex_fixed(UInt64(bitcast[DType.uint32](v)), 8)
    )


def f64_token(v: Float64) -> String:
    """`<decimal>/<16 hex digits>`."""
    return String(v) + "/" + _hex_fixed(bitcast[DType.uint64](v), 16)


def _decimal_agrees_32(text: String, bits: UInt32) raises:
    """The readable half must be the SAME number as the authoritative half,
    to within the one ULP the formatter is measured to lose. Non-finite
    values are skipped: their decimal spelling is `inf`/`nan`, which this
    toolchain's float parser is not asked to read."""
    if ((bits >> UInt32(23)) & UInt32(0xFF)) == UInt32(0xFF):
        return
    var reparsed = bitcast[DType.uint32](Float32(Float64(text)))
    var d = bits - reparsed if bits > reparsed else reparsed - bits
    if d > UInt32(1):
        raise Error(
            "the readable half '" + text + "' and the bits "
            + _hex_fixed(UInt64(bits), 8) + " are different numbers ("
            + String(d) + " ULP apart)"
        )


def _decimal_agrees_64(text: String, bits: UInt64) raises:
    if ((bits >> UInt64(52)) & UInt64(0x7FF)) == UInt64(0x7FF):
        return
    var reparsed = bitcast[DType.uint64](Float64(text))
    var d = bits - reparsed if bits > reparsed else reparsed - bits
    if d > UInt64(1):
        raise Error(
            "the readable half '" + text + "' and the bits "
            + _hex_fixed(bits, 16) + " are different numbers ("
            + String(d) + " ULP apart)"
        )


def parse_f32(tok: String) raises -> Float32:
    var parts = tok.split("/")
    if len(parts) != 2:
        raise Error(
            "malformed float token '" + tok
            + "', want <decimal>/<8 hex digits>"
        )
    var bits = UInt32(_parse_hex(String(parts[1]), 8))
    _decimal_agrees_32(String(parts[0]), bits)
    return bitcast[DType.float32](bits)


def parse_f64(tok: String) raises -> Float64:
    var parts = tok.split("/")
    if len(parts) != 2:
        raise Error(
            "malformed float token '" + tok
            + "', want <decimal>/<16 hex digits>"
        )
    var bits = _parse_hex(String(parts[1]), 16)
    _decimal_agrees_64(String(parts[0]), bits)
    return bitcast[DType.float64](bits)


# ---------------------------------------------------------------------------
# Writing.
# ---------------------------------------------------------------------------


def model_text(tm: TrainedModel) raises -> String:
    """The whole file as one String. Split out from `save_model` so a check
    can look at the bytes without touching the filesystem."""
    var n_features = len(tm.fold_counts)
    if len(tm.borders) != n_features:
        raise Error(
            "TrainedModel has " + String(n_features) + " fold counts and "
            + String(len(tm.borders)) + " border lists"
        )
    if len(tm.one_hot) != 0 and len(tm.one_hot) != n_features:
        raise Error(
            "one_hot flags must be empty or one per feature: got "
            + String(len(tm.one_hot)) + " for " + String(n_features)
        )

    var out = String("")
    out += "# mojolearn model. One record per line, keyword first.\n"
    out += "# Every float is <decimal>/<IEEE-754 bits in hex>; the BITS are\n"
    out += "# what is loaded, because this toolchain's decimal formatter\n"
    out += "# loses one ULP on ~0.46% of float32 values (measured).\n"
    out += "# Format and CTR seam: gbdt/models/model_text.mojo.\n"
    out += (
        String("format ") + String(MODEL_TEXT_NAME) + " "
        + String(MODEL_TEXT_VERSION) + "\n"
    )
    out += (
        String("features ") + String(n_features) + " "
        + String(len(tm.one_hot)) + "\n"
    )
    out += String("trees ") + String(tm.model.size()) + "\n"
    out += String("losses ") + String(len(tm.losses)) + "\n"
    # CTR COLUMN COUNT. Written ONLY when non-zero, so a float-only model's
    # file is byte-identical to what this format wrote before the field
    # existed and the version stays 1 -- the rule THE CTR SEAM sets out.
    #
    # It must travel, and not merely for completeness: `predict_floats`
    # REFUSES a model carrying CTR columns, because a CTR value is a
    # statistic of the learn pool and scoring a new row needs the final CTR
    # tables. A round trip that dropped the count would turn a model that
    # correctly refuses into one that silently scores rows against a grid
    # built for different values. Losing a safety refusal in serialization
    # is worse than losing a leaf.
    if tm.ctr_column_count != 0:
        out += String("ctr_columns ") + String(tm.ctr_column_count) + "\n"

    for f in range(n_features):
        var flag = 1 if len(tm.one_hot) != 0 and tm.one_hot[f] else 0
        var line = (
            String("feature ") + String(f) + " folds "
            + String(tm.fold_counts[f]) + " one_hot " + String(flag)
            + " type float borders " + String(len(tm.borders[f]))
        )
        for b in range(len(tm.borders[f])):
            line += " " + f32_token(tm.borders[f][b])
        out += line + "\n"

    for t in range(tm.model.size()):
        ref weak = tm.model.weak_models[t]
        var depth = weak.structure.get_depth()
        var n_leaves = 1 << depth
        var has_weights = 1 if len(weak.leaf_weights) != 0 else 0
        if len(weak.leaf_values) != n_leaves:
            raise Error(
                "tree " + String(t) + " has depth " + String(depth)
                + " and " + String(len(weak.leaf_values))
                + " leaf values, not " + String(n_leaves)
            )
        if has_weights == 1 and len(weak.leaf_weights) != n_leaves:
            raise Error(
                "tree " + String(t) + " has " + String(len(weak.leaf_weights))
                + " leaf weights, not " + String(n_leaves)
            )
        out += (
            String("tree ") + String(t) + " depth " + String(depth)
            + " dim " + String(weak.dim) + " weights " + String(has_weights)
            + "\n"
        )
        for level in range(depth):
            out += (
                String("split ") + String(t) + " " + String(level) + " "
                + String(Int(weak.structure.splits[level].feature_id)) + " "
                + String(Int(weak.structure.splits[level].bin_idx)) + "\n"
            )
        for i in range(n_leaves):
            out += (
                String("leaf ") + String(t) + " " + String(i) + " "
                + f32_token(weak.leaf_values[i]) + "\n"
            )
        if has_weights == 1:
            for i in range(n_leaves):
                out += (
                    String("weight ") + String(t) + " " + String(i) + " "
                    + f32_token(weak.leaf_weights[i]) + "\n"
                )

    for i in range(len(tm.losses)):
        out += (
            String("loss ") + String(i) + " " + f64_token(tm.losses[i]) + "\n"
        )
    return out^


def save_model(path: String, tm: TrainedModel) raises:
    """Write the model. One `write`, because the file is small and a partial
    file is worse than no file."""
    var text = model_text(tm)
    with open(path, "w") as f:
        f.write(text)


# ---------------------------------------------------------------------------
# Reading.
# ---------------------------------------------------------------------------


def _fields(line: String) raises -> List[String]:
    """Split on spaces, dropping empty pieces (`bench/oracle.txt`'s reader
    does the same, and for the same reason: a doubled space is not a field).
    """
    var out = List[String]()
    for piece in line.split(" "):
        var s = String(piece)
        if s.byte_length() > 0:
            out.append(s)
    return out^


def _expect(t: List[String], i: Int, word: String, kind: String) raises:
    if i >= len(t) or t[i] != word:
        raise Error(
            "a `" + kind + "` record wants '" + word + "' at position "
            + String(i)
        )


def load_model_text(text: String) raises -> TrainedModel:
    """Parse the format above. Strict on purpose: every count in the header
    must be met exactly, every index must be the position it lands in, and
    an unrecognized keyword raises rather than being skipped."""
    var n_features = -1
    var n_flags = -1
    var n_trees = -1
    var n_losses = -1
    var ctr_column_count = 0
    var header_seen = 0

    var fold_counts = List[Int]()
    var one_hot = List[Bool]()
    var borders = List[List[Float32]]()
    var losses = List[Float64]()
    var model = TAdditiveModel()

    var depths = List[Int]()
    var splits_seen = List[Int]()
    var leaves_seen = List[Int]()
    var weights_seen = List[Int]()
    var weights_declared = List[Int]()
    var features_seen = 0
    var trees_seen = 0
    var losses_seen = 0
    var line_no = 0

    for raw in text.split("\n"):
        line_no += 1
        var line = String(raw)
        if line.byte_length() == 0 or line.startswith("#"):
            continue
        var t = _fields(line)
        if len(t) == 0:
            continue
        var kind = t[0]

        try:
            if kind == String("format"):
                if header_seen != 0:
                    raise Error("a second `format` record")
                if len(t) != 3 or t[1] != String(MODEL_TEXT_NAME):
                    raise Error(
                        "not a " + String(MODEL_TEXT_NAME) + " file"
                    )
                if Int(t[2]) != MODEL_TEXT_VERSION:
                    raise Error(
                        "format version " + String(t[2]) + ", this reader"
                        " is version " + String(MODEL_TEXT_VERSION)
                    )
                header_seen = 1
            elif kind == String("features"):
                if header_seen != 1:
                    raise Error("`features` must follow `format`")
                n_features = Int(t[1])
                n_flags = Int(t[2])
                if n_features < 0:
                    raise Error("negative feature count")
                if n_flags != 0 and n_flags != n_features:
                    raise Error(
                        "one_hot flag count " + String(n_flags)
                        + " is neither 0 nor " + String(n_features)
                    )
                header_seen = 2
            elif kind == String("trees"):
                if header_seen != 2:
                    raise Error("`trees` must follow `features`")
                n_trees = Int(t[1])
                header_seen = 3
            elif kind == String("losses"):
                if header_seen != 3:
                    raise Error("`losses` must follow `trees`")
                n_losses = Int(t[1])
                header_seen = 4
            elif kind == String("ctr_columns"):
                if header_seen != 4:
                    raise Error("`ctr_columns` must follow `losses`")
                ctr_column_count = Int(t[1])
            elif kind == String("feature"):
                if header_seen != 4:
                    raise Error("a `feature` record before the header ended")
                if Int(t[1]) != features_seen:
                    raise Error(
                        "features must arrive in order: expected "
                        + String(features_seen) + ", got " + String(t[1])
                    )
                _expect(t, 2, String("folds"), String("feature"))
                _expect(t, 4, String("one_hot"), String("feature"))
                _expect(t, 6, String("type"), String("feature"))
                if t[7] != String("float"):
                    raise Error(
                        "feature " + String(t[1]) + " has type '" + t[7]
                        + "'. This reader carries float features only; a"
                        " categorical or CTR-valued feature needs the CTR"
                        " records described in gbdt/models/model_text.mojo"
                    )
                _expect(t, 8, String("borders"), String("feature"))
                fold_counts.append(Int(t[3]))
                if n_flags != 0:
                    one_hot.append(Int(t[5]) != 0)
                var n_b = Int(t[9])
                if len(t) != 10 + n_b:
                    raise Error(
                        "feature " + String(t[1]) + " declares " + String(n_b)
                        + " borders and carries " + String(len(t) - 10)
                    )
                var bs = List[Float32]()
                for i in range(n_b):
                    bs.append(parse_f32(t[10 + i]))
                borders.append(bs^)
                features_seen += 1
            elif kind == String("tree"):
                if Int(t[1]) != trees_seen:
                    raise Error(
                        "trees must arrive in order: expected "
                        + String(trees_seen) + ", got " + String(t[1])
                    )
                _expect(t, 2, String("depth"), String("tree"))
                _expect(t, 4, String("dim"), String("tree"))
                _expect(t, 6, String("weights"), String("tree"))
                var depth = Int(t[3])
                if depth < 0 or depth > 16:
                    raise Error("tree depth " + String(depth) + " is not sane")
                var st = TObliviousTreeStructure()
                var weak = TObliviousTreeModel(st^)
                weak.dim = Int(t[5])
                model.add_weak_model(weak^)
                depths.append(depth)
                splits_seen.append(0)
                leaves_seen.append(0)
                weights_seen.append(0)
                weights_declared.append(Int(t[7]))
                trees_seen += 1
            elif kind == String("split"):
                var ti = Int(t[1])
                if ti >= trees_seen:
                    raise Error("a `split` for tree " + String(ti)
                                + " before its `tree` record")
                if Int(t[2]) != splits_seen[ti]:
                    raise Error(
                        "tree " + String(ti) + " levels must arrive in"
                        " order: expected " + String(splits_seen[ti])
                        + ", got " + String(t[2])
                    )
                model.weak_models[ti].structure.splits.append(
                    TBinarySplit(Int32(Int(t[3])), Int32(Int(t[4])))
                )
                splits_seen[ti] += 1
            elif kind == String("leaf"):
                var ti = Int(t[1])
                if ti >= trees_seen:
                    raise Error("a `leaf` for tree " + String(ti)
                                + " before its `tree` record")
                if Int(t[2]) != leaves_seen[ti]:
                    raise Error(
                        "tree " + String(ti) + " leaves must arrive in leaf"
                        " order: expected " + String(leaves_seen[ti])
                        + ", got " + String(t[2])
                    )
                model.weak_models[ti].leaf_values.append(parse_f32(t[3]))
                leaves_seen[ti] += 1
            elif kind == String("weight"):
                var ti = Int(t[1])
                if ti >= trees_seen:
                    raise Error("a `weight` for tree " + String(ti)
                                + " before its `tree` record")
                if weights_declared[ti] == 0:
                    raise Error(
                        "tree " + String(ti) + " declared no weights and"
                        " carries a `weight` record"
                    )
                if Int(t[2]) != weights_seen[ti]:
                    raise Error(
                        "tree " + String(ti) + " weights must arrive in leaf"
                        " order: expected " + String(weights_seen[ti])
                        + ", got " + String(t[2])
                    )
                model.weak_models[ti].leaf_weights.append(parse_f32(t[3]))
                weights_seen[ti] += 1
            elif kind == String("loss"):
                if Int(t[1]) != losses_seen:
                    raise Error(
                        "losses must arrive in order: expected "
                        + String(losses_seen) + ", got " + String(t[1])
                    )
                losses.append(parse_f64(t[2]))
                losses_seen += 1
            else:
                raise Error(
                    "unknown record `" + kind + "`. This reader refuses a"
                    " record it does not understand rather than skipping it,"
                    " so that a newer writer's data is never silently lost"
                )
        except e:
            raise Error("line " + String(line_no) + ": " + String(e))

    if header_seen != 4:
        raise Error(
            "the header is incomplete: a model file needs `format`,"
            " `features`, `trees` and `losses` in that order"
        )
    if features_seen != n_features:
        raise Error(
            "declared " + String(n_features) + " features and carries "
            + String(features_seen)
        )
    if trees_seen != n_trees:
        raise Error(
            "declared " + String(n_trees) + " trees and carries "
            + String(trees_seen)
        )
    if losses_seen != n_losses:
        raise Error(
            "declared " + String(n_losses) + " losses and carries "
            + String(losses_seen)
        )
    for t in range(trees_seen):
        var want_leaves = 1 << depths[t]
        if splits_seen[t] != depths[t]:
            raise Error(
                "tree " + String(t) + " declared depth " + String(depths[t])
                + " and carries " + String(splits_seen[t]) + " splits"
            )
        if leaves_seen[t] != want_leaves:
            raise Error(
                "tree " + String(t) + " needs " + String(want_leaves)
                + " leaves and carries " + String(leaves_seen[t])
            )
        var want_weights = want_leaves if weights_declared[t] == 1 else 0
        if weights_seen[t] != want_weights:
            raise Error(
                "tree " + String(t) + " needs " + String(want_weights)
                + " leaf weights and carries " + String(weights_seen[t])
            )

    return TrainedModel(
        model^, fold_counts^, one_hot^, borders^, losses^, ctr_column_count
    )


def load_model(path: String) raises -> TrainedModel:
    var text: String
    with open(path, "r") as f:
        text = f.read()
    return load_model_text(text)
