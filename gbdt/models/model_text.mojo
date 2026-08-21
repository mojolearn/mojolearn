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
    ctr_columns <n>                        (only when n is non-zero)
    feature <f> folds <k> one_hot <0|1> type <float|cat|ctr> borders <n> <tok>...
    ctr_table <column> source <f> type <name> prior_num <tok> prior_denom <tok>
              shift <tok> scale <tok> denom <int> classes <int>
              target_border <int> entries <n>
    ctr_entry <column> <category> <count> [<count> ...]
    tree <t> depth <d> dim <k> weights <0|1>
    split <t> <level> <feature_id> <bin_idx>
    split <t> <level> <feature_id> <bin_idx> split_type take_bin
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

========================== THE CTR SEAM, BUILT =========================
A trained CATEGORICAL model scores raw data through these records. Written
as a plan in this block; this is what landed, and the plan is replaced
rather than annotated.

* The `feature` record's `type` token is `float` for a numeric column,
  `cat` for a one-hot categorical one (its values ARE its bins and its
  splits are equality tests) and `ctr` for a CTR-valued column. A reader
  REFUSES an unknown type by name, so an old reader never half-loads a
  categorical model. `type cat` and the `one_hot` flag must agree, and the
  loader raises if they do not: two spellings of one fact are two chances
  to be wrong.
* The CTR tables go in their own records AFTER the `feature` block and
  before the first `tree`. `ctr_table` mirrors their `TCtrFeature`'s
  `prior_numerator` / `prior_denomerator` (`json_model_helpers.cpp:104-114`)
  plus the `shift`, `scale` and `target_border` of their `TModelCtr`
  (`online_ctr.h:260-266`) and the `CounterDenominator` and
  `TargetClassesCount` of their `TCtrValueTable` (`ctr_value_table.h:104-105`);
  `ctr_entry` is their `ctr_data.hash_map`
  (`json_model_helpers.cpp:440-482`), which for a `FeatureFreq` or
  `Counter` table stores ONE INTEGER per category and not a value. The
  value is formed at apply time by `TModelCtr::Calc`; see
  `gbdt/models/ctr_value_table.mojo` for why the counts and not the values
  are what travels.
* **`ctr_entry` CARRIES THE TARGET-CLASS AXIS, ONE RECORD PER CATEGORY.**
  A `Borders` table's blob is `int[uniqueCategories * TargetClassesCount]`
  (`online_ctr.cpp:910`), so its record carries `classes` counts and
  `entries` stays the CATEGORY count -- the number of `ctr_entry` records,
  not the blob length. `classes` is 0 on a `FeatureFreq` table, where their
  own `TargetClassesCount` default stands, and one count per record. A
  reader that dropped the axis would read a two-class histogram as twice as
  many categories, which is why `entries * max(classes, 1)` is checked
  against the blob it assembled rather than assumed.
* **`ctr_borders` was in the plan and is NOT written**, because in this
  format it would be the same numbers twice. Their `TCtrFeature::Borders`
  exists because a CTR feature is not a `TFloatFeature` in their model and
  carries its own binarization; here a CTR column IS a column of the
  feature table, so its `feature` record already carries exactly those
  borders -- `train()` gives it its own grid
  (`batch_binarized_ctr_calcer.cpp:57-63`, MinEntropy-15 for FeatureFreq)
  and stores it there. A second copy would be a second thing to corrupt.
* A model with no categorical features writes NONE of these records and no
  `split_type` token, so every file this format wrote before they landed
  is byte-identical to what it writes now, and the `format` version stays
  1 for float-only models. MEASURED, not asserted: an 8-tree float-only
  model's 8022 bytes hash the same before and after this change.
* `split` grows a TRAILING `split_type take_bin` pair, and only on a
  one-hot split. Their `TBinarySplit` carries `EBinSplitType` as a member
  (`cuda/data/feature.h:38`) and their `ToSplit` sets it from
  `manager.IsCat` (`cuda/methods/helpers.cpp:164-170`); it is trailing and
  conditional here for exactly the byte-identity rule above. The loader
  cross-checks it against the feature's own type, which is their
  `CB_ENSURE(dataSet.IsOneHot(split.FeatureId))`
  (`add_oblivious_tree_model_doc_parallel.cpp:43`).
* `ctr_columns` is the SAFETY field and it predates the tables: a model
  that carries CTR columns and no tables must load as one that still
  REFUSES to score. The refusal lifts because the tables are PRESENT and
  cover every declared CTR column, never because the count went missing.
======================================================================
"""

from std.memory import bitcast

from gbdt.models.ctr_value_table import (
    TCtrValueTable,
    ctr_type_from_name,
)
from gbdt.ctrs.ctr import ctr_type_name
from gbdt.models.oblivious_model import (
    BIN_SPLIT_TAKE_BIN,
    BIN_SPLIT_TAKE_GREATER,
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

    # which columns a CTR table stands behind, so the `type` token can say
    # so. A column is `ctr` iff a table names it; the count in the header is
    # a SEPARATE, older field and the two are cross-checked on load.
    var table_of_column = List[Int]()
    for _ in range(n_features):
        table_of_column.append(-1)
    for i in range(len(tm.ctr_tables)):
        var c = tm.ctr_tables[i].column
        if c < 0 or c >= n_features:
            raise Error(
                "a CTR table names column " + String(c) + " of "
                + String(n_features)
            )
        if table_of_column[c] != -1:
            raise Error("two CTR tables name column " + String(c))
        table_of_column[c] = i

    for f in range(n_features):
        var is_one_hot = len(tm.one_hot) != 0 and tm.one_hot[f]
        var flag = 1 if is_one_hot else 0
        var kind = String("float")
        if table_of_column[f] >= 0:
            if is_one_hot:
                raise Error(
                    "column " + String(f) + " is flagged one-hot AND carries"
                    " a CTR table; their dispatch gives a one-hot feature no"
                    " CTRs at all (binarizations_manager.cpp:106-109)"
                )
            kind = String("ctr")
        elif is_one_hot:
            kind = String("cat")
        var line = (
            String("feature ") + String(f) + " folds "
            + String(tm.fold_counts[f]) + " one_hot " + String(flag)
            + " type " + kind + " borders " + String(len(tm.borders[f]))
        )
        for b in range(len(tm.borders[f])):
            line += " " + f32_token(tm.borders[f][b])
        out += line + "\n"

    # the tables, in COLUMN order, each followed by its own entries. Their
    # `ctr_data` is a map keyed by the ctr's identity; ours is keyed by the
    # column, because a column is what this model applies.
    for f in range(n_features):
        var ti = table_of_column[f]
        if ti < 0:
            continue
        ref tab = tm.ctr_tables[ti]
        # the TARGET-CLASS AXIS of their blob. 0 on a FeatureFreq table,
        # where their own `TargetClassesCount` default stands; on a Borders
        # table it is the width of every `ctr_entry` record.
        var classes = tab.target_classes_count
        var per_entry = classes if classes > 0 else 1
        if classes < 0:
            raise Error(
                "a CTR table for column " + String(f)
                + " declares a negative target class count"
            )
        if len(tab.counts) % per_entry != 0:
            raise Error(
                "the CTR table for column " + String(f) + " carries "
                + String(len(tab.counts)) + " counts, which is not a whole"
                " number of " + String(per_entry) + "-class histograms"
            )
        var entries = len(tab.counts) // per_entry
        out += (
            String("ctr_table ") + String(f)
            + " source " + String(tab.source_feature)
            + " type " + ctr_type_name(tab.ctr_type)
            + " prior_num " + f32_token(tab.prior_num)
            + " prior_denom " + f32_token(tab.prior_denom)
            + " shift " + f32_token(tab.shift)
            + " scale " + f32_token(tab.scale)
            + " denom " + String(tab.counter_denominator)
            + " classes " + String(classes)
            + " target_border " + String(tab.target_border_idx)
            + " entries " + String(entries) + "\n"
        )
        for c in range(entries):
            var line = (
                String("ctr_entry ") + String(f) + " " + String(c)
            )
            for k in range(per_entry):
                line += " " + String(tab.counts[c * per_entry + k])
            out += line + "\n"

    for t in range(tm.model.size()):
        ref weak = tm.model.weak_models[t]
        var depth = weak.structure.get_depth()
        var n_leaves = 1 << depth
        # A LEAF CARRIES `dim` VALUES, NOT ONE. `dim` is their
        # `OutputDim()` (`oblivious_model.h:130-133`) -- 1 for every
        # single-dimensional loss and `numClasses - 1` for MultiClass --
        # and the values are BIN-MAJOR, `[leaf * dim + d]`, which is the
        # layout `MakeEstimationResult` produces and both apply kernels
        # read. The record index below is that flat index, so a
        # one-dimensional model's bytes do not move.
        if weak.dim < 1:
            raise Error(
                "tree " + String(t) + " has dim " + String(weak.dim)
            )
        var n_values = n_leaves * weak.dim
        var has_weights = 1 if len(weak.leaf_weights) != 0 else 0
        if len(weak.leaf_values) != n_values:
            raise Error(
                "tree " + String(t) + " has depth " + String(depth)
                + ", dim " + String(weak.dim) + " and "
                + String(len(weak.leaf_values))
                + " leaf values, not " + String(n_values)
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
            var line = (
                String("split ") + String(t) + " " + String(level) + " "
                + String(Int(weak.structure.splits[level].feature_id)) + " "
                + String(Int(weak.structure.splits[level].bin_idx))
            )
            # TRAILING AND CONDITIONAL, so a float-only model's bytes do not
            # move. TakeGreater is the absent case because it is every split
            # this format could carry before one-hot splits existed.
            var st = Int(weak.structure.splits[level].split_type)
            if st == BIN_SPLIT_TAKE_BIN:
                line += " split_type take_bin"
            elif st != BIN_SPLIT_TAKE_GREATER:
                raise Error(
                    "tree " + String(t) + " level " + String(level)
                    + " has split type " + String(st)
                    + ", which is neither TakeBin nor TakeGreater"
                )
            out += line + "\n"
        for i in range(n_values):
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

    # the CTR half
    var feature_kinds = List[String]()
    var ctr_tables = List[TCtrValueTable]()
    var ctr_columns = List[Int]()
    var ctr_sources = List[Int]()
    var ctr_types = List[Int]()
    var ctr_prior_num = List[Float32]()
    var ctr_prior_denom = List[Float32]()
    var ctr_shift = List[Float32]()
    var ctr_scale = List[Float32]()
    var ctr_denom = List[Int]()
    var ctr_classes = List[Int]()
    var ctr_border_idx = List[Int]()
    var ctr_declared = List[Int]()
    var ctr_entries_seen = List[Int]()
    var ctr_counts = List[List[Int]]()

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
                if (
                    t[7] != String("float")
                    and t[7] != String("cat")
                    and t[7] != String("ctr")
                ):
                    raise Error(
                        "feature " + String(t[1]) + " has type '" + t[7]
                        + "'. This reader knows float, cat and ctr, and"
                        " REFUSES a type it does not understand rather than"
                        " guessing at the predicate it implies"
                    )
                var flag_here = Int(t[5]) != 0
                if (t[7] == String("cat")) != flag_here:
                    raise Error(
                        "feature " + String(t[1]) + " says type '" + t[7]
                        + "' and one_hot " + String(t[5])
                        + "; a one-hot column is exactly a `cat` column and"
                        " the two spellings must agree"
                    )
                _expect(t, 8, String("borders"), String("feature"))
                fold_counts.append(Int(t[3]))
                if n_flags != 0:
                    one_hot.append(Int(t[5]) != 0)
                feature_kinds.append(t[7])
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
            elif kind == String("ctr_table"):
                if features_seen != n_features:
                    raise Error(
                        "a `ctr_table` before the feature block ended: "
                        + String(features_seen) + " of "
                        + String(n_features) + " features so far"
                    )
                if trees_seen != 0:
                    raise Error("a `ctr_table` after the first `tree`")
                _expect(t, 2, String("source"), String("ctr_table"))
                _expect(t, 4, String("type"), String("ctr_table"))
                _expect(t, 6, String("prior_num"), String("ctr_table"))
                _expect(t, 8, String("prior_denom"), String("ctr_table"))
                _expect(t, 10, String("shift"), String("ctr_table"))
                _expect(t, 12, String("scale"), String("ctr_table"))
                _expect(t, 14, String("denom"), String("ctr_table"))
                _expect(t, 16, String("classes"), String("ctr_table"))
                _expect(t, 18, String("target_border"), String("ctr_table"))
                _expect(t, 20, String("entries"), String("ctr_table"))
                if len(t) != 22:
                    raise Error(
                        "a `ctr_table` record has 22 fields, this one has "
                        + String(len(t))
                    )
                var col = Int(t[1])
                if col < 0 or col >= n_features:
                    raise Error(
                        "ctr_table names column " + String(col) + " of "
                        + String(n_features)
                    )
                for k in range(len(ctr_columns)):
                    if ctr_columns[k] == col:
                        raise Error(
                            "two `ctr_table` records for column "
                            + String(col)
                        )
                if len(ctr_columns) != 0 and col <= ctr_columns[
                    len(ctr_columns) - 1
                ]:
                    raise Error(
                        "ctr_table records must arrive in ascending column"
                        " order: " + String(col) + " after "
                        + String(ctr_columns[len(ctr_columns) - 1])
                    )
                ctr_columns.append(col)
                ctr_sources.append(Int(t[3]))
                ctr_types.append(ctr_type_from_name(t[5]))
                ctr_prior_num.append(parse_f32(t[7]))
                ctr_prior_denom.append(parse_f32(t[9]))
                ctr_shift.append(parse_f32(t[11]))
                ctr_scale.append(parse_f32(t[13]))
                ctr_denom.append(Int(t[15]))
                var n_classes = Int(t[17])
                if n_classes < 0 or n_classes == 1:
                    raise Error(
                        "ctr_table declares " + String(n_classes)
                        + " target classes; it is 0 on a FeatureFreq table"
                        " and at least 2 on a Borders one"
                        " (libs/model/target_classifier.h:32-34)"
                    )
                ctr_classes.append(n_classes)
                var border_idx = Int(t[19])
                if border_idx < 0:
                    raise Error(
                        "ctr_table declares a negative target_border"
                    )
                ctr_border_idx.append(border_idx)
                var n_entries = Int(t[21])
                if n_entries < 0:
                    raise Error("ctr_table declares a negative entry count")
                ctr_declared.append(n_entries)
                ctr_counts.append(List[Int]())
                ctr_entries_seen.append(0)
            elif kind == String("ctr_entry"):
                if len(ctr_columns) == 0:
                    raise Error("a `ctr_entry` before any `ctr_table`")
                var last = len(ctr_columns) - 1
                if Int(t[1]) != ctr_columns[last]:
                    raise Error(
                        "a `ctr_entry` for column " + String(t[1])
                        + " under the `ctr_table` for column "
                        + String(ctr_columns[last])
                    )
                var per_entry = ctr_classes[last] if (
                    ctr_classes[last] > 0
                ) else 1
                if len(t) != 3 + per_entry:
                    raise Error(
                        "a `ctr_entry` record under a table declaring "
                        + String(ctr_classes[last]) + " target classes has "
                        + String(3 + per_entry) + " fields, this one has "
                        + String(len(t))
                    )
                if Int(t[2]) != ctr_entries_seen[last]:
                    raise Error(
                        "column " + String(ctr_columns[last])
                        + " categories must arrive in order: expected "
                        + String(ctr_entries_seen[last]) + ", got "
                        + String(t[2])
                    )
                for k in range(per_entry):
                    ctr_counts[last].append(Int(t[3 + k]))
                ctr_entries_seen[last] += 1
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
                var st = BIN_SPLIT_TAKE_GREATER
                if len(t) == 7:
                    _expect(t, 5, String("split_type"), String("split"))
                    if t[6] != String("take_bin"):
                        raise Error(
                            "a `split` record's split_type is `take_bin` or"
                            " nothing at all (TakeGreater); got '" + t[6]
                            + "'"
                        )
                    st = BIN_SPLIT_TAKE_BIN
                elif len(t) != 5:
                    raise Error(
                        "a `split` record has 5 fields, or 7 with"
                        " `split_type take_bin`; this one has "
                        + String(len(t))
                    )
                model.weak_models[ti].structure.splits.append(
                    TBinarySplit(
                        Int32(Int(t[3])), Int32(Int(t[4])), Int32(st)
                    )
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
        # `dim` values per leaf; see the writer's note on the layout
        var want_leaves = (1 << depths[t]) * model.weak_models[t].dim
        if model.weak_models[t].dim < 1:
            raise Error(
                "tree " + String(t) + " declared dim "
                + String(model.weak_models[t].dim)
            )
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
        # LEAF WEIGHTS ARE PER LEAF, NOT PER VALUE: their `LeafWeights` is
        # `BinCount()` long whatever `Dim` is
        # (`doc_parallel_leaves_estimator.cpp:21-23` CB_ENSUREs exactly
        # that), because a leaf has one weight and `dim` values.
        var want_weights = (
            (1 << depths[t]) if weights_declared[t] == 1 else 0
        )
        if weights_seen[t] != want_weights:
            raise Error(
                "tree " + String(t) + " needs " + String(want_weights)
                + " leaf weights and carries " + String(weights_seen[t])
            )

    # ---- the CTR half, checked as a whole because it is cross-referenced
    for k in range(len(ctr_columns)):
        if ctr_entries_seen[k] != ctr_declared[k]:
            raise Error(
                "ctr_table for column " + String(ctr_columns[k])
                + " declares " + String(ctr_declared[k])
                + " entries and carries " + String(ctr_entries_seen[k])
            )
        # the blob length is entries x the TARGET-CLASS AXIS, checked
        # rather than assumed: a reader that dropped the axis would see a
        # two-class histogram as twice as many categories and every value
        # it produced afterwards would be a different category's
        var per_entry = ctr_classes[k] if ctr_classes[k] > 0 else 1
        if len(ctr_counts[k]) != ctr_declared[k] * per_entry:
            raise Error(
                "ctr_table for column " + String(ctr_columns[k])
                + " declares " + String(ctr_declared[k]) + " entries x "
                + String(per_entry) + " target classes = "
                + String(ctr_declared[k] * per_entry)
                + " counts and carries " + String(len(ctr_counts[k]))
            )
        if feature_kinds[ctr_columns[k]] != String("ctr"):
            raise Error(
                "column " + String(ctr_columns[k]) + " carries a CTR table"
                " and its `feature` record says type '"
                + feature_kinds[ctr_columns[k]] + "'"
            )
        ctr_tables.append(
            TCtrValueTable(
                ctr_columns[k],
                ctr_sources[k],
                ctr_types[k],
                ctr_prior_num[k],
                ctr_prior_denom[k],
                ctr_shift[k],
                ctr_scale[k],
                ctr_denom[k],
                ctr_classes[k],
                ctr_border_idx[k],
                ctr_counts[k].copy(),
            )
        )
    for f in range(features_seen):
        if feature_kinds[f] == String("ctr"):
            var found = False
            for k in range(len(ctr_columns)):
                if ctr_columns[k] == f:
                    found = True
            if not found:
                raise Error(
                    "column " + String(f) + " says type ctr and no"
                    " `ctr_table` record names it. A CTR column without its"
                    " table cannot score a raw row"
                )
    # `ctr_columns` in the header is the SAFETY count and it is older than
    # the tables. It may exceed the table count -- that is a model which
    # still refuses to score, and refusing is the point -- but it must never
    # be SHORT of it, which would be a file claiming fewer CTR columns than
    # it carries.
    if len(ctr_tables) > ctr_column_count:
        raise Error(
            "the header declares " + String(ctr_column_count)
            + " CTR columns and the file carries " + String(len(ctr_tables))
            + " CTR tables"
        )

    # their `CB_ENSURE(dataSet.IsOneHot(split.FeatureId))` for a TakeBin
    # split and `CB_ENSURE(!dataSet.IsOneHot(...))` for the other arm
    # (`add_oblivious_tree_model_doc_parallel.cpp:43-47`), run against the
    # feature table this file carries instead of against a live layout.
    for t in range(trees_seen):
        ref w = model.weak_models[t]
        for lvl in range(len(w.structure.splits)):
            var fid = Int(w.structure.splits[lvl].feature_id)
            if fid < 0 or fid >= features_seen:
                raise Error(
                    "tree " + String(t) + " level " + String(lvl)
                    + " splits on feature " + String(fid) + " of "
                    + String(features_seen)
                )
            var is_bin = (
                Int(w.structure.splits[lvl].split_type) == BIN_SPLIT_TAKE_BIN
            )
            if is_bin != (feature_kinds[fid] == String("cat")):
                raise Error(
                    "tree " + String(t) + " level " + String(lvl)
                    + " is a "
                    + String("TakeBin" if is_bin else "TakeGreater")
                    + " split on feature " + String(fid)
                    + ", whose type is '" + feature_kinds[fid] + "'"
                )

    return TrainedModel(
        model^, fold_counts^, one_hot^, borders^, losses^, ctr_column_count,
        ctr_tables^,
    )


def load_model(path: String) raises -> TrainedModel:
    var text: String
    with open(path, "r") as f:
        text = f.read()
    return load_model_text(text)
