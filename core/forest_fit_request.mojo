# SPDX-License-Identifier: Apache-2.0
"""The native request of a forest fit and the forest binding ABI (lane
calltable-forest, 2026-11-11).

`ForestFitArgs` is what the forest fit Python entries' arguments held, as
plain Mojo values: the X and y addresses, the `params` list as an int view
and a float view of equal length (with a flag byte per position for a value
that has no integer or no float reading, so a read of a missing view refuses
as the former `Int(py=)` / `Float64(py=)` did), the split criterion and the
trailing words (row weights, global tree-ID start, OOB / score / importance
outputs, the largest weight, a DART session and class).

THE ABI (`FOREST_ENTRY_ABI`, one `abi("C")` entry per forest image,
published in `core/forest_entry_table.mojo`; descriptor shape of
`core/native_descriptor_abi.mojo`):

  XF_CAPS             returns the image's capability bits (XF_CAP_*).
  XF_NUMERIC_MODE     returns the compiled tier (0 fast, 1 identical).
  XF_DT_DEFAULT_BINS  returns the bin count a DecisionTreeRegressor left at
                      n_bins=None fits with (128 in an image without one).
  XF_FIT              addrs XF_FIT_WORDS (below); ints/floats = the params
  XF_FIT_DART         views. Returns an export handle owned by the image
                      until XF_EXPORT_RELEASE. XF_FIT_DART is the rf images'
                      DART member fit on a DART session (x = bag rows, y =
                      sampled columns, `session`, `cls`).
  XF_EXPORT_SIZES     addrs [handle, out (Int, 4 + XF_META_MAX)]: writes
                      [trees, nodes, outputs, n_meta, meta...].
  XF_EXPORT_INTO      addrs [handle, offsets, colid, quesval, left, leaves];
                      ints [trees, nodes, outputs]: the model into the
                      caller's five arrays (int32/float32).
  XF_EXPORT_VERIFY    addrs [handle, offsets, colid, quesval, left, leaves]:
                      the diagnostic lists of the same fit, cast to the
                      arrays' dtypes, against the arrays' bytes; returns 0
                      when equal, k + 1 for the first differing array k, or
                      -1 when the diagnostic metadata differs.
  XF_EXPORT_RELEASE   addrs [handle].
  XF_PREDICT          addrs [offsets, colid, quesval, left, leaves, x, out];
                      ints [rows, cols, trees, outputs, regression]: the
                      sequential vote (`rf_predict_proba` / `rf_predict_reg`
                      / `et_predict`); returns rows written.
  XF_RESIDENT_PREPARE addrs [offsets, colid, quesval, left, leaves]; ints
                      [features, trees, outputs, ordered (-1 default)].
  XF_RESIDENT_PREDICT addrs [handle, x, out]; ints [rows, cols, outputs].
  XF_RESIDENT_LABELS  as XF_RESIDENT_PREDICT, int32 class codes out.
  XF_RESIDENT_RELEASE ints [handle] (the resident table's native release).
  XF_STRICT_PREPARE   as XF_RESIDENT_PREPARE without `ordered` (host images).
  XF_STRICT_PREDICT   as XF_RESIDENT_PREDICT.
  XF_STRICT_RELEASE   ints [handle].

The caller owns every buffer until the synchronous call returns.
"""
from std.memory import bitcast

comptime FOREST_ENTRY_ABI = "mojolearn.trees.forest.v1"

comptime XF_CAPS = 0
comptime XF_NUMERIC_MODE = 1
comptime XF_DT_DEFAULT_BINS = 2
comptime XF_FIT = 3
comptime XF_FIT_DART = 4
comptime XF_EXPORT_SIZES = 5
comptime XF_EXPORT_INTO = 6
comptime XF_EXPORT_VERIFY = 7
comptime XF_EXPORT_RELEASE = 8
comptime XF_PREDICT = 9
comptime XF_RESIDENT_PREPARE = 10
comptime XF_RESIDENT_PREDICT = 11
comptime XF_RESIDENT_LABELS = 12
comptime XF_RESIDENT_RELEASE = 13
comptime XF_STRICT_PREPARE = 14
comptime XF_STRICT_PREDICT = 15
comptime XF_STRICT_RELEASE = 16
comptime XF_OP_COUNT = 17

#: capability bits (XF_CAPS)
comptime XF_CAP_DEVICE_SCAN = 1      # the fits refuse a non-finite X cell on the device
comptime XF_CAP_ORDERED_RESIDENT = 2  # the resident snapshot takes the `ordered` word
comptime XF_CAP_FUSED_LABELS = 4      # IDENTICAL resident fused labels (T38)
comptime XF_CAP_RESIDENT_LABELS = 8   # XF_RESIDENT_LABELS is served
comptime XF_CAP_STRICT = 16           # the host strict snapshot (XF_STRICT_*)
comptime XF_CAP_DART = 32             # XF_FIT_DART is served

#: XF_FIT request words: addrs[k]
comptime XF_W_X = 0
comptime XF_W_Y = 1
comptime XF_W_FLAGS = 2
comptime XF_W_SHAPE = 3      # bit 0 classifier, bit 1 row-major X
comptime XF_W_CRITERION = 4  # -1: the ET fits, whose criterion is params[21]
comptime XF_W_WEIGHTS = 5
comptime XF_W_TREE_START = 6
comptime XF_W_OOB = 7
comptime XF_W_SCORE = 8
comptime XF_W_IMPORTANCE = 9
comptime XF_W_WEIGHT_MAX = 10  # the Float64 bits of the largest weight
comptime XF_W_SESSION = 11
comptime XF_W_CLS = 12
comptime XF_FIT_WORDS = 13

#: the params slot holding the fit's tree count in both layouts (the RF
#: list of `_RandomForestBase._fit_params` and the ET list of
#: `extratrees._fit_params`): a tree-ID shard's request overwrites it
comptime XF_P_TREES = 3

comptime XF_SHAPE_CLASSIFIER = 1
comptime XF_SHAPE_ROWMAJOR = 2

#: the most export metadata words an image reports (ET: six)
comptime XF_META_MAX = 8

#: flag bits of `ForestFitArgs.bad`
comptime FOREST_ARG_NO_INT = 1
comptime FOREST_ARG_NO_FLOAT = 2


struct ForestFitArgs(Movable, Sized):
    """One forest fit's request as plain values (see the module docstring).
    `len(args)` and `args.i(k)` / `args.f(k)` read the params list."""

    var x: Int
    var y: Int
    var ints: List[Int]
    var floats: List[Float64]
    var bad: List[UInt8]
    var criterion: Int
    var weights: Int
    var tree_start: Int
    var oob: Int
    var score: Int
    var importance: Int
    var weight_max: Float64
    var session: Int
    var cls: Int

    def __init__(out self, x: Int, y: Int, var ints: List[Int], var floats: List[Float64], var bad: List[UInt8],
                 criterion: Int) raises:
        if len(ints) != len(floats) or len(bad) != len(ints):
            raise Error("forest fit: parameter views differ in length")
        self.x = x
        self.y = y
        self.ints = ints^
        self.floats = floats^
        self.bad = bad^
        self.criterion = criterion
        self.weights = 0
        self.tree_start = 0
        self.oob = 0
        self.score = 0
        self.importance = 0
        self.weight_max = 0.0
        self.session = 0
        self.cls = 0

    def __len__(self) -> Int:
        return len(self.ints)

    def i(self, k: Int) raises -> Int:
        """Slot k as an integer (the former `Int(py=params[k])`)."""
        if k < 0 or k >= len(self.ints):
            raise Error("forest fit: params index " + String(k) + " out of range")
        if self.bad[k] & UInt8(FOREST_ARG_NO_INT):
            raise Error("forest fit: params[" + String(k) + "] must be an integer")
        return self.ints[k]

    def f(self, k: Int) raises -> Float64:
        """Slot k as a float (the former `Float64(py=params[k])`)."""
        if k < 0 or k >= len(self.floats):
            raise Error("forest fit: params index " + String(k) + " out of range")
        if self.bad[k] & UInt8(FOREST_ARG_NO_FLOAT):
            raise Error("forest fit: params[" + String(k) + "] must be a number")
        return self.floats[k]


def forest_set_trees(mut args: ForestFitArgs, trees: Int) raises:
    """A tree-ID shard's tree count in params slot XF_P_TREES (both views,
    an integer reading, so its flag clears)."""
    if len(args.ints) <= XF_P_TREES:
        raise Error("forest fit: the request has no tree-count slot")
    args.ints[XF_P_TREES] = trees
    args.floats[XF_P_TREES] = Float64(trees)
    args.bad[XF_P_TREES] = UInt8(0)


def forest_f64_word(value: Float64) -> Int:
    """A Float64 carried in an address word (XF_W_WEIGHT_MAX)."""
    return Int(bitcast[DType.int64](value))


def forest_word_f64(word: Int) -> Float64:
    return bitcast[DType.float64](Int64(word))


def forest_fit_words(args: ForestFitArgs, shape: Int) -> List[Int]:
    """The XF_FIT address words of a request (the flags at its `bad` list,
    which the caller keeps alive through the call)."""
    var words = List[Int](length=XF_FIT_WORDS, fill=0)
    words[XF_W_X] = args.x
    words[XF_W_Y] = args.y
    words[XF_W_FLAGS] = Int(args.bad.unsafe_ptr()) if len(args.bad) else 0
    words[XF_W_SHAPE] = shape
    words[XF_W_CRITERION] = args.criterion
    words[XF_W_WEIGHTS] = args.weights
    words[XF_W_TREE_START] = args.tree_start
    words[XF_W_OOB] = args.oob
    words[XF_W_SCORE] = args.score
    words[XF_W_IMPORTANCE] = args.importance
    words[XF_W_WEIGHT_MAX] = forest_f64_word(args.weight_max)
    words[XF_W_SESSION] = args.session
    words[XF_W_CLS] = args.cls
    return words^


def forest_fit_args_at(aa: Int, na: Int, pp: Int, np: Int, ff: Int) raises -> ForestFitArgs:
    """The request spans as `ForestFitArgs` (the image side of XF_FIT)."""
    if na != XF_FIT_WORDS:
        raise Error("forest fit: the request needs " + String(XF_FIT_WORDS) + " words")
    var a = MutPointer[Int, MutUntrackedOrigin](unsafe_from_address=aa)
    var ints = List[Int](capacity=np)
    var floats = List[Float64](capacity=np)
    var bad = List[UInt8](length=np, fill=0)
    var ip = MutPointer[Int, MutUntrackedOrigin](unsafe_from_address=pp)
    var fp = MutPointer[Float64, MutUntrackedOrigin](unsafe_from_address=ff)
    var flags = a[XF_W_FLAGS]
    for k in range(np):  # small-loop(the fit's parameter slots): request words, not row data
        ints.append(ip[k])
        floats.append(fp[k])
        if flags:
            bad[k] = MutPointer[UInt8, MutUntrackedOrigin](unsafe_from_address=flags)[k]
    var args = ForestFitArgs(a[XF_W_X], a[XF_W_Y], ints^, floats^, bad^, a[XF_W_CRITERION])
    args.weights = a[XF_W_WEIGHTS]
    args.tree_start = a[XF_W_TREE_START]
    args.oob = a[XF_W_OOB]
    args.score = a[XF_W_SCORE]
    args.importance = a[XF_W_IMPORTANCE]
    args.weight_max = forest_word_f64(a[XF_W_WEIGHT_MAX])
    args.session = a[XF_W_SESSION]
    args.cls = a[XF_W_CLS]
    return args^


def forest_export_sizes_into(out_addr: Int, trees: Int, nodes: Int, outputs: Int, meta: List[Int64]) raises:
    """XF_EXPORT_SIZES' words: [trees, nodes, outputs, n_meta, meta...]."""
    if len(meta) > XF_META_MAX:
        raise Error("forest export: more metadata words than the ABI carries")
    var dst = MutPointer[Int, MutUntrackedOrigin](unsafe_from_address=out_addr)
    dst[0] = trees
    dst[1] = nodes
    dst[2] = outputs
    dst[3] = len(meta)
    for k in range(len(meta)):  # small-loop(meta: at most XF_META_MAX export metadata words)
        dst[4 + k] = Int(meta[k])


def _same_i32(values: List[Int64], address: Int) -> Bool:
    var p = MutPointer[Int32, MutUntrackedOrigin](unsafe_from_address=address)
    for k in range(len(values)):  # small-loop(diagnostic compare): MOJOLEARN_FOREST_EXPORT=verify only, never the default route
        if Int32(values[k]) != p[k]:
            return False
    return True


def _same_f32(values: List[Float64], address: Int) -> Bool:
    var p = MutPointer[Float32, MutUntrackedOrigin](unsafe_from_address=address)
    for k in range(len(values)):  # small-loop(diagnostic compare): MOJOLEARN_FOREST_EXPORT=verify only, never the default route
        if bitcast[DType.uint32](Float32(values[k])) != bitcast[DType.uint32](p[k]):
            return False
    return True


def forest_export_verify_lists(offsets: List[Int64], colid: List[Int64], quesval: List[Float64],
                               left: List[Int64], leaves: List[Float64], dst: List[Int]) -> Int:
    """XF_EXPORT_VERIFY's comparison (the former door's
    `Array.from_list(old[i], dtype)` bytes against the exported arrays):
    0 equal, else k + 1 for the first differing array k. `dst` holds the
    five array addresses."""
    if not _same_i32(offsets, dst[0]):
        return 1
    if not _same_i32(colid, dst[1]):
        return 2
    if not _same_f32(quesval, dst[2]):
        return 3
    if not _same_i32(left, dst[3]):
        return 4
    if not _same_f32(leaves, dst[4]):
        return 5
    return 0
