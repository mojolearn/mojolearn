# SPDX-License-Identifier: Apache-2.0
"""The x_trees glue ABI shared by its publisher and its native callers.

One `abi("C")` entry (`xtrees/glue_runs.mojo::x_trees_glue_v1`, published in
`core/trees_entry_table.mojo` family `TREES_TABLE_X_TREES`) runs every glue
operation below. A call is the descriptor shape of
`core/native_descriptor_abi.mojo`: `operation` (an `XT_*` code), a span of
buffer addresses, and the positional parameters as two parallel spans of
equal length: the int view (read where the operation reads an integer) and
the float view (read where it reads a float). The positions and address
order are those of the operation's former Python entry (`x_trees_<name>`),
documented on its run function. The caller owns every buffer and both spans
until the synchronous call returns. Nothing is retained; no Python object
crosses.
"""

comptime XT_GLUE_ABI = "mojolearn.trees.x_trees.glue.v1"

comptime XT_ACCUMULATE = 0
comptime XT_ACCUMULATE_COLS = 1
comptime XT_ACCUMULATE_ONEHOT = 2
comptime XT_ACCUMULATE_ROWS = 3
comptime XT_APPLY = 4
comptime XT_ARGMAX_ROWS = 5
comptime XT_BAG_ROWS = 6
comptime XT_BINARY_PROBA = 7
comptime XT_CLASS_COUNTS = 8
comptime XT_CLASS_PRIORS = 9
comptime XT_COMPLEMENT_PAIRS = 10
comptime XT_COUNT_EQUAL = 11
comptime XT_COUNT_ROWS = 12
comptime XT_DART_RESCALE = 13
comptime XT_DART_THRESHOLDS = 14
comptime XT_DEVICE_FOLDS = 15
comptime XT_EXACT_SUM = 16
comptime XT_FILL_CLASS_MAJOR_F64 = 17
comptime XT_GATHER_F32 = 18
comptime XT_GATHER_I32 = 19
comptime XT_INDICATOR_CODES = 20
comptime XT_IOTA_I32 = 21
comptime XT_ISOTONIC_FIT = 22
comptime XT_ISOTONIC_PREDICT_STRIDED = 23
comptime XT_LEAF_NUMBERING = 24
comptime XT_LOG64 = 25
comptime XT_MARGIN2 = 26
comptime XT_NORMALIZE_ROWS = 27
comptime XT_ONEHOT_LEAVES = 28
comptime XT_OOB_R2 = 29
comptime XT_PLATT_APPLY_STRIDED = 30
comptime XT_PLATT_FIT = 31
comptime XT_POSITIVE_CODES = 32
comptime XT_PUT_F32 = 33
comptime XT_R2_STEP = 34
comptime XT_REMAP_COLS = 35
comptime XT_ROWS_ANY_UNSET = 36
comptime XT_SAMME_STEP = 37
comptime XT_SAMPLE_INDICES = 38
comptime XT_SCALE = 39
comptime XT_SCALE_TO_F32 = 40
comptime XT_SCATTER = 41
comptime XT_SOFTMAX_ROWS = 42
comptime XT_STACK_W64 = 43
comptime XT_TRANSPOSE_F32 = 44
comptime XT_TRANSPOSE_F64 = 45
comptime XT_UNIFORM = 46
comptime XT_UNSEEN_ROWS = 47
comptime XT_WEIGHTED_MEDIAN = 48
comptime XT_WEIGHTED_SAMPLE = 49
comptime XT_ACCUMULATE_COLS_MANY = 50
comptime XT_OVR_ASSEMBLE = 51
comptime XT_CALIBRATED_PROBA = 52
comptime XT_OP_COUNT = 53


struct XtCall(Copyable, Movable):
    """Plain words of one glue call: span addresses and counts. `bad` is
    0 for a native caller; the Python unpacker sets it to a byte per
    parameter (bit 1: no integer value, bit 2: no float value) so a read of
    a missing view refuses as the former Python conversion did."""

    var addrs: Int
    var n_addrs: Int
    var ints: Int
    var floats: Int
    var n_params: Int
    var bad: Int

    def __init__(out self, addrs: Int, n_addrs: Int, ints: Int, floats: Int, n_params: Int, bad: Int):
        self.addrs = addrs
        self.n_addrs = n_addrs
        self.ints = ints
        self.floats = floats
        self.n_params = n_params
        self.bad = bad


struct XtOwned(Movable):
    """Owning storage behind an `XtCall` (the Python entries' unpack and the
    native callers' typed wrappers). Keep it alive past the call."""

    var addrs: List[Int]
    var ints: List[Int]
    var floats: List[Float64]
    var bad: List[UInt8]

    def __init__(out self, var addrs: List[Int], var ints: List[Int], var floats: List[Float64]):
        self.addrs = addrs^
        self.ints = ints^
        self.floats = floats^
        self.bad = List[UInt8]()

    def view(self) raises -> XtCall:
        if len(self.ints) != len(self.floats):
            raise Error("x_trees glue: parameter views differ in length")
        var bad = 0
        if len(self.bad):
            if len(self.bad) != len(self.ints):
                raise Error("x_trees glue: parameter flags differ in length")
            bad = Int(self.bad.unsafe_ptr())
        return XtCall(Int(self.addrs.unsafe_ptr()), len(self.addrs), Int(self.ints.unsafe_ptr()),
                      Int(self.floats.unsafe_ptr()), len(self.ints), bad)
