# SPDX-License-Identifier: Apache-2.0
"""Native linear fit requests, results and compiled primitive operation ids."""
comptime LINEAR_FIT_ABI = "mojolearn.linear.fit.v1"
comptime RIDGE_FIT_ABI = "mojolearn.x_linear.ridge.v1"
comptime LF_FLAGS = 0
comptime LF_GRAM = 1
comptime LF_RIDGE_RESIDENT = 2
comptime LF_PREP = 3
comptime LF_RIDGE = 4
comptime LF_OLS = 5
comptime LF_OLS_RESIDENT = 6
comptime LF_MODE = 7
comptime LF_VENDOR = 8
comptime LINEAR_FIT_OPERATION_COUNT = 9
comptime LF_HAS_GRAM = 1
comptime LF_HAS_RIDGE_RESIDENT = 2
comptime LF_PAIRED_MEANS = 4
comptime LF_HAS_OLS_RESIDENT = 8
#: The image's LF_GRAM fit and its prediction entry read THE FOLD X WORD (a negative X
#: address naming a slot of the x_prep store, core/fold_x_device.mojo); set only where
#: LF_GRAM always finishes a Ridge fit with alpha > 0, more than one row and column
#: (LINEAR_GRAM_SOLVE and IDN_RIDGE_RESIDENT), so no host route ever reads the word.
comptime LF_HAS_SLOT_X = 16


struct LinearFitRequest(Copyable, Movable):
    var x: Int
    var y: Int
    var weights: Int
    var coef: Int
    var mean: Int
    var rows: Int
    var cols: Int
    var intercept: Bool
    var alpha: Float64
    #: THE FOLD X WORD's prepared X (lane vdf-scope-linear4): the address of
    #: one host Int64 that receives the fold word of a prepared-X slot the fit
    #: left in the fold store (LF_PREP of a fold X keeps its result on the
    #: device); the member frees that slot after the fit. 0: none (a host X).
    var prep_slot: Int

    def __init__(out self, x: Int, y: Int, weights: Int, coef: Int, mean: Int,
                 rows: Int, cols: Int, intercept: Bool, alpha: Float64, prep_slot: Int = 0):
        self.x = x
        self.y = y
        self.weights = weights
        self.coef = coef
        self.mean = mean
        self.rows = rows
        self.cols = cols
        self.intercept = intercept
        self.alpha = alpha
        self.prep_slot = prep_slot


struct LinearFitResult(Copyable, Movable):
    var intercept: Float64
    var ymean: Float64

    def __init__(out self, intercept: Float64 = 0.0, ymean: Float64 = 0.0):
        self.intercept = intercept
        self.ymean = ymean


def linear_fit_check_spans(operation: Int, aa: Int, na: Int, pp: Int, np: Int,
                          ff: Int, nf: Int, result: Int) raises:
    var addresses = 0
    var ints = 0
    var floats = 0
    if operation == LF_FLAGS or operation == LF_MODE or operation == LF_VENDOR:
        pass
    elif operation == LF_GRAM or operation == LF_RIDGE_RESIDENT or operation == LF_OLS_RESIDENT:
        addresses = 5
        ints = 3
        floats = 1
    elif operation == LF_PREP:
        addresses = 9
        ints = 4
    elif operation == LF_RIDGE or operation == LF_OLS:
        addresses = 3
        ints = 2
        floats = 1 if operation == LF_RIDGE else 0
    else:
        raise Error("linear fit ABI: unknown operation")
    if na != addresses or np != ints or nf != floats or not result:
        raise Error("linear fit ABI: invalid request spans")
    if (na and not aa) or (np and not pp) or (nf and not ff):
        raise Error("linear fit ABI: null request span")
    var a = MutPointer[Int, MutUntrackedOrigin](unsafe_from_address=aa)
    for i in range(na):  # driver-loop: over fixed primitive address slots
        if not a[i] and not (operation == LF_PREP and i == 2):
            raise Error("linear fit ABI: null buffer address")


def ridge_fit_check_spans(operation: Int, aa: Int, na: Int, pp: Int, np: Int,
                         ff: Int, nf: Int, result: Int) raises:
    # p: [rows, cols, fit_intercept, sample_weight] and, optionally (lane
    # w3f3-k, the multi-target fit), [4] the target count T >= 1: y is then
    # rows x T row-major (then the rows' weights), the result T*(cols + 1) + 4
    # words (`ridge_fit_result_words`).
    if operation != 0 or na != 3 or (np != 4 and np != 5) or nf != 1 or not aa or not pp or not ff or not result:
        raise Error("x_linear Ridge ABI: invalid request spans")
    var a = MutPointer[Int, MutUntrackedOrigin](unsafe_from_address=aa)
    if not a[0] or not a[1] or not a[2]:
        raise Error("x_linear: null float32 buffer address")
    var p = MutPointer[Int, MutUntrackedOrigin](unsafe_from_address=pp)
    if p[0] <= 0 or p[1] <= 0 or (np == 5 and p[4] < 1):
        raise Error("x_linear: positive dimensions required")


def ridge_fit_targets(np: Int, pp: Int) -> Int:
    """The target count of a ridge fit request (1 without the fifth word)."""
    if np < 5:
        return 1
    return MutPointer[Int, MutUntrackedOrigin](unsafe_from_address=pp)[4]


def ridge_fit_result_words(targets: Int, cols: Int) -> Int:
    """The result words of a ridge fit request: coef T*cols | intercept T |
    alpha | best score | one leave-one-out error | status (x_linear/ridge.mojo
    `ridge_fit`; `cols + 5` for one target)."""
    return targets * (cols + 1) + 4


def ridge_fit_work_words(targets: Int, cols: Int, rows: Int) -> Int:
    """The float work words of a ridge fit request (x_linear/ridge.mojo
    `ridge_fit`'s fw layout, with one spare cols x cols block as before;
    `3 cols^2 + 4 cols + 1 + 2 rows` for one target)."""
    return max(3 * cols * cols + 3 * cols + targets + cols * targets + 2 * rows, 1)


struct LinearOlsPlan(Copyable, Movable):
    var flags: Int
    var tsqr: Bool
    var numeric_mode: Int
    # The decomposition entry's one-entry centered TSQR (XD_FLAGS bit 1),
    # probed ONCE by the door for an unweighted fit with an intercept and
    # read here (lane ols-regression: linear_ols_run probed it a second time).
    var one_entry: Bool

    def __init__(out self, flags: Int, tsqr: Bool, numeric_mode: Int):
        self.flags = flags
        self.tsqr = tsqr
        self.numeric_mode = numeric_mode
        self.one_entry = False
