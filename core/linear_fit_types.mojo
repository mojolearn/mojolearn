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

    def __init__(out self, x: Int, y: Int, weights: Int, coef: Int, mean: Int,
                 rows: Int, cols: Int, intercept: Bool, alpha: Float64):
        self.x = x
        self.y = y
        self.weights = weights
        self.coef = coef
        self.mean = mean
        self.rows = rows
        self.cols = cols
        self.intercept = intercept
        self.alpha = alpha


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
    if operation != 0 or na != 3 or np != 4 or nf != 1 or not aa or not pp or not ff or not result:
        raise Error("x_linear Ridge ABI: invalid request spans")
    var a = MutPointer[Int, MutUntrackedOrigin](unsafe_from_address=aa)
    if not a[0] or not a[1] or not a[2]:
        raise Error("x_linear: null float32 buffer address")
    var p = MutPointer[Int, MutUntrackedOrigin](unsafe_from_address=pp)
    if p[0] <= 0 or p[1] <= 0:
        raise Error("x_linear: positive dimensions required")


struct LinearOlsPlan(Copyable, Movable):
    var flags: Int
    var tsqr: Bool
    var numeric_mode: Int

    def __init__(out self, flags: Int, tsqr: Bool, numeric_mode: Int):
        self.flags = flags
        self.tsqr = tsqr
        self.numeric_mode = numeric_mode
