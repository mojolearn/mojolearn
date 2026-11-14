# SPDX-License-Identifier: Apache-2.0
"""Typed requests crossing the linear estimator's native execution boundary.

The caller owns the admitted buffers until RUN returns. No request owns a
Python object or can inspect its owner. UNPACK and PACK retain those owners.
"""
comptime LINEAR_PREDICT_ABI = "mojolearn.linear.predict.v1"


struct LinearPredictRequest(Copyable, Movable):
    var x: Int
    var coef: Int
    var dst: Int
    var rows: Int
    var cols: Int
    var intercept: Float64
    var wide: Bool
    # The door deferred X's finite test to the entry (a borrowed X it did
    # not scan; lane ols-regression, 2026-11-14): the entry tests the words
    # it reads and returns 1 as its result word when any is NaN or infinite.
    var test_x: Bool

    def __init__(out self, x: Int, coef: Int, dst: Int, rows: Int, cols: Int,
                 intercept: Float64, wide: Bool, test_x: Bool = False):
        self.x = x
        self.coef = coef
        self.dst = dst
        self.rows = rows
        self.cols = cols
        self.intercept = intercept
        self.wide = wide
        self.test_x = test_x


#: Bits of the request's third integer word.
comptime LINEAR_PREDICT_WIDE = 1
comptime LINEAR_PREDICT_TEST_X = 2


def linear_predict_flags(request: LinearPredictRequest) -> Int:
    return (LINEAR_PREDICT_WIDE if request.wide else 0) | (LINEAR_PREDICT_TEST_X if request.test_x else 0)


def linear_predict_request_from_spans(operation: Int, aa: Int, na: Int,
    pp: Int, np: Int, ff: Int, nf: Int, result: Int) raises -> LinearPredictRequest:
    if operation != 0 or na != 3 or np != 3 or nf != 1 or not aa or not pp or not ff or not result:
        raise Error("linear prediction ABI: invalid request spans")
    var a = MutPointer[Int, MutUntrackedOrigin](unsafe_from_address=aa)
    var p = MutPointer[Int, MutUntrackedOrigin](unsafe_from_address=pp)
    var f = MutPointer[Float64, MutUntrackedOrigin](unsafe_from_address=ff)
    if not a[0] or not a[1] or not a[2]:
        raise Error("linear prediction ABI: null buffer address")
    return LinearPredictRequest(a[0], a[1], a[2], p[0], p[1], f[0], (p[2] & LINEAR_PREDICT_WIDE) != 0,
        (p[2] & LINEAR_PREDICT_TEST_X) != 0)
