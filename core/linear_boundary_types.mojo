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
    # The door deferred X's finite test to the entry (a borrowed X it did
    # not scan; lane ols-regression, 2026-11-14): the entry tests the words
    # it reads and returns 1 as its result word when any is NaN or infinite.
    var test_x: Bool
    # THE T WORD (lane w3f3-k): a multi-target Ridge's `targets` coefficient
    # rows (coef targets x cols, dst rows x targets row-major) and the
    # address of its `targets` float64 intercepts (0: one target, whose
    # intercept is `intercept`).
    var targets: Int
    var intercepts: Int

    def __init__(out self, x: Int, coef: Int, dst: Int, rows: Int, cols: Int,
                 intercept: Float64, test_x: Bool = False):
        self.x = x
        self.coef = coef
        self.dst = dst
        self.rows = rows
        self.cols = cols
        self.intercept = intercept
        self.test_x = test_x
        self.targets = 1
        self.intercepts = 0


#: Bits of the request's third integer word. Bit 1 (the float64 widening of
#: the public predict) is gone: predict returns the float32 words to every
#: caller (lane predict-dtype, 2026-11-22).
comptime LINEAR_PREDICT_TEST_X = 2


def linear_predict_flags(request: LinearPredictRequest) -> Int:
    return LINEAR_PREDICT_TEST_X if request.test_x else 0


def linear_predict_request_from_spans(operation: Int, aa: Int, na: Int,
    pp: Int, np: Int, ff: Int, nf: Int, result: Int) raises -> LinearPredictRequest:
    # p: [rows, cols, flags] and optionally [3] the target count T (lane
    # w3f3-k); f: the T intercepts (one without the fourth word).
    if operation != 0 or na != 3 or (np != 3 and np != 4) or not aa or not pp or not ff or not result:
        raise Error("linear prediction ABI: invalid request spans")
    var a = MutPointer[Int, MutUntrackedOrigin](unsafe_from_address=aa)
    var p = MutPointer[Int, MutUntrackedOrigin](unsafe_from_address=pp)
    var f = MutPointer[Float64, MutUntrackedOrigin](unsafe_from_address=ff)
    var targets = p[3] if np == 4 else 1
    if targets < 1 or nf != targets:
        raise Error("linear prediction ABI: invalid request spans")
    if not a[0] or not a[1] or not a[2]:
        raise Error("linear prediction ABI: null buffer address")
    var request = LinearPredictRequest(a[0], a[1], a[2], p[0], p[1], f[0], (p[2] & LINEAR_PREDICT_TEST_X) != 0)
    if targets > 1:
        request.targets = targets
        request.intercepts = ff
    return request^
