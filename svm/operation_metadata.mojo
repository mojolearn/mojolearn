# SPDX-License-Identifier: Apache-2.0
# Stable settings captured at a public SVC operation boundary. No Python owner.
comptime SVC_ABI_NAME = "mojolearn.svm.binary.v1"


struct SvcOperationSettings(Copyable, Movable):
    var kernel: Int
    var degree: Int
    var max_iter: Int
    var nochange_steps: Int
    var features: Int
    var gamma: Float64
    var c: Float64
    var tol: Float64
    var coef0: Float64
    var cache_mib: Float64

    def __init__(out self, kernel: Int, degree: Int, max_iter: Int, nochange_steps: Int,
                 features: Int, gamma: Float64, c: Float64, tol: Float64,
                 coef0: Float64, cache_mib: Float64):
        self.kernel = kernel
        self.degree = degree
        self.max_iter = max_iter
        self.nochange_steps = nochange_steps
        self.features = features
        self.gamma = gamma
        self.c = c
        self.tol = tol
        self.coef0 = coef0
        self.cache_mib = cache_mib

    def fit_ints(self, rows: Int, cols: Int) -> List[Int]:
        return [rows, cols, self.kernel, self.max_iter, self.nochange_steps, self.degree]

    def fit_floats(self) -> List[Float64]:
        return [self.gamma, self.c, self.tol, self.coef0]

    def predict_ints(self, rows: Int, cols: Int, supports: Int, labels: Bool) -> List[Int]:
        return [rows, cols, supports, self.kernel, 1 if labels else 0, self.degree]

    def predict_floats(self, bias: Float64, label0: Float64, label1: Float64) -> List[Float64]:
        return [bias, label0, label1, self.gamma, self.cache_mib, self.coef0]
