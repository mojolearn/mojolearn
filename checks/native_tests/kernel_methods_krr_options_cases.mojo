# SPDX-License-Identifier: Apache-2.0
"""KernelRidge exact properties from test_krr_options.py, on the actual entries.

The original 96x5 integer workload (NumPy seed 3, recorded under
fixtures/kernel_methods_krr_options/) gives an exact linear Gram matrix under
any summation order, so:

  0  kernel='precomputed' on that Gram (alpha 2) returns kernel='linear''s
     dual_coef_ words, its prediction from the 20x96 cross Gram returns the
     linear prediction of X[:20] word for word, and a second prediction
     repeats the first;
  1  a unit sample_weight (the weighted arm, sqrt(1) = 1) returns the
     unweighted rbf (gamma 0.1, alpha 1) dual_coef_ words;
  2  two weighted rbf fits (gamma 0.2, alpha 0.5, w = 0.25 + i % 3) return
     the same dual_coef_ and predict(X) words.

The Gram matrices are Int64 sums rounded once to Float32, the original
`(a.astype(int64) @ a.astype(int64).T).astype(float32)`. The scikit-learn
tolerance comparison stays in Python as an external oracle.
"""
from std.memory import bitcast
from core.host_parallel import host_parallelize

comptime KRR_OPTIONS_FIXTURES = 3
comptime KRR_ROWS = 96
comptime KRR_FEATURES = 5
comptime KRR_QUERIES = 20
comptime KRR_KERNEL_LINEAR = 0
comptime KRR_KERNEL_RBF = 2
comptime KRR_KERNEL_PRECOMPUTED = 4


def require(ok: Bool, message: String) raises:
    if not ok:
        raise Error(message)


trait KrrOptionsCalls:
    @staticmethod
    def fit_predict(x: List[Float32], y: List[Float32], weights: List[Float64], n: Int, d: Int,
                    kernel: Int, gamma: Float64, alpha: Float32, query: List[Float32], q: Int,
                    ) raises -> Tuple[List[Float32], List[Float32], List[Float32]]:
        """The fit, then two predictions of `query`: (dual_coef_, first, second).
        An empty `weights` is the unweighted fit."""
        ...


def recorded(name: String, count: Int) raises -> List[Float32]:
    var result = List[Float32](unsafe_uninit_length=count)
    with open("checks/native_tests/fixtures/kernel_methods_krr_options/" + name, "r") as file:
        var bytes = file.read_bytes()
        require(len(bytes) == count * 4, "KernelRidge options fixture extent changed")
        var bp = bytes.unsafe_ptr()
        var rp = result.unsafe_ptr()
        def word(index: Int) {imm bp, imm rp}:
            var at = index * 4
            var bits = UInt32(bp[at]) | (UInt32(bp[at + 1]) << 8) | (UInt32(bp[at + 2]) << 16) | (UInt32(bp[at + 3]) << 24)
            rp.unsafe_store(index, bitcast[DType.float32](bits))
        host_parallelize(word, count)
        _ = bytes^
    return result^


def integer_gram(x: List[Float32], rows: Int, n: Int, d: Int) raises -> List[Float32]:
    """rows x n Gram of X[:rows] against X, exact Int64 sums, one rounding."""
    var result = List[Float32](unsafe_uninit_length=rows * n)
    var xp = x.unsafe_ptr()
    var rp = result.unsafe_ptr()
    def cell(index: Int) {imm xp, imm rp, imm n, imm d}:
        var i = index // n
        var j = index % n
        var total = Int64(0)
        for f in range(d):  # one dot product's integer terms; exact in any order
            total += Int64(xp[i * d + f]) * Int64(xp[j * d + f])
        rp.unsafe_store(index, Float32(total))
    host_parallelize(cell, rows * n)
    _ = x
    return result^


def same_words(left: List[Float32], right: List[Float32], name: String) raises:
    require(len(left) == len(right), name + ": extents differ")
    var failures = List[Int](length=len(left), fill=0)
    var fp = failures.unsafe_ptr()
    var lp = left.unsafe_ptr()
    var rp = right.unsafe_ptr()
    def cell(index: Int) {imm fp, imm lp, imm rp}:
        fp.unsafe_store(index, Int(bitcast[DType.uint32](lp[index]) != bitcast[DType.uint32](rp[index])))
    host_parallelize(cell, len(left))
    var differ = 0
    for value in failures:
        differ += value
    require(differ == 0, name + ": " + String(differ) + " words differ")
    _ = left
    _ = right


def fixture[Calls: KrrOptionsCalls](unit: Int) raises:
    var x = recorded("x-96x5.f32", KRR_ROWS * KRR_FEATURES)
    var y = recorded("y-96.f32", KRR_ROWS)
    var none = List[Float64]()
    if unit == 0:
        var query = List[Float32](unsafe_uninit_length=KRR_QUERIES * KRR_FEATURES)
        var sp = x.unsafe_ptr()
        var qp = query.unsafe_ptr()
        def prefix(index: Int) {imm sp, imm qp}:  # X[:20]
            qp.unsafe_store(index, sp[index])
        host_parallelize(prefix, KRR_QUERIES * KRR_FEATURES)
        var linear = Calls.fit_predict(x, y, none, KRR_ROWS, KRR_FEATURES, KRR_KERNEL_LINEAR, 1.0, 2.0,
                                       query, KRR_QUERIES)
        var gram = integer_gram(x, KRR_ROWS, KRR_ROWS, KRR_FEATURES)
        var cross = integer_gram(x, KRR_QUERIES, KRR_ROWS, KRR_FEATURES)
        var pre = Calls.fit_predict(gram, y, none, KRR_ROWS, KRR_ROWS, KRR_KERNEL_PRECOMPUTED, 1.0, 2.0,
                                    cross, KRR_QUERIES)
        same_words(pre[0], linear[0], "precomputed vs linear dual_coef_")
        same_words(pre[1], linear[1], "precomputed vs linear predict")
        same_words(pre[1], pre[2], "precomputed repeated predict")
    elif unit == 1:
        var ones = List[Float64](length=KRR_ROWS, fill=1.0)
        var base = Calls.fit_predict(x, y, none, KRR_ROWS, KRR_FEATURES, KRR_KERNEL_RBF, 0.1, 1.0, x, KRR_ROWS)
        var one = Calls.fit_predict(x, y, ones, KRR_ROWS, KRR_FEATURES, KRR_KERNEL_RBF, 0.1, 1.0, x, KRR_ROWS)
        same_words(one[0], base[0], "unit weight vs unweighted dual_coef_")
    else:
        var weights = List[Float64](unsafe_uninit_length=KRR_ROWS)
        var wp = weights.unsafe_ptr()
        def weight(index: Int) {imm wp}:  # the original 0.25 + (arange % 3)
            wp.unsafe_store(index, 0.25 + Float64(index % 3))
        host_parallelize(weight, KRR_ROWS)
        var first = Calls.fit_predict(x, y, weights, KRR_ROWS, KRR_FEATURES, KRR_KERNEL_RBF, 0.2, 0.5, x, KRR_ROWS)
        var second = Calls.fit_predict(x, y, weights, KRR_ROWS, KRR_FEATURES, KRR_KERNEL_RBF, 0.2, 0.5, x, KRR_ROWS)
        same_words(first[0], second[0], "repeated weighted dual_coef_")
        same_words(first[1], second[1], "repeated weighted predict")
