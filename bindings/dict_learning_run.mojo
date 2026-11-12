# SPDX-License-Identifier: Apache-2.0
"""RUN piece of the native dictionary-learning owner (DictionaryLearning,
MiniBatchDictionaryLearning, SparseCoder, SparsePCA, MiniBatchSparsePCA and
`sparse_encode`; bindings/dict_learning_native.mojo is UNPACK and PACK):
`_dict_learning`, `_sparse_encode`, the mini-batch loop and the SparsePCA
ridge projection over the typed kit (bindings/decomp_kit_run.mojo). Same
x_decomp operations, same words, same order: the bits do not move. Every
option arrives as a typed word from UNPACK; no Python (lane
entangled-decompkit, 2026-11-12).

The Gram `G = D D^T` of a query's fitted components is served from the
model's resident record by UNPACK (`g_held`, -1 on a miss); RUN computes it
on a miss and reports the matrix (`gram`) so PACK can keep it for the next
query (N339, N333). The words are the ones this gemm computes, so no bit
changes."""
from std.memory import unsafe_memcpy
from bindings.decomp_kit_run import (
    DKit, DK_SUB, DK_ADD, DK_MUL, DK_DIV, DK_MAXS, DK_MINS, DK_SCALE, DK_SOFT, DK_SQ, DK_SQRT, DK_SELECT,
)
from x_decomp.descriptor_schema import XD_DICT_LEARNING, XD_DICT_MINIBATCH


struct EncodeWords(Copyable, Movable):
    """`_sparse_encode`'s options, read once by UNPACK: `algorithm` the
    position in `_SPARSE_ALGOS` (0 lasso_lars, 1 lasso_cd, 2 lars, 3 omp,
    4 threshold), `count` the nonzero budget (lars, omp), `reg` the alpha,
    `max_iter` and `positive` (lasso), `positive_flag` (threshold)."""
    var algorithm: Int
    var count: Int
    var reg: Float64
    var max_iter: Int
    var positive: Int
    var positive_flag: Bool
    var split_sign: Bool

    def __init__(out self, algorithm: Int, count: Int, reg: Float64, max_iter: Int, positive: Int,
                 positive_flag: Bool, split_sign: Bool):
        self.algorithm = algorithm
        self.count = count
        self.reg = reg
        self.max_iter = max_iter
        self.positive = positive
        self.positive_flag = positive_flag
        self.split_sign = split_sign


struct Gram(Copyable, Movable):
    """The Gram a query's encode used: `served` the record's matrix (-1 on a
    miss), `computed` the one RUN made (-1 when served or not needed)."""
    var served: Int
    var computed: Int

    def __init__(out self, served: Int):
        self.served = served
        self.computed = -1


def dl_gram(mut kit: DKit, d: Int, mut gram: Gram) raises -> Int:
    if gram.served >= 0:
        return gram.served
    var g = kit.mm(d, d, tb=True)
    gram.computed = g
    return g


def dl_split_sign(mut kit: DKit, code: Int) raises -> Int:
    # [max(code, 0), -min(code, 0)] side by side.
    var positive = kit.ew(DK_MAXS, code, scalar=0.0)
    var low = kit.ew(DK_MINS, code, scalar=0.0)
    var negative = kit.ew(DK_SCALE, low, scalar=-1.0)
    return kit.hstack([positive, negative])


def dl_sparse_encode(mut kit: DKit, x: Int, d: Int, words: EncodeWords, init: Int, mut gram: Gram) raises -> Int:
    # `_expansion_decomp._sparse_encode`.
    var index = words.algorithm
    var n = kit.nrows(x)
    var m = kit.ncols(x)
    var kc = kit.nrows(d)
    if index == 2:
        var g = dl_gram(kit, d, gram)
        var q = kit.mm(x, d, tb=True)
        return kit.lars_rows(g, q, m, words.count)
    var q = kit.mm(x, d, tb=True)
    if index == 4:
        var code = kit.ew(DK_SOFT, q, scalar=words.reg)
        if words.positive_flag:
            return kit.ew(DK_MAXS, code, scalar=0.0)
        return code
    var g = dl_gram(kit, d, gram)
    if index == 3:
        return kit.omp_rows(g, q, words.count)
    var w = kit.copy_host(init) if (index == 1 and init >= 0) else kit.zeros(n, kc)
    return kit.lasso_rows(g, q, w, words.reg, words.max_iter, 1e-8, words.positive)


def dl_encode_run(mut kit: DKit, x: Int, d: Int, words: EncodeWords, mut gram: Gram) raises -> Int:
    # `_SparseCoding._encode`: the code, split by sign when asked.
    var code = dl_sparse_encode(kit, x, d, words, -1, gram)
    if words.split_sign:
        return dl_split_sign(kit, code)
    return code


def dl_inverse_run(mut kit: DKit, c: Int, d: Int, split_sign: Bool) raises -> Int:
    # `_SparseCoding.inverse_transform`: code @ components_.
    var code = c
    if split_sign:
        var h = kit.nrows(d)
        var plus = kit.cols_of(code, 0, h)
        var minus = kit.cols_of(code, h, 2 * h)
        code = kit.ew(DK_SUB, plus, minus)
    return kit.mm(code, d)


struct DictWords(Copyable, Movable):
    """`_dict_learning`'s options, read once by UNPACK."""
    var components: Int
    var alpha: Float64
    var max_iter: Int
    var tol: Float64
    var method: Int
    var seed: Int           # masked to 32 bits
    var positive_dict: Bool
    var positive_code: Bool
    var method_max_iter: Int

    def __init__(out self, components: Int, alpha: Float64, max_iter: Int, tol: Float64, method: Int, seed: Int,
                 positive_dict: Bool, positive_code: Bool, method_max_iter: Int):
        self.components = components
        self.alpha = alpha
        self.max_iter = max_iter
        self.tol = tol
        self.method = method
        self.seed = seed
        self.positive_dict = positive_dict
        self.positive_code = positive_code
        self.method_max_iter = method_max_iter


struct DictFit(Movable):
    """`_dict_learning`'s (code, D, errors, n_iter)."""
    var code: Int
    var d: Int
    var errors: List[Float64]
    var done: Int

    def __init__(out self):
        self.code = -1
        self.d = -1
        self.errors = List[Float64]()
        self.done = 0


def dl_dict_learning(mut kit: DKit, x: Int, words: DictWords, ci: Int, di: Int) raises -> DictFit:
    # `_expansion_decomp._dict_learning`.
    var nc = words.components
    var code: Int
    var d: Int
    if ci >= 0 and di >= 0:
        code = ci
        d = di
    else:
        var usv = kit.thin_svd(x, min(kit.nrows(x), kit.ncols(x)), True)
        code = usv[0]
        var st = kit.transpose(usv[1])
        d = kit.ew(DK_MUL, usv[2], st)
    var r = kit.nrows(d)
    if nc <= r:
        code = kit.cols_of(code, 0, nc)
        d = kit.rows_of(d, 0, nc)
    else:
        var code_pad = kit.zeros(kit.nrows(code), nc - r)
        code = kit.hstack([code, code_pad])
        var d_pad = kit.zeros(nc - r, kit.ncols(d))
        d = kit.vstack([d, d_pad])
    # The loop (encode, atom update, cost, stopping test) is ONE call on host
    # copies of the code and the dictionary, written in place.
    code = kit.copy_host(code)
    d = kit.copy_host(d)
    var slots = max(words.max_iter, 1)
    var store = List[Float64](length=slots, fill=0.0)
    var params: List[Int] = [kit.nrows(x), kit.ncols(x), nc, words.method, words.max_iter, words.method_max_iter,
                             Int(words.positive_dict), Int(words.positive_code), words.seed, -1, -1]
    var floats: List[Float64] = [words.alpha, words.tol]
    # ii is the result, the count of recorded errors comes back in `written`.
    var written = List[Int](length=1, fill=0)
    var addresses: List[Int] = [kit.at(x), kit.at(code), kit.at(d), Int(store.unsafe_ptr()), Int(written.unsafe_ptr())]
    var done = kit.xd(XD_DICT_LEARNING, addresses, params, floats)
    kit.wrote_host(code)
    kit.wrote_host(d)
    var recorded = max(min(written[0], slots), 0)
    _ = written^
    var result = DictFit()
    result.errors = List[Float64](length=recorded, fill=0.0)
    if recorded > 0:
        unsafe_memcpy(dest=result.errors.unsafe_ptr(), src=store.unsafe_ptr(), count=recorded)
    _ = store^  # keep the error store alive through the call above
    result.code = code
    result.d = d
    result.done = done
    return result^


struct MiniBatchWords(Copyable, Movable):
    """The mini-batch loop's options, read once by UNPACK (`batch` already
    min(batch_size, n) and positive, `steps` max_iter * per_pass)."""
    var batch: Int
    var seed: Int           # masked to 32 bits
    var shuffle: Bool
    var steps: Int
    var patience: Int       # -1: no limit
    var method: Int
    var transform_max_iter: Int
    var positive_dict: Bool
    var positive_code: Bool
    var alpha: Float64
    var tol: Float64

    def __init__(out self, batch: Int, seed: Int, shuffle: Bool, steps: Int, patience: Int, method: Int,
                 transform_max_iter: Int, positive_dict: Bool, positive_code: Bool, alpha: Float64, tol: Float64):
        self.batch = batch
        self.seed = seed
        self.shuffle = shuffle
        self.steps = steps
        self.patience = patience
        self.method = method
        self.transform_max_iter = transform_max_iter
        self.positive_dict = positive_dict
        self.positive_code = positive_code
        self.alpha = alpha
        self.tol = tol


def dl_minibatch_fit(mut kit: DKit, x: Int, nc: Int, dict_init: Int, words: MiniBatchWords) raises -> Tuple[Int, Int]:
    # MiniBatchDictionaryLearning.fit's steps on X (n x m): (D, n_steps_).
    var n = kit.nrows(x)
    var m = kit.ncols(x)
    var d: Int
    if dict_init >= 0:
        d = dict_init
    else:
        var usv = kit.thin_svd(x, min(n, m), True)
        var st = kit.transpose(usv[1])
        d = kit.ew(DK_MUL, usv[2], st)
    if nc <= kit.nrows(d):
        d = kit.rows_of(d, 0, nc)
    else:
        var pad = kit.zeros(nc - kit.nrows(d), m)
        d = kit.vstack([d, pad])
    var xt = x
    if words.shuffle:
        var draws = kit.rand(1, n, words.seed, 50, 0)
        var order = kit.order(draws)
        xt = kit.take_rows(x, order)
    # The step loop is ONE call on a host copy of the dictionary.
    d = kit.copy_host(d)
    var params: List[Int] = [n, m, nc, words.method, 0, words.transform_max_iter, Int(words.positive_dict),
                             Int(words.positive_code), words.seed, -1, words.patience, words.batch, words.steps]
    var floats: List[Float64] = [words.alpha, words.tol]
    var addresses: List[Int] = [kit.at(xt), kit.at(d)]
    var step = kit.xd(XD_DICT_MINIBATCH, addresses, params, floats) - 1
    kit.wrote_host(d)
    return (d, step + 1)


def dl_normalize(mut kit: DKit, c: Int) raises -> Int:
    # `_BaseSparsePCA._normalize`: each row over its norm (a zero row kept).
    var squares = kit.ew(DK_SQ, c)
    var sums = kit.rowsum(squares)
    var norms = kit.ew(DK_SQRT, sums)
    var one = kit.const(1.0)
    var safe = kit.ew(DK_SELECT, norms, norms, one, 0.0)
    return kit.ew(DK_DIV, c, safe)


def dl_spca_project(mut kit: DKit, x: Int, mean: Int, comps: Int, ridge_alpha: Float32,
                    mut gram: Gram) raises -> Int:
    # `_BaseSparsePCA.transform`: ridge regression of the centered rows (the
    # ridge diagonal and the factor depend on ridge_alpha too and are formed
    # per call).
    var xc = kit.ew(DK_SUB, x, mean)
    var g0 = dl_gram(kit, comps, gram)
    var alpha = kit.const(ridge_alpha)
    var g = kit.diag_add(g0, alpha)
    var factor = kit.lu(g)
    var rhs = kit.mm(comps, xc, tb=True)
    var solved = kit.lu_solve(factor[0], factor[1], rhs)
    return kit.transpose(solved)


def dl_spca_inverse(mut kit: DKit, x: Int, mean: Int, comps: Int) raises -> Int:
    var product = kit.mm(x, comps)
    return kit.ew(DK_ADD, product, mean)


struct SpcaFit(Movable):
    """SparsePCA / MiniBatchSparsePCA fit's results for PACK."""
    var mean: Int
    var comps: Int
    var errors: List[Float64]
    var done: Int           # SparsePCA's n_iter_
    var steps: Int          # MiniBatchSparsePCA's n_steps
    var projected: Int      # fit_transform's projection, -1 when not asked

    def __init__(out self):
        self.mean = -1
        self.comps = -1
        self.errors = List[Float64]()
        self.done = 0
        self.steps = 0
        self.projected = -1


def dl_spca_fit_run(mut kit: DKit, x: Int, vi: Int, ui: Int, minibatch: Bool, dict_words: DictWords,
                    mb_words: MiniBatchWords, encode: EncodeWords, project: Bool,
                    ridge_alpha: Float32) raises -> SpcaFit:
    """SparsePCA / MiniBatchSparsePCA fit (and fit_transform's ridge
    projection of the X, mean and components the fit holds). `vi` / `ui`
    the admitted V_init / U_init (-1 when absent)."""
    var result = SpcaFit()
    var nc = dict_words.components
    var mean = kit.colmean(x)
    var xc = kit.ew(DK_SUB, x, mean)
    var comps: Int
    if not minibatch:
        var ci = -1
        if vi >= 0:
            ci = kit.transpose(vi)
        var di = -1
        if ui >= 0:
            di = kit.transpose(ui)
        var xct = kit.transpose(xc)
        var fit = dl_dict_learning(kit, xct, dict_words, ci, di)
        var code_t = kit.transpose(fit.code)
        comps = dl_normalize(kit, code_t)
        swap(result.errors, fit.errors)  # take the words, leave `fit` whole
        result.done = fit.done
    else:
        # MiniBatchDictionaryLearning(n_components=nc, fit_algorithm=method,
        # transform_algorithm="lasso_lars", transform_alpha=alpha, ...) on X^T.
        var xt = kit.transpose(xc)
        var fit = dl_minibatch_fit(kit, xt, nc, -1, mb_words)
        var gram = Gram(-1)
        var code = dl_sparse_encode(kit, xt, fit[0], encode, -1, gram)
        var code_t = kit.transpose(code)
        comps = dl_normalize(kit, code_t)
        result.steps = fit[1]
    result.mean = mean
    result.comps = comps
    if project:
        var gram = Gram(-1)
        result.projected = dl_spca_project(kit, x, mean, comps, ridge_alpha, gram)
    return result^
