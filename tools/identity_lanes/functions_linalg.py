# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
# ONE LANE PER PUBLIC LINEAR-ALGEBRA FUNCTION (mojolearn.linalg.*, and the
# top-level matmul, lstsq, solve, lu_factor, lu_solve, randomized_svd,
# sparse_encode), sub-lane vdv-cov-func, 2026-11-24.
#
# Every lane calls its function through `ml` with no device= and hands the call
# to `_functions_resident` (tools/identity_lanes/functions.py) with the "train"
# form: on the GPU side every float32 input is handed again as a
# TrainDeviceArray (the training family's device form, which lane vdf-linalg2
# connects for linalg, lowbit and the dense decompositions) and the answer must
# be byte-for-byte the host-input answer of the same side. Integer inputs
# (int8 codes, int32 exponents, uint16 bf16 bits, pivots) stay host arrays.
# Matrices are built with exact elementwise host steps only (no host BLAS).


def _functions_linalg_square(X, n, seed):
    """An (n, n) float32 matrix, diagonally dominant (nonsingular on every
    fixture): the fixture matrix plus n on the diagonal."""
    A = _functions_mat(X, n, n, seed)
    A[np.arange(n), np.arange(n)] += np.float32(n)
    return np.ascontiguousarray(A)


def _functions_linalg_sym(X, n, seed):
    """A symmetric (n, n) float32 matrix: (A + A^T) / 2, exact elementwise."""
    A = _functions_mat(X, n, n, seed)
    return np.ascontiguousarray(((A + A.T) * np.float32(0.5)).astype(np.float32))


@lane("fn-linalg-matmul")
def _(ml, X, yc, yr, Xh=None):
    """linalg.matmul (and the same function as mojolearn.matmul): NN, NT, TN."""
    a = _functions_mat(X, 96, 16, "fn-matmul:a")
    b = _functions_mat(X[96:], 16, 24, "fn-matmul:b")
    c = _functions_mat(X[200:], 24, 16, "fn-matmul:c")
    return _functions_resident(ml, "linalg.matmul", lambda p: dict(
        nn=_functions_d(ml.linalg.matmul(p(a), p(b))),
        nt=_functions_d(ml.linalg.matmul(p(a), p(c), transpose_b=True)),
        tn=_functions_d(ml.linalg.matmul(p(a), p(a), transpose_a=True)),
        top=_functions_d(ml.matmul(p(a), p(b)))), form="train")


@lane("fn-linalg-matmul-bf16")
def _(ml, X, yc, yr, Xh=None):
    """linalg.matmul_bf16: float32 times bf16 bits, and bf16 bits times bf16 bits (both NT).

    The documented operands (python/mojolearn/_linalg_impl.py matmul_bf16): `b` is bf16 BITS (uint16 from to_bf16),
    `a` is float32 or bf16 bits. The earlier second part passed a float32 `b`, which the product refuses by design
    ("b has dtype float32; the bf16 profile takes bf16 BITS", bindings/linalg_ops_lowbit_door.mojo), so the lane was
    REFUSED on every side. Fixture defect; the second part now covers the bits-times-bits form instead
    (FIX w3-fix-13, 2026-11-25).

    The bits operands are made by `to_bf16` of the side's float32 operand (harness fix, lane
    gs-fix): the train form uploads float32 only, so host uint16 bits beside an uploaded `a` were refused as mixed
    device and host arrays on the GPU side. `to_bf16` of a TrainDeviceArray returns the uint16 device array
    (bindings/linalg_ops_lowbit_door.mojo), as fn-lu-solve's pivots come from lu_factor of the uploaded matrix."""
    a = _functions_mat(X, 64, 16, "fn-bf16:a")
    b = _functions_mat(X[64:], 32, 16, "fn-bf16:b")
    return _functions_resident(ml, "linalg.matmul_bf16", lambda p: dict(
        nt=_functions_d(ml.linalg.matmul_bf16(p(a), ml.linalg.to_bf16(p(b)), transpose_b=True)),
        bits=_functions_d(ml.linalg.matmul_bf16(ml.linalg.to_bf16(p(a)), ml.linalg.to_bf16(p(b)),
                                                transpose_b=True))), form="train")


@lane("fn-linalg-to-bf16")
def _(ml, X, yc, yr, Xh=None):
    x = _functions_mat(X, 64, 16, "fn-to-bf16:x")
    return _functions_resident(ml, "linalg.to_bf16", lambda p: dict(
        bits=_functions_d(ml.linalg.to_bf16(p(x)))), form="train")


@lane("fn-linalg-from-bf16")
def _(ml, X, yc, yr, Xh=None):
    x = _functions_mat(X, 64, 16, "fn-from-bf16:x")
    bits = np.ascontiguousarray(np.asarray(ml.linalg.to_bf16(x)))
    return _functions_resident(ml, "linalg.from_bf16", lambda p: dict(
        values=_functions_d(ml.linalg.from_bf16(p(bits)))), form="train")


@lane("fn-linalg-quantize-int8")
def _(ml, X, yc, yr, Xh=None):
    x = _functions_mat(X, 64, 16, "fn-q8:x")
    return _functions_resident(ml, "linalg.quantize_int8", lambda p: dict(
        q=_functions_d(ml.linalg.quantize_int8(p(x)))), form="train")


@lane("fn-linalg-dequantize-int8")
def _(ml, X, yc, yr, Xh=None):
    x = _functions_mat(X, 64, 16, "fn-dq8:x")
    codes, exps = ml.linalg.quantize_int8(x)
    codes, exps = np.ascontiguousarray(np.asarray(codes)), np.ascontiguousarray(np.asarray(exps))
    return _functions_resident(ml, "linalg.dequantize_int8", lambda p: dict(
        x=_functions_d(ml.linalg.dequantize_int8(p(codes), p(exps)))), form="train")


@lane("fn-linalg-matmul-int8")
def _(ml, X, yc, yr, Xh=None):
    """linalg.matmul_int8 from float32 operands and from quantized pairs."""
    a = _functions_mat(X, 64, 16, "fn-mm8:a")
    b = _functions_mat(X[64:], 32, 16, "fn-mm8:b")
    qa, ea = (np.ascontiguousarray(np.asarray(v)) for v in ml.linalg.quantize_int8(a))
    qb, eb = (np.ascontiguousarray(np.asarray(v)) for v in ml.linalg.quantize_int8(b))
    return _functions_resident(ml, "linalg.matmul_int8", lambda p: dict(
        floats=_functions_d(ml.linalg.matmul_int8(p(a), p(b))),
        codes=_functions_d(ml.linalg.matmul_int8((p(qa), p(ea)), (p(qb), p(eb))))), form="train")


@lane("fn-linalg-quantize-int15")
def _(ml, X, yc, yr, Xh=None):
    x = _functions_mat(X, 64, 16, "fn-q15:x")
    return _functions_resident(ml, "linalg.quantize_int15", lambda p: dict(
        planes=_functions_d(ml.linalg.quantize_int15(p(x)))), form="train")


@lane("fn-linalg-dequantize-int15")
def _(ml, X, yc, yr, Xh=None):
    x = _functions_mat(X, 64, 16, "fn-dq15:x")
    hi, lo, ex = (np.ascontiguousarray(np.asarray(v)) for v in ml.linalg.quantize_int15(x))
    return _functions_resident(ml, "linalg.dequantize_int15", lambda p: dict(
        x=_functions_d(ml.linalg.dequantize_int15(p(hi), p(lo), p(ex)))), form="train")


@lane("fn-linalg-matmul-int15")
def _(ml, X, yc, yr, Xh=None):
    """linalg.matmul_int15 from float32 operands and from the planes."""
    a = _functions_mat(X, 64, 16, "fn-mm15:a")
    b = _functions_mat(X[64:], 32, 16, "fn-mm15:b")
    ah, al, ea = (np.ascontiguousarray(np.asarray(v)) for v in ml.linalg.quantize_int15(a))
    bh, bl, eb = (np.ascontiguousarray(np.asarray(v)) for v in ml.linalg.quantize_int15(b))
    return _functions_resident(ml, "linalg.matmul_int15", lambda p: dict(
        floats=_functions_d(ml.linalg.matmul_int15(p(a), p(b))),
        planes=_functions_d(ml.linalg.matmul_int15((p(ah), p(al), p(ea)), (p(bh), p(bl), p(eb))))), form="train")


@lane("fn-linalg-qr")
def _(ml, X, yc, yr, Xh=None):
    """linalg.qr of a tall and a wide matrix, every mode."""
    T = _functions_mat(X, 40, 8, "fn-qr:t")
    W = _functions_mat(X, 5, 8, "fn-qr:w")
    return _functions_resident(ml, "linalg.qr", lambda p: dict(
        r=_functions_d(ml.linalg.qr(p(T))),
        reduced=_functions_d(ml.linalg.qr(p(T), mode="reduced")),
        complete=_functions_d(ml.linalg.qr(p(T), mode="complete")),
        raw=_functions_d(ml.linalg.qr(p(W), mode="raw")),
        wide=_functions_d(ml.linalg.qr(p(W), mode="reduced"))), form="train")


@lane("fn-linalg-eigh")
def _(ml, X, yc, yr, Xh=None):
    G = _functions_linalg_sym(X, 12, "fn-eigh:g")
    return _functions_resident(ml, "linalg.eigh", lambda p: dict(
        lower=_functions_d(ml.linalg.eigh(p(G))),
        upper=_functions_d(ml.linalg.eigh(p(G), UPLO="U"))), form="train")


@lane("fn-linalg-svd")
def _(ml, X, yc, yr, Xh=None):
    T = _functions_mat(X, 40, 8, "fn-svd:t")
    W = _functions_mat(X, 5, 8, "fn-svd:w")
    return _functions_resident(ml, "linalg.svd", lambda p: dict(
        thin=_functions_d(ml.linalg.svd(p(T), full_matrices=False)),
        full=_functions_d(ml.linalg.svd(p(W))),
        values=_functions_d(ml.linalg.svd(p(T), compute_uv=False))), form="train")


@lane("fn-linalg-svdvals")
def _(ml, X, yc, yr, Xh=None):
    T = _functions_mat(X, 40, 8, "fn-svdvals:t")
    W = _functions_mat(X, 5, 8, "fn-svdvals:w")
    return _functions_resident(ml, "linalg.svdvals", lambda p: dict(
        tall=_functions_d(ml.linalg.svdvals(p(T))),
        wide=_functions_d(ml.linalg.svdvals(p(W)))), form="train")


@lane("fn-lstsq")
def _(ml, X, yc, yr, Xh=None):
    a = _functions_mat(X, 300, 6, "fn-lstsq:a")
    b = _functions_cont(X, 300, 7, "fn-lstsq:b")
    B = _functions_mat(X[300:], 300, 3, "fn-lstsq:B")
    return _functions_resident(ml, "lstsq", lambda p: dict(
        vector=_functions_d(ml.lstsq(p(a), p(b))),
        matrix=_functions_d(ml.lstsq(p(a), p(B)))), form="train")


@lane("fn-solve")
def _(ml, X, yc, yr, Xh=None):
    A = _functions_linalg_square(X, 10, "fn-solve:a")
    b = _functions_cont(X, 10, 4, "fn-solve:b")
    B = _functions_mat(X[10:], 10, 3, "fn-solve:B")
    return _functions_resident(ml, "solve", lambda p: dict(
        vector=_functions_d(ml.solve(p(A), p(b))),
        matrix=_functions_d(ml.solve(p(A), p(B)))), form="train")


@lane("fn-lu-factor")
def _(ml, X, yc, yr, Xh=None):
    A = _functions_linalg_square(X, 10, "fn-lu-factor:a")
    return _functions_resident(ml, "lu_factor", lambda p: dict(
        lu=_functions_d(ml.lu_factor(p(A)))), form="train")


@lane("fn-lu-solve")
def _(ml, X, yc, yr, Xh=None):
    """lu_solve on the host factor of a nonsingular matrix: a vector, a matrix,
    and the transposed system."""
    A = _functions_linalg_square(X, 10, "fn-lu-solve:a")
    lu, piv = ml.lu_factor(A)
    lu, piv = np.ascontiguousarray(np.asarray(lu)), np.ascontiguousarray(np.asarray(piv))
    b = _functions_cont(X, 10, 4, "fn-lu-solve:b")
    B = _functions_mat(X[10:], 10, 3, "fn-lu-solve:B")

    def pair(p):
        # The device form of the (lu, piv) pair is lu_factor of the uploaded
        # matrix: piv is an int32 device array (TRAIN_DEV_I32), which only
        # lu_factor makes (train_to_device uploads float32 only). Uploading lu
        # alone left piv on the host: "device and host arrays are mixed"
        # (gd-verify-view, 2026-11-25). fn-lu-factor holds that pair's digest
        # to the host factor's, so both calls solve with the same factor.
        return (lu, piv) if p is _functions_as_given else ml.lu_factor(p(A))

    def run(p):
        lp = pair(p)
        return dict(vector=_functions_d(ml.lu_solve(lp, p(b))),
                    matrix=_functions_d(ml.lu_solve(lp, p(B))),
                    trans=_functions_d(ml.lu_solve(lp, p(B), trans=1)))
    return _functions_resident(ml, "lu_solve", run, form="train")


@lane("fn-randomized-svd")
def _(ml, X, yc, yr, Xh=None):
    M = _functions_mat(X, 600, 8, "fn-rsvd:m")
    return _functions_resident(ml, "randomized_svd", lambda p: dict(
        k3=_functions_d(ml.randomized_svd(p(M), 3, random_state=0)),
        k2=_functions_d(ml.randomized_svd(p(M), 2, n_iter=2, random_state=1))), form="train")


@lane("fn-sparse-encode")
def _(ml, X, yc, yr, Xh=None):
    """sparse_encode of 32 rows over the identity dictionary (lasso_cd, omp)
    and over a fixture dictionary (threshold)."""
    rows = _functions_mat(X, 32, 6, "fn-sparse:rows")
    eye = np.eye(6, dtype=np.float32)
    D = _functions_mat(X[32:], 6, 6, "fn-sparse:dict")
    return _functions_resident(ml, "sparse_encode", lambda p: dict(
        threshold=_functions_d(ml.sparse_encode(p(rows), p(D), algorithm="threshold", alpha=0.125)),
        lasso=_functions_d(ml.sparse_encode(p(rows), p(eye), algorithm="lasso_cd", alpha=0.125, max_iter=30)),
        omp=_functions_d(ml.sparse_encode(p(rows), p(eye), algorithm="omp", n_nonzero_coefs=2))), form="train")


_batch_decl(_FUNCTIONS_BATCH_NA, "fn-linalg-matmul", "fn-linalg-matmul-bf16", "fn-linalg-to-bf16", "fn-linalg-from-bf16",
            "fn-linalg-quantize-int8", "fn-linalg-dequantize-int8", "fn-linalg-matmul-int8", "fn-linalg-quantize-int15",
            "fn-linalg-dequantize-int15", "fn-linalg-matmul-int15", "fn-linalg-qr", "fn-linalg-eigh", "fn-linalg-svd",
            "fn-linalg-svdvals", "fn-lstsq", "fn-solve", "fn-lu-factor", "fn-lu-solve", "fn-randomized-svd",
            "fn-sparse-encode")
