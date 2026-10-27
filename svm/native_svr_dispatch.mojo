# SPDX-License-Identifier: Apache-2.0
# Typed SVR transport to unchanged numerical owners; same packed scalar schema.
from std.memory import memcpy
from core.hostptr import f32_ptr as _f32_ptr, f64_ptr as _f64_ptr, i32_ptr as _i32_ptr, read_f32, copy_f32
from svm.estimator import SvrFitOutputs, svr_fit_host, svr_predict_host


def _c_rows(params: List[Int], slot: Int, n_rows: Int, who: String) raises -> List[Float32]:
    var out = List[Float32]()
    if len(params) <= slot or params[slot] == 0:
        return out^
    return read_f32(params[slot], n_rows)


def native_svr_fit(a: List[Int], params: List[Int], floats: List[Float64]) raises -> Int:
    if len(params) != 9 and len(params) != 10 and len(params) != 12:
        raise Error(
            "svr_fit: params must contain 9 values (10 with the per-row bounds, 12 with"
            " degree and coef0), got " + String(len(params))
        )
    var xp = _f32_ptr(a[0])
    var yp = _f32_ptr(a[1])
    var dp = _f32_ptr(a[2])
    var sip = _i32_ptr(a[3])
    var smp = _f32_ptr(a[4])
    var ip = _f64_ptr(a[5])
    var n_rows = params[0]
    var n_cols = params[1]
    var kernel = params[2]
    var gamma = floats[3]
    var c = floats[4]
    var epsilon = floats[5]
    var tol = floats[6]
    var max_iter = params[7]
    var nochange_steps = params[8]
    if n_rows <= 0 or n_cols <= 0:
        raise Error("svr_fit: n_rows and n_features must both be positive")
    var x = List[Float32]()
    var y = List[Float32]()
    x = read_f32(Int(xp), max(0, n_rows * n_cols))
    y = read_f32(Int(yp), max(0, n_rows))
    var c_rows = _c_rows(params, 9, n_rows, "svr_fit")
    var degree = 3
    var coef0 = 0.0
    if len(params) == 12:
        degree = params[10]
        coef0 = floats[11]
    var res = SvrFitOutputs()
    res = svr_fit_host(
        x, y, n_rows, n_cols, kernel, gamma, c, epsilon, tol, max_iter,
        nochange_steps, c_rows, degree, coef0,
    )
    copy_f32(res.dual_coefs.unsafe_ptr(), dp, res.n_support)
    memcpy(dest=sip, src=res.support_idx.unsafe_ptr(), count=res.n_support)
    copy_f32(res.support_matrix.unsafe_ptr(), smp, res.n_support * n_cols)
    ip.unsafe_store(0, Float64(res.b))
    ip.unsafe_store(1, Float64(res.n_support))
    ip.unsafe_store(2, Float64(res.n_iter))
    return res.n_support



def native_svr_predict(a: List[Int], params: List[Int], floats: List[Float64]) raises -> Int:
    if len(params) != 7 and len(params) != 9:
        raise Error(
            "svr_predict: params must contain 7 values (9 with degree and coef0), got "
            + String(len(params))
        )
    var xp = _f32_ptr(a[0])
    var op = _f32_ptr(a[3])
    var n_rows = params[0]
    var n_cols = params[1]
    var n_support = params[2]
    var b = Float32(floats[3])
    var kernel = params[4]
    var gamma = floats[5]
    var buffer_mib = floats[6]
    var degree = 3
    var coef0 = 0.0
    if len(params) == 9:
        degree = params[7]
        coef0 = floats[8]
    if n_rows <= 0 or n_cols <= 0:
        raise Error("svr_predict: n_rows and n_features must both be positive")
    if n_support < 0:
        raise Error("svr_predict: n_support cannot be negative")
    var x = List[Float32]()
    x = read_f32(Int(xp), max(0, n_rows * n_cols))
    var dual = List[Float32]()
    var support = List[Float32]()
    if n_support > 0:
        var dp = _f32_ptr(a[1])
        var smp = _f32_ptr(a[2])
        dual = read_f32(Int(dp), max(0, n_support))
        if kernel != 4:
            # kernel='precomputed' (4) reads no support rows: X is the
            # cross-kernel and the caller passes no support matrix.
            support = read_f32(Int(smp), max(0, n_support * n_cols))
    var out = List[Float32]()
    out = svr_predict_host(
        x, n_rows, n_cols, support, dual, n_support, b, kernel, gamma,
        buffer_mib, degree, coef0,
    )
    copy_f32(out.unsafe_ptr(), op, n_rows)
    return n_rows


