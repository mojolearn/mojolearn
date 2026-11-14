# SPDX-License-Identifier: Apache-2.0
"""GaussianMixture fit_predict returns the labels of fit then predict
(ported from python/mojolearn/tests/test_ne2_g_decomp_entries.py
`test_gmm_fit_predict_equals_fit_then_predict`, row N401).

The GPU provider writes the fit's own labels from its device buffers when
the fit request carries a labels output (capability MIXTURE_CAP_FIT_PREDICT).
A second fit with the same options and no labels output, followed by the
provider's predict over the fitted state, must give the same label words.
The data are two shifted groups of 40 rows x 3 features (the original's
shape and shift, regenerated deterministically in Mojo; the property does
not depend on NumPy's stream). Options are the public defaults the original
used: full covariance, kmeans init, tol 1e-3, reg_covar 1e-6, max_iter 100,
random_state 0.
"""
from std.math import sin
from core.host_parallel import host_parallelize
from core.mixture_native_api import (
    GmmDriver, GmmFitRequest, GmmScoreRequest, MIXTURE_NATIVE_CAPABILITIES, MIXTURE_CAP_FIT_PREDICT,
    MIXTURE_NATIVE_PREDICT,
)

comptime NE2G_FIXTURES = 1
comptime NE2G_ROWS = 80
comptime NE2G_FEATURES = 3
comptime NE2G_K = 2
comptime NE2G_SENTINEL = Int32(-7)


def require(ok: Bool, message: String) raises:
    if not ok:
        raise Error(message)


def two_groups() -> List[Float32]:
    var count = NE2G_ROWS * NE2G_FEATURES
    var result = List[Float32](unsafe_uninit_length=count)
    var rp = result.unsafe_ptr()
    def cell(index: Int) {imm rp}:
        var row = index // NE2G_FEATURES
        var column = index % NE2G_FEATURES
        var shift = -3.0 if row < NE2G_ROWS // 2 else 3.0
        var noise = sin(1.7 * Float64(row) + 0.9 * Float64(column) + 0.3) * 0.8
        rp.unsafe_store(index, Float32(shift + noise))
    host_parallelize(cell, count)
    return result^


struct GmmState(Movable):
    var weights: List[Float32]
    var means: List[Float32]
    var covariances: List[Float32]
    var precisions: List[Float32]
    var log_det: List[Float32]
    var scalars: List[Float64]

    def __init__(out self):
        self.weights = List[Float32](length=NE2G_K, fill=0)
        self.means = List[Float32](length=NE2G_K * NE2G_FEATURES, fill=0)
        self.covariances = List[Float32](length=NE2G_K * NE2G_FEATURES * NE2G_FEATURES, fill=0)
        self.precisions = List[Float32](length=NE2G_K * NE2G_FEATURES * NE2G_FEATURES, fill=0)
        self.log_det = List[Float32](length=NE2G_K, fill=0)
        self.scalars = List[Float64](length=3, fill=0)

    def fit(mut self, driver: GmmDriver, x: List[Float32], labels: Int) raises -> Int:
        var covariance_type = String("full")
        var init_params = String("kmeans")
        var n_iter = driver.fit(GmmFitRequest(Int(x.unsafe_ptr()), Int(self.weights.unsafe_ptr()),
            Int(self.means.unsafe_ptr()), Int(self.covariances.unsafe_ptr()), Int(self.precisions.unsafe_ptr()),
            Int(self.log_det.unsafe_ptr()), Int(self.scalars.unsafe_ptr()), labels,
            Int(covariance_type.as_c_string_slice().unsafe_ptr()), Int(init_params.as_c_string_slice().unsafe_ptr()),
            NE2G_ROWS, NE2G_FEATURES, NE2G_K, 100, 0, 1e-3, 1e-6))
        _ = covariance_type
        _ = init_params
        _ = x
        return n_iter


def fixture(driver: GmmDriver, unit: Int) raises:
    require(unit == 0, "GMM fit_predict fixture id")
    require((driver.metadata(MIXTURE_NATIVE_CAPABILITIES) & MIXTURE_CAP_FIT_PREDICT) != 0,
            "GPU GMM provider writes fit_predict labels")
    var x = two_groups()
    var fused = List[Int32](length=NE2G_ROWS, fill=NE2G_SENTINEL)
    var first = GmmState()
    _ = first.fit(driver, x, Int(fused.unsafe_ptr()))
    var second = GmmState()
    var n_iter = second.fit(driver, x, 0)
    var predicted = List[Int32](length=NE2G_ROWS, fill=NE2G_SENTINEL)
    driver.score(MIXTURE_NATIVE_PREDICT, GmmScoreRequest(Int(second.weights.unsafe_ptr()),
        Int(second.means.unsafe_ptr()), Int(second.covariances.unsafe_ptr()), Int(second.precisions.unsafe_ptr()),
        Int(second.log_det.unsafe_ptr()), Int(x.unsafe_ptr()), Int(predicted.unsafe_ptr()), NE2G_K,
        NE2G_FEATURES, n_iter, second.scalars[1] != 0, NE2G_ROWS, second.scalars[2]))
    var failures = List[Int](length=NE2G_ROWS, fill=0)
    var fp = failures.unsafe_ptr()
    var up = fused.unsafe_ptr()
    var pp = predicted.unsafe_ptr()
    def row(index: Int) {imm fp, imm up, imm pp}:
        var valid = up[index] >= Int32(0) and up[index] < Int32(NE2G_K)
        fp.unsafe_store(index, Int(not valid or up[index] != pp[index]))
    host_parallelize(row, NE2G_ROWS)
    for failure in failures:
        require(failure == 0, "GMM fit_predict labels differ from fit then predict")
    _ = fused^
    _ = predicted^
    _ = first^
    _ = second^
    _ = x^
