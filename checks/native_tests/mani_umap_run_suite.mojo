# SPDX-License-Identifier: Apache-2.0
"""UMAP door RUN (`x_ann/mani_umap_run.mojo`) without Python.

Each fixture is one typed check or route decision: accepted and refused
values, the refusal wording, the legacy-ABI choice and the transform's reuse
of the fitted embedding. Independent fixtures run across the host's cores.
"""
from core.host_parallel import host_parallelize
from core.cast_in import CAST_KIND_SIGNED, CAST_KIND_UNSIGNED, CAST_KIND_BOOL, CAST_KIND_FLOAT, CAST_KIND_BFLOAT
from x_ann.mani_umap_run import (
    UMAP_MODE_FAST, UMAP_MODE_IDENTICAL, umap_finite, umap_mode_code, umap_check_input, umap_check_neighbors,
    umap_check_spectral_samples, umap_check_pca_components, umap_check_singular_values, umap_check_spread,
    umap_init_shape_matches, umap_init_shape_refusal, umap_check_init_finite, umap_check_target_weight,
    umap_target_neighbors, umap_label_dtype_codable, umap_continuous_target_dims, umap_legacy_abi,
    umap_check_embedding, umap_check_transform_state, umap_check_transform_input, umap_transform_reuses_fit,
    umap_check_transform_output,
)

comptime MANI_UMAP_RUN_FIXTURES = 18


def require(ok: Bool, message: String) raises:
    if not ok:
        raise Error(message)


def infinite() -> Float32:
    # a runtime overflow, not a literal: +inf
    var big = Float32.MAX
    return big * 2


def fixture(unit: Int) raises:
    if unit == 0:
        var x: List[Float32] = [0.0, 1.0, 2.0, 3.0]
        umap_check_input(Int(x.unsafe_ptr()), 4)
        umap_check_init_finite(Int(x.unsafe_ptr()), 4)
        require(umap_finite(Int(x.unsafe_ptr()), 4) and umap_finite(0, 0), "finite words and an empty span")
        _ = x
        umap_check_neighbors(10, 10)
        umap_check_spectral_samples(False, 10, 3)
        umap_check_spectral_samples(True, 1, 3)
        umap_check_pca_components(2, 2, 4)
        umap_check_spread(1.0)
        umap_check_target_weight(0.0)
        umap_check_target_weight(1.0)
        require(umap_mode_code("fast") == UMAP_MODE_FAST and umap_mode_code("identical") == UMAP_MODE_IDENTICAL
                and umap_mode_code("Fast") == -1, "mode codes")
    elif unit == 1:
        try:
            umap_check_neighbors(11, 10)
            raise Error("accepted")
        except e:
            require(String(e) == "UMAP n_neighbors exceeds n_samples", "n_neighbors refusal wording")
    elif unit == 2:
        try:
            umap_check_spectral_samples(False, 9, 3)
            raise Error("accepted")
        except e:
            require(String(e) == "UMAP has too few samples for spectral initialization", "spectral refusal wording")
    elif unit == 3:
        try:
            umap_check_pca_components(3, 2, 5)
            raise Error("accepted")
        except e:
            require(String(e) == "UMAP init='pca' needs n_components <= min(n_samples, n_features)",
                    "pca components refusal wording")
    elif unit == 4:
        var s: List[Float32] = [2.0, 0.0]
        umap_check_singular_values(Int(s.unsafe_ptr()), 1)
        try:
            umap_check_singular_values(Int(s.unsafe_ptr()), 2)
            raise Error("accepted")
        except e:
            require(String(e) == "UMAP init='pca' found a zero singular value", "singular value refusal wording")
        _ = s
    elif unit == 5:
        try:
            umap_check_spread(0.0)
            raise Error("accepted")
        except e:
            require(String(e) == "UMAP init='pca' found a zero spread", "spread refusal wording")
    elif unit == 6:
        require(umap_init_shape_matches(4, 2, 4, 2), "matching init shape")
        require(not umap_init_shape_matches(2, 3, 4, 2), "mismatched init shape")
        require(String(umap_init_shape_refusal("(2, 3)", 4, 2)) == "UMAP init array has shape (2, 3); (4, 2) is needed",
                "init shape refusal wording")
    elif unit == 7:
        var x: List[Float32] = [0.0, infinite()]
        try:
            umap_check_init_finite(Int(x.unsafe_ptr()), 2)
            raise Error("accepted")
        except e:
            require(String(e) == "UMAP init array must be finite", "init finite refusal wording")
        try:
            umap_check_input(Int(x.unsafe_ptr()), 2)
            raise Error("accepted")
        except e:
            require(String(e) == "UMAP input coordinates must be finite", "input finite refusal wording")
        _ = x
    elif unit == 8:
        try:
            umap_check_target_weight(1.5)
            raise Error("accepted")
        except e:
            require(String(e) == "UMAP target_weight must be in [0, 1]", "target_weight refusal wording")
    elif unit == 9:
        require(umap_target_neighbors(-1) == 0 and umap_target_neighbors(2) == 2, "kept target_n_neighbors")
        for refused in range(2):
            try:
                _ = umap_target_neighbors(refused)
                raise Error("accepted")
            except e:
                require(String(e) == "UMAP target_n_neighbors must be -1 or >= 2", "target_n_neighbors refusal")
    elif unit == 10:
        require(umap_label_dtype_codable(CAST_KIND_FLOAT, 4) and umap_label_dtype_codable(CAST_KIND_FLOAT, 8),
                "float32/64 labels")
        require(not umap_label_dtype_codable(CAST_KIND_FLOAT, 2), "float16 labels keep the object path")
        require(umap_label_dtype_codable(CAST_KIND_SIGNED, 8) and umap_label_dtype_codable(CAST_KIND_BOOL, 1),
                "signed and bool labels")
        require(umap_label_dtype_codable(CAST_KIND_UNSIGNED, 4) and not umap_label_dtype_codable(CAST_KIND_UNSIGNED, 8),
                "uint64 labels keep the object path")
        require(not umap_label_dtype_codable(CAST_KIND_BFLOAT, 2), "bfloat16 labels keep the object path")
    elif unit == 11:
        var y: List[Float32] = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]
        var a = Int(y.unsafe_ptr())
        require(umap_continuous_target_dims(1, 6, 0, 6, a, 6) == 1, "1-D target width")
        require(umap_continuous_target_dims(2, 3, 2, 3, a, 6) == 2, "2-D target width")
        for arm in range(2):
            try:
                if arm == 0:
                    _ = umap_continuous_target_dims(3, 3, 2, 3, a, 6)
                else:
                    _ = umap_continuous_target_dims(2, 3, 2, 4, a, 6)
                raise Error("accepted")
            except e:
                require(String(e) == "UMAP continuous y must be finite with one row per sample",
                        "continuous target refusal wording")
        _ = y
    elif unit == 12:
        require(umap_legacy_abi(False, False, True, 2) and umap_legacy_abi(False, False, True, 3), "legacy ABI")
        require(not umap_legacy_abi(True, False, True, 2), "a start takes the extended ABI")
        require(not umap_legacy_abi(False, True, True, 2), "a target takes the extended ABI")
        require(not umap_legacy_abi(False, False, False, 2), "non-default extras take the extended ABI")
        require(not umap_legacy_abi(False, False, True, 4), "4 components take the extended ABI")
    elif unit == 13:
        var e_ok: List[Float32] = [0.0, 1.0]
        umap_check_embedding(True, Int(e_ok.unsafe_ptr()), 2)
        var e_bad: List[Float32] = [0.0, infinite()]
        for arm in range(2):
            try:
                if arm == 0:
                    umap_check_embedding(False, Int(e_ok.unsafe_ptr()), 2)
                else:
                    umap_check_embedding(True, Int(e_bad.unsafe_ptr()), 2)
                raise Error("accepted")
            except e:
                require(String(e) == "RuntimeError: UMAP returned an invalid embedding", "embedding refusal wording")
        _ = e_ok
        _ = e_bad
    elif unit == 14:
        umap_check_transform_state(5, 2, 5, 2, 3, 3)
        try:
            umap_check_transform_state(4, 2, 5, 2, 3, 3)
            raise Error("accepted")
        except e:
            require(String(e) == "UMAP fitted embedding shape differs from fitted configuration",
                    "fitted embedding refusal wording")
        try:
            umap_check_transform_state(5, 2, 5, 2, 4, 3)
            raise Error("accepted")
        except e:
            require(String(e) == "UMAP transform feature count differs from fitted data", "feature count refusal wording")
    elif unit == 15:
        var x: List[Float32] = [infinite()]
        try:
            umap_check_transform_input(Int(x.unsafe_ptr()), 1)
            raise Error("accepted")
        except e:
            require(String(e) == "UMAP transform input coordinates must be finite", "transform input refusal wording")
        _ = x
    elif unit == 16:
        var a: List[Float32] = [1.0, 2.0, 3.0, 4.0]
        var b: List[Float32] = [1.0, 2.0, 3.0, 4.0]
        var c: List[Float32] = [1.0, 2.0, 3.0, 5.0]
        require(umap_transform_reuses_fit(Int(a.unsafe_ptr()), Int(b.unsafe_ptr()), 2, 2, 2), "same bytes reuse the fit")
        require(not umap_transform_reuses_fit(Int(a.unsafe_ptr()), Int(c.unsafe_ptr()), 2, 2, 2), "different bytes run")
        require(not umap_transform_reuses_fit(Int(a.unsafe_ptr()), Int(b.unsafe_ptr()), 1, 2, 2), "different rows run")
        require(umap_transform_reuses_fit(0, 0, 2, 2, 0), "zero features with the fitted row count reuse the fit")
        _ = a
        _ = b
        _ = c
    else:
        var o: List[Float32] = [0.0, 1.0]
        umap_check_transform_output(True, Int(o.unsafe_ptr()), 2)
        try:
            umap_check_transform_output(False, Int(o.unsafe_ptr()), 2)
            raise Error("accepted")
        except e:
            require(String(e) == "RuntimeError: UMAP transform returned an invalid embedding",
                    "transform output refusal wording")
        _ = o


def run_checks() raises:
    var failures = List[Int](length=MANI_UMAP_RUN_FIXTURES, fill=0)
    var fp = failures.unsafe_ptr()
    def task(unit: Int) {imm fp}:
        try:
            fixture(unit)
        except error:
            fp.unsafe_store(unit, 1)
            print("FAIL mani_umap_run fixture", unit, String(error))
    host_parallelize(task, MANI_UMAP_RUN_FIXTURES)
    var failed = 0
    for value in failures:
        failed += value
    _ = failures^
    require(failed == 0, "mani_umap_run failed " + String(failed) + " fixtures")
    print("PASS mani_umap_run:", MANI_UMAP_RUN_FIXTURES, "typed check and route fixtures")
