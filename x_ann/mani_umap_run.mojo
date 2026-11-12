# SPDX-License-Identifier: Apache-2.0
"""UMAP door RUN: the checks on typed values and the typed route decisions.

`bindings/mani_umap_native.mojo` is the UNPACK/PACK side: it reads the
estimator's fields, admits the arrays, builds the start with the decomp kit,
calls the compiled UMAP entries and publishes the fitted attributes. It calls
these checks at the point the public method made them, so the refusal order
and wording are unchanged. Nothing here imports Python.
"""
from core.host_finite_scan import host_all_finite_f32
from core.mani_impl import mani_same_words
from core.cast_in import CAST_KIND_SIGNED, CAST_KIND_UNSIGNED, CAST_KIND_BOOL, CAST_KIND_FLOAT

comptime _F32P = MutPointer[Float32, MutUntrackedOrigin]
comptime _U32P = MutPointer[UInt32, MutUntrackedOrigin]
comptime UMAP_MODE_FAST = 0
comptime UMAP_MODE_IDENTICAL = 1


def umap_finite(address: Int, count: Int) -> Bool:
    return count == 0 or host_all_finite_f32(_F32P(unsafe_from_address=address), count)


def umap_mode_code(mode: String) -> Int:
    """`{"fast": 0, "identical": 1}[mode]`; -1 for any other word (the door
    raises the KeyError with the word's repr)."""
    if mode == "fast":
        return UMAP_MODE_FAST
    if mode == "identical":
        return UMAP_MODE_IDENTICAL
    return -1


def umap_check_input(address: Int, count: Int) raises:
    if not umap_finite(address, count):
        raise Error("UMAP input coordinates must be finite")


def umap_check_neighbors(n_neighbors: Int, n: Int) raises:
    if n_neighbors > n:
        raise Error("UMAP n_neighbors exceeds n_samples")


def umap_check_spectral_samples(has_initial: Bool, n: Int, nc: Int) raises:
    if not has_initial and n < 2 * nc + 4:
        raise Error("UMAP has too few samples for spectral initialization")


def umap_check_pca_components(nc: Int, rows: Int, cols: Int) raises:
    if nc > min(rows, cols):
        raise Error("UMAP init='pca' needs n_components <= min(n_samples, n_features)")


def umap_check_singular_values(address: Int, nc: Int) raises:
    """The first nc singular values of the wide-data SVD are positive."""
    var sp = _F32P(unsafe_from_address=address)
    for j in range(nc):  # small-loop(components: the zero singular value check): bounded by n_components
        if not (Float64(sp[j]) > 0):
            raise Error("UMAP init='pca' found a zero singular value")


def umap_check_spread(peak: Float64) raises:
    if not (peak > 0):
        raise Error("UMAP init='pca' found a zero spread")


def umap_init_shape_matches(rows: Int, cols: Int, n: Int, nc: Int) -> Bool:
    return rows == n and cols == nc


def umap_init_shape_refusal(shape_repr: String, n: Int, nc: Int) -> Error:
    return Error("UMAP init array has shape " + shape_repr + "; (" + String(n) + ", " + String(nc)
                 + ") is needed")


def umap_check_init_finite(address: Int, count: Int) raises:
    if not umap_finite(address, count):
        raise Error("UMAP init array must be finite")


def umap_check_target_weight(weight: Float64) raises:
    if not (weight >= 0 and weight <= 1):
        raise Error("UMAP target_weight must be in [0, 1]")


def umap_target_neighbors(neighbors: Int) raises -> Int:
    """The checked target_n_neighbors kept for the entry: `max(tk, 0)`."""
    if neighbors == 0 or neighbors == 1:
        raise Error("UMAP target_n_neighbors must be -1 or >= 2")
    return max(neighbors, 0)


def umap_label_dtype_codable(kind: Int, itemsize: Int) -> Bool:
    """A 1-D label source whose words one native pass codes: float32/64, any
    signed integer, bool, or an unsigned integer narrower than 8 bytes."""
    if kind == CAST_KIND_FLOAT:
        return itemsize == 4 or itemsize == 8
    return kind == CAST_KIND_SIGNED or kind == CAST_KIND_BOOL or (kind == CAST_KIND_UNSIGNED and itemsize < 8)


def umap_continuous_target_dims(ndim: Int, rows: Int, cols: Int, n: Int, address: Int, count: Int) raises -> Int:
    """The target's width (1 for a 1-D y) after its shape and finite checks."""
    var dims = 1 if ndim == 1 else cols
    if (ndim != 1 and ndim != 2) or rows != n or not umap_finite(address, count):
        raise Error("UMAP continuous y must be finite with one row per sample")
    return dims


def umap_legacy_abi(has_initial: Bool, has_target: Bool, default_extras: Bool, nc: Int) -> Bool:
    """Default controls with a 2-D or 3-D spectral start keep the legacy
    `umap_fit_transform` ABI."""
    return not has_initial and not has_target and default_extras and (nc == 2 or nc == 3)


def umap_check_embedding(columns_match: Bool, address: Int, count: Int) raises:
    if not columns_match or not umap_finite(address, count):
        raise Error("RuntimeError: UMAP returned an invalid embedding")


def umap_check_transform_state(embedding_rows: Int, embedding_cols: Int, training_rows: Int, nc: Int,
                               features: Int, training_features: Int) raises:
    if embedding_rows != training_rows or embedding_cols != nc:
        raise Error("UMAP fitted embedding shape differs from fitted configuration")
    if features != training_features:
        raise Error("UMAP transform feature count differs from fitted data")


def umap_check_transform_input(address: Int, count: Int) raises:
    if not umap_finite(address, count):
        raise Error("UMAP transform input coordinates must be finite")


def umap_transform_reuses_fit(x_address: Int, training_address: Int, queries: Int, training_rows: Int,
                              features: Int) -> Bool:
    """`x.tobytes() == training.tobytes()` for same-shape X: the fitted
    embedding is returned as is."""
    if queries != training_rows:
        return False
    if queries * features == 0:
        return True
    return mani_same_words(_U32P(unsafe_from_address=x_address), _U32P(unsafe_from_address=training_address),
                           queries * features)


def umap_check_transform_output(columns_match: Bool, address: Int, count: Int) raises:
    if not columns_match or not umap_finite(address, count):
        raise Error("RuntimeError: UMAP transform returned an invalid embedding")
