# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Loading the benchmark fixtures, shared by every driver in this directory.

Both arms of every comparison read the SAME bytes off disk, and these are the
functions the Mojo side reads them with. `sklearn_arm.py` takes the same
subset the same way on the scikit-learn side.

Nothing here is timed and nothing here is a port -- it is fixture handling, in
the same category as `checks/fixtures.mojo`.
"""

from extratrees.estimator import (
    count_to_ratio,
    MAX_FEATURES_ALL,
    MAX_FEATURES_LOG2,
    MAX_FEATURES_SQRT,
)


def read_f32(path: String) raises -> List[Float32]:
    """A whole small file as `Float32`. Only the label vector is read this way.
    """
    var f = open(path, "r")
    var b = f.read_bytes()
    f.close()
    var n = len(b) // 4
    var out = List[Float32](length=n, fill=Float32(0.0))
    var src = b.unsafe_ptr().unsafe_bitcast[Float32]()
    for i in range(n):
        out[i] = src[unsafe_offset=i]
    _ = b.unsafe_ptr()
    return out^


def read_column_prefix(
    path: String, total_rows: Int, n_rows: Int, n_features: Int
) raises -> List[Float32]:
    """The first `n_rows` of every column, column-major, WITHOUT reading the
    rest of the file.

    The obvious version reads the whole matrix and then subsets it, and at
    epsilon's 3.2 GB that is 6.4 GB resident before the subset even exists, on
    a 16 GB box that is also holding scikit-learn's copy. Seeking per column
    reads exactly what is asked for. `sklearn_arm.load` takes the same subset
    the same way -- `[:, :n_rows]` on the (n_features, total_rows) view, off a
    memmap -- so both arms see the same matrix, not merely the same
    distribution.
    """
    var f = open(path, "r")
    var out = List[Float32](length=n_rows * n_features, fill=Float32(0.0))
    for c in range(n_features):
        _ = f.seek(c * total_rows * 4)
        var b = f.read_bytes(n_rows * 4)
        if len(b) != n_rows * 4:
            f.close()
            raise Error(
                path
                + ": short read on column "
                + String(c)
                + " -- wanted "
                + String(n_rows * 4)
                + " bytes, got "
                + String(len(b))
            )
        var p = b.unsafe_ptr().unsafe_bitcast[Float32]()
        var dst = c * n_rows
        for r in range(n_rows):
            out[dst + r] = p[unsafe_offset=r]
        _ = b.unsafe_ptr()
    f.close()
    return out^


def dense_class_ids(
    y: List[Float32], n_rows: Int, n_classes: Int
) raises -> List[Float32]:
    """The labels as `0 .. n_classes-1`, by ASCENDING order of distinct value.

    THE MAPPING IS COMPUTED, NOT ASSUMED. covtype's fixture is 1-based and
    epsilon's is `-1 / +1`; a hardcoded `- 1.0` relabels one of them and a
    `- min` turns epsilon into class ids `0` and `2`, which the port's
    `class_ids_for` refuses by name and scikit-learn would silently accept as
    three classes with one empty. `numpy.unique(y, return_inverse=True)` is
    the same function on the scikit-learn side, and it sorts too, so the two
    arms agree on which original label is class 0 rather than merely on how
    many classes there are.
    """
    var uniq = List[Float32]()
    for r in range(n_rows):
        var v = y[r]
        var found = False
        for i in range(len(uniq)):
            if uniq[i] == v:
                found = True
                break
        if not found:
            uniq.append(v)
            if len(uniq) > n_classes:
                raise Error(
                    "the label column has more than "
                    + String(n_classes)
                    + " distinct values"
                )
    for i in range(1, len(uniq)):
        var v = uniq[i]
        var j = i
        while j > 0 and uniq[j - 1] > v:
            uniq[j] = uniq[j - 1]
            j -= 1
        uniq[j] = v
    var out = List[Float32](length=n_rows, fill=Float32(0.0))
    for r in range(n_rows):
        for i in range(len(uniq)):
            if uniq[i] == y[r]:
                out[r] = Float32(i)
                break
    return out^


def row_major(
    col: List[Float32], n_rows: Int, n_features: Int
) raises -> List[Float32]:
    """A row-major copy, for `predict_class_forest`'s `(row, row_offset)`."""
    var out = List[Float32](length=n_rows * n_features, fill=Float32(0.0))
    for c in range(n_features):
        for r in range(n_rows):
            out[r * n_features + c] = col[c * n_rows + r]
    return out^


def all_digits(s: String) -> Bool:
    """Whether every byte of `s` is an ASCII digit.

    The trailing options are order-independent, so `10` (a rep count) has to
    be told apart from `sqrt` and from `k27` without positions to lean on.
    """
    if s.byte_length() == 0:
        return False
    for i in range(s.byte_length()):
        if String("0123456789").find(String(s[byte=i])) < 0:
            return False
    return True


def max_features_code(spec: String, n_features: Int) raises -> Int:
    """`spec` as `resolve_max_features` takes it.

    `sqrt`, `log2` and `all` are scikit-learn's own three; `k<N>` is an
    integer COUNT, which both libraries take literally and which is what makes
    the sampled-column sweep possible on one dataset with nothing else moving.
    """
    if spec == String("sqrt"):
        return MAX_FEATURES_SQRT
    if spec == String("log2"):
        return MAX_FEATURES_LOG2
    if spec == String("all"):
        return MAX_FEATURES_ALL
    var parts = spec.split("k")
    if len(parts) == 2 and String(parts[0]).byte_length() == 0:
        var k = Int(String(parts[1]))
        if k < 1 or k > n_features:
            raise Error(
                "max_features k"
                + String(k)
                + " is outside [1, "
                + String(n_features)
                + "]"
            )
        return k
    raise Error(
        "max_features must be sqrt, log2, all or k<N>; got " + spec
    )
