# SPDX-License-Identifier: Apache-2.0
"""Typed rules of the seven shard drivers (RUN): the numeric-mode, fitted,
width, feature-count and shape refusals of the RBFSampler, scaler, Gaussian
process classifier, forest, reference-shard, parallel-query and distributed
IVF drivers, the reference shard width plan under the signed-int32 shape
contract, and the status words of the native merges.
`bindings/parallel_shard_native_door.mojo` unpacks the Python values into
these arguments, runs the shards on the process pool (one worker per device)
and packs the result; no Python object reaches this file. Every function is
a word or text comparison made once per driver call or once per shard reply;
the partition, the device index admission and the block copies are in
`parallel_shard_common_run.mojo`."""

comptime SHARD_INT32_LIMIT = 2147483647


# ------------------------------------------------------------ shared rules
def shard_mode_identical(present: Bool, exact: Bool, word: String) -> Bool:
    """An estimator's own numeric-mode word admits the identical drivers when
    it is absent or the exact text `identical`."""
    return not present or (exact and word == "identical")


def shard_identical_admit(identical: Bool, refusal: String) raises:
    if not identical:
        raise Error("ValueError: " + refusal)


def shard_fitted_admit(fitted: Bool, refusal: String) raises:
    if not fitted:
        raise Error("ValueError: " + refusal)


def shard_positive_admit(exact: Bool, value: Int, refusal: String) raises:
    """`exact`: the value is an exact int (`value` is read only then)."""
    if not exact or value < 1:
        raise Error("ValueError: " + refusal)


def shard_features_admit(given: Int, fitted: Int) raises:
    if given != fitted:
        raise Error("ValueError: input feature count differs from fit")


def shard_envelope_admit(fields: Int) raises:
    if fields != 3:
        raise Error("parallel shards: invalid worker request envelope")


# ------------------------------------------------------------ RBFSampler
def rbf_features_admit(given: Int, fitted: Int) raises:
    if given != fitted:
        raise Error("ValueError: X has " + String(given) + " features, the fit had " + String(fitted))


def rbf_part_admit(rank: Int, columns: Int, components: Int) raises:
    """One worker block (`columns` is read only for a rank of 2)."""
    if rank != 2 or columns != components:
        raise Error("ValueError: RBFSampler worker returned an invalid result shape")


# ------------------------------------------------------------ scalers
def scaler_input_admit(listed: Bool, dtype: String, rank: Int, rows: Int, columns: Int) raises:
    """`rows` and `columns` are read only for a rank of 2."""
    if not listed and dtype != "<f4":
        raise Error("TypeError: Scaler input must have dtype float32")
    if rank != 2 or rows == 0 or columns == 0:
        raise Error("ValueError: Scaler requires a nonempty two-dimensional input")
    if rows * columns > SHARD_INT32_LIMIT:
        raise Error("ValueError: Scaler exceeds the native Int32 indexing bound")


def scaler_weight_admit(given: Bool) raises:
    if given:
        raise Error("NotImplementedError: scalers do not implement sample_weight")


def scaler_fit_mode_admit(exact: Bool, word: String) raises:
    """The fitted mode word is present, exact text and `identical`."""
    if not exact or word != "identical":
        raise Error("ValueError: parallel scaler transforms require an IDENTICAL fit")


def scaler_block_admit(rows: Int, columns: Int, expected_rows: Int, expected_columns: Int) raises:
    """`columns` is read only when the rows agree."""
    if rows != expected_rows or columns != expected_columns:
        raise Error("ValueError: scaler worker returned an invalid block shape")


# ------------------------------------------------------------ Gaussian process classifier
def gpc_mode_kind_admit(absent: Bool, exact: Bool) raises:
    if not absent and not exact:
        raise Error("ValueError: parallel GPC requires IDENTICAL numeric mode")


def gpc_mode_defaults(absent: Bool, word: String) -> Bool:
    """An absent or empty own word takes the process default mode."""
    return absent or word == ""


def gpc_mode_admit(exact: Bool, word: String) raises:
    if not exact or word != "identical":
        raise Error("ValueError: parallel GPC requires IDENTICAL numeric mode")


def gpc_labels_admit(codes: Int, rows: Int, classes: Int) raises:
    if codes != rows:
        raise Error("ValueError: y length differs from X rows")
    if classes < 2:
        raise Error("ValueError: GaussianProcessClassifier requires at least two classes")


def gpc_method_admit(exact: Bool, method: String) raises:
    if not exact or (method != "predict" and method != "predict_proba"):
        raise Error("ValueError: method must be predict or predict_proba")


def gpc_features_admit(given: Int, fitted: Int) raises:
    if given != fitted:
        raise Error("ValueError: GaussianProcessClassifier: X has " + String(given) + " features, fit saw " + String(fitted))


def gpc_rows_admit(rows: Int, expected: Int) raises:
    if rows != expected:
        raise Error("ValueError: GPC worker returned an invalid row count")


# ------------------------------------------------------------ forest
def forest_count_admit(count: Int) raises:
    if count < 1:
        raise Error("ValueError: n_estimators must be positive")


def forest_offsets_admit(dtype: String) raises:
    if dtype != "<i4":
        raise Error("ValueError: forest worker returned non-int32 node offsets")


# ------------------------------------------------------------ reference shards
def reference_algorithm_admit(exact: Bool, algorithm: String) raises:
    if not exact or (algorithm != "auto" and algorithm != "brute"):
        raise Error("ValueError: reference sharding requires the IDENTICAL brute-force arm")


def reference_knn_admit(identical: Bool, exact: Bool, algorithm: String) raises:
    if not identical or not exact or (algorithm != "auto" and algorithm != "brute"):
        raise Error("ValueError: reference shards require IDENTICAL brute-force KNN")


def reference_k_admit(exact: Bool, k: Int, rows: Int) raises:
    """`exact`: k is an exact int (`k` is read only then)."""
    if not exact or k < 1 or k > rows:
        raise Error("ValueError: n_neighbors must be an integer in [1, reference rows]")


def reference_cell_limit(rows: Int, features: Int) raises -> Int:
    """The most rows one shard may hold so that rows * features stays a
    signed int32 (the native shape contract of the brute-force kernels)."""
    if rows > SHARD_INT32_LIMIT:
        raise Error("ValueError: reference row count exceeds the native signed-int32 shape contract")
    var cell_limit = SHARD_INT32_LIMIT // features
    if cell_limit < 1:
        raise Error("ValueError: feature count exceeds the native signed-int32 shape contract")
    return cell_limit


def reference_width(cell_limit: Int, rows: Int, devices: Int, given: Bool, width: Int) raises -> Int:
    """Rows per reference shard: the caller's width, or an even split over
    the devices capped by the cell limit; a shard above the limit is refused."""
    var chosen = width if given else min(cell_limit, (rows + devices - 1) // devices)
    if min(chosen, rows) > cell_limit:
        raise Error("ValueError: reference shard exceeds the native signed-int32 cell-count contract")
    return chosen


def reference_ids_admit(dtype: String) raises:
    """A shard's ids that are not int64 are widened from int32 or uint32."""
    if dtype != "<i4" and dtype != "<u4":
        raise Error("ValueError: reference shard ids require int32, uint32 or int64, got " + dtype)


def reference_merge_admit(status: Int) raises:
    if status == 1:
        raise Error("RuntimeError: reference worker returned an invalid neighbor index")
    if status == 2:
        raise Error("RuntimeError: reference shards returned too few neighbors")


# ------------------------------------------------------------ distributed IVF
def ivf_open_admit(closed: Bool) raises:
    if closed:
        raise Error("RuntimeError: distributed IVF index is closed")


def ivf_offsets_admit(status: Int) raises:
    if status == 1:
        raise Error("ValueError: IVF original IDs must be a permutation of stored row IDs")
    if status != 0:
        raise Error("ValueError: invalid global IVF list offsets")


def ivf_receipt_admit(stored: Int, rows: Int) raises:
    if stored != rows:
        raise Error("ValueError: IVF storage receipts differ from shard sizes")


def ivf_queries_admit(rows: Int) raises:
    if rows < 1:
        raise Error("ValueError: at least one query is required")


def ivf_features_admit(given: Int, fitted: Int) raises:
    if given != fitted:
        raise Error("ValueError: query features differ from index")


def ivf_parts_admit(parts: Int, shards: Int) raises:
    if parts != shards:
        raise Error("ValueError: incomplete IVF result shards")


def ivf_merge_admit(status: Int) raises:
    if status == 0:
        return
    if status == 1:
        raise Error("ValueError: invalid IVF candidate count")
    if status == 2:
        raise Error("ValueError: invalid IVF local row id")
    if status == 4:
        raise Error("ValueError: NaN distance in an IVF shard result")
    raise Error("ValueError: IVF shard merge failed")
