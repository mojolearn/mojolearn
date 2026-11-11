# SPDX-License-Identifier: Apache-2.0
"""RUN pieces of the linear expansion door (bindings/x_linear_native.mojo):
the typed calls of the x_linear operations entry, the fit request words,
and the byte copies that assemble and slice its float32 words. Typed Mojo
values and addresses only, no Python import (tools/python_free_dirs.sh).
The door's UNPACK admits buffers and allocates the outputs; it releases the
GIL around each operations call exactly where it did before the split; its
PACK publishes the fitted attributes (lane doorsplit-linear, 2026-11-11).
Same operands in the same order, so the bits are unchanged."""
from std.memory import unsafe_memcpy
from core.native_descriptor_abi import NativeDescriptorCall
from core.native_descriptor_call import native_descriptor_call
from core.linear_entry_table_ext import XL_OP_FIT, XL_OP_FIT_FOLD
from core.admitted_refusals import admitted_finite_refuse
from core.entry_table import EntryRef
from core.base_helpers_calls import (
    base_assign_fold_i64, base_check_indices_i64, base_count_fold_hits_i64, base_mark_split_roles_f32,
)

comptime _F32P = MutPointer[Float32, MutUntrackedOrigin]
comptime _I32P = MutPointer[Int32, MutUntrackedOrigin]

# The x_linear fit routes (the `algo` word of an XL_OP_FIT request).
comptime ALGO_SGD = 1
comptime ALGO_GLM = 2
comptime ALGO_HUBER = 3
comptime ALGO_BAYES = 4
comptime ALGO_ARD = 5
comptime ALGO_LARS = 6
comptime ALGO_QUANTILE = 7
comptime ALGO_RIDGE = 8
comptime ALGO_ENETCV = 9
comptime ALGO_LOGCV = 10
comptime ALGO_ISOTONIC = 11
comptime ALGO_ISOTONIC_PREDICT = 12
comptime ALGO_RIDGE_KFOLD = 13
# The decision entries' link words.
comptime LINK_IDENTITY = 0
comptime LINK_EXP = 1
comptime LINK_SIGMOID = 2
# L-BFGS history length of the Huber route's work estimate.
comptime XL_LBFGS_M = 10


def xl_run_call(entry: NativeDescriptorCall, operation: Int, addresses: List[Int], ints: List[Int],
                floats: List[Float64]) raises -> Int:
    """One synchronous call of the x_linear operations entry. The caller
    keeps every addressed buffer alive (and releases the GIL around it)."""
    return native_descriptor_call(entry, operation, addresses, ints, floats)


def xl_run_fit_ints(header: List[Int], ip: List[Int]) -> List[Int]:
    """A fit request's Int words: its fixed header, then the route's
    integer hyperparameters (core/linear_entry_table_ext.mojo)."""
    var words = header.copy()
    for i in range(len(ip)):  # driver-loop: over the route's integer hyperparameters (model structure)
        words.append(ip[i])
    return words^


def xl_run_fit(entry: NativeDescriptorCall, algo: Int, x_addr: Int, x_size: Int, n: Int, d: Int, y_addr: Int,
               y_size: Int, ip: List[Int], fp: List[Float64], n_out: Int, n_fw: Int, n_iw: Int,
               out_addr: Int) raises -> Int:
    """One XL_OP_FIT call into the `n_out` float32 words at `out_addr`.
    `y_addr` addresses `y_size` 32-bit target words (the caller substitutes
    one zero word when there are none)."""
    var header: List[Int] = [algo, n, d, x_size, y_size, n_out, max(n_fw, 1), max(n_iw, 1), len(ip), len(fp)]
    return native_descriptor_call(entry, XL_OP_FIT, [x_addr, y_addr, out_addr], xl_run_fit_ints(header, ip), fp)


def xl_run_fit_fold(entry: NativeDescriptorCall, algo: Int, base_id: Int, base_rows: Int, index_id: Int, n: Int,
                    d: Int, y_addr: Int, y_size: Int, ip: List[Int], fp: List[Float64], n_out: Int, n_fw: Int,
                    n_iw: Int, out_addr: Int) raises -> Tuple[Int, Int]:
    """One XL_OP_FIT_FOLD call: the fit of a model-selection fold whose
    rows the entry gathers on the device out of the resident base. Returns
    the fold's (finite, range) status words (0 = clear) for
    core/admitted_refusals.mojo door_rows_refuse."""
    var words = List[Int](length=2, fill=0)
    var header: List[Int] = [algo, base_id, base_rows, index_id, n, d, n * d, y_size, n_out, max(n_fw, 1),
                             max(n_iw, 1), len(ip), len(fp)]
    _ = native_descriptor_call(entry, XL_OP_FIT_FOLD, [y_addr, out_addr, Int(words.unsafe_ptr())],
                               xl_run_fit_ints(header, ip), fp)
    var status = (words[0], words[1])
    _ = words^
    return status


def xl_run_status_refuse(word_addr: Int, est: String) raises:
    """Refuse a non-finite X a decision entry found in its status word
    (0: no deferred word)."""
    if word_addr != 0:
        admitted_finite_refuse(Int(_I32P(unsafe_from_address=word_addr)[0]), est)


def xl_run_word(addr: Int, index: Int) -> Float32:
    """The float32 word `index` of a result vector."""
    return _F32P(unsafe_from_address=addr)[index]


def xl_run_copy(dst: Int, src: Int, offset: Int, count: Int):
    """`count` float32 words of `src` from word `offset` into `dst`."""
    if count > 0:
        unsafe_memcpy(dest=_F32P(unsafe_from_address=dst), src=_F32P(unsafe_from_address=src) + offset, count=count)


def xl_run_join(dst: Int, first: Int, first_count: Int, second: Int, second_count: Int):
    """`_concat_f32(a, b)`: a's then b's 32-bit words into `dst`."""
    var o = _F32P(unsafe_from_address=dst)
    if first_count > 0:
        unsafe_memcpy(dest=o, src=_F32P(unsafe_from_address=first), count=first_count)
    if second_count > 0:
        unsafe_memcpy(dest=o + first_count, src=_F32P(unsafe_from_address=second), count=second_count)


def xl_run_parts(dst: Int, addrs: List[Int], counts: List[Int]):
    """`_concat_f32_parts(parts)`: the parts' 32-bit words in order."""
    var o = _F32P(unsafe_from_address=dst)
    var at = 0
    for i in range(len(counts)):  # driver-loop: over the target vector's parts
        if counts[i] > 0:
            unsafe_memcpy(dest=o + at, src=_F32P(unsafe_from_address=addrs[i]), count=counts[i])
        at += counts[i]


def xl_parts_total(counts: List[Int]) -> Int:
    var total = 0
    for i in range(len(counts)):  # driver-loop: over the target vector's parts
        total += counts[i]
    return total


def xl_block_dims(coef_words: Int, intercept_words: Int) -> Tuple[Int, Int]:
    """(k, d) of `_coef_block`: k intercepts, d coefficients per class."""
    var k = intercept_words
    return (k, coef_words // max(k, 1))


def xl_run_block(dst: Int, coef: Int, intercept: Int, k: Int, d: Int):
    """`_coef_block(coef, intercept)`: [coef row | intercept] per class,
    k rows of d + 1 float32 words (byte copies)."""
    var o = _F32P(unsafe_from_address=dst)
    var cp = _F32P(unsafe_from_address=coef)
    var bp = _F32P(unsafe_from_address=intercept)
    for j in range(k):  # driver-loop: over the fitted classes' coefficient rows (model structure)
        unsafe_memcpy(dest=o + j * (d + 1), src=cp + j * d, count=d)
        unsafe_memcpy(dest=o + j * (d + 1) + d, src=bp + j, count=1)


def xl_seed_halves(lo: Int, hi: Int) -> Tuple[Int, Int]:
    """`_seed`: the low and high unsigned 32-bit halves of random_state's
    low 64 bits (UNPACK splits the Python integer) as signed Int32 words."""
    var half = 2147483648
    return (lo - 2 * half if lo >= half else lo, hi - 2 * half if hi >= half else hi)


def xl_lbfgs_work(p: Int) -> Int:
    """The Huber route's L-BFGS float work words for p parameters."""
    return 4 * p + 2 * XL_LBFGS_M * p + 2 * XL_LBFGS_M


def xl_sgd_class_floats(mut fp: List[Float64], class_weight: Int, n_classes: Int):
    """Append the SGD fit's class-weight parameters: (w1, w0) for two
    classes; the k positive weights then k ones for more."""
    var cwp = _F32P(unsafe_from_address=class_weight)
    if n_classes == 2:
        fp.append(Float64(cwp[1]))
        fp.append(Float64(cwp[0]))
    else:
        for c in range(n_classes):  # driver-loop: over the classes' positive weights (model structure)
            fp.append(Float64(cwp[c]))
        for c in range(n_classes):  # driver-loop: over the classes' negative weights (model structure)
            fp.append(1.0)


# ------------------------------------------------------------ cv splits


@fieldwise_init
struct XlSplitWords(Movable):
    """The int64 row indices of each cv split as addresses and counts (an
    address is 0 when its side is empty). UNPACK reads them off the
    split Arrays; the Arrays stay alive in the caller."""
    var train: List[Int]
    var train_sizes: List[Int]
    var test: List[Int]
    var test_sizes: List[Int]


def xl_cv_fold_ids_run(key: EntryRef, splits: XlSplitWords, n: Int, fold_ids: Int) raises -> Bool:
    """`_cv_fold_ids(splits, n)` on the base helper table: write each row's
    int32 fold into the n words at `fold_ids` (filled with -1 by the caller) and
    return True when the test sides partition the rows and every train side
    is the complement of its test side; False otherwise."""
    var covered = 0
    for k in range(len(splits.test)):  # driver-loop: over the cv splits
        var size = splits.test_sizes[k]
        if base_assign_fold_i64(key, splits.test[k] if size > 0 else 0, size, n, k, fold_ids) != 0:
            return False
        covered += size
    if covered != n:
        return False
    for k in range(len(splits.train)):  # driver-loop: over the cv splits
        var tr_size = splits.train_sizes[k]
        if tr_size != n - splits.test_sizes[k]:
            return False
        if tr_size > 0:
            if base_check_indices_i64(key, splits.train[k], tr_size, n) != 0:
                return False
            if base_count_fold_hits_i64(key, splits.train[k], tr_size, fold_ids, n, k) != 0:
                return False
    return True


def xl_cv_roles_limit(n: Int, name: String) raises:
    """Per-split roles are float32 row marks: n + 1 must stay exact."""
    if n + 1 > 16777216:
        raise Error("NotImplementedError: mojolearn " + name + ": cv splits that are not a partition are limited "
                    + "to 16,777,215 rows")


def xl_cv_roles_run(key: EntryRef, splits: XlSplitWords, n: Int, roles: Int, name: String) raises:
    """`_cv_roles_f32`: mark each split's train (1) and test (2) rows in its
    n float32 words of `roles` (x_linear/cv_roles.mojo), refusing an index
    outside the rows, a repeated row, or an empty side."""
    for k in range(len(splits.test)):  # driver-loop: over the cv splits
        for side in range(2):  # driver-loop: over the two sides of one split
            var size = splits.train_sizes[k] if side == 0 else splits.test_sizes[k]
            var rows = splits.train[k] if side == 0 else splits.test[k]
            var status = base_mark_split_roles_f32(key, rows if size > 0 else 0, size, n, k, side + 1, roles)
            if status == 1:
                raise Error("mojolearn " + name + ": a cv split holds an index outside [0, n_samples)")
            if status == 2:
                raise Error("mojolearn " + name + ": a cv split repeats a row, or puts a row on both sides")
        if splits.train_sizes[k] == 0 or splits.test_sizes[k] == 0:
            raise Error("mojolearn " + name + ": every cv split needs training and test rows")
