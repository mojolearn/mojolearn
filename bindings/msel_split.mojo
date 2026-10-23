# SPDX-License-Identifier: Apache-2.0
# The splitters of python/mojolearn/model_selection.py, natively (lane
# mcb2-msel): the default folds (`_native_default_folds`), every
# splitter's `split` and `get_n_splits`, `check_cv` and
# `train_test_split`. Each split makes the same helper calls the Python
# made, with the same arguments, and the same random draws in the same order
# (the x_metrics programs of `CounterRng`, `fold_rows` and
# `stratified_fold_rows`, rebuilt with the shared native `_Prog`), so every
# split holds the same rows. A splitter with one split per row or per
# combination (LeaveOneOut, LeavePOut, LeaveOneGroupOut, LeavePGroupsOut)
# returns a lazy `map` over the compiled door, one split per item, as the
# Python generator made them; every other returns its list of splits (the
# Python made them all up front, or in one device program). The public
# `split` stays a generator shell that `yield from`s the result, so its
# refusals are still raised at the first `next`.
from std.memory import memcpy
from std.os import getenv
from std.python import Python, PythonObject
from bindings.py_balanced import py_call, py_type, py_get, py_setitem, py_setattr, py_contains, py_fields
from bindings.classical_arrays import classical_wrap_array
from bindings.native_buffer_owner import native_storage_allocate
from bindings.family_common import (
    FAMILY_I64P, FAMILY_I32P, family_call, family_call_kw, family_isinstance, family_is_str, family_repr, family_str,
)
from bindings.metrics_reg_native import _Prog, _reg_run, _arena_base, _arena_int
from bindings.msel_common import (
    MSEL_NONE, MSEL_FOLD_ROWS_BOUND, msel_b, msel_entry, msel_try, msel_missed, msel_catch, msel_warn, msel_is_bool,
    msel_is_integral, msel_is_int_arg, msel_is_real, msel_int, msel_op, msel_cmp, msel_materialize,
    msel_try_materialize, msel_flatten, msel_encode, msel_labels_list, msel_n_samples, msel_is_array, msel_addr,
    msel_p64, msel_p32, msel_new, msel_new2, msel_i64_list, msel_i32_list, msel_read_i64, msel_slice_i64,
    msel_arange, msel_range_index, msel_split_mask, msel_split_indices, msel_index_copy, msel_as_index,
    MselGroups, MselRng, msel_mix_seed, msel_shuffle_sizes, msel_take_any, msel_lazy, msel_index_i64,
)


# ---------------------------------------------------------------- small pieces


def msel_cls(name: String, context: PythonObject) raises -> PythonObject:
    """A model_selection class connected at import."""
    return context["family_msel_classes"][PythonObject(name)]


def msel_is(model: PythonObject, name: String, context: PythonObject) raises -> Bool:
    return family_isinstance(model, msel_cls(name, context))


def msel_field(model: PythonObject, name: String) raises -> PythonObject:
    """`model.<name>` (the splitter's own attribute, AttributeError kept)."""
    return family_call(msel_b().getattr, model, PythonObject(name))


def msel_sabotage() -> Bool:
    """`_sabotage_requested()`: both variables set to 1."""
    return getenv("MOJOLEARN_FOLD_ORDER_SABOTAGE", "") == "1" and getenv("MOJOLEARN_HOST_ALLOW_SABOTAGE", "") == "1"


def msel_check_splits(n_splits: PythonObject, context: PythonObject) raises -> Int:
    """`_check_splits`."""
    if msel_is_bool(n_splits) or not msel_is_integral(n_splits, context):
        raise Error("The number of folds must be of Integral type. " + family_repr(n_splits) + " was passed.")
    if msel_cmp("le", n_splits, PythonObject(1)):
        raise Error("k-fold cross-validation requires at least one train/test split by setting "
                    + "n_splits=2 or more, got n_splits=" + family_str(n_splits) + ".")
    return msel_int(n_splits)


def msel_new_kfold(name: String, n_splits: PythonObject, shuffle: PythonObject, random_state: PythonObject,
                   context: PythonObject) raises -> PythonObject:
    """`KFold(n_splits, shuffle=..., random_state=...)` (or StratifiedKFold):
    `_KFoldBase.__init__`'s checks, then its three attributes."""
    var k = msel_check_splits(n_splits, context)
    if not msel_is_bool(shuffle):
        raise Error("TypeError: shuffle must be True or False; got " + family_str(shuffle))
    if not Bool(py=family_call(msel_b().bool, shuffle)) and random_state is not None:
        raise Error("Setting a random_state has no effect since shuffle is False. You should "
                    + "leave random_state to its default (None), or set shuffle=True.")
    var cv = native_storage_allocate(msel_cls(name, context))
    py_setattr(cv, "n_splits", PythonObject(k))
    py_setattr(cv, "shuffle", shuffle)
    py_setattr(cv, "random_state", random_state)
    return cv


def msel_truth(value: PythonObject) raises -> Bool:
    return Bool(py=family_call(msel_b().bool, value))


# ---------------------------------------------------------------- fold programs


def _fold_rows_run(mut prog: _Prog, n: Int, k: Int, code: Int, order: Int, context: PythonObject) raises -> PythonObject:
    """`_fold_rows_run`: the `fold_rows` stage, then each fold's (train,
    test) int64 rows read out of the arena."""
    var out_slot = prog.alloc(2 * n * k)
    var out = prog.want(out_slot, 2 * n * k)
    var sz_slot = prog.alloc(k)
    var sz = prog.want(sz_slot, k)
    prog.stage("fold_rows", 1, [n, k, code, order, out, sz])
    var arena = _reg_run(prog, PythonObject(None), context)
    var base = _arena_base(arena)
    var res = Python.list()
    for f in range(k):  # driver-loop: over folds
        var c = _arena_int(arena, sz + f)
        var at = out + 2 * n * f
        var test = msel_new(c, "<i8", context)
        var train = msel_new(n - c, "<i8", context)
        if c > 0:
            memcpy(dest=msel_p64(test), src=FAMILY_I64P(unsafe_from_address=base + 4 * at), count=c)
        if n - c > 0:
            memcpy(dest=msel_p64(train), src=FAMILY_I64P(unsafe_from_address=base + 4 * (at + 2 * c)), count=n - c)
        _ = py_call(res.append, Python.tuple(train, test))
    return res


def msel_fold_rows_shuffled(n: Int, k: Int, mut rng: MselRng, context: PythonObject) raises -> PythonObject:
    """`fold_rows(n, k, rng=rng)`; None past the table bound (no draw made)."""
    if n < 2 or k < 1 or 2 * n * k > MSEL_FOLD_ROWS_BOUND:
        return PythonObject(None)
    var prog = _Prog(context["family_metrics_ops"])
    var order = rng.permute_stage(prog, n)
    var code = prog.scratch(n)
    return _fold_rows_run(prog, n, k, code, order, context)


def msel_stratified_fold_rows(enc: PythonObject, counts: List[Int], alloc: PythonObject, k: Int,
                              mut rng: MselRng, shuffle: Bool, context: PythonObject) raises -> PythonObject:
    """`stratified_fold_rows`: StratifiedKFold's folds in one program; None
    past the table bound."""
    var n = Int(py=enc.size)
    var m = len(counts)
    if n < 2 or k < 1 or 2 * n * k > MSEL_FOLD_ROWS_BOUND:
        return PythonObject(None)
    var prog = _Prog(context["family_metrics_ops"])
    var enc_slot = prog.put_i32(enc)
    var off = prog.alloc(m + 1)
    var ordr = prog.scratch(n)
    prog.stage("group_sort", 1, [enc_slot, n, m, off, ordr])
    var pb = MSEL_NONE
    if shuffle:
        var at = 0
        for c in range(m):  # driver-loop: over classes (one permutation stage each)
            var o = rng.permute_stage(prog, counts[c])
            if c == 0:
                pb = o
            if o != pb + at:
                raise Error("AssertionError: stratified_fold_rows: the class permutations are not contiguous")
            at += counts[c]
    var cum = List[Int]()
    var table = msel_p64(alloc)
    for c in range(m):  # driver-loop: over classes (fold-by-class offsets)
        var acc = 0
        cum.append(0)
        for f in range(k):  # driver-loop: over folds
            acc += Int(table[f * m + c])
            cum.append(acc)
    var cum_slot = prog.put_i32(msel_i32_list(cum, context))
    var code = prog.scratch(n)
    prog.stage("strat_codes", n, [enc_slot, ordr, off, pb, cum_slot, k, code])
    return _fold_rows_run(prog, n, k, code, MSEL_NONE, context)


# ---------------------------------------------------------------- default folds


def msel_discrete_codes(y: PythonObject, context: PythonObject) raises -> PythonObject:
    """`_discrete_codes(y)`: (classes, int32 codes) when y is discrete, else
    None."""
    var b = msel_b()
    var arr = msel_try_materialize(y, "y", Python.tuple(b.TypeError, b.ValueError), context)
    var have = not msel_missed(arr, context)
    var flat = arr
    if have and Int(py=arr.ndim) != 1:
        flat = family_call(arr.reshape, Python.tuple(arr.size))
    if have:
        var dtype = String(py=arr.dtype)
        if (dtype == "<f4" or dtype == "<f8") and Int(py=flat.size) > 0:
            var integral = family_call(msel_entry("reduce_stat", context), PythonObject(msel_addr(flat)),
                                       context["family_msel_native_code"][PythonObject(dtype)], flat.size,
                                       context["family_msel_reduce_integral"])
            if not msel_truth(integral):
                return PythonObject(None)
    return msel_encode(flat if have else y, context)


def _select_folds(fold_store: PythonObject, fold_counts: PythonObject, n: Int, k: Int,
                  context: PythonObject) raises -> PythonObject:
    """`_select_folds`: each fold's (train, test) by `select_fold_i64`."""
    var select = msel_entry("select_fold_i64", context)
    var out = Python.list()
    var counts = msel_p64(fold_counts)
    for fold in range(k):  # driver-loop: over folds
        var size = Int(counts[fold])
        var test = msel_new(size, "<i8", context)
        var train = msel_new(n - size, "<i8", context)
        var scratch = msel_new(n, "<i8", context)
        var got = msel_int(family_call(select, PythonObject(msel_addr(fold_store)), PythonObject(n), PythonObject(fold),
                                       PythonObject(msel_addr(test) if size else msel_addr(scratch)),
                                       PythonObject(msel_addr(train) if n - size else msel_addr(scratch))))
        if got != size:
            raise Error("RuntimeError: mojolearn: select_fold_i64 disagrees with fold_ids")
        _ = py_call(out.append, Python.tuple(train, test))
    return out


def msel_default_folds(y: PythonObject, n_splits: PythonObject, classifier: Bool,
                       context: PythonObject) raises -> PythonObject:
    """`_native_default_folds(y, n_splits, classifier)`."""
    var n = Int(py=family_call(msel_b().len, y))
    if msel_is_bool(n_splits) or not msel_is_integral(n_splits, context) or msel_cmp("lt", n_splits, PythonObject(2)):
        raise Error("cv must specify at least two folds")
    if msel_cmp("gt", n_splits, PythonObject(n)):
        raise Error("cv cannot exceed the number of samples")
    var k = msel_int(n_splits)
    var codes = PythonObject(None)
    var n_classes = 0
    if classifier:
        var got = msel_discrete_codes(y, context)
        if got is not None:
            n_classes = len(got[0])
            codes = got[1]
    var fold_store = msel_new(n, "<i4", context)
    var fold_counts = msel_new(k, "<i8", context)
    var class_counts = msel_new(max(n_classes, 1), "<i8", context)
    _ = family_call(msel_entry("fold_ids", context), PythonObject(0 if codes is None else msel_addr(codes)),
                    PythonObject(n), PythonObject(n_classes), PythonObject(k), PythonObject(msel_addr(class_counts)),
                    PythonObject(msel_addr(fold_store)), PythonObject(msel_addr(fold_counts)))
    if codes is not None:
        var cc = msel_p64(class_counts)
        var most = Int(cc[0]) if n_classes else 0
        var least = Int(cc[0]) if n_classes else 0
        for c in range(n_classes):  # driver-loop: over classes (the fold count checks)
            most = max(most, Int(cc[c]))
            least = min(least, Int(cc[c]))
        if n_classes == 0:
            raise Error("max() iterable argument is empty")
        if most < k:
            raise Error("cv cannot exceed the number of members in every class")
        if least < k:
            msel_warn("The least populated class has fewer members than cv folds", msel_b().UserWarning)
    var out = _select_folds(fold_store, fold_counts, n, k, context)
    if msel_sabotage():
        # THE DORMANT NEGATIVE CONTROL (a verification switch, never set in
        # production): each nonempty fold's smallest test row moves to the
        # previous nonempty fold (the last fold takes fold 0's).
        var nonempty = List[Int]()
        var firsts = List[Int]()
        for f in range(k):  # driver-loop: over folds
            var test = out[f][1]
            if Int(py=test.size) > 0:
                nonempty.append(f)
                firsts.append(Int(msel_p64(test)[0]))
        var store = msel_p32(fold_store)
        for at in range(len(nonempty)):  # driver-loop: over nonempty folds (one row id moved each)
            store[firsts[(at + 1) % len(nonempty)]] = Int32(nonempty[at])
        out = _select_folds(fold_store, fold_counts, n, k, context)
    return out


# ---------------------------------------------------------------- first-seen codes


def msel_first_seen(y: PythonObject, context: PythonObject) raises -> PythonObject:
    """`_first_seen_native(y)`: (int32 codes in first-seen order, k, counts
    list), or None."""
    if y is None:
        return PythonObject(None)
    var b = msel_b()
    var classes = Python.tuple(b.TypeError, b.ValueError, b.OverflowError)
    var n = 0
    try:
        n = msel_n_samples(y)
        var gc = MselGroups(y, n, "y", context)
        if gc.m < 1:
            return PythonObject(None)
        var first = msel_new(gc.m, "<i8", context)
        _ = family_call(family_call(context["family_msel_xm"], PythonObject(None)).x_metrics_first_rows,
                        PythonObject(msel_addr(gc.codes)), PythonObject(n), PythonObject(gc.m),
                        PythonObject(msel_addr(first)))
        var keys = py_call(b.list, first._mv)
        var keywords = Python.dict()
        py_setitem(keywords, "key", py_call(b.getattr, keys, PythonObject("__getitem__")))
        var order = family_call_kw(b.sorted, keywords, py_call(b.range, PythonObject(gc.m)))
        var table = List[Int](length=gc.m, fill=0)
        var counts = Python.list()
        for rank in range(gc.m):  # driver-loop: over classes (first-seen order)
            var c = Int(py=order[rank])
            table[c] = rank
            _ = py_call(counts.append, PythonObject(gc.counts[c]))
        var words = gc.mapped(table, context)
        return Python.tuple(words, PythonObject(gc.m), counts)
    except error:
        if msel_catch(error, classes):
            return PythonObject(None)
        raise Error(String(error))


def _counts_list(counts: PythonObject) raises -> List[Int]:
    var out = List[Int]()
    for i in range(len(counts)):  # driver-loop: over classes
        out.append(Int(py=counts[i]))
    return out^


def _encode_first_seen(values: PythonObject, context: PythonObject) raises -> PythonObject:
    """`_encode_first_seen(values)`: (codes list, k) by dict.fromkeys order
    (the builtins' C loops over the label objects)."""
    var b = msel_b()
    var keys = py_call(b.list, py_call(b.dict.fromkeys, values))
    var index = py_call(b.dict, py_call(b.zip, keys, py_call(b.range, PythonObject(len(keys)))))
    var codes = family_call(b.list, py_call(b.map, py_call(b.getattr, index, PythonObject("__getitem__")), values))
    return Python.tuple(codes, PythonObject(len(index)))


# ---------------------------------------------------------------- KFold family


def msel_kfold_split(x: PythonObject, k: Int, shuffle: Bool, random_state: PythonObject,
                     context: PythonObject) raises -> PythonObject:
    """`KFold.split`."""
    var n = msel_n_samples(x)
    if k > n:
        raise Error("Cannot have number of splits n_splits=" + String(k) + " greater than the "
                    + "number of samples: n_samples=" + String(n) + ".")
    if shuffle:
        var rng = MselRng(random_state, context)
        var got = msel_fold_rows_shuffled(n, k, rng, context)
        if got is not None:
            return got
        var whole = rng.permutation_rows([n], context)[0]
        var out = Python.list()
        var start = 0
        for fold in range(k):  # driver-loop: over folds
            var size = n // k + (1 if fold < n % k else 0)
            _ = py_call(out.append, msel_split_indices(msel_addr(whole) + 8 * start, size, n, context))
            start += size
        _ = whole^
        return out
    return msel_default_folds(py_call(msel_b().range, PythonObject(n)), PythonObject(k), False, context)


def _fold_plan(y: PythonObject, k: Int, context: PythonObject) raises -> PythonObject:
    """`StratifiedKFold._fold_plan`: (enc, n_classes, counts, alloc)."""
    var fs = msel_first_seen(y, context)
    var enc = PythonObject(None)
    var counts = List[Int]()
    var n_classes = 0
    if fs is not None:
        enc = fs[0]
        n_classes = Int(py=fs[1])
        counts = _counts_list(fs[2])
    else:
        var got = msel_encode(y, context)
        var codes = got[1]
        var n = Int(py=codes.size)
        var width = max(len(got[0]), 1)
        enc = msel_new(n, "<i4", context)
        var cnt = msel_new(width, "<i8", context)
        n_classes = msel_int(family_call(msel_entry("first_seen_i32", context), PythonObject(msel_addr(codes)), PythonObject(n),
                                         PythonObject(width), PythonObject(msel_addr(enc)), PythonObject(msel_addr(cnt))))
        var all_counts = msel_read_i64(cnt)
        for c in range(n_classes):  # driver-loop: over classes
            counts.append(all_counts[c])
    var most = 0
    var least = 0
    for c in range(len(counts)):  # driver-loop: over classes (the fold count checks)
        most = counts[c] if c == 0 else max(most, counts[c])
        least = counts[c] if c == 0 else min(least, counts[c])
    if len(counts) == 0 or most < k:
        raise Error("n_splits=" + String(k) + " cannot be greater than the number of members in each class.")
    if least < k:
        msel_warn("The least populated class in y has only " + String(least) + " members, which is less "
                  + "than n_splits=" + String(k) + ".", msel_b().UserWarning)
    var alloc = msel_new2(k, n_classes, "<i8", context)
    var cnt_arr = msel_i64_list(counts, context)
    _ = family_call(msel_entry("strat_alloc_i64", context), PythonObject(msel_addr(cnt_arr)), PythonObject(n_classes),
                    PythonObject(k), PythonObject(msel_addr(alloc)))
    var counts_py = Python.list()
    for c in range(len(counts)):  # driver-loop: over classes
        _ = py_call(counts_py.append, PythonObject(counts[c]))
    return Python.tuple(enc, PythonObject(n_classes), counts_py, alloc)


def msel_stratified_split(y: PythonObject, k: Int, shuffle: Bool, random_state: PythonObject,
                          context: PythonObject) raises -> PythonObject:
    """`StratifiedKFold.split`."""
    var plan = _fold_plan(y, k, context)
    var enc = plan[0]
    var n_classes = Int(py=plan[1])
    var counts = _counts_list(plan[2])
    var alloc = plan[3]
    var rng = MselRng(random_state if shuffle else PythonObject(0), context)
    if k <= 256:
        var got = msel_stratified_fold_rows(enc, counts, alloc, k, rng, shuffle, context)
        if got is not None:
            return got
        if shuffle:
            rng = MselRng(random_state, context)
    # the same assignment by `strat_fold_assign_i32`, then each fold's rows
    var n = Int(py=enc.size)
    var cnt = msel_i64_list(counts, context)
    var perms = PythonObject(None)
    if shuffle:
        perms = msel_new(n, "<i8", context)
        var draws = rng.permutation_rows(counts, context)
        var at = 0
        for c in range(len(counts)):  # driver-loop: over classes (one draw each)
            if counts[c] > 0:
                memcpy(dest=msel_p64(perms) + at, src=msel_p64(draws[c]), count=counts[c])
            at += counts[c]
    var folds = msel_new(n, "<i4", context)
    _ = family_call(msel_entry("strat_fold_assign_i32", context), PythonObject(msel_addr(enc)), PythonObject(n),
                    PythonObject(n_classes), PythonObject(k), PythonObject(msel_addr(alloc)),
                    PythonObject(msel_addr(perms) if perms is not None and n > 0 else 0),
                    PythonObject(msel_addr(cnt)), PythonObject(msel_addr(folds)))
    var sizes = msel_new(k, "<i8", context)
    _ = family_call(msel_entry("bincount_i64", context), PythonObject(msel_addr(folds)), PythonObject(2), PythonObject(n),
                    PythonObject(k), PythonObject(msel_addr(sizes)), PythonObject(0))
    var select = msel_entry("select_fold_i64", context)
    var sz = msel_p64(sizes)
    var out = Python.list()
    for f in range(k):  # driver-loop: over folds
        var size = Int(sz[f])
        var test = msel_new(size, "<i8", context)
        var train = msel_new(n - size, "<i8", context)
        var scratch = msel_new(n, "<i8", context)
        _ = family_call(select, PythonObject(msel_addr(folds)), PythonObject(n), PythonObject(f),
                        PythonObject(msel_addr(test) if size else msel_addr(scratch)),
                        PythonObject(msel_addr(train) if n - size else msel_addr(scratch)))
        _ = py_call(out.append, Python.tuple(train, test))
    return out


def msel_group_kfold_split(x: PythonObject, groups: PythonObject, k: Int, shuffle: Bool,
                           random_state: PythonObject, context: PythonObject) raises -> PythonObject:
    """`GroupKFold.split`."""
    var n = msel_n_samples(x)
    var gc = MselGroups(groups, n, "groups", context)
    var m = gc.m
    if k > m:
        raise Error("Cannot have number of splits n_splits=" + String(k) + " greater than the "
                    + "number of groups: " + String(m) + ".")
    var perm = PythonObject(None)
    if shuffle:
        var rng = MselRng(random_state, context)
        perm = rng.permutation_rows([m], context)[0]
    var to_fold = msel_new(m, "<i4", context)
    var sizes = msel_new(k, "<i8", context)
    var counts = msel_i64_list(gc.counts, context)
    if perm is not None:
        _ = family_call(msel_entry("msel_group_fold_perm_i32", context), PythonObject(msel_addr(counts)), PythonObject(m),
                        PythonObject(k), PythonObject(msel_addr(perm)), PythonObject(msel_addr(to_fold)),
                        PythonObject(msel_addr(sizes)))
    else:
        _ = family_call(msel_entry("group_fold_assign_i32", context), PythonObject(msel_addr(counts)), PythonObject(m),  # cpu-route: sequential greedy fold plan over groups, not rows
                        PythonObject(k), PythonObject(0), PythonObject(msel_addr(to_fold)), PythonObject(msel_addr(sizes)))
    return gc.split_by_fold(to_fold, k, sizes, context)


def msel_strat_group_split(x: PythonObject, y: PythonObject, groups: PythonObject, k: Int, shuffle: Bool,
                           random_state: PythonObject, context: PythonObject) raises -> PythonObject:
    """`StratifiedGroupKFold.split`."""
    var n = msel_n_samples(x)
    var gc = MselGroups(groups, n, "groups", context)
    var fs = msel_first_seen(y, context)
    var yenc = PythonObject(None)
    var n_classes = 0
    if fs is not None:
        yenc = fs[0]
        n_classes = Int(py=fs[1])
    else:
        var got = _encode_first_seen(msel_labels_list(y, "y", context), context)
        var store = family_call(Python.import_module("array").array, PythonObject("i"), got[0])
        yenc = classical_wrap_array(store, Python.tuple(len(store)), "<i4", "C", context)
        n_classes = Int(py=got[1])
    if Int(py=yenc.size) != n:
        raise Error("y has " + String(Int(py=yenc.size)) + " labels for " + String(n) + " samples")
    var m = gc.m
    var perm = PythonObject(None)
    if shuffle:
        var rng = MselRng(random_state, context)
        perm = rng.permutation_rows([m], context)[0]
    var to_fold = msel_new(m, "<i4", context)
    var sizes = msel_new(k, "<i8", context)
    var dist = msel_new(m * n_classes, "<i8", context)
    _ = family_call(msel_entry("bincount2_i32", context), PythonObject(msel_addr(yenc)), PythonObject(msel_addr(gc.codes)),
                    py_call(msel_b().list, Python.tuple(n, n_classes, m)), PythonObject(msel_addr(dist)))
    var bad = msel_int(family_call(msel_entry("strat_group_plan_i32", context),  # cpu-route: sequential greedy fold plan over groups, not rows
                                   py_call(msel_b().list, Python.tuple(msel_addr(dist),
                                                                       msel_addr(perm) if perm is not None else 0,
                                                                       msel_addr(to_fold), msel_addr(sizes))),
                                   py_call(msel_b().list, Python.tuple(n_classes, m, k))))
    if bad:
        raise Error("n_splits=" + String(k) + " cannot be greater than the number of members in "
                    + "each class.")
    return gc.split_by_fold(to_fold, k, sizes, context)


# ---------------------------------------------------------------- the other splitters


def msel_time_series_split(model: PythonObject, x: PythonObject, context: PythonObject) raises -> PythonObject:
    """`TimeSeriesSplit.split`."""
    var n = msel_n_samples(x)
    var k = msel_int(msel_field(model, "n_splits"))
    var folds = k + 1
    var ts_field = msel_field(model, "test_size")
    var ts = msel_int(ts_field) if ts_field is not None else n // folds
    var gap = msel_int(msel_field(model, "gap"))
    var max_train_field = msel_field(model, "max_train_size")
    if folds > n:
        raise Error("Cannot have number of folds=" + String(folds) + " greater than the number of samples="
                    + String(n) + ".")
    if n - gap - ts * k <= 0:
        raise Error("Too many splits=" + String(k) + " for number of samples=" + String(n) + " with "
                    + "test_size=" + String(ts) + " and gap=" + String(gap) + ".")
    if ts == 0:
        raise Error("range() arg 3 must not be zero")
    var out = Python.list()
    var start = n - k * ts
    while start < n:  # driver-loop: over splits
        var end = start - gap
        var lo = 0
        if msel_truth(max_train_field):
            var mt = msel_int(max_train_field)
            if mt < end:
                lo = end - mt
        _ = py_call(out.append, Python.tuple(msel_range_index(lo, end, context),
                                             msel_range_index(start, start + ts, context)))
        start += ts
    return out


def msel_loo_split(x: PythonObject, model: PythonObject, context: PythonObject) raises -> PythonObject:
    """`LeaveOneOut.split`: one split per row, made on demand."""
    var n = msel_n_samples(x)
    if n <= 1:
        raise Error("Cannot perform LeaveOneOut with n_samples=" + String(n) + ".")
    var b = msel_b()
    var items = py_call(b.zip, py_call(Python.import_module("itertools").repeat, PythonObject(n)),
                        py_call(b.range, PythonObject(n)))
    return msel_lazy(model, "family_msel_split_loo_at", items, context)


def msel_loo_at(args: PythonObject, context: PythonObject) raises -> PythonObject:
    """Split s of LeaveOneOut over n rows: test row s, train every other."""
    var n = Int(py=args[0])
    var s = Int(py=args[1])
    var train = msel_new(n - 1, "<i8", context)
    var test = msel_new(1, "<i8", context)
    _ = family_call(msel_entry("leave_range_i64", context), PythonObject(n), PythonObject(s), PythonObject(s + 1),
                    PythonObject(msel_addr(train)), PythonObject(msel_addr(test)))
    return Python.tuple(train, test)


def msel_lpo_split(x: PythonObject, model: PythonObject, context: PythonObject) raises -> PythonObject:
    """`LeavePOut.split`: the combinations in lexicographic order (the
    order `next_combination_i64` stepped through), one split each, made on
    demand."""
    var n = msel_n_samples(x)
    var p_field = msel_field(model, "p")
    if msel_cmp("le", PythonObject(n), p_field):
        raise Error("p=" + family_str(p_field) + " must be strictly less than the number of samples=" + String(n))
    var p = msel_int(p_field)
    if p < 0:
        raise Error("r must be non-negative")
    if p < 1:
        var one = Python.list()
        _ = py_call(one.append, Python.tuple(msel_arange(0, n, context), msel_new(0, "<i8", context)))
        return one
    var b = msel_b()
    var itertools = Python.import_module("itertools")
    var combos = py_call(itertools.combinations, py_call(b.range, PythonObject(n)), PythonObject(p))
    var items = py_call(b.zip, py_call(itertools.repeat, PythonObject(n)), combos)
    return msel_lazy(model, "family_msel_split_lpo_at", items, context)


def msel_lpo_at(args: PythonObject, context: PythonObject) raises -> PythonObject:
    """One LeavePOut split: the combination's rows as the test set."""
    var n = Int(py=args[0])
    var combo = family_call(Python.import_module("array").array, PythonObject("q"), args[1])
    var k = len(combo)
    return msel_split_indices(Int(py=py_call(combo.buffer_info)[0]) if k else 0, k, n, context)


def _group_state(gc: MselGroups) raises -> PythonObject:
    var counts = Python.list()
    for g in range(gc.m):  # driver-loop: over groups
        _ = py_call(counts.append, PythonObject(gc.counts[g]))
    return Python.tuple(gc.codes, PythonObject(gc.n), py_call(msel_b().tuple, counts))


def _state_rows(state: PythonObject, words: PythonObject, want: Int, size: Int,
                context: PythonObject) raises -> PythonObject:
    """`_GroupCodes.rows` from a lazy split's group state."""
    var n = Int(py=state[1])
    var test = msel_new(size, "<i8", context)
    var train = msel_new(n - size, "<i8", context)
    var got = msel_int(family_call(msel_entry("select_fold_i64", context), PythonObject(msel_addr(words)), PythonObject(n),
                                   PythonObject(want), PythonObject(msel_addr(test)), PythonObject(msel_addr(train))))
    if got != size:
        raise Error("RuntimeError: mojolearn: select_fold_i64 disagrees with the group counts")
    return Python.tuple(train, test)


def msel_logo_split(x: PythonObject, groups: PythonObject, model: PythonObject,
                    context: PythonObject) raises -> PythonObject:
    """`LeaveOneGroupOut.split`: one split per group, made on demand."""
    var n = msel_n_samples(x)
    var gc = MselGroups(groups, n, "groups", context)
    if gc.m <= 1:
        raise Error("The groups parameter contains fewer than 2 unique groups (" + family_str(gc.classes) + "). "
                    + "LeaveOneGroupOut expects at least 2.")
    var b = msel_b()
    var items = py_call(b.zip, py_call(Python.import_module("itertools").repeat, _group_state(gc)),
                        py_call(b.range, PythonObject(gc.m)))
    return msel_lazy(model, "family_msel_split_logo_at", items, context)


def msel_logo_at(args: PythonObject, context: PythonObject) raises -> PythonObject:
    var state = args[0]
    var g = Int(py=args[1])
    return _state_rows(state, state[0], g, Int(py=state[2][g]), context)


def msel_lpgo_split(x: PythonObject, groups: PythonObject, model: PythonObject,
                    context: PythonObject) raises -> PythonObject:
    """`LeavePGroupsOut.split`: one split per combination of groups, made
    on demand."""
    var n = msel_n_samples(x)
    var gc = MselGroups(groups, n, "groups", context)
    var n_groups = msel_field(model, "n_groups")
    if msel_cmp("ge", n_groups, PythonObject(gc.m)):
        raise Error("The groups parameter contains fewer than (or equal to) n_groups (" + family_str(n_groups)
                    + ") numbers of unique groups (" + family_str(gc.classes) + ").")
    var b = msel_b()
    var itertools = Python.import_module("itertools")
    var combos = py_call(itertools.combinations, py_call(b.range, PythonObject(gc.m)), n_groups)
    var items = py_call(b.zip, py_call(itertools.repeat, _group_state(gc)), combos)
    return msel_lazy(model, "family_msel_split_lpgo_at", items, context)


def msel_lpgo_at(args: PythonObject, context: PythonObject) raises -> PythonObject:
    var state = args[0]
    var combo = args[1]
    var counts = state[2]
    var m = len(counts)
    var table = msel_new(m, "<i4", context)
    var tp = msel_p32(table)
    var size = 0
    for i in range(len(combo)):  # driver-loop: over the chosen groups of one combination
        var g = Int(py=combo[i])
        tp[g] = 1
        size += Int(py=counts[g])
    var n = Int(py=state[1])
    var words = msel_new(n, "<i4", context)
    _ = family_call(msel_entry("gather_i32", context), PythonObject(msel_addr(table)), PythonObject(m),
                    PythonObject(msel_addr(state[0])), PythonObject(n), PythonObject(msel_addr(words)))
    return _state_rows(state, words, 1, size, context)


def msel_shuffle_split(x: PythonObject, n_splits: Int, test_size: PythonObject, train_size: PythonObject,
                       random_state: PythonObject, default_test: PythonObject, context: PythonObject) raises -> PythonObject:
    """`ShuffleSplit.split`."""
    var n = msel_n_samples(x)
    var sizes = msel_shuffle_sizes(n, test_size, train_size, default_test, context)
    var n_train = sizes[0]
    var n_test = sizes[1]
    var rng = MselRng(random_state, context)
    var want = List[Int](length=max(n_splits, 0), fill=n)
    var perms = rng.permutation_rows(want, context)
    var out = Python.list()
    for s in range(len(want)):  # driver-loop: over splits
        var perm = perms[s]
        _ = py_call(out.append, Python.tuple(msel_slice_i64(perm, n_test, n_train, context),
                                             msel_slice_i64(perm, 0, n_test, context)))
    return out


def msel_group_shuffle_split(x: PythonObject, groups: PythonObject, n_splits: Int, test_size: PythonObject,
                             train_size: PythonObject, random_state: PythonObject, default_test: PythonObject,
                             context: PythonObject) raises -> PythonObject:
    """`GroupShuffleSplit.split`."""
    var gc = MselGroups(groups, msel_n_samples(x), "groups", context)
    var m = gc.m
    var sizes = msel_shuffle_sizes(m, test_size, train_size, default_test, context)
    var n_train = sizes[0]
    var n_test = sizes[1]
    var rng = MselRng(random_state, context)
    var counts = msel_i64_list(gc.counts, context)
    var table = msel_new(m, "<i4", context)
    var sums = msel_new(2, "<i8", context)
    var want = List[Int](length=max(n_splits, 0), fill=m)
    var perms = rng.permutation_rows(want, context)
    var out = Python.list()
    var sp = msel_p64(sums)
    for s in range(len(want)):  # driver-loop: over splits
        _ = family_call(msel_entry("msel_split_table_i32", context), PythonObject(msel_addr(perms[s])), PythonObject(m),
                        PythonObject(n_test), PythonObject(n_train), PythonObject(msel_addr(counts)),
                        PythonObject(msel_addr(table)), PythonObject(msel_addr(sums)))
        var words = gc.mapped_words(table, context)
        var train = gc.only(words, 0, Int(sp[0]), context)
        var test = gc.only(words, 1, Int(sp[1]), context)
        _ = py_call(out.append, Python.tuple(train, test))
    return out


def msel_strat_shuffle_split(y: PythonObject, n_splits: Int, test_size: PythonObject, train_size: PythonObject,
                             random_state: PythonObject, default_test: PythonObject,
                             context: PythonObject) raises -> PythonObject:
    """`StratifiedShuffleSplit._native_split`: every split, in split order
    (the same draws in the same order the generator made)."""
    var b = msel_b()
    var n0 = 0
    if y is not None:
        if not Bool(py=family_call(b.hasattr, y, PythonObject("shape"))):
            n0 = Int(py=family_call(b.len, y))
        else:
            var shape = family_call(b.getattr, y, PythonObject("shape"))
            n0 = msel_int(shape[0]) if Int(py=family_call(b.len, shape)) else 0
    var gc = MselGroups(y, n0, "y", context)
    var n = gc.n
    var k = gc.m
    var sizes = msel_shuffle_sizes(n, test_size, train_size, default_test, context)
    var n_train = sizes[0]
    var n_test = sizes[1]
    if k == 0:
        raise Error("min() iterable argument is empty")
    var least = gc.counts[0]
    for c in range(k):  # driver-loop: over classes
        least = min(least, gc.counts[c])
    if least < 2:
        raise Error("The least populated classes in y have only 1 member, which is too few. The "
                    + "minimum number of groups for any class cannot be less than 2.")
    if n_train < k:
        raise Error("The train_size = " + String(n_train) + " should be greater or equal to the number of classes = " + String(k))
    if n_test < k:
        raise Error("The test_size = " + String(n_test) + " should be greater or equal to the number of classes = " + String(k))
    var rows = Python.list()
    for c in range(k):  # driver-loop: over classes (one row selection each)
        _ = py_call(rows.append, gc.only(gc.codes, c, gc.counts[c], context))
    var gather = msel_entry("gather_i64", context)
    var rng = MselRng(random_state, context)
    var out = Python.list()
    for _ in range(max(n_splits, 0)):  # driver-loop: over splits
        var modes = rng.approximate_modes(gc.counts, n_train, n_test, context)
        var tr_len = 0
        var te_len = 0
        for c in range(k):  # driver-loop: over classes
            tr_len += modes[c]
            te_len += modes[k + c]
        var train = msel_new(tr_len, "<i8", context)
        var test = msel_new(te_len, "<i8", context)
        var at_tr = 0
        var at_te = 0
        var perms = rng.permutation_rows(gc.counts, context)
        for c in range(k):  # driver-loop: over classes (two device gathers each)
            var p0 = msel_addr(perms[c])
            var tab = msel_addr(rows[c])
            _ = family_call(gather, PythonObject(tab), PythonObject(gc.counts[c]), PythonObject(p0), PythonObject(modes[c]),
                            PythonObject(msel_addr(train) + 8 * at_tr))
            _ = family_call(gather, PythonObject(tab), PythonObject(gc.counts[c]), PythonObject(p0 + 8 * modes[c]),
                            PythonObject(modes[k + c]), PythonObject(msel_addr(test) + 8 * at_te))
            at_tr += modes[c]
            at_te += modes[k + c]
        var last = rng.permutation_rows([tr_len, te_len], context)
        var out_tr = msel_new(tr_len, "<i8", context)
        var out_te = msel_new(te_len, "<i8", context)
        _ = family_call(gather, PythonObject(msel_addr(train)), PythonObject(tr_len), PythonObject(msel_addr(last[0])),
                        PythonObject(tr_len), PythonObject(msel_addr(out_tr)))
        _ = family_call(gather, PythonObject(msel_addr(test)), PythonObject(te_len), PythonObject(msel_addr(last[1])),
                        PythonObject(te_len), PythonObject(msel_addr(out_te)))
        _ = py_call(out.append, Python.tuple(out_tr, out_te))
    return out


def msel_predefined_split(model: PythonObject, context: PythonObject) raises -> PythonObject:
    """`PredefinedSplit.split`: each fold value other than -1, in sorted
    order."""
    var folds = msel_field(model, "_folds")
    var gc = MselGroups(folds, Int(py=folds.size), "test_fold", context)
    var out = Python.list()
    for c in range(len(gc.classes)):  # driver-loop: over fold values
        if msel_cmp("ne", gc.classes[c], PythonObject(-1)):
            _ = py_call(out.append, gc.rows(gc.codes, c, gc.counts[c], context))
    return out


def msel_repeated_split(model: PythonObject, x: PythonObject, y: PythonObject, groups: PythonObject,
                        context: PythonObject) raises -> PythonObject:
    """`_Repeated.split`: repeat r is a shuffled run of the splitter with
    seed `_mix_seed(base, r)`."""
    var base = MselRng(msel_field(model, "random_state"), context)
    var cv = msel_field(model, "cv")
    var cvargs = msel_field(model, "cvargs")
    var repeats = msel_int(msel_field(model, "n_repeats"))
    var out = Python.list()
    for r in range(repeats):  # driver-loop: over repeats
        var seed = PythonObject(msel_mix_seed(base.seed, r))
        var keywords = py_call(msel_b().dict, cvargs)
        py_setitem(keywords, "random_state", seed)
        py_setitem(keywords, "shuffle", PythonObject(True))
        var splitter = PythonObject(None)
        var folds = PythonObject(None)
        if cv is msel_cls("KFold", context) or cv is msel_cls("StratifiedKFold", context):
            var name = String("KFold") if cv is msel_cls("KFold", context) else String("StratifiedKFold")
            var n_splits = py_get(keywords, PythonObject("n_splits"), PythonObject(5))
            splitter = msel_new_kfold(name, n_splits, PythonObject(True), seed, context)
            folds = msel_split(splitter, x, y, groups, context)
        else:
            splitter = family_call_kw(cv, keywords)
            folds = family_call(msel_field(splitter, "split"), x, y, groups)
        _ = py_call(out.extend, folds)
    return out


# ---------------------------------------------------------------- dispatch


def msel_split(model: PythonObject, x: PythonObject, y: PythonObject, groups: PythonObject,
               context: PythonObject) raises -> PythonObject:
    """`<splitter>.split(X, y, groups)` of a mojolearn splitter: its splits
    (a list, or a lazy iterator for the per-row and per-combination
    splitters)."""
    if msel_is(model, "StratifiedGroupKFold", context):
        return msel_strat_group_split(x, y, groups, msel_int(msel_field(model, "n_splits")),
                                      msel_truth(msel_field(model, "shuffle")), msel_field(model, "random_state"), context)
    if msel_is(model, "GroupKFold", context):
        return msel_group_kfold_split(x, groups, msel_int(msel_field(model, "n_splits")),
                                      msel_truth(msel_field(model, "shuffle")), msel_field(model, "random_state"), context)
    if msel_is(model, "StratifiedKFold", context):
        return msel_stratified_split(y, msel_int(msel_field(model, "n_splits")), msel_truth(msel_field(model, "shuffle")),
                                     msel_field(model, "random_state"), context)
    if msel_is(model, "KFold", context):
        return msel_kfold_split(x, msel_int(msel_field(model, "n_splits")), msel_truth(msel_field(model, "shuffle")),
                                msel_field(model, "random_state"), context)
    if msel_is(model, "TimeSeriesSplit", context):
        return msel_time_series_split(model, x, context)
    if msel_is(model, "LeaveOneOut", context):
        return msel_loo_split(x, model, context)
    if msel_is(model, "LeavePOut", context):
        return msel_lpo_split(x, model, context)
    if msel_is(model, "LeaveOneGroupOut", context):
        return msel_logo_split(x, groups, model, context)
    if msel_is(model, "LeavePGroupsOut", context):
        return msel_lpgo_split(x, groups, model, context)
    if msel_is(model, "PredefinedSplit", context):
        return msel_predefined_split(model, context)
    if msel_is(model, "ShuffleSplit", context):
        var n_splits = msel_int(msel_field(model, "n_splits"))
        var default_test = msel_field(model, "_default_test_size")
        if msel_is(model, "GroupShuffleSplit", context):
            return msel_group_shuffle_split(x, groups, n_splits, msel_field(model, "test_size"),
                                            msel_field(model, "train_size"), msel_field(model, "random_state"),
                                            default_test, context)
        if msel_is(model, "StratifiedShuffleSplit", context):
            return msel_strat_shuffle_split(y, n_splits, msel_field(model, "test_size"), msel_field(model, "train_size"),
                                            msel_field(model, "random_state"), default_test, context)
        return msel_shuffle_split(x, n_splits, msel_field(model, "test_size"), msel_field(model, "train_size"),
                                  msel_field(model, "random_state"), default_test, context)
    if msel_is(model, "_Repeated", context):
        return msel_repeated_split(model, x, y, groups, context)
    if msel_is(model, "_IterableCV", context):
        return msel_field(model, "_pairs")
    raise Error("NotImplementedError: ")


def msel_n_splits(model: PythonObject, x: PythonObject, y: PythonObject, groups: PythonObject,
                  context: PythonObject) raises -> PythonObject:
    """`<splitter>.get_n_splits(X, y, groups)`."""
    var b = msel_b()
    var math = Python.import_module("math")
    if msel_is(model, "LeaveOneOut", context):
        if x is None:
            raise Error("The 'X' parameter should not be None.")
        return PythonObject(msel_n_samples(x))
    if msel_is(model, "LeavePOut", context):
        return _comb(PythonObject(msel_n_samples(x)), msel_field(model, "p"))
    if msel_is(model, "LeaveOneGroupOut", context):
        return family_call(b.len, py_call(b.set, msel_labels_list(groups, "groups", context)))
    if msel_is(model, "LeavePGroupsOut", context):
        var m = family_call(b.len, py_call(b.set, msel_labels_list(groups, "groups", context)))
        return _comb(m, msel_field(model, "n_groups"))
    if msel_is(model, "PredefinedSplit", context):
        var classes = msel_encode(msel_field(model, "_folds"), context)[0]
        return PythonObject(len(classes) - (1 if py_contains(classes, PythonObject(-1)) else 0))
    if msel_is(model, "_Repeated", context):
        var cv = msel_field(model, "cv")
        var cvargs = msel_field(model, "cvargs")
        var repeats = msel_field(model, "n_repeats")
        var per = PythonObject(None)
        if cv is msel_cls("KFold", context) or cv is msel_cls("StratifiedKFold", context):
            per = PythonObject(msel_check_splits(py_get(cvargs, PythonObject("n_splits"), PythonObject(5)), context))
        else:
            per = family_call(msel_field(family_call_kw(cv, py_call(b.dict, cvargs)), "get_n_splits"), x, y, groups)
        return family_call(msel_op("mul"), per, repeats)
    if msel_is(model, "_IterableCV", context):
        return family_call(b.len, msel_field(model, "_pairs"))
    return msel_field(model, "n_splits")


def _comb(n: PythonObject, k: PythonObject) raises -> PythonObject:
    """`_comb(n, k)`: n choose k in exact integers, 0 outside [0, n]."""
    if msel_cmp("lt", k, PythonObject(0)) or msel_cmp("gt", k, n):
        return PythonObject(0)
    return family_call(Python.import_module("math").comb, n, k)


# ---------------------------------------------------------------- check_cv


def msel_native_discrete(y: PythonObject, context: PythonObject) raises -> PythonObject:
    """`_native_discrete(y)`: True / False, or None when the native test
    does not apply."""
    var b = msel_b()
    var arr = y
    if not msel_is_array(y, context):
        arr = msel_try_materialize(y, "y", Python.tuple(b.TypeError, b.ValueError, b.OverflowError), context)
        if msel_missed(arr, context):
            return PythonObject(None)
    var nd = Int(py=arr.ndim)
    if nd == 2 and Int(py=arr.shape[1]) == 1:
        nd = 1
    if nd != 1 or Int(py=arr.size) < 1:
        return PythonObject(None)
    var dtype = arr.dtype
    if not py_contains(context["family_msel_native_encode"], dtype):
        return PythonObject(None)
    var d = String(py=dtype)
    var integral = msel_entry("reduce_stat", context)
    if d != "<f4" and d != "<f8":
        return PythonObject(True)
    var c = arr if String(py=arr.order) == "C" or Int(py=arr.ndim) == 1 else family_call(arr._as_c)
    return PythonObject(msel_truth(family_call(integral, PythonObject(msel_addr(c)), context["family_msel_native_code"][dtype],
                                               arr.size, context["family_msel_reduce_integral"])))


def msel_check_cv(cv_in: PythonObject, y: PythonObject, classifier: Bool, shuffle: PythonObject,
                  random_state: PythonObject, context: PythonObject) raises -> PythonObject:
    """`check_cv(cv, y, classifier=..., shuffle=..., random_state=...)`."""
    var b = msel_b()
    var cv = PythonObject(5) if cv_in is None else cv_in
    if msel_is_integral(cv, context) and not msel_is_bool(cv):
        var stratify = False
        var native = msel_native_discrete(y, context) if classifier and y is not None else PythonObject(None)
        if native is not None:
            stratify = msel_truth(native)
        elif classifier and y is not None:
            var labels = msel_flatten(y, context)
            var itertools = Python.import_module("itertools")
            var all_str = msel_truth(family_call(b.all, py_call(b.map, b.isinstance, labels, py_call(itertools.repeat, b.str))))
            if all_str:
                stratify = True
            else:
                # every label Integral, or a Real whose float is integer
                # valued (float.is_integer is False for NaN and infinity)
                var kinds = py_call(b.set, py_call(b.map, b.type, labels))
                var integral = context["family_msel_integral"]
                var real = context["family_msel_real"]
                var all_int = msel_truth(family_call(b.all, py_call(b.map, b.issubclass, kinds, py_call(itertools.repeat, integral))))
                if all_int:
                    stratify = True
                elif msel_truth(family_call(b.all, py_call(b.map, b.issubclass, kinds, py_call(itertools.repeat, real)))):
                    stratify = msel_truth(family_call(b.all, py_call(b.map, b.float.is_integer, py_call(b.map, b.float, labels))))
        var rs = random_state if msel_truth(shuffle) else PythonObject(None)
        var kind = String("StratifiedKFold") if stratify else String("KFold")
        return msel_new_kfold(kind, cv, shuffle, rs, context)
    var split = py_call(b.getattr, cv, PythonObject("split"), PythonObject(None))
    if msel_truth(py_call(b.callable, split)):
        return cv
    if family_is_str(cv) or not msel_truth(py_call(b.hasattr, cv, PythonObject("__iter__"))):
        raise Error("Expected cv as an integer, cross-validation object (from "
                    + "sklearn.model_selection) or an iterable. Got " + family_repr(cv) + ".")
    return msel_iterable_cv(cv, context)


def msel_iterable_cv(cv: PythonObject, context: PythonObject) raises -> PythonObject:
    """`_IterableCV(cv)`: the caller's index pairs copied once."""
    var pairs = Python.list()
    var it = family_call(msel_b().iter, cv)
    var b = msel_b()
    while True:  # driver-loop: over the caller's folds
        var item = msel_try(b.next, Python.tuple(it), b.StopIteration, context)
        if msel_missed(item, context):
            break
        var pair = family_call(b.tuple, item)
        if len(pair) < 2:
            raise Error("not enough values to unpack (expected 2, got " + String(len(pair)) + ")")
        if len(pair) > 2:
            raise Error("too many values to unpack (expected 2)")
        var tr = pair[0]
        var te = pair[1]
        _ = py_call(pairs.append, Python.tuple(msel_index_copy(tr, context), msel_index_copy(te, context)))
    var out = native_storage_allocate(msel_cls("_IterableCV", context))
    py_setattr(out, "_pairs", pairs)
    return out


# ---------------------------------------------------------------- train_test_split


def msel_train_test_split(arrays: PythonObject, test_size: PythonObject, train_size: PythonObject,
                          random_state: PythonObject, shuffle: PythonObject, stratify: PythonObject,
                          context: PythonObject) raises -> PythonObject:
    """`train_test_split(*arrays, ...)`."""
    var b = msel_b()
    if len(arrays) == 0:
        raise Error("At least one array required as input")
    var n = msel_n_samples(arrays[0])
    var lengths = Python.list()
    var mismatch = False
    for i in range(len(arrays)):  # driver-loop: over the input arrays
        var m = msel_n_samples(arrays[i])
        _ = py_call(lengths.append, PythonObject(m))
        mismatch = mismatch or m != n
    if mismatch:
        raise Error("Found input variables with inconsistent numbers of samples: " + family_str(lengths))
    var sizes = msel_shuffle_sizes(n, test_size, train_size, PythonObject(0.25), context)
    var n_train = sizes[0]
    var n_test = sizes[1]
    var train = PythonObject(None)
    var test = PythonObject(None)
    if not msel_truth(shuffle):
        if stratify is not None:
            raise Error("Stratified train/test split is not implemented for shuffle=False")
        train = msel_range_index(0, n_train, context)
        test = msel_range_index(n_train, n_train + n_test, context)
    else:
        var split: PythonObject
        if stratify is not None:
            split = msel_strat_shuffle_split(stratify, 1, PythonObject(n_test), PythonObject(n_train), random_state,
                                             PythonObject(0.1), context)[0]
        else:
            split = msel_shuffle_split(arrays[0], 1, PythonObject(n_test), PythonObject(n_train), random_state,
                                       PythonObject(0.1), context)[0]
        train = split[0]
        test = split[1]
    var out = Python.list()
    for i in range(len(arrays)):  # driver-loop: over the input arrays
        _ = py_call(out.append, msel_take_any(arrays[i], train, context))
        _ = py_call(out.append, msel_take_any(arrays[i], test, context))
    return out
