# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""`weights='distance'` for the k-NN classifier and regressor.

NO UPSTREAM. ORIGINAL WORK, and this file says so rather than hunting for
something to credit.

WHY THERE IS NO UPSTREAM AND WHY THAT IS NOT A REASON TO REFUSE
----------------------------------------------------------------
cuML REFUSES this parameter. `kneighbors_classifier.pyx:191-193` and
`kneighbors_regressor.pyx:188-190` both raise "Only uniform weighting
strategy is supported currently", and `_params_from_cpu` (`:134-135` /
`:146-147`) raises `UnsupportedOnGPU` at the scikit-learn boundary. There
is no cuVS kernel, no RAFT primitive and no cuML C++ entry to transliterate.
`neighbors/NOT_IMPLEMENTED.tsv` carried a row saying exactly that, and the
row's REASON -- "there is no upstream GPU kernel to port" -- was withdrawn
on 2026-09-01: a refusal is legitimate only when the thing is genuinely
impossible here or when refusing IS the correct behaviour for the input.
"a prior implementation does not have it" is neither. The row is now a
DERIVATION_MAP row with `no-upstream` status, which is what it always
should have been.

WHAT IT IS: scikit-learn's SEMANTICS, not scikit-learn's bits
--------------------------------------------------------------
scikit-learn is this tree's oracle for SEMANTICS on paths cuML does not
carry, exactly as `kde/impl/neighbors/kernel_density.mojo` already treats
`_binary_tree.pxi`. The definition is `sklearn/neighbors/_base.py:74-114`,
the dense-array branch at `:108-113`:

    with np.errstate(divide="ignore"):
        dist = 1.0 / dist
    inf_mask = np.isinf(dist)
    inf_row = np.any(inf_mask, axis=1)
    dist[inf_row] = inf_mask[inf_row]
    return dist

Three sentences, and the third is the one everybody gets wrong. A row that
contains ANY exact-zero distance is REPLACED WHOLESALE by its infinity
mask: the exact matches get weight 1.0 and EVERY OTHER NEIGHBOUR IN THAT
ROW gets 0.0, not merely a smaller weight. Their comment at `:96-98` says
so. It is a row-level replacement, not a per-element clamp, and a port
that only special-cased the zero element itself would agree with sklearn
on every fixture without a duplicate point and disagree on every fixture
with one. `check_knn_distance_weights` plants a duplicate for that reason.

The consumers, also scikit-learn's:

    _classification.py:398-410   proba[row, class] += w[row, slot], in slot
                                 order, then each row divided by its own
                                 sum. `predict` is `weighted_mode` over the
                                 same accumulation, which is the argmax
                                 this tree already computes in
                                 `class_vote_kernel`.
    _regression.py               pred = sum_j y[nbr_j] * w[row,j]
                                        / sum_j w[row,j]

THE UNIFORM ARM IS NOT TOUCHED. `class_probs_kernel` and
`regress_avg_kernel` in `neighbors/impl/selection/knn.mojo` are cuML's,
transliterated, and they still run byte for byte when `weights='uniform'`.
Their `1/k` per slot is PRE-normalized where the weighted kernels here
accumulate raw and normalize afterwards, which is the difference between
cuML's formulation and scikit-learn's; making one call the other would
have moved cuML's bits to accommodate a parameter cuML does not have.

THE DISTANCES MUST BE THE REAL ONES, NOT THE SQUARED ONES
-----------------------------------------------------------
`neighbors/estimator.mojo`'s classifier and regressor call the search with
`return_sqrt = False`, because a UNIFORM vote only needs the ORDER and the
squared distance is monotone in the distance, so the root is a launch
saved. A DISTANCE weight is not order-only: `1/d^2` is not `1/d`. So the
weighted arm asks the search for the rooted distance. That is a real
behavioural difference between the two arms of the same estimator and it
is stated here, in the estimator's dispatch, and in the Python docstring,
rather than being left for someone to find.

THE NUMERIC CONTRACT
--------------------
Nothing new is invented. `1/d` is `identical_div` (row 49, DEVIATION 740),
the accumulations are serial per row in slot order exactly as cuML's two
kernels already are, so there is no fold width and no lane primitive
anywhere in this file, and every stored intermediate goes through `ftz`
(row 10). One thread owns one query row in both kernels, which is cuML's
own decomposition for the same job.

DEVIATION 554 (2026-09-01): THE WEIGHTS ARE COMPUTED ON THE HOST.
scikit-learn computes them in numpy over the returned `(n_queries, k)`
distance matrix, after the search. This does the same, on the host, for a
reason that is about the identity column and not about convenience: the
zero test is `d == 0.0`, the row test is "does ANY slot in this row hold a
zero", and the row-level replacement then depends on that reduction. Done
on the device that is a per-row any-reduction whose shape would be a new
thing to pin; done on the host over `n_queries * k` floats it is a serial
loop with one order everywhere. `n_queries * k` is the size of the answer
the caller is already receiving, so this adds no asymptotic cost to a
search that just read `n_queries * n_index` distances.

DEVIATION 555 (2026-09-01): A ROW WHOSE WEIGHTS ALL UNDERFLOW TO ZERO IS
REFUSED BY NAME. scikit-learn raises "All neighbors of some sample is
getting zero weights" (`_classification.py:392-396`) only for a USER
CALLABLE, because `1/d` cannot underflow for any finite d it admits. It
can here: `d` above about 2^127 makes `1/d` subnormal, and row 10 flushes
a subnormal to zero on an FTZ column and keeps it on a denormal-honoring
one, so the row's normalizer would be `+0.0` on Metal and tiny-but-normal
on CUDA and the two columns would disagree about the whole row. Refusing
is the only answer that is the same on three columns, and it is input
validation, which is where a refusal is still right.
"""

from std.gpu import block_dim, block_idx, thread_idx

from checks.numerics import ftz, identical_div


#: `weights` as a value. Their strings are `'uniform'` and `'distance'`
#: (`_classification.py:195`); `python/mojolearn/neighbors.py` maps them.
comptime WEIGHTS_UNIFORM = 0
comptime WEIGHTS_DISTANCE = 1

#: NO BLOCK WIDTH IS DECLARED HERE ON PURPOSE. Both kernels below are
#: launched by `neighbors/impl/selection/knn.mojo` at its own `KNN_TPB_X`
#: (cuML's `template <int TPB_X = 32>`), because they sit in that file's
#: loop beside the uniform kernels and a second constant here would be a
#: second place to change and a chance for the two arms to drift.


def weights_from_name(name: String) raises -> Int:
    """`'uniform'` or `'distance'`. A callable is NOT ported and is refused
    by name; sklearn accepts one (`_base.py:116`) and there is no way to
    call a Python function from inside a GPU kernel, which is the one kind
    of reason that still justifies a refusal."""
    if name == "uniform":
        return WEIGHTS_UNIFORM
    if name == "distance":
        return WEIGHTS_DISTANCE
    if name == "callable":
        raise Error(
            "mojolearn k-NN: weights=<callable> is NOT PORTED; a Python"
            " function cannot run inside a GPU kernel. 'uniform' and"
            " 'distance' are ported."
        )
    raise Error(
        "mojolearn k-NN: weights='" + name + "' is not a weighting; use"
        " 'uniform' or 'distance'"
    )


def host_distance_weights(
    dist: List[Float32], n_queries: Int, k: Int
) raises -> List[Float32]:
    """`_get_weights(dist, 'distance')`, `_base.py:108-113`, on the host.

    `dist` is the `n_queries x k` matrix the search returned, ASCENDING in
    each row (`knn_search_traced` sorts it, so slot 0 is the nearest and a
    zero, if there is one, is at slot 0). `w` is written the same shape.

    Their three sentences, in their order, per row:

      1. `w = 1/d` for every slot. `identical_div`, so `1/0` is `+inf` with
         the sign rule and the flush model pinned (DEVIATION 740) rather
         than whatever the column does.
      2. `inf_row = any(isinf(w))`.
      3. if `inf_row`: `w = isinf(w)` -- 1.0 at the infinite slots, 0.0
         everywhere else IN THAT ROW. THE WHOLE ROW IS REPLACED.

    Raises on a row whose weights are all zero (DEVIATION 555).

    HOST LISTS IN AND OUT, not pointers, and that is a deliberate
    narrowing: this function is host-only, its caller already has the
    distances in host memory, and `archive/plans/UNWIRED.md:31` records that a pointer
    from `enqueue_create_host_buffer` is not interchangeable with an
    arbitrary host pointer on this stack -- SILENTLY. A `List` boundary
    cannot express that mistake.
    """
    if n_queries <= 0 or k <= 0:
        raise Error(
            "host_distance_weights: n_queries and k must be positive, got "
            + String(n_queries) + ", " + String(k)
        )
    if len(dist) != n_queries * k:
        raise Error(
            "host_distance_weights: dist holds " + String(len(dist))
            + " values, expected " + String(n_queries * k)
        )
    var w = List[Float32](capacity=n_queries * k)
    for _ in range(n_queries * k):
        w.append(Float32(0.0))
    var pos_inf = Float32(1.0) / Float32(0.0)
    for i in range(n_queries):
        var base = i * k
        # SENTENCE 1: `dist = 1.0 / dist` under `errstate(divide="ignore")`.
        var any_inf = False
        for j in range(k):
            var d = ftz(dist[base + j])
            var v = ftz(identical_div(Float32(1.0), d))
            # SENTENCE 2, folded into the same pass: `np.isinf` is true for
            # BOTH signs. A negative distance cannot occur -- every op in
            # `distance_ops.mojo` returns a non-negative value, and cosine's
            # is in [0, 2] up to round-off -- so a `-inf` here would be a
            # bug elsewhere, and it is detected below rather than silently
            # weighted.
            if v == pos_inf or v == -pos_inf:
                any_inf = True
            w[base + j] = v

        # SENTENCE 3: `dist[inf_row] = inf_mask[inf_row]`. THE WHOLE ROW.
        if any_inf:
            for j in range(k):
                var v = w[base + j]
                if v == pos_inf:
                    w[base + j] = Float32(1.0)
                elif v == -pos_inf:
                    raise Error(
                        "host_distance_weights: query row "
                        + String(i)
                        + " slot "
                        + String(j)
                        + " has a NEGATIVE distance, which no ported metric"
                        " can produce; refusing rather than weighting it"
                    )
                else:
                    w[base + j] = Float32(0.0)
            continue

        # DEVIATION 555: the normalizer must be a normal float on every
        # column. Checked per row so the message can name the row and the
        # distance that caused it. Only the no-inf rows need it: an inf row
        # has at least one weight of exactly 1.0 by sentence 3.
        var s = Float32(0.0)
        for j in range(k):
            s = ftz(s + ftz(w[base + j]))
        if s <= Float32(0.0):
            raise Error(
                "knn weights='distance': every neighbour of query row "
                + String(i)
                + " has 1/d that underflows float32 (the nearest is at"
                " distance "
                + String(dist[base])
                + "), so the row's normalizer is zero on an FTZ column and"
                " not on a denormal-honoring one (DEVIATION 555)"
            )
    return w^


def weighted_class_probs_kernel(
    out_p: MutPointer[Float32, MutAnyOrigin],
    knn_indices: MutPointer[UInt32, MutAnyOrigin],
    labels: MutPointer[Int32, MutAnyOrigin],
    weights: MutPointer[Float32, MutAnyOrigin],
    n_uniq_labels_in: Int32,
    n_samples_in: Int32,
    n_neighbors_in: Int32,
):
    """`_classification.py:404-410` as a kernel: accumulate `w[row, slot]`
    into the slot's class IN SLOT ORDER, then divide the row by its own
    sum.

    Shaped as cuML's `class_probs_kernel` (`src_prims/selection/knn.cuh:
    52-71`) -- one thread per query row, `out` zeroed by the caller,
    `labels` the MONOTONIC 0-based relabelling so `out_label` is a column
    index -- because that is the right decomposition for this job and
    because a reader diffing the two kernels should see one difference,
    the weight, and not two.

    THE TWO-PASS SHAPE IS scikit-learn's AND IS NOT AN OPTIMIZATION
    OPPORTUNITY. `sum_j w_j` could be accumulated in the same pass as the
    scatter, but sklearn normalizes AFTER the full accumulation
    (`:409-410`), and a running normalizer would divide by a different
    number at every slot. The second loop is over `n_uniq` classes, not
    over `k`, so it also costs less than the first.
    """
    var row = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    var n_neighbors = Int(n_neighbors_in)
    var i = row * n_neighbors
    if row >= Int(n_samples_in):
        return
    var n_uniq = Int(n_uniq_labels_in)

    # pass 1: the scatter, slot order, one serial fold per (row, class)
    for j in range(n_neighbors):
        var nbr = Int(knn_indices.unsafe_load(i + j))
        var out_label = Int(labels.unsafe_load(nbr))
        var out_idx = row * n_uniq + out_label
        out_p.unsafe_store(
            out_idx,
            ftz(
                out_p.unsafe_load(out_idx)
                + ftz(weights.unsafe_load(i + j))
            ),
        )

    # pass 2: `proba_k /= proba_k.sum(axis=1)`. The sum is taken over the
    # CLASS axis of the accumulated row, not over the weight vector, which
    # is what sklearn does and is not the same float: the class sums were
    # each folded in slot order and re-adding them in class order is a
    # different association. Following their expression, not their value.
    var s = Float32(0.0)
    for c in range(n_uniq):
        s = ftz(s + ftz(out_p.unsafe_load(row * n_uniq + c)))
    for c in range(n_uniq):
        var oi = row * n_uniq + c
        out_p.unsafe_store(
            oi, ftz(identical_div(ftz(out_p.unsafe_load(oi)), s))
        )


def weighted_regress_avg_kernel(
    out_p: MutPointer[Float32, MutAnyOrigin],
    knn_indices: MutPointer[UInt32, MutAnyOrigin],
    labels: MutPointer[Float32, MutAnyOrigin],
    weights: MutPointer[Float32, MutAnyOrigin],
    n_samples_in: Int32,
    n_neighbors_in: Int32,
    n_outputs_in: Int32,
    output_offset_in: Int32,
):
    """`sklearn/neighbors/_regression.py`'s weighted branch:
    `sum_j y[nbr_j] * w_j / sum_j w_j`, both folds serial in slot order.

    Shaped as cuML's `regress_avg_kernel` (`knn.cuh:112-131`) with `1/k`
    replaced by the row's own normalizer, and with the SAME multi-output
    layout (`out[row * n_outputs + output_offset]`) so the two arms write
    the same buffer the same way.

    The product is a plain `*` and NOT `identical_mul_add(y, w, pred)`,
    which would be one rounding instead of two. That is deliberate:
    sklearn's expression is `np.sum(_y[neigh_ind] * weights, axis=1)`, a
    materialized product then a sum, and this file's contract is
    sklearn's SEMANTICS. Pinning it to an fma here would make the ported
    arithmetic differ from the reference it is checked against for a
    reason no ledger row asks for. Both operands and the running sum go
    through `ftz`, which is row 10 and is a mode question, not a shape
    question.
    """
    var row = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    var n_neighbors = Int(n_neighbors_in)
    var i = row * n_neighbors
    if row >= Int(n_samples_in):
        return
    var num = Float32(0.0)
    var den = Float32(0.0)
    for j in range(n_neighbors):
        var nbr = Int(knn_indices.unsafe_load(i + j))
        var wv = ftz(weights.unsafe_load(i + j))
        var yv = ftz(labels.unsafe_load(nbr))
        num = ftz(num + ftz(yv * wv))
        den = ftz(den + wv)
    out_p.unsafe_store(
        row * Int(n_outputs_in) + Int(output_offset_in),
        ftz(identical_div(num, den)),
    )
