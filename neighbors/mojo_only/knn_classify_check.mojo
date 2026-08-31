# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Gates for the k-NN CLASSIFIER: `neighbors/estimator.mojo::
knn_classifier_predict` over the cuML port in `neighbors/ported/knn/knn.mojo`
and `neighbors/ported/selection/knn.mojo`.

Six claims, each a named check, each on a HASHED fixture (a uniform or
ramp fixture hides a permutation -- `uniform-test-data-hides-permutation`):

1. `check_getuniquelabels`: the sorted distinct set of an int32 array with
   negative values, gaps and repeats, and `make_monotonic`'s rank map over
   it. Pure integer; this is the part of the port that decides what every
   class index downstream MEANS.
2. `check_knn_classify_matches_host_transcription`: device `predict` and
   `predict_proba` equal a HOST transcription of cuML's kernels, bit for
   bit -- the same serial fold over the same sorted neighbour slots, with
   the same `ftz` seams -- on a 3-class fixture whose labels are
   `{-1, 0, 1}` (a negative label, so the monotonic map is not the
   identity). The neighbour slots come from `knn_search`, the same call
   the estimator makes, so the transcription checks the VOTE and not the
   search (which `estimator_check.mojo` and `knn_identity_check.mojo`
   gate).
3. `check_knn_classify_ties_go_to_lowest_class`: two classes, `k = 4`, so
   2-2 votes occur; every tied row returns the LOWER class and a
   `[0.5, 0.5]` proba row. The check REFUSES ITSELF if no row ties -- a
   tie check on a fixture without ties gates nothing.
4. `check_knn_classify_reach_by_sabotage`: find a row with a 3-2 vote
   where the runner-up is the LOWER class, flip one of the winner's
   neighbours to the runner-up's label (one int32 in `y`), and the vote
   must move to the runner-up -- through `class_probs_kernel`'s label
   read, the fold, and `class_vote_kernel`'s tie rule in one motion. Then
   `k = 0` must be REFUSED by name, not clamped.
5. `check_knn_classify_multi_output_layout`: two label columns at once
   equal the two single-output answers column for column, which is the
   `row * n_outputs + output_offset` layout and cuML's `vector<int*> y`.
6. `check_knn_classify_run_twice_identical`: two predicts, same bits.

The scikit-learn comparison is NOT here: a Mojo check cannot import numpy.
It is `tools/knn_sklearn_oracle.py`, run under `pixi run -e gbmbench
python3`, in both numeric modes. Cited so it is not assumed to be missing.
"""

from max.gpu.host import DeviceBuffer, DeviceContext, HostBuffer

from mojo_only.numerics import ftz
from neighbors.estimator import knn_classifier_predict, knn_search
from neighbors.ported.label.classlabels import (
    getUniquelabels,
    make_monotonic,
)


comptime CLF_INDEX = 2000
comptime CLF_QUERIES = 64
comptime CLF_FEATURES = 8
comptime CLF_K = 5


def _coord(row: Int, feature: Int, salt: Int) -> Float32:
    """A hashed coordinate. Distinct per (row, feature), not a ramp. The
    same mix `estimator_check.mojo` uses."""
    var z = (
        UInt64(row) * 0x9E3779B97F4A7C15
        + UInt64(feature) * 0xBF58476D1CE4E5B9
        + UInt64(salt) * 0x94D049BB133111EB
    )
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9
    z = (z ^ (z >> 27)) * 0x94D049BB133111EB
    z = z ^ (z >> 31)
    return Float32(Float64(z % 1000000) / 1000000.0)


def _hash_label(row: Int, n_classes: Int, salt: Int) -> Int32:
    """A hashed class in `[0, n_classes)`, shifted down by one so the set
    includes a NEGATIVE label and the monotonic map is not the identity."""
    var z = UInt64(row) * 0x9E3779B97F4A7C15 + UInt64(salt) * 0xD1B54A32D192ED03
    z = (z ^ (z >> 29)) * 0xBF58476D1CE4E5B9
    z = z ^ (z >> 32)
    return Int32(Int(z % UInt64(n_classes))) - 1


struct ClfFixture:
    """Index, queries, labels on the host; the sorted neighbour slots from
    `knn_search` (the estimator's own search) for the transcription."""

    var ctx: DeviceContext
    var h_index: HostBuffer[DType.float32]
    var h_query: HostBuffer[DType.float32]
    var h_y: HostBuffer[DType.int32]
    var h_dist: HostBuffer[DType.float32]
    var h_idx: HostBuffer[DType.uint32]
    var n_classes: Int

    def __init__(out self, n_classes: Int, label_salt: Int) raises:
        self.ctx = DeviceContext()
        self.h_index = self.ctx.enqueue_create_host_buffer[DType.float32](
            CLF_INDEX * CLF_FEATURES
        )
        self.h_query = self.ctx.enqueue_create_host_buffer[DType.float32](
            CLF_QUERIES * CLF_FEATURES
        )
        self.h_y = self.ctx.enqueue_create_host_buffer[DType.int32](CLF_INDEX)
        self.h_dist = self.ctx.enqueue_create_host_buffer[DType.float32](
            CLF_QUERIES * CLF_K
        )
        self.h_idx = self.ctx.enqueue_create_host_buffer[DType.uint32](
            CLF_QUERIES * CLF_K
        )
        self.ctx.synchronize()
        self.n_classes = n_classes
        for j in range(CLF_INDEX):
            for f in range(CLF_FEATURES):
                self.h_index.unsafe_ptr().unsafe_store(
                    j * CLF_FEATURES + f, _coord(j, f, 11)
                )
            self.h_y.unsafe_ptr().unsafe_store(
                j, _hash_label(j, n_classes, label_salt)
            )
        for i in range(CLF_QUERIES):
            for f in range(CLF_FEATURES):
                self.h_query.unsafe_ptr().unsafe_store(
                    i * CLF_FEATURES + f, _coord(i, f, 29)
                )

    def search(mut self, k: Int) raises:
        """The neighbour slots the estimator will vote over: `knn_search`
        with `return_sqrt=False`, which is what `knn_classifier_predict`
        passes (the vote never reads a distance)."""
        _ = knn_search(
            self.ctx,
            self.h_index.unsafe_ptr(),
            CLF_INDEX,
            self.h_query.unsafe_ptr(),
            CLF_QUERIES,
            CLF_FEATURES,
            k,
            self.h_dist.unsafe_ptr(),
            self.h_idx.unsafe_ptr(),
            False,
        )

    def uniq(self) -> List[Int32]:
        """Sorted distinct labels, by insertion on the host."""
        var u = List[Int32]()
        for j in range(CLF_INDEX):
            var v = self.h_y.unsafe_ptr().unsafe_load(j)
            var lo = 0
            var hi = len(u)
            while lo < hi:
                var mid = (lo + hi) // 2
                if u[mid] < v:
                    lo = mid + 1
                else:
                    hi = mid
            if lo < len(u) and u[lo] == v:
                continue
            u.insert(lo, v)
        return u^

    def transcribe(
        self, k: Int, mut proba: List[Float32], mut labels: List[Int32]
    ):
        """cuML's `class_probs_kernel` + `class_vote_kernel` on the host,
        over the slots `search` left in `h_idx`: per row, in slot order,
        `proba[row, cls] += 1/k` (Float32, every seam through `ftz`, which
        is what the device kernel does under IDENTICAL and is inert under
        FAST on an FTZ device); then the FIRST maximum over classes."""
        var u = self.uniq()
        var nu = len(u)
        proba.clear()
        labels.clear()
        var inv = ftz(Float32(1.0) / Float32(k))
        for row in range(CLF_QUERIES):
            var acc = List[Float32]()
            for _ in range(nu):
                acc.append(Float32(0.0))
            for s in range(k):
                var nb = Int(self.h_idx.unsafe_ptr().unsafe_load(row * k + s))
                var lab = self.h_y.unsafe_ptr().unsafe_load(nb)
                var c = 0
                while u[c] != lab:
                    c += 1
                acc[c] = ftz(acc[c] + inv)
            var cur_max = Float32(-1.0)
            var cur = -1
            for c in range(nu):
                if acc[c] > cur_max:
                    cur_max = acc[c]
                    cur = c
            for c in range(nu):
                proba.append(acc[c])
            labels.append(u[cur])

    def predict(
        mut self,
        k: Int,
        want_proba: Bool,
        mut out_labels: HostBuffer[DType.int32],
        mut out_proba: HostBuffer[DType.float32],
        mut out_uniq: HostBuffer[DType.int32],
    ) raises -> Int:
        var n_classes = List[Int]()
        n_classes.append(self.n_classes)
        return knn_classifier_predict(
            self.ctx,
            self.h_index.unsafe_ptr(),
            CLF_INDEX,
            self.h_query.unsafe_ptr(),
            CLF_QUERIES,
            CLF_FEATURES,
            k,
            self.h_y.unsafe_ptr(),
            1,
            n_classes,
            out_labels.unsafe_ptr(),
            out_proba.unsafe_ptr(),
            out_uniq.unsafe_ptr(),
            want_proba,
        )


def check_getuniquelabels() raises:
    """`getUniquelabels` and `make_monotonic` on `[-7, 3, 3, 1000000, -7,
    0, 3]`: the set is `[-7, 0, 3, 1000000]` and the 1-based ranks are
    `[1, 3, 3, 4, 1, 2, 3]`."""
    var ctx = DeviceContext()
    var n = 7
    var h = ctx.enqueue_create_host_buffer[DType.int32](n)
    var d = ctx.enqueue_create_buffer[DType.int32](n)
    var m = ctx.enqueue_create_buffer[DType.int32](n)
    ctx.synchronize()
    var vals = List[Int32]()
    vals.append(-7)
    vals.append(3)
    vals.append(3)
    vals.append(1000000)
    vals.append(-7)
    vals.append(0)
    vals.append(3)
    for i in range(n):
        h.unsafe_ptr().unsafe_store(i, vals[i])
    ctx.enqueue_copy(dst_buf=d, src_ptr=h.unsafe_ptr())
    ctx.synchronize()
    var u = getUniquelabels(ctx, d, n)
    if (
        len(u) != 4
        or u[0] != -7
        or u[1] != 0
        or u[2] != 3
        or u[3] != 1000000
    ):
        raise Error(
            "check_getuniquelabels: want [-7, 0, 3, 1000000], got "
            + String(len(u))
            + " values"
        )
    make_monotonic(ctx, m, d, n, False)
    var hm = ctx.enqueue_create_host_buffer[DType.int32](n)
    ctx.enqueue_copy(dst_ptr=hm.unsafe_ptr(), src_buf=m)
    ctx.synchronize()
    var want = List[Int32]()
    want.append(1)
    want.append(3)
    want.append(3)
    want.append(4)
    want.append(1)
    want.append(2)
    want.append(3)
    for i in range(n):
        if hm.unsafe_ptr().unsafe_load(i) != want[i]:
            raise Error(
                "check_getuniquelabels: make_monotonic["
                + String(i)
                + "] = "
                + String(hm.unsafe_ptr().unsafe_load(i))
                + ", want "
                + String(want[i])
            )
    _ = h^
    _ = hm^
    _ = d^
    _ = m^
    print(
        "check_getuniquelabels: OK (negative label, gap, repeats -> sorted"
        " set of 4; monotonic ranks as computed by hand)"
    )


def check_knn_classify_matches_host_transcription() raises:
    """Claim 2. Three classes `{-1, 0, 1}`, `k = 5`, 64 queries."""
    var fx = ClfFixture(3, 101)
    fx.search(CLF_K)
    var want_p = List[Float32]()
    var want_l = List[Int32]()
    fx.transcribe(CLF_K, want_p, want_l)

    var out_l = fx.ctx.enqueue_create_host_buffer[DType.int32](CLF_QUERIES)
    var out_p = fx.ctx.enqueue_create_host_buffer[DType.float32](
        CLF_QUERIES * 3
    )
    var out_u = fx.ctx.enqueue_create_host_buffer[DType.int32](3)
    fx.ctx.synchronize()
    _ = fx.predict(CLF_K, False, out_l, out_p, out_u)
    var wrong_l = 0
    for i in range(CLF_QUERIES):
        if out_l.unsafe_ptr().unsafe_load(i) != want_l[i]:
            wrong_l += 1
    _ = fx.predict(CLF_K, True, out_l, out_p, out_u)
    var wrong_p = 0
    for i in range(CLF_QUERIES * 3):
        # BIT comparison, not tolerance: the claim is one arithmetic.
        var got = out_p.unsafe_ptr().unsafe_load(i)
        if got.to_bits() != want_p[i].to_bits():
            wrong_p += 1
    var u_ok = (
        out_u.unsafe_ptr().unsafe_load(0) == -1
        and out_u.unsafe_ptr().unsafe_load(1) == 0
        and out_u.unsafe_ptr().unsafe_load(2) == 1
    )
    if wrong_l != 0 or wrong_p != 0 or not u_ok:
        raise Error(
            "check_knn_classify_matches_host_transcription: "
            + String(wrong_l)
            + " of "
            + String(CLF_QUERIES)
            + " labels and "
            + String(wrong_p)
            + " of "
            + String(CLF_QUERIES * 3)
            + " proba cells differ from the host transcription; uniq ok="
            + String(u_ok)
        )
    # The fixture must have exercised more than one class and more than
    # one outcome, or the transcription agreed about nothing.
    var seen_neg = 0
    var seen_pos = 0
    for i in range(CLF_QUERIES):
        if want_l[i] == -1:
            seen_neg += 1
        if want_l[i] == 1:
            seen_pos += 1
    if seen_neg == 0 or seen_pos == 0:
        raise Error(
            "check_knn_classify_matches_host_transcription: the fixture"
            " produced only one class; it gates nothing"
        )
    _ = out_l^
    _ = out_p^
    _ = out_u^
    print(
        "check_knn_classify_matches_host_transcription: OK ("
        + String(CLF_QUERIES)
        + " labels and "
        + String(CLF_QUERIES * 3)
        + " proba cells bit-identical to the host fold; labels {-1,0,1},"
        " -1 predicted "
        + String(seen_neg)
        + "x, +1 "
        + String(seen_pos)
        + "x)"
    )


def check_knn_classify_ties_go_to_lowest_class() raises:
    """Claim 3. Two classes `{-1, 0}`, `k = 4`: 2-2 rows return -1 and
    `[0.5, 0.5]`."""
    var fx = ClfFixture(2, 202)
    var k = 4
    fx.search(k)
    var out_l = fx.ctx.enqueue_create_host_buffer[DType.int32](CLF_QUERIES)
    var out_p = fx.ctx.enqueue_create_host_buffer[DType.float32](
        CLF_QUERIES * 2
    )
    var out_u = fx.ctx.enqueue_create_host_buffer[DType.int32](2)
    fx.ctx.synchronize()
    _ = fx.predict(k, False, out_l, out_p, out_u)
    # Count the votes on the host from the SAME slots, to know which rows
    # tie; the assertion is on the device's answer for those rows.
    var ties = 0
    var bad = 0
    for row in range(CLF_QUERIES):
        var n_neg = 0
        for s in range(k):
            var nb = Int(fx.h_idx.unsafe_ptr().unsafe_load(row * k + s))
            if fx.h_y.unsafe_ptr().unsafe_load(nb) == -1:
                n_neg += 1
        if n_neg == 2:
            ties += 1
            if out_l.unsafe_ptr().unsafe_load(row) != -1:
                bad += 1
    _ = fx.predict(k, True, out_l, out_p, out_u)
    for row in range(CLF_QUERIES):
        var n_neg = 0
        for s in range(k):
            var nb = Int(fx.h_idx.unsafe_ptr().unsafe_load(row * k + s))
            if fx.h_y.unsafe_ptr().unsafe_load(nb) == -1:
                n_neg += 1
        if n_neg == 2:
            if (
                out_p.unsafe_ptr().unsafe_load(row * 2) != Float32(0.5)
                or out_p.unsafe_ptr().unsafe_load(row * 2 + 1) != Float32(0.5)
            ):
                bad += 1
    if ties == 0:
        raise Error(
            "check_knn_classify_ties_go_to_lowest_class: NO ROW TIED; the"
            " fixture gates nothing -- change the label salt"
        )
    if bad != 0:
        raise Error(
            "check_knn_classify_ties_go_to_lowest_class: "
            + String(bad)
            + " tied rows did not resolve to the lowest class / [0.5, 0.5]"
            + " (of "
            + String(ties)
            + " ties)"
        )
    _ = out_l^
    _ = out_p^
    _ = out_u^
    print(
        "check_knn_classify_ties_go_to_lowest_class: OK ("
        + String(ties)
        + " of "
        + String(CLF_QUERIES)
        + " rows tied 2-2 at k=4; every one returned -1 and [0.5, 0.5])"
    )


def check_knn_classify_reach_by_sabotage() raises:
    """Claim 4. A 3-2 row whose runner-up is the LOWER class; flip one
    winning neighbour's label to the runner-up; the vote must move. Then
    `k = 0` is refused."""
    var fx = ClfFixture(3, 101)
    fx.search(CLF_K)
    var out_l = fx.ctx.enqueue_create_host_buffer[DType.int32](CLF_QUERIES)
    var out_p = fx.ctx.enqueue_create_host_buffer[DType.float32](
        CLF_QUERIES * 3
    )
    var out_u = fx.ctx.enqueue_create_host_buffer[DType.int32](3)
    fx.ctx.synchronize()
    _ = fx.predict(CLF_K, False, out_l, out_p, out_u)

    # Find the row: counts over {-1, 0, 1}; winner W has 3, runner-up R has
    # 2, and R < W (so after the flip R wins 3-2, not by a tie rule).
    var row_found = -1
    var flip_nb = -1
    var winner = Int32(0)
    var runner = Int32(0)
    for row in range(CLF_QUERIES):
        var cnt = List[Int]()
        cnt.append(0)
        cnt.append(0)
        cnt.append(0)
        for s in range(CLF_K):
            var nb = Int(fx.h_idx.unsafe_ptr().unsafe_load(row * CLF_K + s))
            cnt[Int(fx.h_y.unsafe_ptr().unsafe_load(nb)) + 1] += 1
        for w in range(3):
            for r in range(w):
                if cnt[w] == 3 and cnt[r] == 2:
                    row_found = row
                    winner = Int32(w - 1)
                    runner = Int32(r - 1)
        if row_found >= 0:
            for s in range(CLF_K):
                var nb = Int(
                    fx.h_idx.unsafe_ptr().unsafe_load(row * CLF_K + s)
                )
                if fx.h_y.unsafe_ptr().unsafe_load(nb) == winner:
                    flip_nb = nb
            break
    if row_found < 0:
        raise Error(
            "check_knn_classify_reach_by_sabotage: no 3-2 row with a lower"
            " runner-up; the fixture cannot host the sabotage"
        )
    if out_l.unsafe_ptr().unsafe_load(row_found) != winner:
        raise Error(
            "check_knn_classify_reach_by_sabotage: the chosen row did not"
            " predict its 3-vote winner before the sabotage"
        )
    # THE SABOTAGE: one int32 in y.
    var saved = fx.h_y.unsafe_ptr().unsafe_load(flip_nb)
    fx.h_y.unsafe_ptr().unsafe_store(flip_nb, runner)
    _ = fx.predict(CLF_K, False, out_l, out_p, out_u)
    var moved_to = out_l.unsafe_ptr().unsafe_load(row_found)
    fx.h_y.unsafe_ptr().unsafe_store(flip_nb, saved)
    if moved_to != runner:
        raise Error(
            "check_knn_classify_reach_by_sabotage: flipping neighbour "
            + String(flip_nb)
            + " of row "
            + String(row_found)
            + " from "
            + String(winner)
            + " to "
            + String(runner)
            + " left the vote at "
            + String(moved_to)
            + "; the label read or the fold is not reached"
        )

    # k = 0: refused by name, never clamped.
    var refused = False
    try:
        _ = fx.predict(0, False, out_l, out_p, out_u)
    except e:
        refused = String(e).find("k must be positive") >= 0
    if not refused:
        raise Error(
            "check_knn_classify_reach_by_sabotage: k=0 was not refused by"
            " name"
        )
    _ = out_l^
    _ = out_p^
    _ = out_u^
    print(
        "check_knn_classify_reach_by_sabotage: OK (row "
        + String(row_found)
        + " moved "
        + String(winner)
        + " -> "
        + String(runner)
        + " on one flipped label; k=0 refused by name)"
    )


def check_knn_classify_multi_output_layout() raises:
    """Claim 5. `y` with two columns (the 3-class and the 2-class label
    sets) equals the two single-output answers, column for column."""
    var fx3 = ClfFixture(3, 101)
    var fx2 = ClfFixture(2, 202)
    var ctx = fx3.ctx
    var out_l3 = ctx.enqueue_create_host_buffer[DType.int32](CLF_QUERIES)
    var out_l2 = ctx.enqueue_create_host_buffer[DType.int32](CLF_QUERIES)
    var out_p = ctx.enqueue_create_host_buffer[DType.float32](CLF_QUERIES * 3)
    var out_u = ctx.enqueue_create_host_buffer[DType.int32](3)
    var y2 = ctx.enqueue_create_host_buffer[DType.int32](2 * CLF_INDEX)
    var out_l = ctx.enqueue_create_host_buffer[DType.int32](2 * CLF_QUERIES)
    var out_p2 = ctx.enqueue_create_host_buffer[DType.float32](
        CLF_QUERIES * 5
    )
    var out_u2 = ctx.enqueue_create_host_buffer[DType.int32](5)
    ctx.synchronize()
    _ = fx3.predict(CLF_K, False, out_l3, out_p, out_u)
    _ = fx2.predict(CLF_K, False, out_l2, out_p, out_u)
    for j in range(CLF_INDEX):
        y2.unsafe_ptr().unsafe_store(j, fx3.h_y.unsafe_ptr().unsafe_load(j))
        y2.unsafe_ptr().unsafe_store(
            CLF_INDEX + j, fx2.h_y.unsafe_ptr().unsafe_load(j)
        )
    var n_classes = List[Int]()
    n_classes.append(3)
    n_classes.append(2)
    _ = knn_classifier_predict(
        ctx,
        fx3.h_index.unsafe_ptr(),
        CLF_INDEX,
        fx3.h_query.unsafe_ptr(),
        CLF_QUERIES,
        CLF_FEATURES,
        CLF_K,
        y2.unsafe_ptr(),
        2,
        n_classes,
        out_l.unsafe_ptr(),
        out_p2.unsafe_ptr(),
        out_u2.unsafe_ptr(),
        False,
    )
    var bad = 0
    for i in range(CLF_QUERIES):
        if out_l.unsafe_ptr().unsafe_load(i * 2) != out_l3.unsafe_ptr().unsafe_load(i):
            bad += 1
        if out_l.unsafe_ptr().unsafe_load(i * 2 + 1) != out_l2.unsafe_ptr().unsafe_load(i):
            bad += 1
    if bad != 0:
        raise Error(
            "check_knn_classify_multi_output_layout: "
            + String(bad)
            + " cells of the 2-output answer differ from the single-output"
            " answers"
        )
    # And the wrong class count is refused before anything is written.
    var n_bad = List[Int]()
    n_bad.append(3)
    n_bad.append(7)
    var refused = False
    try:
        _ = knn_classifier_predict(
            ctx,
            fx3.h_index.unsafe_ptr(),
            CLF_INDEX,
            fx3.h_query.unsafe_ptr(),
            CLF_QUERIES,
            CLF_FEATURES,
            CLF_K,
            y2.unsafe_ptr(),
            2,
            n_bad,
            out_l.unsafe_ptr(),
            out_p2.unsafe_ptr(),
            out_u2.unsafe_ptr(),
            False,
        )
    except e:
        refused = String(e).find("class sets is wrong") >= 0
    if not refused:
        raise Error(
            "check_knn_classify_multi_output_layout: a wrong caller class"
            " count (7 for a 2-class column) was not refused by name"
        )
    _ = out_l3^
    _ = out_l2^
    _ = out_p^
    _ = out_u^
    _ = y2^
    _ = out_l^
    _ = out_p2^
    _ = out_u2^
    print(
        "check_knn_classify_multi_output_layout: OK (2 outputs x "
        + String(CLF_QUERIES)
        + " rows equal the single-output answers; wrong class count refused)"
    )


def check_knn_classify_run_twice_identical() raises:
    """Claim 6."""
    var fx = ClfFixture(3, 101)
    var a_l = fx.ctx.enqueue_create_host_buffer[DType.int32](CLF_QUERIES)
    var b_l = fx.ctx.enqueue_create_host_buffer[DType.int32](CLF_QUERIES)
    var a_p = fx.ctx.enqueue_create_host_buffer[DType.float32](CLF_QUERIES * 3)
    var b_p = fx.ctx.enqueue_create_host_buffer[DType.float32](CLF_QUERIES * 3)
    var u = fx.ctx.enqueue_create_host_buffer[DType.int32](3)
    fx.ctx.synchronize()
    _ = fx.predict(CLF_K, True, a_l, a_p, u)
    _ = fx.predict(CLF_K, False, a_l, b_p, u)
    _ = fx.predict(CLF_K, True, b_l, b_p, u)
    _ = fx.predict(CLF_K, False, b_l, a_p, u)
    var bad = 0
    for i in range(CLF_QUERIES):
        if a_l.unsafe_ptr().unsafe_load(i) != b_l.unsafe_ptr().unsafe_load(i):
            bad += 1
    for i in range(CLF_QUERIES * 3):
        if (
            a_p.unsafe_ptr().unsafe_load(i).to_bits()
            != b_p.unsafe_ptr().unsafe_load(i).to_bits()
        ):
            bad += 1
    if bad != 0:
        raise Error(
            "check_knn_classify_run_twice_identical: "
            + String(bad)
            + " cells differ between two runs"
        )
    _ = a_l^
    _ = b_l^
    _ = a_p^
    _ = b_p^
    _ = u^
    print(
        "check_knn_classify_run_twice_identical: OK (labels and proba"
        " bit-identical across two predicts)"
    )


def main() raises:
    check_getuniquelabels()
    check_knn_classify_matches_host_transcription()
    check_knn_classify_ties_go_to_lowest_class()
    check_knn_classify_reach_by_sabotage()
    check_knn_classify_multi_output_layout()
    check_knn_classify_run_twice_identical()
