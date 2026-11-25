# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
# LANE nn-blocks-scatter's IDENTITY LANE (2026-11-25): ops.Session.scatter_add, segment_sum, segment_mean and
# segment_max (autograd/nn_scatter.mojo), forward and backward, and a captured Program of them.
#
# Verify through `device=` (docs/plans/VERIFY_THROUGH_DEVICE.md): the lane is device-agnostic, it opens its
# sessions with no `device=` and the harness's view gives every Session the side, so the same lane runs the host
# column (device="cpu") and the device column (device="gpu") and the two digests are compared with the reference.
#
# Loaded by tools/identity_break.py `_load_lane_fragments` in the harness namespace: np, _h, _fit, _hw,
# _same_bytes, _batch_decl and `lane` are the harness's; helpers here are prefixed `_nn_scatter_`. The inputs are
# built from the hashed stream (`_hw`) and fixed integer tables, nothing shared with another lane. Reference OWED
# (recorded once, at the release's one verification).

# A small graph: 7 nodes, 19 directed edges. Node 5 receives no edge (an EMPTY segment: sum 0, mean 0, max 0, no
# gradient). Edges 3 and 4 (and 11 and 12) are the same (src, dst) pair, so their messages are equal and the
# segment max of node 2 (and node 6) is a TIE: the gradient goes to the lower edge. Node 0 receives five edges
# (repeated ids, a long fold).
_NN_SCATTER_SRC = (1, 2, 3, 4, 4, 0, 6, 2, 1, 3, 0, 5, 5, 6, 2, 4, 1, 3, 6)
_NN_SCATTER_DST = (0, 0, 1, 2, 2, 0, 3, 4, 6, 0, 1, 6, 6, 3, 0, 4, 1, 2, 3)
_NN_SCATTER_NODES = 7


def _nn_scatter_first(r):
    """A Program's one output (run returns the outputs; a single output may come back bare)."""
    if isinstance(r, (list, tuple)):
        return np.asarray(r[0])
    return np.asarray(r)


@lane("ops-scatter-segment")
def _(ml, X, yc, yr, Xh=None):
    """ops.Session's indexed writes and grouped reductions (lane nn-blocks-scatter, 2026-11-25): one message-passing
    layer of a graph neural network in a recording session. Node features h (7, 4) gathered by edge source (take),
    a linear map w (4, 3) to edge messages (19, 3), whose values mix magnitudes 1e3 and 1e-2 so the order of every
    addition shows in the last bits; the messages aggregated into the 7 nodes by segment_sum, segment_mean and
    segment_max (an empty node, repeated targets, tied maxima) and by scatter_add along dim 0 into a base x (7, 3)
    with the target index expanded over the columns; a scatter_add along dim 1 of a (3, 5) base with a (3, 6) source
    and repeated column indices. The loss is the sum of each result times a fixed weight; backward. Parts: every
    forward result, the loss, the gradients of h, w, both bases and the dim-1 source. Then the same segment_mean
    and scatter_add captured as a Program with the ids as inputs: run on the captured arrays (the same bytes as the
    session's own result, checked) and on new arrays. Sabotage arms coverage-autograd-nn_scatter_max_order_gpu and
    coverage-autograd-nn_scatter_mean_empty_cpu fault one side each: the device side folds segment_max's rows in
    reversed order (the tied maxima's gradients of h and w move), the host side makes the empty segment's mean NaN
    (the forward, the loss and the program parts move)."""
    E, N, F, M = len(_NN_SCATTER_SRC), _NN_SCATTER_NODES, 4, 3
    src = np.asarray(_NN_SCATTER_SRC, dtype=np.int64)
    dst = np.asarray(_NN_SCATTER_DST, dtype=np.int64)
    h0 = _hw((N, F), "ops-sct:h", -1.0, 1.0)
    scale = np.where(np.arange(N)[:, None] % 2 == 0, np.float32(1.0e3), np.float32(1.0e-2)).astype(np.float32)
    h0 = np.ascontiguousarray((h0 * scale).astype(np.float32))
    w0 = _hw((F, M), "ops-sct:w", -0.5, 0.5)
    x0 = _hw((N, M), "ops-sct:x", -1.0, 1.0)
    b0 = _hw((3, 5), "ops-sct:b", -1.0, 1.0)
    s0 = _hw((3, 6), "ops-sct:s", -100.0, 100.0)
    col = np.asarray([[4, 0, 4, 1, 4, 0], [2, 2, 2, 3, 0, 1], [0, 1, 2, 3, 4, 4]], dtype=np.int64)
    weights = [_hw((N, M), f"ops-sct:c{k}", -1.0, 1.0) for k in range(4)]  # glue: four fixed loss weights
    wb = _hw((3, 5), "ops-sct:cb", -1.0, 1.0)
    parts = {}
    with ml.ops.Session(record=True, numeric_mode="identical") as s:
        h = s.tensor(h0, requires_grad=True)
        w = s.tensor(w0, requires_grad=True)
        x = s.tensor(x0, requires_grad=True)
        b = s.tensor(b0, requires_grad=True)
        sb = s.tensor(s0, requires_grad=True)
        msg = s.matmul(s.take(h, s.ids(src), axis=0), w)
        ids = s.ids(dst)
        agg = [s.segment_sum(msg, ids, N), s.segment_mean(msg, ids, N), s.segment_max(msg, ids, N),
               s.scatter_add(x, 0, s.ids(np.repeat(dst[:, None], M, axis=1)), msg)]
        side = s.scatter_add(b, 1, s.ids(col), sb)
        terms = [s.sum(s.mul(a, s.tensor(c))) for a, c in zip(agg, weights)]  # glue: four loss terms
        terms.append(s.sum(s.mul(side, s.tensor(wb))))
        loss = terms[0]
        for t in terms[1:]:  # driver-loop: over the five loss terms
            loss = s.add(loss, t)
        values = [np.asarray(s.numpy(a)) for a in agg]  # glue: one array per result
        parts["forward"] = _h(*values, np.asarray(s.numpy(side)))
        parts["loss"] = _h(np.asarray(s.numpy(loss)))
        s.backward(loss)
        parts["grad"] = _h(*[np.asarray(s.grad(t)) for t in (h, w, x, b, sb)])  # glue: one array per input
    with ml.ops.Session(numeric_mode="identical") as s:
        msg0 = np.ascontiguousarray((h0[src] @ w0).astype(np.float32))  # cpu-route: lane input built before the run
        d = s.tensor(msg0)
        ids = s.ids(dst)
        base = s.tensor(x0)
        idx = s.ids(np.repeat(dst[:, None], M, axis=1))
        mean = s.segment_mean(d, ids, N)
        scat = s.scatter_add(base, 0, idx, d)
        eager = [np.asarray(s.numpy(mean)), np.asarray(s.numpy(scat))]
        prog_mean = s.capture(inputs=[d, ids], outputs=[mean])
        prog_scat = s.capture(inputs=[base, idx, d], outputs=[scat])
        again = [_nn_scatter_first(prog_mean.run(msg0, dst)),
                 _nn_scatter_first(prog_scat.run(x0, np.repeat(dst[:, None], M, axis=1), msg0))]
        _same_bytes("Program segment_mean", again[0], "Session.segment_mean", eager[0])
        _same_bytes("Program scatter_add", again[1], "Session.scatter_add", eager[1])
        dst2 = np.asarray(_NN_SCATTER_DST[::-1], dtype=np.int64)
        msg2 = np.ascontiguousarray((msg0[::-1] * np.float32(-0.5)).astype(np.float32))
        other = [_nn_scatter_first(prog_mean.run(msg2, dst2)),
                 _nn_scatter_first(prog_scat.run(x0, np.repeat(dst2[:, None], M, axis=1), msg2))]
        parts["program"] = _h(*eager, *other)
    return _fit(parts)


_batch_decl("n/a:training-step (one seeded forward and backward through the ops session and a captured program; "
            "the forward outputs are parts, digested whole)", "ops-scatter-segment")
