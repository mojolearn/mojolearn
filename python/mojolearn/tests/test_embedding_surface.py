# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The gate for the Python surface of `mojolearn.Embedding` (2026-09-14): a
gate on the WIRING (the two params lists, the padding row, the carried
accumulator, every refusal by name) with NumPy as an independent reference
for the gather and the ascending fold on normal values, and plan="sort"
(PLAN_SORT) held bit-identical to plan="scan". The profile's
arithmetic and its sixteen sabotage arms are gated by `pixi run
check-embedding` and `tools/embedding_sabotage_arm.sh`.

    cd python && python3 -m mojolearn.tests.test_embedding_surface

Exit 2 naming `bindings/build_embedding.sh` when unbuilt.
"""
import sys

import numpy as np

import mojolearn
from mojolearn import Embedding
from mojolearn.tests._expose_d_harness import Report, bind_or_exit, mode, run

V, D, T = 37, 5, 211


def _data(seed=0):
    rng = np.random.default_rng(seed)
    w = rng.uniform(-1.0, 1.0, (V, D)).astype(np.float32)
    ids = rng.integers(0, V, T).astype(np.int32)
    dy = rng.uniform(-1.0, 1.0, (T, D)).astype(np.float32)
    return w, ids, dy


def _numpy_fold(ids, dy, pad=None):
    """contract 5.1 with NumPy: +0.0 seed, then dw[v] += dy[t] in ascending t."""
    dw = np.zeros((V, D), np.float32)
    for t in range(ids.shape[0]):
        if pad is not None and ids[t] == pad:
            continue
        dw[ids[t]] += dy[t]
    if pad is not None:
        dw[pad] = np.float32(0.0)
    return dw


def _bits(a):
    return np.ascontiguousarray(np.asarray(a, dtype=np.float32)).view(np.uint32)


def arm_forward_backward(rep):
    w, ids, dy = _data()
    e = Embedding(V, D, weight=w)
    y = np.asarray(e.forward(ids))
    rep.check("FOLD", y.shape == (T, D), "forward shape (T, d)", y.shape)
    rep.check("FOLD", np.array_equal(_bits(y), _bits(w[ids])), "forward is weight[ids] bit for bit")
    y2 = np.asarray(e(ids.reshape(1, T)))
    rep.check("FOLD", y2.shape == (1, T, D), "a 2-D ids keeps its shape plus d", y2.shape)
    dw = np.asarray(e.backward(ids, dy))
    ref = _numpy_fold(ids, dy)
    if mode() == "identical":
        rep.check("FOLD", np.array_equal(_bits(dw), _bits(ref)), "backward equals NumPy's ascending += fold bit for bit")
    else:
        rep.report_only("FOLD", np.array_equal(_bits(dw), _bits(ref)), "backward against NumPy's ascending fold")
    empty_rows = sorted(set(range(V)) - set(ids.tolist()))
    rep.check("FOLD", all(np.all(_bits(dw[r]) == 0) for r in empty_rows), "rows with no id are +0.0 stored", empty_rows[:4])
    dw2 = np.asarray(Embedding(V, D, weight=w).backward(ids, dy))
    rep.check("FOLD", np.array_equal(_bits(dw), _bits(dw2)), "two calls agree bit for bit on this box")


def arm_padding_and_carry(rep):
    w, ids, dy = _data(1)
    pad = int(ids[3])
    e = Embedding(V, D, padding_idx=pad, weight=w)
    dw = np.asarray(e.backward(ids, dy))
    rep.check("PAD", np.all(_bits(dw[pad]) == 0), "row padding_idx is +0.0 stored")
    rep.check("PAD", np.array_equal(_bits(dw), _bits(_numpy_fold(ids, dy, pad))), "padded fold equals NumPy's with the pad positions dropped")
    rep.check("PAD", np.array_equal(_bits(np.asarray(e.forward(ids))), _bits(w[ids])), "the forward gathers the padding row like any other")
    neg = Embedding(V, D, padding_idx=pad - V, weight=w)
    rep.check("PAD", neg.padding_idx == pad, "a negative padding_idx counts from the end", neg.padding_idx)
    for t0 in (1, 70, 150, T - 1):
        g1 = e.backward(ids[:t0], dy[:t0])
        g2 = np.asarray(e.backward(ids[t0:], dy[t0:], grad=g1))
        rep.check("CARRY", np.array_equal(_bits(g2), _bits(dw)), f"the carried split at t0={t0} reproduces the unsplit gradient")
    g1 = np.asarray(e.backward(ids[:70], dy[:70]))
    before = g1.copy()
    e.backward(ids[70:], dy[70:], grad=g1)
    rep.check("CARRY", np.array_equal(_bits(g1), _bits(before)), "grad= is copied, never written")


def arm_plan_sort(rep):
    """plan="sort" is PLAN_SORT (contract 6.2), the device total-key sort. The plan
    is an execution plan and not the specification, so every gradient below must
    equal plan="scan"'s bit for bit: plain, padded, carried, and a degenerate
    one-id run whose length is not a power of two (the sort's sentinel slack)."""
    w, ids, dy = _data(2)
    scan = Embedding(V, D, weight=w)
    sort = Embedding(V, D, weight=w, plan="sort")
    rep.check("PLAN", scan.plan == "scan" and sort.plan == "sort", "plan is stored, default 'scan'", (scan.plan, sort.plan))
    dw_scan = np.asarray(scan.backward(ids, dy))
    dw_sort = np.asarray(sort.backward(ids, dy))
    rep.check("PLAN", np.array_equal(_bits(dw_sort), _bits(dw_scan)), "plan='sort' backward equals plan='scan' bit for bit")
    if mode() == "identical":
        rep.check("PLAN", np.array_equal(_bits(dw_sort), _bits(_numpy_fold(ids, dy))), "plan='sort' equals NumPy's ascending += fold bit for bit")
    pad = int(ids[7])
    p_scan = np.asarray(Embedding(V, D, padding_idx=pad, weight=w).backward(ids, dy))
    p_sort = np.asarray(Embedding(V, D, padding_idx=pad, weight=w, plan="sort").backward(ids, dy))
    rep.check("PLAN", np.array_equal(_bits(p_sort), _bits(p_scan)), "padded plan='sort' equals padded plan='scan' bit for bit")
    rep.check("PLAN", np.all(_bits(p_sort[pad]) == 0), "plan='sort' stores +0.0 in row padding_idx")
    g1 = sort.backward(ids[:97], dy[:97])
    g2 = np.asarray(sort.backward(ids[97:], dy[97:], grad=g1))
    rep.check("PLAN", np.array_equal(_bits(g2), _bits(dw_scan)), "plan='sort' carried split at t0=97 reproduces the unsplit scan gradient")
    hot = np.full(T, 5, np.int32)
    rep.check("PLAN", np.array_equal(_bits(np.asarray(sort.backward(hot, dy))), _bits(np.asarray(scan.backward(hot, dy)))),
              "one id at every position (R = T = 211, not a power of two) agrees across plans")
    rep.check("PLAN", np.array_equal(_bits(np.asarray(sort.forward(ids))), _bits(w[ids])), "the forward ignores plan")
    for bad in ("radix", "SORT", 1, None):
        rep.raises("PLAN", ValueError, "plan must be", f"plan={bad!r} refused by name", Embedding, V, D, weight=w, plan=bad)
    fp = Embedding.from_pretrained(w, plan="sort")
    rep.check("PLAN", fp.plan == "sort", "from_pretrained carries plan", fp.plan)


def arm_refusals(rep):
    w, ids, dy = _data()
    rep.raises("REFUSE", ValueError, "max_norm", "max_norm by name", Embedding, V, D, max_norm=1.0, weight=w)
    rep.raises("REFUSE", ValueError, "scale_grad_by_freq", "scale_grad_by_freq by name", Embedding, V, D, scale_grad_by_freq=True, weight=w)
    rep.raises("REFUSE", ValueError, "sparse", "sparse by name", Embedding, V, D, sparse=True, weight=w)
    rep.raises("REFUSE", ValueError, "weight is required", "a missing weight by name", Embedding, V, D)
    rep.raises("REFUSE", ValueError, "padding_idx", "padding_idx out of range", Embedding, V, D, padding_idx=V, weight=w)
    rep.raises("REFUSE", ValueError, "weight has shape", "a weight of the wrong shape", Embedding, V, D + 1, weight=w)
    e = Embedding(V, D, weight=w)
    bad = ids.copy(); bad[5] = V
    rep.raises("REFUSE", Exception, "REFUSED", "an id at V, refused on the Mojo host (contract 8)", e.forward, bad)
    bad[5] = -1
    rep.raises("REFUSE", Exception, "REFUSED", "an id at -1 in the backward, refused on the Mojo host", e.backward, bad, dy)
    nan = dy.copy(); nan[2, 1] = np.float32("nan")
    rep.raises("REFUSE", Exception, "NaN in dY", "a NaN in dY, refused on the Mojo host (contract 9.1)", e.backward, ids, nan)
    inf_w = w.copy(); inf_w[0, 0] = np.float32("inf")
    rep.raises("REFUSE", Exception, "infinity in W", "an infinity in W, refused on the Mojo host", Embedding(V, D, weight=inf_w).forward, ids)
    rep.raises("REFUSE", ValueError, "dy has shape", "dy of the wrong shape", e.backward, ids, dy[:, :2])


def arm_provenance(rep):
    rep.check("PROVENANCE", "Embedding" in mojolearn.__all__ and "embedding" in mojolearn.__all__, "Embedding and mojolearn.embedding exported")
    rep.check("PROVENANCE", "Embedding" not in mojolearn._NOT_YET, "Embedding is no longer a named absence")
    w, _, _ = _data()
    rep.check("PROVENANCE", Embedding(V, D, weight=w).numeric_mode_used() == mode(), "numeric_mode_used() is the process default")


def main(out=sys.stdout):
    bind_or_exit("_mojolearn_embedding", "build_embedding.sh")
    rep = Report("test_embedding_surface")
    return run("test_embedding_surface", [("FOLD", arm_forward_backward), ("PAD", arm_padding_and_carry),
                                          ("PLAN", arm_plan_sort),
                                          ("REFUSE", arm_refusals), ("PROVENANCE", arm_provenance)], rep, out)


if __name__ == "__main__":
    sys.exit(main())
