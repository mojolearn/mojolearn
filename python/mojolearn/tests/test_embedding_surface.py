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


# The FOLD, PAD/CARRY and PLAN arithmetic (forward = W[ids], the ascending
# += fold, the +0.0 padding row, carried splits, plan="sort" against
# plan="scan") moved to checks/native_tests/embedding_surface.mojo
# (lane tmig-lm-1). The wiring of those arms stays here.
def arm_wiring(rep):
    w, ids, dy = _data(1)
    pad = int(ids[3])
    neg = Embedding(V, D, padding_idx=pad - V, weight=w)
    rep.check("PAD", neg.padding_idx == pad, "a negative padding_idx counts from the end", neg.padding_idx)
    e = Embedding(V, D, padding_idx=pad, weight=w)
    g1 = np.asarray(e.backward(ids[:70], dy[:70]))
    before = g1.copy()
    e.backward(ids[70:], dy[70:], grad=g1)
    rep.check("CARRY", np.array_equal(before.view(np.uint32), g1.view(np.uint32)), "grad= is copied, never written")
    scan = Embedding(V, D, weight=w, plan="scan")
    sort = Embedding(V, D, weight=w, plan="sort")
    auto = Embedding(V, D, weight=w)
    rep.check("PLAN", scan.plan == "scan" and sort.plan == "sort" and auto.plan == "auto", "plan is stored, default 'auto'", (scan.plan, sort.plan, auto.plan))
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
    return run("test_embedding_surface", [("WIRING", arm_wiring),
                                          ("REFUSE", arm_refusals), ("PROVENANCE", arm_provenance)], rep, out)


if __name__ == "__main__":
    sys.exit(main())
