# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The gate for the Python surface of `mojolearn.IVFIndex` (2026-09-14): a
gate on the WIRING (the params list, the distances, ids and candidate counts
landing in the caller's arrays, every refusal by name), with brute force in
NumPy as the reference for the neighbor sets when every list is probed. The
arithmetic and the list layout are gated by `pixi run check-ivf`.

    cd python && python3 -m mojolearn.tests.test_ivf_surface

Exit 2 naming `bindings/build_ivf.sh` when unbuilt.
"""
import sys

import numpy as np

import mojolearn
from mojolearn import IVFIndex
from mojolearn.tests._expose_d_harness import Report, bind_or_exit, mode, run


def _data(seed=0):
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((1024, 8)).astype(np.float32)
    q = rng.standard_normal((32, 8)).astype(np.float32)
    return x, q


def arm_search(rep):
    x, q = _data()
    m = IVFIndex(n_lists=8, n_probes=2, n_neighbors=5, random_state=1).fit(x)
    d, i = m.search(q)
    d, i = np.asarray(d), np.asarray(i)
    cand = np.asarray(m.n_candidates_)
    rep.check("SEARCH", d.shape == (32, 5) and d.dtype == np.float32, "distances float32 (m, k)", (d.shape, d.dtype))
    rep.check("SEARCH", i.shape == (32, 5) and i.dtype == np.int32, "indices int32 (m, k)", (i.shape, i.dtype))
    rep.check("SEARCH", np.all((i >= 0) & (i < 1024)), "every id is a fit row")
    rep.check("SEARCH", np.all(np.diff(d, axis=1) >= 0), "distances ascend along each query")
    rep.check("SEARCH", cand.shape == (32,) and np.all(cand >= 5) and np.all(cand <= 1024), "n_candidates_ (m,) within [k, n]", (cand.min(), cand.max()))
    full = IVFIndex(n_lists=8, n_probes=8, n_neighbors=5, random_state=1).fit(x)
    fd, fi = (np.asarray(a) for a in full.search(q))
    brute = np.argsort(((q[:, None, :].astype(np.float64) - x[None, :, :]) ** 2).sum(-1), axis=1, kind="stable")[:, :5]
    rep.check("SEARCH", np.array_equal(np.sort(fi, axis=1), np.sort(brute.astype(np.int32), axis=1)), "probing every list returns brute force's neighbor sets")
    rep.check("SEARCH", np.all(np.asarray(full.n_candidates_) == 1024), "probing every list examines every row")
    d2, i2 = (np.asarray(a) for a in IVFIndex(n_lists=8, n_probes=2, n_neighbors=5, random_state=1).fit(x).search(q))
    same = np.array_equal(i, i2) and np.array_equal(d.view(np.uint32), d2.view(np.uint32))
    if mode() == "identical":
        rep.check("SEARCH", same, "two builds and searches agree bit for bit on this box")
    else:
        rep.report_only("SEARCH", same, "two builds and searches")


def arm_refusals(rep):
    x, q = _data()
    rep.raises("REFUSE", ValueError, "metric", "metric='cosine' by name", IVFIndex(n_lists=4, n_probes=1, metric="cosine").fit(x).search, q)
    rep.raises("REFUSE", ValueError, "metric code", "an unknown integer metric code", IVFIndex(n_lists=4, n_probes=1, metric=7).fit(x).search, q)
    rep.raises("REFUSE", ValueError, "L2SqrtExpanded", "metric='euclidean' refused by name (all-zero distances on the M4, 2026-09-14)", IVFIndex(n_lists=4, n_probes=1, metric="euclidean").fit(x).search, q)
    rep.raises("REFUSE", ValueError, "L2SqrtExpanded", "metric code 1 refused by name", IVFIndex(n_lists=4, n_probes=1, metric=1).fit(x).search, q)
    rep.raises("REFUSE", TypeError, "n_lists", "n_lists as a float", IVFIndex(n_lists=4.0, n_probes=1).fit(x).search, q)
    rep.raises("REFUSE", ValueError, "call fit", "search before fit", IVFIndex(n_lists=4, n_probes=1).search, q)
    rep.raises("REFUSE", ValueError, "features", "queries with another width", IVFIndex(n_lists=4, n_probes=1).fit(x).search, q[:, :4])
    rep.raises("REFUSE", Exception, "", "n_probes > n_lists, refused on the Mojo host (policy 2)", IVFIndex(n_lists=4, n_probes=5).fit(x).search, q)


def arm_provenance(rep):
    rep.check("PROVENANCE", "IVFIndex" in mojolearn.__all__, "IVFIndex exported")
    rep.check("PROVENANCE", "IVFIndex" not in mojolearn._NOT_YET, "IVFIndex is no longer a named absence")
    rep.check("PROVENANCE", IVFIndex(n_lists=4, n_probes=1).numeric_mode_used() == mode(), "numeric_mode_used() is the process default")


def main(out=sys.stdout):
    bind_or_exit("_mojolearn_ivf", "build_ivf.sh")
    rep = Report("test_ivf_surface")
    return run("test_ivf_surface", [("SEARCH", arm_search), ("REFUSE", arm_refusals), ("PROVENANCE", arm_provenance)], rep, out)


if __name__ == "__main__":
    sys.exit(main())
