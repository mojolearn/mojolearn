# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The gate for the Python surface of `mojolearn.IVFIndex` (2026-09-14): a
gate on the WIRING (the params list, the distances, ids and candidate counts
landing in the caller's arrays, every refusal by name), with brute force in
NumPy formerly the reference for the neighbor sets. The arithmetic, the list
layout, brute force at full probe, the rooted metric, the short-fill and the
repeat build are gated by `pixi run check-ivf`; the sample-filter short-fill
by `pixi run check-native-ann-ivf-surface-gpu`
(`checks/native_tests/ann_ivf_surface.mojo`).

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
    d, i = (np.asarray(a) for a in m.search(q))
    cand = np.asarray(m.n_candidates_)
    rep.check("SEARCH", d.shape == (32, 5) and d.dtype == np.float32, "distances float32 (m, k)", (d.shape, d.dtype))
    rep.check("SEARCH", i.shape == (32, 5) and i.dtype == np.int32, "indices int32 (m, k)", (i.shape, i.dtype))
    rep.check("SEARCH", cand.shape == (32,), "n_candidates_ (m,)", cand.shape)


def arm_euclidean(rep):
    x, q = _data()
    m = IVFIndex(n_lists=8, n_probes=8, n_neighbors=5, metric="euclidean", random_state=1).fit(x)
    ed, ei = (np.asarray(a) for a in m.search(q))
    rep.check("EUCLIDEAN", ed.shape == (32, 5) and ed.dtype == np.float32 and ei.dtype == np.int32, "distances float32 and indices int32 (m, k)", (ed.shape, ed.dtype, ei.dtype))
    rep.check("EUCLIDEAN", IVFIndex(n_lists=8, n_probes=2, n_neighbors=5, metric=1, random_state=1).fit(x).search(q)[0].shape == (32, 5), "metric code 1 (L2SqrtExpanded) accepted")


def arm_short_fill(rep):
    """A short index publishes (m, k) outputs; the short-fill values are
    gated in Mojo (see the module docstring)."""
    x, q = _data()
    small = np.ascontiguousarray(x[:16])
    for metric in ("sqeuclidean", "euclidean"):
        m = IVFIndex(n_lists=2, n_probes=2, n_neighbors=20, metric=metric, random_state=1).fit(small)
        d, i = (np.asarray(a) for a in m.search(q))
        rep.check("SHORT", d.shape == (32, 20) and i.shape == (32, 20), f"{metric}: (m, k) outputs for a short index", (d.shape, i.shape))


def arm_refusals(rep):
    x, q = _data()
    # Validation may refuse during construction or fit, before search exists.
    # Defer the complete operation so every named refusal reaches the checker.
    rep.raises("REFUSE", ValueError, "metric", "metric='cosine' by name", lambda: IVFIndex(n_lists=4, n_probes=1, metric="cosine").fit(x).search(q))
    rep.raises("REFUSE", ValueError, "metric code", "an unknown integer metric code", lambda: IVFIndex(n_lists=4, n_probes=1, metric=7).fit(x).search(q))
    rep.raises("REFUSE", ValueError, "metric codes", "metric code 2 (neither L2 code)", lambda: IVFIndex(n_lists=4, n_probes=1, metric=2).fit(x).search(q))
    rep.raises("REFUSE", TypeError, "n_lists", "n_lists as a float", lambda: IVFIndex(n_lists=4.0, n_probes=1).fit(x).search(q))
    rep.raises("REFUSE", ValueError, "call fit", "search before fit", lambda: IVFIndex(n_lists=4, n_probes=1).search(q))
    rep.raises("REFUSE", ValueError, "features", "queries with another width", lambda: IVFIndex(n_lists=4, n_probes=1).fit(x).search(q[:, :4]))
    rep.raises("REFUSE", Exception, "", "n_probes > n_lists, refused on the Mojo host (policy 2)", lambda: IVFIndex(n_lists=4, n_probes=5).fit(x).search(q))


def arm_provenance(rep):
    rep.check("PROVENANCE", "IVFIndex" in mojolearn.__all__, "IVFIndex exported")
    rep.check("PROVENANCE", "IVFIndex" not in mojolearn._NOT_YET, "IVFIndex is no longer a named absence")
    rep.check("PROVENANCE", IVFIndex(n_lists=4, n_probes=1).numeric_mode_used() == mode(), "numeric_mode_used() is the process default")


def main(out=sys.stdout):
    bind_or_exit("_mojolearn_ivf", "build_ivf.sh")
    rep = Report("test_ivf_surface")
    return run("test_ivf_surface", [("SEARCH", arm_search), ("EUCLIDEAN", arm_euclidean), ("SHORT", arm_short_fill), ("REFUSE", arm_refusals), ("PROVENANCE", arm_provenance)], rep, out)


if __name__ == "__main__":
    sys.exit(main())
