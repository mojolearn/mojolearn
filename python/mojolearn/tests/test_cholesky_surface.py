# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The gate for the Python surface of `mojolearn.Cholesky` (workstream D,
2026-09-14): the door of `cholesky/estimator.mojo` through
`bindings/_mojolearn_gp.mojo`. Nothing here re-gates the arithmetic
(`pixi run check-cholesky` does, with its sabotage arms); this is a gate on
the WIRING: the two params lists, the four scalars, `info` traveling from
fit to solve, and every refusal made to fire by name.

    cd python && python3 -m mojolearn.tests.test_cholesky_surface

Exit 2 naming `bindings/build_gp.sh` when the binding is unbuilt. Written
on one Apple M4 with no built binary in the worktree; the first run is
owed on a box.
"""
import sys

import numpy as np

import mojolearn
from mojolearn import Cholesky
from mojolearn.tests._expose_d_harness import Report, bind_or_exit, mode, run


def _spd(n, seed=0):
    rng = np.random.default_rng(seed)
    m = rng.standard_normal((n, n)).astype(np.float64)
    a = m @ m.T + n * np.eye(n)
    return np.ascontiguousarray(a.astype(np.float32))


def arm_fit(rep):
    n = 24
    a = _spd(n)
    c = Cholesky(jitter=0.0).fit(a)
    rep.check("FIT", c.info_ == 0, "info_ == 0 on an SPD matrix", c.info_)
    L = np.asarray(c.L_)
    rep.check("FIT", L.shape == (n, n), "L_ is (n, n)", L.shape)
    rep.check("FIT", np.all(np.triu(L, 1) == 0.0), "strict upper triangle of L_ is +0.0")
    rec = (L.astype(np.float64) @ L.astype(np.float64).T)
    rep.check("FIT", np.max(np.abs(rec - a)) < 1e-3 * np.max(np.abs(a)), "L L^T reproduces A at 1e-3 relative",
              float(np.max(np.abs(rec - a))))
    ld = 2.0 * np.sum(np.log(np.diag(L).astype(np.float64)))
    rep.check("FIT", abs(c.logdet_ - ld) < 1e-3 * max(1.0, abs(ld)), "logdet_ matches 2 sum log diag(L) at 1e-3", (c.logdet_, ld))
    rep.check("FIT", c.jitter_ == 0.0 and c.nb_ >= 1 and c.n_ == n, "jitter_, nb_ and n_ read back", (c.jitter_, c.nb_, c.n_))
    d = Cholesky().fit(a)
    rep.check("FIT", d.jitter_ == d.profile_jitter() and d.jitter_ > 0.0, "the default jitter is the profile's pinned ridge, read from the binding",
              (d.jitter_, d.profile_jitter()))
    return c, a


def arm_solve(rep):
    c, a = arm_fit.result
    n = a.shape[0]
    rng = np.random.default_rng(1)
    b = rng.standard_normal((n, 3)).astype(np.float32)
    x = np.asarray(c.solve(b))
    rep.check("SOLVE", x.shape == (n, 3) and x.dtype == np.float32, "solve returns float32 (n, nrhs)", (x.shape, x.dtype))
    res = a.astype(np.float64) @ x.astype(np.float64) - b
    rep.check("SOLVE", np.max(np.abs(res)) < 1e-3 * np.max(np.abs(b)), "A X = B at 1e-3 relative", float(np.max(np.abs(res))))
    x1 = np.asarray(c.solve(b[:, 0]))
    rep.check("SOLVE", x1.shape == (n,), "a 1-D right-hand side comes back 1-D", x1.shape)
    same = np.array_equal(x1.view(np.uint32), x[:, 0].view(np.uint32))
    if mode() == "identical":
        rep.check("SOLVE", same, "one column solved alone equals that column of the batch, bit for bit")
    else:
        rep.report_only("SOLVE", same, "column alone vs in batch")


def arm_failed_factor(rep):
    n = 8
    a = -np.eye(n, dtype=np.float32)
    c = Cholesky(jitter=0.0).fit(a)
    rep.check("FAILED", c.info_ != 0, "a negative definite matrix reports info_ != 0 as a RESULT (DEVIATION 1634)", c.info_)
    rep.raises("FAILED", Exception, "FAILED", "solve against a failed factor is refused by name in Mojo", c.solve, np.ones((n,), np.float32))
    rep.raises("FAILED", ValueError, "info=", "logdet_ on a failed factor is refused by name", lambda: c.logdet_)


def arm_refusals(rep):
    a = _spd(8)
    rep.raises("REFUSE", ValueError, "square", "a non-square A", Cholesky().fit, a[:, :4])
    rep.raises("REFUSE", ValueError, "fit before solve", "solve before fit", Cholesky().solve, np.ones(8, np.float32))
    c = Cholesky(jitter=0.0).fit(a)
    rep.raises("REFUSE", ValueError, "rows", "B with the wrong row count", c.solve, np.ones((5,), np.float32))
    rep.raises("REFUSE", TypeError, "jitter", "a jitter that is not a number", Cholesky, "0.1")
    bad = a.copy(); bad[0, 1] += np.float32(1.0)
    rep.raises("REFUSE", Exception, "symmetric", "a non-symmetric matrix, refused on the Mojo host by name", Cholesky(jitter=0.0).fit, bad)
    nan = a.copy(); nan[2, 2] = np.float32("nan")
    rep.raises("REFUSE", Exception, "", "a NaN cell, refused on the Mojo host", Cholesky(jitter=0.0).fit, nan)
    rep.raises("REFUSE", Exception, "", "an unpinned jitter (0.5), refused on the Mojo host (DEVIATION 1637)", Cholesky(jitter=0.5).fit, a)


def arm_provenance(rep):
    c = Cholesky()
    rep.check("PROVENANCE", c.numeric_mode_used() == mode(), "numeric_mode_used() is the process default", (c.numeric_mode_used(), mode()))
    rep.check("PROVENANCE", "Cholesky" in mojolearn.__all__ and "Cholesky" in mojolearn.linalg.__all__, "Cholesky is exported from mojolearn and mojolearn.linalg")


def main(out=sys.stdout):
    bind_or_exit("_mojolearn_gp", "build_gp.sh")
    rep = Report("test_cholesky_surface")

    def fit_then_solve(r):
        arm_fit.result = arm_fit(r)
        arm_solve(r)

    return run("test_cholesky_surface", [("FIT+SOLVE", fit_then_solve), ("FAILED", arm_failed_factor),
                                         ("REFUSE", arm_refusals), ("PROVENANCE", arm_provenance)], rep, out)


if __name__ == "__main__":
    sys.exit(main())
