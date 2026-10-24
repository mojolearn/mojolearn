# SPDX-License-Identifier: Apache-2.0
"""IterativeImputer runs as one native call per public method
(bindings/prep_iterimp_native.mojo, through the prep family door). The former
Python route is kept below as the oracle: the door must give the same bytes and
the same fitted attributes on small fixtures."""
import array
import ast
import inspect
import copy
import pathlib

import numpy as np
import pytest

_HERE = pathlib.Path(__file__).resolve().parents[1]


def _method_bodies():
    tree = ast.parse((_HERE / "_expansion_prep.py").read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "IterativeImputer")
    return {f.name: f for f in cls.body if isinstance(f, ast.FunctionDef)}


def test_public_methods_are_one_native_call():
    """Pure source check: fit, fit_transform and transform each return one _ENCODE_CALL."""
    bodies = _method_bodies()
    for name in ("fit", "fit_transform", "transform"):
        body = [s for s in bodies[name].body
                if not (isinstance(s, ast.Expr) and isinstance(s.value, ast.Constant))]
        assert len(body) == 1 and isinstance(body[0], ast.Return), name
        call = body[0].value
        assert isinstance(call, ast.Call) and getattr(call.func, "id", "") == "_ENCODE_CALL", name
        assert isinstance(call.args[1], ast.Constant) and call.args[1].value == name, name
    assert set(bodies) >= {"fit", "fit_transform", "transform"}
    # the former helpers are gone from the class
    assert not set(bodies) & {"_fit_user", "_ii_take", "_ii_put", "_prepare", "_neighbours_device"}


# ---------------------------------------------------------------- the former Python route (oracle)
try:
    import mojolearn as ml
    from mojolearn._expansion_prep import (
        Array, SimpleImputer, _Prog, _x2d, _cs, _mode, _seed_words, _NONE, _II_CH, as_f32_c)
except Exception:  # no built bindings: only the source check above runs
    ml = None
numbers = __import__("numbers")
_INF = float("inf")
needs_bindings = pytest.mark.skipif(ml is None, reason="needs the built mojolearn bindings")


def _splitmix64(state):
    state = (state + 0x9E3779B97F4A7C15) & 0xFFFFFFFFFFFFFFFF
    z = state
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & 0xFFFFFFFFFFFFFFFF
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & 0xFFFFFFFFFFFFFFFF
    return state, z ^ (z >> 31)




def _mark_missing(pr, xo, count, missing_values):
    """Stages that turn a numeric `missing_values` into NaN; returns the offset
    to read (the input itself when missing_values is NaN)."""
    if missing_values is None or _is_nan_value(missing_values):
        return xo
    val = pr.put_scalar(missing_values)
    out = pr.alloc(count)
    pr.stage("mark_missing", count, xo, count, val, out)
    return out




class _NeighbourDraws:
    """IterativeImputer's n_nearest_features draws, one list per (round,
    feature) step in fit order. Every step is drawn up front in one device
    program (`_neighbours_device`, every tier: lane cpu2-l3-prep deleted the
    host route that read the |corr| matrix back and called the base binding's
    `weighted_pick_i32` per step); `next` then hands them out and moves the
    instance's splitmix64 state past the words the steps taken so far own
    (k per step: state + k * golden each)."""

    def __init__(self, imp, Xf, n, dk, mode, orders):
        self.imp, self.dk, self.at = imp, dk, 0
        js = [j for order in orders for j in order]  # glue: the (round, feature) step list
        self.k, self.start = int(imp.n_nearest_features), imp._rng
        self.lists = imp._neighbours_device(Xf, n, dk, mode, js, self.k) if js else []

    def next(self, j):
        out = self.lists[self.at]
        self.at += 1
        self.imp._rng = (self.start + self.at * self.k * 0x9E3779B97F4A7C15) & 0xFFFFFFFFFFFFFFFF
        return out




def _clone(est):
    """A fresh unfitted copy (the reference's `clone`: the same class and
    get_params(deep=False))."""
    return type(est)(**est.get_params(deep=False)) if hasattr(est, "get_params") else copy.deepcopy(est)




def _as_list(v):
    from . import _user_objects
    return _user_objects.call("floats", v.tolist() if hasattr(v, "tolist") else v).tolist()




def _pred_f64(v):
    """A user estimator's predictions as a float64 Array holding the values
    `_as_list` reads (`float(x)` per element): a float32 or float64 vector
    widened natively (exact), anything else through `_as_list`."""
    from ._buffer import _materialize, _has_buffer, as_f64_c
    if isinstance(v, Array) or (not isinstance(v, (list, tuple)) and _has_buffer(v)):
        a, _ = _materialize(v, "predict")
        if a.ndim == 1 and a.dtype in ("<f4", "<f8") and a.size:
            return as_f64_c(a, ndim=1, name="predict")[0]
    vals = _as_list(v)
    return Array._owned(array.array("d", vals), (len(vals),), "<f8", "C")




class _Former(ml.IterativeImputer if ml is not None else object):
    """The Python route as it stood before the native door."""

    def _refuse(self):
        mi = self.max_iter
        if isinstance(mi, bool) or not isinstance(mi, numbers.Integral) or mi < 0:
            raise ValueError("The 'max_iter' parameter of IterativeImputer must be an int in the range "
                             f"[0, inf). Got {mi!r} instead.")
        if self.imputation_order not in ("ascending", "descending", "roman", "arabic", "random"):
            raise ValueError(f"mojolearn: invalid imputation_order {self.imputation_order!r}")
        nnf = self.n_nearest_features
        if nnf is not None and (not isinstance(nnf, numbers.Integral) or nnf < 1):
            raise ValueError(f"mojolearn: n_nearest_features must be an int >= 1; got {nnf!r}")
        if self.sample_posterior and self.estimator is not None and \
                "return_std" not in inspect.signature(self.estimator.predict).parameters:
            raise ValueError("mojolearn: If 'sample_posterior' is True, the estimator must support "
                             "'return_std' in its 'predict' method.")

    def _bounds(self, d):
        def per(v):
            vals = list(v) if isinstance(v, (list, tuple)) or hasattr(v, "tolist") else [v] * d
            vals = vals.tolist() if hasattr(vals, "tolist") else vals
            return [float(x) for x in vals]  # glue: converts the min_value / max_value arguments (vals-sized: per-column bound arguments)
        lo, hi = per(self.min_value), per(self.max_value)
        return [v for pair in zip(lo, hi) for v in pair]  # glue: interleaves the bound arguments (lo-sized: per-column bound arguments)

    def _prepare(self, pr, arr, Xf, inout=False):
        """Arena: the filled block, its missing mask, the per-column bounds.
        inout: the program imputes the filled block in place and returns it."""
        n, d = arr.shape
        dk = len(self._keep)
        xo = _mark_missing(pr, pr.put(arr), n * d, self.missing_values)
        fo = pr.put(Xf, inout=inout)
        ko = pr.put_list(self._keep)
        mo = pr.alloc(n * dk)
        pr.stage("nan_mask", n * dk, xo, n, d, ko, dk, mo)
        bo = pr.put_list(self._bounds_k)
        return fo, mo, bo

    def _draw(self):
        """The next splitmix64 word of the instance's draw stream."""
        self._rng, z = _splitmix64(self._rng)
        return z

    def _neighbours_device(self, Xf, n, dk, mode, js, k):
        """Lane fam2-prep-metrics (every tier since lane cpu2-l3-prep): every
        (round, feature) step's n_nearest_features draw in ONE program: the
        |corr| matrix (ii_mean, ii_gram, p2m_abscorr_*) stays on the device and one
        thread per step draws its k features from it (f2_wpick,
        x_prep/fam2.mojo; step c takes splitmix64 words c*k .. c*k + k - 1 of
        the instance's stream). Returns each step's ascending picks."""
        pr = _Prog()
        fo, mz = pr.put(Xf), pr.alloc(n * dk)
        means, cnt, g, flag = pr.alloc(dk), pr.alloc(1), pr.alloc(dk * dk), pr.alloc(1)
        pr.stage("ii_mean", dk, fo, n, dk, mz, 0, means, cnt, flag)
        pr.stage("ii_gram", dk * dk, fo, n, dk, mz, 0, means, g, flag)
        m = pr.alloc(dk * dk)
        pr.stage("p2m_abscorr_cell", dk * dk, g, dk, m)
        pr.stage("p2m_abscorr_norm", dk, m, dk)
        calls = len(js)
        fl, out = pr.work(calls * dk), pr.alloc(calls * (k + 1))
        pr.stage("f2_wpick", calls, m, dk, pr.put_ints(js), k, _seed_words(pr, self._rng), fl, out)
        pr.run(mode)
        words = pr.get_i32(out, calls * (k + 1)).tolist()
        lists = []
        for c in range(calls):  # glue: one feature list per step
            got = words[c * (k + 1) + k]
            if got < k:
                raise ValueError("mojolearn: IterativeImputer: no feature left to draw")
            lists.append(words[c * (k + 1):c * (k + 1) + got])
        return lists

    def fit_transform(self, X, y=None):
        self._refuse()
        arr = _x2d(X)
        n, d = arr.shape
        mode = _mode()
        self._rng = 0 if self.random_state is None else int(self.random_state) & 0xFFFFFFFFFFFFFFFF
        self._post_step = 0
        self.initial_imputer_ = SimpleImputer(missing_values=self.missing_values, strategy=self.initial_strategy,
                                              fill_value=self.fill_value,
                                              keep_empty_features=self.keep_empty_features).fit(arr)
        Xf = self.initial_imputer_.transform(arr)
        self._keep = list(self.initial_imputer_._keep)
        dk = len(self._keep)
        bounds = self._bounds(d)
        self._bounds_k = [bounds[2 * c + h] for c in self._keep for h in (0, 1)]  # glue: per-kept-column bound arguments (_keep-sized: kept feature indices)
        # missing counts per kept column, and the tolerance scale, from the device
        pr = _Prog()
        xo = _mark_missing(pr, pr.put(arr), n * d, self.missing_values)
        st = pr.alloc(6 * d)
        _cs(pr, mode, xo, n, d, st)
        rounds = int(self.max_iter)
        # lane cpu2-l3-prep: the missing counts (c2_ii_miss), each round's imputation
        # order (c2_ii_pos + c2_ii_ord, or c2_ii_rand: the reference's Fisher-Yates on the
        # instance's splitmix64 stream, one round a thread) and the tolerance scale
        # (c2_colmax) on the device; the orders come back as the fit's control lists
        R = max(rounds, 1)
        ordw, lens, sc = pr.alloc(max(R * dk, 1)), pr.alloc(R), pr.alloc(1)
        pr.stage("c2_colmax", 1, st + 5 * d, d, sc)
        if dk:
            missd = pr.work(dk)
            pr.stage("c2_ii_miss", dk, st, pr.put_ints(self._keep), n, missd)
            if self.imputation_order == "random":
                pr.stage("c2_ii_rand", R, missd, dk, 1 if self.skip_complete else 0, _seed_words(pr, self._rng),
                         ordw, lens)
            else:
                pos = pr.work(dk)
                mcode = {"ascending": 0, "descending": 1, "roman": 2, "arabic": 3}[self.imputation_order]
                pr.stage("c2_ii_pos", dk, missd, dk, mcode, pos)
                pr.stage("c2_ii_ord", dk, missd, dk, pos, R, ordw, lens)
        pr.run(mode)
        scale = pr.values(sc, 1)[0]
        if scale == _INF:
            # the device's largest |observed value| (c2_colmax) is infinite
            # exactly when an observed entry is; scikit-learn refuses it
            raise ValueError("Input X contains infinity or a value too large for dtype('float64').")
        self._indicator = [j for j, c in enumerate(pr.values(st, d)) if int(c) < n] if self.add_indicator else []  # glue: indicator column index list from device counts (d-sized: feature count)
        ln = pr.get_i32(lens, R).tolist() if dk else [0] * R
        ow = pr.get_i32(ordw, R * dk).tolist() if dk else []
        orders = [ow[r * dk:r * dk + ln[r]] for r in range(rounds)]  # glue: the device's orders as control lists
        self.n_features_with_missing_ = ln[0]
        if self.imputation_order == "random":
            # the draws the rounds took: (m - 1) a round over m candidates
            m = ln[0] if self.skip_complete else dk
            self._rng = (self._rng + rounds * max(m - 1, 0) * 0x9E3779B97F4A7C15) & 0xFFFFFFFFFFFFFFFF
        self.numeric_mode_, self.n_features_in_ = mode, d
        nnf = self.n_nearest_features
        corr = _NeighbourDraws(self, Xf, n, dk, mode, orders) if nnf is not None and nnf < dk else None
        if self.estimator is not None:
            return self._with_indicator(arr, self._fit_user(arr, Xf, orders, corr, float(self.tol) * scale))
        pr = _Prog()
        fo, mo, bo = self._prepare(pr, arr, Xf, inout=True)
        tol = pr.put_scalar(float(self.tol) * scale)
        flag, niter = pr.alloc(1), pr.alloc(1)
        prev = pr.alloc(n * dk)
        cnt, g = pr.alloc(1), pr.alloc(dk * dk)
        p1 = max(dk - 1, 1)
        gs, eig, vec, w = pr.alloc(p1 * p1), pr.alloc(p1), pr.alloc(p1 * p1), pr.alloc(p1)
        seq, extra = [], []
        seed = self._rng & 0x7FFFFFFF
        # the reference checks convergence only without sample_posterior
        conv = not self.sample_posterior
        rowabs = None
        for r in range(rounds):  # glue: drives the device imputation rounds (rounds-sized: imputation rounds)
            if orders[r] and conv:
                pr.stage("ii_snapshot", n * dk, fo, prev, flag)
            for j in orders[r]:  # glue: drives one device regression per feature (orders-sized: per-round feature orders)
                coef, inter, means = pr.alloc(dk), pr.alloc(1), pr.alloc(dk)
                nbl = corr.next(j) if corr is not None else None
                nb1 = pr.put_list([1 if a in nbl else 0 for a in range(dk)]) + 1 if nbl is not None else 0  # glue: neighbor-feature mask of one step (dk-sized: kept feature count)
                pp = len(nbl) if nbl is not None else dk - 1
                al = pr.alloc(2) if self.sample_posterior else -1
                seq.append((j, coef, inter))
                pr.stage("ii_mean", dk, fo, n, dk, mo, j, means, cnt, flag)
                pr.stage("ii_gram", dk * dk, fo, n, dk, mo, j, means, g, flag)
                if pp > 0:
                    pr.stage("ii_sub", 1, g, dk, j, gs, flag, nb1)
                    pr.stage("eigh", 1, gs, pp, 0, eig, vec, 1)  # q[5]=1: cyclic eigh_unit, not round-robin
                pr.stage("ii_br", 1, g, dk, j, eig, vec, means, cnt, coef, inter, flag, w, nb1, al + 1)
                if self.sample_posterior:
                    sig = pr.alloc(max(pp, 1) ** 2)
                    pr.stage("ii_sigma", pp * pp, eig, vec, pp, al, sig, flag)
                    key = pr.put_ints([seed, self._post_step])
                    self._post_step += 1
                    pr.stage("ii_post", n, fo, n, dk, mo, j, coef, inter, bo, flag, means, sig, al, nb1, key)
                    extra.append((nbl, sig, al, means, pp))
                else:
                    pr.stage("ii_predict", n, fo, n, dk, mo, j, coef, inter, bo, flag)
            if orders[r] and conv:
                if rowabs is None:
                    rowabs = pr.alloc(n)
                pr.stage("ii_rowabs", n, fo, prev, dk, rowabs, flag)
                pr.stage("ii_conv", 1, fo, prev, n * dk, tol, flag, niter, dk, rowabs + 1)
        pr.run(mode)
        any_order = any(orders)
        done = (int(pr.values(niter, 1)[0]) if conv else rounds) if any_order else 0
        self.n_iter_ = done if any_order else min(1, rounds)
        steps = sum(len(o) for o in orders[:done])  # glue: count of finished imputation steps (orders-sized: per-round feature orders)
        self.imputation_sequence_ = [(j, pr.get(c, dk), pr.get(i, 1)) for j, c, i in seq[:steps]]  # glue: fitted imputation sequence views (seq-sized: finished imputation steps)
        self._posterior = [(nbl, pr.get(sg, max(pp, 1) ** 2), pr.get(al, 2), pr.get(mn, dk))
                           for nbl, sg, al, mn, pp in extra[:steps]]  # glue: fitted imputation sequence views (extra-sized: finished imputation steps)
        return self._with_indicator(arr, pr.get(fo, (n, dk)))

    def _with_indicator(self, arr, Xt):
        """Xt, then MissingIndicator's columns (add_indicator)."""
        m = len(self._indicator)
        if not m:
            return Xt
        n, d = arr.shape
        pr = _Prog()
        xo = _mark_missing(pr, pr.put(arr), n * d, self.missing_values)
        mo = pr.alloc(n * m)
        pr.stage("nan_mask", n * m, xo, n, d, pr.put_list(self._indicator), m, mo)
        dk = Xt.shape[1]
        to = pr.put(Xt)
        hc = pr.output(n * (dk + m))
        pr.stage("hcat", n * (dk + m), to, dk, mo, m, hc)
        pr.run(self.numeric_mode_)
        return pr.get(hc, (n, dk + m))

    def _fit_user(self, arr, Xf, orders, corr, tol):
        """estimator=<any>: the reference's rounds over copies of the
        estimator, its fit / predict on the float32 blocks. The data-sized
        plumbing runs as x_prep units (`_ii_take`, `_ii_put`; the device on a
        GPU install, the same units on the host on a CPU-only one): the
        observed / missing rows of feature j by a chunked count, scan and
        write, the estimator's blocks by a gather, the clipped predictions by
        a scatter, and the round's stop by `ii_rowabs` / `ii_conv`, the
        default estimator's own test (the matrix inf-norm of the round's
        change < tol, float32)."""
        n, d = arr.shape
        dk = len(self._keep)
        mode = _mode()
        mask = self._mask_arr(arr)
        Xt = as_f32_c(Xf, name="X")[0].copy()
        seq = []
        done = 0
        for r, order in enumerate(orders):  # cpu-route: the user estimator, fitted once per imputation round (user code)
            check = not self.sample_posterior and bool(order)
            prev = Xt.copy() if check else None
            for j in order:  # cpu-route: the user estimator, fitted once per feature (user code)
                nbl = corr.next(j) if corr is not None else [a for a in range(dk) if a != j]  # glue: the other feature ids
                est = _clone(self.estimator)
                Xo, yo, Xm, rows, m = self._ii_take(Xt, mask, j, nbl, mode, fit=True)
                est.fit(Xo, yo)
                seq.append((j, nbl, est))
                Xt = self._ii_put(Xt, Xm, rows, m, j, est, mode)
            done = r + 1
            if check and self._ii_stop(Xt, prev, tol, mode):
                break
        self.n_iter_ = done if any(orders) else min(1, len(orders))
        self.imputation_sequence_ = seq
        return Xt

    def _ii_rows(self, pr, mo, n, dk, j, missing):
        """Stages: the ascending rows whose mask word for feature j is (or is
        not) zero, as int32 words, and their count (a float slot)."""
        ch = max(_II_CH, -(-n // _II_CH))
        nch = -(-n // ch)
        cnt, off, tot, rows = pr.alloc(nch), pr.alloc(nch), pr.alloc(1), pr.alloc(n)
        pr.stage("ii_rcount", nch, mo, n, dk, j, 1 if missing else 0, ch, cnt)
        pr.stage("uniq_scan", 1, cnt, nch, off, tot)
        pr.stage("ii_rwrite", nch, mo, n, dk, j, 1 if missing else 0, ch, off, rows)
        return rows, tot

    def _ii_take(self, Xt, mask, j, nbl, mode, fit):
        """One program: feature j's missing rows and the estimator's predict
        block Xt[mis][:, nbl] and, with fit, its training blocks
        Xt[obs][:, nbl] and Xt[obs, j]. Returns (Xo, yo, Xm, mis rows as int32
        words, m)."""
        n, dk = Xt.shape
        nc = len(nbl)
        pr = _Prog()
        xo, mo = pr.put(Xt), pr.put(mask)
        co = pr.put_list(nbl or [0])
        jo = pr.put_list([j])
        rm, tm = self._ii_rows(pr, mo, n, dk, j, True)
        gm = pr.alloc(n * nc)
        if nc:
            pr.stage("ii_gather", n * nc, xo, dk, rm, tm, co, nc, gm)
        if fit:
            ro, to = self._ii_rows(pr, mo, n, dk, j, False)
            go, gy = pr.alloc(n * nc), pr.alloc(n)
            if nc:
                pr.stage("ii_gather", n * nc, xo, dk, ro, to, co, nc, go)
            pr.stage("ii_gather", n, xo, dk, ro, to, jo, 1, gy)
        pr.run(mode)
        m = int(pr.values(tm, 1)[0])
        Xo = yo = None
        if fit:
            mob = int(pr.values(to, 1)[0])
            Xo, yo = self._ii_block(pr, go, gy, mob, nc)
        Xm, _ = self._ii_block(pr, gm, None, m, nc)
        rows = pr.get_i32(rm, (m,)) if m else None
        return Xo, yo, Xm, rows, m

    @staticmethod
    def _ii_block(pr, xo, yo, m, nc):
        """(X block (m, nc), y (m,)) as float32 Arrays; a 0-row block is
        `from_list([])`, a 0-column one its m empty rows, as before."""
        if m == 0:
            return Array.from_list([], "<f4"), (Array.from_list([], "<f4") if yo is not None else None)
        X = pr.get(xo, (m, nc)) if nc else Array((m, 0), "<f4")
        return X, (pr.get(yo, (m,)) if yo is not None else None)

    def _ii_put(self, Xt, Xm, rows, m, j, est, mode):
        """Predict on the missing rows' block, then (one program) the clipped
        float32 store into Xt[rows, j]. sample_posterior's truncated normal
        draw stays Python float64 on the pinned normal cdf / inverse cdf
        (DEVIATION 6902) and is stored unclipped."""
        if not m:
            return Xt
        n, dk = Xt.shape
        lo, hi = self._bounds_k[2 * j], self._bounds_k[2 * j + 1]
        if not self.sample_posterior:
            v = _pred_f64(est.predict(Xm))
            clip = 1
        else:
            mus, sig = est.predict(Xm, return_std=True)
            v = self._truncnorm_draws(mus, sig, lo, hi)
            clip = 0
        k = min(v.size, m)      # the reference's zip stops at the shorter
        if k == 0:
            return Xt
        if v.size != k:
            v = Array._owned(array.array("d", v.tolist()[:k]), (k,), "<f8", "C")
        pr = _Prog()
        xo = pr.put(Xt, inout=True)
        ro = pr.put_words(rows)
        vo = pr.put(v.astype("<f4"))
        bo = pr.put_list([lo, hi])
        pr.stage("ii_scatter", k, xo, dk, ro, j, vo, bo, clip)
        pr.run(mode)
        return pr.get(xo, (n, dk))

    def _ii_stop(self, Xt, prev, tol, mode):
        """The round's stop: the inf-norm of Xt - prev below tol."""
        n, dk = Xt.shape
        pr = _Prog()
        xo, po = pr.put(Xt), pr.put(prev)
        to = pr.put_scalar(tol)
        flag, niter, rowabs = pr.alloc(1), pr.alloc(1), pr.alloc(n)
        pr.stage("ii_rowabs", n, xo, po, dk, rowabs, flag)
        pr.stage("ii_conv", 1, xo, po, n * dk, to, flag, niter, dk, rowabs + 1)
        pr.run(mode)
        return pr.values(flag, 1)[0] != 0

    def _mask_arr(self, arr):
        """`_mask`'s words (1.0 missing, 0.0 observed) as the float32 (n, dk) Array."""
        n, d = arr.shape
        pr = _Prog()
        xo = _mark_missing(pr, pr.put(arr), n * d, self.missing_values)
        mo = pr.alloc(n * len(self._keep))
        pr.stage("nan_mask", n * len(self._keep), xo, n, d, pr.put_list(self._keep), len(self._keep), mo)
        pr.run(_mode())
        return as_f32_c(pr.get(mo, (n, len(self._keep))), name="mask")[0]

    def _truncnorm_draws(self, mus, sig, lo, hi):
        """`_impute_one_feature`'s rule per entry, in Mojo (the core helper
        `truncnorm_draws`, bindings/normal_dist_helpers.mojo; lane py-runtime
        round 3, it was Python float64 per entry): mu beyond a bound -> the
        bound, sigma <= 0 -> mu, else the inversion of the truncated normal
        at a 53-bit splitmix64 uniform on the pinned normal cdf / inverse cdf
        (DEVIATION 6902), the instance's stream advanced by the entries that
        draw. The reference's zip stops at the shorter of the two outputs."""
        from ._buffer import _native
        from . import _user_objects
        # the user estimator's predictions read once natively (user_objects floats)
        mu = _user_objects.call("floats", _user_objects.call("flatten", mus))
        sd = _user_objects.call("floats", _user_objects.call("flatten", sig))
        k = min(len(mu), len(sd))
        out = array.array("d", bytes(8 * k))
        st = int(self._rng)
        hi_lo = _native("truncnorm_draws")(mu.buffer_info()[0], sd.buffer_info()[0], k, [float(lo), float(hi)],
                                           [st >> 32, st & 0xFFFFFFFF], out.buffer_info()[0])
        self._rng = (int(hi_lo[0]) << 32) | int(hi_lo[1])
        return Array._owned(out, (k,), "<f8", "C")

    def fit(self, X, y=None):
        self.fit_transform(X)
        return self

    def transform(self, X):
        if not hasattr(self, "imputation_sequence_"):
            raise RuntimeError("mojolearn: this IterativeImputer instance is not fitted yet")
        arr = _x2d(X)
        self._check_width(arr)
        n, d = arr.shape
        dk = len(self._keep)
        Xf = self.initial_imputer_.transform(arr)
        if self.estimator is not None:
            mode = self.numeric_mode_
            mask = self._mask_arr(arr)
            Xt = as_f32_c(Xf, name="X")[0].copy()
            for j, nbl, est in self.imputation_sequence_:  # glue: drives one device regression per step (imputation_sequence_-sized: fitted imputation steps)
                _, _, Xm, rows, m = self._ii_take(Xt, mask, j, nbl, mode, fit=False)
                Xt = self._ii_put(Xt, Xm, rows, m, j, est, mode)
            return self._with_indicator(arr, Xt)
        pr = _Prog()
        fo, mo, bo = self._prepare(pr, arr, Xf, inout=True)
        seed = self._rng & 0x7FFFFFFF
        for s, (j, coef, inter) in enumerate(self.imputation_sequence_):  # glue: drives one device regression per step (imputation_sequence_-sized: fitted imputation steps)
            co, io = pr.put(coef), pr.put(inter)
            if self.sample_posterior:
                nbl, sig, al, means = self._posterior[s]
                nb1 = pr.put_list([1 if a in nbl else 0 for a in range(dk)]) + 1 if nbl is not None else 0  # glue: neighbor-feature mask of one step (dk-sized: kept feature count)
                key = pr.put_ints([seed, self._post_step])
                self._post_step += 1
                pr.stage("ii_post", n, fo, n, dk, mo, j, co, io, bo, _NONE, pr.put(means), pr.put(sig), pr.put(al),
                         nb1, key)
            else:
                pr.stage("ii_predict", n, fo, n, dk, mo, j, co, io, bo, _NONE)
        pr.run(self.numeric_mode_)
        return self._with_indicator(arr, pr.get(fo, (n, dk)))




def _data(seed=0, n=200, d=5):
    rng = np.random.default_rng(seed)
    Z = rng.standard_normal((n, 2))
    X = (Z @ rng.standard_normal((2, d)) + 0.1 * rng.standard_normal((n, d))).astype(np.float32)
    X[rng.random((n, d)) < 0.15] = np.nan
    return X


def _same(a, b):
    a, b = np.asarray(a), np.asarray(b)
    assert a.shape == b.shape and a.dtype == b.dtype
    assert a.tobytes() == b.tobytes()


def _same_attrs(m, o):
    for name in ("n_iter_", "n_features_with_missing_", "n_features_in_", "numeric_mode_", "_keep", "_indicator",
                 "_rng", "_post_step"):
        assert getattr(m, name) == getattr(o, name), name
    assert [float(v) for v in m._bounds_k] == [float(v) for v in o._bounds_k]
    _same(m.initial_imputer_.statistics_, o.initial_imputer_.statistics_)
    assert len(m.imputation_sequence_) == len(o.imputation_sequence_)
    for s, t in zip(m.imputation_sequence_, o.imputation_sequence_):
        assert s[0] == t[0]
        if isinstance(s[1], list):
            assert s[1] == t[1]
        else:
            _same(s[1], t[1])
            _same(s[2], t[2])
    if hasattr(o, "_posterior"):
        assert len(m._posterior) == len(o._posterior)
        for s, t in zip(m._posterior, o._posterior):
            assert s[0] == t[0]
            for u, v in zip(s[1:], t[1:]):
                _same(u, v)


CASES = [
    dict(),
    dict(max_iter=3, imputation_order="descending", initial_strategy="median"),
    dict(min_value=-1.0, max_value=1.0, imputation_order="roman"),
    dict(imputation_order="random", random_state=3, add_indicator=True),
    dict(n_nearest_features=2, random_state=5, imputation_order="arabic"),
    dict(sample_posterior=True, random_state=4, max_iter=3, min_value=-0.5, max_value=0.5),
    dict(sample_posterior=True, n_nearest_features=3, random_state=6, max_iter=2),
    dict(max_iter=0),
    dict(skip_complete=True, imputation_order="random", random_state=1, missing_values=0.0),
]


@needs_bindings
@pytest.mark.parametrize("kw", CASES)
def test_door_matches_former_route(kw):
    X, Xh = _data(0), _data(1)
    if kw.get("missing_values") == 0.0:
        X, Xh = np.nan_to_num(X), np.nan_to_num(Xh)
    m, o = ml.IterativeImputer(**kw), _Former(**kw)
    _same(m.fit_transform(X), o.fit_transform(X))
    _same_attrs(m, o)
    _same(m.transform(Xh), o.transform(Xh))
    assert m._post_step == o._post_step and m._rng == o._rng
    assert ml.IterativeImputer(**kw).fit(X).n_iter_ == m.n_iter_


@needs_bindings
@pytest.mark.parametrize("kw", [dict(max_iter=3), dict(max_iter=2, add_indicator=True, n_nearest_features=3,
                                                       random_state=2)])
def test_door_matches_former_route_mojolearn_estimator(kw):
    X, Xh = _data(2), _data(3)
    m = ml.IterativeImputer(estimator=ml.Ridge(alpha=2.0), **kw)
    o = _Former(estimator=ml.Ridge(alpha=2.0), **kw)
    _same(m.fit_transform(X), o.fit_transform(X))
    assert m.n_iter_ == o.n_iter_ and m._rng == o._rng
    assert [(j, nbl) for j, nbl, _ in m.imputation_sequence_] == [(j, nbl) for j, nbl, _ in o.imputation_sequence_]
    _same(m.transform(Xh), o.transform(Xh))


class _PlainRegressor:
    def get_params(self, deep=True):
        return {}

    def fit(self, X, y):
        return self

    def predict(self, X):
        return np.zeros(len(X), dtype=np.float32)


@needs_bindings
def test_refusals():
    X = _data(0)
    with pytest.raises(TypeError, match="IterativeImputer requires a mojolearn estimator; got _PlainRegressor"):
        ml.IterativeImputer(estimator=_PlainRegressor()).fit(X)
    with pytest.raises(ValueError, match="return_std"):
        ml.IterativeImputer(estimator=ml.Ridge(), sample_posterior=True).fit(X)
    with pytest.raises(ValueError, match="'max_iter' parameter"):
        ml.IterativeImputer(max_iter=-1).fit(X)
    with pytest.raises(ValueError, match="invalid imputation_order"):
        ml.IterativeImputer(imputation_order="sideways").fit(X)
    with pytest.raises(ValueError, match="n_nearest_features"):
        ml.IterativeImputer(n_nearest_features=0).fit(X)
    with pytest.raises(RuntimeError, match="not fitted"):
        ml.IterativeImputer().transform(X)
    with pytest.raises(ValueError, match="features"):
        ml.IterativeImputer(max_iter=1).fit(X).transform(X[:, :2])
    Xi = X.copy()
    Xi[0, 0] = np.inf
    with pytest.raises(ValueError, match="infinity"):
        ml.IterativeImputer().fit(Xi)
