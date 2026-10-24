# SPDX-License-Identifier: Apache-2.0
"""The former Python paths of LocalOutlierFactor, NearestCentroid and
KNNImputer (python/mojolearn/_expansion_neighbors.py before lane
native-neighbors-20261024 made each public method one native call), frozen
as subclasses so the tests compare the native door with them bit for bit.
Test-only oracle; never imported by the package."""
import os

from mojolearn import _portable_math as math
from mojolearn._array import Array
from mojolearn._buffer import addr, addr_ro, empty, zeros
from mojolearn._lazy_out import _empty_out
from mojolearn._mode import available_if
from mojolearn._expansion_neighbors import (
    LocalOutlierFactor, NearestCentroid, KNNImputer, _f32, _i32, _f32_scalar, _p2m_sort_rows,
    _class_array, _argmax_labels, _accuracy, _U_IDENTITY, _UNCOMPACT_IMPUTE, _OLD_ITEMS,
)


class LegacyLocalOutlierFactor(LocalOutlierFactor):
    def _check_params(self):
        if not (self.metric == "euclidean" or (self.metric == "minkowski" and self.p == 2)):
            raise NotImplementedError(
                f"LocalOutlierFactor: metric={self.metric!r} (p={self.p!r}) is not implemented; "
                "euclidean (or minkowski with p=2) only")
        if self.metric_params:
            raise NotImplementedError("LocalOutlierFactor: metric_params is not implemented")
        if self.contamination != "auto" and not (0.0 < float(self.contamination) <= 0.5):
            raise ValueError("contamination must be 'auto' or in (0, 0.5]")
        if int(self.n_neighbors) < 1:
            raise ValueError("n_neighbors must be >= 1")

    def fit(self, X, y=None):
        """Fit the model on the training data.

        Parameters
        ----------
        X : array-like or scipy.sparse matrix of shape (n_samples, n_features)
            Training data; at least 2 samples. Sparse input is densified.
        y : None
            Ignored.

        Returns
        -------
        self : LocalOutlierFactor
            The fitted estimator.
        """
        self._check_params()
        X = _f32(X)
        n, d = X.shape
        if n < 2:
            # scikit-learn's check_array wording (ensure_min_samples=2)
            raise ValueError(f"Found array with {n} sample(s) (shape=({n}, {d})) while a minimum of 2 is required.")
        k = min(int(self.n_neighbors), n - 1)
        dist, idx = self._knn(X, X, k, True)
        lrd = _empty_out((n,), "<f4")
        self._op("lof_lrd", [(dist, 0), (idx, 0), (dist, 0), (lrd, 1)], (k, n, n))
        score = _empty_out((n,), "<f4")
        self._op("lof_score", [(idx, 0), (lrd, 0), (lrd, 0), (score, 1)], (k, n, n))
        self._fit_X, self._fit_dist, self._lrd = X, dist, lrd
        self.n_neighbors_ = k
        self.n_features_in_ = d
        self.n_samples_fit_ = n
        self.negative_outlier_factor_ = score
        if self.contamination == "auto":
            self.offset_ = -1.5
        else:
            q = 100.0 * float(self.contamination)
            self.offset_ = _f32_scalar(self._sorted_percentile(score, q))
        return self

    def _sorted_percentile(self, score, q):
        """numpy.percentile (method 'linear') of score with the sort on the device (`p2m_row_sort`
        over one row: ascending by value, ties by position, as `sorted`);
        only the two order statistics it reads come back to Python."""
        n = score.shape[0]
        srt = _p2m_sort_rows(self, _i32([0, n], "indptr"), zeros((n,), "<i4"), score)[1]
        pos = (q / 100.0) * (n - 1)
        lo = math.floor(pos)
        hi = min(lo + 1, n - 1)
        t = pos - lo
        a, b = srt[lo], srt[hi]
        diff = b - a
        return b - diff * (1.0 - t) if t >= 0.5 else a + diff * t

    @available_if(lambda self: not self.novelty)
    def fit_predict(self, X, y=None):
        """Fit the model and label the training samples.

        Available only with ``novelty=False``; otherwise raises
        ``AttributeError``.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        y : None
            Ignored.

        Returns
        -------
        labels : Array of shape (n_samples,), int64
            -1 for outliers, 1 for inliers.
        """
        if self.novelty:
            raise AttributeError("fit_predict is not available when novelty=True; use predict on new data")
        self.fit(X)
        off = self.offset_
        score = self.negative_outlier_factor_
        lab = empty((score.size,), "<i4")
        if score.size:
            self._op("p2m_sign_label", [(score, 0), (lab, 1)], (score.size, 0), (off,))
        return lab.astype("<i8")

    def _novelty(self, what):
        if not self.novelty:
            raise AttributeError(f"{what} is not available when novelty=False; use fit_predict")

    @available_if(lambda self: self.novelty)
    def score_samples(self, X):
        """Negative local outlier factor of new samples.

        Available only with ``novelty=True``; otherwise raises
        ``AttributeError``. Lower scores are more abnormal.

        Parameters
        ----------
        X : array-like of shape (n_queries, n_features)
            New samples.

        Returns
        -------
        scores : Array of shape (n_queries,), float64
        """
        self._novelty("score_samples")
        Q = _f32(X)
        k, nf = self.n_neighbors_, self.n_samples_fit_
        dist, idx = self._knn(Q, self._fit_X, k, False)
        nq = Q.shape[0]
        lrd = _empty_out((nq,), "<f4")
        self._op("lof_lrd", [(dist, 0), (idx, 0), (self._fit_dist, 0), (lrd, 1)], (k, nq, nf))
        score = _empty_out((nq,), "<f4")
        self._op("lof_score", [(idx, 0), (self._lrd, 0), (lrd, 0), (score, 1)], (k, nq, nf))
        return score

    @available_if(lambda self: self.novelty)
    def decision_function(self, X):
        """Shifted outlier score of new samples: ``score_samples(X) - offset_``.

        Available only with ``novelty=True``. Negative values are outliers.

        Parameters
        ----------
        X : array-like of shape (n_queries, n_features)
            New samples.

        Returns
        -------
        scores : Array of shape (n_queries,), float64
        """
        self._novelty("decision_function")
        return self._unary(self.score_samples(X), _U_IDENTITY, 1.0, -self.offset_)

    @available_if(lambda self: self.novelty)
    def predict(self, X):
        """Label new samples as inliers or outliers.

        Available only with ``novelty=True``; otherwise raises
        ``AttributeError``.

        Parameters
        ----------
        X : array-like of shape (n_queries, n_features)
            New samples.

        Returns
        -------
        labels : Array of shape (n_queries,), int64
            -1 for outliers, 1 for inliers.
        """
        self._novelty("predict")
        dec = self.decision_function(X)
        lab = empty((dec.size,), "<i4")
        if dec.size:
            self._op("p2m_sign_label", [(dec, 0), (lab, 1)], (dec.size, 1), (0.0,))
        return lab.astype("<i8")


class LegacyNearestCentroid(NearestCentroid):
    def fit(self, X, y):
        """Fit the class centroids.

        Parameters
        ----------
        X : array-like or scipy.sparse matrix of shape (n_samples, n_features)
            Training data.
        y : array-like of shape (n_samples,)
            Class labels; at least two classes.

        Returns
        -------
        self : NearestCentroid
            The fitted estimator.

        Raises
        ------
        ValueError
            If there are fewer than two classes, the priors are invalid, or
            every feature has zero variance.
        """
        if self.metric not in ("euclidean", "manhattan"):
            raise ValueError("NearestCentroid: metric must be 'euclidean' or 'manhattan'")
        X = _f32(X)
        n, d = X.shape
        # the native encoder's int32 codes, the class counts from the device
        # (`p2m_class_counts`, an atomic add per row; lane pyglue-numeric:
        # a Python counting loop on every column but FAST + Apple)
        from mojolearn._labels import encode_labels
        classes, lab = encode_labels(y)
        if lab.size != n:
            raise ValueError("X and y have different numbers of rows")
        C = len(classes)
        if C < 2:
            raise ValueError(f"The number of classes has to be greater than one; got {C} class")
        nk = _empty_out((C,), "<f4")
        info = empty((1,), "<i4")
        self._op("p2m_class_counts", [(lab, 0), (nk, 1), (info, 1)], (n, C))
        counts = [int(v) for v in nk.tolist()]          # glue: the k class counts (priors, offsets)
        if self.priors == "empirical":
            # count / n per class in the base binding (lane py-runtime-b;
            # the same binary64 words as the Python quotient)
            from mojolearn._buffer import _native
            pa = empty((C,), "<f8")
            ca = Array.from_list(counts, "<i8")
            _native("class_ratio_f64")(addr_ro(ca, name="counts"), C, n, 1, addr(pa, name="class_prior_"))
            prior = pa.tolist()  # glue: k class priors for the checks below
        elif self.priors == "uniform":
            prior = [1.0 / C] * C
        else:
            prior = [float(v) for v in (self.priors.tolist() if hasattr(self.priors, "tolist") else self.priors)]  # glue: user priors argument checks
            if len(prior) != C:
                raise ValueError("priors must have one entry per class")
            if any(p < 0 for p in prior):  # glue: user priors argument checks
                raise ValueError("priors must be non-negative")
            tot = math.fsum(prior)  # glue: the native fsum (core fsum_f64) of the user priors argument
            if not math.isclose(tot, 1.0, rel_tol=1e-5, abs_tol=1e-8):
                prior = [p / tot for p in prior]  # glue: the user priors argument rescaled to sum one (k sized)
        self.class_prior_ = Array.from_list(prior, "<f8")
        if self.metric == "euclidean":
            cent = _empty_out((C, d), "<f4")
            if os.environ.get("MOJOLEARN_NC_SPLIT_OPS", "") == "1":
                self._op("group_mean", [(X, 0), (lab, 0), (cent, 1)], (n, d, C))
        else:
            # the per-class medians on the device (cpu-gpu-cleanup w2-pyglue,
            # x_neighbors/sort_items.mojo): a bitonic sort of every feature's
            # rows by (class, value, row) over a power-of-two pad, then each
            # class's middle value(s) at its offset; offsets are the class
            # counts' prefix (integer bookkeeping over the classes)
            start = [0] * (C + 1)
            for c in range(C):  # glue: class offsets prefix over k classes
                start[c + 1] = start[c] + counts[c]
            p = 1
            while p < n:
                p *= 2
            lg = p.bit_length() - 1
            cent = _empty_out((C, d), "<f4")
            self._op("nc_median", [(X, 0), (lab, 0), (_i32(start, "start"), 0), (cent, 1)],
                     (n, d, C, p, lg * (lg + 1) // 2))
        stats = _empty_out((d,), "<f4")
        new_cent = _empty_out((C, d), "<f4")
        devs = _empty_out((C, d), "<f4")
        dsc = None
        if self.metric == "euclidean" and os.environ.get("MOJOLEARN_NC_SPLIT_OPS", "") != "1":
            # lane/neural-pass95: the class means, the within-class std and the
            # dataset centroid in one op (x_neighbors/items.mojo nc_stats_item):
            # X goes to the device once; MOJOLEARN_NC_SPLIT_OPS=1 restores the three ops
            dsc = _empty_out((d,), "<f4")
            self._op("nc_stats", [(X, 0), (lab, 0), (nk, 0), (cent, 1), (stats, 1), (dsc, 1)], (n, d, C))
        else:
            self._op("nc_std", [(X, 0), (lab, 0), (cent, 0), (stats, 1)], (n, d, C))
        # the median of the d stds and the all-zero test on the device
        # (x_neighbors/sort_items.mojo nc_med_std: a bitonic sort, then the
        # middle value(s)); two floats come back
        p = 1
        while p < d:
            p *= 2
        lg = p.bit_length() - 1
        ms = _empty_out((2,), "<f4")
        self._op("nc_med_std", [(stats, 0), (ms, 1)], (d, p, lg * (lg + 1) // 2))
        med_std, all_zero = ms.tolist()
        if all_zero and self._ptp_zero(X):
            raise ValueError("All features have zero variance. Division by zero.")
        shrink = float(self.shrink_threshold) if self.shrink_threshold else 0.0
        if dsc is not None:
            self._op("nc_shrink_d", [(dsc, 0), (cent, 0), (nk, 0), (stats, 0), (new_cent, 1), (devs, 1)],
                     (n, d, C, 1 if shrink else 0), (med_std, shrink))
        else:
            self._op("nc_shrink", [(X, 0), (cent, 0), (nk, 0), (stats, 0), (new_cent, 1), (devs, 1)],
                     (n, d, C, 1 if shrink else 0), (med_std, shrink))
        self.centroids_ = new_cent
        self.deviations_ = devs
        self.within_class_std_dev_ = stats
        self.classes_ = classes
        self._codes_classes = classes
        self.n_features_in_ = d
        return self

    def _ptp_zero(self, X):
        """Every feature constant (sklearn's ptp == 0): `p2m_const_cols`."""
        n, d = X.shape
        flag = empty((1,), "<i4")
        self._op("p2m_const_cols", [(X, 0), (flag, 1)], (n, d))
        return flag.tolist()[0] == 0

    def _uniform(self):
        C = len(self.classes_)
        return all(math.isclose(p, 1.0 / C, rel_tol=1e-5, abs_tol=1e-8) for p in self.class_prior_.tolist())  # glue: uniform prior check over k classes

    def predict(self, X):
        """Predict the class of each sample.

        With uniform priors this is the class of the nearest centroid;
        otherwise the class with the largest :meth:`decision_function` value.

        Parameters
        ----------
        X : array-like of shape (n_queries, n_features)
            Samples to classify.

        Returns
        -------
        labels : Array or list of shape (n_queries,)
            Predicted class labels.
        """
        Q = _f32(X)
        if self._uniform():
            D = self._sqdist(Q, self.centroids_) if self.metric == "euclidean" else self._l1dist(Q, self.centroids_)
            _, idx = self._knn_select(D, 1, False)
            # TOMBSTONE: MOJOLEARN_NC_FAST_CLS1_PREDICT (DROPPED-noise) deleted 2026-10-03 by 9decae29f; code recoverable at 9decae29f^.
            # Restore: git apply experiments/removed/MOJOLEARN_NC_FAST_CLS1_PREDICT.patch; record in docs/TOMBSTONES.md.
            return _class_array(self.classes_, idx.reshape((idx.shape[0],)))
        return _argmax_labels(self, self.classes_, self._decision_matrix(Q))

    def decision_function(self, X):
        """Discriminant score of each sample for each class.

        Available for ``metric='euclidean'`` only; otherwise raises
        ``AttributeError``.

        Parameters
        ----------
        X : array-like of shape (n_queries, n_features)
            Samples to score.

        Returns
        -------
        scores : Array of shape (n_queries, n_classes), float32
            For two classes, shape (n_queries,): the second class's score
            minus the first's, positive for ``classes_[1]``.
        """
        dec = self._decision_matrix(X)
        if dec.shape[1] != 2:
            return dec
        # scikit-learn's binary decision_function, scores[:, 1] - scores[:, 0]:
        # one product with the exact weights (-1, 1), so each entry is the
        # difference rounded once
        signs = Array.from_list([-1.0, 1.0], "<f4").reshape((2, 1))
        diff = self._matmul(dec, signs)
        return diff.reshape((dec.shape[0],))

    def _decision_matrix(self, X):
        """The (n_queries, n_classes) discriminant scores."""
        if self.metric != "euclidean":
            raise AttributeError("decision_function is available for metric='euclidean' only")
        Q = _f32(X)
        C = len(self.classes_)
        prior = Array.from_list(self.class_prior_.tolist(), "<f4")
        std = self.within_class_std_dev_
        out = _empty_out((Q.shape[0], C), "<f4")
        self._op("nc_decision", [(Q, 0), (self.centroids_, 0), (std, 0), (prior, 0), (out, 1)],
                 (Q.shape[0], Q.shape[1], C))
        return out

    def predict_proba(self, X):
        """Class probabilities: the softmax of :meth:`decision_function`.

        Available for ``metric='euclidean'`` only.

        Parameters
        ----------
        X : array-like of shape (n_queries, n_features)
            Samples to score.

        Returns
        -------
        proba : Array of shape (n_queries, n_classes)
            Probabilities, columns in the order of ``classes_``.
        """
        dec = self._decision_matrix(X)
        if hasattr(self._bind(), "xn_softmax64"):
            # lane apple-fast-q-clf (x_neighbors/proba64_nc.mojo, QUALITY-FIX, FAST
            # default; -D MOJOLEARN_PROBA64_QOLD has no export): float64 probabilities,
            # the winner's 1 - c exact (float32 saturated at 1.0: Istella log loss
            # 4.299 vs scikit-learn 4.118, accuracy equal)
            out = empty(dec.shape, "<f8")
            self._op("softmax64", [(dec, 0), (out, 1)], dec.shape)
            return out
        out = empty(dec.shape, "<f4")
        self._op("softmax", [(dec, 0), (out, 1)], dec.shape)
        return out


    def score(self, X, y, sample_weight=None):
        return _accuracy(y, self.predict(X), sample_weight)


class LegacyKNNImputer(KNNImputer):
    def _check(self):
        mv = self.missing_values
        import numbers
        if mv is None or isinstance(mv, (str, bool)) or not isinstance(mv, numbers.Real):
            raise NotImplementedError("KNNImputer: missing_values must be a number or NaN (float input)")
        if self.metric != "nan_euclidean":
            raise NotImplementedError("KNNImputer: metric must be 'nan_euclidean'")
        if self.weights not in ("uniform", "distance"):
            raise NotImplementedError("KNNImputer: weights must be 'uniform' or 'distance'")

    def _nan_cells(self, X, colmiss_only=0):
        """(cells, colmiss, count): the flat indices of the NaN cells of X
        ascending (an int32 Array of n * d slots, the first `count` used), the
        NaN count per column as a list, and the count (xn_nan_cells).
        colmiss_only=1 (fit): a build with MOJOLEARN_XN_FAST_NAN_COLMISS_ONLY
        may skip the cell list; the counts are the same integers."""
        n, d = X.shape
        # FAST Apple default MOJOLEARN_XN_FAST_NAN_FIT_LEAN_GPU (_OFF rollback): a binary
        # whose fit-time op ignores the cell list gets a one-slot one (no
        # n * d allocation)
        lean = colmiss_only and int(getattr(self._bind(), "x_neighbors_nc_fit_lean", lambda: 0)())
        cells = empty((1 if lean else max(n * d, 1),), "<i4")
        colmiss = empty((max(d, 1),), "<i4")
        info = empty((1,), "<i4")
        self._op("nan_cells", [(X, 0), (cells, 1), (colmiss, 1), (info, 1)], (n, d, colmiss_only))
        return cells, colmiss.tolist(), int(info.tolist()[0])

    def _masked(self, X):
        X = _f32(X)
        mv = self.missing_values
        if mv == mv and X.size:           # a number: its cells become NaN
            out = empty(X.shape, "<f4")
            self._op("p2m_mask_value", [(X, 0), (out, 1)], (X.size,), (_f32_scalar(mv),))
            X = out
        return X

    def fit(self, X, y=None):
        """Store the training data used as donors.

        Parameters
        ----------
        X : array-like or scipy.sparse matrix of shape (n_samples, n_features)
            Training data, with missing values marked by ``missing_values``.
        y : None
            Ignored.

        Returns
        -------
        self : KNNImputer
            The fitted imputer.
        """
        self._check()
        X = self._masked(X)
        n, d = X.shape
        # lane/neural-pass71 (2026-10-01): the column flags from one native
        # pass over the cells (xn_nan_cells), no list of the matrix
        cm = self._nan_cells(X, 1)[1]
        self._valid = [cm[f] < n for f in range(d)]  # glue: column masks over the d features
        self._miss_cols = [f for f in range(d) if cm[f] > 0]  # glue: column masks over the d features
        self._fit_X = X
        self.n_features_in_ = d
        return self

    def transform(self, X):
        """Impute the missing values of X.

        Parameters
        ----------
        X : array-like or scipy.sparse matrix of shape (n_samples, n_features)
            Data to impute; the number of features must match ``fit``
            (``ValueError`` otherwise).

        Returns
        -------
        X_imputed : Array of shape (n_samples, n_features_out), float32
            Imputed data, without the features that were entirely missing at
            ``fit`` (unless ``keep_empty_features``), followed by the indicator
            columns when ``add_indicator`` is True.
        """
        X = self._masked(X)
        n, d = X.shape
        if d != self.n_features_in_:
            raise ValueError("X has a different number of features than during fit")
        m = self._fit_X.shape[0]
        out = _empty_out((n, d), "<f4")
        k = int(self.n_neighbors)
        if k < 1:
            raise ValueError("n_neighbors must be >= 1")
        if _UNCOMPACT_IMPUTE:
            self._op("knn_impute", [(X, 0), (self._fit_X, 0), (out, 1)],
                     (n, m, d, k, 1 if self.weights == "distance" else 0))
        else:
            # one GPU thread per MISSING cell (`knn_impute_cells`): the same
            # item statements; a present cell keeps x, as the item stores it
            cells, _, nc = self._nan_cells(X)        # lane/neural-pass71: no Python walk
            out = X.copy()
            if nc:
                self._op("knn_impute_cells" if _OLD_ITEMS else "knn_impute_tiled",
                         [(cells, 0), (X, 0), (self._fit_X, 0), (out, 1)],
                         (n, m, d, k, 1 if self.weights == "distance" else 0, nc))
        keep = [f for f in range(d) if self._valid[f]]  # glue: column masks over the d features
        if self.keep_empty_features:
            if n and not all(self._valid):
                flags = _i32([0 if v else 1 for v in self._valid], "flags")   # glue: d fit flags
                z = _empty_out((n, d), "<f4")
                self._op("p2m_zero_cols", [(out, 0), (flags, 0), (z, 1)], (n, d))
                out = z
        elif len(keep) != d:
            out = self._take_cols(out, keep)
        if self.add_indicator and self._miss_cols and n:
            c = out.shape[1]
            q = len(self._miss_cols)
            res = _empty_out((n, c + q), "<f4")
            self._op("p2m_nan_indicator",
                     [(X, 0), (out if c else X, 0), (_i32(self._miss_cols, "cols"), 0), (res, 1)], (n, d, c, q))
            out = res
        return out

    def fit_transform(self, X, y=None):
        """Fit to X, then impute X: ``fit(X).transform(X)``.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Data with missing values.
        y : None
            Ignored.

        Returns
        -------
        X_imputed : Array of shape (n_samples, n_features_out), float32
        """
        return self.fit(X).transform(X)
