# SPDX-License-Identifier: Apache-2.0
"""Frozen b32103c00 KernelDensity fit / score_samples / score Python path
(python/mojolearn/density.py before its methods became one native call each),
used only as an independent test oracle: `fit(model, X, y, sample_weight)`,
`scores(model, X, want_total)` (score_samples: False; score: True)."""
from mojolearn._sample_weight import check_sample_weight
from mojolearn._buffer import addr, addr_ro, all_finite, as_f32_dense_c, empty


def validate_params(self):
    """The constructor checks, run first in `fit` and `load`. Every
    read site converts `bandwidth` with `float()`."""
    bandwidth, algorithm, kernel, metric = self.bandwidth, self.algorithm, self.kernel, self.metric
    atol, rtol, breadth_first, leaf_size = self.atol, self.rtol, self.breadth_first, self.leaf_size
    metric_params = self.metric_params
    if isinstance(bandwidth, str):
        raise NotImplementedError(
            "mojolearn KernelDensity: bandwidth='%s' is refused; compute "
            "the rule on the host and pass the float (sklearn's scott is "
            "n ** (-1/(d+4)), silverman is (n (d+2) / 4) ** (-1/(d+4)))"
            % bandwidth
        )
    bandwidth = bandwidth if type(bandwidth) is float else float(bandwidth)
    if not (bandwidth > 0.0) or bandwidth != bandwidth:
        raise ValueError(
            "mojolearn KernelDensity: bandwidth must be positive, got "
            f"{bandwidth!r}"
        )
    if algorithm not in (None, "auto"):
        raise ValueError(
            "mojolearn KernelDensity: algorithm=%r is refused; this is "
            "cuML's brute-force KDE, there is no tree to choose" % (algorithm,)
        )
    if atol != 0 or rtol != 0 or not breadth_first or leaf_size != 40:
        raise ValueError(
            "mojolearn KernelDensity: atol/rtol/breadth_first/leaf_size "
            "are tree-traversal knobs; this is brute force (refused)"
        )
    if not isinstance(kernel, str) or not isinstance(metric, str):
        raise ValueError("mojolearn KernelDensity: kernel and metric are names")
    # cuML's rule, `kernel_density.py:302-307`: at most ONE entry, and
    # its VALUE is taken regardless of its key.
    if metric_params:
        if len(metric_params) != 1:
            raise ValueError(
                "mojolearn KernelDensity: only metrics with a single "
                "argument are supported (cuML raises the same, "
                "kernel_density.py:304-306)"
            )
        p = float(list(metric_params.values())[0])
        if p != 2.0:
            raise NotImplementedError(
                "mojolearn KernelDensity: metric_params=%r asks for "
                "Minkowski p=%r. The KERNEL computes any finite "
                "positive normal p (kde/impl/distance/distance.mojo, "
                "DEVIATION 552) and kde/estimator.mojo takes it as "
                "`metric_arg`; what is missing is a SLOT IN THE "
                "BINDING -- kde_score_samples_binding length-checks "
                "its params list at exactly 5 entries "
                "(bindings/_mojolearn_estimators.mojo:647, and the same "
                "check in bindings/_mojolearn_estimators_host.mojo:213). "
                "Pass p=2, "
                "or apply the two-line binding change in "
                "kde/README.md's HAND-OFF." % (metric_params, p)
            )

def resident_fit_handle(self, binding):
    """The handle of the device-resident copy of the fit set (DEVIATION
    3003, the KDE door of DEVIATION 2921), prepared on the first call
    and reused while the training array, the weights, the kernel, the
    metric and the bandwidth keep their identity; None where the loaded
    binding has no `kde_fit_prepare` (the CPU host binding, a CPU-only
    install), in which case `score_samples` takes the per-call
    upload. An absent name is ImportError on a host binding and
    AttributeError on a module, and both mean "no residency here"."""
    try:
        prepare = binding.kde_fit_prepare
    except (ImportError, AttributeError):
        return None
    w = self._w
    key = (addr_ro(self._x, name="_x"), tuple(self._x.shape),
           addr_ro(w, name="w") if w is not None else 0,
           str(self.kernel), str(self.metric), float(self.bandwidth))
    cached = getattr(self, "_resident", None)
    if cached is not None and cached[0] == key:
        return cached[1]
    if cached is not None:
        validate_params(self)
    release_resident_fit(self)
    handle = int(prepare(
        key[0], key[2],
        # ORDER MATCHES bindings/_mojolearn_estimators.mojo::kde_fit_prepare_binding.
        [int(self._x.shape[0]), int(self._x.shape[1]), float(self.bandwidth),
         1 if w is not None else 0],
        self.kernel, self.metric,
    ))
    # The minting binding travels with the handle: the release (here, and
    # the native finalizer when the instance is dropped) is its own
    # `kde_fit_release`.
    self._resident = (key, handle, binding)
    return handle

def release_resident_fit(self):
    """Drop the device copy, if one is held. Quiet on a binding that
    cannot be reached any more (interpreter shutdown) and on a handle
    already released (a copied instance)."""
    cached = getattr(self, "_resident", None)
    if cached is None:
        return
    self._resident = None
    try:
        cached[2].kde_fit_release(cached[1])
    except Exception:  # noqa: BLE001
        pass

# No Python `__del__`: dropping an instance releases its device fit
# through the native finalizer connected below the class
# (`_native_archive_bridge.lifetime`, bindings/host_archive_lifetime.mojo).

def fit(self, X, y=None, sample_weight=None):
    """Store the training data.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Training rows, converted to float32; every value must be
        finite.
    y : ignored
    sample_weight : array-like of shape (n_samples,), default=None
        Finite, non-negative per-row weights with a positive sum.

    Returns
    -------
    self

    Raises
    ------
    ValueError
        For a non-finite value or invalid ``sample_weight``.
    """
    # A refit drops the device copy of the previous fit set HERE, as
    # NearestNeighbors.fit does (DEVIATION 2921's key rule).
    release_resident_fit(self)
    x, self.input_copied_ = as_f32_dense_c(X, ndim=2, name="X")
    if not all_finite(x):
        # DEVIATION 604 refuses a non-finite fit set at the first
        # score_samples; refusing it here as well names the input at the
        # call that supplied it, as scikit-learn's fit does.
        raise ValueError(
            "mojolearn KernelDensity: X contains a NaN or an infinity; "
            "refused by name at fit (DEVIATION 604)"
        )
    # Opt-in immutable snapshot: transport-only copy of caller bytes.
    # Residency never relies on external mutable pointer identity.
    binding = self._bind("_mojolearn_estimators")
    immutable = bool(getattr(binding, "kde_fast_immutable_fit", lambda: 0)())
    if immutable:
        x = x.copy()
        self.input_copied_ = True
    self._x = x  # owned snapshot when the experiment is enabled
    self.n_features_in_ = x.shape[1]
    self.n_samples_fit_ = x.shape[0]
    if sample_weight is not None:
        # scikit-learn's _check_sample_weight (shape, finite, >= 0, not
        # all zero, its wording) by the base binding's reductions
        w = check_sample_weight(sample_weight, x.shape[0])
        self._w = w.copy() if immutable else w
    else:
        self._w = None
    return self

def scores(self, X, want_total):
    """score_samples(X), or with want_total their sum, folded in the
    binding (on the device on a GPU install: `device_sum_f32_fixed`;
    the host binding's `host_sum_f32_fixed`, the same order)."""
    if not hasattr(self, "_x"):
        raise ValueError("mojolearn KernelDensity: call fit() first")
    q, _ = as_f32_dense_c(X, ndim=2, name="X")
    if q.shape[1] != self.n_features_in_:
        raise ValueError(
            f"mojolearn KernelDensity: X has {q.shape[1]} features, "
            f"fit saw {self.n_features_in_}"
        )
    out = empty((q.shape[0],), "<f4")
    w = self._w
    binding = self._bind("_mojolearn_estimators")
    # THE FIT SET STAYS ON THE DEVICE (DEVIATION 3003): the first call
    # validates and uploads it, every later call scores the device copy
    # through `kde_score_samples_resident`, validating and uploading the
    # queries only; the score is the same statements over the same
    # bytes. A binding without that entry takes the per-call path.
    handle = resident_fit_handle(self, binding)
    if handle is not None:
        got = binding.kde_score_samples_resident(
            handle,
            addr_ro(q, name="q"),
            addr(out, name="out"),
            # ORDER MATCHES bindings/_mojolearn_estimators.mojo::kde_score_samples_resident_binding.
            [int(q.shape[0]), int(self.n_features_in_), float(self.bandwidth), 1 if want_total else 0],
            self.kernel,
            self.metric,
        )
        return float(got) if want_total else out
    got = binding.kde_score_samples(
        addr_ro(self._x, name="_x"),
        addr_ro(q, name="q"),
        addr_ro(w, name="w") if w is not None else 0,
        addr(out, name="out"),
        # ORDER MATCHES bindings/_mojolearn_estimators.mojo::kde_score_samples_binding.
        [
            int(self._x.shape[0]),
            int(q.shape[0]),
            int(self.n_features_in_),
            float(self.bandwidth),
            1 if w is not None else 0,
            1 if want_total else 0,
        ],
        self.kernel,
        self.metric,
    )
    return float(got) if want_total else out

# The per-row float32 scores are summed in the binding's fixed order (`core/device_fold.mojo`: on the
# device on a GPU install, the same order on the host column); lane pyglue-numeric replaced a host
# sum of the downloaded scores.
