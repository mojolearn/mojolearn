# SPDX-License-Identifier: Apache-2.0
# The native parameter glue the release audit found no lane for (lane
# coverage-gap-lanes-20261025). Three entries of the tokenizer host binding
# are installed at import, not called by name, so no static reach sees them:
#   linear_constructor_native_call  the field-only __init__ of the linear models
#                                   (python/mojolearn/_native_linear_constructors.py)
#   params_setter_native_call       their set_params, which also drops fitted state
#                                   (python/mojolearn/_native_params_setter.py)
#   declared_params_native_call     the shallow set_params of the prep and decomp
#                                   families (python/mojolearn/_native_declared_params.py)
#   public_attribute_native_call    __getattribute__ of every connected class
#                                   (python/mojolearn/_native_public_attributes.py)
#   full_reset_params_native_call   set_params of the 90 field-only classes (KMeans, PCA,
#                                   SVC, ...), which also clears the fit
#                                   (python/mojolearn/_native_full_reset_params.py)
# The lane sets parameters through the public calls a user makes and digests
# what they change: a dropped or reverted assignment moves the refit's outputs.
# Sizes are correctness-check sizes, never product dispatch rules.
#
# tools/identity_break.py executes this file in ITS OWN namespace after every
# helper and registry exists (`_load_lane_fragments`): spell every lane name
# literally, add only these lanes and rebind no existing name.


@lane("public-set-params")
def _(ml, X, yc, yr, Xh=None):
    rows = np.ascontiguousarray(X[:256, :6], dtype=np.float32)
    target = np.ascontiguousarray(yr[:256], dtype=np.float32)
    # The linear family: the native constructor stores the keywords, the
    # native setter replaces two of them and drops the fitted fields.
    huber = ml.HuberRegressor(epsilon=1.5, max_iter=40, alpha=1e-3)
    before = huber.fit(rows, target).predict(rows)
    huber.set_params(alpha=0.5, epsilon=1.2)
    if hasattr(huber, "coef_"):
        raise AssertionError("HuberRegressor.set_params must drop the fitted coefficients")
    params = huber.get_params()
    after = huber.fit(rows, target).predict(rows)
    # The declared prep family: a shallow assignment on an unfitted expansion.
    expansion = ml.PolynomialFeatures(degree=2)
    expansion.set_params(degree=3, include_bias=False)
    expanded = expansion.fit_transform(rows[:, :3])
    # The declared decomp family: the new size and seed replace the old ones.
    projection = ml.GaussianRandomProjection(n_components=3, random_state=1)
    projection.set_params(n_components=4, random_state=7)
    projected = projection.fit_transform(rows)
    # The full-reset family (python/mojolearn/_native_full_reset_params.py,
    # full_reset_params_native_call): set_params replaces the constructor
    # fields and clears the fit, so the refit runs on the new settings.
    km = ml.KMeans(n_clusters=3, n_init=1, max_iter=20, random_state=0)
    km.fit(rows)
    km.set_params(n_clusters=4, random_state=1)
    if hasattr(km, "cluster_centers_"):
        raise AssertionError("KMeans.set_params must drop the fitted centers")
    km_params = km.get_params()
    km.fit(rows)
    settings = np.asarray([params["alpha"], params["epsilon"], params["max_iter"],
                           expansion.get_params()["degree"], projection.get_params()["n_components"]],
                          dtype=np.float64)
    return _fit(dict(linear=_h(before, after, huber.coef_, np.float64(huber.intercept_), np.float64(huber.scale_)),
                     prep=_h(expanded, expansion.powers_), decomp=_h(projected, projection.components_),
                     settings=_h(settings),
                     full_reset=_h(np.asarray(km.cluster_centers_), np.asarray(km.labels_), np.float64(km.inertia_),
                                   np.asarray([km_params["n_clusters"], km_params["random_state"]], dtype=np.int64))))


_batch_decl("n/a:coverage-probe (the lane checks public call paths the release audit found no lane for; "
            "its outputs are train parts, digested whole)",
            "public-set-params")
