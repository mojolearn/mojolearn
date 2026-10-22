# SPDX-License-Identifier: Apache-2.0
"""Frozen 32011bb32 density constructors, used only as independent test oracles."""

KDE_DEFAULTS = dict(bandwidth=1.0, kernel="gaussian", metric="euclidean", algorithm="auto",
                    atol=0, rtol=0, breadth_first=True, leaf_size=40, metric_params=None)
IFOREST_DEFAULTS = dict(n_estimators=100, max_samples="auto", max_depth=None, max_features=1.0,
                        bootstrap=False, contamination="auto", random_state=None,
                        warm_start=False, n_jobs=None, verbose=False, output_type=None)


def initialize(model, family, **options):
    """Preserve stored values and the original NumericModeMixin constructor rule."""
    defaults = KDE_DEFAULTS if family == "kde" else IFOREST_DEFAULTS
    for name, value in defaults.items():
        setattr(model, name, options.get(name, value))
    mode = options.get("numeric_mode")
    if mode is not None or not hasattr(model, "numeric_mode"):
        model.numeric_mode = mode


def construct(cls, family, **options):
    model = object.__new__(cls)
    initialize(model, family, **options)
    return model
