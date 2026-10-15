# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Exceptions raised by mojolearn estimators."""

import sys

__all__ = ["NotFittedError"]


class NotFittedError(ValueError, AttributeError):
    """Raised when an estimator is used before it is fitted.

    It subclasses both ``ValueError`` and ``AttributeError``, as
    scikit-learn's ``NotFittedError`` does. When scikit-learn is imported, the
    error raised is also an instance of ``sklearn.exceptions.NotFittedError``,
    so ``except NotFittedError`` works with either import.

    Examples
    --------
    >>> from mojolearn.exceptions import NotFittedError
    >>> from mojolearn import Ridge
    >>> try:
    ...     Ridge().predict([[1.0, 2.0]])
    ... except NotFittedError as err:
    ...     print("fit first")
    fit first
    """


# The subclass that also derives from scikit-learn's NotFittedError, one per
# scikit-learn class object. Built only when scikit-learn is already imported,
# so importing mojolearn never imports scikit-learn.
_BRIDGES = {}


def not_fitted_error(message):
    """The `NotFittedError` instance to raise with `message`."""
    module = sys.modules.get("sklearn.exceptions")
    base = getattr(module, "NotFittedError", None)
    if base is None or issubclass(NotFittedError, base):
        return NotFittedError(message)
    bridge = _BRIDGES.get(base)
    if bridge is None:
        bridge = type("NotFittedError", (NotFittedError, base),
                      {"__module__": __name__, "__doc__": NotFittedError.__doc__,
                       # pickles (joblib workers) rebuild it through this function
                       "__reduce__": lambda self: (not_fitted_error, self.args[:1])})
        _BRIDGES[base] = bridge
    return bridge(message)
