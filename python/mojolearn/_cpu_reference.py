# SPDX-License-Identifier: Apache-2.0
"""Private, scoped access to CPU training for the numerical verifier.

This is not a public CPU training API. Source reference builds may contain
more native bindings than an inference wheel. No environment variable turns
ordinary estimator fitting into CPU training.
"""
from contextlib import contextmanager
from contextvars import ContextVar

_active = ContextVar("mojolearn_cpu_reference", default=False)


@contextmanager
def reference_training():
    """Allow reference fits for this verification call, restoring on failure."""
    token = _active.set(True)
    try:
        yield
    finally:
        _active.reset(token)


def require_training(estimator):
    from . import _backend
    # A class whose `fit` computes an inference answer rather than training
    # a model says so by name. `Cholesky` is the one: its fit factors a
    # given matrix, the public CPU inference surface of
    # lane/inference-embedding-ivf-cholesky (2026-09-15).
    if getattr(type(estimator), "_CPU_FIT_IS_INFERENCE", False) is True:
        return
    is_cpu = _backend._CPU_ONLY is not None or getattr(estimator, "_HOST_INFERENCE_ONLY", False)
    if is_cpu and not _active.get():
        raise NotImplementedError(
            "mojolearn: public CPU estimators support inference from saved models; "
            "fit/training is reserved for the internal bitwise verifier. "
            "Train on a supported GPU and load the saved model for CPU inference. "
            "The published LanguageModelHostTrainer remains supported."
        )
