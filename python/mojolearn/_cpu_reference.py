# SPDX-License-Identifier: Apache-2.0
"""CPU training is public (2026-09-20).

Every estimator that has a CPU host binding fits on a CPU-only install, in
the same arithmetic the GPU columns are checked against. Until this date an
ordinary `fit` refused on CPU and only the verifier could train there, from
inside `reference_training()`. The context manager and `require_training`
stay so their callers keep working; neither one refuses anything.
"""
from contextlib import contextmanager
from contextvars import ContextVar

_active = ContextVar("mojolearn_cpu_reference", default=True)


@contextmanager
def reference_training():
    """Kept for callers; CPU fits no longer need it."""
    token = _active.set(True)
    try:
        yield
    finally:
        _active.reset(token)


def require_training(estimator):
    """Kept for callers; CPU training is public and nothing is refused."""
    return None
