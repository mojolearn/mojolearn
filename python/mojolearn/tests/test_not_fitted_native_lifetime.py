# SPDX-License-Identifier: Apache-2.0
"""Public fitted-error reduction retains payload and singleton ownership."""
import gc
import pickle
import sys
from types import ModuleType

import numpy as np
import pytest

from mojolearn import Ridge
from mojolearn.exceptions import NotFittedError
from mojolearn._verify_native_runtime import audit_native_call


def _error(monkeypatch):
    # The optional-package fixture has exactly sklearn's established native
    # exception bases. It does not claim qualification of a sklearn release.
    module = ModuleType("sklearn.exceptions")
    class SklearnNotFittedError(ValueError, AttributeError):
        pass
    module.NotFittedError = SklearnNotFittedError
    monkeypatch.setitem(sys.modules, module.__name__, module)
    with pytest.raises(NotFittedError) as caught:
        Ridge().predict(np.ones((2, 1), dtype="f4"))
    assert isinstance(caught.value, SklearnNotFittedError)
    return caught.value


@pytest.mark.parametrize('args', [(), ('one',), ('one', 'discarded')])
def test_public_error_reduction_preserves_payload_without_losing_none(monkeypatch, args):
    error = _error(monkeypatch)
    error.args = args
    operation = error.__reduce__
    result = audit_native_call(operation).assert_native()
    assert result[1] == args[:1]
    del result
    for _ in range(128):
        result = operation()
        del result
    gc.collect()
    enabled = gc.isenabled()
    gc.disable()
    try:
        before = sys.getrefcount(None)
        for _ in range(128):
            result = operation()
            del result
        after = sys.getrefcount(None)
    finally:
        if enabled:
            gc.enable()
    # CPython bookkeeping can retire a few unrelated None references. This
    # fixed allowance is independent of call count and cannot admit the
    # original demonstrated one-reference loss per reduction. CPython3.11
    # is essential coverage: later interpreters make None immortal.
    assert after >= before - 16, (before, after, after - before)
    if args:
        restored = pickle.loads(pickle.dumps(error))
        assert type(restored) is type(error)
        assert restored.args == args[:1]
