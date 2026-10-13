"""Binary probabilities must keep borrowed native input buffers alive."""
import gc
import weakref

import numpy as np

from mojolearn import _expansion_trees as trees
from mojolearn._array import Array


def test_binary_probability_zero_column_survives_native_call(monkeypatch):
    class TrackedArray(Array):
        pass

    allocated = []

    def tracked_zeros(shape, dtype):
        value = TrackedArray(shape, dtype)
        allocated.append(weakref.ref(value))
        return value

    class Native:
        def x_trees_stack_w64(self, columns, out, dimensions):
            gc.collect()
            assert allocated[-1]() is not None, "zero column was freed before native code consumed it"
            np.testing.assert_array_equal(np.asarray(allocated[-1]()), 0)

        def x_trees_softmax_rows(self, out, dimensions):
            pass

    model = trees.DARTClassifier()
    monkeypatch.setattr(trees, "zeros", tracked_zeros)
    monkeypatch.setattr(model, "_raw", lambda X: Array((len(X),), "<f8"))
    monkeypatch.setattr(model, "_bind", lambda: Native())
    assert model.predict_proba(np.zeros((19, 3), np.float32)).shape == (19, 2)
    gc.collect()
    assert allocated[-1]() is None, "temporary ownership must end after the call"
