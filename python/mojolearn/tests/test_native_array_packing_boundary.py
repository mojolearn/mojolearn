# SPDX-License-Identifier: Apache-2.0
"""Public Array packing rejects Python hooks before touching native storage."""
from array import array

import pytest

from mojolearn import Array, NMF
from mojolearn._array import _ARRAY_CONTEXT


def _nmf_input():
    np = pytest.importorskip("numpy")
    return np.array([[1, 2, 1], [3, 1, 2], [1, 1, 4], [2, 3, 1]], dtype=np.float32)


def _model():
    return NMF(n_components=2, init="random", solver="cd", max_iter=2,
               random_state=7, numeric_mode="identical")


@pytest.mark.parametrize("hook", ["__setattr__", "__getattribute__", "_addr"])
def test_nmf_output_packing_refuses_rebound_array_hooks(hook, monkeypatch):
    source = _nmf_input()
    model = _model()
    # Prime the same class/storage admission, then mutate the class. The
    # output packer must invalidate that admission before allocating an owner.
    held = Array.from_buffer(array("f", [1.0, 2.0]))
    called = []

    def forbidden(*args):
        called.append(hook)
        raise AssertionError("Python Array storage hook executed")

    replacement = property(forbidden, forbidden) if hook == "_addr" else forbidden
    with monkeypatch.context() as patch:
        patch.setattr(Array, hook, replacement, raising=False)
        with pytest.raises(TypeError, match="native (allocation|storage)"):
            model.fit(source)
    assert called == []
    assert held.shape == (2,)
    assert "components_" not in vars(model)


@pytest.mark.parametrize("change", ["descriptor", "missing", "foreign_key"])
def test_array_storage_registry_changes_invalidate_admission(change, monkeypatch):
    storage = array("f", [1.0, 2.0])
    held = Array.from_buffer(storage)
    slots = _ARRAY_CONTEXT["from_buffer_slots"]
    called = []

    def forbidden(*args):
        called.append("descriptor")
        raise AssertionError("Python storage descriptor executed")

    class Key:
        def __hash__(self):
            return hash("_addr")

        def __eq__(self, other):
            called.append("equality")
            raise AssertionError("Python storage key equality executed")

    with monkeypatch.context() as patch:
        if change == "descriptor":
            patch.setitem(slots, "_addr", property(forbidden, forbidden))
        elif change == "missing":
            patch.delitem(slots, "_addr")
        else:
            # Remove the colliding native key before inserting the trap;
            # setup itself must not invoke its equality callback.
            patch.delitem(slots, "_addr")
            patch.setitem(slots, Key(), None)
        with pytest.raises(TypeError, match="storage descriptors"):
            Array.from_buffer(storage)
    assert called == []
    assert held._store is storage


def test_array_passive_subclass_keeps_borrowed_storage_and_metadata():
    class Passive(Array):
        pass

    storage = array("i", [3, -7, 12])
    result = Passive.from_buffer(storage)
    assert type(result) is Passive
    assert result._store is storage
    assert result._addr == storage.buffer_info()[0]
    assert result.shape == (3,) and result.strides == (storage.itemsize,)
    assert result.itemsize == storage.itemsize and result.nbytes == 3 * storage.itemsize
    assert result.order == "C" and result._readonly is False
    storage[1] = 23
    assert result._mv.tobytes() == storage.tobytes()
    del storage
    assert result._mv.tobytes() == array("i", [3, 23, 12]).tobytes()


def test_nmf_output_array_keeps_native_owner_and_layout():
    model = _model().fit(_nmf_input())
    result = vars(model)["components_"]
    assert type(result) is Array
    assert type(result._store) is array
    assert result._addr == result._store.buffer_info()[0]
    assert result._base is None and result._pin is None
    assert result._mv.obj is result._store
    assert result.shape == (2, 3) and result.strides == (12, 4)
    assert result.dtype == "<f4" and result.itemsize == 4
    assert result.ndim == 2 and result.size == 6 and result.nbytes == 24
    assert result.order == "C" and result._readonly is False
    retained = result._mv
    expected = result._store.tobytes()
    del result, model
    assert retained.tobytes() == expected
