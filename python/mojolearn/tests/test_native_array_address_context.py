# SPDX-License-Identifier: Apache-2.0
"""The common native Array address reader admits mutable context keys first."""
import pytest

from mojolearn import MultiLabelBinarizer
import mojolearn._array as array_module


def _packing_case():
    rows = [[1, 3], [2], [1, 2]]
    model = MultiLabelBinarizer()

    def pack():
        # Integer multilabel input always creates native word/offset Arrays,
        # even with NumPy loaded. Their addresses reach classical_address ->
        # array_storage_address before the existing numerical provider call.
        return model.fit_transform(rows).tolist()

    assert pack() == [[1, 0, 1], [0, 1, 0], [1, 1, 0]]
    return rows, pack


@pytest.mark.parametrize("kind", ["mapping", "dict_subclass"])
def test_array_address_context_rebinding_refuses_before_mapping_hooks(kind, monkeypatch):
    rows, pack = _packing_case()
    called = []

    def forbidden(self, key, *args):
        called.append(key)
        raise AssertionError("Python Array context mapping hook executed")

    class Mapping:
        __getitem__ = forbidden
        get = forbidden

    class DictSubclass(dict):
        __getitem__ = forbidden
        get = forbidden

    replacement = Mapping() if kind == "mapping" else DictSubclass(array_module._ARRAY_CONTEXT)
    with monkeypatch.context() as patch:
        patch.setattr(array_module, "_ARRAY_CONTEXT", replacement)
        # The import-time module connection already pins this export. Its
        # existing refusal must remain ahead of any context subscripting.
        with pytest.raises(TypeError, match="native module connection export changed"):
            pack()
    assert called == []
    assert pack() == [[1, 0, 1], [0, 1, 0], [1, 1, 0]]
    assert rows == [[1, 3], [2], [1, 2]]


@pytest.mark.parametrize("field", ["array_class", "from_buffer_slots"])
@pytest.mark.parametrize("kind", ["foreign", "str_subclass"])
def test_array_address_context_mutation_invalidates_key_admission(field, kind, monkeypatch):
    rows, pack = _packing_case()
    context = array_module._ARRAY_CONTEXT
    original = context[field]
    called = []

    def forbidden(self, other):
        called.append(other)
        raise AssertionError("Python Array context key equality executed")

    class ForeignKey:
        def __hash__(self):
            return hash(field)

        __eq__ = forbidden

    class StringKey(str):
        __hash__ = str.__hash__
        __eq__ = forbidden

    key = ForeignKey() if kind == "foreign" else StringKey(field)
    with monkeypatch.context() as patch:
        # Remove the native collision target first: setup/cleanup must not
        # invoke equality either. The earlier packing call primed this same dict.
        patch.delitem(context, field)
        patch.setitem(context, key, original)
        with pytest.raises(TypeError, match="Array native address context requires native string keys"):
            pack()
    assert called == []
    assert pack() == [[1, 0, 1], [0, 1, 0], [1, 1, 0]]
    assert rows == [[1, 3], [2], [1, 2]]


def test_array_address_context_allows_unconsumed_native_string_metadata(monkeypatch):
    rows, pack = _packing_case()
    context = array_module._ARRAY_CONTEXT
    with monkeypatch.context() as patch:
        patch.setitem(context, "address_test_metadata", object())
        assert pack() == [[1, 0, 1], [0, 1, 0], [1, 1, 0]]
        # Value mutation also invalidates the stamp, without imposing a new
        # schema on unconsumed context values or changing the input rows.
        patch.setitem(context, "address_test_metadata", object())
        assert pack() == [[1, 0, 1], [0, 1, 0], [1, 1, 0]]
    assert rows == [[1, 3], [2], [1, 2]]
