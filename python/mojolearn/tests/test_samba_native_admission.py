# SPDX-License-Identifier: Apache-2.0
"""Native public Samba refusal before user protocols, with unchanged state."""
import array
import sys
from collections import defaultdict

import numpy as np
import pytest

from mojolearn import Array, SambaConfig, SambaStack
from mojolearn._verify_native_runtime import audit_native_call
from mojolearn.tests.test_samba_checkpoint_native import _assert_refusal


def _model():
    config = SambaConfig(13, 8, ["attention"], n_heads=2, intermediate=11,
                         dropout=0.125, tie_embeddings=False)
    weights = {name: np.full(shape, 0.025, dtype="f4") for name, shape in config.registry()}
    return SambaStack(config, weights=weights), np.array([[1, 3], [2, 4]], dtype="i4")


def _identity(model):
    return (model.flat.tobytes(), model.generator.state_dict(), model.optimizer.t,
            None if model.optimizer._m is None else model.optimizer._m.tobytes(),
            None if model.optimizer._v is None else model.optimizer._v.tobytes())


@pytest.mark.parametrize("field", ["vocab", "norm_eps", "layers", "tie_embeddings"])
def test_config_refuses_conversion_and_iteration_callbacks_before_execution(field):
    calls = []

    class Protocol:
        def __int__(self):
            calls.append("int")
            return 13

        def __float__(self):
            calls.append("float")
            return 1e-5

        def __iter__(self):
            calls.append("iter")
            return iter(["attention"])

        def __bool__(self):
            calls.append("bool")
            return True

    arguments = dict(vocab=13, d_model=8, layers=["attention"], n_heads=2, intermediate=11)
    arguments[field] = Protocol()
    config = object.__new__(SambaConfig)
    _assert_refusal(audit_native_call(config.__init__, **arguments), TypeError)
    assert calls == []
    assert vars(config) == {}


def test_config_dict_refuses_custom_mapping_before_getitem():
    calls = []

    class Mapping:
        def __getitem__(self, key):
            calls.append(key)
            return 1

    _assert_refusal(audit_native_call(SambaConfig.from_dict, Mapping()), TypeError)
    assert calls == []


@pytest.mark.parametrize("field", ["layers", "tie_embeddings", "norm_eps"])
def test_native_array_in_scalar_or_config_slot_cannot_invoke_python_protocols(monkeypatch, field):
    value = Array.from_list([1], "<i4")
    calls = []

    def callback(*args):
        calls.append("array protocol")
        raise AssertionError("Python Array conversion")

    config = object.__new__(SambaConfig)
    arguments = dict(vocab=13, d_model=8, layers=["attention"], n_heads=2, intermediate=11)
    arguments[field] = value
    with monkeypatch.context() as patch:
        for name in ("__bool__", "__iter__", "__len__", "__getitem__"):
            patch.setattr(Array, name, callback)
        result = audit_native_call(config.__init__, **arguments)
    _assert_refusal(result, TypeError)
    assert calls == []
    assert vars(config) == {}


def test_array_in_stored_sequence_slot_refused_before_python_indexing(monkeypatch):
    model, ids = _model()
    before = _identity(model)
    original = model.offsets
    model.offsets = Array.from_list(original, "<i4")
    calls = []

    def callback(*args):
        calls.append("index")
        raise AssertionError("Python Array indexing")

    with monkeypatch.context() as patch:
        patch.setattr(Array, "__getitem__", callback)
        result = audit_native_call(model.train_step, ids, ids)
    model.offsets = original
    _assert_refusal(result, TypeError)
    assert calls == []
    assert _identity(model) == before


def test_defaultdict_factory_is_refused_before_missing_config_lookup():
    calls = []

    def factory():
        calls.append("factory")
        return 13

    _assert_refusal(audit_native_call(SambaConfig.from_dict, defaultdict(factory)), TypeError)
    assert calls == []


def test_registered_record_key_refused_before_replaced_hash(monkeypatch):
    model, _ = _model()
    key = SambaConfig(13, 8, ["attention"], n_heads=2, intermediate=11)
    weights = {key: model.flat}
    calls = []

    def hash_callback(self):
        calls.append("hash")
        return 0

    with monkeypatch.context() as patch:
        patch.setattr(SambaConfig, "__hash__", hash_callback)
        result = audit_native_call(model.load_weights, weights)
    _assert_refusal(result, TypeError)
    assert calls == []


@pytest.mark.parametrize("operation", ["forward", "loss", "train_step"])
def test_ndarray_subclass_refused_before_properties_or_array_protocol(operation):
    model, ids = _model()
    before = _identity(model)
    calls = []

    class ArraySubclass(np.ndarray):
        def __getattribute__(self, name):
            calls.append(name)
            return super().__getattribute__(name)

    argument = ids.view(ArraySubclass)
    calls.clear()
    arguments = (argument,) if operation == "forward" else (argument, ids)
    _assert_refusal(audit_native_call(getattr(model, operation), *arguments), TypeError)
    assert calls == []
    assert _identity(model) == before


@pytest.mark.parametrize("location", ["config", "state", "optimizer", "array", "lock"])
def test_stored_metadata_protocol_refusal_precedes_training_or_state_mutation(location):
    model, ids = _model()
    before = _identity(model)
    calls = []

    class Number:
        def __int__(self):
            calls.append("int")
            return 2

        def __float__(self):
            calls.append("float")
            return 0.001

        def acquire(self):
            calls.append("acquire")

        def release(self):
            calls.append("release")

    if location == "config":
        old = model.config.norm_eps
        model.config.norm_eps = Number()
        result = audit_native_call(model.train_step, ids, ids)
        model.config.norm_eps = old
    elif location == "state":
        state = model.allocate_state(2, 8)
        state.layers[0].cached_tokens = Number()
        result = audit_native_call(model.forward, ids, state)
    elif location == "optimizer":
        old = model.optimizer.lr
        model.optimizer.lr = Number()
        result = audit_native_call(model.train_step, ids, ids)
        model.optimizer.lr = old
    elif location == "array":
        old = model.flat.shape
        model.flat.shape = (Number(),)
        result = audit_native_call(model.train_step, ids, ids)
        model.flat.shape = old
    else:
        old = model._blocks[0]._runtime_lock
        model._blocks[0]._runtime_lock = Number()
        result = audit_native_call(model.train_step, ids, ids)
        model._blocks[0]._runtime_lock = old
    _assert_refusal(result, TypeError)
    assert calls == []
    assert _identity(model) == before


@pytest.mark.parametrize("owner,name", [(SambaConfig, "vocab"), (Array, "shape")])
def test_replaced_public_descriptor_refused_without_invoking_property(monkeypatch, owner, name):
    model, ids = _model()
    before = _identity(model)
    calls = []

    def get_value(self):
        calls.append("property")
        return 13

    with monkeypatch.context() as patch:
        patch.setattr(owner, name, property(get_value), raising=False)
        result = audit_native_call(model.forward, ids)
    _assert_refusal(result, TypeError)
    assert calls == []
    assert _identity(model) == before


def test_custom_metaclass_and_config_subclass_refused_before_attribute_hooks():
    calls = []

    class Meta(type):
        def __getattribute__(cls, name):
            calls.append(name)
            return super().__getattribute__(name)

    class ConfigSubclass(SambaConfig, metaclass=Meta):
        def __getattribute__(self, name):
            calls.append(name)
            return super().__getattribute__(name)

    value = object.__new__(ConfigSubclass)
    model = object.__new__(SambaStack)
    calls.clear()
    _assert_refusal(audit_native_call(model.__init__, value), TypeError)
    assert calls == []
    assert vars(model) == {}


def test_standard_numpy_scalars_and_plain_native_buffers_remain_supported():
    config = object.__new__(SambaConfig)
    audit_native_call(config.__init__, np.int32(13), np.int64(8), ("attention",),
                      n_heads=np.int32(2), intermediate=np.int64(11),
                      norm_eps=np.float32(1e-5)).assert_native()
    registry = audit_native_call(config.registry).assert_native()
    weights = {name: np.asfortranarray(np.full(shape, 0.025, dtype="f4")) for name, shape in registry}
    model = object.__new__(SambaStack)
    audit_native_call(model.__init__, config, weights=weights).assert_native()
    ids = np.array([[1, 3], [2, 4]], dtype="i4")
    expected = model.forward(ids).tobytes()
    storage = array.array("i", [1, 3, 2, 4])
    plain = memoryview(storage).cast("B").cast("i", shape=(2, 2))
    assert audit_native_call(model.forward, plain).assert_native().tobytes() == expected
    assert audit_native_call(model.forward, np.asfortranarray(ids)).assert_native().tobytes() == expected


@pytest.mark.skipif(sys.version_info < (3, 12), reason="Python-defined buffer exporters require Python 3.12")
def test_python_buffer_exporter_is_refused_before_buffer_callback():
    model, ids = _model()
    calls = []

    class Exporter:
        def __buffer__(self, flags):
            calls.append("buffer")
            return memoryview(ids)

    _assert_refusal(audit_native_call(model.forward, Exporter()), TypeError)
    assert calls == []
