# SPDX-License-Identifier: Apache-2.0
"""Negative controls for the owner-authorized artifact-only import API."""
import pytest
from mojolearn import import_model
from mojolearn._verify_native_runtime import audit_native_call


def test_python_models_and_serializers_are_refused_without_callbacks():
    calls = []
    class Source:
        def __getattribute__(self, name):
            calls.append(name)
            raise AssertionError("source hook invoked")
        def forward(self, *args):
            calls.append("forward")
        def SerializeToString(self):
            calls.append("serialize")
        def __fspath__(self):
            calls.append("path")
    audit = audit_native_call(import_model, Source(), numeric_mode="identical")
    assert isinstance(audit.exception, TypeError)
    assert not audit.frames and not audit.forbidden_native, audit.report()
    assert not audit.unresolved_native and not audit.wrapper_work, audit.report()
    assert not audit.unobserved_threads, audit.report()
    assert calls == []


def test_example_inputs_is_removed_from_the_public_signature():
    with pytest.raises(TypeError, match="example_inputs"):
        import_model(object(), example_inputs=object())
