"""Native family attribute reads (bindings/family_raw_attr_pack.mojo) against CPython's generic lookup.

Compile tests/native_family_raw_attr_probe_pyapi.mojo as a shared library (the extension module
`raw_attr_probe`) and set MOJOLEARN_FAMILY_RAW_ATTR_PROBE to its path to run these controls, as
tools/test_array_metadata_owner.py does for its probe. Without that compiled probe the native
controls have not run, and they are reported as skipped for that reason, never as passed.
Test-only descriptors exercise CPython generic lookup and exception preservation;
this probe does not expand the product's admission of caller protocols.
"""
import importlib.util
import os

import pytest

import mojolearn  # noqa: F401  (loads the package the probe links against)


@pytest.fixture(scope="module")
def native():
    path = os.environ.get("MOJOLEARN_FAMILY_RAW_ATTR_PROBE")
    if not path:
        pytest.skip("the raw_attr_probe extension has not been compiled for this test run "
                    "(set MOJOLEARN_FAMILY_RAW_ATTR_PROBE)")
    spec = importlib.util.spec_from_file_location("raw_attr_probe", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_missing(native):
    sentinel = object()
    assert native.read(object(), "absent", sentinel) is sentinel


def test_value(native):
    class Model:
        pass
    m = Model()
    m.value = object()
    assert native.read(m, "value", None) is m.value


def test_descriptor_precedence(native):
    expected = object()

    class Model:
        @property
        def value(self):
            return expected
    m = Model()
    m.__dict__["value"] = "shadow"
    assert native.read(m, "value", None) is expected


def test_attribute_error(native):
    class Model:
        @property
        def value(self):
            raise AttributeError("missing")
    sentinel = object()
    assert native.read(Model(), "value", sentinel) is sentinel


def test_other_exception(native):
    failure = ValueError("preserve original")

    class Model:
        @property
        def value(self):
            raise failure
    with pytest.raises(ValueError) as got:
        native.read(Model(), "value", None)
    assert got.value is failure


def test_no_getattr(native):
    class Model:
        def __getattr__(self, name):
            raise AssertionError("hook called")
    sentinel = object()
    assert native.read(Model(), "absent", sentinel) is sentinel
