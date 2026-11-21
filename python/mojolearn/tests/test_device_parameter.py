# SPDX-License-Identifier: Apache-2.0
"""`device=None | "gpu" | "cpu"` at the Python boundary, over EVERY estimator class that takes it.

One table, read from the package (`mojolearn._mode.device_report()`): the classes whose constructor is the shared
wrapper, and the classes whose own constructor names `device` and declare `_mojolearn_device`. For each:

  * the default is None and `get_params` returns what was passed;
  * an unknown value is refused in words at construction and at `set_params`, and the refused `set_params` leaves
    the estimator as it was;
  * an explicit `"gpu"` runs on a GPU installation and is refused in words that name `device="cpu"` on a CPU-only one;
  * an explicit `"cpu"` always runs on a CPU-only installation; beside a GPU set it is accepted exactly for the
    classes the generated table (`mojolearn._device_table.CPU_CLASSES`) answers True for, and refused for every
    other class in the table's own sentence (nothing runs on a device its caller did not name);
  * clone (`type(est)(**est.get_params())`) and pickle carry it;
  * `vendor_used()` of a `device="cpu"` estimator answers `cpu`.

No model is fitted here (the per-family device tests fit: test_linear_device_boundary.py,
test_logistic_device_boundary.py, test_family_device_boundary.py). A class whose constructor needs arguments this
file cannot invent is skipped by name.
"""
import inspect
import pickle

import pytest

import mojolearn
from mojolearn import _device_table, _families, _mode

GPU_INSTALL = mojolearn.vendor() != "cpu"


def _load_every_family():
    # every lazily loaded family, so the table holds every class
    loader = getattr(_families, "load_all", None)
    if loader is not None:
        loader()


def _classes():
    _load_every_family()
    rows = []
    for cls, answer in sorted(_device_table.CPU_CLASSES.items(), key=lambda item: item[0].__module__ + item[0].__name__):
        if not cls.__module__.startswith("mojolearn.") or cls.__module__.startswith("mojolearn.tests"):
            continue
        if cls.__name__.startswith("_") or getattr(cls, "_HOST_INFERENCE_ONLY", False):
            continue
        if cls.__init__ is object.__init__:
            continue
        parameters = inspect.signature(cls.__init__).parameters
        if "device" not in parameters or parameters["device"].default is not None:
            continue  # no device keyword, or an older device keyword of its own (default "gpu")
        required = [p.name for p in list(parameters.values())[1:]
                    if p.default is p.empty and p.kind in (p.POSITIONAL_OR_KEYWORD, p.KEYWORD_ONLY)]
        rows.append(pytest.param(cls, answer, tuple(required), id=cls.__module__.rsplit(".", 1)[-1] + "." + cls.__name__))
    return rows


CLASSES = _classes()


def _build(cls, required, **keywords):
    if required:
        pytest.skip(f"{cls.__name__} needs constructor arguments: {', '.join(required)}")
    return cls(**keywords)


def test_the_table_is_not_empty():
    assert CLASSES, "no estimator class takes device="
    assert all(answer is True or isinstance(answer, str) for answer in _device_table.CPU_CLASSES.values())


def _package_classes():
    # every class the package defines, found from `object` down: independent of the table's own derivation
    _load_every_family()
    found, todo = set(), list(object.__subclasses__())
    while todo:
        cls = todo.pop()
        if cls in found:
            continue
        found.add(cls)
        todo.extend(type.__subclasses__(cls))
    return sorted((cls for cls in found
                   if cls.__module__.startswith("mojolearn.") and not cls.__module__.startswith("mojolearn.tests")),
                  key=lambda cls: cls.__module__ + "." + cls.__qualname__)


def test_every_marked_class_is_in_the_table_whatever_its_base():
    # `_mojolearn_device = True` is the class's own statement that it runs on both devices: the compiled
    # constructor check reads the TABLE, so a marked class without a row would be refused `device="cpu"`.
    missing = [cls.__module__ + "." + cls.__qualname__ for cls in _package_classes()
               if getattr(cls, "_mojolearn_device", False) is True and _device_table.CPU_CLASSES.get(cls) is not True]
    assert not missing, f"marked classes without a True row in the device table: {missing}"


def test_every_estimator_that_takes_device_is_in_the_table_whatever_its_base():
    missing = []
    for cls in _package_classes():
        if cls.__init__ is object.__init__ or not hasattr(cls, "get_params") or cls in _device_table.CPU_CLASSES:
            continue
        try:
            parameters = inspect.signature(cls.__init__).parameters
        except (TypeError, ValueError):
            continue
        if "device" in parameters and cls.__module__ in __import__("sys").modules and \
                vars(__import__("sys").modules[cls.__module__]).get(cls.__name__) is cls:
            missing.append(cls.__module__ + "." + cls.__qualname__)
    assert not missing, f"estimator classes that take device= without a row in the device table: {missing}"


def test_the_report_lists_every_class_of_the_table():
    rows = _mode.device_report()
    assert len(rows) == len(_device_table.CPU_CLASSES)
    assert all(len(row) == 3 for row in rows)


@pytest.mark.parametrize("cls, answer, required", CLASSES)
def test_default_is_none_and_get_params_returns_it(cls, answer, required):
    est = _build(cls, required)
    assert est.get_params(deep=False)["device"] is None
    assert est.device is None


@pytest.mark.parametrize("cls, answer, required", CLASSES)
@pytest.mark.parametrize("bad", ["tpu", "GPU", " cpu", "", "None", 0, False, b"cpu"])
def test_unknown_device_is_refused_in_words(cls, answer, required, bad):
    _build(cls, required)
    with pytest.raises((ValueError, TypeError), match="device"):
        cls(device=bad)


@pytest.mark.parametrize("cls, answer, required", CLASSES)
def test_explicit_gpu(cls, answer, required):
    _build(cls, required)
    if GPU_INSTALL:
        assert cls(device="gpu").get_params(deep=False)["device"] == "gpu"
    else:
        with pytest.raises(ValueError, match="device='cpu'"):
            cls(device="gpu")


@pytest.mark.parametrize("cls, answer, required", CLASSES)
def test_explicit_cpu_follows_the_generated_table(cls, answer, required):
    _build(cls, required)
    if not GPU_INSTALL or answer is True:
        est = cls(device="cpu")
        assert est.get_params(deep=False)["device"] == "cpu"
        assert est.vendor_used() == "cpu"
    else:
        with pytest.raises(ValueError) as refused:
            cls(device="cpu")
        text = str(refused.value)
        assert "device='cpu' is not available for " + cls.__name__ in text and answer in text
        assert "device=None" in text


@pytest.mark.parametrize("cls, answer, required", CLASSES)
def test_set_params_checks_the_device_before_the_estimator_changes(cls, answer, required):
    est = _build(cls, required)
    before = dict(est.get_params(deep=False))
    with pytest.raises((ValueError, TypeError), match="device"):
        est.set_params(device="tpu")
    assert est.get_params(deep=False) == before
    if GPU_INSTALL and answer is not True:
        with pytest.raises(ValueError, match="device='cpu' is not available"):
            est.set_params(device="cpu")
        assert est.get_params(deep=False) == before
    ok = "gpu" if GPU_INSTALL else "cpu"
    assert est.set_params(device=ok) is est
    assert est.get_params(deep=False)["device"] == ok
    assert est.set_params(device=None).get_params(deep=False)["device"] is None


@pytest.mark.parametrize("cls, answer, required", CLASSES)
def test_clone_and_pickle_carry_the_device(cls, answer, required):
    _build(cls, required)
    values = [None, "gpu" if GPU_INSTALL else "cpu"]
    if GPU_INSTALL and answer is True:
        values.append("cpu")
    for device in values:
        est = cls(device=device)
        twin = type(est)(**est.get_params(deep=False))
        assert twin.get_params(deep=False)["device"] == device
        try:
            data = pickle.dumps(est)
        except Exception as error:  # a class that does not pickle today is not this file's subject
            pytest.skip(f"{cls.__name__} does not pickle: {type(error).__name__}")
        assert pickle.loads(data).get_params(deep=False)["device"] == device


@pytest.mark.parametrize("cls, answer, required", CLASSES)
def test_a_constructor_run_again_keeps_the_rule(cls, answer, required):
    est = _build(cls, required)
    with pytest.raises((ValueError, TypeError), match="device"):
        est.__init__(device="tpu")
    est.__init__()
    assert est.get_params(deep=False)["device"] is None
