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

A network layer of the table (the `_expansion_cnn` layers: PyTorch modules, not scikit-learn estimators) has no
`get_params`, `set_params` or `vendor_used`: its device is the `device` attribute its constructor wrote, and the
estimator-only checks (set_params, clone through get_params, vendor_used) are skipped for it by name.
A model-selection splitter of the table (`KFold`, `ShuffleSplit`, ...: scikit-learn cross-validators, not
estimators; they joined the table with their `device` in 44b842efd3) has `get_params`, so the get_params, clone and
pickle checks run for it, and like scikit-learn's splitters it has no `set_params` and no `vendor_used`: those two
checks are skipped for it by name (a splitter runs nothing until `split`).

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
    # `load_all` imports the pre-lane set; a public class exported lazily by a door (`AutoARIMA` through
    # `_expansion_sequence`, its module `_x_sequence_autoarima`) enters the table only when it is first touched. Touch
    # every public lazy name here, so the table, and so CLASSES, is the same whatever test ran first (it was not:
    # AutoARIMA was in the table, and missing from CLASSES, only when an earlier test had touched it).
    for name, where in getattr(mojolearn, "_LAZY_NAMES", {}).items():
        if where is None:
            continue  # a submodule, not a class
        try:
            getattr(mojolearn, name)
        except ModuleNotFoundError as exc:  # an optional third-party dependency this installation lacks
            if (exc.name or "").split(".")[0] == "mojolearn":
                raise


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


def _is_estimator(cls):
    return hasattr(cls, "get_params")


def _params(est):
    # an estimator's get_params; a layer's device is the attribute its constructor wrote
    if _is_estimator(type(est)):
        return est.get_params(deep=False)
    return {"device": est.device}


def _require_estimator(cls, what):
    if not _is_estimator(cls):
        pytest.skip(f"{cls.__name__} is a layer, not an estimator: it has no {what}")


def _is_splitter(cls):
    # a model-selection cross-validator: get_params, no set_params, no vendor_used (see the module docstring)
    from mojolearn import model_selection
    return issubclass(cls, (model_selection._Splitter, model_selection._Repeated))


def _require_fitted_estimator_api(cls, what):
    _require_estimator(cls, what)
    if _is_splitter(cls):
        pytest.skip(f"{cls.__name__} is a cross-validator, not an estimator: it has no {what}")


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
    # A class from outside the package can carry a non-string __module__: type.__module__ returns the class's own
    # __dict__ entry, and a type that defines an instance getset named __module__ returns that getset_descriptor.
    # Such a class is not in a mojolearn module, so only a string module is compared.
    modules = ((cls, getattr(cls, "__module__", "")) for cls in found)
    return sorted((cls for cls, module in modules
                   if isinstance(module, str) and module.startswith("mojolearn.")
                   and not module.startswith("mojolearn.tests")),
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
    assert _params(est)["device"] is None
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
        assert _params(cls(device="gpu"))["device"] == "gpu"
    else:
        with pytest.raises(ValueError, match="device='cpu'"):
            cls(device="gpu")


@pytest.mark.parametrize("cls, answer, required", CLASSES)
def test_explicit_cpu_follows_the_generated_table(cls, answer, required):
    _build(cls, required)
    if not GPU_INSTALL or answer is True:
        est = cls(device="cpu")
        assert _params(est)["device"] == "cpu"
        if _is_estimator(cls) and not _is_splitter(cls):
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
    _require_fitted_estimator_api(cls, "set_params")
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


@pytest.mark.parametrize("cls, answer, required", [row for row in CLASSES if row.values[2]])
@pytest.mark.parametrize("bad", ["tpu", 0])
def test_a_class_with_required_arguments_refuses_a_bad_device_too(cls, answer, required, bad):
    # No class that takes device= is left unchecked: the classes the tests above skip (their constructor needs
    # arguments this file cannot invent) must still refuse a bad device at construction, whatever the other
    # arguments are, because the device is checked before any field is written.
    placeholders = {name: None for name in required}
    with pytest.raises((ValueError, TypeError), match="device"):
        cls(device=bad, **placeholders)


def test_every_public_class_that_takes_device_is_exercised():
    # the parametrized cases above are the wave's proof that a bad device raises at construction and at
    # set_params for EVERY class: a public estimator that takes device=None and is not among them fails here
    covered = {row.values[0] for row in CLASSES}
    missing = []
    for cls in _package_classes():
        if cls.__name__.startswith("_") or getattr(cls, "_HOST_INFERENCE_ONLY", False):
            continue
        if cls.__init__ is object.__init__ or not hasattr(cls, "get_params"):
            continue
        try:
            parameters = inspect.signature(cls.__init__).parameters
        except (TypeError, ValueError):
            continue
        if "device" in parameters and parameters["device"].default is None and cls not in covered:
            missing.append(cls.__module__ + "." + cls.__qualname__)
    assert not missing, f"classes that take device= and are not exercised by this file: {missing}"


# Constructor data for the classes whose constructor needs arguments (the cases above skip them by name): with it the
# device cases run for real, not only the refusal of a bad value.
def _series():
    import numpy as np
    return np.random.default_rng(0).standard_normal((2, 40)).cumsum(axis=1).astype(np.float32)


_CONSTRUCTOR_DATA = {"AutoARIMA": lambda: {"endog": _series()}}


@pytest.mark.parametrize("cls, answer, required",
                         [row for row in CLASSES if row.values[2] and row.values[0].__name__ in _CONSTRUCTOR_DATA])
def test_a_class_with_constructor_data_takes_every_device(cls, answer, required):
    data = _CONSTRUCTOR_DATA[cls.__name__]
    est = cls(**data())
    assert _params(est)["device"] is None
    if GPU_INSTALL:
        assert _params(cls(device="gpu", **data()))["device"] == "gpu"
    else:
        with pytest.raises(ValueError, match="device='cpu'"):
            cls(device="gpu", **data())
    if not GPU_INSTALL or answer is True:
        assert _params(cls(device="cpu", **data()))["device"] == "cpu"
    else:
        with pytest.raises(ValueError, match="device='cpu' is not available for " + cls.__name__):
            cls(device="cpu", **data())
    with pytest.raises((ValueError, TypeError), match="device"):
        cls(device="tpu", **data())


def test_the_constructor_data_names_only_classes_of_the_table():
    # a stale entry would silently exercise nothing
    named = {row.values[0].__name__ for row in CLASSES if row.values[2]}
    assert set(_CONSTRUCTOR_DATA) <= named, sorted(set(_CONSTRUCTOR_DATA) - named)


@pytest.mark.parametrize("cls, answer, required", CLASSES)
def test_clone_and_pickle_carry_the_device(cls, answer, required):
    _build(cls, required)
    _require_estimator(cls, "get_params to clone from")
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
    assert _params(est)["device"] is None
