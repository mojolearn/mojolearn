# SPDX-License-Identifier: Apache-2.0
"""Scaler `__sklearn_tags__` is built natively from scikit-learn's own tag
records: cold (scikit-learn not imported yet), warm, scikit-learn absent,
class metadata overrides and their admission, and a NumPy/scikit-learn-free
`import mojolearn`."""
import dataclasses
import json
import subprocess
import sys

import pytest

SCALERS = ("StandardScaler", "MinMaxScaler", "MaxAbsScaler", "RobustScaler")
FLOAT32_FIXED = ("StandardScaler", "MinMaxScaler")


def _run(code):
    done = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=300)
    assert done.returncode == 0, done.stderr[-2000:]
    return done.stdout.strip().splitlines()[-1]


def test_import_mojolearn_imports_neither_sklearn_nor_numpy():
    out = _run("import sys, json, mojolearn; "
               "print(json.dumps([m for m in ('sklearn', 'numpy', 'scipy') if m in sys.modules]))")
    assert json.loads(out) == []


def test_cold_call_returns_exact_sklearn_records():
    pytest.importorskip("sklearn")
    code = f"""
import sys, json, dataclasses, mojolearn
assert 'sklearn' not in sys.modules
out = {{}}
for name in {SCALERS!r}:
    tags = getattr(mojolearn, name)().__sklearn_tags__()
    from sklearn.utils import Tags, TargetTags, InputTags, TransformerTags
    assert type(tags) is Tags and type(tags.target_tags) is TargetTags
    assert type(tags.input_tags) is InputTags and type(tags.transformer_tags) is TransformerTags
    out[name] = dataclasses.asdict(tags)
print(json.dumps(out))
"""
    cold = json.loads(_run(code))
    for name in SCALERS:
        assert cold[name] == _expected(_cls(name)()), name


def _cls(name):
    import mojolearn
    return getattr(mojolearn, name)


def _expected(model):
    # The legacy Python answer: the shared `_mode.sklearn_tags` with the
    # float32-preserving transformer tags of StandardScaler/MinMaxScaler.
    from sklearn.utils import TransformerTags
    from mojolearn._mode import sklearn_tags
    tags = sklearn_tags(model)
    if type(model).__name__ in FLOAT32_FIXED or any(c.__name__ in FLOAT32_FIXED for c in type(model).__mro__):
        tags.transformer_tags = TransformerTags(preserves_dtype=["float32"])
    return dataclasses.asdict(tags)


@pytest.mark.parametrize("name", SCALERS)
def test_warm_call_matches_legacy_without_python_tag_constructors(name, monkeypatch):
    pytest.importorskip("sklearn")
    from sklearn.utils import Tags, TargetTags, InputTags, TransformerTags
    model = _cls(name)()
    expected = _expected(model)

    def poison(*args, **kwargs):
        raise AssertionError("Python tag constructor reached")

    from mojolearn import _mode
    with monkeypatch.context() as patch:
        patch.setattr(_mode, "sklearn_tags", poison)
        for tag_class in (Tags, TargetTags, InputTags, TransformerTags):
            patch.setattr(tag_class, "__init__", poison)
        first = model.__sklearn_tags__()
        second = model.__sklearn_tags__()
    assert type(first) is Tags
    assert dataclasses.asdict(first) == expected
    assert first is not second and first.input_tags is not second.input_tags
    assert first.transformer_tags.preserves_dtype is not second.transformer_tags.preserves_dtype
    first.transformer_tags.preserves_dtype.append("float64")
    assert dataclasses.asdict(second) == expected


def test_sklearn_absent_raises_import_error():
    code = f"""
import sys, importlib.abc
class Block(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path=None, target=None):
        if name == 'sklearn' or name.startswith('sklearn.'):
            raise ModuleNotFoundError("No module named 'sklearn'", name='sklearn')
sys.meta_path.insert(0, Block())
import mojolearn
seen = []
for name in {SCALERS!r}:
    try:
        getattr(mojolearn, name)().__sklearn_tags__()
    except ImportError as error:
        seen.append(name)
print(','.join(seen))
"""
    assert _run(code) == ",".join(SCALERS)


def test_class_metadata_overrides_are_admitted():
    pytest.importorskip("sklearn")
    import mojolearn

    class PCA(mojolearn.StandardScaler):  # a shared sparse-input class name
        _sklearn_input_tags = {"allow_nan": True}
        _sklearn_target_tags = {"positive_only": True}
        _sklearn_requires_fit = False

    class Regressing(mojolearn.MinMaxScaler):
        _estimator_type = "regressor"

    class Dtypes(mojolearn.MaxAbsScaler):
        _sklearn_preserves_dtype = ("float32", "float64")

    tags = PCA().__sklearn_tags__()
    assert dataclasses.asdict(tags) == _expected(PCA())
    assert tags.input_tags.sparse is True and tags.input_tags.allow_nan is True
    assert tags.target_tags.positive_only is True and tags.requires_fit is False
    tags = Regressing().__sklearn_tags__()
    assert dataclasses.asdict(tags) == _expected(Regressing())
    assert tags.estimator_type == "regressor" and tags.target_tags.required is True
    assert tags.regressor_tags is not None and tags.classifier_tags is None
    tags = Dtypes().__sklearn_tags__()
    assert tags.transformer_tags.preserves_dtype == ["float32", "float64"]
    assert dataclasses.asdict(tags) == _expected(Dtypes())
    assert RobustScaler_allows_nan()


def RobustScaler_allows_nan():
    import mojolearn
    return mojolearn.RobustScaler().__sklearn_tags__().input_tags.allow_nan is True


@pytest.mark.parametrize("attrs, error", [
    ({"_sklearn_input_tags": [("allow_nan", True)]}, TypeError),
    ({"_sklearn_input_tags": {"allow_nan": 1}}, TypeError),
    ({"_sklearn_target_tags": {"no_such_tag": True}}, AttributeError),
    ({"_sklearn_requires_fit": 0}, TypeError),
    ({"_estimator_type": 3}, TypeError),
])
@pytest.mark.parametrize("name", SCALERS)
def test_invalid_metadata_is_refused(name, attrs, error):
    pytest.importorskip("sklearn")
    sub = type("Sub", (_cls(name),), dict(attrs))
    with pytest.raises(error):
        sub().__sklearn_tags__()


@pytest.mark.parametrize("name", SCALERS)
def test_tag_callback_is_refused_without_invocation(name):
    pytest.importorskip("sklearn")
    calls = []

    def hook(self):
        calls.append(1)
        return {"required": True}

    sub = type("Sub", (_cls(name),), {"_sklearn_instance_target_tags": hook})
    with pytest.raises(TypeError, match="_sklearn_instance_target_tags"):
        sub().__sklearn_tags__()
    assert calls == []
