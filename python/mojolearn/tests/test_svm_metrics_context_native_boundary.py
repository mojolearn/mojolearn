"""Live metrics metadata is admitted before any Python lookup/string hook."""
import numpy as np
import pytest

from mojolearn.svm import LinearSVC

__mojolearn_user_boundary__ = True


@pytest.fixture
def operation():
    from mojolearn import _expansion_metrics

    source = np.array([[-1], [1]] * 4, dtype=np.float32)
    labels = [-5, 9] * 4
    model = LinearSVC(numeric_mode="identical").fit(source, labels)
    return model, source, labels, _expansion_metrics


def _contexts(module):
    context = dict(module._PROG_CONTEXT)
    context["prog_types"] = dict(context["prog_types"])
    context["backend_fields"] = dict(context["backend_fields"])
    context["prog_bindings"] = dict(context["prog_bindings"])
    mode = context["backend_fields"]["_DEFAULT_MODE"]
    context["prog_bindings"][mode] = dict(context["prog_bindings"][mode])
    return context, mode


@pytest.mark.parametrize("level", ["context", "prog_types", "backend_fields", "prog_bindings", "entries"])
def test_metrics_mapping_hooks_are_refused_before_lookup(operation, monkeypatch, level):
    model, source, labels, module = operation
    context, mode = _contexts(module)
    calls = []

    class MappingHook:
        def __getitem__(self, key):
            calls.append(key)
            raise AssertionError("Python metadata lookup")

    if level == "context":
        context = MappingHook()
    elif level == "entries":
        context["prog_bindings"][mode] = MappingHook()
    else:
        context[level] = MappingHook()
    monkeypatch.setattr(module, "_PROG_CONTEXT", context)
    with pytest.raises(TypeError, match="exact string-keyed dictionaries"):
        model.score(source, labels)
    assert calls == []


@pytest.mark.parametrize("level", ["context", "prog_types", "backend_fields", "prog_bindings", "entries"])
def test_metrics_colliding_key_hooks_are_refused_before_lookup(operation, monkeypatch, level):
    model, source, labels, module = operation
    context, mode = _contexts(module)
    maps = {"context": context, "prog_types": context["prog_types"],
            "backend_fields": context["backend_fields"], "prog_bindings": context["prog_bindings"],
            "entries": context["prog_bindings"][mode]}
    names = {"context": "prog_types", "prog_types": "str", "backend_fields": "_DEFAULT_MODE",
             "prog_bindings": mode, "entries": "prog_family"}
    calls = []
    armed = False

    class KeyHook:
        def __hash__(self):
            return hash(names[level])

        def __eq__(self, other):
            if armed:
                calls.append(other)
                raise AssertionError("Python key equality")
            return False

    maps[level][KeyHook()] = None
    armed = True
    monkeypatch.setattr(module, "_PROG_CONTEXT", context)
    with pytest.raises(TypeError, match="exact string-keyed dictionaries"):
        model.score(source, labels)
    assert calls == []


@pytest.mark.parametrize("slot", ["prog_family", "prog_mode_refusal"])
def test_metrics_diagnostic_string_hooks_are_refused(operation, monkeypatch, slot):
    model, source, labels, module = operation
    context, mode = _contexts(module)
    calls = []

    class StringHook:
        def __str__(self):
            calls.append(True)
            raise AssertionError("Python diagnostic formatting")

    if slot == "prog_family":
        context["prog_bindings"][mode][slot] = StringHook()
    else:
        context["backend_fields"]["_DEFAULT_MODE"] = 0
        context[slot] = StringHook()
    monkeypatch.setattr(module, "_PROG_CONTEXT", context)
    with pytest.raises(TypeError, match="requires a native string"):
        model.score(source, labels)
    assert calls == []
