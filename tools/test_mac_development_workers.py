"""Recorded Mac development workers remain exact admission inputs; no builds."""
import copy

import pytest

import release_requirement_dependencies as deps
import release_retained_compile_evidence as compiled
from release_requirements import digest


def group(name, jobs):
    env = {} if jobs is None else {"MOJOLEARN_COMPILE_JOBS": jobs}
    return dict(mode="identical", members=[name], environment=env,
                host_toolchain={"sdk": "recorded-sdk"})


def collect(monkeypatch, groups):
    monkeypatch.setattr(compiled, "_dev_facts", lambda *args: dict(
        source_revision="a" * 40, groups=groups))
    return compiled._dev_collect("unused", {}, {})


def test_mixed_recorded_workers_are_carried_per_binding(monkeypatch):
    result = collect(monkeypatch, {"a": group("one", "10"), "b": group("two", "2")})
    mapping = result["configuration"]["dependency_context"]["selected_binding_compile_jobs"]
    assert mapping == {"osx-arm64/metal/apple/identical/one": "10",
                       "osx-arm64/metal/apple/identical/two": "2"}
    assert result["configuration"]["build_flags"] == {}
    key = "osx-arm64/metal/apple/identical/one"
    source = dict(bindings=[dict(key=k, identity=dict(flags={"compile_jobs": "2"})) for k in mapping])
    original = copy.deepcopy(source)
    selected = deps._selected_binding_flags(source, result["configuration"])
    assert source == original
    actual = {row["key"]: row["identity"] for row in selected["bindings"]}
    assert actual[key]["flags"]["compile_jobs"] == "10"
    assert digest(actual[key]) != digest(original["bindings"][0]["identity"])
    compiled.check_flags(actual[key]["flags"], {"MOJOLEARN_COMPILE_JOBS": "10"})
    with pytest.raises(ValueError, match="compiled flag differs"):
        compiled.check_flags(actual[key]["flags"], {"MOJOLEARN_COMPILE_JOBS": "2"})


@pytest.mark.parametrize("jobs", [None, "", "0", "-1", "1.5", " 10", "01", 10, True])
def test_absent_or_malformed_workers_are_not_defaulted(monkeypatch, jobs):
    with pytest.raises(ValueError, match="worker count is missing or malformed"):
        collect(monkeypatch, {"a": group("one", jobs)})


def test_conflicting_worker_records_are_refused(monkeypatch):
    with pytest.raises(ValueError, match="worker counts conflict"):
        collect(monkeypatch, {"a": group("one", "10"), "b": group("one", "2")})


def test_duplicate_binding_groups_are_not_silently_overwritten(monkeypatch):
    with pytest.raises(ValueError, match="groups conflict"):
        collect(monkeypatch, {"a": group("one", "10"), "b": group("one", "10")})
