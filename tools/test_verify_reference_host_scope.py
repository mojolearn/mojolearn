"""Inert JSON-only producer tests; never import the product or run a fit."""
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("vref_scope_tested", ROOT / "python/mojolearn/_verify_reference.py")
VREF = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(VREF)
A, B = "a" * 16, "b" * 16


def table(tmp_path, monkeypatch, columns, host_only=True):
    harness = SimpleNamespace(LANES={"x": None}, FIXTURES=["base"], LANE_REVISIONS={},
        __file__=__file__, _h=lambda _: "f" * 16, fixture=lambda _: (0, 0, 0),
        heldout=lambda _: 0, BATCH_ALONE=16, BATCH_SPLIT=(3, 7))
    paths = []
    for stem, vendor, value in columns:
        path = tmp_path / f"{stem}.json"
        path.write_text(json.dumps(dict(mode="identical", commit="a" * 40, vendor=vendor,
            cells={"x/base": dict(verdict="STABLE", hashes=[value])},
            fixtures={"base": dict(X="f" * 16, y_clf="f" * 16, y_reg="f" * 16)},
            heldout={"base": dict(X="f" * 16)})))
        paths.append(str(path))
    monkeypatch.setattr(VREF, "_host_only_source_lanes", lambda _: {"x"} if host_only else set())
    monkeypatch.setattr(VREF, "_commit_time", lambda *a: 1)
    return VREF.build_table(paths, harness, str(tmp_path), parts=("train",))


def test_cpu_only_single_execution_is_retained_with_honest_scope(tmp_path, monkeypatch):
    result = table(tmp_path, monkeypatch, [("cpu", "cpu-test", A)])
    entry = result["cells"]["x/base"]["train"]
    assert entry == dict(ref=A, cols={"cpu": 0}, scope="cpu-host-route-only")
    assert result["format"] == "mojolearn.verify-reference.v1"
    assert result["records"][0]["commit"] == "a" * 40


def test_gpu_observations_preserved_but_never_decide_host_arithmetic(tmp_path, monkeypatch):
    result = table(tmp_path, monkeypatch, [("cpu", "cpu-test", A),
        ("zz-nvidia", "nvidia-test", B), ("zzz-ptx", "nvidia-ptx-test", B)])
    entry = result["cells"]["x/base"]["train"]
    assert entry["ref"] == A and not entry.get("conflict")
    assert entry["cols"]["nvidia"][1] == B and entry["cols"]["ptx"][1] == B
    assert not result["ptx_divergent"]
    assert len(result["records"]) == 3


@pytest.mark.parametrize("columns", [[("cpu", "cpu-test", A)],
    [("nv", "nvidia-test", A), ("ptx", "nvidia-ptx-test", A)]])
def test_ordinary_gpu_lane_still_needs_independent_corroboration(tmp_path, monkeypatch, columns):
    assert not table(tmp_path, monkeypatch, columns, host_only=False)["cells"]


def test_gpu_records_cannot_replace_missing_host_observation(tmp_path, monkeypatch):
    assert not table(tmp_path, monkeypatch, [("amd", "amd-test", A),
        ("nv", "nvidia-test", A)])["cells"]


def test_ordinary_ptx_divergence_still_blocks(tmp_path, monkeypatch):
    result = table(tmp_path, monkeypatch, [("amd", "amd-test", A),
        ("nv", "nvidia-test", A), ("ptx", "nvidia-ptx-test", B)], host_only=False)
    assert len(result["ptx_divergent"]) == 1


def test_no_source_tooling_does_not_grant_exemptions(tmp_path):
    assert VREF._host_only_source_lanes(tmp_path) == frozenset()


def test_producer_reads_explicit_frozen_source_with_current_tooling(tmp_path, monkeypatch):
    (tmp_path / 'tools').mkdir()
    (tmp_path / 'tools/lane_applicability.py').write_text('# frozen source marker\n')
    calls = []
    def run(command, **kwargs):
        calls.append(command)
        return SimpleNamespace(stdout='["x"]')
    monkeypatch.setattr(VREF.subprocess, 'run', run)
    assert VREF._host_only_source_lanes(tmp_path) == {'x'}
    assert calls[0][-2:] == ['--source-root', str(tmp_path)]
    assert Path(calls[0][1]) == ROOT / 'tools/identity_admission_guard.py'
