"""Reference publication with inert JSON and mocked verifier calls; no native code."""
import copy
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("admission_guard_tested", ROOT / "tools/identity_admission_guard.py")
GUARD = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(GUARD)
HASH = "a" * 16
OTHER = "b" * 16


def tables():
    records = [dict(class_=cls, commit="1" * 40) for cls in ("apple", "nvidia", "amd", "cpu", "ptx")]
    records = [{"class": r["class_"], "commit": "0" * 40 if r["class_"] == "cpu" else r["commit"]} for r in records]
    entry = dict(ref=HASH, cols=dict(apple=0, nvidia=1, amd=2))
    base = dict(format="mojolearn.verify-reference.v1", fixtures={"base": {}, "odd": {}},
                heldout={}, records=records, cells={"historical/base": {
                    "train": dict(ref=None, cols={}, conflict=True)}}, ptx_divergent=[],
                lane_revisions={"historical": "old"}, admission_policy={"legacy": True})
    candidate = copy.deepcopy(base)
    for fixture in base["fixtures"]:
        candidate["cells"][f"selected/{fixture}"] = {
            part: copy.deepcopy(entry) for part in ("train", "infer", "model", "batch")}
    return base, candidate


def test_retained_agreeing_gpu_columns_and_unselected_legacy_debt_are_preserved():
    base, candidate = tables()
    assert GUARD.validate(base, candidate, ["selected"])["admissible"]
    # A new CPU-only observation can use all three existing GPU witnesses.
    candidate["cells"]["selected/base"]["train"]["cols"]["cpu"] = 3
    assert GUARD.validate(base, candidate, ["selected"])["admissible"]


@pytest.mark.parametrize("cls", ["apple", "nvidia", "amd"])
@pytest.mark.parametrize("conflict_flag", [True, False])
def test_each_gpu_disagreement_refuses_even_without_a_conflict_flag(cls, conflict_flag):
    base, candidate = tables()
    entry = candidate["cells"]["selected/base"]["train"]
    entry["cols"][cls] = [entry["cols"][cls], OTHER]
    if conflict_flag:
        entry.update(ref=None, conflict=True)
    assert not GUARD.validate(base, candidate, ["selected"])["admissible"]


@pytest.mark.parametrize("change", ["missing_class", "missing_cell", "missing_part", "invalid_index", "wrong_class"])
def test_missing_or_invalid_selected_witness_refuses(change):
    base, candidate = tables()
    entry = candidate["cells"]["selected/base"]["train"]
    if change == "missing_class":
        del entry["cols"]["apple"]
    elif change == "missing_cell":
        del candidate["cells"]["selected/odd"]
    elif change == "missing_part":
        del candidate["cells"]["selected/base"]["model"]
    elif change == "invalid_index":
        entry["cols"]["apple"] = 100
    else:
        entry["cols"]["apple"] = 1
    assert not GUARD.validate(base, candidate, ["selected"])["admissible"]


def test_older_cpu_disagreement_remains_superseded_but_builder_conflict_refuses():
    base, candidate = tables()
    entry = candidate["cells"]["selected/base"]["train"]
    entry["cols"]["cpu"] = [3, OTHER]
    assert GUARD.validate(base, candidate, ["selected"])["admissible"]
    entry.update(ref=None, conflict=True)
    assert not GUARD.validate(base, candidate, ["selected"])["admissible"]


def test_na_parts_and_gpu_only_cells_do_not_invent_a_cpu_requirement():
    base, candidate = tables()
    for cell in candidate["cells"].values():
        if "model" in cell:
            cell["model"] = dict(ref="n/a:no-model", cols=dict(apple=0, nvidia=1, amd=2))
    assert GUARD.validate(base, candidate, ["selected"])["admissible"]


@pytest.mark.parametrize("summary_present", [False, True])
def test_ptx_disagreement_blocks_even_when_scoped_merge_keeps_old_summary(summary_present):
    base, candidate = tables()
    candidate["cells"]["selected/base"]["train"]["cols"]["ptx"] = [4, OTHER]
    if summary_present:
        base["ptx_divergent"] = candidate["ptx_divergent"] = [{"cell": "selected/base"}]
    assert not GUARD.validate(base, candidate, ["selected"])["admissible"]


def test_agreeing_ptx_does_not_replace_a_missing_gpu_witness():
    base, candidate = tables()
    cols = candidate["cells"]["selected/base"]["train"]["cols"]
    cols["ptx"] = 4
    assert GUARD.validate(base, candidate, ["selected"])["admissible"]
    del cols["apple"]
    assert not GUARD.validate(base, candidate, ["selected"])["admissible"]


@pytest.mark.parametrize("change", ["cell", "record", "revision", "policy"])
def test_unselected_history_cannot_change(change):
    base, candidate = tables()
    if change == "cell":
        candidate["cells"]["historical/base"]["train"]["ref"] = HASH
    elif change == "record":
        candidate["records"][0]["commit"] = "2" * 40
    elif change == "revision":
        candidate["lane_revisions"]["historical"] = "new"
    else:
        candidate["admission_policy"] = {"silently_qualified": True}
    assert not GUARD.validate(base, candidate, ["selected"])["admissible"]


def test_publication_is_atomic_and_refuses_an_intervening_edit(tmp_path):
    base, candidate = tables()
    old, new, destination = (tmp_path / name for name in ("base.json", "candidate.json", "table.json"))
    old.write_text(json.dumps(base))
    new.write_text(json.dumps(candidate))
    destination.write_bytes(old.read_bytes())
    GUARD.publish(new, old, destination, ["selected"])
    assert destination.read_bytes() == new.read_bytes()
    with pytest.raises(ValueError, match="changed during admission"):
        GUARD.publish(new, old, destination, ["selected"])


def test_selection_preserves_whole_original_record_and_rejects_absent_lane(tmp_path):
    path = tmp_path / "column.json"
    path.write_text(json.dumps({"cells": {"selected/base": {}, "held/base": {"verdict": "STABLE"}}}))
    original = path.read_bytes()
    assert GUARD.selected_lanes([path]) == ["held", "selected"]
    assert GUARD.selected_lanes([path], "selected") == ["selected"]
    assert path.read_bytes() == original
    with pytest.raises(ValueError, match="absent"):
        GUARD.selected_lanes([path], "unknown")


def wrapper_fixture(tmp_path, candidate, base, fail_selftest=False):
    repo = tmp_path / "repo"
    (repo / "tools").mkdir(parents=True)
    for name in ("admit_identity_columns.sh", "identity_admission_guard.py"):
        shutil.copyfile(ROOT / "tools" / name, repo / "tools" / name)
    table = repo / "python/mojolearn/verify_reference/table.json"
    table.parent.mkdir(parents=True)
    table.write_text(json.dumps(base))
    for name in (".libs/libMojolearnMath.so", ".dylibs/libMojolearnMath.dylib"):
        path = table.parent.parent / name
        path.parent.mkdir(parents=True)
        path.write_bytes(b"inert; never loaded")
    incoming = tmp_path / "incoming"
    incoming.mkdir()
    raw = incoming / "apple.identical.json"
    raw.write_text(json.dumps({"vendor": "apple-inert", "commit": "1" * 40,
                               "cells": {"selected/base": {}, "held/base": {"verdict": "DIVERGENT"}}}))
    stub = tmp_path / "candidate.json"
    stub.write_text(json.dumps(candidate))
    shim = tmp_path / "inert_python.py"
    shim.write_text('''import json, os, pathlib, sys
a = sys.argv[1:]
if a[0].endswith("identity_host_preflight.py"):
    raise SystemExit(0)
if a[0].endswith("identity_columns.py"):
    if a[1] == "check":
        for path in a[2:]:
            print("IDCOLUMN " + path + " admissible=yes class=apple vendor=apple-inert commit=" + "1"*40 + " repeats=1")
    else:
        pathlib.Path(a[a.index("--out")+1]).write_text("{}")
    raise SystemExit(0)
if a[:3] == ["-m", "mojolearn", "verify"]:
    if "--emit-reference" in a:
        assert "--reference-table" in a and a[a.index("--lanes")+1] == "selected"
        pathlib.Path(a[a.index("--emit-reference")+1]).write_bytes(pathlib.Path(os.environ["STUB_CANDIDATE"]).read_bytes())
        print("{}")
    if "--self-test" in a and os.environ.get("FAIL_SELFTEST") == "1":
        raise SystemExit(7)
    raise SystemExit(0)
os.execv(sys.executable, [sys.executable] + a)
''')
    bindir = tmp_path / "bin"
    bindir.mkdir()
    git = bindir / "git"
    git.write_text('#!/bin/sh\ncase "$1" in\ncat-file) exit 0;;\nlog) echo 1; exit 0;;\n*) exit 91;;\nesac\n')
    git.chmod(0o755)
    evidence = tmp_path / "evidence"
    env = dict(os.environ, PATH=str(bindir) + os.pathsep + os.environ["PATH"],
               PYTHON=f"{sys.executable} {shim}", STUB_CANDIDATE=str(stub),
               EVIDENCE=str(evidence), ADMIT_DATE="2026-10-23", REF_TAG="inert",
               FAIL_SELFTEST="1" if fail_selftest else "0")
    return repo, table, raw, env


@pytest.mark.parametrize("problem", ["apple", "nvidia", "amd", "missing", "selftest"])
def test_wrapper_failure_cannot_mutate_shipped_table(tmp_path, problem):
    base, candidate = tables()
    if problem in ("apple", "nvidia", "amd"):
        cols = candidate["cells"]["selected/base"]["train"]["cols"]
        cols[problem] = [cols[problem], OTHER]
    elif problem == "missing":
        del candidate["cells"]["selected/base"]["train"]
    repo, table, raw, env = wrapper_fixture(tmp_path, candidate, base, problem == "selftest")
    before, original = table.read_bytes(), raw.read_bytes()
    result = subprocess.run(["bash", str(repo / "tools/admit_identity_columns.sh"), "--lanes", "selected", str(raw.parent)],
                            env=env, capture_output=True, text=True)
    assert result.returncode != 0, result.stdout + result.stderr
    assert table.read_bytes() == before
    assert raw.read_bytes() == original


@pytest.mark.parametrize("check_only", [False, True])
def test_wrapper_checks_candidate_and_preserves_original_unselected_record(tmp_path, check_only):
    base, candidate = tables()
    repo, table, raw, env = wrapper_fixture(tmp_path, candidate, base)
    before, original = table.read_bytes(), raw.read_bytes()
    result = subprocess.run(["bash", str(repo / "tools/admit_identity_columns.sh"),
                             *(["--check-only"] if check_only else []), "--lanes", "selected", str(raw.parent)],
                            env=env, capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
    assert raw.read_bytes() == original
    if check_only:
        assert table.read_bytes() == before
        assert not (repo / "bench").exists()
    else:
        assert json.loads(table.read_text()) == candidate
        retained = repo / "bench/results/identity_break/2026-10-23-inert/apple-inert" / raw.name
        assert retained.read_bytes() == original
