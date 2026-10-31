"""Reference publication with inert JSON and mocked verifier calls; no native code."""
import copy
import hashlib
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


@pytest.fixture(autouse=True)
def inert_scope(monkeypatch):
    monkeypatch.setattr(GUARD, "host_only_declarations", lambda: frozenset())


def tables():
    records = [dict(class_=cls, commit="1" * 40) for cls in ("apple", "nvidia", "amd", "cpu", "ptx")]
    records = [{"class": r["class_"], "commit": "0" * 40 if r["class_"] == "cpu" else r["commit"]} for r in records]
    entry = dict(ref=HASH, cols=dict(apple=0, nvidia=1, amd=2, cpu=3))
    base = dict(format="mojolearn.verify-reference.v1", fixtures={"base": {}, "odd": {}},
                heldout={}, records=records, cells={"historical/base": {
                    "train": dict(ref=None, cols={}, conflict=True)}}, ptx_divergent=[],
                lane_revisions={"historical": "old"}, admission_policy={"legacy": True})
    candidate = copy.deepcopy(base)
    for fixture in base["fixtures"]:
        candidate["cells"][f"selected/{fixture}"] = {
            part: copy.deepcopy(entry) for part in ("train", "infer", "model", "batch")}
    return base, candidate


NO_DECLARATIONS = (frozenset(), ())
GPU_ONLY = (frozenset({"selected"}), ())


def test_retained_agreeing_columns_and_unselected_legacy_debt_are_preserved():
    base, candidate = tables()
    assert GUARD.validate(base, candidate, ["selected"], NO_DECLARATIONS)["admissible"]
    assert GUARD.validate(base, candidate, ["selected"])["admissible"]


def test_cpu_equal_to_the_gpu_reference_passes():
    base, candidate = tables()
    report = GUARD.validate(base, candidate, ["selected"], NO_DECLARATIONS)
    assert report["admissible"] and report["numerical_parts"] == 8


@pytest.mark.parametrize("superseded", [False, True])
def test_cpu_disagreement_refuses_naming_the_cell(superseded):
    base, candidate = tables()
    entry = candidate["cells"]["selected/base"]["train"]
    # An older-commit cpu value (build_table's superseded pair) refuses too.
    entry["cols"]["cpu"] = [3, OTHER]
    report = GUARD.validate(base, candidate, ["selected"], GPU_ONLY if superseded else NO_DECLARATIONS)
    assert not report["admissible"]
    assert report["problems"] == [
        f"selected/base/train: cpu disagrees with the GPU reference (cpu={OTHER}, gpu={HASH})"]


def _timed(base, candidate, cpu_time, gpu_time):
    for table in (base, candidate):
        for record in table["records"]:
            record["commit_time"] = cpu_time if record["class"] == "cpu" else gpu_time
    candidate["cells"]["selected/base"]["train"]["cols"]["cpu"] = [3, OTHER]


def test_older_cpu_record_asks_for_a_re_record():
    base, candidate = tables()
    _timed(base, candidate, 1, 2)
    report = GUARD.validate(base, candidate, ["selected"], NO_DECLARATIONS)
    assert report["problems"] == [
        "selected/base/train: cpu column older than the GPU reference "
        f"(cpu@{'0' * 9} vs gpu@{'1' * 9}); re-record the cpu column"]


@pytest.mark.parametrize("cpu_time", [2, 3])
def test_same_or_newer_cpu_record_is_a_disagreement_to_debug(cpu_time):
    base, candidate = tables()
    _timed(base, candidate, cpu_time, 2)
    report = GUARD.validate(base, candidate, ["selected"], NO_DECLARATIONS)
    assert report["problems"] == [
        f"selected/base/train: cpu disagrees with the GPU reference (cpu={OTHER}, gpu={HASH})"]


def test_missing_cpu_column_refuses():
    base, candidate = tables()
    del candidate["cells"]["selected/odd"]["infer"]["cols"]["cpu"]
    report = GUARD.validate(base, candidate, ["selected"], NO_DECLARATIONS)
    assert report["problems"] == ["selected/odd/infer: no cpu column"]


def test_cpu_absent_or_na_on_a_declared_gpu_only_lane_passes():
    base, candidate = tables()
    for cell in (candidate["cells"][f"selected/{f}"] for f in ("base", "odd")):
        for part, entry in cell.items():
            if part == "train":
                entry["cols"]["cpu"] = [3, "n/a:cpu-by-design (no CPU implementation)"]
            else:
                entry["cols"].pop("cpu", None)
    assert GUARD.validate(base, candidate, ["selected"], GPU_ONLY)["admissible"]
    assert GUARD.validate(base, candidate, ["selected"], (frozenset(), ("sel",)))["admissible"]
    report = GUARD.validate(base, candidate, ["selected"], NO_DECLARATIONS)
    assert not report["admissible"]
    assert any("cpu N/A is not declared" in p for p in report["problems"])
    assert any(p.endswith(": no cpu column") for p in report["problems"])


def test_gpu_saved_file_model_na_passes_only_on_the_model_part():
    base, candidate = tables()
    na = "n/a:gpu-saved-file (a CPU column loads the GPU column's model and writes none)"
    for fixture in ("base", "odd"):
        candidate["cells"][f"selected/{fixture}"]["model"]["cols"]["cpu"] = [3, na]
    assert GUARD.validate(base, candidate, ["selected"], NO_DECLARATIONS)["admissible"]
    candidate["cells"]["selected/base"]["infer"]["cols"]["cpu"] = [3, na]
    assert not GUARD.validate(base, candidate, ["selected"], NO_DECLARATIONS)["admissible"]


def test_gpu_only_declarations_are_read_from_the_harness_source():
    lanes, prefixes = GUARD.gpu_only_declarations()
    assert "par-" in prefixes
    assert 'optim-resident-io' in lanes
    # Ranking now has CPU routes; its older GPU-only exemption must not return.
    assert 'gbdt-pair-logit-pointwise' not in lanes


def test_gpu_only_reader_combines_harness_and_lane_declarations(tmp_path):
    (tmp_path/'identity_break.py').write_text(
        'GPU_ONLY_LANES = {"harness-lane": "GPU operation"}\nRECORD_EXCLUDED_PREFIXES = ("cooperative-",)\n')
    (tmp_path/'identity_lanes').mkdir()
    (tmp_path/'identity_lanes/example.py').write_text('GPU_ONLY_LANES["module-lane"] = "GPU operation"\n')
    assert GUARD.gpu_only_declarations(tmp_path) == (
        frozenset({'harness-lane', 'module-lane'}), ('cooperative-',))


def test_gpu_only_declarations_absent_harness_reads_strict(tmp_path):
    assert GUARD.gpu_only_declarations(tmp_path) == (frozenset(), ())


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


def test_builder_conflict_refuses():
    base, candidate = tables()
    entry = candidate["cells"]["selected/base"]["train"]
    entry["cols"]["cpu"] = [3, OTHER]
    entry.update(ref=None, conflict=True)
    assert not GUARD.validate(base, candidate, ["selected"])["admissible"]


def test_na_parts_do_not_invent_a_cpu_requirement():
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
    for name in ("admit_identity_columns.sh", "identity_admission_guard.py", "retain_run_summary.py"):
        shutil.copyfile(ROOT / "tools" / name, repo / "tools" / name)
    table = repo / "python/mojolearn/verify_reference/table.json"
    table.parent.mkdir(parents=True)
    table.write_text(json.dumps(base))
    shutil.copyfile(ROOT / 'python/mojolearn/_verify_reference.py', table.parent.parent / '_verify_reference.py')
    for name in (".libs/libMojolearnMath.so", ".dylibs/libMojolearnMath.dylib"):
        path = table.parent.parent / name
        path.parent.mkdir(parents=True)
        path.write_bytes(b"inert; never loaded")
    incoming = tmp_path / "incoming"
    incoming.mkdir()
    raw = incoming / "apple.identical.json"
    def git(*args):
        return subprocess.check_output(['git', *args], cwd=repo, stderr=subprocess.STDOUT, text=True).strip()
    git('init', '-b', 'main')
    git('config', 'user.name', 'Admission test')
    git('config', 'user.email', 'admission@example.invalid')
    git('add', '.')
    git('commit', '-m', 'source fixture')
    source_commit = git('rev-parse', 'HEAD')
    raw.write_text(json.dumps({"vendor": "apple-inert", "commit": source_commit,
                               "cells": {"selected/base": {}, "held/base": {"verdict": "DIVERGENT"}}}))
    stub = tmp_path / "candidate.json"
    stub.write_text(json.dumps(candidate))
    shim = tmp_path / "inert_python.py"
    shim.write_text('''import json, os, pathlib, runpy, sys
a = sys.argv[1:]
if a[0].endswith("identity_host_preflight.py"):
    raise SystemExit(0)
if a[0].endswith("identity_columns.py"):
    if a[1] == "check":
        for path in a[2:]:
            print("IDCOLUMN " + path + " admissible=yes class=apple vendor=apple-inert commit=" + json.loads(pathlib.Path(path).read_text())["commit"] + " repeats=1")
    else:
        pathlib.Path(a[a.index("--out")+1]).write_text("{}")
    raise SystemExit(0)
if a[:3] == ["-m", "mojolearn", "verify"]:
    if "--emit-reference" in a:
        assert "--reference-table" in a and a[a.index("--lanes")+1] == "selected"
        records = []
        for index, arg in enumerate(a):
            if arg == "--records":
                records.append(a[index+1])
        discover = runpy.run_path("python/mojolearn/_verify_reference.py")["repository_record_paths"]
        inputs = discover(records, pathlib.Path.cwd())
        with open(os.environ["EMIT_CALLS"], "a") as log:
            log.write(json.dumps([str(p) for p in inputs]) + "\\n")
        pathlib.Path(a[a.index("--emit-reference")+1]).write_bytes(pathlib.Path(os.environ["STUB_CANDIDATE"]).read_bytes())
        print("{}")
    if "--self-test" in a and os.environ.get("FAIL_SELFTEST") == "1":
        raise SystemExit(7)
    raise SystemExit(0)
os.execv(sys.executable, [sys.executable] + a)
''')
    evidence = tmp_path / "evidence"
    env = dict(os.environ,
               PYTHON=f"{sys.executable} {shim}", STUB_CANDIDATE=str(stub),
               EMIT_CALLS=str(tmp_path/'emit-calls.jsonl'),
               EVIDENCE=str(evidence), ADMIT_DATE="2026-10-23", REF_TAG="inert",
               FAIL_SELFTEST="1" if fail_selftest else "0")
    return repo, table, raw, env


@pytest.mark.parametrize("problem", ["apple", "nvidia", "amd", "cpu", "nocpu", "missing", "selftest"])
@pytest.mark.parametrize('check_only', [False, True])
def test_wrapper_failure_cannot_mutate_shipped_table(tmp_path, problem, check_only):
    base, candidate = tables()
    if problem == "nocpu":
        del candidate["cells"]["selected/base"]["train"]["cols"]["cpu"]
    elif problem in ("apple", "nvidia", "amd", "cpu"):
        cols = candidate["cells"]["selected/base"]["train"]["cols"]
        cols[problem] = [cols[problem], OTHER]
    elif problem == "missing":
        del candidate["cells"]["selected/base"]["train"]
    repo, table, raw, env = wrapper_fixture(tmp_path, candidate, base, problem == "selftest")
    before, original = table.read_bytes(), raw.read_bytes()
    result = subprocess.run(["bash", str(repo / "tools/admit_identity_columns.sh"),
                            *(['--check-only'] if check_only else []), "--lanes", "selected", str(raw.parent)],
                            env=env, capture_output=True, text=True)
    assert result.returncode != 0, result.stdout + result.stderr
    assert table.read_bytes() == before
    assert raw.read_bytes() == original
    if check_only:
        assert not list((repo/'bench/results/identity_break').rglob('*.json'))


@pytest.mark.parametrize("check_only", [False, True])
def test_wrapper_checks_candidate_and_preserves_original_unselected_record(tmp_path, check_only):
    base, candidate = tables()
    repo, table, raw, env = wrapper_fixture(tmp_path, candidate, base)
    before, original = table.read_bytes(), raw.read_bytes()
    head = subprocess.check_output(['git', '-C', str(repo), 'rev-parse', 'HEAD'])
    result = subprocess.run(["bash", str(repo / "tools/admit_identity_columns.sh"),
                             *(["--check-only"] if check_only else []), "--lanes", "selected", str(raw.parent)],
                            env=env, capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
    assert raw.read_bytes() == original
    if check_only:
        assert table.read_bytes() == before
        assert not (repo / 'bench/results/reference_admissions').exists()
        assert not list((repo/'bench/results/identity_break').rglob('*.json'))
    else:
        assert json.loads(table.read_text()) == candidate
        retained = repo / 'bench/results/identity_break/retained' / hashlib.sha256(original).hexdigest() / raw.name
        assert retained.read_bytes() == original
        assert (repo / 'bench/results/reference_admissions/2026-10-23-inert/agreement-summary.json').exists()
    calls = Path(env['EMIT_CALLS']).read_text().splitlines()
    assert len(calls) == 1
    assert len(list((repo/'bench/results/identity_break').rglob('*.identical.json'))) == (0 if check_only else 1)
    assert all(Path(p).is_relative_to(repo/'bench/results/identity_break') for p in json.loads(calls[0]))
    assert subprocess.check_output(['git', '-C', str(repo), 'rev-parse', 'HEAD']) == head


@pytest.mark.parametrize('change', ['corrupted-copy', 'symlink'])
def test_admission_refuses_unsafe_input_without_mutating_table(tmp_path, change):
    base, candidate = tables()
    repo, table, raw, env = wrapper_fixture(tmp_path, candidate, base)
    if change == 'corrupted-copy':
        retained = repo/'bench/results/identity_break/retained'/hashlib.sha256(raw.read_bytes()).hexdigest()/raw.name
        retained.parent.mkdir(parents=True)
        retained.write_text('{}')
    else:
        target = raw.with_suffix('.original')
        raw.rename(target)
        raw.symlink_to(target)
    before = table.read_bytes()
    result = subprocess.run(['bash', str(repo/'tools/admit_identity_columns.sh'), '--lanes', 'selected', str(raw.parent)],
                            env=env, capture_output=True, text=True)
    assert result.returncode != 0
    assert 'could not retain incoming records' in result.stderr
    assert table.read_bytes() == before
    assert not Path(env['EMIT_CALLS']).exists()


def test_admission_reuses_an_uncommitted_repository_copy(tmp_path):
    base, candidate = tables()
    repo, table, raw, env = wrapper_fixture(tmp_path, candidate, base)
    retained = repo/'bench/results/identity_break/new-run'/raw.name
    retained.parent.mkdir(parents=True)
    retained.write_bytes(raw.read_bytes())
    result = subprocess.run(['bash', str(repo/'tools/admit_identity_columns.sh'), '--lanes', 'selected', str(raw.parent)],
                            env=env, capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
    assert json.loads(table.read_text()) == candidate
    assert list((repo/'bench/results/identity_break').rglob('*.identical.json')) == [retained]


def test_source_derived_host_only_cpu_is_sufficient(monkeypatch):
    monkeypatch.setattr(GUARD, "host_only_declarations", lambda: {"selected"})
    base, candidate = tables()
    for key, cell in candidate["cells"].items():
        if key.startswith("selected/"):
            for entry in cell.values():
                entry["cols"] = {"cpu": 3}
                entry["scope"] = "cpu-host-route-only"
    report = GUARD.validate(base, candidate, ["selected"], NO_DECLARATIONS)
    assert report["admissible"] and report["numerical_parts"] == 8
    # Scope metadata is descriptive, never an exemption supplied by a caller.
    monkeypatch.setattr(GUARD, "host_only_declarations", lambda: frozenset())
    assert not GUARD.validate(base, candidate, ["selected"], NO_DECLARATIONS)["admissible"]


@pytest.mark.parametrize("value", [None, [3, OTHER], True, [99, HASH]])
def test_host_only_still_requires_a_valid_agreeing_cpu(monkeypatch, value):
    monkeypatch.setattr(GUARD, "host_only_declarations", lambda: {"selected"})
    base, candidate = tables()
    candidate["cells"]["selected/base"]["train"]["cols"]["cpu"] = value
    assert not GUARD.validate(base, candidate, ["selected"], NO_DECLARATIONS)["admissible"]


def test_host_only_historical_gpu_observations_do_not_claim_gpu_execution(monkeypatch):
    monkeypatch.setattr(GUARD, "host_only_declarations", lambda: {"selected"})
    base, candidate = tables()
    candidate["cells"]["selected/base"]["train"]["cols"].update(ptx=[4, OTHER], nvidia=[1, OTHER])
    assert GUARD.validate(base, candidate, ["selected"], NO_DECLARATIONS)["admissible"]


def test_host_only_na_needs_cpu_observation_too(monkeypatch):
    monkeypatch.setattr(GUARD, "host_only_declarations", lambda: {"selected"})
    base, candidate = tables()
    entry = candidate["cells"]["selected/base"]["train"]
    entry["ref"] = "n/a:declared"
    del entry["cols"]["cpu"]
    assert not GUARD.validate(base, candidate, ["selected"], NO_DECLARATIONS)["admissible"]


@pytest.mark.parametrize('preview', [False, True])
def test_repository_inputs_keep_verifier_companions(tmp_path, preview):
    from test_retain_run_summary import verifier_column
    root=tmp_path/'repo';(root/'.git').mkdir(parents=True);(root/'tools').mkdir()
    original=tmp_path/'producer';original.mkdir()
    column,report,launch=verifier_column(original)
    preview_dir=root/'bench/results/identity_break/.admission-preview-case' if preview else None
    [retained]=GUARD.repository_inputs(root,[column],preview_dir=preview_dir)
    assert retained.with_name(report.name).read_bytes()==report.read_bytes()
    assert retained.with_name(launch.name).read_bytes()==launch.read_bytes()
    assert retained.with_name('amd.provenance.txt').is_file()
    if not preview:
        # Reusing an old exact column repairs missing companion bytes from the
        # supplied packet without rewriting the original record.
        retained.with_name(launch.name).unlink()
        assert GUARD.repository_inputs(root,[column])==[retained]
        assert retained.with_name(launch.name).read_bytes()==launch.read_bytes()


def test_retired_par_lanes_in_old_records_are_ignored_not_selected(tmp_path):
    # Records taken before multi-GPU verification was retired (0.8.63) carry
    # par-* cells, all REFUSED; admission ignores them rather than refusing.
    record = tmp_path / "old.identical.json"
    record.write_text(json.dumps({"cells": {"kmeans/base": {}, "par-kmeans/base": {"verdict": "REFUSED"},
                                            "par-forest/ties": {"verdict": "REFUSED"}}}))
    assert GUARD.selected_lanes([record]) == ["kmeans"]
    assert GUARD.selected_lanes([record], "kmeans,par-kmeans") == ["kmeans"]
    with pytest.raises(ValueError):
        GUARD.selected_lanes([record], "par-kmeans")


def test_guard_and_reference_agree_on_the_retired_prefixes():
    import ast
    tree = ast.parse((ROOT / "python/mojolearn/_verify_reference.py").read_text())
    value = next(ast.literal_eval(node.value) for node in tree.body if isinstance(node, ast.Assign)
                 and any(getattr(t, "id", None) == "RETIRED_LANE_PREFIXES" for t in node.targets))
    assert value == GUARD.RETIRED_LANE_PREFIXES == ("par-",)
