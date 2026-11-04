#!/usr/bin/env python3
"""Identity ledger rows judged by the reference admission, without builds, GPUs or the real selector.

A throwaway Git repository holds a frozen tree with a stand-in tools/lane_select.py (lane -> files,
version-literal-only edits inert, as the real selector's ``release_label_only``), an admitted
reference table and its column records. The real frozen-view worker runs that selector.
"""
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import pytest

import release_identity_evidence as E
import release_identity_reference as IR
import release_requirement_dependencies as D
import release_requirements as R

FAKE_SELECTOR = '''
import importlib.util, re, subprocess
from pathlib import Path
ROOT = str(Path(__file__).resolve().parent.parent)
VERSION = "python/mojolearn/_version.py"
LANE_FILES = {"ols": ["core/ols.mojo", "core/shared.mojo", "core/a1.mojo", "core/a2.mojo", "core/a3.mojo",
                      "core/a4.mojo", "core/a5.mojo", VERSION],
              "ridge": ["core/ridge.mojo", "core/shared.mojo"]}

def identity_break():
    spec = importlib.util.spec_from_file_location("fake_harness", ROOT + "/tools/identity_break.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

def lane_sources():
    return {lane: set(files) for lane, files in LANE_FILES.items()}, {}

def _python_closure(paths):
    return []

def changed_paths(ref):
    out = subprocess.check_output(["git", "-C", ROOT, "diff", "--name-only", ref, "HEAD"], text=True)
    return sorted(line for line in out.splitlines() if line)

def _label_view(text):
    return re.sub(r'__version__ = "[0-9.]+"', '__version__ = "<release-label>"', text)

def select(paths, ref=None, sources=None, backend=None, stop_when=None):
    lanes, by_path, unattributed = set(), {}, []
    for path in paths:
        if path == VERSION:
            old = subprocess.check_output(["git", "-C", ROOT, "show", ref + ":" + path], text=True)
            if _label_view(old) == _label_view(Path(ROOT, path).read_text()):
                continue
        if path.startswith("unknown/"):
            unattributed.append(path)
            continue
        hit = {lane for lane, files in (sources or lane_sources()[0]).items() if path in files}
        if hit:
            by_path[path] = sorted(hit)
            lanes |= hit
    return dict(lanes=sorted(lanes), by_path=by_path, unattributed=unattributed, stopped_early=False)
'''
HARNESS = 'FIXTURES = ["base"]\nLANES = {"ols": None, "ridge": None}\nLANE_REVISIONS = {}\n'
CLASSES = ("apple", "nvidia", "amd", "cpu")
PARTS = ("train", "infer", "model", "batch")


def h16(*parts):
    return hashlib.sha256("/".join(parts).encode()).hexdigest()[:16]


class Repo:
    def __init__(self, root):
        self.root = root
        root.mkdir()
        self.git("init", "-q")
        self.git("config", "user.email", "test@example.test")
        self.git("config", "user.name", "Reference Test")
        self.write("tools/release_reuse.py", "from pathlib import Path\nROOT=Path(__file__).resolve().parent.parent\n")
        self.write("tools/bincache.py", "from pathlib import Path\nROOT=Path(__file__).resolve().parent.parent\n")
        self.write("tools/lane_select.py", FAKE_SELECTOR)
        self.write("tools/identity_break.py", HARNESS)
        self.write("tools/release.py", "# release tooling\n")
        for name in ("ols", "shared", "a1", "a2", "a3", "a4", "a5", "ridge"):
            self.write(f"core/{name}.mojo", f"fn {name}(): pass\n")
        self.write("python/mojolearn/_version.py", '__version__ = "0.8.65"\n')

    def git(self, *args):
        return subprocess.check_output(["git", "-C", str(self.root), *args], text=True,
                                       stderr=subprocess.DEVNULL).strip()

    def write(self, rel, text):
        path = self.root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)

    def commit(self, message):
        self.git("add", "-A")
        self.git("commit", "-q", "-m", message)
        return self.git("rev-parse", "HEAD")

    def admit(self, record_commit, *, conflict=False, other_harness=False, legacy=None, legacy_ref="n/a:legacy",
              salt=""):
        """Column records at ``record_commit`` and the table that admits them, committed together."""
        records, cells = [], {}
        for index, cls in enumerate(CLASSES):
            body = dict(commit=record_commit, cells={
                f"{lane}/base": dict(verdict="STABLE", hashes=[h16(lane + salt, "train")],
                                     **{p: [h16(lane + salt, p)] for p in PARTS[1:]},
                                     **{p + "_verdict": "STABLE" for p in PARTS[1:]})
                for lane in ("ols", "ridge")})
            raw = json.dumps(body, sort_keys=True)
            sha = hashlib.sha256(raw.encode()).hexdigest()
            self.write(f"bench/results/identity_break/col/{cls}.identical.json", raw)
            records.append(dict(**{"class": cls}, commit=record_commit, dir="col", file=f"{cls}.identical.json",
                                sha256=sha, vendor=cls))
        for lane in ("ols", "ridge"):
            cells[f"{lane}/base"] = {p: dict(ref=h16(lane + salt, p), cols={c: i for i, c in enumerate(CLASSES)})
                                     for p in PARTS}
        if conflict:
            cells["ols/base"]["infer"]["conflict"] = True
        if legacy:
            # An optional part only an older Apple record carries (0.8.65: the legacy rlpair n/a rows).
            body = dict(commit=legacy, cells={"ols/base": dict(rlpair=[legacy_ref], rlpair_verdict=(
                "N/A" if legacy_ref.startswith("n/a:") else "STABLE"))})
            raw = json.dumps(body, sort_keys=True)
            self.write("bench/results/identity_break/old/apple.identical.json", raw)
            records.append(dict(**{"class": "apple"}, commit=legacy, dir="old", file="apple.identical.json",
                                sha256=hashlib.sha256(raw.encode()).hexdigest(), vendor="apple"))
            cells["ols/base"]["rlpair"] = dict(ref=legacy_ref, cols={"apple": len(records) - 1})
        harness = hashlib.sha256((self.root / "tools/identity_break.py").read_bytes()).hexdigest()
        table = dict(cells=cells, records=records, harness_sha256=harness, lane_revisions={},
                     lane_admission={lane: dict(harness_sha256="0" * 64 if other_harness else harness)
                                     for lane in ("ols", "ridge")})
        self.write(IR.TABLE_PATH, json.dumps(table, sort_keys=True))
        self.write("bench/results/reference_admissions/2026-11-04-test/README.md",
                   "".join(f"`bench/results/identity_break/retained/{r['sha256']}/{r['file']}`\n" for r in records))
        return self.commit("admit reference columns")


def wanted(vendor, lane, revision, native="n"):
    coverage = [E.BASE_COVERAGE, "lane:" + lane]
    inputs = dict(unit=dict(kind="identity", vendor=vendor, lanes=[lane], release_modes=["identical"]),
                  native_sources={"core/ols.mojo": native, "python/mojolearn/_version.py": native},
                  execution_environment={lane: dict(target="linux-64", bindings={})},
                  contract=dict(id=f"identity-{vendor}/{lane}", kind="identity", stage="pre", scope="s",
                                coverage=coverage))
    return dict(source_revision=revision, inputs=inputs, fingerprint=R.digest(inputs), scope="s",
                coverage=coverage, kind="identity", stage="pre", missing_inputs=[], required=True,
                dependency_provenance=dict(identity_applicability=dict(required_columns=list(CLASSES))))


def manifest(repo, revision, lanes=("ols", "ridge"), native="n"):
    rows = {f"identity-{v}/{lane}": wanted(v, lane, revision, native) for v in CLASSES for lane in lanes}
    return dict(schema=R.SCHEMA, version="0.8.66", source_revision=revision, requirements=rows,
                source_root=str(repo.root), configuration={})


def plan(repo, tmp_path, revision, *, attempts=(), explain=False, **kwargs):
    doc = dict(manifest(repo, revision, **kwargs), attempts=list(attempts), runs=[])
    return R.RequirementsLedger.evaluate(doc, root=str(repo.root), log_dir=tmp_path / "deps", explain=explain)


@pytest.fixture
def repo(tmp_path, monkeypatch):
    IR._MEMO.clear()
    D._REVISIONS.clear()
    # Never read the real release records (~/mojolearn-evidence/release) from a test.
    monkeypatch.setenv(IR.RELEASE_RECORDS_ENV, str(tmp_path / "release-records"))
    made = Repo(tmp_path / "repo")
    made.record = made.commit("frozen sources")
    return made


def test_unchanged_closure_passes_with_the_admission_as_evidence(repo, tmp_path):
    freeze = repo.admit(repo.record)
    rows = plan(repo, tmp_path, freeze)
    assert {row["status"] for row in rows.values()} == {"passed"}, rows
    row = rows["identity-nvidia/ols"]
    assert row["action"] == "reuse" and row["tested_revision"] == repo.record
    assert row["reference"]["decision"] == "passed" and row["reference"]["release_floor"] is None
    paths = {item["path"] for item in row["evidence"]}
    assert IR.TABLE_PATH in paths
    assert "bench/results/identity_break/col/nvidia.identical.json" in paths
    assert "bench/results/reference_admissions/2026-11-04-test/README.md" in paths
    assert all(item["revision"] == freeze for item in row["evidence"])
    assert {item["cls"] for item in row["reference"]["basis"]} == {"nvidia"}
    # No new run is owed for a passed row.
    ledger = R.RequirementsLedger(tmp_path / "ledger.json", source_root=str(repo.root),
                                  evidence_log_dir=tmp_path / "deps")
    ledger.sync(manifest(repo, freeze))
    assert ledger.begin("identity-nvidia/ols") is None
    assert ledger.gate(stage="pre")[0]


def test_changed_closure_is_untested_with_the_first_five_files(repo, tmp_path):
    repo.admit(repo.record)
    for name in ("ols", "shared", "a1", "a2", "a3", "a4", "a5"):
        repo.write(f"core/{name}.mojo", f"fn {name}(): return 1\n")
    freeze = repo.commit("edit the ols closure")
    rows = plan(repo, tmp_path, freeze)
    row = rows["identity-amd/ols"]
    assert row["status"] == "untested" and row["action"] == "run"
    reason = row["reasons"][0]
    assert reason.startswith(f"closure changed since {repo.record[:12]} (amd hip): 7 file(s): ")
    assert reason.endswith("core/a1.mojo, core/a2.mojo, core/a3.mojo, core/a4.mojo, core/a5.mojo")
    # core/shared.mojo also reaches ridge; a file only ols reads leaves ridge alone.
    assert rows["identity-amd/ridge"]["status"] == "untested"


def test_change_in_one_lane_leaves_the_other_passed(repo, tmp_path):
    repo.admit(repo.record)
    repo.write("core/ridge.mojo", "fn ridge(): return 2\n")
    freeze = repo.commit("edit ridge only")
    rows = plan(repo, tmp_path, freeze)
    assert all(rows[f"identity-{v}/ols"]["status"] == "passed" for v in CLASSES)
    assert all(rows[f"identity-{v}/ridge"]["status"] == "untested" for v in CLASSES)
    assert "1 file(s): core/ridge.mojo" in rows["identity-cpu/ridge"]["reasons"][0]


def test_version_only_diff_passes(repo, tmp_path):
    repo.admit(repo.record)
    repo.write("python/mojolearn/_version.py", '__version__ = "0.8.66"\n')
    freeze = repo.commit("Release 0.8.66")
    # The row's own inputs hash _version.py, so its fingerprint moved; the closure did not.
    rows = plan(repo, tmp_path, freeze, native="bumped")
    assert {row["status"] for row in rows.values()} == {"passed"}, rows
    repo.write("python/mojolearn/_version.py", '__version__ = "0.8.66"\nDEBUG = True\n')
    later = repo.commit("not a label-only change")
    IR._MEMO.clear()
    rows = plan(repo, tmp_path, later)
    assert rows["identity-apple/ols"]["status"] == "untested"
    assert "python/mojolearn/_version.py" in rows["identity-apple/ols"]["reasons"][0]
    assert rows["identity-apple/ridge"]["status"] == "passed"


@pytest.mark.parametrize("legacy_ref, status", [("n/a:legacy", "passed"), ("0123456789abcdef", "untested")])
def test_declared_na_optional_part_is_not_a_closure_basis(repo, tmp_path, legacy_ref, status):
    repo.write("core/ols.mojo", "fn ols(): return 4\n")
    recorded = repo.commit("edit after the legacy record")
    freeze = repo.admit(recorded, legacy=repo.record, legacy_ref=legacy_ref)
    row = plan(repo, tmp_path, freeze)["identity-apple/ols"]
    assert row["status"] == status, row["reasons"]
    if status == "untested":
        assert row["reasons"][0].startswith(f"closure changed since {repo.record[:12]}")


def test_unattributed_path_never_passes(repo, tmp_path):
    repo.admit(repo.record)
    repo.write("unknown/thing.mojo", "fn x(): pass\n")
    freeze = repo.commit("a path nothing places")
    row = plan(repo, tmp_path, freeze)["identity-nvidia/ridge"]
    assert row["status"] == "untested" and "unattributed path(s) 1 file(s): unknown/thing.mojo" in row["reasons"][0]


def test_table_conflict_or_other_harness_is_refused_not_relaxed(repo, tmp_path):
    freeze = repo.admit(repo.record, conflict=True)
    rows = plan(repo, tmp_path, freeze)
    assert rows["identity-apple/ols"]["status"] == "untested"
    assert "conflict" in rows["identity-apple/ols"]["reasons"][0]
    assert rows["identity-apple/ridge"]["status"] == "passed"
    IR._MEMO.clear()
    freeze = repo.admit(repo.record, other_harness=True)
    rows = plan(repo, tmp_path, freeze)
    assert all(row["status"] == "untested" and "another harness" in row["reasons"][0] for row in rows.values())


def test_failed_attempt_on_an_unchanged_closure_is_repaired(repo, tmp_path):
    freeze = repo.admit(repo.record)
    failed = dict(id="f1", requirement="identity-nvidia/ols", run_id="r1", status="failed",
                  fingerprint="0" * 64, source_revision=repo.record, summary="digest differs")
    rows = plan(repo, tmp_path, freeze, attempts=[failed])
    assert rows["identity-nvidia/ols"]["status"] == "untested"
    assert rows["identity-nvidia/ols"]["action"] == "repair"
    assert "contradicts the admitted reference" in rows["identity-nvidia/ols"]["reasons"][0]
    assert rows["identity-amd/ols"]["status"] == "passed"


def tag_release(repo, commit, version="0.8.64"):
    for vendor in ("nvidia", "amd"):
        repo.git("tag", "-a", f"alpha-api-{version}-{vendor}-20261102", "-m", "release", commit)


def test_a_release_tag_alone_is_not_a_floor(repo, tmp_path):
    # ols changed after its record and before the release; the table admitting the old record was
    # committed only after the tag, so the release holds no receipt for it (0.8.64/0.8.65 shipped
    # under owner waivers: a published release is not verification).
    repo.write("core/ols.mojo", "fn ols(): return 3\n")
    tagged = repo.commit("changed before the last release")
    tag_release(repo, tagged)
    freeze = repo.admit(repo.record)
    row = plan(repo, tmp_path, freeze)["identity-apple/ols"]
    assert row["status"] == "untested"
    assert row["reference"]["release_floor"]["commit"] == tagged
    item = row["reference"]["basis"][0]
    assert item["basis"] == "record" and item["base"] == repo.record
    assert item["floor_refused"].startswith(f"no admitted receipt at the floor: no admitted reference table at {tagged[:12]}")
    assert row["reasons"][0].startswith(f"closure changed since {repo.record[:12]} (apple metal): 1 file(s): core/ols.mojo")


def test_floor_without_receipt_falls_back_to_the_record_revision(repo, tmp_path):
    # The table at the floor admits the record, but the lane's closure moved between the record and
    # the floor: the floor's own admission owed a record, so it is no receipt.
    repo.admit(repo.record)
    repo.write("core/ols.mojo", "fn ols(): return 3\n")
    tagged = repo.commit("changed before the last release")
    tag_release(repo, tagged)
    repo.write("docs/notes.md", "after the release\n")
    freeze = repo.commit("after the release")
    rows = plan(repo, tmp_path, freeze)
    row = rows["identity-nvidia/ols"]
    assert row["status"] == "untested"
    item = row["reference"]["basis"][0]
    assert item["basis"] == "record" and item["base"] == repo.record
    assert f"closure changed between its record {repo.record[:12]} and the floor {tagged[:12]}" in item["floor_refused"]
    assert "core/ols.mojo" in row["reasons"][0]
    # ridge's closure never moved: it holds a receipt and is measured from the floor.
    ridge = rows["identity-nvidia/ridge"]
    assert ridge["status"] == "passed" and ridge["reference"]["basis"][0]["basis"] == "release-floor"


def receipted_floor(repo):
    """Admitted at A (records at R); ridge changes; release tagged; a docs commit; the freeze."""
    repo.admit(repo.record)
    repo.write("core/ridge.mojo", "fn ridge(): return 2\n")
    tagged = repo.commit("ridge changed before the last release")
    tag_release(repo, tagged)
    return tagged


def test_floor_with_receipt_passes_from_the_floor(repo, tmp_path):
    tagged = receipted_floor(repo)
    repo.write("docs/notes.md", "after the release\n")
    freeze = repo.commit("after the release")
    rows = plan(repo, tmp_path, freeze)
    row = rows["identity-amd/ols"]
    assert row["status"] == "passed", row["reasons"]
    item = row["reference"]["basis"][0]
    assert item["basis"] == "release-floor" and item["base"] == tagged and item["commit"] == repo.record
    assert IR.TABLE_PATH in item["floor_receipt"]
    # The receipt's evidence (the floor's table and its admitted record) is cited at the floor.
    floor_evidence = {(e["path"], e["revision"]) for e in row["evidence"] if e["revision"] == tagged}
    assert (IR.TABLE_PATH, tagged) in floor_evidence
    assert ("bench/results/identity_break/col/amd.identical.json", tagged) in floor_evidence
    # ridge changed between its record and the floor: no receipt, measured from its record.
    ridge = rows["identity-amd/ridge"]
    assert ridge["status"] == "untested" and ridge["reference"]["basis"][0]["basis"] == "record"


def test_floor_receipt_needs_the_same_digests(repo, tmp_path):
    tagged = receipted_floor(repo)
    freeze = repo.admit(repo.record, salt="-readmitted")
    row = plan(repo, tmp_path, freeze)["identity-cpu/ols"]
    item = row["reference"]["basis"][0]
    assert item["basis"] == "record"
    assert "the floor's admitted table carries" in item["floor_refused"]
    assert tagged[:12] in item["floor_refused"]


@pytest.mark.parametrize("source", ["committed", "release-record", "reuse-waiver"])
def test_waived_floor_is_never_a_floor(repo, tmp_path, monkeypatch, source):
    receipted_floor(repo)
    if source == "committed":
        repo.write("docs/release/0.8.64-owner-waiver.json", json.dumps(dict(
            approved_by="owner", reason="one-time", prefixes=["identity-apple/ols", "adapters-"])))
        # Another release's waiver does not reach this floor.
        repo.write("docs/release/0.8.63-owner-waiver.json", json.dumps(dict(
            approved_by="owner", reason="older", prefixes=["identity-"])))
    elif source == "release-record":
        state = tmp_path / "release-records" / "0.8.64" / "state.json"
        state.parent.mkdir(parents=True)
        state.write_text(json.dumps(dict(owner_waivers=[dict(
            approved_by="owner", reason="one-time", file=str(tmp_path / "gone.json"), waived=["identity-apple/ols"])])))
    else:
        raw = (repo.root / "bench/results/identity_break/col/apple.identical.json").read_bytes()
        repo.write("docs/release/0.8.64-identity-reuse-waiver.json", json.dumps(dict(
            approved_by="owner", reason="reuse", records_sha256=[hashlib.sha256(raw).hexdigest()])))
    repo.write("docs/notes.md", "after the release\n")
    freeze = repo.commit("after the release")
    rows = plan(repo, tmp_path, freeze)
    item = rows["identity-apple/ols"]["reference"]["basis"][0]
    assert item["basis"] == "record" and item["base"] == repo.record
    want = "identity-reuse waiver" if source == "reuse-waiver" else "rests on a waived row: identity-apple/ols"
    assert want in item["floor_refused"]
    # The row itself is still judged, from its record: its closure did not change, so it passes.
    assert rows["identity-apple/ols"]["status"] == "passed"
    # Rows the waiver does not name keep the floor.
    assert rows["identity-nvidia/ols"]["reference"]["basis"][0]["basis"] == "release-floor"


def test_floor_waivers_read_both_sources(repo, tmp_path):
    repo.write("docs/release/0.8.64-owner-waiver.json", json.dumps(dict(prefixes=["identity-cpu/"])))
    revision = repo.commit("waiver")
    state = tmp_path / "records" / "0.8.64" / "state.json"
    state.parent.mkdir(parents=True)
    named = tmp_path / "w.json"
    named.write_text(json.dumps(dict(prefixes=["identity-amd/ridge"])))
    state.write_text(json.dumps(dict(owner_waivers=[dict(file=str(named), waived=["identity-apple/ols"])])))
    floor = dict(commit=revision, tags=dict(nvidia="alpha-api-0.8.64-nvidia-x", amd="alpha-api-0.8.64-amd-x"))
    got = IR.floor_waivers(repo.root, revision, floor, records_dir=tmp_path / "records")
    assert IR.waived(got, "identity-cpu/ols") == "identity-cpu/"
    assert IR.waived(got, "identity-apple/ols") == "identity-apple/ols"
    assert IR.waived(got, "identity-amd/ridge") == "identity-amd/ridge"
    assert IR.waived(got, "identity-nvidia/ols") is None
    assert IR.floor_versions(floor) == ["0.8.64"]
    assert IR.floor_waivers(repo.root, revision, None)["sources"] == []


def test_a_release_tag_at_the_freeze_itself_never_vouches_for_it(repo, tmp_path):
    repo.write("core/ols.mojo", "fn ols(): return 3\n")
    freeze = repo.admit(repo.record)
    for vendor in ("nvidia", "amd"):
        repo.git("tag", "-a", f"alpha-api-0.8.66-{vendor}-20261105", "-m", "release", freeze)
    row = plan(repo, tmp_path, freeze)["identity-apple/ols"]
    assert row["status"] == "untested" and row["reference"]["release_floor"] is None


def test_no_admitted_table_is_unrecorded_not_a_broken_analysis(repo, tmp_path):
    rows = plan(repo, tmp_path, repo.record)
    assert rows["identity-apple/ols"]["action"] == "run"
    assert rows["identity-apple/ols"]["reasons"] == ["no qualification evidence recorded for this requirement"]


def test_explain_runs_the_selector_only_as_a_diagnostic(repo, tmp_path, monkeypatch):
    freeze = repo.admit(repo.record)
    calls = []

    def selector(doc, name, old, wanted, row, **kwargs):
        calls.append(name)
        row.update(dependency_decision="blocked", reasons=["dependency equivalence blocked: selector provenance unresolved"])
    monkeypatch.setattr(R.RequirementsLedger, "_equivalence", staticmethod(selector))
    monkeypatch.setattr(R.RequirementsLedger, "_attempt_requirement", staticmethod(lambda attempt: {}))
    older = dict(id="p1", requirement="identity-nvidia/ols", run_id="r0", status="passed",
                 fingerprint="1" * 64, source_revision=repo.record, summary="ok")
    rows = plan(repo, tmp_path, freeze, attempts=[older])
    assert rows["identity-nvidia/ols"]["status"] == "passed" and not calls
    rows = plan(repo, tmp_path, freeze, attempts=[older], explain=True)
    row = rows["identity-nvidia/ols"]
    assert calls == ["identity-nvidia/ols"]
    assert row["status"] == "passed"
    assert row["selector_diagnostic"]["dependency_decision"] == "blocked"


def test_unit_shape_and_classes():
    row = wanted("nvidia", "ols", "f" * 40)
    assert IR.unit_of("identity-nvidia/ols", row) == ("ols", "nvidia")
    assert IR.unit_of("identity-amd/ols", row) is None
    assert IR.unit_of("identity", row) is None
    row["inputs"]["execution_environment"]["ols"]["bindings"] = {"linux-64/cuda/sm_80/identical/x": {}}
    assert IR.unit_classes(row, "nvidia") == (["nvidia", "ptx"], ["nvidia", "ptx", "apple", "amd"])
    assert IR.unit_classes(wanted("cpu", "ols", "f" * 40), "cpu") == (["cpu"], ["cpu", "apple", "nvidia", "amd"])


def test_class_backends_match_the_recorder():
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import identity_columns
    assert {k: v for k, v in IR.CLASS_BACKEND.items() if v} == identity_columns.CLASS_BACKEND
    assert IR.OPTIONAL_PARTS == identity_columns._OPTIONAL_PARTS


# ------------------------------------------------------------ tooling after a freeze
def tracked_repo(tmp_path):
    repo = Repo(tmp_path / "tracked")
    return repo, repo.commit("frozen")


def test_tooling_only_diff_is_not_refused_and_the_view_keeps_frozen_bytes(tmp_path):
    repo, frozen = tracked_repo(tmp_path)
    (repo.root / "tools/release.py").write_text("# a later tooling fix\n")
    (repo.root / "docs").mkdir()
    state = D._tracked_state(repo.root, frozen, tooling_from_revision=True)
    assert D.tooling_overlay(state) == ["tools/release.py"]
    with pytest.raises(ValueError, match="differs from requested revision: tools/release.py"):
        D._tracked_state(repo.root, frozen)
    repo.git("add", "tools/release.py")
    assert D.tooling_overlay(D._tracked_state(repo.root, frozen, tooling_from_revision=True)) == ["tools/release.py"]
    got = D.frozen_snapshot(repo.root, frozen, dict(extra_paths=["tools/release.py"]), log_dir=tmp_path / "logs")
    assert got["extra_sources"]["tools/release.py"] == hashlib.sha256(b"# release tooling\n").hexdigest()
    assert (repo.root / "tools/release.py").read_text() == "# a later tooling fix\n"


def test_bit_determining_diff_is_still_refused(tmp_path):
    repo, frozen = tracked_repo(tmp_path)
    for rel in ("core/ols.mojo", "python/mojolearn/_version.py", "tools/identity_break.py"):
        original = (repo.root / rel).read_text()
        (repo.root / rel).write_text(original + "# edited\n")
        with pytest.raises(ValueError, match="differs from requested revision: " + rel):
            D._tracked_state(repo.root, frozen, tooling_from_revision=True)
        (repo.root / rel).write_text(original)
    (repo.root / "core/ols.mojo").write_text("staged\n")
    repo.git("add", "core/ols.mojo")
    with pytest.raises(ValueError, match="staged changes"):
        D._tracked_state(repo.root, frozen, tooling_from_revision=True)


def test_tooling_scope():
    assert D.tooling_only("tools/release.py") and D.tooling_only("docs/release/IDENTITY_CLOSURES.md")
    assert D.tooling_only("CHANGELOG.md")
    for path in ("tools/identity_break.py", "tools/identity_lanes/core.py", "core/ols.mojo",
                 "bindings/_mojolearn.mojo", "python/mojolearn/host_surface.py", "packaging/linux/build_sets.sh",
                 "bench/results/identity_break/col/a.json", "pixi.toml", "pyproject.toml"):
        assert not D.tooling_only(path), path


@pytest.mark.skipif(sys.platform != "darwin", reason="macOS clonefile source-view path")
def test_source_view_records_tooling_commit_and_overlay(tmp_path):
    repo, frozen = tracked_repo(tmp_path)
    (repo.root / "tools/release.py").write_text("# a later tooling fix\n")
    D.frozen_snapshot(repo.root, frozen, dict(extra_paths=[]), log_dir=tmp_path / "logs")
    report = json.loads(next((tmp_path / "logs").glob("*.json")).read_text())
    assert report["source_view"]["tooling_from_revision"] == ["tools/release.py"]
    assert "commit" in report["source_view"]["tooling"]
