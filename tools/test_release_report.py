# SPDX-License-Identifier: Apache-2.0
"""tools/release_report.py (the release status report) and its tools/release.py hooks (--report, the totals
line --plan embeds). Fake ledgers for every state and the ready rule; the 0.8.65 fixtures
(tools/testdata/blocked-rows-0865.tsv, owner-waiver-0865.json) replayed (a) through the waiver and (b) through
the reference-admission judgement in a throwaway Git repository. Nothing here builds, rents or reaches a box."""
import hashlib
import json
from pathlib import Path
import subprocess

import pytest

import release_identity_evidence as E
import release_identity_reference as IR
import release_plan as rp
import release_report as rr
import release_requirement_dependencies as D
import release_requirements as R

TESTDATA = Path(__file__).resolve().parent / "testdata"
TSV = TESTDATA / "blocked-rows-0865.tsv"
WAIVER = TESTDATA / "owner-waiver-0865.json"
C = "c" * 40
OLD = "a" * 40


def row(status="untested", action="run", **kw):
    out = dict(status=status, action=action, required=True, stage="pre", reasons=[], evidence=[], source_revision=C)
    out.update(kw)
    return out


def fake_ledger():
    """One row per state (before waivers) plus one the waiver names."""
    return {
        "compile-apple": row("passed", "reuse", tested_revision=C, reasons=["same declared inputs and verified retained evidence"],
                             evidence=[dict(path="/r/compile-receipts/x.json", sha256="0")]),
        "identity-amd/ols": row("passed", "reuse", tested_revision=OLD, dependency_decision="reference-closure-unchanged",
                                reasons=["admitted reference at cccccccccccc; closure unchanged since aaaaaaaaaaaa"],
                                evidence=[dict(path=IR.TABLE_PATH, revision=C, sha256="1")]),
        "runtime-amd": row("failed", "repair", reasons=["previous attempt failed: timed out at 900 s"]),
        "identity-nvidia/ridge": row(reasons=["closure changed since aaaaaaaaaaaa (ptx cuda): 1 file(s): core/ridge.mojo"],
                                     dependency_decision="reference-closure-changed"),
        "adapters-nvidia": row(action="complete-provenance", reasons=["adapter evidence belongs to another vendor"]),
        "performance-amd": row(required=False, reasons=["no qualification evidence recorded for this requirement"]),
        "coverage-source": row(reasons=["no qualification evidence recorded for this requirement"]),
    }


def waiver_file(tmp_path, prefixes=("coverage-source",), names=()):
    path = tmp_path / "waiver.json"
    path.write_text(json.dumps(dict(prefixes=list(prefixes), waived=list(names), approved_by="Owner", reason="backlog")))
    return path


def by_id(report):
    return {r["id"]: r for r in report["rows"]}


# ---------------------------------------------------------------- every state
def test_every_state_from_a_fake_ledger(tmp_path):
    report = rr.build("0.8.66", C, fake_ledger(), waivers=rr.waivers_from([waiver_file(tmp_path)]))
    rows = by_id(report)
    assert {k: v["state"] for k, v in rows.items()} == {
        "compile-apple": "passed", "identity-amd/ols": "reused", "runtime-amd": "failed",
        "identity-nvidia/ridge": "missing-execution", "adapters-nvidia": "blocked-provenance",
        "performance-amd": "not-applicable", "coverage-source": "waived"}
    assert set(r["state"] for r in report["rows"]) == set(rr.STATES)
    assert report["totals_line"] == ("REPORT 0.8.66 passed=1 reused=1 failed=1 missing=1 blocked=1 n/a=1 waived=1  "
                                     "ready=NO")
    for r in report["rows"]:
        if r["state"] in ("passed", "reused", "not-applicable"):
            assert r["next_action"] is None
        else:
            assert r["next_action"], r
    assert rows["compile-apple"]["evidence"] == ["/r/compile-receipts/x.json"]
    assert rows["identity-amd/ols"]["evidence"] == [IR.TABLE_PATH + "@" + C[:12]]


def test_a_waiver_stays_visible_and_never_becomes_passed(tmp_path):
    waivers = rr.waivers_from([waiver_file(tmp_path, prefixes=("coverage-source", "compile-", "runtime-"))])
    rows = by_id(rr.build("0.8.66", C, fake_ledger(), waivers=waivers))
    w = rows["coverage-source"]
    assert w["state"] == "waived" and w["underlying_state"] == "missing-execution"
    assert w["waiver"]["approved_by"] == "Owner" and w["waiver"]["reason"] == "backlog"
    assert w["waiver"]["rule"] == "coverage-source"
    # a failed row named by the waiver is waived, its failure kept as the underlying state
    assert rows["runtime-amd"]["state"] == "waived" and rows["runtime-amd"]["underlying_state"] == "failed"
    # a row that passed on evidence stays passed; the waiver is recorded as unused, never as the reason
    assert rows["compile-apple"]["state"] == "passed" and rows["compile-apple"]["waiver_unused"]["rule"] == "compile-"
    md = rr.markdown(rr.build("0.8.66", C, fake_ledger(), waivers=waivers))
    assert "| coverage-source | waived (missing-execution) |" in md and "WAIVED by Owner (coverage-source): backlog" in md


def test_ready_needs_zero_failed_missing_and_blocked(tmp_path):
    ok = {k: v for k, v in fake_ledger().items() if k in ("compile-apple", "identity-amd/ols", "performance-amd",
                                                          "coverage-source")}
    report = rr.build("0.8.66", C, ok, waivers=rr.waivers_from([waiver_file(tmp_path)]))
    assert report["ready"] and report["totals_line"].endswith("waived=1  ready=YES")
    for bad in ("runtime-amd", "identity-nvidia/ridge", "adapters-nvidia"):
        rows = dict(ok, **{bad: fake_ledger()[bad]})
        assert not rr.build("0.8.66", C, rows)["ready"], bad
    # an owed release step blocks readiness too; a skipped one is not applicable
    steps = {"gate-amd": ("owed", ""), "gpu-column-amd": ("done", "2026: SKIPPED: no GPU smoke")}
    report = rr.build("0.8.66", C, ok, steps, waivers=rr.waivers_from([waiver_file(tmp_path)]))
    assert by_id(report)["gpu-column-amd"]["state"] == "not-applicable" and not report["ready"]
    assert by_id(report)["gate-amd"]["next_action"] == "tools/release.py 0.8.66 --only gate-amd"


def test_step_states_and_their_next_actions():
    steps = {"gate-nvidia": ("done", "x: PASS"), "reuse-plan": ("done", "x: against 0.8.64 pypi: 74 REUSE, 150 BUILD"),
             "linux-assemble": ("failed", "x: refused"), "publish-amd": ("held", "x: needs --publish", True),
             "published-amd": ("owed", "")}
    rows = by_id(rr.build("0.8.66", C, {}, steps))
    assert rows["gate-nvidia"]["state"] == "passed" and rows["reuse-plan"]["state"] == "reused"
    assert rows["linux-assemble"]["state"] == "failed"
    assert rows["linux-assemble"]["next_action"].startswith("tools/release.py 0.8.66 --only linux-assemble")
    assert rows["publish-amd"]["next_action"] == "tools/release.py 0.8.66 --only publish-amd --publish pypi"
    assert rows["published-amd"]["stage"] == "post" and rows["gate-nvidia"]["stage"] == "pre"
    assert rr.build("0.8.66", C, {}, steps, stage="post")["totals"]["missing-execution"] == 2


def test_next_action_catalogue():
    na = lambda name, r, **kw: rr.next_action("0.8.66", C, name, rr.ledger_state(name, r, C)[0], r, **kw)
    assert na("identity-nvidia/ridge", fake_ledger()["identity-nvidia/ridge"]) == "tools/reference_refresh.sh cccccccccccc ptx"
    assert na("identity-amd/x", row(reasons=["closure changed since a (amd hip): 1 file(s): a"])) == \
        "tools/reference_refresh.sh cccccccccccc amd"
    assert na("identity-cpu/x", row("failed", "repair")).startswith("fix the code")
    assert "--report" in na("identity-apple/x", row(action="complete-provenance", dependency_decision="blocked",
                                                     reasons=["dependency equivalence blocked: selector provenance unresolved"]))
    assert na("runtime-nvidia", row()) == "tools/release.py 0.8.66 --only gate-nvidia"
    assert na("adapters-apple", row()) == "tools/release.py 0.8.66 --only gate-apple"
    assert na("compile-amd", row()) == "tools/release.py 0.8.66 --only linux-builds"
    assert na("compile-apple", row()) == "tools/release.py 0.8.66 --only macos-build"
    assert na("published-nvidia", row(stage="post")) == "tools/release.py 0.8.66 --only published-nvidia"
    assert na("runtime-cpu", row()).startswith("add a producer for runtime-cpu")
    assert "--adapter coverage-source" in na("coverage-source", row())
    assert na("runtime-amd", row("failed", "repair")) == "fix the failure it recorded, then tools/release.py 0.8.66 --only gate-amd"
    # evidence retained but not here: restore it, by R2 tag when one is named, else from the plan's restore source
    tagged = row(action="restore-evidence", reasons=["evidence unavailable: /gone/report.json"],
                 evidence=[dict(path="/gone/report.json", sha256="0", tag="rel0866-gate-amd")])
    assert rr.ledger_state("runtime-amd", tagged)[0] == "blocked-provenance"
    assert na("runtime-amd", tagged) == "restore retained evidence: tools/box_persist.py fetch --tag rel0866-gate-amd"
    plain = row(action="restore-evidence", reasons=["evidence unavailable: /gone/report.json"])
    assert na("runtime-amd", plain, plan_restore={"runtime-amd": "tools/box_persist.py fetch --tag t1"}) == \
        "restore retained evidence: tools/box_persist.py fetch --tag t1"
    assert na("runtime-amd", plain).startswith("restore retained evidence: /gone/report.json (no retained copy named")
    assert na("candidate-amd", row(action="resume", active_run="run-7")).startswith("wait for or resume the unfinished run run-7")


def test_state_logged_waivers_are_read_back(tmp_path):
    w = waiver_file(tmp_path, prefixes=("adapters-",))
    state = dict(commit=C, owner_waivers=[dict(file=str(w), approved_by="Owner", reason="backlog", waived=["gate-amd"]),
                                          dict(file="/gone.json", approved_by="Owner", reason="gate", waived=["gate-nvidia"])])
    waivers = rr.waivers_from([w], state)
    rows = by_id(rr.build("0.8.66", C, fake_ledger(), {"gate-nvidia": ("owed", ""), "gate-amd": ("owed", "")}, waivers))
    assert rows["adapters-nvidia"]["state"] == "waived" and rows["adapters-nvidia"]["underlying_state"] == "blocked-provenance"
    assert rows["gate-nvidia"]["state"] == "waived" and rows["gate-nvidia"]["waiver"]["reason"] == "gate"
    assert rows["gate-amd"]["state"] == "waived"


def test_cli_writes_report_json_and_markdown_and_exits_3_when_not_ready(tmp_path, capsys):
    rows = tmp_path / "rows.json"
    rows.write_text(json.dumps(fake_ledger()))
    out = tmp_path / "out" / "report.json"
    assert rr.main(["--version", "0.8.66", "--commit", C, "--rows", str(rows), "--json", str(out)]) == 3
    doc = json.loads(out.read_text())
    assert doc["schema"] == rr.SCHEMA and doc["runs_nothing"] is True and len(doc["rows"]) == 7
    assert out.with_suffix(".md").read_text().startswith("# Release status: 0.8.66 at cccccccccccc")
    assert capsys.readouterr().out.splitlines()[0].startswith("REPORT 0.8.66 passed=1")


# ---------------------------------------------------------------- 0.8.65 regression fixture
def blocked_0865():
    return rr.rows_from_blocked_tsv(TSV)


def test_0865_fixture_with_the_waiver_shows_exactly_those_rows_waived():
    rows = blocked_0865()
    waiver = json.loads(WAIVER.read_text())
    assert len(rows) == 455 and len(waiver["prefixes"]) == 451
    report = rr.build("0.8.65", "6519a54c6579", rows, waivers=rr.waivers_from([WAIVER]))
    waived = {r["id"] for r in report["rows"] if r["state"] == "waived"}
    assert waived == set(waiver["prefixes"]) & set(rows) == set(waiver["prefixes"])
    assert report["totals"]["passed"] == report["totals"]["reused"] == 0   # nothing passed by the waiver
    assert all(r["waiver"]["approved_by"] == "Andrew Hendel" for r in report["rows"] if r["state"] == "waived")
    assert report["waived_underlying"] == {"blocked-provenance": 395, "missing-execution": 56}
    assert {r["id"] for r in report["rows"] if r["state"] == "missing-execution"} == {
        "candidate-nvidia", "published-amd", "published-apple", "published-nvidia"}
    assert report["totals_line"] == "REPORT 0.8.65 passed=0 reused=0 failed=0 missing=4 blocked=0 n/a=0 waived=451  ready=NO"
    # without the waiver nothing is waived and the 395 selector-blocked identity rows are blocked provenance
    bare = rr.build("0.8.65", "6519a54c6579", rows)
    assert bare["totals"]["waived"] == 0 and bare["totals"]["blocked-provenance"] == 395
    assert sum(r["id"].startswith("identity-") for r in bare["rows"]) == 442


# ------------- (b) the 442 identity rows replayed through the reference-admission judgement
SELECTOR = '''
import importlib.util, json, re, subprocess
from pathlib import Path
ROOT = str(Path(__file__).resolve().parent.parent)
VERSION = "python/mojolearn/_version.py"

def identity_break():
    spec = importlib.util.spec_from_file_location("fake_harness", ROOT + "/tools/identity_break.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

def lane_sources():
    lanes = json.loads(Path(ROOT, "tools/fake_lane_files.json").read_text())
    return {lane: set(files) for lane, files in lanes.items()}, {}

def changed_paths(ref):
    out = subprocess.check_output(["git", "-C", ROOT, "diff", "--name-only", ref, "HEAD"], text=True)
    return sorted(line for line in out.splitlines() if line)

def _label_view(text):
    return re.sub(r'__version__ = "[0-9.]+"', '__version__ = "<release-label>"', text)

def select(paths, ref=None, sources=None, backend=None, stop_when=None):
    lanes, by_path = set(), {}
    for path in paths:
        if path == VERSION:
            old = subprocess.check_output(["git", "-C", ROOT, "show", ref + ":" + path], text=True)
            if _label_view(old) == _label_view(Path(ROOT, path).read_text()):
                continue
        hit = {lane for lane, files in (sources or lane_sources()[0]).items() if path in files}
        if hit:
            by_path[path] = sorted(hit)
            lanes |= hit
    return dict(lanes=sorted(lanes), by_path=by_path, unattributed=[], stopped_early=False)
'''
PARTS = ("train", "infer", "model", "batch")


def h16(*parts):
    return hashlib.sha256("/".join(parts).encode()).hexdigest()[:16]


class Repo:
    def __init__(self, root):
        self.root = root
        root.mkdir()
        for args in (("init", "-q"), ("config", "user.email", "t@example.test"), ("config", "user.name", "Report Test")):
            self.git(*args)

    def git(self, *args):
        return subprocess.check_output(["git", "-C", str(self.root), *args], text=True, stderr=subprocess.DEVNULL).strip()

    def write(self, rel, text):
        path = self.root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)

    def commit(self, message):
        self.git("add", "-A")
        self.git("commit", "-q", "-m", message)
        return self.git("rev-parse", "HEAD")


def identity_units_0865():
    """{(vendor, lane): tested revision} of the 442 blocked identity rows, from the fixture."""
    out = {}
    for name, r in blocked_0865().items():
        if name.startswith("identity-"):
            vendor, lane = name[len("identity-"):].split("/", 1)
            out[(vendor, lane)] = r["tested_revision"]
    return out


def build_0865_like_repo(tmp_path, units):
    """A frozen tree shaped like 0.8.65: per-lane closures, column records at the four record commits the
    fixture names (apple 0412c8ee2f/4f7cdf9c2f/ba151faf37, nvidia/amd/cpu 4f7cdf9c2f, hf lanes ccc8c26dfc),
    tooling and docs commits after them, four column admissions (one README each), then the version bump
    (the hf lanes read python/mojolearn/_version.py) as the freeze."""
    repo = Repo(tmp_path / "repo")
    lanes = sorted({lane for _, lane in units})
    files = {lane: [f"core/{lane}.mojo"] + (["python/mojolearn/_version.py"] if lane.startswith("hf-") else [])
             for lane in lanes}
    repo.write("tools/fake_lane_files.json", json.dumps(files, sort_keys=True))
    repo.write("tools/lane_select.py", SELECTOR)
    repo.write("tools/identity_break.py", "FIXTURES = ['base']\nLANES = {%s}\nLANE_REVISIONS = {}\n"
               % ", ".join(f"{lane!r}: None" for lane in lanes))
    for name in ("release_reuse.py", "bincache.py"):
        repo.write("tools/" + name, "from pathlib import Path\nROOT=Path(__file__).resolve().parent.parent\n")
    for lane in lanes:
        repo.write(f"core/{lane}.mojo", f"fn {lane.replace('-', '_')}(): pass\n")
    repo.write("python/mojolearn/_version.py", '__version__ = "0.8.64"\n')
    commits = {}
    for short in ("4f7cdf9c2f", "0412c8ee2f", "ccc8c26dfc", "ba151faf37"):
        repo.write(f"docs/notes/{short}.md", "record commit stand-in\n")
        commits[short] = repo.commit("record commit stand-in for " + short)
    for i in range(3):   # merges that move no closure (0.8.65: 153 commits of tooling, docs and other lanes)
        repo.write(f"tools/release_tool_{i}.py", "# tooling\n")
        repo.commit(f"tooling {i}")
    # column records: one file per (class, record commit), the cells of the lanes resting on it
    records, index, cells = [], {}, {}
    for (vendor, lane), short in sorted(units.items()):
        index.setdefault((vendor, short), []).append(lane)
    harness = hashlib.sha256((repo.root / "tools/identity_break.py").read_bytes()).hexdigest()
    readmes = {}
    for (vendor, short), held in sorted(index.items()):
        body = dict(commit=commits[short], device_class=vendor, cells={
            f"{lane}/base": dict(verdict="STABLE", hashes=[h16(lane, "train")], **{p: [h16(lane, p)] for p in PARTS[1:]},
                                 **{p + "_verdict": "STABLE" for p in PARTS[1:]}) for lane in held})
        raw = json.dumps(body, sort_keys=True)
        sha = hashlib.sha256(raw.encode()).hexdigest()
        rel = f"{vendor}-{short}"
        repo.write(f"bench/results/identity_break/{rel}/{vendor}.identical.json", raw)
        records.append(dict(**{"class": vendor}, commit=commits[short], dir=rel, file=f"{vendor}.identical.json",
                            sha256=sha, vendor=vendor))
        readmes.setdefault(vendor, []).append(f"`bench/results/identity_break/retained/{sha}/{vendor}.identical.json`\n")
        for lane in held:
            for p in PARTS:
                cells.setdefault(f"{lane}/base", {}).setdefault(p, dict(ref=h16(lane, p), cols={}))["cols"][vendor] = \
                    len(records) - 1
    table = dict(cells=cells, records=records, harness_sha256=harness, lane_revisions={},
                 lane_admission={lane: dict(harness_sha256=harness) for lane in lanes})
    repo.write(IR.TABLE_PATH, json.dumps(table, sort_keys=True))
    for vendor, lines in readmes.items():   # one admission per column, as 2026-11-04-* did
        repo.write(f"bench/results/reference_admissions/2026-11-04-{vendor}/README.md", "".join(lines))
    repo.commit("admit the four identity columns")
    repo.write("python/mojolearn/_version.py", '__version__ = "0.8.65"\n')
    repo.write("CHANGELOG.md", "## 0.8.65\n")
    return repo, repo.commit("Release 0.8.65"), commits


def wanted(vendor, lane, revision, required_columns):
    coverage = [E.BASE_COVERAGE, "lane:" + lane]
    inputs = dict(unit=dict(kind="identity", vendor=vendor, lanes=[lane], release_modes=["identical"]),
                  native_sources={f"core/{lane}.mojo": "n"},
                  execution_environment={lane: dict(target="linux-64", bindings={})},
                  contract=dict(id=f"identity-{vendor}/{lane}", kind="identity", stage="pre", scope="s", coverage=coverage))
    return dict(source_revision=revision, inputs=inputs, fingerprint=R.digest(inputs), scope="s", coverage=coverage,
                kind="identity", stage="pre", missing_inputs=[], required=True,
                dependency_provenance=dict(identity_applicability=dict(required_columns=required_columns)))


def test_0865_identity_rows_replayed_through_the_reference_admission_are_never_blocked_provenance(tmp_path, monkeypatch):
    IR._MEMO.clear()
    D._REVISIONS.clear()
    monkeypatch.setenv(IR.RELEASE_RECORDS_ENV, str(tmp_path / "release-records"))
    units = identity_units_0865()
    assert len(units) == 442
    repo, freeze, commits = build_0865_like_repo(tmp_path, units)
    columns = {}
    for vendor, lane in units:
        columns.setdefault(lane, set()).add(vendor)
    requirements = {f"identity-{v}/{lane}": wanted(v, lane, freeze, sorted(columns[lane])) for v, lane in units}
    doc = dict(schema=R.SCHEMA, version="0.8.65", source_revision=freeze, requirements=requirements,
               source_root=str(repo.root), configuration={}, attempts=[], runs=[])
    rows = R.RequirementsLedger.evaluate(doc, root=str(repo.root), log_dir=tmp_path / "deps")
    report = rr.build("0.8.65", freeze, rows, waivers=rr.waivers_from([WAIVER]))
    t = report["totals"]
    assert t["blocked-provenance"] == 0 and t["failed"] == 0 and t["missing-execution"] == 0
    assert t["reused"] == 442 and t["waived"] == 0    # verified work needs no waiver
    assert sum(r.get("waiver_unused") is not None for r in report["rows"]) == 442
    sample = by_id(report)["identity-amd/hf-causal-lm"]
    assert sample["tested_revision"] == commits["ccc8c26dfc"]   # the version bump did not reach the hf lane
    assert any(e.startswith(IR.TABLE_PATH + "@") for e in sample["evidence"])
    assert report["totals_line"] == "REPORT 0.8.65 passed=0 reused=442 failed=0 missing=0 blocked=0 n/a=0 waived=0  ready=YES"


# ---------------------------------------------------------------- tools/release.py hooks
def release_with_rows(tmp_path, rows, **kw):
    from test_release_no_hands import make
    r = make(tmp_path, **kw)
    r.readiness_status = lambda write=False: rows
    return r


def test_release_report_writes_report_json_and_md_and_runs_nothing(tmp_path):
    import release
    r = release_with_rows(tmp_path, fake_ledger(), report=True, report_json=str(tmp_path / "rep" / "report.json"),
                          owner_waiver=str(waiver_file(tmp_path)))
    assert r.readonly
    assert r.go() == 3
    doc = json.loads((tmp_path / "rep" / "report.json").read_text())
    rows = {x["id"]: x for x in doc["rows"]}
    assert rows["coverage-source"]["state"] == "waived"
    assert rows["gate-amd"]["section"] == "step" and rows["gate-amd"]["state"] == "missing-execution"
    assert not any(k.startswith("published-smoke-") for k in rows)   # unrequested: not selected
    assert any(line.startswith("REPORT 0.8.66 ") for line in r.lines)
    assert (tmp_path / "rep" / "report.md").is_file() and r.calls == []
    assert release.Release.release_report.__doc__


def test_release_plan_embeds_the_report_totals_line(tmp_path):
    r = release_with_rows(tmp_path, fake_ledger(), plan=True, plan_json=str(tmp_path / "plan" / "plan.json"),
                          dry_run=True, plan_columns="")
    r.plan_facts = lambda: dict(version="0.8.66", commit=r.commit, changes=[], groups={}, lanes={}, gates={}, notes=[])
    r.release_plan()
    plan = json.loads((tmp_path / "plan" / "plan.json").read_text())
    assert plan["report_totals"].startswith("REPORT 0.8.66 passed=1 reused=1 failed=1 ")
    assert plan["report_totals"] in rp.lines(plan) and plan["report_totals"] in r.lines


def test_report_refuses_options_that_run_something():
    import release
    for extra in (["--status"], ["--plan"], ["--only", "gate-amd"], ["--dry-run"]):
        with pytest.raises(SystemExit):
            release.main(["0.8.66", "--report", *extra])


def test_a_changed_closure_in_the_replay_is_missing_execution_with_a_refresh_not_blocked(tmp_path, monkeypatch):
    IR._MEMO.clear()
    D._REVISIONS.clear()
    monkeypatch.setenv(IR.RELEASE_RECORDS_ENV, str(tmp_path / "release-records"))
    units = {k: v for k, v in identity_units_0865().items() if k[1] in ("mlp", "hf-tokenizer")}
    assert len(units) == 8
    repo, _, _ = build_0865_like_repo(tmp_path, units)
    repo.write("core/mlp.mojo", "fn mlp(): return 1\n")
    freeze = repo.commit("the mlp kernel changed after its record")
    requirements = {f"identity-{v}/{lane}": wanted(v, lane, freeze, ["amd", "apple", "cpu", "nvidia"]) for v, lane in units}
    doc = dict(schema=R.SCHEMA, version="0.8.65", source_revision=freeze, requirements=requirements,
               source_root=str(repo.root), configuration={}, attempts=[], runs=[])
    report = rr.build("0.8.65", freeze, R.RequirementsLedger.evaluate(doc, root=str(repo.root), log_dir=tmp_path / "d"))
    rows = by_id(report)
    assert report["totals"]["blocked-provenance"] == 0 and report["totals"]["missing-execution"] == 4
    assert rows["identity-amd/mlp"]["next_action"] == f"tools/reference_refresh.sh {freeze[:12]} amd"
    assert rows["identity-amd/hf-tokenizer"]["state"] == "reused"
