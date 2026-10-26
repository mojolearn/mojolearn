"""Synthetic identity column records through the real ledger; no product import or device work."""
import copy
import json
from pathlib import Path

import pytest

import release_identity_evidence as E
import release_requirement_receipts as receipts
import release_requirements as R

SOURCE = "1" * 40
OTHER = "2" * 40
LINUX = "Linux-6.8.0-59-generic-x86_64-with-glibc2.35"
MAC = "macOS-26.5-arm64-arm-64bit"
FIXTURES = {"base": {"X": "a" * 16, "y_clf": "b" * 16, "y_reg": "c" * 16},
            "ties": {"X": "d" * 16, "y_clf": "e" * 16, "y_reg": "f" * 16}}
HELDOUT = {"base": {"X": "0" * 16}, "ties": {"X": "9" * 16}}
LABELS = {"apple": "apple-m4", "nvidia": "nvidia-l40s-sm89", "amd": "amd-mi325x-gfx942",
          "cpu": "cpu-amd-epyc-9575f", "ptx": "nvidia-ptx-l40s-sm80"}
BACKENDS = {"apple": "metal", "nvidia": "cuda", "amd": "hip", "cpu": "cpu", "ptx": "cuda"}


def synthetic_scope(wanted, columns=("apple", "nvidia", "amd", "cpu"), kind="estimator-both-routes"):
    vendor = wanted["inputs"]["unit"]["vendor"]
    return dict(schema="mojolearn.release-identity-applicability.v1", source_revision=wanted["source_revision"],
                lane=E.unit_lane(wanted), vendor=vendor, kind=kind, claim_devices=1,
                applicable=vendor in columns, required_columns=list(columns),
                excluded={v: "no applicable arithmetic" for v in E.VENDORS if v not in columns},
                separate_requirements=[])


@pytest.fixture(autouse=True)
def no_git_declarations(monkeypatch):
    monkeypatch.setattr(E, "unit_scope", synthetic_scope)
    E._RECORDS.clear()
    monkeypatch.setattr(E, "declarations", lambda revision: (frozenset({"gpu-lane"}), ("par-",)))


def cell(digest, model=None):
    return {"verdict": "STABLE", "hashes": [digest], "infer": [digest[::-1]], "infer_verdict": "STABLE",
            "model": [model or digest[:8] + "00000000"], "model_verdict": "STABLE",
            "batch": [digest[1:] + "0"], "batch_verdict": "STABLE",
            "stepfull": ["n/a:not-sequence"], "stepfull_verdict": "N/A",
            "batchgrad": ["n/a:no-backward"], "batchgrad_verdict": "N/A",
            "batchscale": ["n/a:no-batch-scale"], "batchscale_verdict": "N/A",
            "ragged": ["n/a:not-ragged"], "ragged_verdict": "N/A"}


def record(cls, lanes, *, commit=SOURCE, platform=None, digests=None, **extra):
    label = LABELS[cls]
    platform = platform or (MAC if cls == "apple" else LINUX)
    digests = digests or {}
    j = dict(mode="identical", repeats=1, platform=platform, vendor=label, commit=commit, heldout_seed=1,
             fixtures=copy.deepcopy(FIXTURES), heldout=copy.deepcopy(HELDOUT), complete=True, partial_column=False,
             parts_omitted=[], parts_collected=["batch", "batchgrad", "batchscale", "ragged", "rlpair", "stepfull"],
             lane_revisions={}, batch_revisions={}, batch_protocol={"alone": 16},
             package=dict(par_devices="0", binding_provenance_protocol="loaded-native-v2",
                          bindings=[dict(module="mojolearn._x", file="/x.so", sha256="e" * 64, size=10)]),
             cells={f"{lane}/{f}": cell(digests.get((lane, f), "1234567890abcdef")) for lane in lanes for f in FIXTURES})
    options = dict(require_backend=BACKENDS[cls], require_cpu=cls == "cpu", vendor=None if cls == "cpu" else label,
                   lanes=",".join(lanes), repeats=1, skip="", fixtures="", partial_column="")
    j["resume_signature"] = dict(schema=1, source_sha256="a" * 64, environment_sha256="b" * 64, options=options,
                                 provenance={k: j[k] for k in ("commit", "mode", "vendor", "fixtures", "heldout")})
    if cls == "cpu":
        j["host"] = dict(arch="x86_64" if platform.startswith("Linux") else "arm64", column="cpu")
    j.update(extra)
    return j


def write_record(directory, name, j, *, repeats=1, dirty="none"):
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{name}.identical.json"
    path.write_text(json.dumps(j))
    sha = E.reference(path)["sha256"]
    (directory / f"{name}.provenance.txt").write_text(
        f"commit={j['commit']}\nlabel={j['vendor']}\nrepeats={repeats} cpu_threads=3\n"
        f"dirty_tracked={dirty}\nrecord_sha256={sha}\n")
    return path


def unit(vendor, lane, target=None, ptx=True, revision=SOURCE, native="f" * 64):
    target = target or ("osx-arm64" if vendor == "apple" else "linux-64")
    bindings = {f"{target}/host/-/host/_mojolearn_core_host": {"arch": ""}}
    if vendor == "nvidia":
        bindings[f"{target}/cuda/sm_89/identical/_mojolearn"] = {"arch": "sm_89"}
        if ptx:
            bindings[f"{target}/cuda/sm_80/identical/_mojolearn"] = {"arch": "sm_80"}
    inputs = dict(unit=dict(schema="mojolearn.release-dependencies.v1", kind="identity", vendor=vendor,
                            release_modes=["identical"], build_flags={}, lanes=[lane]),
                  native_sources={"bindings/x.mojo": native},
                  execution_environment={lane: dict(target=target, toolchain={"platform": target}, bindings=bindings)},
                  configuration={}, contract=dict(id=f"identity-{vendor}/{lane}", scope="s", kind="identity",
                                                  stage="pre", coverage=[E.BASE_COVERAGE, "lane:" + lane]))
    return dict(source_revision=revision, inputs=inputs, fingerprint=R.digest(inputs), scope="s",
                coverage=[E.BASE_COVERAGE, "lane:" + lane], missing_inputs=[], kind="identity", stage="pre",
                provenance_sources=[], required=True)


def ledger_with(tmp_path, lanes=("ols",), cpu_target="linux-64", ptx=True):
    reqs = {f"identity-{v}/{lane}": unit(v, lane, cpu_target if v == "cpu" else None, ptx)
            for v in E.VENDORS for lane in lanes}
    ledger = R.RequirementsLedger(tmp_path / "ledger.json")
    ledger.sync(dict(schema=R.SCHEMA, version="0.0.1", source_revision=SOURCE, requirements=reqs, configuration={}))
    return ledger


def fake_check(records, root, source, out):
    path = Path(out) / "identity-columns-check.log"
    Path(out).mkdir(parents=True, exist_ok=True)
    path.write_text(E.CHECK_HEADER + source + " rc=0\n"
                    + "".join(f"IDCOLUMN {Path(p).resolve()} admissible=yes class=x\n" for p in records))
    return E.reference(path)


def columns(tmp_path, lanes=("ols",), classes=("apple", "nvidia", "amd", "cpu", "ptx"), **per_class):
    directory = tmp_path / "records"
    for cls in classes:
        j = per_class.get(cls) or record(cls, list(lanes))
        write_record(directory, LABELS[cls], j)
    return directory


def begin_all(ledger, tmp_path):
    out = tmp_path / "out"
    return [path for v in E.VENDORS if (path := E.begin(ledger, v, out)[0]) is not None]


def run(ledger, tmp_path, records, launches=None, **kwargs):
    return E.admit(ledger, [records], launches if launches is not None else [], tmp_path / "out", check=fake_check,
                   **kwargs)


def test_complete_columns_pass_every_unit_and_survive_reevaluation(tmp_path):
    ledger = ledger_with(tmp_path)
    assert all(R.producer_admission_available(w) for w in ledger.read()["requirements"].values())
    launches = begin_all(ledger, tmp_path)
    assert all(ledger.plan()[f"identity-{v}/ols"]["action"] == "resume" for v in E.VENDORS)
    result = run(ledger, tmp_path, columns(tmp_path), launches)
    assert result["passed"] == 4 and not result["problems"] and not result["waiting"]
    rows = ledger.plan()
    for v in E.VENDORS:
        row = rows[f"identity-{v}/ols"]
        assert row["status"] == "passed" and row["action"] == "reuse", row["reasons"]
        assert row["receipt"]["identity_detail"]["launch_phase"] == "before-execution"
        assert row["receipt"]["coverage"][1]["expected"] == 2 * len(E.ALL_PARTS[:-1])
    assert rows["identity-nvidia/ols"]["receipt"]["identity_detail"]["ptx_required"] is True
    assert ledger.gate(stage="pre")[0]


def test_begin_at_admission_is_labelled(tmp_path):
    ledger = ledger_with(tmp_path)
    result = run(ledger, tmp_path, columns(tmp_path), begin_missing=True)
    assert result["passed"] == 4
    assert ledger.plan()["identity-amd/ols"]["receipt"]["identity_detail"]["launch_phase"] == "at-admission"


def test_records_without_a_launch_stay_untested(tmp_path):
    ledger = ledger_with(tmp_path)
    result = run(ledger, tmp_path, columns(tmp_path))
    assert result["recorded"] == 0 and len(result["waiting"]) == 4
    assert all(r["status"] == "untested" for r in ledger.plan().values())


def test_linux_cpu_column_never_satisfies_an_osx_cpu_target(tmp_path):
    ledger = ledger_with(tmp_path, cpu_target="osx-arm64")
    result = run(ledger, tmp_path, columns(tmp_path), begin_all(ledger, tmp_path))
    assert result["passed"] == 3
    assert "differs from the frozen execution target osx-arm64" in result["problems"]["identity-cpu/ols"]
    assert ledger.plan()["identity-cpu/ols"]["status"] != "passed"


def test_cpu_digest_difference_fails_the_contract(tmp_path):
    cpu = record("cpu", ["ols"], digests={("ols", "ties"): "fedcba0987654321"})
    ledger = ledger_with(tmp_path)
    result = run(ledger, tmp_path, columns(tmp_path, cpu=cpu), begin_all(ledger, tmp_path))
    assert result["failed"] == 4 and result["passed"] == 0
    rows = ledger.plan()
    assert rows["identity-cpu/ols"]["status"] == "failed" and rows["identity-cpu/ols"]["action"] == "repair"
    detail = rows["identity-cpu/ols"]["receipt"]["identity_detail"]
    assert any("cpu" in p and "ols/ties" in p for p in detail["problems"])
    # The GPU columns agree among themselves but the common contract does not.
    gpu = rows["identity-amd/ols"]["receipt"]["coverage"]
    assert gpu[1]["failed"] == 0 and gpu[0]["failed"] > 0


def test_one_gpu_vendor_differing_fails_every_unit(tmp_path):
    amd = record("amd", ["ols"], digests={("ols", "base"): "0000000000000001"})
    ledger = ledger_with(tmp_path)
    result = run(ledger, tmp_path, columns(tmp_path, amd=amd), begin_all(ledger, tmp_path))
    assert result["failed"] == 4


def test_ptx_column_is_required_and_compared_for_nvidia(tmp_path):
    ledger = ledger_with(tmp_path)
    launches = begin_all(ledger, tmp_path)
    result = run(ledger, tmp_path, columns(tmp_path, classes=("apple", "nvidia", "amd", "cpu")), launches)
    assert result["passed"] == 3 and "ptx" in result["waiting"]["identity-nvidia/ols"]
    ptx = record("ptx", ["ols"], digests={("ols", "base"): "0000000000000002"})
    write_record(tmp_path / "records", LABELS["ptx"], ptx)
    E._RECORDS.clear()
    result = run(ledger, tmp_path, tmp_path / "records", launches)
    assert result["failed"] == 1 and ledger.plan()["identity-nvidia/ols"]["status"] == "failed"


def test_nvidia_without_the_ptx_set_needs_no_ptx_column(tmp_path):
    ledger = ledger_with(tmp_path, ptx=False)
    result = run(ledger, tmp_path, columns(tmp_path, classes=("apple", "nvidia", "amd", "cpu")),
                 begin_all(ledger, tmp_path))
    assert result["passed"] == 4


@pytest.mark.parametrize("change,reason", [
    (dict(commit=OTHER), "commit witness"),
    (dict(repeats=2), "repeats"),
    (dict(partial_column=True, parts_omitted=["batch"]), "partial column"),
    (dict(batch_sabotage=True), "sabotage"),
    (dict(complete=False), "incomplete"),
    (dict(platform="Windows-11"), "platform"),
])
def test_refused_records_leave_units_unrecorded(tmp_path, change, reason):
    amd = record("amd", ["ols"])
    amd.update(change)
    if "commit" in change:
        amd["resume_signature"]["provenance"]["commit"] = change["commit"]
    ledger = ledger_with(tmp_path)
    result = run(ledger, tmp_path, columns(tmp_path, amd=amd), begin_all(ledger, tmp_path))
    assert result["recorded"] == 0 and len(result["refused_records"]) == 1
    assert reason in " ".join(next(iter(result["refused_records"].values())))
    assert all(r["status"] == "untested" for r in ledger.plan().values())


def test_dirty_tree_provenance_is_refused(tmp_path):
    ledger = ledger_with(tmp_path)
    directory = columns(tmp_path)
    write_record(directory, LABELS["amd"], record("amd", ["ols"]), dirty="tools/x.py")
    result = run(ledger, tmp_path, directory, begin_all(ledger, tmp_path))
    assert result["recorded"] == 0 and "provenance" in " ".join(sum(result["refused_records"].values(), []))


def test_shard_records_admit_per_lane_and_repeats_are_refused(tmp_path):
    lanes = ("ols", "ridge")
    ledger = ledger_with(tmp_path, lanes=lanes)
    directory = columns(tmp_path, lanes=lanes, classes=("apple", "amd", "cpu", "ptx"))
    write_record(directory, "nvidia-l40s-sm89.s0of2", record("nvidia", ["ols"]))
    write_record(directory, "nvidia-l40s-sm89.s1of2", record("nvidia", ["ridge"]))
    result = run(ledger, tmp_path, directory, begin_all(ledger, tmp_path))
    assert result["passed"] == 8
    second = tmp_path / "again"
    ledger2 = ledger_with(tmp_path / "l2", lanes=lanes)
    directory2 = columns(second, lanes=lanes)
    write_record(directory2, "amd-mi325x-gfx942.lanes-0000aaaa", record("amd", ["ols"]))
    result = E.admit(ledger2, [directory2], begin_all(ledger2, second), second / "out", check=fake_check)
    assert all("repeat records" in result["problems"][f"identity-{v}/ols"] for v in E.VENDORS)
    assert result["passed"] == 4  # ridge is unaffected


def test_declared_gpu_only_lane_needs_no_cpu_value(tmp_path):
    ledger = ledger_with(tmp_path, lanes=("gpu-lane",))
    cpu = record("cpu", ["gpu-lane"])
    for c in cpu["cells"].values():
        c.update(verdict="REFUSED", hashes=["refused"])
    result = run(ledger, tmp_path, columns(tmp_path, lanes=("gpu-lane",), cpu=cpu), begin_all(ledger, tmp_path))
    assert result["passed"] == 4
    detail = ledger.plan()["identity-cpu/gpu-lane"]["receipt"]["identity_detail"]
    assert detail["gpu_only_declared"] is True and detail["counts"]["cpu_declared_na"] > 0


def test_undeclared_cpu_absence_fails(tmp_path):
    ledger = ledger_with(tmp_path)
    cpu = record("cpu", ["ols"])
    for c in cpu["cells"].values():
        c.update(verdict="REFUSED", hashes=["refused"])
    result = run(ledger, tmp_path, columns(tmp_path, cpu=cpu), begin_all(ledger, tmp_path))
    assert result["failed"] == 4


def test_changed_record_bytes_withdraw_the_pass(tmp_path):
    ledger = ledger_with(tmp_path)
    directory = columns(tmp_path)
    run(ledger, tmp_path, directory, begin_all(ledger, tmp_path))
    assert ledger.plan()["identity-amd/ols"]["status"] == "passed"
    path = directory / f"{LABELS['amd']}.identical.json"
    j = json.loads(path.read_text())
    j["cells"]["ols/base"]["hashes"] = ["0000000000000003"]
    path.write_text(json.dumps(j))
    E._RECORDS.clear()
    row = ledger.plan()["identity-amd/ols"]
    assert row["status"] != "passed" and row["action"] == "complete-provenance"


def test_receipt_counts_cannot_be_edited(tmp_path):
    ledger = ledger_with(tmp_path)
    run(ledger, tmp_path, columns(tmp_path), begin_all(ledger, tmp_path))
    doc = ledger.read()
    attempt = next(a for a in doc["attempts"] if a["requirement"] == "identity-apple/ols")
    receipt = copy.deepcopy(attempt["receipt"])
    receipt["identity_detail"]["counts"]["slots"] += 1
    wanted = dict(attempt["manifest"], inputs=attempt["inputs"])
    assert receipts.validate_receipt("identity-apple/ols", wanted, receipt)


def test_supports_requirement_shapes():
    good = unit("amd", "ols")
    assert E.supports_requirement(good)
    for edit in (lambda w: w.update(missing_inputs=["build_flags"]),
                 lambda w: w["inputs"]["unit"].update(lanes=["ols", "ridge"]),
                 lambda w: w.update(coverage=[E.BASE_COVERAGE]),
                 lambda w: w["inputs"]["execution_environment"]["ols"].update(target="win-64"),
                 lambda w: w["inputs"]["unit"].update(vendor="ptx")):
        bad = copy.deepcopy(good)
        edit(bad)
        assert not E.supports_requirement(bad)


def test_configuration_merge_keeps_retained_context():
    saved = dict(dependency_context=dict(cpu_target="osx-arm64", compiler_sha256={"linux-64": "c" * 64}),
                 build_flags={"A": "1"})
    supplied = json.loads((Path(__file__).with_name("release_identity_targets.json")).read_text())
    merged = R.merge_configuration(saved, supplied)
    assert merged["dependency_context"] == dict(cpu_target="linux-64", compiler_sha256={"linux-64": "c" * 64})
    assert saved["dependency_context"]["cpu_target"] == "osx-arm64"
    with pytest.raises(ValueError):
        R.merge_configuration(saved, ["x"])


# ------------------------------------------------- refreeze: fingerprint-equal begun revisions
def refreeze(ledger, lanes=("ols",), changed=(), revision=OTHER):
    """Sync the ledger to a later frozen source; ``changed`` lanes get different inputs."""
    reqs = {f"identity-{v}/{lane}": unit(v, lane, revision=revision,
                                         native=("0" * 64 if lane in changed else "f" * 64))
            for v in E.VENDORS for lane in lanes}
    ledger.sync(dict(schema=R.SCHEMA, version="0.0.1", source_revision=revision, requirements=reqs, configuration={}))


def finish_unrecorded(ledger, tmp_path, names):
    """Close begun runs without a result (the operator's abort before a re-begin)."""
    note = tmp_path / "abort.txt"
    note.write_text("aborted: inputs changed at the refreeze\n")
    rows = ledger.plan()
    for name in names:
        ledger.record(rows[name]["active_run"], "failed", [str(note)], "aborted before completion: refreeze", 1, 1, 0, 1)


def test_equal_fingerprint_older_revision_records_admit_with_provenance(tmp_path):
    ledger = ledger_with(tmp_path)
    launches = begin_all(ledger, tmp_path)
    before = {n: w["fingerprint"] for n, w in ledger.read()["requirements"].items()}
    refreeze(ledger)
    assert {n: w["fingerprint"] for n, w in ledger.read()["requirements"].items()} == before
    rows = ledger.plan()
    assert all(rows[f"identity-{v}/ols"]["active_revision"] == SOURCE for v in E.VENDORS)
    # Records made at the begun revision; an extra lane that no unit declares is ignored with a note.
    directory = columns(tmp_path, lanes=("ols", "lasso"))
    result = run(ledger, tmp_path, directory, launches)
    assert result["passed"] == 4 and not result["problems"] and not result["refused_records"], result
    assert all(x["lanes"] == ["lasso"] and x["commit"] == SOURCE for x in result["ignored"].values())
    assert len(result["ignored"]) == 5
    rows = ledger.plan()
    for v in E.VENDORS:
        row = rows[f"identity-{v}/ols"]
        assert row["status"] == "passed" and row["action"] == "reuse", row["reasons"]
        revisions = row["receipt"]["identity_detail"]["revisions"]
        assert revisions["begun_revision"] == SOURCE and revisions["frozen_revision"] == OTHER
        assert revisions["input_fingerprint_equal"] is True
        assert set(revisions["record_commits"].values()) == {SOURCE}
        assert row["receipt"]["source_revision"] == SOURCE
    assert ledger.gate(stage="pre")[0]


def test_records_at_the_frozen_source_do_not_satisfy_a_unit_begun_earlier(tmp_path):
    ledger = ledger_with(tmp_path)
    launches = begin_all(ledger, tmp_path)
    refreeze(ledger)
    directory = tmp_path / "records"
    for cls in ("apple", "nvidia", "amd", "cpu", "ptx"):
        write_record(directory, LABELS[cls], record(cls, ["ols"], commit=OTHER))
    result = run(ledger, tmp_path, directory, launches)
    assert result["recorded"] == 0 and len(result["ignored"]) == 5
    assert not result["problems"] and not result["waiting"]


def test_changed_fingerprint_older_revision_records_are_refused(tmp_path):
    ledger = ledger_with(tmp_path)
    launches = begin_all(ledger, tmp_path)
    refreeze(ledger, changed=("ols",))
    rows = ledger.plan()
    assert all(rows[f"identity-{v}/ols"]["active_input_fingerprint"] != rows[f"identity-{v}/ols"]["input_fingerprint"]
               for v in E.VENDORS)
    result = run(ledger, tmp_path, columns(tmp_path), launches)
    assert result["recorded"] == 0
    assert all("inputs that differ" in result["problems"][f"identity-{v}/ols"] for v in E.VENDORS)
    assert not result["ignored"]  # refused by unit, not silently ignored
    assert all(r["status"] == "untested" for r in ledger.plan().values())


def test_stale_begun_revision_never_admits_without_a_unit(tmp_path):
    """A record at a revision no begun unit carries is refused by its commit witness."""
    ledger = ledger_with(tmp_path)
    launches = begin_all(ledger, tmp_path)
    amd = record("amd", ["ols"], commit="3" * 40)
    amd["resume_signature"]["provenance"]["commit"] = "3" * 40
    result = run(ledger, tmp_path, columns(tmp_path, amd=amd), launches)
    assert result["recorded"] == 0
    assert "commit witness" in " ".join(sum(result["refused_records"].values(), []))


def test_columns_from_different_fingerprint_equal_revisions(tmp_path):
    ledger = ledger_with(tmp_path)
    out = tmp_path / "out"
    launches = [E.begin(ledger, v, out)[0] for v in ("apple", "amd", "cpu")]
    refreeze(ledger)
    launches.append(E.begin(ledger, "nvidia", out)[0])
    assert ledger.plan()["identity-nvidia/ols"]["active_revision"] == OTHER
    directory = tmp_path / "records"
    for cls in ("apple", "amd", "cpu"):
        write_record(directory, LABELS[cls], record(cls, ["ols"], commit=SOURCE))
    for cls in ("nvidia", "ptx"):
        write_record(directory, LABELS[cls], record(cls, ["ols"], commit=OTHER))
    result = run(ledger, tmp_path, directory, launches)
    assert result["passed"] == 4 and not result["problems"] and not result["ignored"], result
    rows = ledger.plan()
    nvidia = rows["identity-nvidia/ols"]["receipt"]["identity_detail"]["revisions"]
    assert nvidia["begun_revision"] == OTHER and nvidia["frozen_revision"] == OTHER
    assert nvidia["record_commits"] == dict(apple=SOURCE, amd=SOURCE, cpu=SOURCE, nvidia=OTHER, ptx=OTHER)
    apple = rows["identity-apple/ols"]["receipt"]["identity_detail"]["revisions"]
    assert apple["begun_revision"] == SOURCE and apple["record_commits"]["nvidia"] == OTHER
    assert all(rows[f"identity-{v}/ols"]["status"] == "passed" for v in E.VENDORS)


def test_columns_from_different_revisions_still_compare_bits(tmp_path):
    ledger = ledger_with(tmp_path)
    out = tmp_path / "out"
    launches = [E.begin(ledger, v, out)[0] for v in ("apple", "amd", "cpu")]
    refreeze(ledger)
    launches.append(E.begin(ledger, "nvidia", out)[0])
    directory = tmp_path / "records"
    for cls in ("apple", "amd", "cpu"):
        write_record(directory, LABELS[cls], record(cls, ["ols"], commit=SOURCE))
    write_record(directory, LABELS["ptx"], record("ptx", ["ols"], commit=OTHER))
    write_record(directory, LABELS["nvidia"], record("nvidia", ["ols"], commit=OTHER,
                                                    digests={("ols", "base"): "0000000000000004"}))
    result = run(ledger, tmp_path, directory, launches)
    assert result["failed"] == 4 and result["passed"] == 0


def test_passed_column_at_an_earlier_revision_serves_a_later_unit(tmp_path):
    """Units admitted earlier keep supplying their column to a unit admitted after the refreeze."""
    ledger = ledger_with(tmp_path)
    launches = begin_all(ledger, tmp_path)
    directory = columns(tmp_path, classes=("apple", "nvidia", "amd", "cpu"))
    result = run(ledger, tmp_path, directory, launches)
    assert result["passed"] == 3 and "ptx" in result["waiting"]["identity-nvidia/ols"]
    refreeze(ledger)
    write_record(directory, LABELS["ptx"], record("ptx", ["ols"]))
    E._RECORDS.clear()
    result = run(ledger, tmp_path, directory, launches)
    assert result["passed"] == 1 and not result["problems"], result
    assert ledger.plan()["identity-nvidia/ols"]["status"] == "passed"


def test_repeat_rule_is_per_unit_revision(tmp_path):
    lanes = ("ols", "ridge")
    ledger = ledger_with(tmp_path, lanes=lanes)
    launches = begin_all(ledger, tmp_path)
    refreeze(ledger, lanes=lanes, changed=("ridge",))
    finish_unrecorded(ledger, tmp_path, [f"identity-{v}/ridge" for v in E.VENDORS])
    out = tmp_path / "out"
    launches += [E.begin(ledger, v, out, ["ridge"])[0] for v in E.VENDORS]
    rows = ledger.plan()
    assert all(rows[f"identity-{v}/ridge"]["active_revision"] == OTHER for v in E.VENDORS)
    # A full column at the begun revision covers ridge too; the ridge units use the later LANES= record.
    directory = columns(tmp_path, lanes=lanes)
    for cls in ("apple", "nvidia", "amd", "cpu", "ptx"):
        write_record(directory, LABELS[cls] + ".lanes-ridge", record(cls, ["ridge"], commit=OTHER))
    result = run(ledger, tmp_path, directory, launches)
    assert result["passed"] == 8 and not result["problems"], result
    assert sorted(x["lanes"] == ["ridge"] and x["commit"] == SOURCE for x in result["ignored"].values()) == [True] * 5
    rows = ledger.plan()
    assert rows["identity-amd/ridge"]["receipt"]["identity_detail"]["revisions"]["record_commits"]["amd"] == OTHER
    assert rows["identity-amd/ols"]["receipt"]["identity_detail"]["revisions"]["record_commits"]["amd"] == SOURCE
    # Two records of one column at the revision a unit uses are still a repeat.
    second = tmp_path / "again"
    ledger2 = ledger_with(second, lanes=lanes)
    launches2 = begin_all(ledger2, second)
    refreeze(ledger2, lanes=lanes)
    directory2 = columns(second, lanes=lanes)
    write_record(directory2, "amd-mi325x-gfx942.lanes-0000aaaa", record("amd", ["ols"]))
    write_record(directory2, "amd-mi325x-gfx942.lanes-0000bbbb", record("amd", ["ridge"], commit=OTHER))
    result = E.admit(ledger2, [directory2], launches2, second / "out", check=fake_check)
    assert all("repeat records" in result["problems"][f"identity-{v}/ols"] for v in E.VENDORS)
    assert result["passed"] == 4  # ridge: its unit uses the begun revision; the later record is ignored
    assert any(x["commit"] == OTHER for x in result["ignored"].values())


def test_report_without_fingerprint_equality_is_refused_on_reevaluation(tmp_path):
    ledger = ledger_with(tmp_path)
    launches = begin_all(ledger, tmp_path)
    refreeze(ledger)
    run(ledger, tmp_path, columns(tmp_path), launches)
    attempt = next(a for a in ledger.read()["attempts"] if a["requirement"] == "identity-amd/ols")
    wanted = dict(attempt["manifest"], inputs=attempt["inputs"])
    report_path = Path(attempt["receipt"]["producer_report"]["path"])
    report = json.loads(report_path.read_text())
    report["revisions"]["frozen_fingerprint"] = "0" * 64
    with pytest.raises(ValueError, match="cannot qualify"):
        E.column_revisions(report["revisions"], SOURCE, "amd", wanted, E.required_classes(wanted))
    report["revisions"]["frozen_fingerprint"] = wanted["fingerprint"]
    report["revisions"]["columns"]["apple"] = "3" * 40
    with pytest.raises(ValueError, match="not a fingerprint-equal run"):
        E.column_revisions(report["revisions"], SOURCE, "amd", wanted, E.required_classes(wanted))


@pytest.fixture
def version_label_units(monkeypatch):
    import hashlib
    import release_requirement_dependencies as D
    old_bytes = b'# version only\n__version__ = "0.8.58"\n'
    new_bytes = b'# version only\n__version__ = "0.8.59"\n'
    sources = {SOURCE: old_bytes, OTHER: new_bytes}
    anchor = b'version only in unconsumed sidecar'
    monkeypatch.setattr(D, 'VERSION_LABEL_ANCHORS', {'reviewed-consumer': hashlib.sha256(anchor).hexdigest()})
    monkeypatch.setattr(D, '_git_bytes', lambda root, revision, path: sources[revision] if path == D.VERSION_PATH else anchor)
    monkeypatch.setattr(D, '_revision', lambda root, revision: revision)
    def make(vendor, revision):
        wanted = unit(vendor, 'sequence-export-roundtrip', revision=revision)
        wanted['inputs']['native_sources'][D.VERSION_PATH] = hashlib.sha256(sources[revision]).hexdigest()
        wanted['fingerprint'] = R.digest(wanted['inputs'])
        return wanted
    return make, sources


def test_version_only_refreeze_preserves_begun_fingerprints_and_admits(tmp_path, version_label_units):
    make, _ = version_label_units
    lane = 'sequence-export-roundtrip'
    ledger = R.RequirementsLedger(tmp_path / 'ledger.json', source_root=tmp_path)
    def sync(revision):
        ledger.sync(dict(schema=R.SCHEMA, version='0.8.59', source_revision=revision, configuration={},
                         requirements={f'identity-{v}/{lane}': make(v, revision) for v in E.VENDORS}))
    sync(SOURCE)
    launches = begin_all(ledger, tmp_path)
    original = copy.deepcopy(ledger.read()['runs'])
    sync(OTHER)
    result = E.admit(ledger, [columns(tmp_path, lanes=(lane,))], launches, tmp_path / 'out', check=fake_check)
    assert result['passed'] == 4, result
    assert ledger.read()['runs'] == original
    rows = ledger.plan()
    assert all(row['action'] == 'reuse' for row in rows.values())
    assert all(row['tested_revision'] == SOURCE for row in rows.values())
    assert all(row['equivalence_proof']['kind'] == 'unconsumed-release-version-literal' for row in rows.values())
    attempt = ledger.read()['attempts'][0]
    report = E.load(attempt['receipt']['producer_report']['path'])
    wanted = dict(attempt['manifest'], inputs=attempt['inputs'])
    report['revisions']['version_label_proof']['version_source_sha256']['new'] = '0' * 64
    with pytest.raises(ValueError, match='cannot qualify'):
        E.column_revisions(report['revisions'], SOURCE, wanted['inputs']['unit']['vendor'], wanted, E.required_classes(wanted))


@pytest.mark.parametrize('change', ['source', 'flags', 'lane', 'consumer', 'literal_code', 'version_hash', 'missing'])
def test_version_proof_refuses_meaningful_or_unreviewed_changes(version_label_units, monkeypatch, change):
    import release_requirement_dependencies as D
    make, sources = version_label_units
    old, new = make('amd', SOURCE), make('amd', OTHER)
    assert D.version_label_proof('.', old, new)
    if change == 'source':
        new['inputs']['native_sources']['bindings/x.mojo'] = 'changed'
    elif change == 'flags':
        new['inputs']['unit']['build_flags'] = {'changed': True}
    elif change == 'lane':
        for wanted in (old, new):
            wanted['inputs']['unit']['lanes'] = ['checkpoint-version-consumer']
            wanted['fingerprint'] = R.digest(wanted['inputs'])
    elif change == 'consumer':
        monkeypatch.setattr(D, 'VERSION_LABEL_ANCHORS', {'reviewed-consumer': '0' * 64})
    elif change == 'literal_code':
        sources[OTHER] += b'execute()\n'
    elif change == 'version_hash':
        new['inputs']['native_sources'][D.VERSION_PATH] = '0' * 64
    elif change == 'missing':
        new['missing_inputs'] = ['compiler']
    new['fingerprint'] = R.digest(new['inputs'])
    assert D.version_label_proof('.', old, new) is None


def scoped_ledger(tmp_path, monkeypatch, columns, kind):
    lane = 'host-lane' if columns == ('cpu',) else 'gpu-lane'
    monkeypatch.setattr(E, 'unit_scope', lambda wanted: synthetic_scope(wanted, columns, kind))
    ledger = ledger_with(tmp_path, lanes=(lane,))
    doc = ledger.read()
    for wanted in doc['requirements'].values():
        wanted['required'] = wanted['inputs']['unit']['vendor'] in columns
    ledger.sync(doc)
    return ledger, lane


def test_host_only_lane_admits_original_cpu_run_without_gpu_records(tmp_path, monkeypatch):
    ledger, lane = scoped_ledger(tmp_path, monkeypatch, ('cpu',), 'cpu-host-route-only')
    launches = begin_all(ledger, tmp_path)
    original = copy.deepcopy(ledger.read()['runs'])
    result = E.admit(ledger, [columns(tmp_path, lanes=(lane,), classes=('cpu',))], launches,
                     tmp_path / 'out', check=fake_check)
    assert result['passed'] == 1 and not result['problems']
    assert ledger.read()['runs'] == original
    rows = ledger.plan()
    assert rows['identity-cpu/host-lane']['status'] == 'passed'
    assert all(rows[f'identity-{v}/host-lane']['required'] is False for v in E.GPU_CLASSES)
    detail = rows['identity-cpu/host-lane']['receipt']['identity_detail']
    assert set(detail['records']) == {'cpu'}
    assert 'no GPU arithmetic' in detail['comparison']


def test_host_only_unusable_cell_is_failure_not_scope_success(tmp_path, monkeypatch):
    ledger, lane = scoped_ledger(tmp_path, monkeypatch, ('cpu',), 'cpu-host-route-only')
    cpu = record('cpu', [lane])
    cpu['cells'][f'{lane}/base']['verdict'] = 'DIVERGENT'
    result = E.admit(ledger, [columns(tmp_path, lanes=(lane,), classes=('cpu',), cpu=cpu)],
                     begin_all(ledger, tmp_path), tmp_path / 'out', check=fake_check)
    assert result['passed'] == 0
    assert ledger.plan()['identity-cpu/host-lane']['status'] == 'failed'


def test_gpu_only_scope_requires_no_cpu_record_but_keeps_ptx_and_three_gpus(tmp_path, monkeypatch):
    ledger, lane = scoped_ledger(tmp_path, monkeypatch, E.GPU_CLASSES, 'gpu-only')
    result = E.admit(ledger, [columns(tmp_path, lanes=(lane,), classes=(*E.GPU_CLASSES, 'ptx'))],
                     begin_all(ledger, tmp_path), tmp_path / 'out', check=fake_check)
    assert result['passed'] == 3
    rows = ledger.plan()
    assert rows['identity-cpu/gpu-lane']['required'] is False
    nv = rows['identity-nvidia/gpu-lane']['receipt']['identity_detail']
    assert set(nv['records']) == {*E.GPU_CLASSES, 'ptx'}
    assert nv['counts']['cpu_declared_na'] > 0


def test_host_scope_cannot_qualify_a_gpu_unit(tmp_path, monkeypatch):
    wanted = unit('amd', 'host-lane')
    monkeypatch.setattr(E, 'unit_scope', lambda w: synthetic_scope(w, ('cpu',), 'cpu-host-route-only'))
    with pytest.raises(ValueError, match='inapplicable'):
        E.required_classes(wanted)
