# SPDX-License-Identifier: Apache-2.0
"""tools/release_check_units.py: the device checks as lane-like units, reused by the identity rule (same device
class, same key -> the old good record, the check does not run). Temporary wheels and stores only."""
import json
import pathlib
import sys
import zipfile

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import release_check_units as U  # noqa: E402


def wheel(path, table=b"{}", binding=b"ELF", version_py=b"__version__ = '0.8.66'\\n"):
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w") as z:
        z.writestr("mojolearn/_backend.py", b"x = 1\\n")
        z.writestr("mojolearn/_mojolearn.so", binding)
        z.writestr("mojolearn/verify_reference/table.json", table)
        z.writestr("mojolearn/_version.py", version_py)
        z.writestr("mojolearn/identity_columns/COMMIT", b"c" * 40)
        z.writestr("mojolearn_core-0.8.66.dist-info/RECORD", b"r")
    return path


def test_every_check_is_a_unit():
    names = U.units()
    assert names[:2] == [U.INSTALL_LOAD, U.ADAPTERS] and len(names) == 2 + 24
    assert all(n.startswith("autograd/") for n in names[2:])


def test_install_load_key_ignores_the_table_and_metadata_but_not_a_binding(tmp_path):
    a = U.unit_key(U.INSTALL_LOAD, ROOT, U.wheel_members([wheel(tmp_path / "a.whl")]))
    b = U.unit_key(U.INSTALL_LOAD, ROOT, U.wheel_members([wheel(tmp_path / "b.whl", table=b'{"x": 1}',
                                                                 version_py=b"__version__ = '0.8.67'\\n")]))
    c = U.unit_key(U.INSTALL_LOAD, ROOT, U.wheel_members([wheel(tmp_path / "c.whl", binding=b"ELF2")]))
    assert a["digest"] == b["digest"] != c["digest"]
    assert "mojolearn/verify_reference/table.json" not in a["members"]


def rec(key, device="nvidia-sm_89", status="passed", at="2026-11-05T00:00:00Z"):
    return dict(schema=U.SCHEMA, unit=key["unit"], device_class=device, status=status, key=key, at=at)


def test_the_identity_rule_reuses_only_the_same_key_on_the_same_class(tmp_path):
    key = U.unit_key(U.INSTALL_LOAD, ROOT, U.wheel_members([wheel(tmp_path / "a.whl")]))
    assert U.decide(key, [rec(key)], "nvidia-sm_89")[0] == "reuse"
    assert U.decide(key, [rec(key, device="amd-gfx942")], "nvidia-sm_89")[0] == "run"
    assert U.decide(key, [rec(key, status="failed")], "nvidia-sm_89")[0] == "run"
    other = dict(key, digest="0" * 64, members=dict(key["members"], **{"mojolearn/_mojolearn.so": "f" * 64}))
    decision, why = U.decide(key, [rec(other)], "nvidia-sm_89")
    assert decision == "run" and "mojolearn/_mojolearn.so" in why
    assert U.decide(dict(key, closure_available=False), [rec(key)], "nvidia-sm_89")[0] == "run"


def test_plan_reads_the_store_and_skips_compiled_checks_on_cpu(tmp_path, monkeypatch):
    # fast: the closure walks are the tooling's own (tested there); here a fixed closure stands in
    monkeypatch.setattr(U, "python_closure", lambda root, seeds: ["python/mojolearn/__init__.py"])
    w = wheel(tmp_path / "a.whl")
    doc = U.plan("cpu", "cpu-linux-x86_64", ROOT, [w], tmp_path / "store", reuse=True)
    assert all("source" not in U.entry(r["unit"]) for r in doc["units"] if r["unit"].startswith("autograd/"))
    assert doc["reused"] == 0 and doc["owed"]
    row = next(r for r in doc["units"] if r["unit"] == U.INSTALL_LOAD)
    d = tmp_path / "store" / "cpu-linux-x86_64" / U.safe(U.INSTALL_LOAD)
    d.mkdir(parents=True)
    (d / "k.json").write_text(json.dumps(rec(row["key"], device="cpu-linux-x86_64")))
    again = U.plan("cpu", "cpu-linux-x86_64", ROOT, [w], tmp_path / "store", reuse=True)
    assert U.INSTALL_LOAD not in again["owed"] and again["reused"] == 1
    # the default (no --reuse-unchanged): every unit runs, whatever the store holds
    full = U.plan("cpu", "cpu-linux-x86_64", ROOT, [w], tmp_path / "store")
    assert full["reused"] == 0 and U.INSTALL_LOAD in full["owed"] and len(full["owed"]) == len(full["units"])


def test_summary_errors():
    assert U.summary_errors(None) == ["no check-units summary"]
    doc = dict(schema=U.SCHEMA + "+summary", units={"adapters": dict(status="failed"), "install-load": dict(status="reused")})
    assert U.summary_errors(doc) == ["unit adapters did not pass"]


def test_a_plan_that_owes_nothing_composes_its_summary_and_one_that_owes_refuses(tmp_path):
    import pytest
    plan = dict(schema=U.PLAN_SCHEMA, vendor="amd", device_class="amd-gfx942", reused=1, owed=[],
                units=[dict(unit="adapters", decision="reuse", key=dict(digest="d" * 64))])
    doc = U.compose_reused(plan, tmp_path / "units", "c" * 40)
    assert U.summary_errors(json.loads((tmp_path / "units" / U.SUMMARY).read_text()), plan) == [] and doc["ok"]
    with pytest.raises(ValueError, match="owes 1 unit"):
        U.compose_reused(dict(plan, owed=["adapters"]), tmp_path / "u2", "c" * 40)


def test_a_check_that_passed_outside_the_runner_gets_its_record(tmp_path):
    key = dict(unit="adapters", digest="e" * 64, files={}, natives={}, members={}, closure_available=True)
    plan = dict(vendor="apple", device_class="apple-m2-pro", units=[dict(unit="adapters", decision="run", key=key)])
    U.record_passed(tmp_path, plan, "adapters", "c" * 40, "2026-11-06T00:00:00Z")
    (got,) = U.records(tmp_path, "apple-m2-pro", "adapters")
    assert got["status"] == "passed" and U.decide(key, [got], "apple-m2-pro")[0] == "reuse"
