"""Tooling test for .github/workflows/apple-fast-check.yml, tools/ops/apple_fast_check.py and the Apple kinds of
tools/ops/native_checks_steps.sh (lane fx-apple-fast-ci, 2026-11-20). Nothing here compiles or touches a GPU."""

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


afc = _load("apple_fast_check", ROOT / "tools/ops/apple_fast_check.py")
WF = yaml.safe_load((ROOT / ".github/workflows/apple-fast-check.yml").read_text())


def test_workflow_is_manual_and_only_the_named_jobs_run_on_a_mac():
    on = WF.get("on", WF.get(True))
    assert set(on) == {"workflow_dispatch"}  # light-checks.yml refuses anything else
    inputs = on["workflow_dispatch"]["inputs"]
    assert inputs["step"]["options"] == ["parse", "build", "smoke"] and inputs["ref"]["required"] is True
    assert inputs["runner"]["default"] == "macos-15"  # the free standard Apple-silicon runner
    assert set(WF["jobs"]) == {"plan", "parse", "build", "smoke", "summary"}
    for job in ("parse", "build", "smoke"):
        assert WF["jobs"][job]["runs-on"] == "${{ inputs.runner || 'macos-15' }}"
    for job in ("plan", "summary"):
        assert WF["jobs"][job]["runs-on"].startswith("ubuntu")
    assert WF["permissions"] == {"contents": "read"}
    for job in ("parse", "build"):
        assert WF["jobs"][job]["strategy"]["fail-fast"] is False  # every file's errors, not the first


def test_parse_is_the_shared_parse_step_with_the_apple_kinds():
    steps = WF["jobs"]["parse"]["steps"]
    run = [s for s in steps if s.get("run", "").strip() == "bash .wf/tools/ops/native_checks_steps.sh parse-group"]
    assert len(run) == 1
    env = run[0]["env"]
    assert env["KIND"] == "${{ matrix.kind }}" and env["CLS"] == "${{ matrix.cls }}" and "APPLE_ACCEL" in env
    assert steps[-1]["if"] == "always()" and "out/*.txt" in steps[-1]["with"]["path"]
    script = (ROOT / "tools/ops/native_checks_steps.sh").read_text()
    # the legacy fast set carries no IDENTICAL define; the identical set does; a metal1 file keeps its own flag
    assert 'apple-fast) flags="$accel_flags" ;;' in script
    assert 'apple-identical) flags="-D MOJOLEARN_NUMERIC_IDENTICAL=1 $accel_flags" ;;' in script
    assert 'metal1) accel_flags="--target-accelerator metal:1" ;;' in script
    assert 'target="--target-cpu apple-m1"' in script and '-j "$cores"' in script
    # nothing in the parse step needs a tool macOS lacks without a fallback
    assert "command -v timeout" in script and "vm_stat" in script and "getconf _NPROCESSORS_ONLN" in script


def test_build_and_smoke_call_the_tool():
    build = "\n".join(s.get("run", "") for s in WF["jobs"]["build"]["steps"])
    assert "apple_fast_check.py build-group" in build and "--accel" in build
    smoke = WF["jobs"]["smoke"]
    assert "inputs.step == 'smoke'" in smoke["if"] and "!cancelled()" in smoke["if"]
    assert any("apple_fast_check_install.sh built out" in s.get("run", "") for s in smoke["steps"])
    install = (ROOT / "tools/ops/apple_fast_check_install.sh").read_text()
    for needle in ("apple_fast_check.py\" place", "stage_dylibs.py", "portable_math/stage.py", "-m build --wheel",
                   "apple_fast_check.py\" smoke", "SOURCE TREE"):
        assert needle in install, needle


def test_plan_covers_every_member_once(monkeypatch):
    monkeypatch.chdir(ROOT)
    merged = _load("merged_bindings_t", ROOT / "tools/merged_bindings.py")
    for units in (0, 3):
        rows = afc.plan_rows(["fast", "identical", "host"], units)
        assert len({r["key"] for r in rows}) == len(rows)
        for name in ("fast", "identical", "host"):
            want = sorted(n for _s, _c, mem in merged.plan(name, "darwin", True) for n in mem)
            got = sorted(n for r in rows if r["set"] == name for n in r["members"].split(","))
            assert got == want, (name, units)
            assert all(r["kind"] == afc.KIND[name] for r in rows if r["set"] == name)
        if units:
            assert all(len(r["members"].split(",")) <= units for r in rows)
    whole = afc.plan_rows(["fast", "identical"])
    assert {r["cls"] for r in whole} == {"default", "metal1"}
    assert all(r["cls"] == "host" for r in afc.plan_rows(["host"]))
    # the byte LM is in both Apple device sets (merged_build.sh's macOS default)
    for name in ("fast", "identical"):
        assert any("_mojolearn_byte_lm" in r["members"].split(",") for r in whole if r["set"] == name)


def test_smoke_cell_tells_ran_refused_crashed_and_timeout(tmp_path, monkeypatch):
    (tmp_path / "logs").mkdir()
    child = r'''
import json, os, signal, sys, time
case = sys.argv[1]
if case == "ok": print("SMOKE " + json.dumps(dict(status="ran", digest="abc")))
elif case == "no": print("SMOKE " + json.dumps(dict(status="refused", message="ValueError: nope")))
elif case == "die": os.kill(os.getpid(), signal.SIGSEGV)
elif case == "hang": time.sleep(60)
'''
    monkeypatch.setattr(afc, "SMOKE_CHILD", child)
    assert afc.smoke_cell("ok", "fast", tmp_path, 30)["status"] == "ran"
    refused = afc.smoke_cell("no", "fast", tmp_path, 30)
    assert refused["status"] == "refused" and "nope" in refused["message"]
    died = afc.smoke_cell("die", "fast", tmp_path, 30)
    assert died["status"] == "crashed" and "SIGSEGV" in died["message"]
    assert afc.smoke_cell("hang", "fast", tmp_path, 2)["status"] == "timeout"


def test_smoke_child_names_the_five_fits_and_both_modes():
    assert afc.SMOKE_MODES == ("fast", "identical")
    for needle in ("RandomForestClassifier", "Ridge(", "KMeans(", "MLPClassifier(", "TransformerBlock("):
        assert needle in afc.SMOKE_CHILD
    compile(afc.SMOKE_CHILD, "smoke_child", "exec")


def test_summary_tables_and_missing(tmp_path):
    a = tmp_path / "apple-parse-fast-a"; a.mkdir()
    (a / "result-fast-a.txt").write_text("key=fast-a stem=a kind=pass rc=0 seconds=10 host_used_peak_bytes=2147483648\n")
    b = tmp_path / "apple-parse-fast-b"; b.mkdir()
    (b / "result-fast-b.txt").write_text("key=fast-b stem=b kind=compile rc=1 seconds=7\n")
    (b / "fast-b.log").write_text("x.mojo:3:1: error: no such thing\nnote\n")
    c = tmp_path / "apple-build-fast-c"; c.mkdir()
    (c / "result-fast-c.txt").write_text("key=fast-c set=fast stem=c cls=default kind=pass rc=0 seconds=5 bytes=9 kernels=present jobs=3\n")
    planned = tmp_path / "planned.json"
    planned.write_text(json.dumps([dict(step="parse", key="fast-a"), dict(step="parse", key="fast-b"),
                                   dict(step="parse", key="fast-z"), dict(step="build", key="fast-c")]))
    out = tmp_path / "SUMMARY.md"
    assert afc.main(["summary", "--results", str(tmp_path), "--out", str(out), "--planned", str(planned),
                     "--title", "t"]) == 0
    text = out.read_text()
    assert "## parse: 1 of 2 passed" in text and "## build: 1 of 1 passed" in text
    assert "error: no such thing" in text and "MISSING parse fast-z" in text and "| fast-a | pass | 0 | 10 | 2.0 |" in text


def test_place_puts_each_set_in_its_directory(tmp_path, monkeypatch):
    monkeypatch.chdir(ROOT)
    art = tmp_path / "built"
    for key, rel in (("fast-_mojolearn_native", "pkg/_mojolearn_native.so"),
                     ("identical-_mojolearn_native", "pkg/identical/_mojolearn_native.so"),
                     ("host-_mojolearn_native_host", "pkg/host/_mojolearn_native_host.so")):
        f = art / f"apple-build-{key}" / rel
        f.parent.mkdir(parents=True)
        f.write_bytes(key.encode())
    pkg = tmp_path / "pkg-tree"
    got = subprocess.run([sys.executable, str(ROOT / "tools/ops/apple_fast_check.py"), "place", "--artifacts",
                          str(art), "--pkg", str(pkg)], capture_output=True, text=True, cwd=ROOT)
    assert got.returncode == 0, got.stderr
    assert (pkg / "_mojolearn_native.so").read_bytes() == b"fast-_mojolearn_native"
    assert (pkg / "identical/_mojolearn_native.so").read_bytes() == b"identical-_mojolearn_native"
    assert (pkg / "host/_mojolearn_native_host.so").exists()
    index = json.loads((pkg / "identical/_mojolearn_merged.json").read_text())
    assert "_mojolearn_native.so" in index["files"]
    assert "MISSING fast/_mojolearn_native_trees.so" in got.stdout
