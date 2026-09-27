# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""CPU tests for tools/cross_compile_check.py: the selection, the flags, and
the PASS/FAIL/TIMEOUT/STALLED verdicts against a fake compiler. No Mojo.

    python3 -m pytest -q tools/test_cross_compile_check.py
"""
import sys
import time
import json as json_mod
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
import cross_compile_check as xcc  # noqa: E402

PY = sys.executable


def facts(**closures):
    return {"closures": {k: (None if v is None else dict(scope="closure", digest=v, sources=["a.mojo"]))
                         for k, v in closures.items()}}


# ---------------------------------------------------------------- selection
def test_changed_scripts_new_changed_unchanged():
    ref = facts(**{"build_a.sh": "1", "build_b.sh": "2", "build_gone.sh": "3"})
    head = facts(**{"build_a.sh": "1", "build_b.sh": "9", "build_new.sh": "4", "build_gone.sh": None})
    got = xcc.changed_scripts(ref, head)
    assert set(got) == {"build_b.sh", "build_new.sh"}
    assert got["build_new.sh"] == "new at head"


def test_a_scope_change_alone_is_a_change():
    ref = facts(**{"build_a.sh": "1"})
    head = facts(**{"build_a.sh": "1"})
    head["closures"]["build_a.sh"]["scope"] = "tree"
    assert "build_a.sh" in xcc.changed_scripts(ref, head)


ROWS = [("_m_a", "build_a.sh", "fast"), ("_m_a", "build_a.sh", "identical"),
        ("_m_b", "build_b.sh", "identical"),
        ("_m_c", "build_c.sh", "fast"), ("_m_c", "build_c.sh", "deterministic"),
        ("_m_c", "build_c.sh", "identical")]


def test_jobs_follow_the_tiers_each_binding_ships_in():
    jobs = xcc.plan_jobs({"build_a.sh", "build_b.sh"}, ROWS, ["sm_89"])
    assert jobs == [("_m_a", "build_a.sh", "fast", "sm_89"), ("_m_a", "build_a.sh", "identical", "sm_89"),
                    ("_m_b", "build_b.sh", "identical", "sm_89")]


def test_limit_caps_bindings_not_jobs_and_archs_multiply():
    jobs = xcc.plan_jobs({"build_a.sh", "build_b.sh", "build_c.sh"}, ROWS, ["sm_89", "gfx942"], limit=1)
    assert {j[0] for j in jobs} == {"_m_a"}
    assert len(jobs) == 4


def test_only_and_tiers_filter():
    jobs = xcc.plan_jobs({"build_a.sh", "build_c.sh"}, ROWS, ["gfx942"], tiers={"fast"}, only={"_m_c"})
    assert jobs == [("_m_c", "build_c.sh", "fast", "gfx942")]


def test_unchanged_selects_nothing():
    assert xcc.plan_jobs(set(), ROWS, ["sm_89"]) == []


def test_real_linux_lists_ship_svm_fast_and_identical():
    """FAST svm ships on Linux again (d2270a3f9 reverted the Apple-only
    69a519c15); the gate builds what the Linux lists ship."""
    rows = xcc.linux_rows()
    tiers = {t for n, _s, t in rows if n == "_mojolearn_svm"}
    assert tiers == {"fast", "identical"}
    assert all(not s.endswith("_host.sh") for _n, s, _t in rows)


def test_compile_cmd_flags():
    cmd = xcc.compile_cmd(["mojo"], "bindings/_x.mojo", [".", "bindings"], "identical", "gfx942", "/o/x.so")
    s = " ".join(cmd)
    assert "--target-accelerator gfx942" in s and "-j 1" in s and "--emit shared-lib" in s
    assert "-D MOJOLEARN_NUMERIC_IDENTICAL=1" in s and "-D MOJOLEARN_COLUMN_AMD" in s
    fast = " ".join(xcc.compile_cmd(["mojo"], "b.mojo", ["."], "fast", "sm_89", "/o"))
    assert "MOJOLEARN_NUMERIC" not in fast and "-D MOJOLEARN_COLUMN_NVIDIA" in fast
    det = " ".join(xcc.compile_cmd(["mojo"], "b.mojo", ["."], "deterministic", "sm_90a", "/o"))
    assert "-D MOJOLEARN_NUMERIC_DETERMINISTIC=1" in det


# ---------------------------------------------------------------- verdicts
FAKE = r"""
import sys, time, pathlib
mode = sys.argv[1]
out = sys.argv[sys.argv.index("-o") + 1] if "-o" in sys.argv else None
if mode == "pass":
    pathlib.Path(out).write_bytes(b"so")
elif mode == "fail":
    print("x.mojo:1:1: error: nope"); sys.exit(1)
elif mode == "noout":
    sys.exit(0)
elif mode == "sleep":
    time.sleep(60)
elif mode == "spin":
    t = time.time()
    while time.time() - t < 60:
        pass
"""


def fake(tmp_path, mode):
    f = tmp_path / "fake_mojo.py"
    f.write_text(FAKE)
    out = tmp_path / "x.so"
    return [PY, str(f), mode, "-o", str(out)], out


def test_pass(tmp_path):
    cmd, out = fake(tmp_path, "pass")
    v, secs, _ = xcc.run_one(cmd, out, timeout=30, stall=0, poll=0.05)
    assert v == "PASS"


def test_fail_names_the_error(tmp_path):
    cmd, out = fake(tmp_path, "fail")
    v, _, tail = xcc.run_one(cmd, out, timeout=30, stall=0, poll=0.05)
    assert v == "FAIL" and any("error: nope" in ln for ln in tail)


def test_exit_zero_without_the_library_is_a_fail(tmp_path):
    cmd, out = fake(tmp_path, "noout")
    assert xcc.run_one(cmd, out, timeout=30, stall=0, poll=0.05)[0] == "FAIL"


def test_timeout_kills_a_busy_compiler(tmp_path):
    cmd, out = fake(tmp_path, "spin")
    t0 = time.monotonic()
    v, secs, _ = xcc.run_one(cmd, out, timeout=1.0, stall=0, poll=0.05)
    assert v == "TIMEOUT" and time.monotonic() - t0 < 10


def test_a_busy_compiler_is_not_stalled(tmp_path):
    cmd, out = fake(tmp_path, "spin")
    v, _, _ = xcc.run_one(cmd, out, timeout=2.5, stall=1.5, poll=0.1)
    assert v == "TIMEOUT"


def test_stalled_compiler_is_called_early(tmp_path):
    cmd, out = fake(tmp_path, "sleep")
    t0 = time.monotonic()
    v, _, tail = xcc.run_one(cmd, out, timeout=50, stall=1.0, poll=0.1)
    assert v == "STALLED" and time.monotonic() - t0 < 20 and "flat" in tail[0]


def test_stall_logic_with_a_mocked_clock(tmp_path):
    """The CPU probe and the clock are injectable: flat CPU for `stall`
    seconds is STALLED, CPU that moves is not."""
    cmd, out = fake(tmp_path, "sleep")
    now = [0.0]
    cpu = iter([1.0, 2.0, 3.0] + [3.0] * 100)
    v, _, _ = xcc.run_one(cmd, out, timeout=1000, stall=10, poll=0,
                          cpu_probe=lambda pid: next(cpu), clock=lambda: now[0],
                          sleep=lambda s: now.__setitem__(0, now[0] + 4))
    assert v == "STALLED" and now[0] <= 30


# ---------------------------------------------------------------- end to end, mocked compiler
def _patch(monkeypatch, changed=("build_svm.sh",), plan=True):
    import release_reuse
    import bincache
    ref = {"closures": {}}
    head = {"closures": {}}
    for _n, s, _t in xcc.linux_rows():
        ref["closures"][s] = dict(scope="closure", digest="same", sources=[])
        head["closures"][s] = dict(scope="closure", digest="new" if s in changed else "same", sources=[])
    monkeypatch.setattr(release_reuse, "facts_for_commit", lambda c, cache: ref if c == "REF" else head)
    if not plan:
        monkeypatch.setattr(bincache, "script_plan", lambda *a, **k: None)


def test_main_runs_the_changed_binding_once_per_arch_and_tier(tmp_path, monkeypatch, capsys):
    _patch(monkeypatch)
    f = tmp_path / "fake_mojo.py"
    f.write_text(FAKE)
    rc = xcc.main(["--ref", "REF", "--archs", "sm_89,gfx942", "--mojo", f"{PY} {f} pass", "--stall", "0",
                   "--cache", str(tmp_path)])
    out = capsys.readouterr().out
    assert rc == 0, out
    lines = [ln for ln in out.splitlines() if ln.strip().startswith("PASS")]
    assert len(lines) == 4 and all("_mojolearn_svm" in ln for ln in lines)
    assert sorted(ln.split()[2] + " " + ln.split()[3] for ln in lines) == [
        "fast gfx942", "fast sm_89", "identical gfx942", "identical sm_89"]


def test_main_exits_nonzero_on_a_timeout(tmp_path, monkeypatch, capsys):
    _patch(monkeypatch)
    f = tmp_path / "fake_mojo.py"
    f.write_text(FAKE)
    rc = xcc.main(["--ref", "REF", "--archs", "sm_89", "--mojo", f"{PY} {f} spin", "--stall", "0",
                   "--timeout", "1", "--cache", str(tmp_path)])
    assert rc == 1 and "TIMEOUT" in capsys.readouterr().out


def test_main_refuses_an_unreadable_compile_line(tmp_path, monkeypatch, capsys):
    _patch(monkeypatch, plan=False)
    rc = xcc.main(["--ref", "REF", "--archs", "sm_89", "--mojo", "false", "--cache", str(tmp_path)])
    assert rc == 2 and "REFUSED" in capsys.readouterr().out


def test_main_with_nothing_changed_compiles_nothing(tmp_path, monkeypatch, capsys):
    _patch(monkeypatch, changed=())
    rc = xcc.main(["--ref", "REF", "--mojo", "false", "--cache", str(tmp_path)])
    assert rc == 0 and "nothing to cross-compile" in capsys.readouterr().out


def test_list_json_and_summarize_name_every_non_pass(tmp_path):
    jobs = [("_mojolearn_gp", "build_gp.sh", "fast", "gfx942"),
            ("_mojolearn_gp", "build_gp.sh", "identical", "gfx942"),
            ("_mojolearn_svm", "build_svm.sh", "fast", "sm_89")]
    plan = tmp_path / "plan.json"
    inc = xcc.write_matrix(plan, "v0.8.22", jobs)
    assert [j["key"] for j in inc] == ["_mojolearn_gp-fast-gfx942", "_mojolearn_gp-identical-gfx942",
                                       "_mojolearn_svm-fast-sm_89"]
    res = tmp_path / "artifacts"
    for key, verdict in (("_mojolearn_gp-fast-gfx942", "PASS"), ("_mojolearn_gp-identical-gfx942", "FAIL")):
        d = res / ("xcc-" + key)
        d.mkdir(parents=True)
        name, tier, arch = key.rsplit("-", 2)
        (d / "result.json").write_text(json_mod.dumps(dict(results=[dict(
            name=name, tier=tier, arch=arch, verdict=verdict, seconds=12.0, detail=["error: boom"])])))
        (d / "time.txt").write_text("\tMaximum resident set size (kbytes): 2097152\n")
    md = tmp_path / "summary.md"
    assert xcc.summarize(str(plan), str(res), str(md)) == 1
    text = md.read_text()
    assert "1 of 3 PASS" in text
    assert "_mojolearn_gp identical gfx942 FAIL" in text
    assert "_mojolearn_svm fast sm_89 MISSING" in text
    assert "| 2048 |" in text
    for d in res.iterdir():
        doc = json_mod.loads((d / "result.json").read_text())
        doc["results"][0]["verdict"] = "PASS"
        (d / "result.json").write_text(json_mod.dumps(doc))
    assert xcc.summarize(str(plan), str(res)) == 1, "a planned job with no result is never a pass"
    xcc.write_matrix(plan, "v0.8.22", jobs[:2])
    assert xcc.summarize(str(plan), str(res)) == 0
