# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""CPU tests for tools/cross_compile_check.py: the selection, the flags, and
the PASS/FAIL/TIMEOUT/STALLED verdicts against a fake compiler. No Mojo.

    python3 -m pytest -q tools/test_cross_compile_check.py
"""
import sys
import time
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


def test_real_linux_lists_ship_svm_identical_only():
    """main ships FAST svm Apple-only on Linux (69a519c15); the gate must build
    what the Linux lists ship, not what the Mac ships."""
    rows = xcc.linux_rows()
    tiers = {t for n, _s, t in rows if n == "_mojolearn_svm"}
    assert tiers == {"identical"}
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
    assert len(lines) == 2 and all("_mojolearn_svm" in ln and "identical" in ln for ln in lines)


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
