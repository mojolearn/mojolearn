# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""CPU tests of tools/compile_watch.py, the one stall rule of tools/cross_compile_check.py and of every release
build leg (packaging/linux/binding_timeout.sh). No compiler.

    python3 -m pytest -q tools/test_compile_watch.py
"""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
import compile_watch as cw  # noqa: E402


def test_flat_cpu_and_no_output_for_the_window_is_stalled():
    sc = cw.StallClock(10, 0.0)
    assert not sc.observe(0.0, 1.0, 0)
    assert not sc.observe(5.0, 1.2, 0)
    assert sc.observe(10.0, 1.3, 0)
    assert "flat at 1 s for 10 s" in sc.describe()


def test_cpu_that_moves_or_output_that_grows_is_not_stalled():
    sc = cw.StallClock(10, 0.0)
    for t in range(0, 100, 4):
        assert not sc.observe(float(t), float(t), 0)
    sc = cw.StallClock(10, 0.0)
    for t in range(0, 100, 4):
        assert not sc.observe(float(t), 1.0, t)  # the log grows, the CPU does not
    sc = cw.StallClock(10, 0.0)
    assert not sc.observe(0.0, 5.0, 0)
    assert not sc.observe(8.0, 2.0, 0), "a finished child lowers the sum: a new phase, not a stall"
    assert not sc.observe(16.0, 2.0, 0)
    assert sc.observe(18.0, 2.0, 0)


def test_stall_zero_and_an_unreadable_probe_never_stall():
    sc = cw.StallClock(0, 0.0)
    assert not any(sc.observe(float(t), 1.0, 0) for t in range(0, 1000, 50))
    sc = cw.StallClock(10, 0.0)
    assert not any(sc.observe(float(t), None, 0) for t in range(0, 1000, 50))


def test_first_error_lines():
    text = "a\nx.mojo:1:2: error: one\nb\nerror: two\n" + "\n".join(f"l{i}" for i in range(20))
    assert cw.first_error_lines(text) == ["x.mojo:1:2: error: one", "error: two"]
    assert cw.first_error_lines("\n".join(f"l{i}" for i in range(20))) == [f"l{i}" for i in range(14, 20)]


def _stat(pid, ppid, utime, stime=0, cutime=0, cstime=0, comm="sh (x)"):
    fields = ["S", ppid, ppid, ppid, 0, -1, 0, 0, 0, 0, 0, utime, stime, cutime, cstime, 20, 0, 1]
    return f"{pid} ({comm}) " + " ".join(str(f) for f in fields) + "\n"


def test_tree_cpu_sums_the_process_tree_only(tmp_path):
    tick = os.sysconf("SC_CLK_TCK")
    rows = {10: (1, 1 * tick, 0), 11: (10, 2 * tick, tick), 12: (11, 3 * tick, 0), 99: (1, 50 * tick, 0)}
    for pid, (ppid, u, s) in rows.items():
        (tmp_path / str(pid)).mkdir()
        (tmp_path / str(pid) / "stat").write_text(_stat(pid, ppid, u, s))
    (tmp_path / "self").mkdir()
    assert cw.tree_cpu_seconds(10, proc=str(tmp_path)) == 7.0
    assert cw.tree_cpu_seconds(12, proc=str(tmp_path)) == 3.0
    assert cw.tree_cpu_seconds(1234, proc=str(tmp_path)) is None


def test_watch_marks_a_stall_and_returns_when_the_build_ends(tmp_path):
    log, marker = tmp_path / "b.log", tmp_path / "b.log.stalled"
    log.write_text("start\n")
    now = [0.0]
    rc = cw.watch(1, log, 10, marker, poll=0, clock=lambda: now[0],
                  sleep=lambda s: now.__setitem__(0, now[0] + 4), cpu_probe=lambda pid: 3.0, is_alive=lambda p: True)
    assert rc == 3 and "flat at 3 s" in marker.read_text()
    marker.unlink()
    alive = iter([True, True, False])
    rc = cw.watch(1, log, 10, marker, poll=0, clock=lambda: 0.0, sleep=lambda s: None,
                  cpu_probe=lambda pid: 3.0, is_alive=lambda p: next(alive))
    assert rc == 0 and not marker.exists()


def test_cross_compile_check_uses_this_rule():
    text = (ROOT / "tools" / "cross_compile_check.py").read_text()
    assert "compile_watch.StallClock(" in text and "compile_watch.first_error_lines(" in text
    shell = (ROOT / "packaging" / "linux" / "binding_timeout.sh").read_text()
    assert "tools/compile_watch.py" in shell
