"""The steward kills EVERY process of a job (tools/apple_steward.py _run_job).

2026-09-29: a timed-out speed job on do-amd left its grandchildren spinning on
the GPU for 7 h 46 min, because the timeout killed only the `sh -c` child. A
job now runs in its own session with a job tag in its environment; on a
timeout, and after any exit, the group and every tagged process (even one that
left the group with setsid) are killed.
"""
import os
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import apple_steward as st  # noqa: E402

# a child that leaves the job's process group and outlives its parent
ESCAPE = ("import os, time; os.setsid(); open(os.environ['PIDFILE'], 'w').write(str(os.getpid())); "
          "time.sleep(600)")


def _alive(pid):
    r = subprocess.run(["ps", "-o", "stat=", "-p", str(pid)], capture_output=True, text=True)
    return bool(r.stdout.strip()) and not r.stdout.strip().startswith("Z")


def _wait_pidfile(p):
    end = time.time() + 20
    while time.time() < end:
        if p.is_file() and p.read_text().strip():
            return int(p.read_text())
        time.sleep(0.1)
    raise AssertionError("the escaping child never started")


def _spawn(tmp, script, timeout):
    pidfile = tmp / "escaped.pid"
    env = dict(os.environ, PIDFILE=str(pidfile), PY=sys.executable)
    rc, rec = st._run_job(["sh", "-c", script], timeout, env=env,
                          stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return rc, rec, pidfile


def test_timeout_kills_group_and_escaped_child(tmp_path):
    # the escaped child starts, then the job hangs past its timeout
    script = ('"$PY" -c "%s" & sleep 1; sleep 600 & sleep 600' % ESCAPE)
    rc, rec, pidfile = _spawn(tmp_path, script, 4)
    pid = _wait_pidfile(pidfile)
    assert rc == 124
    assert rec["timed_out"] is True
    assert pid in rec["killed"]
    assert rec["survivors"] == []
    assert not _alive(pid)


def test_leftovers_after_a_normal_exit_are_killed(tmp_path):
    # the command exits 0 but leaves a setsid child behind (as the board's drivers do)
    script = '"$PY" -c "%s" & sleep 2; exit 0' % ESCAPE
    rc, rec, pidfile = _spawn(tmp_path, script, 60)
    pid = _wait_pidfile(pidfile)
    assert rc == 0
    assert rec["timed_out"] is False
    assert pid in rec["killed"]
    assert rec["survivors"] == []
    assert not _alive(pid)


def test_clean_job_records_nothing(tmp_path):
    rc, rec, _ = _spawn(tmp_path, "exit 3", 60)
    assert rc == 3
    assert rec == {}


def test_other_jobs_are_left_alone(tmp_path):
    # a process without this job's tag, in another session, survives
    other = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(600)"], start_new_session=True)
    try:
        rc, rec, _ = _spawn(tmp_path, "sleep 600", 2)
        assert rc == 124
        assert other.pid not in rec["killed"]
        assert other.poll() is None
    finally:
        other.kill()
        other.wait()


def test_a_request_stranded_in_working_gets_a_fail_verdict(tmp_path, monkeypatch):
    """A restart (or systemd stopping the service after an OOM kill) leaves the
    running request in working/; the next start gives it a FAIL verdict."""
    import json
    work, done = tmp_path / "working", tmp_path / "done"
    work.mkdir()
    done.mkdir()
    monkeypatch.setattr(st, "WORK", work)
    monkeypatch.setattr(st, "DONE", done)
    (work / "1790000000000-speed-x-abc.do-amd.json").write_text(json.dumps(
        {"name": "1790000000000-speed-x-abc", "kind": "speed", "lane": "x"}))
    (work / "1790000000001-speed-y-def.m2pro.json").write_text("{}")   # another steward's: untouched
    st._recover_stranded("do-amd")
    v = json.loads((done / "1790000000000-speed-x-abc" / "verdict.json").read_text())
    assert v["result"] == "FAIL" and v["failed_step"].startswith("interrupted")
    assert [p.name for p in work.iterdir()] == ["1790000000001-speed-y-def.m2pro.json"]
