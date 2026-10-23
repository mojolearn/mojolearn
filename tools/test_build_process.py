"""Real inert child processes exercise build bounds; no compiler or GPU."""
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_process as guard


def run(tmp_path, code, **limits):
    return guard.run([sys.executable, '-c', code], cwd=tmp_path, env=os.environ.copy(),
                     log=tmp_path/'build.log', seconds=limits.pop('seconds', 5),
                     poll=.03, **limits)


def alive(pid):
    path = Path('/proc')/str(pid)/'stat'
    if path.exists() and path.read_text().rsplit(')', 1)[1].split()[0] == 'Z':
        return False
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False


def assert_gone(pid):
    deadline = time.monotonic()+3
    while alive(pid) and time.monotonic() < deadline:
        time.sleep(.03)
    assert not alive(pid)


@pytest.mark.parametrize('limits', [dict(seconds=float('nan')), dict(seconds=0),
    dict(stall_seconds=float('inf')), dict(rss_bytes=-1), dict(min_available_bytes=-1)])
def test_bad_limits_refuse_before_spawn(tmp_path, limits):
    with pytest.raises(ValueError):
        run(tmp_path, 'raise AssertionError("must not execute")', **limits)
    assert not (tmp_path/'build.log').exists()


def test_success_and_failure_retain_full_log_and_status(tmp_path):
    result = run(tmp_path, 'print("diagnostic"); raise SystemExit(7)')
    assert result['complete'] and result['status']=='failed' and result['exit_code']==7
    assert (tmp_path/'build.log').read_text().strip()=='diagnostic'
    assert json.loads((tmp_path/'build.log.execution.json').read_text())==result


def test_deadline_stops_busy_compiler(tmp_path):
    result = run(tmp_path, 'while True: pass', seconds=.3)
    assert result['status']=='timeout' and result['exit_code']==124
    assert_gone(result['pid'])


def test_stall_and_memory_limits_are_distinct(tmp_path):
    for policy, probe, expected in [
        (dict(stall_seconds=.1), lambda pid, seen: (0, 1, 100), 'stalled'),
        (dict(rss_bytes=50), lambda pid, seen: (1, 100, 100), 'rss-limit'),
        (dict(min_available_bytes=50), lambda pid, seen: (1, 1, 25), 'memory-pressure'),
    ]:
        result = run(tmp_path, 'import time;time.sleep(60)', probe=probe, **policy)
        assert result['status']==expected and result['exit_code']==124
        assert_gone(result['pid'])


def test_probe_failure_still_terminates_group(tmp_path):
    def broken(pid, seen):
        raise RuntimeError('probe failed')
    result = run(tmp_path, 'import time;time.sleep(60)', probe=broken)
    assert result['status']=='controller-error' and result['exit_code']==125
    assert 'probe failed' in result['error']
    assert_gone(result['pid'])


def test_successful_launcher_cannot_leave_child_running(tmp_path):
    child = 'import signal,time;signal.signal(signal.SIGTERM,signal.SIG_IGN);time.sleep(60)'
    code = ('import subprocess,sys,pathlib,time;'
            f'p=subprocess.Popen([sys.executable,"-c",{child!r}]);'
            'pathlib.Path("child.pid").write_text(str(p.pid));time.sleep(.15)')
    result = run(tmp_path, code)
    pid = int((tmp_path/'child.pid').read_text())
    try:
        assert result['exit_code']==0
        assert_gone(pid)
    finally:
        if alive(pid):
            os.kill(pid, signal.SIGKILL)


def test_cpu_sample_retains_finished_children(tmp_path, monkeypatch):
    rows = iter(['10 9 00:05 10\n11 9 00:10 20\n', '10 9 00:06 10\n12 9 00:01 30\n'])
    monkeypatch.setattr(guard.subprocess, 'check_output', lambda *a, **k: next(rows))
    monkeypatch.setattr(guard, 'available_memory', lambda: 100000)
    seen = {}
    assert guard.sample_group(9, seen)==(15, 30*1024, 100000)
    assert guard.sample_group(9, seen)==(17, 40*1024, 100000)


def test_sigterm_controller_stops_worker_group(tmp_path):
    code = ('import sys;sys.path.insert(0,'+repr(str(Path(guard.__file__).parent))+');'
            'import build_process,os,pathlib;'
            'r=build_process.run([sys.executable,"-c",'
            '"import os,pathlib,time;pathlib.Path(\"worker.pid\").write_text(str(os.getpid()));time.sleep(60)"],'
            'cwd=".",env=os.environ.copy(),log="guard.log",seconds=30,poll=.05);'
            'raise SystemExit(r["exit_code"])')
    proc = subprocess.Popen([sys.executable, '-c', code], cwd=tmp_path)
    pid = None
    try:
        deadline = time.monotonic()+5
        while not (tmp_path/'worker.pid').exists() and time.monotonic()<deadline:
            time.sleep(.03)
        pid = int((tmp_path/'worker.pid').read_text())
        proc.terminate()
        assert proc.wait(timeout=5)==130
        assert_gone(pid)
        receipt = json.loads((tmp_path/'guard.log.execution.json').read_text())
        assert receipt['status']=='interrupted' and receipt['complete']
    finally:
        if proc.poll() is None:
            proc.kill();proc.wait()
        if pid and alive(pid):
            os.kill(pid, signal.SIGKILL)
