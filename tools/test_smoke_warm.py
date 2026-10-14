"""tools/smoke_warm.py: the smoke's warm process (written 2026-10-14, not run by the lane)."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import pytest

import smoke_warm as W
import gpu_gap_campaign as C

pytestmark = pytest.mark.skipif(not hasattr(os, "fork"), reason="the warm process forks")


@pytest.fixture
def server(tmp_path):
    """A warm server with no preloads (stdlib only) and its launcher."""
    import tempfile
    sock = Path(tempfile.mkdtemp(prefix="mlswt")) / "s.sock"
    launcher = W.write_launcher(tmp_path / "warm", sys.executable, sock)
    proc = subprocess.Popen([sys.executable, str(Path(W.__file__)), "serve", "--sock", str(sock),
                             "--launcher", str(launcher), "--no-preload"],
                            stdout=(tmp_path / "server.log").open("wb"), stderr=subprocess.STDOUT,
                            start_new_session=True)
    deadline = time.time() + 30
    while not Path(str(sock) + ".ready").exists() and time.time() < deadline:
        time.sleep(0.05)
    yield sock, launcher
    W.control(sock, "shutdown", timeout=10)
    proc.wait(timeout=20)


def test_parse_argv_takes_scripts_c_and_m_and_sends_other_flags_cold():
    assert W.parse_argv(["-u", "x.py", "a"]) == ("script", "x.py", ["a"], {"-u"})
    assert W.parse_argv(["-c", "print(1)", "z"]) == ("-c", "print(1)", ["z"], set())
    assert W.parse_argv(["-m", "pip", "show"])[:3] == ("-m", "pip", ["show"])
    for argv in (["-X", "dev", "x.py"], ["-I", "x.py"], [], ["-"]):
        assert W.parse_argv(argv)[0] is None


def test_launch_relays_output_exit_status_argv_cwd_and_env(server, tmp_path):
    sock, launcher = server
    code = "import os, sys; print(sys.argv, os.getcwd(), os.environ.get('SMOKE_X')); sys.exit(7)"
    env = dict(os.environ, SMOKE_X="y")
    p = subprocess.run([str(launcher), "-c", code, "a"], capture_output=True, text=True, cwd=tmp_path, env=env,
                       timeout=60)
    assert p.returncode == 7
    assert p.stdout.strip() == str(["-c", "a"]) + " " + os.path.realpath(tmp_path) + " y" or \
        p.stdout.strip() == str(["-c", "a"]) + " " + str(tmp_path) + " y"
    stats = W.control(sock, "end")
    assert stats["warm"] == 1 and stats["cold"] == []


def test_script_runs_as_main_with_its_directory_first_on_the_path(server, tmp_path):
    sock, launcher = server
    (tmp_path / "helper_mod_smoke.py").write_text("VALUE = 41\n")
    script = tmp_path / "prog.py"
    script.write_text("import sys, helper_mod_smoke\nif __name__ == '__main__':\n"
                      "    print(helper_mod_smoke.VALUE + 1, sys.argv[1:])\n    raise SystemExit(0)\n")
    p = subprocess.run([str(launcher), "-u", str(script), "q"], capture_output=True, text=True, timeout=60)
    assert p.returncode == 0 and p.stdout.strip() == "42 ['q']"


def test_unhandled_exception_exits_1_with_a_traceback(server):
    _, launcher = server
    p = subprocess.run([str(launcher), "-c", "raise RuntimeError('boom')"], capture_output=True, text=True,
                       timeout=60)
    assert p.returncode == 1 and "RuntimeError: boom" in p.stderr


def test_unsupported_flag_runs_cold_and_is_counted(server):
    sock, launcher = server
    p = subprocess.run([str(launcher), "-X", "utf8", "-c", "print('cold ok')"], capture_output=True, text=True,
                       timeout=60)
    assert p.returncode == 0 and p.stdout.strip() == "cold ok"
    stats = W.control(sock, "end")
    assert stats["warm"] == 0 and len(stats["cold"]) == 1 and "interpreter flag -X" in stats["cold"][0]


def test_nested_launch_through_sys_executable_is_warm_too(server):
    sock, launcher = server
    code = ("import subprocess, sys; "
            "sys.exit(subprocess.run([sys.executable, '-c', 'import sys; sys.exit(5)']).returncode)")
    p = subprocess.run([str(launcher), "-c", code], timeout=60)
    assert p.returncode == 5
    assert W.control(sock, "end")["warm"] == 2


def test_timeout_kills_the_recipe_and_end_reaps_what_it_left(server, tmp_path):
    sock, launcher = server
    marker = tmp_path / "late"
    code = ("import subprocess, sys, time; "
            "subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(3); open(%r, \"w\").close()'], "
            "start_new_session=True); time.sleep(30)" % str(marker))
    rc, seconds = C.run_bounded([str(launcher), "-c", code], tmp_path / "log", 1.0)
    assert rc is None and seconds < 10
    stats = W.control(sock, "end")
    assert stats["warm"] == 2
    time.sleep(4)
    assert not marker.exists()                       # the grandchild in its own session was killed too


def test_signal_death_is_relayed_as_a_signal(server):
    _, launcher = server
    p = subprocess.run([str(launcher), "-c", "import os, signal; os.kill(os.getpid(), signal.SIGTERM)"],
                       timeout=60)
    assert p.returncode == -15


def test_no_server_means_a_cold_interpreter(tmp_path):
    launcher = W.write_launcher(tmp_path / "warm", sys.executable, tmp_path / "missing.sock")
    p = subprocess.run([str(launcher), "-c", "print('still runs')"], capture_output=True, text=True, timeout=60)
    assert p.returncode == 0 and p.stdout.strip() == "still runs"


def test_launcher_prefix_keeps_the_real_lib_directory(tmp_path):
    launcher = W.write_launcher(tmp_path / "warm", sys.executable, tmp_path / "s.sock")
    real_lib = Path(os.path.dirname(os.path.dirname(os.path.abspath(sys.executable)))) / "lib"
    if real_lib.is_dir():
        assert os.path.realpath(launcher.parent.parent / "lib") == os.path.realpath(real_lib)


def test_cold_reasons_for_library_paths_and_interpreter_settings(tmp_path):
    srv = W.Server(tmp_path / "s.sock", dict(launcher="x"))
    srv.env0 = {"LD_LIBRARY_PATH": str(tmp_path)}
    srv.ld0 = {os.path.realpath(tmp_path)}
    ok = dict(env={"LD_LIBRARY_PATH": str(tmp_path), "PYTHONUNBUFFERED": "1"})
    assert srv.cold_reason(ok, "script", "x.py", set()) is None
    extra = dict(env={"LD_LIBRARY_PATH": "/opt/other/lib64"})
    assert "LD_LIBRARY_PATH adds" in srv.cold_reason(extra, "script", "x.py", set())
    seed = dict(env={"PYTHONHASHSEED": "0"})
    assert "PYTHONHASHSEED" in srv.cold_reason(seed, "script", "x.py", set())


def test_campaign_records_the_warm_report_and_each_recipe_launch_counts(tmp_path, monkeypatch):
    ids = ['algos/adam/synthetic/rows=full', 'algos/adamw/synthetic/rows=full']
    manifest = tmp_path / 'manifest.json'
    manifest.write_text(json.dumps({'amd': ids}))

    class FakeWarm:
        def __init__(self, args, out):
            self.report = dict(status='serving', import_seconds=31.5, starts=[dict(preloads={})])
            self.ended = 0

        def start(self):
            return self.report

        def serving(self):
            return True

        def python(self):
            return '/warm/bin/python'

        def end_recipe(self):
            self.ended += 1
            return dict(warm=3, cold=[], killed=0)

        def stop(self):
            pass
    monkeypatch.setattr(C, 'Warm', FakeWarm)
    monkeypatch.setattr(C, 'smoke_checks', lambda args, plans, out, warm=None: ({}, {}))
    cmds = []

    def bounded(cmd, log, timeout, env=None):
        cmds.append(cmd)
        if len(cmds) == 1:
            raise RuntimeError('harness bug')          # recorded; the loop goes on
        return None, timeout
    monkeypatch.setattr(C, 'run_bounded', bounded)
    monkeypatch.setattr(C, 'retain', lambda archive, members: {'sha256': 'x'})
    out = tmp_path / 'o'
    C.main(['--vendor', 'amd', '--manifest', str(manifest), '--out', str(out), '--cache', str(tmp_path),
            '--smoke', '--smoke-warm', 'on'])
    doc = json.loads((out / 'smoke' / 'smoke.json').read_text())
    assert doc['warm']['import_seconds'] == 31.5
    rows = {r['id']: r for r in doc['rows'].values()}
    assert rows[ids[0]]['status'] == 'FAILED' and 'harness bug' in rows[ids[0]]['problems'][0]
    assert rows[ids[1]]['status'] == 'STARTED' and rows[ids[1]]['launches']['warm'] == 3
    assert all(c[0] == '/warm/bin/python' and c[c.index('--python-env') + 1] == '/warm/bin/python' for c in cmds)
    assert 'WARM serving import_s=31.5' in (out / 'smoke' / 'smoke.txt').read_text()
