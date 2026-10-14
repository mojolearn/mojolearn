"""The campaign smoke's in-process worker (tools/smoke_worker.py, tools/smoke_inproc.py)."""
import json
import os
import subprocess
import sys
import textwrap
import time
from types import SimpleNamespace

import pytest

import gpu_gap_campaign as C
import smoke_inproc as SI
import smoke_worker as SW


# ---------------------------------------------------------------------------
# one recipe under its own alarm; the loop goes on
# ---------------------------------------------------------------------------

def test_the_alarm_records_a_timeout_and_the_next_recipe_still_runs(tmp_path):
    t0 = time.time()
    reply = SW.run_recipe(lambda argv: time.sleep(30), [], tmp_path / 'a.log', 0.3)
    assert reply['status'] == 'timeout' and reply['rc'] is None
    assert time.time() - t0 < 10
    assert 'TIMER FIRED' in (tmp_path / 'a.log').read_text()
    assert any('<lambda>' in f for f in reply['stopped_at'])    # how far it got
    reply = SW.run_recipe(lambda argv: 0, [], tmp_path / 'b.log', 5)
    assert reply['status'] == 'done' and reply['rc'] == 0


def test_a_broad_handler_in_a_driver_cannot_swallow_the_alarm(tmp_path):
    def stubborn(argv):
        try:
            time.sleep(30)
        except BaseException:   # a driver's `except BaseException` (the timer keeps firing)
            pass
        time.sleep(30)
    t0 = time.time()
    reply = SW.run_recipe(stubborn, [], tmp_path / 'a.log', 0.3)
    assert reply['status'] == 'timeout' and time.time() - t0 < 10


def test_an_exception_is_recorded_and_the_next_recipe_still_runs(tmp_path):
    def broken(argv):
        raise ValueError('no lr in the read-back')
    reply = SW.run_recipe(broken, [], tmp_path / 'a.log', 5)
    assert reply['status'] == 'error' and 'no lr in the read-back' in reply['error']
    assert 'Traceback' in (tmp_path / 'a.log').read_text()
    reply = SW.run_recipe(lambda argv: (_ for _ in ()).throw(SystemExit('bench_board: REFUSING')), [],
                          tmp_path / 'b.log', 5)
    assert reply['status'] == 'done' and reply['rc'] == 1
    assert 'REFUSING' in (tmp_path / 'b.log').read_text()
    reply = SW.run_recipe(lambda argv: 1, [], tmp_path / 'c.log', 5)
    assert reply['status'] == 'done' and reply['rc'] == 1


def test_a_recipe_output_goes_to_its_log_and_the_alarm_is_disarmed_after_it(tmp_path):
    def loud(argv):
        print('bench_board: params probe', argv)
        os.write(2, b'from C\n')
        return 0
    reply = SW.run_recipe(loud, ['x'], tmp_path / 'a.log', 0.5)
    text = (tmp_path / 'a.log').read_text()
    assert reply['status'] == 'done' and "bench_board: params probe ['x']" in text and 'from C' in text
    time.sleep(0.8)          # no stray alarm after the recipe finished


def test_processes_a_timed_out_recipe_left_behind_are_killed(tmp_path):
    started = []

    def spawner(argv):
        p = subprocess.Popen(['sleep', '60'], start_new_session=True)
        started.append(p)
        time.sleep(30)
    reply = SW.run_recipe(spawner, [], tmp_path / 'a.log', 0.3)
    assert reply['status'] == 'timeout'
    assert started[0].pid in reply['children_killed']
    assert started[0].wait(timeout=10) is not None


# ---------------------------------------------------------------------------
# the outer watchdog
# ---------------------------------------------------------------------------

FAKE_WORKER = textwrap.dedent('''
    import json, os, sys, time
    proto = os.fdopen(os.dup(1), "w", buffering=1)
    proto.write(json.dumps({"event": "ready", "report": {"import_seconds": 0.1, "gpu_init_seconds": 0.0,
                                                         "imports": {"json": {"status": "ok"}}}}) + "\\n")
    for line in sys.stdin:
        req = json.loads(line)
        if req["op"] == "quit":
            break
        if req["op"] == "warm-box":
            proto.write(json.dumps({"event": "warm-box", "facts": {"status": "done"}}) + "\\n")
            continue
        if "hang" in req.get("argv", []):
            time.sleep(600)       # a C call that never returns to Python
        proto.write(json.dumps({"event": "recipe", "key": req.get("key"), "status": "done", "rc": 0}) + "\\n")
''')


def _worker(tmp_path, grace=0.5):
    script = tmp_path / 'fake_worker.py'
    script.write_text(FAKE_WORKER)
    args = SimpleNamespace(smoke_load_timeout=60.0, smoke_grace=grace, python_env=sys.executable, vendor='amd')
    return C.SmokeWorker(args, tmp_path, dict(os.environ), worker_cmd=[sys.executable, str(script)])


def test_a_hung_worker_is_killed_and_restarted_and_the_next_recipe_runs(tmp_path):
    w = _worker(tmp_path)
    try:
        rep = w.start()
        assert rep['status'] == 'serving' and rep['import_seconds'] == 0.1
        assert w.warm({'op': 'warm-box', 'argv': []}, 10) == {'status': 'done'}
        first = w.proc.pid
        t0 = time.time()
        reply, restart = w.recipe('identical|a', ['hang'], tmp_path / 'a.log', 0.2)
        assert reply is None and time.time() - t0 < 30
        assert restart['after'] == 'identical|a' and restart['started'] and 'killed' in restart['reason']
        assert w.alive() and w.proc.pid != first
        assert len(w.report['starts']) == 2 and len(w.report['restarts']) == 1   # load cost recorded again
        assert w.report['starts'][-1]['box_facts']['status'] == 'done'           # box facts read again
        reply, restart = w.recipe('identical|b', ['ok'], tmp_path / 'b.log', 0.2)
        assert reply['status'] == 'done' and restart is None
    finally:
        w.stop()
    assert not w.alive()


def test_a_worker_that_cannot_load_is_recorded(tmp_path):
    script = tmp_path / 'dead.py'
    script.write_text('import sys; sys.exit(3)\n')
    args = SimpleNamespace(smoke_load_timeout=20.0, smoke_grace=0.5, python_env=sys.executable, vendor='amd')
    w = C.SmokeWorker(args, tmp_path, dict(os.environ), worker_cmd=[sys.executable, str(script)])
    rep = w.start()
    assert rep['status'] == 'failed' and 'no load report' in rep['starts'][-1]['problem']
    assert C.worker_check(rep)['status'] == 'failed'


# ---------------------------------------------------------------------------
# the import report
# ---------------------------------------------------------------------------

def test_import_classification_absent_installed_but_broken_and_ok(tmp_path, monkeypatch):
    pkg = tmp_path / 'brokenpkg_smoke'
    pkg.mkdir()
    (pkg / '__init__.py').write_text('import no_such_dependency_for_smoke\n')
    monkeypatch.syspath_prepend(str(tmp_path))
    assert SW.import_one('json', 'library')['status'] == 'ok'
    assert SW.import_one('no_such_library_for_smoke', 'library')['status'] == 'absent'
    assert SW.import_one('no_such_library_for_smoke.sub', 'library')['status'] == 'absent'
    broken = SW.import_one('brokenpkg_smoke', 'library')
    assert broken['status'] == 'failed' and 'no_such_dependency_for_smoke' in broken['error']
    assert SW.import_one('no_such_harness_module', 'module')['status'] == 'failed'   # ours is never absent
    assert SW.import_one(str(tmp_path / 'missing_driver.py'), 'driver',
                         importer=SI.driver_module)['status'] == 'failed'


def test_preload_records_every_import_and_the_total(tmp_path):
    driver = tmp_path / 'tiny_driver.py'
    driver.write_text('X = 1\n')
    report = SW.preload({}, libraries=('json', 'no_such_library_for_smoke'), modules=('smoke_inproc',),
                        drivers=(driver,))
    imp = report['imports']
    assert imp['json']['status'] == 'ok' and imp['no_such_library_for_smoke']['status'] == 'absent'
    assert imp['smoke_inproc']['status'] == 'ok' and imp['tiny_driver.py']['status'] == 'ok'
    assert 'mojolearn' in imp and isinstance(report['import_seconds'], float)
    assert all('seconds' in r for r in imp.values())


def test_gpu_init_is_recorded_separately_and_never_raises(monkeypatch):
    class BadTorch:
        class cuda:
            @staticmethod
            def is_available():
                return False
    monkeypatch.setitem(sys.modules, 'torch', BadTorch)
    monkeypatch.delitem(sys.modules, 'cupy', raising=False)
    report = SW.gpu_init({}, 'nvidia')
    assert report['gpu_init']['torch']['status'] == 'failed'
    assert report['gpu_init']['cupy']['status'] == 'absent'
    assert 'gpu_init_seconds' in report


# ---------------------------------------------------------------------------
# the in-process seams
# ---------------------------------------------------------------------------

def test_in_process_is_true_only_in_the_marked_process(monkeypatch):
    monkeypatch.setenv(SI.ENV_FLAG, str(os.getpid() + 1))
    assert not SI.in_process()
    monkeypatch.setenv(SI.ENV_FLAG, str(os.getpid()))
    assert SI.in_process()
    out = subprocess.run([sys.executable, '-c', 'import os, sys; sys.path.insert(0, %r); import smoke_inproc; '
                          'print(smoke_inproc.in_process())' % os.path.dirname(SI.__file__)],
                         capture_output=True, text=True, env=dict(os.environ)).stdout.strip()
    assert out == 'False'      # a child inherits the variable, not the role


def test_scoped_env_applies_exactly_records_and_restores(monkeypatch):
    monkeypatch.setenv('SMOKE_KEEP', 'a')
    monkeypatch.setenv('SMOKE_GONE', 'b')
    monkeypatch.setenv(SI.ENV_FLAG, str(os.getpid()))
    target = {k: v for k, v in os.environ.items() if k != 'SMOKE_GONE'}
    target.update(SMOKE_NEW='c', OMP_NUM_THREADS='7')
    target.pop(SI.ENV_FLAG)
    rec = {}
    with SI.scoped_env(target, rec):
        assert os.environ['SMOKE_NEW'] == 'c' and 'SMOKE_GONE' not in os.environ
        assert os.environ[SI.ENV_FLAG] == str(os.getpid())   # the worker keeps its role
        os.environ['SMOKE_ADDED_BY_CALL'] = 'x'
    assert os.environ['SMOKE_GONE'] == 'b' and 'SMOKE_NEW' not in os.environ
    assert 'SMOKE_ADDED_BY_CALL' not in os.environ
    assert 'SMOKE_NEW' in rec['applied'] and 'SMOKE_GONE' in rec['unset']
    assert rec['load_time_only'] == ['OMP_NUM_THREADS'] and rec['process_start'] == []


def test_scoped_env_switches_mojolearn_numeric_mode_in_process_and_back(monkeypatch):
    calls = []

    class Backend:
        mode = 'identical'

        @staticmethod
        def default_mode():
            return Backend.mode

    def set_numeric_mode(mode):
        prev, Backend.mode = Backend.mode, mode
        calls.append(mode)
        return prev
    monkeypatch.setitem(sys.modules, 'mojolearn', SimpleNamespace(_backend=Backend, set_numeric_mode=set_numeric_mode))
    rec = {}
    with SI.scoped_env(dict(os.environ, MOJOLEARN_NUMERIC_MODE='fast'), rec):
        assert Backend.mode == 'fast'
    assert calls == ['fast', 'identical'] and Backend.mode == 'identical'
    assert rec['numeric_mode']['via'] == 'set_numeric_mode' and rec['numeric_mode']['requested'] == 'fast'


def test_process_start_settings_need_another_process(monkeypatch):
    env = dict(os.environ, LD_PRELOAD='/x/libfoo.so')
    assert SI.env_difference(env)['process_start'] == ['LD_PRELOAD']
    assert not SI.can_run_here([sys.executable, '-u', 'drv.py', 'race'], env)
    assert SI.can_run_here([sys.executable, '-u', 'drv.py', 'race'], dict(os.environ))
    assert not SI.can_run_here(['pixi', 'run', 'python3', 'drv.py'], dict(os.environ))


FAKE_DRIVER = textwrap.dedent('''
    import argparse, json, os, sys

    def build_parser():
        ap = argparse.ArgumentParser()
        sub = ap.add_subparsers(dest="cmd")
        w = sub.add_parser("worker")
        w.add_argument("--arm")
        r = sub.add_parser("race")
        r.add_argument("--code", type=int, default=0)
        return ap

    def worker_ready(args):
        print("constructing", args.arm)
        return {"event": "ready", "pid": os.getpid(), "params": {"seed": 7, "lr": 0.01, "arm": args.arm},
                "info": {"library": "fake", "version": "1.0",
                         "mode": os.environ.get("MOJOLEARN_NUMERIC_MODE")}}

    def probe_ready(argv):
        return worker_ready(build_parser().parse_args(argv))

    def main(argv=None):
        args = build_parser().parse_args(argv)
        if args.cmd == "worker":
            proto = os.fdopen(os.dup(1), "w", buffering=1)
            os.dup2(2, 1)
            proto.write(json.dumps(worker_ready(args)) + "\\n")
            return 0
        print("race env", os.environ.get("SMOKE_DRIVER_ENV"))
        raise SystemExit(args.code)

    if __name__ == "__main__":
        sys.exit(main())
''')


def test_the_in_process_arm_record_equals_the_worker_subprocess_record(tmp_path, monkeypatch):
    driver = tmp_path / 'fake_driver.py'
    driver.write_text(FAKE_DRIVER)
    monkeypatch.setenv(SI.ENV_FLAG, str(os.getpid()))
    env = dict(os.environ, MOJOLEARN_NUMERIC_MODE='identical')
    cmd = [sys.executable, str(driver), 'worker', '--arm', 'torch-gpu']
    sub = subprocess.run(cmd, capture_output=True, text=True, env=env)
    want = json.loads(sub.stdout.strip().splitlines()[-1])
    w = SI.start_worker('torch-gpu', cmd, env, str(tmp_path / 'arm.log'), str(tmp_path), True,
                        lambda *a: pytest.fail('a subprocess worker was started in process'))
    assert isinstance(w, SI.InProcessWorker)
    got = w.read(600)
    assert got['params'] == want['params'] and got['event'] == 'ready'
    proc = got['info'].pop('probe_process')
    assert got['info'] == want['info']
    assert proc['process'].startswith('in-process') and 'numeric_mode' in proc
    assert 'constructing torch-gpu' in (tmp_path / 'arm.log').read_text()
    w.kill('params_only', None)
    w.log.close()
    with pytest.raises(RuntimeError):
        SI.start_worker('torch-gpu', cmd, env, str(tmp_path / 'arm2.log'), str(tmp_path), False, None)


def test_outside_the_worker_the_subprocess_worker_is_used(tmp_path, monkeypatch):
    monkeypatch.delenv(SI.ENV_FLAG, raising=False)
    made = []
    SI.start_worker('a', ['py', 'x.py'], {}, 'l', '.', True, lambda *a: made.append(a) or 'subprocess')
    assert made and made[0][0] == 'a'


def test_run_driver_calls_main_under_the_driver_environment_and_returns_its_code(tmp_path, monkeypatch):
    driver = tmp_path / 'fake_driver2.py'
    driver.write_text(FAKE_DRIVER)
    monkeypatch.delenv('SMOKE_DRIVER_ENV', raising=False)
    log = tmp_path / 'probe.log'
    rc, rec = SI.run_driver([sys.executable, '-u', str(driver), 'race', '--code', '3'],
                            dict(os.environ, SMOKE_DRIVER_ENV='on'), str(log), cwd=str(tmp_path))
    assert rc == 3 and 'race env on' in log.read_text()
    assert 'SMOKE_DRIVER_ENV' not in os.environ and rec['process'].startswith('in-process')
    rc, _ = SI.run_driver([sys.executable, str(driver), 'race'], dict(os.environ), str(log), cwd=str(tmp_path))
    assert rc == 0


def test_box_facts_are_read_once_per_worker(monkeypatch):
    SI.clear_once()
    n = []
    assert SI.once(('k',), lambda: n.append(1) or 5) == 5
    assert SI.once(('k',), lambda: n.append(1) or 6) == 5
    assert n == [1]
    SI.clear_once()


def test_torch_switches_a_recipe_flipped_are_put_back(monkeypatch):
    # smoke-fixes (2026-10-14): a neural recipe imports mamba/corpus/gen_corpus.py, which turns on
    # torch.use_deterministic_algorithms(True) and one thread at import; in the one worker process that
    # leaked into the gcn torch arms on MPS (scatter_reduce_mps refuses deterministic mode)
    state = dict(det=False, warn=False, threads=8)

    def use(on, warn_only=False):
        state.update(det=on, warn=warn_only)
    torch = SimpleNamespace(are_deterministic_algorithms_enabled=lambda: state['det'],
                            is_deterministic_algorithms_warn_only_enabled=lambda: state['warn'],
                            get_num_threads=lambda: state['threads'], use_deterministic_algorithms=use,
                            set_num_threads=lambda n: state.update(threads=n))
    monkeypatch.setitem(sys.modules, 'torch', torch)
    saved = SW.torch_switches()
    assert SW.restore_torch_switches(saved) == {}
    use(True)
    state['threads'] = 1                                       # what gen_corpus.py does at import
    assert SW.restore_torch_switches(saved) == {'deterministic_algorithms': [True, False], 'num_threads': [1, 8]}
    assert state == dict(det=False, warn=False, threads=8)
