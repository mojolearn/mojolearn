import ast
import importlib.util
import os
from pathlib import Path
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('bounded_benchmark', HERE/'bounded.py')
bounded = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bounded)


def test_timeout_kills_term_ignoring_worker_after_leader_exits(tmp_path):
    marker = tmp_path/'escaped'
    script = '''import os, signal, time
pid = os.fork()
if pid == 0:
 signal.signal(signal.SIGTERM, signal.SIG_IGN)
 time.sleep(.7)
 open(%r, 'w').write('escaped')
else:
 time.sleep(10)
''' % str(marker)
    start = time.monotonic()
    assert bounded.run([sys.executable,'-c',script], .2) == 124
    assert time.monotonic()-start < 3
    time.sleep(.7)
    assert not marker.exists()


def test_failed_command_and_successful_command_preserve_status():
    assert bounded.run([sys.executable,'-c','raise SystemExit(7)'],2) == 7
    assert bounded.run([sys.executable,'-c','print("TIME before=1 after=2")'],2) == 0


def test_timed_out_phase_records_failure_and_later_phase_continues(tmp_path):
    result=subprocess.run(['bash','-c',
        'source "$STATUS"; benchmark "$PY" -c "import time; time.sleep(3)" | tail -1; '
        'echo later > "$EV/later"; job_finish'],
        env=dict(os.environ, STATUS=str(HERE/'job_status.sh'), EV=str(tmp_path),
                 PY=sys.executable, BENCH_TIMEOUT='.1'),capture_output=True,text=True,timeout=5)
    assert result.returncode != 0
    assert (tmp_path/'later').exists()
    assert 'TIMING INCOMPLETE' in result.stderr


def test_light_shapes_bound_setup_not_just_queries(monkeypatch, capsys):
    # Execute the benchmark's actual shape selection without importing NumPy/native code.
    tree=ast.parse((HERE/'kern_bench.py').read_text())
    start=next(i for i,n in enumerate(tree.body) if isinstance(n,ast.Assign)
               and any(isinstance(t,ast.Tuple) for t in n.targets))
    stop=next(i for i,n in enumerate(tree.body) if isinstance(n,ast.Assign)
              and any(isinstance(t,ast.Name) and t.id=='Xf' for t in n.targets))
    for small in [True,False]:
        ns=dict(small=small,col='cpu',scale='cpu' if small else 'full')
        exec(compile(ast.Module(body=tree.body[start:stop],type_ignores=[]),'shapes','exec'),ns)
        assert ns['nf'] == (512 if small else 5000)
        assert ns['n'] == (512 if small else 10000)
        assert ns['nq'] == (1024 if small else 200000)
    assert 'fit_rows=512' in capsys.readouterr().out
