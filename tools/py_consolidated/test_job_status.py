"""No native imports, compilation, hardware, or external jobs."""
import ast
import os
from pathlib import Path
import subprocess
import types
import sys
import pytest

HERE = Path(__file__).resolve().parent

@pytest.mark.parametrize('body', [
    'false | tail -1',
    '(exit 7) | tee "$EV/output" | tail -1',
    'phase() { false; echo later-phase; }; phase',
    '(false; echo subshell-continued)',
])
def test_failed_phase_survives_filters_and_later_success(tmp_path, body):
    result = subprocess.run(['bash', '-c',
        'source "$STATUS"; ' + body + '; echo finished > "$EV/later"; job_finish'],
        env=dict(os.environ, EV=str(tmp_path), STATUS=str(HERE/'job_status.sh')),
        capture_output=True, text=True)
    assert result.returncode != 0
    assert (tmp_path/'later').read_text().strip() == 'finished'
    assert (tmp_path/'job-failures.tsv').stat().st_size
    assert 'JOB FAILED/INCOMPLETE' in result.stdout


def test_successful_timing_values_do_not_fail(tmp_path):
    result = subprocess.run(['bash', '-c',
        'source "$STATUS"; printf "TIME before 1.0\\nTIME after 2.0\\n" | tail -2; job_finish'],
        env=dict(os.environ, EV=str(tmp_path), STATUS=str(HERE/'job_status.sh')),
        capture_output=True, text=True)
    assert result.returncode == 0
    assert 'JOB PASS' in result.stdout


def test_missing_timing_binding_is_incomplete(monkeypatch, capsys):
    tree = ast.parse((HERE/'timing.py').read_text())
    main = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'main')
    import argparse
    def missing(_):
        raise ImportError('binding missing')
    ns = dict(argparse=argparse, CASES={'missing': missing}, COL='cpu')
    monkeypatch.setitem(sys.modules, 'mojolearn', types.ModuleType('mojolearn'))
    monkeypatch.setattr(sys, 'argv', ['timing.py'])
    exec(compile(ast.Module(body=[main], type_ignores=[]), 'timing.py', 'exec'), ns)
    assert ns['main']() == 2
    assert 'ERROR ImportError: binding missing' in capsys.readouterr().out

@pytest.mark.parametrize('verdict', ['AGREE', 'DISAGREE', 'KNOWN REFUSAL', 'ERROR'])
def test_arms_exit_matches_required_lane_outcome(tmp_path, monkeypatch, verdict):
    import concurrent.futures as cf
    import json
    import time
    class Alc(types.ModuleType):
        ROOT = tmp_path
        def ensure_portable_math(self, *a): pass
        def needed_bindings(self, lanes): return {l: ['binding'] for l in lanes}
        def ensure_built(self, *a, **k): pass
        def load_harness(self): return object()
        def gpu_backend(self): return 'test'
        def run_arm(self, *a, **k): return 0
        def compare(self, *a): return verdict, ''
    monkeypatch.setitem(sys.modules, 'algos_lane_check', Alc('algos_lane_check'))
    monkeypatch.setenv('NO_PROBE', '1')
    tree = ast.parse((HERE/'check.py').read_text())
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'arms')
    ns = dict(Path=Path, sys=sys, os=os, time=time, cf=cf, json=json, now=lambda:'now', single_device_plan=lambda tree, lanes: (lanes, {}))
    exec(compile(ast.Module(body=[fn], type_ignores=[]), 'check.py', 'exec'), ns)
    assert ns['arms'](tmp_path, tmp_path/'out', ['lane']) == (0 if verdict == 'AGREE' else 1)
