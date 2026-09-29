import subprocess
import sys
from pathlib import Path
from python_test_inventory import inventory


def test_inventory_never_imports_gate_and_covers_each_file(tmp_path):
    files = {
        'test_function.py': 'def test_something(): pass\n',
        'test_class.py': 'import unittest\nclass Something(unittest.TestCase): pass\n',
        'test_gate.py': 'import pytest\nraise SystemExit(0)\n',
        'test_direct.py': 'from unittest import TestCase\nclass Something(TestCase): pass\n',
    }
    for name, source in files.items():
        (tmp_path / name).write_text(source)
    result = inventory(tmp_path)
    assert [p.name for p in result['gates']] == ['test_gate.py']
    assert len(result['pytest']) == 3
    assert {p.name for paths in result.values() for p in paths} == set(files)


def test_entrypoint_preserves_pytest_failure(tmp_path):
    (tmp_path / 'test_failure.py').write_text('def test_failure(): assert False\n')
    (tmp_path / 'test_gate.py').write_text('raise SystemExit(0)\n')
    run = subprocess.run([sys.executable, str(Path(__file__).with_name('run_python_tests.py')),
                          '--tests-root', str(tmp_path)], capture_output=True, text=True)
    assert run.returncode == 1
    assert '1 failed' in run.stdout
    assert 'INTERNALERROR' not in run.stdout
