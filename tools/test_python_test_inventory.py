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


def test_worker_count_and_serial_files(tmp_path):
    sys.path.insert(0, str(Path(__file__).parent))
    import run_python_tests as runner
    assert runner.worker_count({'MOJOLEARN_TEST_WORKERS': '1'}) == 1
    assert runner.worker_count({}) >= 1
    assert runner.user_chose_workers(['-n', '4']) and runner.user_chose_workers(['-p', 'no:xdist'])
    assert not runner.user_chose_workers(['--continue-on-collection-errors'])
    (tmp_path / 'test_a.py').write_text('def test_a(): pass\n')
    (tmp_path / 'test_b.py').write_text('import pytest\n@pytest.mark.serial\ndef test_b(): pass\n')
    assert [p.name for p in runner.serial_files(sorted(tmp_path.glob('test_*.py')))] == ['test_b.py']


def test_entrypoint_serial_override_preserves_failure(tmp_path):
    (tmp_path / 'test_failure.py').write_text('def test_failure(): assert False\n')
    run = subprocess.run([sys.executable, str(Path(__file__).with_name('run_python_tests.py')),
                          '--tests-root', str(tmp_path)], capture_output=True, text=True,
                         env=dict(__import__('os').environ, MOJOLEARN_TEST_WORKERS='1'))
    assert run.returncode == 1
    assert '1 failed' in run.stdout
