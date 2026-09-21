# SPDX-License-Identifier: Apache-2.0
"""Reject independently introduced platform math from a wheel payload."""
import importlib.util
from pathlib import Path
import subprocess
import sys

import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from stage import build
spec = importlib.util.spec_from_file_location('math_wheel_audit', HERE / 'wheel.py')
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


@pytest.fixture
def payload(tmp_path):
    path = tmp_path / ('libMojolearnMath.dylib' if sys.platform == 'darwin' else 'libMojolearnMath.so')
    build(path)
    return tmp_path


def test_owned_library_has_no_math_imports(payload):
    assert audit.audit_tree(payload)['platform_math_free']


@pytest.mark.parametrize('name,content', [('libm.so.6', b'not an ELF'),
                                         ('module.py', b'import math\n'),
                                         ('module.py', b'from math import sqrt\n')])
def test_rejects_payload_regressions(payload, name, content):
    (payload / name).write_bytes(content)
    with pytest.raises(ValueError, match='platform math audit failed'):
        audit.audit_tree(payload)


def test_rejects_native_math_import(payload):
    source = payload / 'foreign.c'
    source.write_text('extern double sin(double); double foreign(double x) { return sin(x); }\n')
    flags = ['clang', '-dynamiclib'] if sys.platform == 'darwin' else ['cc', '-shared', '-fPIC']
    subprocess.run(flags + [str(source), '-o', str(payload / 'foreign.so')], check=True)
    with pytest.raises(ValueError, match="math imports=\\['sin'\\]"):
        audit.audit_tree(payload)


@pytest.mark.parametrize('name,content', [
    ('mojolearn/feature.py', b'import numpy as np\n'),
    ('mojolearn/feature.py', b'from numpy import asarray\n'),
    ('mojolearn/feature.py', b'__import__("numpy")\n'),
    ('mojolearn/feature.py', b'importlib.import_module("numpy.linalg")\n'),
    ('numpy/__init__.py', b''),
    ('numpy.libs/libopenblas.so', b''),
    ('mojolearn.dist-info/METADATA', b'Requires-Dist: numpy>=1.24; extra == "test"\n'),
])
def test_numpy_policy_rejects_runtime_and_payload_dependencies(tmp_path, name, content):
    path = tmp_path / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    assert audit.numpy_errors(path, name)


@pytest.mark.parametrize("relative", ["mojolearn/_identity_break.py", "mojolearn/_verify_par.py"])
def test_numpy_remains_available_to_independent_verification(tmp_path, relative):
    path = tmp_path / '_identity_break.py'
    path.write_text('import numpy as np\n')
    assert not audit.numpy_errors(path, relative)


@pytest.mark.parametrize('content', ['import scipy\n', 'import sklearn\n', 'from requests import get\n',
                                     'def f():\n    import requests\n'])
def test_dependency_policy_rejects_a_package_the_wheel_does_not_provide(tmp_path, content):
    module = tmp_path / 'mojolearn' / 'module.py'
    module.parent.mkdir()
    module.write_text(content)
    assert audit.dependency_errors(module, 'mojolearn/module.py')


@pytest.mark.parametrize('content', ['import json\nfrom . import _backend\nimport mojolearn\n',
                                     'def tags(self):\n    from sklearn.utils import Tags\n'])
def test_dependency_policy_allows_the_standard_library_and_lazy_interop(tmp_path, content):
    module = tmp_path / 'mojolearn' / 'module.py'
    module.parent.mkdir()
    module.write_text(content)
    assert audit.dependency_errors(module, 'mojolearn/module.py') == []


def test_the_shipped_package_imports_nothing_the_wheel_does_not_provide():
    root = HERE.parents[1] / 'python'
    paths = [root / 'mojolearn_diagnostics.py'] + [
        p for p in (root / 'mojolearn').rglob('*.py') if 'tests' not in p.relative_to(root).parts]
    assert paths and not [e for p in paths for e in audit.dependency_errors(p, p.relative_to(root).as_posix())]
