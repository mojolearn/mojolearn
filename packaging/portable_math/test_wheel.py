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
    ('mojolearn.dist-info/METADATA', b'Provides-Extra: verify\nRequires-Dist: numpy>=1.26.4\n'),
    ('mojolearn.dist-info/METADATA', b'Provides-Extra: verify\nRequires-Dist: numpy>=1.26.4; extra == "verify" or python_version >= "3"\n'),
])
def test_numpy_policy_rejects_runtime_and_payload_dependencies(tmp_path, name, content):
    path = tmp_path / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    assert audit.numpy_errors(path, name)


@pytest.mark.parametrize('extra', ['numpy', 'verify'])
def test_numpy_verification_extra_is_optional_metadata(tmp_path, extra):
    path = tmp_path / 'METADATA'
    path.write_text(f'Provides-Extra: {extra}\nRequires-Dist: numpy>=1.26.4; extra == "{extra}"\n')
    assert not audit.numpy_errors(path, 'mojolearn.dist-info/METADATA')


def test_numpy_runtime_guard_is_narrow_and_lazy(tmp_path):
    path = tmp_path / '_optional_numpy.py'
    path.write_text('def require_numpy(feature):\n    import numpy\n    return numpy\n')
    assert not audit.numpy_errors(path, 'mojolearn/_optional_numpy.py')
    assert audit.numpy_errors(path, 'mojolearn/other.py')
    path.write_text('import numpy\n')
    assert audit.numpy_errors(path, 'mojolearn/_optional_numpy.py')


def test_shipped_runtime_numpy_imports_use_only_the_guard():
    root = HERE.parents[1] / 'python'
    errors = [error for path in (root / 'mojolearn').rglob('*.py')
              if 'tests' not in path.relative_to(root).parts
              for error in audit.numpy_errors(path, path.relative_to(root).as_posix())]
    assert not errors


def test_compensated_sum_exception_cannot_admit_platform_transcendentals(tmp_path):
    path = tmp_path / '_portable_math.py'
    body = 'import math as _cmath\ndef fsum(vals):\n    return _cmath.fsum(vals)\n'
    path.write_text(body)
    assert audit.python_math_errors(path, 'mojolearn/_portable_math.py') == []
    assert audit.python_math_errors(path, 'mojolearn/other.py')
    for bad in (body.replace('_cmath.fsum', '_cmath.exp'),
                body.replace('return _cmath.fsum(vals)', 'return getattr(_cmath, "fsum")(vals)'),
                body + '\nleaked = _cmath\n', body + '\nleaked = _cmath.fsum\n',
                body.replace('def fsum(', 'def other('),
                '__import__("math")\n', 'importlib.import_module("cmath")\n'):
        path.write_text(bad)
        assert audit.python_math_errors(path, 'mojolearn/_portable_math.py')


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
    shipped = [(p, p.relative_to(root).as_posix()) for p in paths]
    # packaging/linux/pack_wheel.py and the macOS build ship these tools under these names
    tools = HERE.parents[1] / 'tools'
    shipped += [(tools / 'identity_break.py', 'mojolearn/_identity_break.py'),
                (tools / 'identity_trace_diff.py', 'mojolearn/_identity_trace_diff.py')]
    assert paths and not [e for p, rel in shipped for e in audit.dependency_errors(p, rel)]


def test_dependency_policy_allows_the_source_tree_check_only_where_it_is_skipped(tmp_path):
    body = 'def run():\n    import lane_applicability\n'
    module = tmp_path / 'mojolearn' / '_identity_break.py'
    module.parent.mkdir()
    module.write_text(body)
    assert audit.dependency_errors(module, 'mojolearn/_identity_break.py') == []
    other = tmp_path / 'mojolearn' / 'other.py'
    other.write_text(body)
    assert audit.dependency_errors(other, 'mojolearn/other.py')
    module.write_text('import lane_applicability\n')
    assert audit.dependency_errors(module, 'mojolearn/_identity_break.py')


def test_fast_tier_gpu_sets_are_the_only_math_exemption():
    import wheel
    ok = ["mojolearn/cuda/sm_89/_mojolearn_gp.so", "mojolearn/hip/gfx942/_mojolearn.so",
          "mojolearn/_mojolearn_gp.so"]
    enforced = ["mojolearn/cuda/sm_89/identical/_mojolearn_gp.so",
                "mojolearn/hip/gfx942/deterministic/_mojolearn_gbdt.so",
                "mojolearn/host/_mojolearn_gp_host.so", "mojolearn/identical/_mojolearn_gp.so",
                "mojolearn/deterministic/_mojolearn_gbdt.so", "mojolearn/.libs/libfoo.so"]
    assert "__sincosf_stret" in wheel.MATH_SYMBOLS
    assert all(wheel.FAST_SET.match(p) for p in ok)
    assert not any(wheel.FAST_SET.match(p) for p in enforced)


def test_optional_sparse_return_adapter_remains_narrow(tmp_path):
    path = tmp_path / '_expansion_neighbors.py'
    relative = 'mojolearn/_expansion_neighbors.py'
    body = 'class AdditiveChi2Sampler:\n    def transform(self, X):\n        if sparse is not None:\n            import scipy.sparse as sp\n'
    path.write_text(body)
    assert audit.dependency_errors(path, relative) == []
    assert audit.dependency_errors(path, 'mojolearn/other.py')
    for changed in (body.replace('scipy.sparse', 'scipy.linalg'),
                    body.replace('if sparse is not None', 'if True'),
                    body.replace('AdditiveChi2Sampler', 'OtherEstimator'),
                    body.replace('transform', 'fit')):
        path.write_text(changed)
        assert audit.dependency_errors(path, relative)
    path.write_text('import scipy.sparse\n')
    assert audit.dependency_errors(path, relative)


def test_sparse_adapter_does_not_add_a_numpy_runtime_dependency(tmp_path):
    import ast
    source = HERE.parents[1] / 'python' / 'mojolearn' / '_expansion_neighbors.py'
    tree = ast.parse(source.read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'AdditiveChi2Sampler')
    path = tmp_path / 'adapter.py'
    path.write_text(ast.unparse(cls))
    assert audit.numpy_errors(path, 'mojolearn/_expansion_neighbors.py') == []
