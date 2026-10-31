# SPDX-License-Identifier: Apache-2.0
"""Reject independently introduced platform math from a wheel payload."""
import importlib.util
from pathlib import Path
import subprocess
import sys

import pytest


def test_optional_python314_stdlib_requires_importerror_guard(tmp_path, monkeypatch):
    monkeypatch.setattr(sys, 'stdlib_module_names', sys.stdlib_module_names - {'compression'})
    path = tmp_path / 'module.py'
    path.write_text('try:\n    from compression import zstd\nexcept ImportError:\n    zstd = None\n')
    assert audit.dependency_errors(path, 'mojolearn/module.py') == []
    path.write_text('from compression import zstd\n')
    assert audit.dependency_errors(path, 'mojolearn/module.py')
    path.write_text('try:\n    from compression import zstd\nexcept ValueError:\n    zstd = None\n')
    assert audit.dependency_errors(path, 'mojolearn/module.py')

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


@pytest.mark.parametrize('name', sorted(audit.NUMPY_ORACLES))
def test_exact_verifier_controllers_may_use_reference_packages(tmp_path, name):
    path = tmp_path / 'verifier.py'
    path.write_text('import numpy\nimport math\n')
    assert not audit.numpy_errors(path, name)
    assert not audit.python_math_errors(path, name)
    assert audit.numpy_errors(path, 'mojolearn/_verify_new_runtime.py')
    assert audit.python_math_errors(path, 'mojolearn/_verify_new_runtime.py')


@pytest.mark.parametrize('name,scope', [
    ('mojolearn/_native_xcnn_bridge.py', 'connect'),
    ('mojolearn/importers/_native_reader_bridge.py', 'connect'),
    ('mojolearn/resample.py', None),
])
def test_native_metadata_import_guard_is_location_and_statement_scoped(tmp_path, name, scope):
    path = tmp_path / 'setup.py'
    guarded = 'try:\n    import numpy as np\nexcept ImportError:\n    np = None\n'
    def scoped(text):
        return ('def ' + scope + '():\n' + ''.join('    ' + line + '\n' for line in text.splitlines())) if scope else text
    path.write_text(scoped(guarded))
    assert not audit.numpy_errors(path, name)
    assert audit.numpy_errors(path, 'mojolearn/unrelated.py')
    path.write_text(scoped('import numpy\n'))
    assert audit.numpy_errors(path, name)
    path.write_text(scoped(guarded.replace('ImportError', 'ValueError')))
    assert audit.numpy_errors(path, name)
    path.write_text(scoped('def later():\n' + ''.join('    ' + line + '\n' for line in guarded.splitlines())))
    assert audit.numpy_errors(path, name)


def test_rng_setup_admits_only_exact_native_metadata_import(tmp_path):
    path = tmp_path / 'rng.py'
    text = ('def _connect_pcs_rng():\n    try:\n'
            '        from numpy.random import mtrand, BitGenerator, MT19937, PCG64, PCG64DXSM, Philox, SFC64\n'
            '    except ImportError:\n        return\n')
    path.write_text(text)
    assert not audit.numpy_errors(path, 'mojolearn/_expansion_neighbors.py')
    path.write_text(text.replace('SFC64', 'default_rng'))
    assert audit.numpy_errors(path, 'mojolearn/_expansion_neighbors.py')


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


def test_native_registry_may_connect_only_exact_math_exports(tmp_path):
    path = tmp_path / '_native_loaded_modules.py'
    body = "import math\nconnect_module('math', math, ('floor', 'ceil', 'comb'))\n"
    path.write_text(body)
    rel = 'mojolearn/_native_loaded_modules.py'
    assert audit.python_math_errors(path, rel) == []
    assert audit.python_math_errors(path, 'mojolearn/other.py')
    for bad in (body.replace("'comb'", "'sin'"), body.replace("'floor', ", "'sqrt', 'floor', "),
                body.replace("('floor', 'ceil', 'comb')", "tuple(n for n in vars(math))"),
                body + 'leaked = math.exp\n', body + "connect_module('math', math, ('exp',))\n",
                body.replace('import math', 'import math as m'), body + 'from math import log\n'):
        path.write_text(bad)
        assert audit.python_math_errors(path, rel), bad


@pytest.mark.parametrize("relative", ["mojolearn/_identity_break.py", "mojolearn/_verify_all.py"])
def test_numpy_remains_available_to_independent_verification(tmp_path, relative):
    path = tmp_path / '_identity_break.py'
    path.write_text('import numpy as np\n')
    assert not audit.numpy_errors(path, relative)


NATIVE_VERIFIER_NAMES = (
    "_verify_distributed_program", "_verify_generation", "_verify_graph",
    "_verify_partitioned_program",
)


@pytest.mark.parametrize("name", NATIVE_VERIFIER_NAMES)
def test_native_verifier_numpy_classification_is_exact(tmp_path, name):
    source = HERE.parents[1] / "python/mojolearn" / (name + ".py")
    relative = "mojolearn/" + name + ".py"
    assert not audit.numpy_errors(source, relative)
    # The same oracle import remains illegal if moved into an execution
    # module or a similarly named new module; no prefix exemption is granted.
    path = tmp_path / (name + "_runtime.py")
    path.write_text("def predict(x):\n    import numpy as np\n    return np.asarray(x)\n")
    assert audit.numpy_errors(path, "mojolearn/" + path.name)
    assert audit.numpy_errors(path, "mojolearn/models/causal_lm.py")
    assert audit.numpy_errors(path, "mojolearn/ops.py")


def test_native_numpy_verifiers_are_lazy_and_have_no_product_callers():
    import ast
    package = HERE.parents[1] / "python/mojolearn"
    names = set(NATIVE_VERIFIER_NAMES)
    for name in names:
        tree = ast.parse((package / (name + ".py")).read_bytes())
        lazy = {id(node) for function in tree.body if isinstance(function, (ast.FunctionDef, ast.AsyncFunctionDef))
                for node in ast.walk(function)}
        imports = [node for node in ast.walk(tree) if
                   isinstance(node, ast.Import) and any(a.name.split(".")[0] == "numpy" for a in node.names)
                   or isinstance(node, ast.ImportFrom) and (node.module or "").split(".")[0] == "numpy"]
        assert imports and all(id(node) in lazy for node in imports), name
    callers = set()
    for path in package.rglob("*.py"):
        relative = path.relative_to(package)
        if "tests" in relative.parts:
            continue
        tree = ast.parse(path.read_bytes())
        for node in ast.walk(tree):
            references = []
            if isinstance(node, ast.Import):
                references = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                references = [node.module or ""] + [alias.name for alias in node.names]
            elif isinstance(node, ast.Call) and node.args and isinstance(node.args[0], ast.Constant):
                if isinstance(node.func, ast.Name) and node.func.id == "__import__" or isinstance(node.func, ast.Attribute) and node.func.attr == "import_module":
                    references = [str(node.args[0].value)]
            if any(value.split(".")[-1] in names for value in references):
                callers.add(relative.as_posix())
    assert callers == {"__main__.py"}, callers


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
          "mojolearn/_mojolearn_gp.so",
          # the merged-bindings layout: one native file per set directory
          "mojolearn/cuda/sm_89/_mojolearn_native.so", "mojolearn/cuda_ptx/sm_80/_mojolearn_native.so",
          "mojolearn/_mojolearn_native.so"]
    enforced = ["mojolearn/cuda/sm_89/identical/_mojolearn_gp.so",
                "mojolearn/host/_mojolearn_gp_host.so", "mojolearn/identical/_mojolearn_gp.so",
                "mojolearn/.libs/libfoo.so",
                "mojolearn/cuda/sm_89/identical/_mojolearn_native.so",
                "mojolearn/hip/gfx942/identical/_mojolearn_native.so",
                "mojolearn/host/_mojolearn_native_host.so", "mojolearn/identical/_mojolearn_native.so"]
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


def test_metadata_only_vendor_aggregate_needs_no_elf(tmp_path):
    import json
    dist = tmp_path / 'mojolearn_nvidia-1.0.dist-info'
    dist.mkdir()
    (dist / 'gpu_plugin.json').write_text(json.dumps({'role': 'aggregate'}))
    assert audit.audit_tree(tmp_path)['binaries'] == []
    (tmp_path / 'payload.so').write_bytes(b'inert')
    with pytest.raises(ValueError, match='aggregate contains non-metadata'):
        audit.audit_tree(tmp_path)


def test_empty_payload_cannot_use_aggregate_exception(tmp_path):
    import json
    dist = tmp_path / 'mojolearn_nvidia_sm89-1.0.dist-info'
    dist.mkdir()
    (dist / 'gpu_payload.json').write_text(json.dumps({'role': 'payload'}))
    with pytest.raises(ValueError, match='no native binaries'):
        audit.audit_tree(tmp_path)


def test_baseline_manifest_is_rechecked_against_final_bytes(payload):
    import json
    directory = payload / 'mojolearn/cuda_ptx/sm_80'
    directory.mkdir(parents=True)
    binary = directory / 'probe.so'
    binary.write_bytes(b'fixture\0.version 8.1\n.target sm_80\n.address_size 64\n'
                       b'.visible .entry probe() {\nret;\n}\n\0')
    manifest = audit.ptx_baseline.audit_tree(directory, 'a' * 40, 'test compiler')
    assert not manifest['errors']
    (directory / 'PTX_BASELINE.json').write_text(json.dumps(manifest))
    report = audit.audit_tree(payload)
    assert report['identical_cuda_machine_code'] is True
    assert report['ptx_sets'][0]['schema'] == 'mojolearn.ptx-set.v2'
    assert report['ptx_sets'][0]['code_format'] == 'ptx'
    assert 'identical_qualified' not in report['ptx_sets'][0]
    manifest['component_sources'] = {'probe.so': {
        'binary_sha256': manifest['files'][0]['sha256'],
        'source_commit': 'b' * 40, 'source_closure_sha256': 'c' * 64,
        'source_closure_ref': 'retained/source.json'}}
    (directory / 'PTX_BASELINE.json').write_text(json.dumps(manifest))
    audit.audit_tree(payload)
    manifest['unexpected'] = True
    (directory / 'PTX_BASELINE.json').write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match='final bytes differ'):
        audit.audit_tree(payload)
    del manifest['unexpected']
    manifest['component_sources']['probe.so']['binary_sha256'] = 'd' * 64
    (directory / 'PTX_BASELINE.json').write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match='component source evidence'):
        audit.audit_tree(payload)
    del manifest['component_sources']
    (directory / 'PTX_BASELINE.json').write_text(json.dumps(manifest))
    binary.write_bytes(binary.read_bytes() + b'changed')
    with pytest.raises(ValueError, match='complete GPU binding bytes'):
        audit.audit_tree(payload)
