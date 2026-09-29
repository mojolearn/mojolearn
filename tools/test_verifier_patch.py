"""Verifier patch allowlist must leave algorithms, references and binaries intact."""
import importlib.util
from pathlib import Path

spec = importlib.util.spec_from_file_location('verifier_patch', Path(__file__).resolve().parents[1] / 'packaging/verifier_patch.py')
patch = importlib.util.module_from_spec(spec)
spec.loader.exec_module(patch)


def test_core_patch_preserves_all_numerical_members(tmp_path):
    source = tmp_path / 'python/mojolearn'
    source.mkdir(parents=True)
    for name in patch.VERIFIER_FILES:
        (source / name).write_text('# new verifier\n')
    base = {'mojolearn/identical/model.so': b'unchanged binary',
            'mojolearn/algorithm.py': b'unchanged algorithm',
            'mojolearn/references.json': b'unchanged hashes',
            'mojolearn/_version.py': b'__version__ = "0.8.25"',
            'mojolearn-0.8.25.dist-info/METADATA': b'Version: 0.8.25\n',
            'mojolearn-0.8.25.dist-info/RECORD': b'old record'}
    name, files = patch.expected(base, 'mojolearn-0.8.25-py3-none-macosx_11_0_arm64.whl', '0.8.26', tmp_path, 'a' * 40)
    assert name == 'mojolearn-0.8.26-py3-none-macosx_11_0_arm64.whl'
    for n in ('mojolearn/identical/model.so', 'mojolearn/algorithm.py', 'mojolearn/references.json'):
        assert files[n] == base[n]
    assert files['mojolearn/_verification_profiles.py'] == b'# new verifier\n'
    assert b'0.8.26' in files['mojolearn-0.8.26.dist-info/METADATA']
    assert 'mojolearn-0.8.25.dist-info/RECORD' not in files


def test_plugin_patch_only_versions_metadata(tmp_path):
    base = {'mojolearn/cuda/model.so': b'unchanged binary',
            'mojolearn_nvidia-0.8.25.dist-info/gpu_plugin.json': b'{"requires":"mojolearn==0.8.25"}',
            'mojolearn_nvidia-0.8.25.dist-info/LINUX_PAYLOAD.json': b'{"version":"0.8.25","original_native_provenance":true}'}
    _, files = patch.expected(base, 'mojolearn_nvidia-0.8.25-py3-none-manylinux_2_35_x86_64.whl', '0.8.26', tmp_path, 'a' * 40)
    assert files['mojolearn/cuda/model.so'] == base['mojolearn/cuda/model.so']
    assert b'0.8.26' in files['mojolearn_nvidia-0.8.26.dist-info/gpu_plugin.json']
    assert files['mojolearn_nvidia-0.8.26.dist-info/LINUX_PAYLOAD.json'] == base['mojolearn_nvidia-0.8.25.dist-info/LINUX_PAYLOAD.json']
    assert not any(n.endswith('.py') for n in files)
