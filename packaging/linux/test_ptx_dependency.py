"""Exercise the actual pip dependency resolver over inert, locally packed wheels."""
import hashlib
import json
import subprocess
import sys
import zipfile

import pytest

from test_split_wheels import make_sets, members, pw, wheel_api_audit


def packed(root):
    sets = make_sets(root)
    out = root / 'wheels'
    args = [a for directory in sets for a in ('--set', directory)]
    assert pw.main(args + ['--out', str(out)], _gates=False) == 0
    return out


def resolve(wheels, destination, project='mojolearn'):
    return subprocess.run([
        sys.executable, '-m', 'pip', 'download', '--disable-pip-version-check',
        '--no-cache-dir', '--no-index', '--find-links', str(wheels),
        '--only-binary=:all:', '--platform', pw.SPLIT_PLAT,
        '--dest', str(destination), f'{project}=={pw.read_version()}',
    ], capture_output=True, text=True, timeout=60)


#: 2026-10-20: what each Linux install resolves to. mojolearn (file-free)
#: pulls the core and both vendors; a vendor pulls the core; the core pulls no
#: vendor. 2026-11-05: the NVIDIA wheel carries its PTX set, no PTX package.
RESOLVES = {
    'mojolearn': {'mojolearn', 'mojolearn_core', 'mojolearn_nvidia', 'mojolearn_amd'},
    'mojolearn-nvidia': {'mojolearn_core', 'mojolearn_nvidia'},
    'mojolearn-amd': {'mojolearn_core', 'mojolearn_amd'},
    'mojolearn-core': {'mojolearn_core'},
}


@pytest.mark.parametrize('project', sorted(RESOLVES))
def test_pip_resolves_each_install_from_its_requirements(tmp_path, project):
    wheels = packed(tmp_path)
    destination = tmp_path / 'resolved'
    result = resolve(wheels, destination, project)
    assert result.returncode == 0, result.stdout + result.stderr
    assert {p.name.split('-')[0] for p in destination.glob('*.whl')} == RESOLVES[project]
    for p in destination.glob('*.whl'):
        assert hashlib.sha256(p.read_bytes()).digest() == hashlib.sha256((wheels / p.name).read_bytes()).digest()


def test_pip_refuses_a_vendor_without_the_core(tmp_path):
    wheels = packed(tmp_path)
    next(wheels.glob('mojolearn_core-*.whl')).unlink()
    result = resolve(wheels, tmp_path / 'resolved', 'mojolearn-amd')
    assert result.returncode != 0
    assert 'mojolearn-core' in result.stderr


def test_no_separate_ptx_distribution_is_packed_or_required(tmp_path):
    wheels = packed(tmp_path)
    assert not list(wheels.glob('mojolearn_nvidia_ptx-*.whl'))
    native = next(wheels.glob('mojolearn_nvidia-*.whl'))
    files = members(native)
    metadata = next(data for name, data in files.items() if name.endswith('.dist-info/METADATA')).decode()
    assert 'mojolearn-nvidia-ptx' not in metadata
    assert any(name.startswith('mojolearn/cuda_portable/sm_80/') for name in files)
    assert any(name.startswith('mojolearn/cuda_native/sm_89/') for name in files)


def test_audit_binds_the_nvidia_marker_to_its_own_ptx_bytes(tmp_path):
    wheels = packed(tmp_path)
    native = next(wheels.glob('mojolearn_nvidia-*.whl'))
    files = members(native)
    marker_name = next(n for n in files if n.endswith('/gpu_plugin.json'))
    marker = json.loads(files[marker_name])
    marker['bundled_ptx']['manifest_sha256'] = '0' * 64
    files[marker_name] = json.dumps(marker).encode()
    with zipfile.ZipFile(native, 'w') as z:
        for name, data in files.items():
            z.writestr(name, data)
    problems = wheel_api_audit.split_audit(wheels.glob('*.whl'))['problems']
    assert any('invalid PTX slot' in p for p in problems)


def test_a_ptx_only_wheel_kind_no_longer_exists(tmp_path):
    sets = make_sets(tmp_path, sets=(('cuda', 'sm_80'),), include_byte_lm=True)
    args = [a for directory in sets for a in ('--set', directory)]
    with pytest.raises(SystemExit, match='name each'):
        pw.main(args + ['--profile', 'release-split', '--wheels', 'nvidia-ptx',
                       '--build-proof', 'cuda-sm_80.json', '--out', str(tmp_path / 'out')], _gates=False)
