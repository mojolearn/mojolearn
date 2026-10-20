"""Architecture publish ordering and actual installed-target admission."""
import hashlib
import json
from pathlib import Path
import runpy
import zipfile

import pytest
import release
import release_reuse
import index_release_check as index
import qualify_verifier_wheel as qualifier


def test_registry_parity_and_no_experimental_publication():
    registry = runpy.run_path(str(release.ROOT / 'python/mojolearn/gpu_plugins.py'))
    rows = registry['distribution_rows']()
    # 2026-10-20: mojolearn-core (no vendor pin), the vendors, and the
    # file-free Linux mojolearn requiring the core and both vendors
    assert set(index.PROJECTS) == {'mojolearn', 'mojolearn-core', *(r['distribution'] for r in rows)}
    assert index.REQUIRES['mojolearn-core'] == ()
    assert registry['core_requirements']('1.2.3') == []
    assert index.REQUIRES['mojolearn'] == tuple(r.split('==')[0] for r in registry['meta_requirements']('1.2.3'))
    assert index.REQUIRES['mojolearn'] == ('mojolearn-core', 'mojolearn-nvidia', 'mojolearn-amd')
    assert qualifier.PLUGIN_DISTRIBUTIONS == {r['wheel_name']: r['distribution'] for r in rows}
    assert 'nvidia-ptx80' not in release.SPLIT_PACKAGES
    assert 'nvidia-ptx80' not in registry['PAYLOADS']
    assert all(r['release_enabled'] for r in rows)
    for row in rows:
        expected = registry['package_requirements'](row['profile'], '1.2.3')
        assert index.REQUIRES[row['distribution']] == tuple(r.split('==')[0] for r in expected)
        assert expected[0] == 'mojolearn-core==1.2.3'


def test_payloads_cannot_publish_before_own_architecture_checks():
    # 2026-10-08, releases rent nothing: the Hopper column gates publish-nvidia only when a Hopper box is held
    # (Release.hopper_required adds the need); otherwise publish-nvidia waits for it to settle (AFTER)
    for vendor, columns in release.NATIVE_COLUMNS.items():
        for column in columns:
            if column == 'nvidia-hopper':
                assert 'gpu-column-nvidia-hopper' in release.AFTER['publish-nvidia']
                continue
            assert 'gpu-column-' + column in release.NEEDS['publish-' + vendor]
        assert 'linux-joint-diff' not in release.NEEDS['publish-' + vendor]   # deleted 2026-10-10: identity runs ONCE
    # 2026-10-20 order: mojolearn-core FIRST (after both GPU columns settle),
    # then PTX and AMD, then NVIDIA, then the Linux mojolearn LAST
    assert release.NEEDS['publish-core-linux'] == ['linux-pack']
    assert {'gpu-column-nvidia', 'gpu-column-amd'} <= set(release.AFTER['publish-core-linux'])
    assert 'publish-core-linux' in release.NEEDS['publish-nvidia-ptx']
    assert 'publish-core-linux' in release.NEEDS['publish-amd']
    assert 'publish-nvidia-ptx' in release.NEEDS['publish-nvidia']
    assert set(release.NEEDS['publish-meta-linux']) == {'publish-core-linux', 'publish-nvidia', 'publish-amd'}
    assert not any(s in release.NEEDS['publish-core-linux']
                   for s in ('publish-nvidia', 'publish-nvidia-ptx', 'publish-amd', 'publish-meta-linux'))
    assert 'linux-joint-diff' not in release.AFTER


def test_target_admission_requires_actual_installed_arch(tmp_path):
    runner = object.__new__(release.Release)
    receipt = tmp_path / 'results.json'
    receipt.write_text(json.dumps({'installed': {'gpu_arch': 'sm_89'}}))
    assert runner.column_arch_ok(tmp_path, ('sm_89',))
    assert not runner.column_arch_ok(tmp_path, ('sm_90', 'sm_90a'))
    receipt.write_text(json.dumps({'installed': {'vendor': 'cuda'}}))
    assert not runner.column_arch_ok(tmp_path, ('sm_89',))


# the core published as mojolearn_core-* since 2026-10-20, as mojolearn-* before
@pytest.mark.parametrize('core_name', ['mojolearn_core', 'mojolearn'])
def test_synthetic_reuse_normalizes_roots_without_changing_bytes(tmp_path, core_name):
    core = tmp_path / f'{core_name}-1-py3-none-manylinux.whl'
    payload = tmp_path / 'mojolearn_nvidia-1-py3-none-manylinux.whl'
    with zipfile.ZipFile(core, 'w') as z:
        z.writestr('mojolearn/__init__.py', b'core')
        z.writestr(f'{core_name}-1.dist-info/LINUX_PAYLOAD.json', json.dumps({
            'extensions': {'mojolearn/cuda_native/sm_89/identical/kernel.so': 'hash'}, 'split': {}}))
    with zipfile.ZipFile(payload, 'w') as z:
        z.writestr('mojolearn/cuda_native/sm_89/identical/kernel.so', b'exact bytes')
    merged = release.merge_split([payload, core], tmp_path / 'merged.whl')
    with zipfile.ZipFile(merged) as z:
        assert z.read('mojolearn/cuda/sm_89/identical/kernel.so') == b'exact bytes'
        metadata = json.loads(z.read(f'{core_name}-1.dist-info/LINUX_PAYLOAD.json'))
        assert metadata['extensions'] == {'mojolearn/cuda/sm_89/identical/kernel.so': 'hash'}
    # the PTX slot's installed root maps back to its set path, like cuda_native
    assert release_reuse.legacy_archive_path('mojolearn/cuda_ptx/sm_80/kernel.so') == 'mojolearn/cuda/sm_80/kernel.so'


def test_missing_payload_index_release_is_refused():
    from test_index_release_check import run
    _, errors = run(missing=('mojolearn-nvidia',))
    assert len(errors) == 1 and 'mojolearn-nvidia' in errors[0]
