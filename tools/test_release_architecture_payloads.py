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
    assert set(index.PROJECTS) == {'mojolearn', *(r['distribution'] for r in rows)}
    assert qualifier.PLUGIN_DISTRIBUTIONS == {r['wheel_name']: r['distribution'] for r in rows}
    assert 'nvidia-ptx80' not in release.SPLIT_PACKAGES
    for row in rows:
        expected = registry['package_requirements'](row['profile'], '1.2.3')
        assert index.REQUIRES[row['distribution']] == tuple(r.split('==')[0] for r in expected)


def test_payloads_cannot_publish_before_own_architecture_checks():
    # 2026-10-08, releases rent nothing: the Hopper column gates publish-nvidia only when a Hopper box is held
    # (Release.hopper_required adds the need); otherwise publish-nvidia waits for it to settle (AFTER)
    for vendor, columns in release.NATIVE_COLUMNS.items():
        for column in columns:
            if column == 'nvidia-hopper':
                assert 'gpu-column-nvidia-hopper' in release.AFTER['publish-nvidia']
                continue
            assert 'gpu-column-' + column in release.NEEDS['publish-' + vendor]
        assert 'linux-joint-diff' in release.NEEDS['publish-' + vendor]
    assert {'publish-nvidia', 'publish-amd'} <= set(release.NEEDS['publish-core-linux'])
    assert 'gpu-column-nvidia-hopper' in release.AFTER['linux-joint-diff']


def test_target_admission_requires_actual_installed_arch(tmp_path):
    runner = object.__new__(release.Release)
    receipt = tmp_path / 'results.json'
    receipt.write_text(json.dumps({'installed': {'gpu_arch': 'sm_89'}}))
    assert runner.column_arch_ok(tmp_path, ('sm_89',))
    assert not runner.column_arch_ok(tmp_path, ('sm_90', 'sm_90a'))
    receipt.write_text(json.dumps({'installed': {'vendor': 'cuda'}}))
    assert not runner.column_arch_ok(tmp_path, ('sm_89',))


def test_synthetic_reuse_normalizes_roots_without_changing_bytes(tmp_path):
    core = tmp_path / 'mojolearn-1-py3-none-manylinux.whl'
    payload = tmp_path / 'mojolearn_nvidia-1-py3-none-manylinux.whl'
    with zipfile.ZipFile(core, 'w') as z:
        z.writestr('mojolearn/__init__.py', b'core')
        z.writestr('mojolearn-1.dist-info/LINUX_PAYLOAD.json', json.dumps({
            'extensions': {'mojolearn/cuda_native/sm_89/identical/kernel.so': 'hash'}, 'split': {}}))
    with zipfile.ZipFile(payload, 'w') as z:
        z.writestr('mojolearn/cuda_native/sm_89/identical/kernel.so', b'exact bytes')
    merged = release.merge_split([core, payload], tmp_path / 'merged.whl')
    with zipfile.ZipFile(merged) as z:
        assert z.read('mojolearn/cuda/sm_89/identical/kernel.so') == b'exact bytes'
        metadata = json.loads(z.read('mojolearn-1.dist-info/LINUX_PAYLOAD.json'))
        assert metadata['extensions'] == {'mojolearn/cuda/sm_89/identical/kernel.so': 'hash'}
    assert release_reuse.legacy_archive_path('mojolearn/cuda_ptx/sm_80/kernel.so') == 'mojolearn/cuda_ptx/sm_80/kernel.so'


def test_missing_payload_index_release_is_refused():
    from test_index_release_check import run
    _, errors = run(missing=('mojolearn-nvidia',))
    assert len(errors) == 1 and 'mojolearn-nvidia' in errors[0]
