# SPDX-License-Identifier: Apache-2.0
"""Portable build format gates, independent of CUDA/GPU availability."""
import hashlib
from pathlib import Path
import subprocess
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ptx_baseline as baseline
from ptx_contract import patch_bytes

PTX = (b'.version 8.1\n.target sm_80\n.address_size 64\n.visible .entry k() {\n'
       b'\tmul.f32 \t%r4, %r2, %r3;\n\tret;\n}\n')


def blob(ptx=PTX):
    return b'ELF-placeholder\0' + ptx + b'\0'


def test_requires_explicit_baseline_arch():
    for arch in ('', 'sm_89', 'sm_90a', 'gfx942'):
        with pytest.raises(ValueError, match='sm_80'):
            baseline.validate_config('ptx', arch)
    baseline.validate_config('ptx', 'sm_80')
    baseline.validate_config('native', 'gfx942')
    # the old experimental format name is gone (Andrew 2026-10-10: PTX is a normal target; no flag)
    with pytest.raises(ValueError, match='unsupported'):
        baseline.validate_config('ptx-baseline', 'sm_80')


def test_preserves_rounding_safeguard_and_reports_approx_without_certifying(tmp_path):
    data = blob(PTX.replace(b'\tret;', b'\trsqrt.approx.f32 %r4, %r2;\n\tret;'))
    assert baseline.audit_binary(data)[1] == ['IDENTICAL PTX contains unpinned floating arithmetic']
    patched, _, _ = patch_bytes(data)
    path = tmp_path / 'identical' / 'binding.so'
    path.parent.mkdir()
    path.write_bytes(patched)
    report = baseline.audit_tree(tmp_path, 'a' * 40, 'Mojo 1.0.0')
    assert report['errors'] == []
    assert report['schema'] == 'mojolearn.ptx-set.v2'
    assert report['code_format'] == 'ptx'
    for field in ('identical_qualified', 'experimental', 'qualification_required'):
        assert field not in report
    row = report['files'][0]
    assert row['sha256'] == hashlib.sha256(patched).hexdigest()
    assert row['ptx_modules'][0]['approx'] == {'rsqrt.approx.f32': 1}
    assert row['ptx_modules'][0]['ptx_isa'] == '8.1'


@pytest.mark.parametrize('target', [b'sm_89', b'sm_90a', b'sm_80, texmode_independent'])
def test_refuses_higher_or_feature_specific_targets(target):
    data, _, _ = patch_bytes(blob(PTX.replace(b'sm_80', target)))
    assert any('non-baseline' in e for e in baseline.audit_binary(data)[1])


def test_refuses_fatbin_and_bare_cubin():
    data, _, _ = patch_bytes(blob())
    assert any('fatbin' in e for e in baseline.audit_binary(data + baseline._FATBIN)[1])
    elf = bytearray(24)
    elf[:6] = b'\x7fELF\x02\x01'
    elf[18:20] = (190).to_bytes(2, 'little')
    assert any('CUDA ELF' in e for e in baseline.audit_binary(data + elf)[1])


def test_missing_malformed_and_duplicate_headers():
    for ptx in (PTX.replace(b'.target sm_80', b'.target nonsense'),
                PTX.replace(b'.version 8.1', b'.version 8.1\n.version 8.1'),
                PTX.replace(b'.address_size 64', b'.address_size 32')):
        assert baseline.audit_binary(blob(ptx))[1]


def test_empty_or_native_only_set_refused_and_host_ignored(tmp_path):
    host = tmp_path / 'host'
    host.mkdir()
    (host / 'host.so').write_bytes(blob())
    report = baseline.audit_tree(tmp_path, 'b' * 40, 'Mojo 1.0.0')
    assert report['files'] == []
    assert report['errors'] == ['PTX set contains no PTX modules']


def test_fast_is_reported_without_identical_rounding_claim(tmp_path):
    (tmp_path / 'fast.so').write_bytes(blob())
    report = baseline.audit_tree(tmp_path, 'b' * 40, 'Mojo 1.0.0')
    assert report['errors'] == []
    assert report['files'][0]['numeric_mode'] == 'fast'
    assert 'identical_qualified' not in report


def test_cli_invalid_format_fails_before_any_compilation(tmp_path):
    script = Path(__file__).with_name('build_sets.sh')
    import os
    env = dict(os.environ, MOJOLEARN_CUDA_CODE_FORMAT='ptx', MOJOLEARN_GPU_ARCHS='gfx942')
    result = subprocess.run(['bash', str(script), str(tmp_path / 'out')], env=env,
                            capture_output=True, text=True)
    assert result.returncode == 2
    assert 'requires explicit' in result.stderr
    assert not (tmp_path / 'out').exists()


def test_device_free_binding_requires_registered_same_tier_delegates(tmp_path):
    glue = tmp_path / '_mojolearn_x_trees.so'
    glue.write_bytes(b'host-only wrapper')
    assert 'missing PTX device delegates' in baseline.audit_tree(tmp_path, 'a' * 40, 'Mojo')['errors'][0]
    for name in ('_mojolearn_rf', '_mojolearn_gbdt'):
        (tmp_path / (name + '.so')).write_bytes(blob())
    report = baseline.audit_tree(tmp_path, 'a' * 40, 'Mojo')
    assert report['errors'] == []
    row = next(r for r in report['files'] if r['file'] == glue.name)
    assert len(row['delegates']) == 2
    (tmp_path / 'unregistered.so').write_bytes(b'host wrapper')
    assert any('unregistered binary' in e for e in baseline.audit_tree(tmp_path, 'a' * 40, 'Mojo')['errors'])
