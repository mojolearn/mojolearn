"""Apple optional ByteLM release layout, exercised without compiling or publishing."""
import os
from pathlib import Path
import subprocess
import sys
import types

import pytest
import release_reuse as reuse

ROOT = Path(__file__).resolve().parents[1]
BUILD = (ROOT / 'packaging/macos/build_release_wheel.sh').read_text()
VERIFY = (ROOT / 'packaging/macos/verify_wheel.sh').read_text()
BYTE = '_mojolearn_byte_lm'
PROFILE = 'mojolearn.byte-lm.b2-l32-d32-h4-kv2-ff64-v256-blocks2.fp32.v1'


def test_manifest_apple_adds_fast_without_changing_linux():
    apple = [b for b in reuse.bindings(reuse.MACOS) if b.name == BYTE]
    assert {b.tier for b in apple} == {'fast', 'identical'}
    assert {b.archive_path for b in apple} == {'mojolearn/_mojolearn_byte_lm.so',
                                              'mojolearn/identical/_mojolearn_byte_lm.so'}
    assert {b.tier for b in reuse.bindings(reuse.LINUX) if b.name == BYTE} == {'identical'}
    for vendor in (None, 'cuda', 'hip', 'metal'):
        assert BYTE not in reuse.tier_names('deterministic', True, vendor=vendor)
        assert BYTE not in reuse.tier_names('fast', False, vendor=vendor)


@pytest.mark.parametrize('enabled', [0, 1])
def test_shell_build_plan_agrees_with_apple_manifest(enabled):
    function = BUILD[BUILD.index('build_pairs() {'):BUILD.index('# HEAVIEST FIRST.')]
    env = dict(os.environ, MODES='fast deterministic identical', BUILD_SCRIPTS='build_gbdt.sh',
               IDENTICAL_ONLY_SCRIPTS='', FAST_CLASSICAL_SCRIPTS='build_training.sh',
               HOST_FAMILIES='byte_lm', PACKAGE_BYTE_LM=str(enabled))
    rows = subprocess.check_output(['sh', '-c', function+'\nbuild_pairs'], env=env, text=True).splitlines()
    assert [row for row in rows if row.endswith(' build_byte_lm.sh')] == (
        ['identical build_byte_lm.sh', 'fast build_byte_lm.sh'] if enabled else [])
    assert len(rows) == len(set(rows))


@pytest.mark.parametrize('mode,prefix', [('fast', ''), ('identical', 'identical/')])
def test_output_mapping_and_build_readback(mode, prefix, tmp_path):
    start = BUILD.index('    case "$script" in\n        build_*_host.sh) f=')
    mapping = BUILD[start:BUILD.index('    # A REUSED BINDING', start)]
    env = dict(os.environ, script='build_byte_lm.sh', mode=mode)
    result = subprocess.check_output(['sh', '-c', mapping+'\nprintf "%s" "$output"'], env=env, text=True)
    assert result == 'python/mojolearn/'+prefix+BYTE+'.so'
    body = BUILD.split("<<'PYBYTE'\n", 1)[1].split('\nPYBYTE', 1)[0]
    binding = tmp_path/'fake.py'
    code = 0 if mode == 'fast' else 1
    def write(value):
        binding.write_text(f'def byte_lm_numeric_mode(): return {value}\ndef byte_lm_vendor(): return "metal"\ndef byte_lm_profile(): return {PROFILE!r}\n')
    write(code)
    subprocess.run([sys.executable, '-B', '-c', body, str(binding), mode], check=True, capture_output=True)
    write(2)
    assert subprocess.run([sys.executable, '-B', '-c', body, str(binding), mode], capture_output=True).returncode != 0


def test_installed_fast_verifier_rejects_identical_fallback(monkeypatch):
    body = VERIFY.split("<<'PYBYTEFAST'\n", 1)[1].split('\nPYBYTEFAST', 1)[0]
    package = Path(sys.prefix)/'lib'/'fixture_mojolearn'
    native = types.ModuleType(BYTE)
    native.__file__ = str(package/(BYTE+'.so'))
    native.byte_lm_numeric_mode = lambda: 0
    native.byte_lm_vendor = lambda: 'metal'
    native.byte_lm_profile = lambda: PROFILE
    mod = types.ModuleType('mojolearn');mod.__file__ = str(package/'__init__.py')
    setattr(mod, BYTE, native)
    monkeypatch.setitem(sys.modules, 'mojolearn', mod)
    exec(compile(body, '<installed-fast-verifier>', 'exec'), {})
    native.__file__ = str(package/'identical'/(BYTE+'.so'))
    with pytest.raises(AssertionError):exec(compile(body, '<installed-fast-verifier>', 'exec'), {})
