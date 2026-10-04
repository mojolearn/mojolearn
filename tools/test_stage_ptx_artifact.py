import json
from pathlib import Path
import pytest
from stage_ptx_artifact import sha, stage

SHA = '8ae4346b1fba67dce59f5a61da0b73c5febd0644'


def fixture(tmp_path):
    source = tmp_path / 'input'
    base = source / 'ptx-baseline'
    tree = base / 'build/sets/cuda/sm_80'
    tree.mkdir(parents=True)
    (tree / '.libs').mkdir()
    for name in ['_mojolearn.so', '_mojolearn_host.so', '.libs/runtime.so', 'readback.txt']:
        (tree / name).write_bytes(name.encode())
    common = dict(source_commit=SHA, source_dirty=False, code_format='ptx-baseline', identical_qualified=False)
    manifest = dict(common, files=[dict(file='_mojolearn.so', sha256=sha(tree / '_mojolearn.so'))])
    (tree / 'PTX_BASELINE.json').write_text(json.dumps(manifest))
    runtime = dict(extensions=[dict(path=name, sha256=sha(tree / name)) for name in ['_mojolearn.so', '_mojolearn_host.so']],
                   staged_libs=[dict(name='runtime.so', sha256=sha(tree / '.libs/runtime.so'))])
    (tree / 'manifest.json').write_text(json.dumps(runtime))
    (base / 'libMojolearnMath.so').write_bytes(b'helper')
    proof = dict(common, manifest_sha256=sha(tree / 'PTX_BASELINE.json'), readback_sha256=sha(tree / 'readback.txt'),
                 runtime_manifest_sha256=sha(tree / 'manifest.json'), portable_math_sha256=sha(base / 'libMojolearnMath.so'))
    (base / 'experimental-build.json').write_text(json.dumps(proof))
    (base / 'build.log').write_text('retained compiler log')
    return source, base, tree


def test_venv_symlinks_never_traversed_and_complete_bytes_preserved(tmp_path):
    source, base, tree = fixture(tmp_path)
    tools = base / 'tools/bin'
    tools.mkdir(parents=True)
    (tools / 'python').symlink_to('/root/inaccessible/python')
    (tree / 'ignored-link').symlink_to('/root/inaccessible')
    output = tmp_path / 'output'
    report = stage(source, output, SHA, True)
    assert report['complete']
    assert not (output / 'ptx-baseline/tools').exists()
    assert not list(output.rglob('ignored-link'))
    for name, digest in report['files'].items():
        assert sha(source / name) == sha(output / name) == digest
    assert (output / 'ptx-baseline/build/sets/cuda/sm_80/.libs/runtime.so').is_file()
    assert (output / 'ptx-baseline/build.log').read_text() == 'retained compiler log'


@pytest.mark.parametrize('member', ['_mojolearn.so', '_mojolearn_host.so', '.libs/runtime.so', 'readback.txt'])
def test_missing_required_payload_fails_success_admission(tmp_path, member):
    source, _, tree = fixture(tmp_path)
    (tree / member).unlink()
    with pytest.raises(ValueError, match='incomplete'):
        stage(source, tmp_path / 'output', SHA, True)
    assert json.loads((tmp_path / 'output/retention.json').read_text())['complete'] is False


def test_failure_logs_survive_without_successful_manifest(tmp_path):
    source = tmp_path / 'input'
    (source / 'bootstrap').mkdir(parents=True)
    (source / 'bootstrap/pixi.log').write_text('failure details')
    report = stage(source, tmp_path / 'output', SHA)
    assert not report['complete']
    assert (tmp_path / 'output/bootstrap/pixi.log').read_text() == 'failure details'


def test_different_source_refused(tmp_path):
    source, _, _ = fixture(tmp_path)
    with pytest.raises(ValueError, match='source differs'):
        stage(source, tmp_path / 'output', 'f' * 40, True)
