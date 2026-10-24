"""Inert candidate artifact and publication-order regression checks."""
import base64
import csv
import hashlib
import io
import json
from pathlib import Path
from types import SimpleNamespace
import zipfile

import pytest

import candidate_wheel_install as C
import gate_autograd_artifacts as A
import pypi_wheel_install as P
import release
import release_lq_smoke as LQ

VERSION = '0.8.56'
COMMIT = 'a'*40
TAG = 'py3-none-manylinux_2_28_x86_64'


def wheel(tmp_path, name, *, commit=COMMIT, version=VERSION, requires=(), corrupt=False, modes=None):
    base = name.replace('-', '_')+'-'+version
    dist = base+'.dist-info'
    files = {
        dist+'/METADATA': ('Metadata-Version: 2.1\nName: '+name+'\nVersion: '+version+'\n'+
                           ''.join('Requires-Dist: '+value+'\n' for value in requires)).encode(),
        dist+'/WHEEL': ('Wheel-Version: 1.0\nRoot-Is-Purelib: false\nTag: '+TAG+'\n').encode(),
        dist+'/LINUX_PAYLOAD.json': json.dumps(dict(schema='mojolearn.linux-payload.v1', version=version,
            source_commit=commit, split=dict(distribution=name), **({'release_modes': modes} if modes is not None else {}))).encode(),
        name+'/fixture.txt': b'inert package fixture, never executable',
    }
    if name == P.CORE:
        files['mojolearn/identity_columns/COMMIT'] = (commit+'\n').encode()
    output = io.StringIO()
    writer = csv.writer(output)
    for member, value in files.items():
        writer.writerow((member, 'sha256='+base64.urlsafe_b64encode(hashlib.sha256(value).digest()).rstrip(b'=').decode(), len(value)))
    writer.writerow((dist+'/RECORD', '', ''))
    files[dist+'/RECORD'] = output.getvalue().encode()
    if corrupt:
        files[name+'/fixture.txt'] += b'changed after RECORD'
    path = tmp_path/(base+'-'+TAG+'.whl')
    with zipfile.ZipFile(path, 'w') as archive:
        for member, value in files.items():
            archive.writestr(member, value)
    return path


def wheels(tmp_path, vendor='nvidia'):
    return [wheel(tmp_path, name, requires=(() if name == P.CORE else (P.CORE+'=='+VERSION,)))
            for name in P.distributions(vendor)]


def loaded(manifest):
    return {'import mojolearn': {'module_file': '/venv/lib/python3.12/site-packages/mojolearn/__init__.py'},
            **{row['distribution']: dict(version=VERSION, location='/venv/lib/python3.12/site-packages',
                direct_url=dict(url=Path(row['path']).as_uri(), archive_info=dict(hashes=dict(sha256=row['sha256']))))
               for row in manifest['files']}}


def artifacts(tmp_path, vendor='nvidia'):
    directory = tmp_path/'autograd'
    directory.mkdir()
    rows = []
    for name, source in A.expected(vendor).items():
        path = directory/name
        path.write_bytes(b'inert fixture; tests never execute this file')
        path.chmod(0o755)
        rows.append(dict(file=name, source=source, source_sha256='b'*64,
                         command=['mojo', 'build', '--target-cpu', 'x86-64-v3', source],
                         exit_code=0, sha256=A.sha(path)))
    doc = dict(schema=A.SCHEMA, state='built', source_commit=COMMIT, vendor=vendor, target='sm_89',
               target_cpu='x86-64-v3', compiler=dict(sha256='c'*64), files=rows)
    (directory/'build-manifest.json').write_text(json.dumps(doc))
    return directory


def test_candidate_requires_every_vendor_distribution_and_one_frozen_version(tmp_path):
    paths = wheels(tmp_path)
    manifest = C.build_manifest(VERSION, 'nvidia', paths, COMMIT)
    assert manifest['source'] == 'candidate' and 'all_digests_match' not in manifest
    assert C.identity(manifest)['wheels'].keys() == set(P.distributions('nvidia'))
    with pytest.raises(ValueError, match='exactly the core'):
        C.build_manifest(VERSION, 'nvidia', paths[:-1], COMMIT)
    with pytest.raises(ValueError, match='provenance'):
        C.build_manifest(VERSION, 'nvidia', paths, 'd'*40)
    with pytest.raises(ValueError, match='version'):
        C.build_manifest('0.8.57', 'nvidia', paths, COMMIT)


def test_candidate_refuses_record_tampering_and_hidden_pypi_mojolearn_dependency(tmp_path):
    corrupt = wheel(tmp_path, P.CORE, corrupt=True)
    with pytest.raises(ValueError, match='RECORD mismatch'):
        C.wheel_record(corrupt, VERSION, COMMIT)
    paths = wheels(tmp_path, 'amd')
    paths[0] = wheel(tmp_path, P.CORE, requires=('mojolearn=='+VERSION,))
    with pytest.raises(ValueError, match='unprovided'):
        C.build_manifest(VERSION, 'amd', paths, COMMIT)


def test_candidate_modes_match_and_missing_metadata_is_historical_dual(tmp_path):
    paths = wheels(tmp_path, 'amd')
    assert C.build_manifest(VERSION, 'amd', paths, COMMIT)['release_modes'] == ['fast', 'identical']
    paths[0] = wheel(tmp_path, P.CORE, modes=['identical'])
    with pytest.raises(ValueError, match='release modes differ'):
        C.build_manifest(VERSION, 'amd', paths, COMMIT)


def test_candidate_loaded_receipt_requires_exact_hash_without_claiming_pypi(tmp_path):
    manifest = C.build_manifest(VERSION, 'amd', wheels(tmp_path, 'amd'), COMMIT)
    receipt = loaded(manifest)
    assert P.refusal(manifest, receipt) is None
    receipt[P.CORE]['direct_url']['archive_info'] = {}
    assert 'exact artifact hash' in P.refusal(manifest, receipt)
    receipt = loaded(manifest)
    receipt[P.CORE]['direct_url']['archive_info']['hashes']['sha256'] = '0'*64
    assert 'hash' in P.refusal(manifest, receipt)


@pytest.mark.parametrize('artifact_modes', [None, ['fast', 'identical'], ['identical']])
def test_release_candidate_modes_match_controller_even_on_cached_identity(tmp_path, artifact_modes):
    paths = [wheel(tmp_path, name, modes=artifact_modes) for name in P.distributions('amd')]
    runner = object.__new__(release.Release)
    runner.version, runner.dry = VERSION, False
    runner.state = {'commit': COMMIT}
    runner.release_modes = ('identical',)
    runner.candidate_gate_wheels = lambda vendor: paths
    if artifact_modes != ['identical']:
        with pytest.raises(release.StepFailed, match='selected release modes'):
            runner.candidate_gate_identity('amd')
        return
    assert runner.candidate_gate_identity('amd')['release_modes'] == ['identical']
    assert runner.candidate_gate_identity('amd')['release_modes'] == ['identical']
    runner.release_modes = ('fast', 'identical')
    with pytest.raises(release.StepFailed, match='selected release modes'):
        runner.candidate_gate_identity('amd')


def test_candidate_install_passes_local_files_and_checks_interpreter_tags(tmp_path, monkeypatch):
    paths = wheels(tmp_path, 'amd')
    calls = []
    monkeypatch.setattr(C.subprocess, 'run', lambda *a, **kw: SimpleNamespace(stdout=json.dumps([TAG])))
    monkeypatch.setattr(P, 'run', lambda argv, log: calls.append(argv) or 0)
    monkeypatch.setattr(P, 'loaded_receipt', lambda python, manifest: loaded(manifest))
    args = SimpleNamespace(version=VERSION, vendor='amd', wheel=paths, source_commit=COMMIT,
                           python='/fake/python', log=str(tmp_path/'install.log'), manifest=str(tmp_path/'manifest.json'))
    assert C.install(args) == 0
    assert len(calls) == 2 and all(str(path.resolve()) in command for command in calls for path in paths)
    assert not any(arg.startswith('mojolearn') and '==' in arg for command in calls for arg in command)
    assert '--no-deps' in calls[1] and '--force-reinstall' in calls[1]
    monkeypatch.setattr(C.subprocess, 'run', lambda *a, **kw: SimpleNamespace(stdout=json.dumps(['cp313-cp313-win_amd64'])))
    with pytest.raises(ValueError, match='ABI/platform'):
        C.install(args)
    assert len(calls) == 2


def test_lq_stages_all_eight_source_bound_binaries_and_candidate_wheels(tmp_path):
    paths, directory = wheels(tmp_path), artifacts(tmp_path)
    uploads, argv = LQ.gate_plan('nvidia', VERSION, '/root/results-gate/test', paths, COMMIT, directory)
    assert len(uploads) == 3+8+1
    assert argv.count('--candidate-wheel') == 3
    assert '--source-commit' in argv and COMMIT in argv
    assert len([remote for _, remote in uploads if '/out/autograd-bin/' in remote]) == 9
    (directory/next(iter(A.expected('nvidia')))).write_bytes(b'replaced')
    with pytest.raises(ValueError, match='binary/provenance'):
        LQ.gate_plan('nvidia', VERSION, '/root/results-gate/test', paths, COMMIT, directory)


def test_published_gate_mode_stays_explicit_and_uploads_no_candidate(tmp_path):
    uploads, argv = LQ.gate_plan('amd', VERSION, '/root/results-gate/published')
    assert uploads == [] and '--candidate-wheel' not in argv


def test_autograd_resume_reuses_verified_completed_bytes_without_compiling(tmp_path, monkeypatch):
    monkeypatch.setattr(A.platform, 'system', lambda: 'Linux')
    monkeypatch.setattr(A.platform, 'machine', lambda: 'x86_64')
    root = tmp_path/'source'
    root.mkdir()
    compiler = tmp_path/'mojo'
    compiler.write_bytes(b'inert compiler fixture')
    compiler.chmod(0o755)
    for source in A.expected('nvidia').values():
        path = root/source
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('inert source fixture\n')
    monkeypatch.setattr(A, 'ROOT', root)
    monkeypatch.setattr(A.AS, 'ROOT', root)
    def output(command, **kwargs):
        if command[0] == 'git':
            return COMMIT+'\n' if 'rev-parse' in command else ''
        return 'fixture compiler version\n'
    calls = []
    def compile_fixture(command, **kwargs):
        calls.append(command)
        path = Path(command[command.index('-o')+1])
        path.write_bytes(b'inert output fixture')
        path.chmod(0o755)
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr(A.subprocess, 'check_output', output)
    monkeypatch.setattr(A.subprocess, 'run', compile_fixture)
    args = SimpleNamespace(out=str(tmp_path/'outputs'), source_commit=COMMIT, vendor='nvidia',
                           target='sm_89', mojo=str(compiler), compile_jobs=2, resume=False)
    assert A.build(args) == 0 and len(calls) == 8
    manifest = json.loads((Path(args.out)/'build-manifest.json').read_text())
    assert manifest['target_cpu'] == 'x86-64-v3'
    assert all(command[command.index('--target-cpu')+1] == 'x86-64-v3' for command in calls)
    args.resume = True
    assert A.build(args) == 0 and len(calls) == 8
    args.compile_jobs = 3
    with pytest.raises(RuntimeError, match='completed autograd artifact changed'):
        A.build(args)
    assert len(calls) == 8


@pytest.mark.parametrize('change', ['missing-receipt-target', 'host-native-command', 'missing-command-flag'])
def test_autograd_admission_refuses_unpinned_host_code(tmp_path, change):
    directory = artifacts(tmp_path)
    path = directory/'build-manifest.json'
    doc = json.loads(path.read_text())
    if change == 'missing-receipt-target':
        doc.pop('target_cpu')
    elif change == 'host-native-command':
        command = doc['files'][0]['command']
        command[command.index('--target-cpu')+1] = 'native'
    else:
        doc['files'][0]['command'] = ['mojo', 'build', doc['files'][0]['source']]
    path.write_text(json.dumps(doc))
    with pytest.raises(ValueError, match='host CPU baseline'):
        A.validate(path, 'nvidia', COMMIT)


@pytest.mark.parametrize('system,machine', [('Darwin', 'arm64'), ('Linux', 'aarch64')])
def test_gpu_gate_builder_refuses_a_different_host_abi(monkeypatch, system, machine):
    monkeypatch.setattr(A.platform, 'system', lambda: system)
    monkeypatch.setattr(A.platform, 'machine', lambda: machine)
    with pytest.raises(RuntimeError, match='Linux x86-64 build host'):
        A.build(SimpleNamespace())


def test_gate_receipt_cannot_reuse_published_or_other_candidate_hashes(tmp_path):
    manifest = C.build_manifest(VERSION, 'amd', wheels(tmp_path, 'amd'), COMMIT)
    identity = C.identity(manifest)
    output = tmp_path/'gate'
    output.mkdir()
    runner = object.__new__(release.Release)
    runner.version = VERSION
    runner.candidate_gate_identity = lambda vendor: identity
    runner.gate_out = lambda vendor: output
    for mode, candidate in [('published', None), ('candidate', dict(identity, wheels={'mojolearn-core': '0'*64}))]:
        (output/'gate-smoke.json').write_text(json.dumps(dict(version=VERSION, vendor='amd', verdict='PASS',
                                                           input_mode=mode, candidate=candidate)))
        ok, reason = runner.gate_ok('amd')
        assert not ok and 'candidate wheel hashes' in reason


def test_direct_publisher_call_refuses_before_any_external_action_without_both_gates():
    runner = object.__new__(release.Release)
    runner.args = SimpleNamespace(publish='pypi')
    runner.dry = False
    runner.gate_ok = lambda vendor: (vendor == 'nvidia', 'not qualified')
    with pytest.raises(release.StepFailed, match='both candidate gates'):
        runner.publish('linux', None, None)


def test_no_publication_precedes_both_candidate_gates_and_graph_has_no_cycles():
    graph = {name: set(deps) | set(release.AFTER.get(name, ())) for name, deps in release.NEEDS.items()}
    def ancestors(node, active=()):
        assert node not in active, 'dependency cycle: '+repr(active+(node,))
        return set(graph[node]) | set().union(*(ancestors(dep, active+(node,)) for dep in graph[node]))
    for name in graph:
        dependencies = ancestors(name)
        if name.startswith('publish-'):
            assert {'gate-nvidia', 'gate-amd'} <= dependencies
        if name in ('gate-' + v for v in release.GATE_VENDORS):  # gate-apple gates the Mac wheel only
            assert 'linux-pack' in dependencies
            assert not any(dep.startswith('publish-') for dep in dependencies)
