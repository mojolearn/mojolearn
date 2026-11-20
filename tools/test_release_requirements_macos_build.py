"""tools/release_requirements_macos_build.py: the macOS release build as a retained compile producer.

Fixture repositories, caches and wheels only; nothing compiles. The packaging
replay (install_name_tool/strip/codesign on real Mach-O) is replaced by a
deterministic stand-in, so these tests pin the chain logic, not the Apple tools.
"""
import copy
import hashlib
import json
from pathlib import Path
import subprocess
import zipfile

import pytest

import bincache
import release_compile_subunit_evidence as sub
import release_ledger_admission as admission
import release_requirements_macos_build as mrb
import release_retained_compile_evidence as compiled
from release_requirements import RequirementsLedger, digest

VERSION = '0.9.1'
TOOLCHAIN = dict(platform='osx-arm64', packages=['max-26.5.0-release', 'mojo-1.0.0-release'])
APPLE = dict(xcode='Xcode 26.6', sdk='26.5')
PINS = {'schema': 'mojolearn.adapter-pins.v1', 'torch': {'requirement': 'torch>=2.13,<2.14', 'series': '2.13', 'build': {
    'metal': {'version': '2.13.0', 'backend': 'mps', 'wheel_sha256': None}}},
    'max': {'requirement': 'max==26.5.0', 'version': '26.5.0'}}
DEVICE_FLAGS = dict(compile_jobs='1', gpu_archs=None, numeric_mode='identical', package_byte_lm='1')
HOST_FLAGS = dict(DEVICE_FLAGS, target_column='cpu')
STAGE = ('def add_rpath(p, v):\n    pass\n\n\ndef strip_build_rpaths(p):\n    return []\n\n\n'
         'def sign(p):\n    pass\n')
SHRINK = ('path = pathlib.Path(tmp) / "b.so"\nsubprocess.run(["strip", "-x", str(path)])\n'
          'subprocess.run(["codesign", "-f", "-s", "-", str(path)])\n')
GROUPS = {('identical', '_mojolearn_native'): ['_mojolearn', '_mojolearn_svm'],
          ('host', '_mojolearn_native_host'): ['_mojolearn_core_host']}


def put(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(value, bytes):
        path.write_bytes(value)
    else:
        path.write_text(value if isinstance(value, str) else json.dumps(value, indent=2, sort_keys=True) + '\n')
    return path


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def git(root, *args):
    return subprocess.check_output(['git', '-C', str(root), *args], text=True).strip()


def shipped(raw):
    """The stand-in packaging transform: shipped bytes are a function of the raw bytes."""
    return b'packaged:' + raw


@pytest.fixture(autouse=True)
def stand_in_replay(monkeypatch):
    monkeypatch.setattr(mrb, 'replay_tools', lambda: None)
    monkeypatch.setattr(mrb, 'replay', lambda raw, tier, stem, stage, scratch:
                        (hashlib.sha256(b'staged:' + raw).hexdigest(), hashlib.sha256(shipped(raw)).hexdigest()))
    monkeypatch.setitem(admission.ADAPTERS, mrb.NAME, ('release_requirements_macos_build', 'compile'))


@pytest.fixture
def repo(tmp_path):
    root = tmp_path / 'repo'
    root.mkdir()
    git(root, 'init', '-q')
    git(root, 'config', 'user.email', 'fixture@example.invalid')
    git(root, 'config', 'user.name', 'Fixture')
    put(root / 'tools/release_requirements.json', {'schema': 'fixture', 'requirements': []})
    put(root / 'tools/merged_bindings.py', 'MERGE = 1\n')
    put(root / 'bindings/merged_groups.json', {'dirs': {}})
    put(root / 'bindings/merged_build.sh', 'mojo build merged\n')
    put(root / 'bindings/build_host_family.sh', 'host family\n')
    put(root / 'bindings/native.mojo', 'fn native(): pass\n')
    put(root / 'core/host_predict_threads.mojo', 'fn threads(): pass\n')
    put(root / 'x_linear/ops.mojo', 'fn ops(): pass\n')
    put(root / 'adapters/native/c_api.mojo', 'fn c_api(): pass\n')
    put(root / 'adapters/torch/registration.cpp', 'int r;\n')
    put(root / 'packaging/adapters/pins.json', PINS)
    put(root / 'packaging/adapters/build.sh', 'mojo build adapters\n')
    put(root / 'packaging/macos/stage_dylibs.py', STAGE)
    put(root / 'packaging/macos/shrink_wheel.py', SHRINK)
    put(root / 'pixi.toml', '[workspace]\nname = "fixture"\n')
    put(root / 'pixi.lock', 'lock\n')
    git(root, 'add', '.')
    git(root, 'commit', '-qm', 'fixture source')
    return root


def fake_revision_closure(root, revision, script, args, environ, plat=None):
    """Stand-in for bincache.revision_closure (which walks the real merged-group plan): the
    group's closure digest is a function of bindings/native.mojo at REVISION and the group."""
    text = git(root, 'show', f'{revision}:bindings/native.mojo')
    stem = args[2]
    tier = 'host' if args[0] == 'host' else 'identical'
    return dict(source=dict(scope='closure', digest=hashlib.sha256((text + stem).encode()).hexdigest(), files=3),
                inputs=dict(digest=hashlib.sha256(b'bindings/merged_build.sh').hexdigest(),
                            files=['bindings/merged_build.sh']),
                merged=[dict(dir=tier, stem=stem, flags_class='host' if tier == 'host' else 'default',
                             members=GROUPS[(tier, stem)], root_sha256='a' * 64)],
                readers=({'MOJOLEARN_COMPILE_JOBS', 'MOJOLEARN_NUMERIC_MODE', 'MOJOLEARN_PACKAGE_BYTE_LM',
                          'MOJOLEARN_TARGET_COLUMN'}, []))


def side_commit(root, rel, text):
    """A commit on top of HEAD that changes REL, without moving HEAD (an earlier or other revision)."""
    import os
    import tempfile
    env = dict(os.environ, GIT_INDEX_FILE=tempfile.mktemp(dir=str(root.parent)))
    run = lambda *a, **k: subprocess.run(['git', '-C', str(root), *a], env=env, check=True, capture_output=True,
                                         text=True, **k).stdout.strip()
    blob = run('hash-object', '-w', '--stdin', input=text)
    run('read-tree', 'HEAD')
    run('update-index', '--add', '--cacheinfo', f'100644,{blob},{rel}')
    return run('commit-tree', run('write-tree'), '-p', 'HEAD', '-m', 'side')


def adapter_sources(root):
    rels = ['adapters/native/c_api.mojo', 'adapters/torch/registration.cpp', 'packaging/adapters/build.sh',
            'packaging/adapters/pins.json', 'x_linear/ops.mojo', 'core/host_predict_threads.mojo']
    return {rel: sha(root / rel) for rel in rels}


def adapter_manifest(root, revision, files, jobs='1'):
    native = ['pixi', 'run', 'mojo', 'build', '--emit', 'shared-lib', '-j', jobs, '-D', 'MOJOLEARN_NUMERIC_IDENTICAL=1',
              'adapters/native/c_api.mojo', '-o', '/out/libmojolearn_adapter.so']
    pin = PINS['torch']['build']['metal']
    return dict(schema='mojolearn.adapters.v1', abi=1, vendor='metal', arch='apple', platform='macos',
                source_commit=revision, numeric_mode='identical',
                torch=dict(version=pin['version'], series='2.13', backend='mps', wheel_sha256='2' * 64),
                torch_version_pin=pin['version'], max=dict(version='26.5.0'), mojo=dict(conda='1.0.0'),
                requirements=dict(torch=PINS['torch']['requirement'], max=PINS['max']['requirement']),
                commands=dict(native=native, max=['pixi', 'run', 'mojo', 'precompile', 'adapters/max_ops', '-o', 'x'],
                              torch=['c++', 'adapters/torch/registration.cpp', '-o', 'y']),
                sources={r: s for r, s in adapter_sources(root).items() if r.startswith(('adapters/', 'packaging/adapters/'))},
                files={role: dict(name=name, sha256=hashlib.sha256(files[name]).hexdigest(), bytes=len(files[name]))
                       for role, name in sub.ADAPTER_FILES.items()})


@pytest.fixture
def build(tmp_path, repo):
    """A retained release build: the bincache directory, the build log and the shipped wheel."""
    revision = git(repo, 'rev-parse', 'HEAD')
    tree = compiled.Tree.at(repo, revision)
    toolchain = dict(TOOLCHAIN, pixi_lock_sha256=sha(repo / 'pixi.lock'))
    out = tmp_path / 'build'

    def fields_for(tier, stem, source=None, merged_key=None, env=None):
        environment = {'MOJOLEARN_COMPILE_JOBS': '1', 'MOJOLEARN_MERGED_BINDINGS': '1',
                       'MOJOLEARN_MERGED_KEY': merged_key or mrb._merged_key(tree),
                       'MOJOLEARN_NUMERIC_MODE': 'identical', 'MOJOLEARN_PACKAGE_BYTE_LM': '1',
                       'MOJOLEARN_REUSE_DIR': '', 'MOJOLEARN_REUSE_PLAN': '',
                       # never compiler inputs (release_requirements.build_flags drops them)
                       'MOJOLEARN_RUNPOD_KEY_FILE': '/home/key', 'MOJOLEARN_SLOT_TOKEN': 'abc123'}
        if tier == 'host':
            environment['MOJOLEARN_TARGET_COLUMN'] = 'cpu'
        environment.update(env or {})
        inputs = ['bindings/merged_build.sh']
        return dict(schema=bincache.SCHEMA, script='bindings/merged_build.sh',
                    args=[mrb.KINDS[tier], '--group', stem], source=source or sub._tree_digest(tree, mrb.SCRIPT),
                    toolchain=toolchain, numeric_mode='identical', gpu_archs='', target_column='',
                    build_env=environment, device_arch='none', image='local', os=dict(id='darwin'),
                    repo_path=str(repo),
                    local=dict(inputs=dict(digest=mrb._inputs_digest(tree, inputs), files=inputs),
                               outputs=[f'python/mojolearn/{tier}/{stem}.so'], shell='sh', host_toolchain=APPLE))

    def fields_v2(tier, stem, at=None):
        """A closure-v2 key (tools/bincache.py KEY_RULE) of the group at revision AT (default the wheel's)."""
        want = fake_revision_closure(repo, at or revision, mrb.SCRIPT, [mrb.KINDS[tier], '--group', stem], {})
        environment = {'MOJOLEARN_COMPILE_JOBS': '1', 'MOJOLEARN_NUMERIC_MODE': 'identical',
                       'MOJOLEARN_PACKAGE_BYTE_LM': '1'}
        if tier == 'host':
            environment['MOJOLEARN_TARGET_COLUMN'] = 'cpu'
        return dict(schema=bincache.SCHEMA, key_rule=bincache.KEY_RULE, script=mrb.SCRIPT,
                    args=[mrb.KINDS[tier], '--group', stem], source=want['source'], inputs=want['inputs'],
                    merged=want['merged'], toolchain=toolchain, numeric_mode='identical', gpu_archs='',
                    target_column='', build_env=environment, image='local', os=dict(id='darwin'),
                    repo_path=str(repo),
                    local=dict(outputs=[f'python/mojolearn/{tier}/{stem}.so'], shell='sh', host_toolchain=APPLE))

    adapter_files = {'libmojolearn_adapter.so': b'adapter', 'mojolearn_torch.so': b'torch', 'mojolearn_ops.mojoc': b'ops'}

    def write(outcome='miss+built-cached', fields=None, wheel_bytes=None, jobs='1', v2=False, origin=None,
              placement=True, named=True):
        """OUTCOME: one build-log outcome for every group, or {stem: outcome}. A `hit` group's
        archive names its original compile (ORIGIN, default the wheel's revision) unless NAMED
        is False, and the cache's placements.jsonl echoes it unless PLACEMENT is False."""
        if out.exists():
            import shutil
            shutil.rmtree(out)
        cache = out / 'cache'
        rows, log, placed = [], ['== building %d merged files, 4 at a time' % len(GROUPS)], []
        for (tier, stem) in GROUPS:
            how = outcome.get(stem, 'miss+built-cached') if isinstance(outcome, dict) else outcome
            f = (fields or {}).get(stem) or (fields_v2(tier, stem) if v2 else fields_for(tier, stem))
            key = bincache.key_of(f)
            rel = f['local']['outputs'][0]
            stage = out / 'stage'
            put(stage / rel, ('raw ' + stem).encode())
            cache.mkdir(parents=True, exist_ok=True)
            compiled_by = dict(source_commit=origin or revision, built='2026-11-07T09:12:00Z', build_seconds=512.5,
                               builder=dict(host='fixture-mac', leg='local'))
            extra = compiled_by if (how != 'hit' or named) else {}
            bincache.pack(str(cache / (key + '.tar.gz')), key, f, stage, [rel], extra)
            put(cache / 'provenance/keys' / (key + '.json'), f)
            files = f'{rel}={sha(stage / rel)}'
            rows.append(f'2026-10-31T18:00:00Z\tbindings/merged_build.sh\t{how}\t{key}\t1.0\t{files}')
            if how == 'hit':
                log.append(f'BINCACHE seeded bindings/merged_build.sh key={key[:16]} from /fixture/bincache-mac')
                if placement:
                    placed.append(json.dumps(dict(
                        script='bindings/merged_build.sh', outcome='hit', key=key,
                        files=[dict(path=rel, sha256=sha(stage / rel))],
                        archive_source_commit=extra.get('source_commit', ''), archive_built=extra.get('built', ''),
                        archive_build_seconds=extra.get('build_seconds'), archive_builder=extra.get('builder', {})),
                        sort_keys=True))
            log.append(f'BINCACHE {how} bindings/merged_build.sh key={key[:16]} seconds=1.0')
        put(cache / 'provenance/provenance.tsv', '\n'.join(rows) + '\n')
        if placed:
            put(cache / 'provenance/placements.jsonl', '\n'.join(placed) + '\n')
        put(out / 'macos-build.log', '\n'.join(log) + '\nshrink_wheel: done\n')
        wheel = out / f'mojolearn-{VERSION}-py3-none-macosx_26_0_arm64.whl'
        wheel.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(wheel, 'w') as archive:
            archive.writestr('mojolearn/identity_columns/COMMIT', revision + '\n')
            for tier in mrb.KINDS:
                archive.writestr(f'mojolearn/{tier}/_mojolearn_merged.json', json.dumps(dict(format=1, files={
                    stem + '.so': members for (t, stem), members in GROUPS.items() if t == tier})))
            for (tier, stem) in GROUPS:
                data = (wheel_bytes or {}).get(stem) or shipped(('raw ' + stem).encode())
                archive.writestr(f'mojolearn/{tier}/{stem}.so', data)
            for name, data in adapter_files.items():
                archive.writestr('mojolearn/_adapters/' + name, data)
            archive.writestr('mojolearn/_adapters/adapters.json',
                             json.dumps(adapter_manifest(repo, revision, adapter_files, jobs)))
        return dict(wheel=str(wheel), log=str(out / 'macos-build.log'), cache=str(cache))

    rels = ['bindings/native.mojo', 'pixi.lock']
    builders = {'packaging/macos/build_release_wheel.sh': 'f' * 64}
    unit = dict(schema='mojolearn.release-dependencies.v1', kind='compile', vendor='apple', release_modes=['identical'],
                build_flags={}, compile_variants=[])

    def identity(tier, name, flags=None):
        host = tier == 'host'
        return dict(schema='mojolearn.binding-identity.v2', target='osx-arm64', vendor='host' if host else 'metal',
                    arch='' if host else 'apple', tier=tier, name=name, script='build.sh', toolchain=toolchain,
                    host_toolchain=APPLE, closure=dict(scope='closure', digest=compiled.closure_digest(tree, rels)),
                    generators=None, flags=flags or (HOST_FLAGS if host else DEVICE_FLAGS), builders=builders,
                    builder_rule='builder-view.v1')

    def row(rid, inputs, coverage, sources):
        inputs = dict(inputs, configuration={}, contract=dict(id=rid, scope='Integrated identical build',
                                                              coverage=coverage, kind='compile', stage='pre'))
        return rid, dict(source_revision=revision, inputs=inputs, fingerprint=digest(inputs),
                         scope='Integrated identical build', coverage=coverage, missing_inputs=[], kind='compile',
                         stage='pre', provenance_sources=sorted(sources), required=True,
                         parent_requirement=rid.split('/', 1)[0])

    def originals(flags=None):
        rows_ = []
        for (tier, _stem), members in GROUPS.items():
            for name in members:
                b = identity(tier, name, (flags or {}).get(name))
                key = compiled.binding_key(b)
                rows_.append(row('compile-apple/' + key, dict(unit=unit, binding=b), [compiled.BASE_COVERAGE, key],
                                 rels + list(builders)))
        environment = dict(target='osx-arm64', vendor='metal', arch='apple', toolchain=toolchain, flags=DEVICE_FLAGS,
                           host_toolchain=APPLE, builders=builders)
        srcs = adapter_sources(repo)
        rows_.append(row('compile-apple/adapters/apple', dict(unit=unit, adapter_sources=srcs, environment=environment),
                         sub.ADAPTER_COVERAGE, list(srcs) + list(builders)))
        return dict(rows_)

    return dict(write=write, originals=originals, fields_for=fields_for, fields_v2=fields_v2, revision=revision)


def ledger_for(path, wanted):
    ledger = RequirementsLedger(path / 'ledger.json')
    revision = next(iter(wanted.values()))['source_revision']
    ledger.sync(dict(schema='mojolearn.release-requirements.v1', version=VERSION, source_revision=revision,
                     requirements=copy.deepcopy(wanted), source_root='/fixture',
                     configuration=dict(dependency_context=dict(cpu_target='linux-64', apple_toolchain=APPLE))))
    return ledger


def use(monkeypatch, rows):
    monkeypatch.setattr(admission, 'reconstruct', lambda *a, **k: copy.deepcopy(rows))


def prepare(repo, path, wanted, producer):
    return admission.prepare(mrb.NAME, repo, ledger_for(path, wanted).path, path / 'review', producer)


def test_admits_every_compiled_binding_and_the_adapters(tmp_path, repo, build, monkeypatch):
    wanted = build['originals']()
    assert all(mrb.supports_requirement(w) for w in wanted.values())
    use(monkeypatch, wanted)
    producer = build['write']()
    review = prepare(repo, tmp_path, wanted, producer)
    assert review['admitted'] == sorted(wanted) and not review['refused']
    assert review['reconstruction_configuration']['dependency_context']['cpu_target'] == 'linux-64'
    result = admission.admit(tmp_path / 'ledger.json', repo, tmp_path / 'review/report.json')
    assert set(result['actions'].values()) == {'reuse'}
    plan = RequirementsLedger(tmp_path / 'ledger.json').plan()
    assert all(plan[name]['status'] == 'passed' for name in wanted)


def test_the_report_reverifies_from_the_retained_bytes(tmp_path, repo, build, monkeypatch):
    wanted = build['originals']()
    use(monkeypatch, wanted)
    producer = build['write']()
    prepare(repo, tmp_path, wanted, producer)
    report = json.loads((tmp_path / 'review/report.json').read_text())
    assert mrb.ADAPTERS[mrb.NAME].verify(report) == report['facts']
    # a later change to the shipped wheel breaks the retained chain
    build['write'](wheel_bytes={'_mojolearn_native': b'other bytes'})
    with pytest.raises(ValueError, match='shipped bytes are not the compiled bytes'):
        mrb.ADAPTERS[mrb.NAME].verify(report)


@pytest.mark.parametrize('case, match', [
    ('hit', "a placed group's compile key is not closure-v2"),
    ('shipped', 'shipped bytes are not the compiled bytes after packaging'),
    ('source', 'compiled source does not reproduce'),
    ('merged', 'merged group plan differs'),
])
def test_a_broken_chain_admits_nothing(tmp_path, repo, build, monkeypatch, case, match):
    wanted = build['originals']()
    use(monkeypatch, wanted)
    if case == 'hit':
        producer = build['write'](outcome='hit')
    elif case == 'shipped':
        producer = build['write'](wheel_bytes={'_mojolearn_native_host': b'rebuilt elsewhere'})
    elif case == 'source':
        producer = build['write'](fields={'_mojolearn_native': build['fields_for'](
            'identical', '_mojolearn_native', source=dict(scope='tree', digest='0' * 64, files=1))})
    else:
        producer = build['write'](fields={'_mojolearn_native': build['fields_for'](
            'identical', '_mojolearn_native', merged_key='1' * 64)})
    with pytest.raises(ValueError, match=match):
        prepare(repo, tmp_path, wanted, producer)


def test_tampered_archive_bytes_are_refused(tmp_path, repo, build, monkeypatch):
    wanted = build['originals']()
    use(monkeypatch, wanted)
    producer = build['write']()
    import tarfile
    arc = next(Path(producer['cache']).glob('*.tar.gz'))
    with tarfile.open(arc) as tf:
        members = {m.name: tf.extractfile(m).read() for m in tf.getmembers()}
    with tarfile.open(arc, 'w:gz') as tf:
        for name, data in members.items():
            data = data if name == 'manifest.json' else b'swapped'
            info = tarfile.TarInfo(name)
            info.size = len(data)
            import io
            tf.addfile(info, io.BytesIO(data))
    with pytest.raises(ValueError, match='sha256 mismatch'):
        prepare(repo, tmp_path, wanted, producer)


def test_other_flags_are_refused_per_unit(tmp_path, repo, build, monkeypatch):
    wanted = build['originals'](flags={'_mojolearn_svm': dict(DEVICE_FLAGS, compile_jobs='2')})
    use(monkeypatch, wanted)
    review = prepare(repo, tmp_path, wanted, build['write']())
    svm = 'compile-apple/osx-arm64/metal/apple/identical/_mojolearn_svm'
    assert 'MOJOLEARN_COMPILE_JOBS' in review['refused'][svm]
    assert svm not in review['admitted'] and len(review['admitted']) == len(wanted) - 1
    # an undeclared build switch in the compile environment is refused as well
    env = {'MOJOLEARN_BUILD_DEFINES': 'X=1'}
    use(monkeypatch, build['originals']())
    review = admission.prepare(mrb.NAME, repo, ledger_for(tmp_path / 'b', build['originals']()).path,
                               tmp_path / 'b/review', build['write'](fields={'_mojolearn_native_host': build['fields_for'](
                                   'host', '_mojolearn_native_host', env=env)}))
    host = 'compile-apple/osx-arm64/host/-/host/_mojolearn_core_host'
    assert 'declared build flags differ' in review['refused'][host]


def test_adapters_with_other_compile_jobs_are_refused(tmp_path, repo, build, monkeypatch):
    wanted = build['originals']()
    use(monkeypatch, wanted)
    review = prepare(repo, tmp_path, wanted, build['write'](jobs='4'))
    assert 'compile jobs' in review['refused']['compile-apple/adapters/apple']


def test_supports_only_the_mac_contract():
    assert not mrb.supports_requirement(dict(kind='compile', inputs=dict(binding=dict(target='linux-64'))))
    assert not mrb.supports_requirement(dict(kind='source', inputs={}))


# ---------------------------------------------------------------- closure-v2 keys and placed groups (2026-11-07)


@pytest.fixture
def v2(monkeypatch):
    monkeypatch.setattr(bincache, 'revision_closure', fake_revision_closure)


def test_closure_v2_groups_compiled_in_this_build_are_admitted(tmp_path, repo, build, monkeypatch, v2):
    wanted = build['originals']()
    use(monkeypatch, wanted)
    review = prepare(repo, tmp_path, wanted, build['write'](v2=True))
    assert review['admitted'] == sorted(wanted) and not review['refused']
    facts = json.loads((tmp_path / 'review/report.json').read_text())['facts']
    assert facts['compiles'] == dict(built=sorted(s for _, s in GROUPS), placed=[])
    assert facts['groups']['_mojolearn_native']['compile']['outcome'] == 'built'


def test_placed_groups_are_admitted_with_their_original_compile(tmp_path, repo, build, monkeypatch, v2):
    wanted = build['originals']()
    use(monkeypatch, wanted)
    producer = build['write'](v2=True, outcome='hit')
    review = prepare(repo, tmp_path, wanted, producer)
    assert review['admitted'] == sorted(wanted) and not review['refused']
    report = json.loads((tmp_path / 'review/report.json').read_text())
    facts = report['facts']
    assert facts['compiles'] == dict(built=[], placed=sorted(s for _, s in GROUPS))
    assert facts['groups']['_mojolearn_native_host']['compile'] == dict(
        outcome='placed', source_commit=build['revision'], compiled_at='2026-11-07T09:12:00Z', build_seconds=512.5,
        builder=dict(host='fixture-mac', leg='local'))
    assert any(e['path'].endswith('placements.jsonl') for e in facts['evidence'])
    assert mrb.ADAPTERS[mrb.NAME].verify(report) == facts
    result = admission.admit(tmp_path / 'ledger.json', repo, tmp_path / 'review/report.json')
    assert set(result['actions'].values()) == {'reuse'}


def test_one_build_may_compile_some_groups_and_place_others(tmp_path, repo, build, monkeypatch, v2):
    wanted = build['originals']()
    use(monkeypatch, wanted)
    review = prepare(repo, tmp_path, wanted, build['write'](v2=True, outcome={'_mojolearn_native': 'hit'}))
    assert review['admitted'] == sorted(wanted)
    facts = json.loads((tmp_path / 'review/report.json').read_text())['facts']
    assert facts['compiles'] == dict(built=['_mojolearn_native_host'], placed=['_mojolearn_native'])


def test_a_group_compiled_at_another_revision_with_the_same_closure_is_placed(tmp_path, repo, build, monkeypatch, v2):
    earlier = side_commit(repo, 'docs/unrelated.md', 'not in any closure\n')
    wanted = build['originals']()
    use(monkeypatch, wanted)
    review = prepare(repo, tmp_path, wanted, build['write'](v2=True, outcome='hit', origin=earlier))
    assert review['admitted'] == sorted(wanted)
    facts = json.loads((tmp_path / 'review/report.json').read_text())['facts']
    assert {g['compile']['source_commit'] for g in facts['groups'].values()} == {earlier}


@pytest.mark.parametrize('case, match', [
    ('no-placement-record', 'bincache placement record is missing'),
    ('archive-names-no-compile', 'does not name its original compile'),
    ('stale-key', 'compiled source does not reproduce from the tested revision'),
    ('origin-elsewhere', "key does not reproduce at the revision it names"),
    ('merged-key', 'compiled source does not reproduce from the tested revision'),
])
def test_a_placed_group_without_its_compile_admits_nothing(tmp_path, repo, build, monkeypatch, v2, case, match):
    wanted = build['originals']()
    use(monkeypatch, wanted)
    if case == 'no-placement-record':
        producer = build['write'](v2=True, outcome='hit', placement=False)
    elif case == 'archive-names-no-compile':
        producer = build['write'](v2=True, outcome='hit', named=False)
    elif case == 'stale-key':
        older = side_commit(repo, 'bindings/native.mojo', 'fn native(): return\n')
        producer = build['write'](v2=True, outcome='hit', origin=older, fields={
            '_mojolearn_native': build['fields_v2']('identical', '_mojolearn_native', at=older)})
    elif case == 'origin-elsewhere':
        other = side_commit(repo, 'bindings/native.mojo', 'fn native(): return\n')
        producer = build['write'](v2=True, outcome='hit', origin=other)
    else:
        f = build['fields_v2']('identical', '_mojolearn_native')
        f['build_env']['MOJOLEARN_MERGED_KEY'] = '1' * 64
        producer = build['write'](v2=True, outcome='hit', fields={'_mojolearn_native': f})
    with pytest.raises(ValueError, match=match):
        prepare(repo, tmp_path, wanted, producer)


def test_the_build_log_admits_only_compiled_or_placed_groups(tmp_path, repo, build, monkeypatch, v2):
    wanted = build['originals']()
    use(monkeypatch, wanted)
    producer = build['write'](v2=True, outcome='bypass-destination-exists+built-not-cached:conflict')
    with pytest.raises(ValueError, match='neither compiled and cached in this build nor placed'):
        prepare(repo, tmp_path, wanted, producer)
