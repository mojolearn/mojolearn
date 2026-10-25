"""Compile sub-unit admissions (runtime, adapters, Apple pack bindings) on synthetic producers.

Nothing is compiled, installed or run: each test writes the retained records a
producer leaves and checks that only exact inputs are admitted.
"""
import copy
import hashlib
import json
from pathlib import Path, PurePosixPath
import subprocess
import zipfile

import pytest

import bincache
import release_compile_subunit_evidence as sub
import release_host_composition as witness
import release_ledger_admission as admission
import release_retained_compile_evidence as compiled
import release_reuse
from release_requirements import RequirementsLedger, digest, producer_admission_available

VERSION = '0.9.1'
LINUX_TOOLCHAIN = dict(platform='linux-64', packages=['max-26.5.0-release', 'mojo-1.0.0-release'],
                       pixi_lock_sha256='c' * 64)
MAC_TOOLCHAIN = dict(platform='osx-arm64', packages=['max-26.5.0-release', 'mojo-1.0.0-release'],
                     pixi_lock_sha256='c' * 64)
APPLE = dict(xcode='Xcode 26.6', sdk='26.5')
COMPILER = 'e' * 64
IMAGE = 'rocm/dev@sha256:' + 'f' * 64
PINS = {'schema': 'mojolearn.adapter-pins.v1', 'torch': {'requirement': 'torch>=2.13,<2.14', 'series': '2.13', 'build': {
    'hip': {'version': '2.13.0+rocm7.2', 'backend': 'rocm', 'wheel_sha256': '1' * 64},
    'metal': {'version': '2.13.0', 'backend': 'mps', 'wheel_sha256': None}}},
    'max': {'requirement': 'max==26.5.0', 'version': '26.5.0'}}
LEG_FLAGS = dict(compile_jobs='8', gpu_archs='gfx942', linux_cpu='x86-64-v3', numeric_mode='identical',
                 package_byte_lm='1', target_column='amd')
MAC_FLAGS = dict(compile_jobs='1', gpu_archs=None, numeric_mode='identical', package_byte_lm='1')


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


@pytest.fixture
def repo(tmp_path):
    root = tmp_path / 'repo'
    root.mkdir()
    git(root, 'init', '-q')
    git(root, 'config', 'user.email', 'fixture@example.invalid')
    git(root, 'config', 'user.name', 'Fixture')
    put(root / 'tools/release_requirements.json', {'schema': 'fixture', 'requirements': []})
    put(root / 'packaging/portable_math/portable_math.c', 'double mojolearn_log10(double x);\n')
    put(root / 'packaging/portable_math/powers_of_ten.h', '#define P 10\n')
    put(root / 'packaging/adapters/pins.json', PINS)
    put(root / 'packaging/adapters/build.sh', 'mojo build adapters\n')
    put(root / 'adapters/native/c_api.mojo', 'fn c_api(): pass\n')
    put(root / 'adapters/torch/registration.cpp', 'int r;\n')
    put(root / 'x_linear/ops.mojo', 'fn ops(): pass\n')
    # Reached from adapters/native/c_api.mojo by import, outside adapters/ and every device
    # closure (the real 0.8.57 pack shape: core/host_predict_threads.mojo, sequence/*.mojo).
    put(root / 'core/host_predict_threads.mojo', 'fn threads(): pass\n')
    put(root / 'bindings/native.mojo', 'fn native(): pass\n')
    put(root / 'bindings/merged_build.sh', 'mojo build merged\n')
    put(root / 'bindings/build_host_family.sh', 'host family\n')
    put(root / 'pixi.toml', '[workspace]\nname = "fixture"\n')
    put(root / 'pixi.lock', 'lock\n')
    git(root, 'add', '.')
    git(root, 'commit', '-qm', 'fixture source')
    return root


def ledger_for(tmp_path, wanted, configuration=None):
    ledger = RequirementsLedger(tmp_path / 'ledger.json')
    revision = next(iter(wanted.values()))['source_revision']
    ledger.sync(dict(schema='mojolearn.release-requirements.v1', version=VERSION, source_revision=revision,
                     requirements=copy.deepcopy(wanted), source_root='/fixture', configuration=configuration or {}))
    return ledger


def row(revision, rid, inputs, coverage, sources=()):
    inputs = dict(inputs, configuration={}, contract=dict(id=rid, scope='Integrated identical build',
                                                          coverage=coverage, kind='compile', stage='pre'))
    return rid, dict(source_revision=revision, inputs=inputs, fingerprint=digest(inputs), scope='Integrated identical build',
                     coverage=coverage, missing_inputs=[], kind='compile', stage='pre', provenance_sources=sorted(sources),
                     required=True, parent_requirement=rid.split('/', 1)[0])


def adapter_sources(root):
    rels = ['adapters/native/c_api.mojo', 'adapters/torch/registration.cpp', 'packaging/adapters/build.sh',
            'packaging/adapters/pins.json', 'x_linear/ops.mojo']
    return {rel: sha(root / rel) for rel in rels}


def adapter_manifest(root, revision, vendor, arch, platform, files, flags, abi_define=True):
    native = ['pixi', 'run', 'mojo', 'build', '--emit', 'shared-lib', '-j', flags['compile_jobs'],
              '-D', 'MOJOLEARN_NUMERIC_IDENTICAL=1']
    if flags.get('linux_cpu'):
        native += ['--target-cpu', flags['linux_cpu'], '--target-accelerator', arch]
    if flags.get('target_column'):
        native += ['-D', 'MOJOLEARN_COLUMN_' + flags['target_column'].upper()]
    native += ['adapters/native/c_api.mojo', '-o', '/out/libmojolearn_adapter.so']
    pin = PINS['torch']['build'][vendor]
    return dict(schema='mojolearn.adapters.v1', abi=1, vendor=vendor, arch=arch, platform=platform,
                source_commit=revision, numeric_mode='identical',
                torch=dict(version=pin['version'], series='2.13', backend=pin['backend'], wheel_sha256=pin['wheel_sha256'] or '2' * 64),
                torch_version_pin=pin['version'], torch_version_built=pin['version'],
                max=dict(version='26.5.0'), mojo=dict(version='Mojo 1.0.0', conda='1.0.0'),
                requirements=dict(torch=PINS['torch']['requirement'], max=PINS['max']['requirement']),
                commands=dict(native=native, max=['pixi', 'run', 'mojo', 'precompile', 'adapters/max_ops', '-o', 'x'],
                              torch=['c++', 'adapters/torch/registration.cpp', '-o', 'y']),
                sources={r: s for r, s in adapter_sources(root).items() if r.startswith(('adapters/', 'packaging/adapters/'))},
                files={role: dict(name=name, sha256=hashlib.sha256(files[name]).hexdigest(), bytes=len(files[name]))
                       for role, name in sub.ADAPTER_FILES.items()})


# ---------------------------------------------------------------- Linux leg
@pytest.fixture
def leg(tmp_path, repo):
    revision = git(repo, 'rev-parse', 'HEAD')
    legs = tmp_path / 'legs'
    base = legs / 'hip-gfx942'
    build = base / 'hip-gfx942/release-build/build'
    set_root = build / 'sets/hip/gfx942'
    runtime = dict(schema='mojolearn.binding-identity.v2', target='linux-64', what='runtime', toolchain=LINUX_TOOLCHAIN,
                   image=dict(cuda='runpod/pytorch:x', hip=IMAGE), portable_math='9' * 64, builders={}, builder_rule='v1')
    device = dict(target='linux-64', vendor='hip', arch='gfx942', tier='identical', name='_mojolearn', toolchain=LINUX_TOOLCHAIN,
                  flags=LEG_FLAGS, image=IMAGE, closure=dict(scope='closure', digest='0' * 64))
    host = dict(device, vendor='host', arch='', tier='host', name='_mojolearn_core_host', image=dict(cuda='a', hip=IMAGE))
    inventory = [[rel, sha(repo / rel)] for rel in ('adapters/native/c_api.mojo', 'adapters/torch/registration.cpp',
                 'bindings/native.mojo', 'x_linear/ops.mojo', 'packaging/adapters/build.sh', 'packaging/adapters/pins.json')]
    put(build / 'source-inventory.json', inventory)
    capture = put(build / 'compile-inputs.json', dict(
        schema='mojolearn.linux.release-compile-inputs.v1', phase='before-build', source_commit=revision, vendor='hip',
        arch='gfx942', toolchain=LINUX_TOOLCHAIN, installed_packages=LINUX_TOOLCHAIN['packages'] + ['python-3.14'],
        compiler_sha256=COMPILER, source_inventory_sha256=sha(build / 'source-inventory.json'),
        source_inventory_digest=digest(inventory), build_flags={'MOJOLEARN_GPU_ARCHS': 'gfx942', 'MOJOLEARN_TARGET_COLUMN': 'amd'},
        binding_identities={compiled.binding_key(device): device, compiled.binding_key(host): host}))
    libs = {'libAsyncRTMojoBindings.so': b'runtime bytes', 'libMojolearnMath.so': b'math bytes'}
    for name, data in libs.items():
        put(set_root / '.libs' / name, data)
    files = {'libmojolearn_adapter.so': b'adapter', 'mojolearn_torch.so': b'torch', 'mojolearn_ops.mojoc': b'ops'}
    for name, data in files.items():
        put(set_root / '_adapters' / name, data)
    manifest = adapter_manifest(repo, revision, 'hip', 'gfx942', 'linux', files, LEG_FLAGS)
    put(set_root / '_adapters/adapters.json', manifest)
    set_manifest = put(set_root / 'manifest.json', dict(
        staged_libs=[dict(name=n, sha256=hashlib.sha256(d).hexdigest(), bytes=len(d)) for n, d in libs.items()],
        portable_math=dict(helper='libMojolearnMath.so', build=dict(
            source_sha256=sha(repo / 'packaging/portable_math/portable_math.c'),
            constants_sha256=sha(repo / 'packaging/portable_math/powers_of_ten.h'),
            output_sha256=hashlib.sha256(libs['libMojolearnMath.so']).hexdigest(), command=['cc'])),
        adapters=[dict(path='_adapters/' + n, sha256=hashlib.sha256(d).hexdigest()) for n, d in files.items()
                  if n.endswith('.so')]))
    put(build / 'build-provenance.json', dict(
        schema='mojolearn.linux.build-provenance.v1', action='build', complete=True, build_exit=0, source_commit=revision,
        release_modes=['identical'], source_inventory=inventory,
        source_sha256=hashlib.sha256(json.dumps(inventory, separators=(',', ':')).encode()).hexdigest(),
        compile_inputs_sha256=sha(capture), set_manifests={'mojolearn/hip/gfx942/manifest.json': sha(set_manifest)},
        adapters={'mojolearn/hip/gfx942/_adapters': dict(manifest_sha256=sha(set_root / '_adapters/adapters.json'),
            files=dict({n: hashlib.sha256(d).hexdigest() for n, d in files.items()},
                       **{'adapters.json': sha(set_root / '_adapters/adapters.json')}))}))
    put(legs / 'hip-gfx942.exit', '0\n')
    put(base / 'LEG/create_request.json', dict(imageName=IMAGE))
    put(legs / 'hip-gfx942.provenance.json', dict(
        schema='mojolearn.release-leg-provenance.v1', leg='hip-gfx942', vendor='hip', arch='gfx942',
        source_commit=revision, tooling_commit=revision, tooling_dirty=[], overlay=None, route_overlay_env={},
        release_build=str(base / 'hip-gfx942/release-build'),
        set_identity=dict(runtime=release_reuse.digest_of(runtime))))
    unit = dict(schema='mojolearn.release-dependencies.v1', kind='compile', vendor='amd', release_modes=['identical'],
                build_flags={}, compile_variants=[], compiler_sha256=COMPILER)
    environment = {k: device[k] for k in ('target', 'vendor', 'arch', 'toolchain', 'flags', 'image')}
    environment['builders'] = {'packaging/linux/build_sets.sh': '3' * 64}

    def originals(runtime_identity=runtime, sources=None, env=environment):
        srcs = sources or adapter_sources(repo)
        return dict([
            row(revision, 'compile-amd/runtime', dict(unit=unit, runtime=runtime_identity), sub.RUNTIME_COVERAGE),
            row(revision, 'compile-amd/adapters/gfx942', dict(unit=unit, adapter_sources=srcs, environment=env),
                sub.ADAPTER_COVERAGE, list(srcs) + list(env['builders']))])

    return dict(dir=base, legs=legs, revision=revision, originals=originals, runtime=runtime, environment=environment,
                set_root=set_root, build=build)


def use(monkeypatch, rows):
    monkeypatch.setattr(admission, 'reconstruct', lambda *a, **k: copy.deepcopy(rows))


def test_leg_runtime_and_adapters_are_admitted_and_reused(tmp_path, repo, leg, monkeypatch):
    wanted = leg['originals']()
    assert all(sub.supports_requirement(w) and producer_admission_available(w) for w in wanted.values())
    use(monkeypatch, wanted)
    ledger = ledger_for(tmp_path, wanted)
    review = admission.prepare('linux-release-leg', repo, ledger.path, tmp_path / 'review', dict(leg=leg['dir']))
    assert review['admitted'] == sorted(wanted) and not review['refused']
    assert review['reconstruction_configuration']['dependency_context']['compiler_sha256'] == {'linux-64': COMPILER}
    result = admission.admit(ledger.path, repo, tmp_path / 'review/report.json')
    assert set(result['actions'].values()) == {'reuse'}
    plan = ledger.plan()
    assert all(plan[name]['status'] == 'passed' for name in wanted)


def refused(tmp_path, repo, leg, rows, monkeypatch):
    use(monkeypatch, rows)
    ledger = ledger_for(tmp_path, rows)
    with pytest.raises(ValueError, match='no requirement could be admitted'):
        admission.prepare('linux-release-leg', repo, ledger.path, tmp_path / 'review', dict(leg=leg['dir']))
    return json.loads((tmp_path / 'review/review.json').read_text())['refused']


def test_leg_refuses_other_runtime_identity_flags_and_unbound_sources(tmp_path, repo, leg, monkeypatch):
    env = copy.deepcopy(leg['environment'])
    env['flags'] = dict(env['flags'], compile_jobs='2')
    sources = dict(adapter_sources(repo), **{'bindings/merged_build.sh': sha(repo / 'bindings/merged_build.sh')})
    rows = leg['originals'](runtime_identity=dict(leg['runtime'], portable_math='8' * 64), env=env)
    reasons = refused(tmp_path / 'a', repo, leg, rows, monkeypatch)
    assert 'another runtime identity' in reasons['compile-amd/runtime']
    assert 'device identities differ' in reasons['compile-amd/adapters/gfx942']
    rows = leg['originals'](sources=sources)
    use(monkeypatch, rows)
    review = admission.prepare('linux-release-leg', repo, ledger_for(tmp_path / 'b', rows).path,
                               tmp_path / 'b/review', dict(leg=leg['dir']))
    assert review['admitted'] == ['compile-amd/runtime']
    assert 'not bound to the compiled bytes' in review['refused']['compile-amd/adapters/gfx942']


def test_leg_changed_bytes_dirty_tooling_and_ptx_leg_are_refused(tmp_path, repo, leg, monkeypatch):
    use(monkeypatch, leg['originals']())
    ledger = ledger_for(tmp_path, leg['originals']())
    lib = leg['set_root'] / '.libs/libAsyncRTMojoBindings.so'
    lib.write_bytes(b'other')
    with pytest.raises(ValueError, match='staged runtime library differs'):
        admission.prepare('linux-release-leg', repo, ledger.path, tmp_path / 'a', dict(leg=leg['dir']))
    lib.write_bytes(b'runtime bytes')
    adapter = leg['set_root'] / '_adapters/mojolearn_torch.so'
    adapter.write_bytes(b'other')
    with pytest.raises(ValueError, match='shipped adapter bytes'):
        admission.prepare('linux-release-leg', repo, ledger.path, tmp_path / 'b', dict(leg=leg['dir']))
    adapter.write_bytes(b'torch')
    launch_path = leg['legs'] / 'hip-gfx942.provenance.json'
    launch = json.loads(launch_path.read_text())
    put(launch_path, dict(launch, tooling_dirty=['tools/x.py']))
    with pytest.raises(ValueError, match='dirty or overlaid'):
        admission.prepare('linux-release-leg', repo, ledger.path, tmp_path / 'c', dict(leg=leg['dir']))
    put(launch_path, launch)
    put(repo / 'x_linear/ops.mojo', 'fn ops(): changed\n')
    git(repo, 'commit', '-qam', 'later edit')
    for path in (leg['build'] / 'compile-inputs.json', leg['build'] / 'build-provenance.json', launch_path):
        doc = json.loads(path.read_text())
        doc['source_commit'] = git(repo, 'rev-parse', 'HEAD')
        if 'tooling_commit' in doc:
            doc['tooling_commit'] = doc['source_commit']
        put(path, doc)
    with pytest.raises(ValueError):
        admission.prepare('linux-release-leg', repo, ledger.path, tmp_path / 'd', dict(leg=leg['dir']))
    facts = dict(vendor='cuda', arch='sm_80', adapters=None)
    with pytest.raises(ValueError, match='PTX leg'):
        sub._leg_units(facts)


def test_binding_units_and_partial_coverage_are_not_subunits():
    binding = dict(target='linux-64', vendor='hip', arch='gfx942', tier='identical', name='_mojolearn')
    unit = dict(kind='compile', vendor='amd', release_modes=['identical'], build_flags={}, compile_variants=[],
                compiler_sha256=COMPILER)
    assert not sub.supports_requirement(dict(kind='compile', inputs=dict(unit=unit, binding=binding),
                                             coverage=[sub.BASE_COVERAGE, compiled.binding_key(binding)], missing_inputs=[]))
    runtime = dict(target='linux-64', what='runtime')
    assert sub.supports_requirement(dict(kind='compile', inputs=dict(unit=unit, runtime=runtime),
                                         coverage=sub.RUNTIME_COVERAGE, missing_inputs=[]))
    assert not sub.supports_requirement(dict(kind='compile', inputs=dict(unit=unit, runtime=runtime),
                                             coverage=sub.RUNTIME_COVERAGE[:1], missing_inputs=[]))
    assert not sub.supports_requirement(dict(kind='compile', inputs=dict(unit=dict(unit, build_flags={'X': '1'}),
                                             runtime=runtime), coverage=sub.RUNTIME_COVERAGE, missing_inputs=[]))


# ---------------------------------------------------------------- Apple pack
def closure_digest(root, rels):
    value = hashlib.sha256()
    for rel in sorted(rels, key=lambda name: PurePosixPath(name).parts):
        value.update(rel.encode() + b'\0' + sha(root / rel).encode() + b'\n')
    return value.hexdigest()


@pytest.fixture
def pack(tmp_path, repo):
    revision = git(repo, 'rev-parse', 'HEAD')
    import apple_retained_groups as arg
    out = tmp_path / 'pack'
    admission_doc = dict(schema=arg.SCHEMA, action='admit-original-device-groups', complete=True,
                         admitted_source_commit=revision, source_root=str(repo), compiler_sha256=COMPILER,
                         host_toolchain=APPLE, admitted_dependency_files={'x_linear/ops.mojo': sha(repo / 'x_linear/ops.mojo')})
    admission_path = put(tmp_path / 'device-admission.json', admission_doc)
    put(out / 'launch.json', dict(schema='mojolearn.apple.retained-pack-launch.v1', source_commit=revision,
                                  admission=witness.witness(admission_path), compiler_sha256=COMPILER))
    tree = compiled.Tree.at(repo, revision)

    def key_fields(source=None, env=None, outputs=('python/mojolearn/host/_mojolearn_native_host.so',)):
        return dict(schema='mojolearn-bincache-v1', script='bindings/merged_build.sh', numeric_mode='identical',
                    os=dict(id='darwin'), repo_path=str(repo), toolchain=MAC_TOOLCHAIN,
                    build_env=env or {'MOJOLEARN_COMPILE_JOBS': '1', 'MOJOLEARN_MERGED_BINDINGS': '1',
                                      'MOJOLEARN_MERGED_KEY': 'k', 'MOJOLEARN_NUMERIC_MODE': 'identical',
                                      'MOJOLEARN_PACKAGE_BYTE_LM': '1', 'MOJOLEARN_REUSE_DIR': '',
                                      'MOJOLEARN_REUSE_PLAN': '', 'MOJOLEARN_TARGET_COLUMN': 'cpu'},
                    local=dict(host_toolchain=APPLE, outputs=list(outputs)),
                    source=source or sub._tree_digest(tree, 'bindings/merged_build.sh'))

    files = {'libmojolearn_adapter.so': b'adapter', 'mojolearn_torch.so': b'torch', 'mojolearn_ops.mojoc': b'ops'}

    def write(fields=None, outcome='miss+built-cached', mac_flags=MAC_FLAGS):
        fields = fields or key_fields()
        key = bincache.key_of(fields)
        put(out / 'cache-provenance/keys' / (key + '.json'), fields)
        output = fields['local']['outputs'][0]
        put(out / 'cache-provenance/provenance.tsv',
            f'2026-10-25T00:00:00Z\tbindings/merged_build.sh\t{outcome}\t{key}\t1.0\t{output}={"7" * 64}\n'
            'x\tbindings/merged_build.sh\tretained\tabc\t1.0\tpython/mojolearn/identical/_mojolearn_native.so=' + '8' * 64 + '\n')
        wheel = out / 'mojolearn-0.9.1-cp314-macosx_14_0_arm64.whl'
        with zipfile.ZipFile(wheel, 'w') as archive:
            archive.writestr('mojolearn/host/_mojolearn_native_host.so', b'host bytes')
            archive.writestr('mojolearn/host/_mojolearn_merged.json',
                             json.dumps(dict(files={'_mojolearn_native_host.so': ['_mojolearn_core_host']})))
            for name, data in files.items():
                archive.writestr('mojolearn/_adapters/' + name, data)
            archive.writestr('mojolearn/_adapters/adapters.json',
                             json.dumps(adapter_manifest(repo, revision, 'metal', 'apple', 'macos', files, mac_flags)))
        put(out / 'pack.rc', '0\n')
        put(out / 'pack-proof.json', dict(schema=arg.PACK_SCHEMA, complete=True, pack_exit=0, source_commit=revision,
            launch=witness.witness(out / 'launch.json'), admission=witness.witness(admission_path),
            wheel=dict(path=str(wheel.resolve()), sha256=sha(wheel))))

    write()
    rels = ['bindings/native.mojo', 'pixi.lock']
    host = dict(schema='mojolearn.binding-identity.v2', target='osx-arm64', vendor='host', arch='', tier='host',
                name='_mojolearn_core_host', script='build_native.sh', toolchain=MAC_TOOLCHAIN, host_toolchain=APPLE,
                closure=dict(scope='closure', digest=closure_digest(repo, rels)), generators=None,
                flags=dict(numeric_mode='identical', target_column='cpu', gpu_archs=None, compile_jobs='1', package_byte_lm='1'),
                builders={'packaging/macos/build_release_wheel.sh': 'f' * 64}, builder_rule='builder-view.v1')
    unit = dict(schema='mojolearn.release-dependencies.v1', kind='compile', vendor='apple', release_modes=['identical'],
                build_flags={}, compile_variants=[])
    environment = dict(target='osx-arm64', vendor='metal', arch='apple', toolchain=MAC_TOOLCHAIN, flags=MAC_FLAGS,
                       host_toolchain=APPLE, builders={'packaging/macos/build_release_wheel.sh': '4' * 64})

    def originals(binding=host):
        key = compiled.binding_key(binding)
        srcs = dict(adapter_sources(repo), **{'core/host_predict_threads.mojo': sha(repo / 'core/host_predict_threads.mojo')})
        return dict([
            row(revision, 'compile-apple/' + key, dict(unit=unit, binding=binding), [compiled.BASE_COVERAGE, key],
                rels + list(binding['builders'])),
            row(revision, 'compile-apple/adapters/apple', dict(unit=unit, adapter_sources=srcs, environment=environment),
                sub.ADAPTER_COVERAGE, list(srcs) + list(environment['builders']))])

    return dict(dir=out, originals=originals, write=write, key_fields=key_fields, host=host)


def test_apple_pack_admits_compiled_host_bindings_and_adapters(tmp_path, repo, pack, monkeypatch):
    wanted = pack['originals']()
    assert all(producer_admission_available(w) for w in wanted.values())
    use(monkeypatch, wanted)
    ledger = ledger_for(tmp_path, wanted)
    review = admission.prepare('apple-pack', repo, ledger.path, tmp_path / 'review', dict(pack=pack['dir']))
    assert review['admitted'] == sorted(wanted) and not review['refused']
    result = admission.admit(ledger.path, repo, tmp_path / 'review/report.json')
    assert set(result['actions'].values()) == {'reuse'}
    assert all(ledger.plan()[name]['status'] == 'passed' for name in wanted)


def test_apple_pack_refuses_hits_other_sources_flags_and_closures(tmp_path, repo, pack, monkeypatch):
    wanted = pack['originals']()
    use(monkeypatch, wanted)
    ledger = ledger_for(tmp_path, wanted)
    host_id = next(k for k in wanted if '/host/' in k)
    # a cache hit is not a compilation in this run, and without a compiled whole-tree key
    # nothing binds the adapter's imported Mojo sources outside adapters/
    pack['write'](outcome='hit')
    with pytest.raises(ValueError, match='no requirement could be admitted'):
        admission.prepare('apple-pack', repo, ledger.path, tmp_path / 'a', dict(pack=pack['dir']))
    reasons = json.loads((tmp_path / 'a/review.json').read_text())['refused']
    assert 'core/host_predict_threads.mojo' in reasons['compile-apple/adapters/apple']
    pack['write'](fields=pack['key_fields'](source=dict(scope='tree', digest='0' * 64, files=1)))
    with pytest.raises(ValueError, match='does not reproduce'):
        admission.prepare('apple-pack', repo, ledger.path, tmp_path / 'b', dict(pack=pack['dir']))
    env = dict(pack['key_fields']()['build_env'], MOJOLEARN_COMPILE_JOBS='2')
    pack['write'](fields=pack['key_fields'](env=env), mac_flags=dict(MAC_FLAGS, compile_jobs='2'))
    with pytest.raises(ValueError, match='no requirement could be admitted'):
        admission.prepare('apple-pack', repo, ledger.path, tmp_path / 'c', dict(pack=pack['dir']))
    reasons = json.loads((tmp_path / 'c/review.json').read_text())['refused']
    assert 'MOJOLEARN_COMPILE_JOBS' in reasons[host_id]
    assert 'compile jobs' in reasons['compile-apple/adapters/apple']
    pack['write']()
    bad = pack['originals'](binding=dict(pack['host'], closure=dict(scope='closure', digest='0' * 64)))
    use(monkeypatch, bad)
    review = admission.prepare('apple-pack', repo, ledger_for(tmp_path / 'd', bad).path, tmp_path / 'd/review',
                               dict(pack=pack['dir']))
    assert 'closure digest' in review['refused'][host_id]


def test_apple_pack_requires_a_completed_pack(tmp_path, repo, pack, monkeypatch):
    use(monkeypatch, pack['originals']())
    ledger = ledger_for(tmp_path, pack['originals']())
    put(pack['dir'] / 'pack.rc', '1\n')
    with pytest.raises(ValueError, match='did not complete'):
        admission.prepare('apple-pack', repo, ledger.path, tmp_path / 'a', dict(pack=pack['dir']))
