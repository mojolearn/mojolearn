"""Compose a separately compiled native adapter with retained framework frontends.

This is packaging evidence, never a runtime qualification. Original build and
staging witnesses remain unchanged; only the named native library is replaced.
"""
import copy
import hashlib
import json
from pathlib import Path
import subprocess
import tarfile

import release_host_composition as hc
import release_macos_assembly as ma


def require(ok, message):
    hc.require(ok, 'Native adapter composition: ' + message)


def command_flags(command):
    """Ignore output paths and compiler worker count, retaining all code flags."""
    command = list(command)
    require(command[:4] == ['pixi', 'run', 'mojo', 'build'], 'unknown build command')
    args = command[4:]
    flags = []
    pairs = {'-j', '-o', '--emit', '--target-cpu', '--target-accelerator', '-D', '-I'}
    while args:
        item = args.pop(0)
        if item in pairs:
            require(bool(args), 'missing build option value')
            value = args.pop(0)
            if item not in ('-j', '-o'):
                flags.append((item, value))
        else:
            flags.append((item, ''))
    return sorted(flags)


def validate(refresh, before, current, compiler, leg, original_adapters):
    require(refresh.get('schema') == 'mojolearn.linux.native-adapter-composition.v1', 'unknown schema')
    build = refresh['build']
    vendor, target = leg.split('/')
    require(build['component'] == 'adapter-native' and build['state'] == 'built'
            and build['vendor'] == {'cuda': 'nvidia', 'hip': 'amd'}[vendor]
            and build['target'] == target, 'build target/state differs')
    require(refresh['compiler_sha256'] == compiler, 'compiler differs')
    require(refresh['captured_inputs'] == current['adapter_sources'], 'compiled source closure differs')
    ma.native_source_refresh(before['adapter_sources'], current['adapter_sources'],
                             refresh['captured_inputs'], refresh['frontend_inputs'])
    require(before.get('builder_phases', {}).get('adapters', before['adapter_builders'])
            == current.get('builder_phases', {}).get('adapters', current['adapter_builders']),
            'frontend build policy changed')
    old, new = hc.decoded(refresh['original_manifest']), hc.decoded(refresh['manifest'])
    key = 'mojolearn/' + leg + '/_adapters'
    require(original_adapters[key]['manifest_sha256'] == refresh['original_manifest']['sha256'],
            'original manifest differs')
    require(command_flags(old['commands']['native']) == command_flags(refresh['command']), 'native flags differ')
    require(new['commands']['native'] == refresh['command'], 'new manifest command differs')
    for role in ('torch', 'max'):
        require(new['files'][role] == old['files'][role], 'retained frontend bytes differ')
    expected = copy.deepcopy(old)
    expected.update(source_commit=new['source_commit'], source_commit_scope=new['source_commit_scope'],
                    sources=new['sources'], artifact_origins=new['artifact_origins'])
    expected['commands']['native'] = refresh['command']
    expected['files']['native'] = new['files']['native']
    expected['unstaged']['native'] = dict(name='libmojolearn_adapter.so',
                                         sha256=build['sha256'], bytes=refresh['unstaged_bytes'])
    require(new == expected, 'unrelated adapter manifest metadata changed')
    require(new['files']['native']['sha256'] == refresh['staged_sha256'], 'staged native differs')
    files = dict(original_adapters[key]['files'])
    files.update({'libmojolearn_adapter.so': refresh['staged_sha256'],
                  'adapters.json': refresh['manifest']['sha256']})
    return {key: dict(files=files, manifest_sha256=files['adapters.json'])}


def prepare(build_dir, device_set, source_root, compiler, output, staging_command):
    """Verify a CPU build capture, then apply ordinary ELF loader staging."""
    import retained_release_composition as rc
    build_dir, device_set, root, output = map(Path, (build_dir, device_set, source_root, output))
    require(not output.exists(), 'destination exists')
    summary = json.loads((build_dir / 'summary.json').read_text())
    require((build_dir / 'artifacts/build.rc').read_text().strip() == '0'
            and 'exit_code=0' in (build_dir / 'GHA/status.txt').read_text(), 'build failed')
    require(compiler in (build_dir / 'GHA/compiler.txt').read_text(), 'compiler capture differs')
    native = build_dir / 'artifacts/libmojolearn_adapter.so'
    require(hc.file_sha(native) == summary['sha256'], 'raw native bytes changed')
    captured = json.loads((build_dir / 'adapter-inputs.json').read_text())
    require(captured == rc.adapter_inputs(root), 'captured inputs differ from current source')
    with tarfile.open(build_dir / 'adapter-source-closure.tar.gz') as archive:
        for name, value in captured.items():
            require(hashlib.sha256(archive.extractfile(name).read()).hexdigest() == value, 'captured source bytes differ')
        for name in ('pixi.lock', 'pixi.toml', 'tools/build_native_adapters.sh'):
            require(archive.extractfile(name).read() == (root / name).read_bytes(), 'build dependency/recipe changed')
    original_path = device_set / '_adapters/adapters.json'
    old = json.loads(original_path.read_text())
    inputs = {role: ma.adapter_role_inputs(root, role) for role in ('torch', 'max')}
    for role, values in inputs.items():
        for name, value in values.items():
            raw = subprocess.check_output(['git', '-C', str(root), 'show', old['source_commit'] + ':' + name])
            require(hashlib.sha256(raw).hexdigest() == value, 'retained ' + role + ' input changed: ' + name)
    command = json.loads((build_dir / 'artifacts/build-command.json').read_text())
    require(command_flags(command) == command_flags(old['commands']['native']), 'code generation flags changed')
    output.mkdir(parents=True)
    staged = output / 'libmojolearn_adapter.so'
    staged.write_bytes(native.read_bytes())
    argv = [str(x).replace('{artifact}', str(staged)) for x in staging_command]
    with (output / 'staging.log').open('w') as log:
        result = subprocess.run(argv, stdout=log, stderr=subprocess.STDOUT)
    require(result.returncode == 0, 'ELF loader staging failed')
    api = ma.module(root / 'packaging/adapters/manifest.py', 'native_adapter_manifest')
    sources = ma.native_source_refresh(old['sources'], api.source_digests(root), captured, inputs)
    new = copy.deepcopy(old)
    new.update(source_commit=ma.revision(root), source_commit_scope='admitted package source; compilation origins are per artifact',
               sources=sources, artifact_origins={
                   'native': dict(action='separate-native-build', source_commit=summary['source_commit'], sha256=summary['sha256']),
                   **{role: dict(action='retained-original-build', source_commit=old['source_commit'],
                                 sha256=old['files'][role]['sha256']) for role in ('torch', 'max')}})
    new['commands']['native'] = command
    new['files']['native'] = dict(name=staged.name, bytes=staged.stat().st_size, sha256=hc.file_sha(staged))
    new['unstaged']['native'] = dict(name=staged.name, bytes=native.stat().st_size, sha256=hc.file_sha(native))
    rc.write(output / 'adapters.json', new)
    refresh = dict(schema='mojolearn.linux.native-adapter-composition.v1', build=summary,
                   compiler_sha256=compiler, captured_inputs=captured, frontend_inputs=inputs,
                   command=command, original_manifest=hc.witness(original_path), manifest=hc.witness(output / 'adapters.json'),
                   unstaged_bytes=native.stat().st_size, staged_sha256=hc.file_sha(staged),
                   staging=dict(command=argv, exit_code=0, log=hc.witness(output / 'staging.log')),
                   evidence={str(p.relative_to(build_dir)):dict(path=str(p), sha256=hc.file_sha(p))
                             for p in build_dir.rglob('*') if p.is_file()})
    rc.write(output / 'native-refresh.json', refresh)
    return refresh
