#!/usr/bin/env python3
"""Admit unchanged merged device artifacts with an independently built host.

Assembly is not compilation. Original build proofs, before-build captures,
source revisions, and staged bytes stay intact as raw witnesses. The new proof
names the package source separately and requires exact compiler, target, flags,
source-closure, adapter, and runtime inputs for every artifact it admits.
"""
import argparse
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import tarfile
import tempfile

import release_host_composition as hc

SCHEMA = 'mojolearn.linux.retained-composition.v1'
ACTION = 'assemble-retained-native'
PTX_FIELD = 'retained_composition'


def require(ok, why):
    hc.require(ok, 'Retained composition: ' + why)


def sha_inventory(inventory):
    return hashlib.sha256(json.dumps(inventory, separators=(',', ':')).encode()).hexdigest()


def write(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, sort_keys=True, indent=2) + '\n')


def adapter_inputs(root):
    """Resolve both native adapter frontends without importing product code."""
    import bincache
    root = Path(root).resolve()
    todo = [root / 'adapters/native/c_api.mojo', *(root / 'adapters/max_ops').rglob('*.mojo')]
    seen = set()
    while todo:
        path = todo.pop().resolve()
        if path in seen:
            continue
        require(path.is_relative_to(root) and path.is_file(), 'missing or escaping adapter source')
        seen.add(path)
        for module, names in bincache.parse_imports(path.read_text()):
            found = bincache.resolve(module, names, path, [root, root / 'bindings'])
            require(found is not None, 'unresolved adapter import: ' + module)
            todo.extend(found)
    todo = [p for p in (root / 'adapters/torch').rglob('*') if p.suffix in ('.cpp', '.h', '.hpp')]
    while todo:
        path = todo.pop().resolve()
        if path in seen:
            continue
        require(path.is_relative_to(root) and path.is_file(), 'missing or escaping adapter C++ source')
        seen.add(path)
        for include in re.findall(r'^\s*#\s*include\s*"([^"\n]+)"', path.read_text(), re.M):
            found = next((p for p in (path.parent / include, root / 'adapters/native' / include) if p.is_file()), None)
            require(found is not None, 'unresolved adapter C++ include: ' + include)
            todo.append(found)
    seen.update(root / name for name in ('packaging/adapters/build.sh', 'packaging/adapters/fetch_torch.py',
                                        'packaging/adapters/manifest.py', 'packaging/adapters/pins.json'))
    return {p.relative_to(root).as_posix(): hc.file_sha(p) for p in sorted(seen)}


@lru_cache(maxsize=8)
def source_context(root, revision):
    """Read committed source once per revision; never compile or import it."""
    import release_reuse as rr
    from check_linux_release_qualification import is_native_source, native_inventory
    root = Path(root)
    with tempfile.TemporaryDirectory(prefix='mojolearn-retained-inputs-') as directory:
        # Include Python package bytes in the original inventory witness, while
        # using only each compiled artifact's closure for the reuse decision.
        proc = subprocess.Popen(['git', '-C', str(root), 'archive', '--format=tar', revision],
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        with tarfile.open(fileobj=proc.stdout, mode='r|') as archive:
            for member in archive:
                if member.isfile() and (rr.wanted(member.name) or is_native_source(member.name)):
                    hc.relative(member.name)
                    archive.extract(member, directory, filter='data')
        error = proc.stderr.read().decode()
        proc.stdout.close()
        proc.stderr.close()
        require(proc.wait() == 0, 'cannot read committed source archive: ' + error)
        facts = rr.tree_facts(directory)
        identities = {b.key: rr.identity(b, facts) for b in rr.bindings(rr.LINUX, ['identical'])}
        require(all(not rr.unreadable(i) and i['closure']['scope'] == 'closure' for i in identities.values()),
                'unresolved binding identity')
        runtime = rr.runtime_identity(facts)
        require(not rr.unreadable_runtime(runtime), 'unresolved runtime identity')
        adapters = adapter_inputs(Path(directory))
        inventory = native_inventory(directory)
        import binding_stamps
        merged_inputs = {name: hc.file_sha(Path(directory) / name) for name in binding_stamps.MERGED_INPUTS}
        for name in hc.HOST_FORMAT_SOURCES:
            require(b'MOJOLEARN_CUDA_CODE_FORMAT' not in (Path(directory) / name).read_bytes(),
                    'host builder now consumes CUDA format; explicit support is required')
        builders = {name: rr.builder_view(name, (Path(directory) / name).read_bytes(),
                    {'packaging/adapters/build.sh', 'adapters'}) for name in rr.LINUX_BUILDERS}
    return dict(binding_identities=identities, runtime_identity=runtime, source_inventory=inventory,
                adapter_sources=adapters, adapter_builders=builders, merged_build_inputs=merged_inputs)


def current_source(root):
    from check_linux_release_qualification import tracked_native_inventory
    root = Path(root).resolve()
    revision = subprocess.check_output(['git', '-C', str(root), 'rev-parse', 'HEAD'], text=True).strip()
    require(re.fullmatch('[0-9a-f]{40}', revision), 'missing current source revision')
    require(not subprocess.check_output(['git', '-C', str(root), 'status', '--porcelain', '--untracked-files=no'],
                                       text=True).strip(), 'current tracked source is dirty')
    return revision, tracked_native_inventory(root), source_context(str(root), revision)


def _device_keys(leg, context):
    prefix = 'linux-64/' + leg + '/identical/'
    return {k: v for k, v in context['binding_identities'].items() if k.startswith(prefix)}


def _host_keys(context):
    return {k: v for k, v in context['binding_identities'].items() if k.startswith('linux-64/host/-/host/')}


def _source_record(source, compiler):
    contract, variant = hc._record(source, compiler)
    proof, capture = hc.decoded(source['proof']), hc.decoded(source['capture'])
    prefix = 'mojolearn/' + source['leg'] + '/'
    identities = {k: v for k, v in capture['binding_identities'].items() if '/host/' not in k}
    import release_reuse as rr
    expected = {b.key for b in rr.bindings(rr.LINUX, ['identical'])
                if not b.host and b.vendor + '/' + b.arch == source['leg']}
    require(set(identities) == expected, 'incomplete original device identity inventory')
    require(proof.get('extensions') and all(p.startswith(prefix + 'identical/')
            and re.fullmatch('[0-9a-f]{64}', v) for p, v in proof['extensions'].items()), 'invalid device outputs')
    indexes = {k: v for k, v in proof['merged_indexes'].items() if not k.startswith(prefix + 'host/')}
    require(indexes and set(indexes) == {prefix + 'identical/' + hc.INDEX}, 'incomplete device index proof')
    members = [name for row in indexes.values() for names in row['files'].values() for name in names]
    require(len(members) == len(set(members)) and set(members) == {i['name'] for i in identities.values()},
            'device index members differ from original identities')
    expected_files = {prefix + 'identical/' + name for row in indexes.values() for name in row['files']}
    require(set(proof['extensions']) == expected_files, 'device index files differ from original outputs')
    for row in indexes.values():
        require(re.fullmatch('[0-9a-f]{64}', row.get('sha256', '')), 'invalid device index digest')
    stage = hc.decoded(source['manifest'])
    staged = {prefix + row['path']: row['sha256'] for row in stage['extensions']}
    require(all(staged.get(name) == value for name, value in proof['extensions'].items()),
            'original device stage differs from original output proof')
    readback = [line.split() for line in hc.raw(source['readback']).decode().splitlines()
                if line.split()[:1] == ['identical']]
    require(len(readback) == len(expected) and all(len(row) == 3 and row[2] == capture['vendor'] for row in readback)
            and {row[1] for row in readback} == set(members), 'incomplete original device vendor readback')
    return contract, variant, proof, capture


def _projection(device, host, compiler):
    dc, dv, dp, di = _source_record(device, compiler)
    if host.get('kind') == 'independent-host-build':
        from release_host_build import record
        hc0, hv, hp, hi = record(host, compiler)
    else:
        hc0, hv, hp, hi = _source_record(host, compiler)
    require(dc['runtime_libraries'] == hc0['runtime_libraries'], 'device and host runtime bytes differ')
    require(dc['toolchain'] == hc0['toolchain'] and dc['installed_packages'] == hc0['installed_packages'],
            'device and host compiler/package environment differs')
    prefix, hprefix = 'mojolearn/' + device['leg'] + '/', 'mojolearn/' + host['leg'] + '/'
    host_outputs = {prefix + p[len(hprefix):]: v for p, v in hp['host_extension'].items()}
    indexes = {p: v for p, v in dp['merged_indexes'].items() if not p.startswith(prefix + 'host/')}
    indexes.update({prefix + p[len(hprefix):]: v for p, v in hp['merged_indexes'].items() if p.startswith(hprefix + 'host/')})
    origins = {p: dict(source_commit=dp['source_commit'], proof_sha256=device['proof']['sha256'], sha256=v,
                        leg=device['leg']) for p, v in dp['extensions'].items()}
    origins.update({p: dict(source_commit=hp['source_commit'], proof_sha256=host['proof']['sha256'], sha256=v,
                            leg=host['leg']) for p, v in host_outputs.items()})
    return dict(extensions=dp['extensions'], host_extension=host_outputs, merged_indexes=indexes,
                adapters=dp.get('adapters', {}), runtime_libraries=dc['runtime_libraries'], artifact_origins=origins)


def validate(proof, source_root=None):
    require(proof.get('schema') == SCHEMA and proof.get('action') == ACTION and proof.get('complete') is True,
            'unsupported or incomplete assembly proof')
    require('build_exit' not in proof and proof.get('binding_layout') == 'merged'
            and proof.get('release_modes') == ['identical'], 'assembly must not impersonate compilation')
    compiler = proof.get('compiler_sha256', '')
    require(isinstance(compiler, str) and re.fullmatch('[0-9a-f]{64}', compiler), 'missing declared compiler identity')
    device, host = proof['original_device'], proof['original_host']
    projection = _projection(device, host, compiler)
    require(all(proof.get(k) == v for k, v in projection.items()), 'assembled artifact/provenance projection differs')
    revision, inventory = proof.get('source_commit'), proof.get('source_inventory')
    require(isinstance(revision, str) and re.fullmatch('[0-9a-f]{40}', revision), 'missing admitted package revision')
    require(isinstance(inventory, list) and inventory and len(dict(inventory)) == len(inventory)
            and sha_inventory(inventory) == proof.get('source_sha256'), 'invalid admitted source inventory')
    for name, value in inventory:
        hc.relative(name)
        require(isinstance(value, str) and re.fullmatch('[0-9a-f]{64}', value), 'invalid admitted source digest')
    current = proof['admitted_inputs']
    before, hbefore = proof['device_inputs'], proof['host_inputs']
    dcapture, hcapture = hc.decoded(device['capture']), hc.decoded(host['capture'])
    require(current.get('source_inventory') == inventory, 'admitted input inventory differs')
    for source, inputs in ((device, before), (host, hbefore)):
        require(inputs.get('source_inventory') == hc.decoded(source['inventory']),
                'original inventory differs from committed source inputs')
    require(_device_keys(device['leg'], before) == _device_keys(device['leg'], current)
            == {k: v for k, v in dcapture['binding_identities'].items() if '/host/' not in k},
            'device compiler/target/flags/source identity changed')
    require(_host_keys(hbefore) == _host_keys(current)
            == {k: v for k, v in hcapture['binding_identities'].items() if '/host/' in k},
            'selected host compiler/flags/source identity changed')
    require(before['runtime_identity'] == hbefore['runtime_identity'] == current['runtime_identity'],
            'runtime compiler/flags/source identity changed')
    require(before['merged_build_inputs'] == hbefore['merged_build_inputs'] == current['merged_build_inputs'],
            'merged compiler script, defines, generator, or manifest changed')
    if host.get('kind') == 'independent-host-build':
        require(hcapture['runtime_identity'] == hbefore['runtime_identity']
                and hcapture['merged_build_inputs'] == hbefore['merged_build_inputs'], 'independent host runtime/build inputs differ')
    if projection['adapters']:
        require(before['adapter_sources'] == current['adapter_sources']
                and before['adapter_builders'] == current['adapter_builders'], 'adapter source/build inputs changed')
    if source_root is not None:
        root = str(Path(source_root).resolve())
        actual_revision, actual_inventory, actual = current_source(root)
        require(actual_revision == revision and actual_inventory == inventory and actual == current,
                'admission does not match current committed source')
        for source, captured in ((device, before), (host, hbefore)):
            original_revision = hc.decoded(source['proof'])['source_commit']
            require(source_context(root, original_revision) == captured, 'original source inputs do not reproduce')
    return True


def validate_bytes(proof, root):
    """Validate each shipped byte against its original producer, including indexes."""
    validate(proof)
    root = Path(root)
    prefix = 'mojolearn/' + proof['original_device']['leg'] + '/'
    for key in ('extensions', 'host_extension'):
        for name, value in proof[key].items():
            require(name.startswith(prefix) and hc.file_sha(root / hc.relative(name[len(prefix):])) == value,
                    'shipped artifact differs from original producer: ' + name)
    for name, row in proof['merged_indexes'].items():
        p = root / hc.relative(name[len(prefix):])
        require(hc.file_sha(p) == row['sha256'] and json.loads(p.read_text())['files'] == row['files'],
                'shipped merged index differs from original producer')
    runtime = {p.name: hc.file_sha(p) for p in (root / '.libs').iterdir() if p.is_file()}
    require(runtime == proof['runtime_libraries'], 'shipped runtime inventory/bytes differ')
    adapters = {p.name: hc.file_sha(p) for p in (root / '_adapters').glob('*') if p.is_file()}
    expected = {prefix + '_adapters': dict(files=adapters, manifest_sha256=adapters.get('adapters.json'))} if adapters else {}
    require(expected == proof['adapters'], 'shipped adapters differ from original producer')
    return True


def ptx_manifest(proof, original_manifest):
    """Only the current core binding changes; original arithmetic metadata remains."""
    doc = hc.decoded(original_manifest)
    require('release_admission' not in doc and PTX_FIELD not in doc, 'PTX original is already an admission')
    require(doc.get('source_commit') == hc.decoded(proof['original_device']['proof'])['source_commit'],
            'PTX original source differs')
    doc['source_commit'] = proof['source_commit']
    doc[PTX_FIELD] = {'proof': {'utf8': json.dumps(proof, sort_keys=True, separators=(',', ':'))},
                       'original_manifest': original_manifest}
    record = doc[PTX_FIELD]['proof']
    record['sha256'] = hashlib.sha256(record['utf8'].encode()).hexdigest()
    return doc


def validate_ptx(doc, root):
    admission = doc[PTX_FIELD]
    proof, original = hc.decoded(admission['proof']), hc.decoded(admission['original_manifest'])
    require(proof['original_device']['leg'] == 'cuda/sm_80', 'retained PTX proof has another target')
    require(ptx_manifest(proof, admission['original_manifest']) == doc, 'retained PTX arithmetic metadata changed')
    require(original.get('source_dirty') is False and doc['source_commit'] == proof['source_commit'],
            'dirty or mismatched PTX source')
    validate_bytes(proof, root)
    from verify_linux_surface_qualification import load_gpu_plugins
    prefix = 'mojolearn/cuda/sm_80/'
    load_gpu_plugins().validate_baseline_manifest(original,
        {name[len(prefix):]: value for name, value in proof['extensions'].items()},
        indexes={name[len(prefix):]: row['sha256'] for name, row in proof['merged_indexes'].items()
                 if name.startswith(prefix + 'identical/')})
    return True


def prepare(device_build, host_build, source_root, compiler, output):
    """Create a new set and assembly proof; originals and their proofs stay untouched."""
    device, device_set = hc.inspect_build(device_build)
    if (Path(host_build) / 'host-build-proof.json').is_file():
        from release_host_build import inspect
        host, host_set = inspect(host_build)
    else:
        host, host_set = hc.inspect_build(host_build)
    revision, inventory, context = current_source(source_root)
    root = str(Path(source_root).resolve())
    dp, hp = hc.decoded(device['proof']), hc.decoded(host['proof'])
    proof = dict(schema=SCHEMA, action=ACTION, complete=True, binding_layout='merged',
                 release_modes=['identical'], source_commit=revision, source_inventory=inventory,
                 source_sha256=sha_inventory(inventory), compiler_sha256=compiler,
                 original_device=device, original_host=host, admitted_inputs=context,
                 device_inputs=source_context(root, dp['source_commit']),
                 host_inputs=source_context(root, hp['source_commit']),
                 qualification='Retained build input/byte admission; not runtime, identity, or installed-wheel qualification',
                 **_projection(device, host, compiler))
    validate(proof, source_root)
    # Check originals before copying. Each source is checked with its own host.
    for original, directory in ((device, device_set), (host, host_set)):
        if original.get('kind') == 'independent-host-build':
            from release_host_build import validate_bytes as validate_host_bytes
            validate_host_bytes(original, directory, compiler)
            continue
        original_projection = _projection(original, original, compiler)
        for group in ('extensions', 'host_extension'):
            prefix = 'mojolearn/' + original['leg'] + '/'
            for path, value in original_projection[group].items():
                require(hc.file_sha(directory / hc.relative(path[len(prefix):])) == value, 'original artifact bytes changed')
    output = Path(output).resolve()
    require(not output.exists(), 'assembly destination already exists')
    output.parent.mkdir(parents=True, exist_ok=True)
    pending = Path(tempfile.mkdtemp(prefix=output.name + '.pending-', dir=output.parent))
    try:
        destination = pending / 'set'
        shutil.copytree(device_set, destination, ignore=lambda p, ns: ['host'] if Path(p) == device_set else [])
        shutil.copytree(host_set / 'host', destination / 'host')
        # Keep the old stage manifest as a raw original witness; do not claim it
        # staged the replacement host. The assembly receipt owns the new set.
        validate_bytes(proof, destination)
        if device['leg'] == 'cuda/sm_80':
            doc = ptx_manifest(proof, hc.witness(device_set / 'PTX_BASELINE.json'))
            validate_ptx(doc, destination)
            write(destination / 'PTX_BASELINE.json', doc)
        write(pending / 'assembly-proof.json', proof)
        pending.rename(output)
    except BaseException:
        shutil.rmtree(pending)
        raise
    return proof


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device-build', type=Path, required=True)
    parser.add_argument('--host-build', type=Path, required=True)
    parser.add_argument('--source-root', type=Path, required=True)
    parser.add_argument('--compiler-sha256', required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args(argv)
    proof = prepare(args.device_build, args.host_build, args.source_root, args.compiler_sha256, args.out)
    print(json.dumps({'schema': SCHEMA, 'action': ACTION, 'source_commit': proof['source_commit'],
                      'original_device': hc.decoded(proof['original_device']['proof'])['source_commit'],
                      'original_host': hc.decoded(proof['original_host']['proof'])['source_commit'],
                      'output': str(args.out)}))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
