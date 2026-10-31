#!/usr/bin/env python3
"""Witnessed assembly of one retained CPU host artifact, without rebuilding it.

Original complete-build proofs, manifests and binaries are never modified. A
composition is a packaging choice, not a claim of byte reproducibility or of
numerical equivalence between the retained compiler outputs.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import re
import shutil
import tempfile

# v2 (0.8.59): the merged host set is SEVERAL files, one per host group of
# bindings/merged_groups.json; every per-file field is a {file: value} map.
SCHEMA = 'mojolearn.retained-host-composition.v2'
BUILD_SCHEMA = 'mojolearn.linux.build-provenance.v1'
CAPTURE_SCHEMA = 'mojolearn.linux.release-compile-inputs.v1'
INDEX = '_mojolearn_merged.json'
HOST_FORMAT_SOURCES = ('bindings/merged_build.sh', 'bindings/build_defines.sh')


def require(ok, why):
    if not ok:
        raise ValueError(why)


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':')).encode()


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def file_sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def witness(path):
    raw = Path(path).read_bytes()
    return {'sha256': hashlib.sha256(raw).hexdigest(), 'utf8': raw.decode()}


def raw(record):
    require(isinstance(record, dict) and isinstance(record.get('utf8'), str), 'Missing raw witness')
    data = record['utf8'].encode()
    require(hashlib.sha256(data).hexdigest() == record.get('sha256'), 'Raw witness hash changed')
    return data


def decoded(record):
    return json.loads(raw(record))


def relative(path):
    p = Path(path)
    require(not p.is_absolute() and '..' not in p.parts and str(p) == path, 'Unsafe artifact path')
    return p


def host_plan():
    """{'<stem>.so': [members]}: the merged host files the Linux release plans
    (tools/merged_bindings.py, byte LM on), from this checkout's manifest."""
    import merged_bindings
    return {stem + '.so': list(members) for stem, _cls, members in merged_bindings.plan('host', 'linux', True)}


def host_files():
    """The merged host file names, sorted."""
    return sorted(host_plan())


def check_host_index(index, names, why):
    """A merged host index: format 1, exactly the planned files, each with its
    planned members, together exactly `names`."""
    plan = host_plan()
    files = index.get('files', {}) if isinstance(index, dict) else {}
    seen = [n for members in files.values() if isinstance(members, list) for n in members]
    require(type(index.get('format')) is int and index['format'] == 1
            and set(files) == set(plan) and all(isinstance(m, list) and sorted(m) == sorted(plan[f])
                                                for f, m in files.items())
            and len(seen) == len(set(seen)) == len(names) and set(seen) == set(names), why)


def expected_names():
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'python/mojolearn'))
    from host_surface import wheel_bindings
    return set(wheel_bindings())


def shard_evidence_root(build_root):
    """Locate retained GHA receipts beside a downloaded complete build."""
    root = Path(build_root).resolve()
    for parent in (root, *list(root.parents)[:3]):
        candidate = parent / 'GHA'
        if (candidate / 'shards').is_dir() and (candidate / 'assemble/hot_sha256.tsv').is_file():
            return candidate
    return None


def _tsv(witness_record):
    return [line.split('\t') for line in raw(witness_record).decode().splitlines() if line.strip()]


def validate_shard_origin(receipt, filename, cache, key):
    """Bind an assembly hot hit to an actual successful original shard compile."""
    require(isinstance(receipt, dict), 'Missing original shard compilation receipt')
    token = 'host:merged_build.sh:' + filename[:-3]
    rows = [r for r in _tsv(receipt['shard']) if r[0] == token]
    require(len(rows) == 1 and len(rows[0]) == 5 and rows[0][1] == 'ok'
            and rows[0][2].startswith('miss+built-') and rows[0][4] == cache['key'],
            'Original shard did not compile this host group successfully')
    require(decoded(receipt['key']) == key and digest(key) == cache['key'], 'Original shard compiler inputs differ')
    prov = [r for r in _tsv(receipt['provenance']) if len(r) == 6 and r[3] == cache['key']]
    require(len(prov) == 1 and prov[0][1] == 'bindings/merged_build.sh' and prov[0][2] == rows[0][2]
            and prov[0][5] == 'python/mojolearn/host/' + filename + '=' + cache['unstaged_sha256'],
            'Original shard compiler output differs')
    archives = []
    for name in ('archive', 'assembly_archive'):
        hashes = [r for r in _tsv(receipt[name]) if r[0] == cache['key']]
        require(len(hashes) == 1 and len(hashes[0]) == 2 and re.fullmatch('[0-9a-f]{64}', hashes[0][1]),
                'Missing original shard archive hash')
        archives.append(hashes[0][1])
    require(archives[0] == archives[1], 'Assembly restored another shard archive')


def _record(source, compiler):
    proof, capture, manifest, index, key = (decoded(source[n]) for n in
        ('proof', 'capture', 'manifest', 'index', 'host_cache_key'))
    require(proof.get('schema') == BUILD_SCHEMA and proof.get('action') == 'build'
            and proof.get('complete') is True and type(proof.get('build_exit')) is int
            and proof['build_exit'] == 0, 'Incomplete or unsupported original build')
    require(capture.get('schema') == CAPTURE_SCHEMA and capture.get('phase') == 'before-build',
            'Missing before-build compiler capture')
    require(proof.get('compile_inputs_sha256') == source['capture']['sha256'], 'Capture does not match original proof')
    commit = proof.get('source_commit')
    require(isinstance(commit, str) and re.fullmatch('[0-9a-f]{40}', commit)
            and capture.get('source_commit') == commit, 'Original source commit differs')
    require(capture.get('compiler_sha256') == compiler, 'Captured compiler differs from declared compiler bytes')
    require(proof.get('binding_layout') == manifest.get('binding_layout') == 'merged'
            and proof.get('release_modes') == ['identical'], 'Only complete merged identical builds are supported')
    vendor, arch = capture.get('vendor'), capture.get('arch')
    require(vendor in ('cuda', 'hip') and isinstance(arch, str)
            and re.fullmatch('sm_[0-9]+[a-z]*|gfx[0-9a-f]+', arch), 'Unsupported original target')
    leg = vendor + '/' + arch
    require(source.get('leg') == leg, 'Witness leg differs from compiler capture')
    prefix = 'mojolearn/' + leg + '/'
    files = host_files()
    host_paths = {f: 'host/' + f for f in files}
    require(set(proof.get('host_extension', {})) == {prefix + p for p in host_paths.values()},
            'Incomplete merged host output proof')
    host_sha = {f: proof['host_extension'][prefix + p] for f, p in host_paths.items()}
    require(all(isinstance(v, str) and re.fullmatch('[0-9a-f]{64}', v) for v in host_sha.values()),
            'Missing original host digest')
    names = expected_names()
    # The installed index uses files as its explicit file-to-members table.
    require(type(index.get('format')) is int and index['format'] == 1, 'Unsupported merged host index format')
    check_host_index(index, names, 'Incomplete or changed merged host index')
    proof_index = proof.get('merged_indexes', {}).get(prefix + 'host/' + INDEX, {})
    require(proof_index == {'sha256': source['index']['sha256'], 'files': index['files']},
            'Host index does not match original proof')
    readback = raw(source['readback']).decode().splitlines()
    host_rows = [line.split() for line in readback if line.split()[:1] == ['host']]
    require(len(host_rows) == len(names) and all(len(r) == 3 and r[2] == 'cpu' for r in host_rows)
            and {r[1] for r in host_rows} == names, 'Incomplete CPU host readback')
    identities = {k: v for k, v in capture.get('binding_identities', {}).items() if '/host/' in k}
    require(set(identities) == {'linux-64/host/-/host/' + n for n in names}, 'Incomplete host source identities')
    import release_reuse
    for ident in identities.values():
        require(ident.get('schema') == 'mojolearn.binding-identity.v2' and ident.get('target') == 'linux-64'
                and ident.get('vendor') == ident.get('tier') == 'host' and ident.get('arch') == ''
                and not release_reuse.unreadable(ident), 'Unresolved or unsupported host identity')
        require(ident.get('closure', {}).get('digest') and ident.get('closure', {}).get('scope'), 'Missing host source closure')
        flags = ident.get('flags', {})
        require(flags.get('target_column') == 'cpu' and not flags.get('gpu_archs')
                and flags.get('numeric_mode') == 'identical' and flags.get('compile_jobs') == '2'
                and flags.get('linux_cpu') == 'x86-64-v3' and flags.get('package_byte_lm') == '1',
                'Unsupported host compiler flags')
    inventory = proof.get('source_inventory')
    require(isinstance(inventory, list) and inventory and len(dict(inventory)) == len(inventory), 'Invalid source inventory')
    require(hashlib.sha256(json.dumps(inventory, separators=(',', ':')).encode()).hexdigest()
            == proof.get('source_sha256'), 'Source inventory digest differs')
    require(capture.get('source_inventory_digest') == digest(inventory), 'Before-build source digest differs')
    packages = capture.get('installed_packages')
    toolchain = capture.get('toolchain')
    require(isinstance(packages, list) and packages and all(type(p) is str for p in packages)
            and isinstance(toolchain, dict) and toolchain.get('packages')
            and set(toolchain['packages']) <= set(packages), 'Missing installed compiler package provenance')
    require(capture.get('source_inventory_sha256') == source['inventory']['sha256']
            and decoded(source['inventory']) == inventory, 'Before-build source inventory differs')
    require(proof.get('set_manifests', {}).get(prefix + 'manifest.json') == source['manifest']['sha256'],
            'Original stage manifest differs from build proof')
    staged = {}
    for f, host_path in host_paths.items():
        extensions = [r for r in manifest.get('extensions', []) if r.get('path') == host_path]
        require(len(extensions) == 1 and extensions[0].get('sha256') == host_sha[f], 'Original host stage hash differs')
        staged[f] = extensions[0]
    libs = manifest.get('staged_libs', [])
    require(libs and len({r['name'] for r in libs}) == len(libs), 'Missing or duplicate staged libraries')
    libraries = {r['name']: r['sha256'] for r in libs}
    for name, value in libraries.items():
        require(Path(name).name == name and re.fullmatch('[0-9a-f]{64}', value), 'Unsafe runtime library')
    require(isinstance(key, dict) and set(key) == set(files), 'Missing original host compiler/cache witness')
    normalized, caches, formats = {}, {}, set()
    for f, host_path in host_paths.items():
        one, cache = key[f], proof.get('bincache', {}).get(prefix + host_path, {})
        require(cache.get('key') == digest(one)
                and re.fullmatch('[0-9a-f]{40}', cache.get('compiled_from_commit', ''))
                and re.fullmatch('[0-9a-f]{64}', cache.get('unstaged_sha256', '')),
                'Missing original successful host compiler/cache witness')
        if cache.get('outcome') == 'hot-hit':
            validate_shard_origin(source.get('host_shard_origins', {}).get(f), f, cache, one)
        else:
            require(cache.get('outcome') == 'miss+built-uploaded' and cache['compiled_from_commit'] == commit,
                    'Missing original successful host compiler/cache witness')
        # one group per compile: bindings/merged_build.sh host --group <stem>,
        # optionally with the fixed generated-root directory (--rootdir <dir>,
        # packaging/linux/build_sets.sh since 0.8.63). The rootdir stays in the
        # normalized cache inputs, so every leg's contract must name the same one.
        args = one.get('args')
        require(isinstance(args, list) and args[:3] == ['host', '--group', f[:-len('.so')]]
                and (len(args) == 3 or (len(args) == 5 and args[3] == '--rootdir'
                                        and isinstance(args[4], str) and args[4].startswith('/'))),
                'Cache witness is not an identical CPU host invocation')
        require(one.get('schema') == 'mojolearn-bincache-v1' and one.get('script') == 'bindings/merged_build.sh'
                and one.get('target_column') == 'cpu'
                and one.get('numeric_mode') == 'identical' and one.get('device_arch') == 'none'
                and one.get('gpu_archs') == '', 'Cache witness is not an identical CPU host invocation')
        env = one.get('build_env', {})
        require(env.get('MOJOLEARN_TARGET_COLUMN') == 'cpu' and env.get('MOJOLEARN_COMPILE_JOBS') == '2'
                and env.get('MOJOLEARN_LINUX_CPU') == 'x86-64-v3' and not env.get('MOJOLEARN_GPU_ARCHS'), 'Actual host invocation flags differ')
        normalized[f] = copy.deepcopy(one)
        formats.add(normalized[f]['build_env'].pop('MOJOLEARN_CUDA_CODE_FORMAT', None))
        caches[f] = cache
    require(len(formats) == 1 and formats <= {'native', 'ptx'}, 'Unknown inherited CUDA code format')
    code_format = formats.pop()
    # This project-only flag is consumed by GPU format staging, not the host
    # command in merged_build.sh. No compiler/source/target flag is dropped.
    contract = dict(source_commit=commit, source_inventory=inventory, compiler_sha256=compiler,
        host_identities=identities, host_cache_inputs=normalized,
        group_compilation_sources={f:c['compiled_from_commit'] for f,c in caches.items()},
        installed_packages=capture.get('installed_packages'),
        toolchain=capture.get('toolchain'), index=index, runtime_libraries=libraries,
        host_dependencies={f: {k: row.get(k) for k in ('needed', 'runpath', 'rpath')} for f, row in staged.items()})
    variant = dict(leg=leg, host_sha256=host_sha, unstaged_sha256={f: c['unstaged_sha256'] for f, c in caches.items()},
        cache_key={f: c['key'] for f, c in caches.items()},
        proof_sha256=source['proof']['sha256'], inherited_cuda_code_format=code_format)
    return contract, variant


def validate_document(doc):
    require(doc.get('schema') == SCHEMA and doc.get('action') == 'select-retained-host', 'Unsupported host composition schema')
    compiler = doc.get('expected_compiler_sha256', '')
    require(isinstance(compiler, str) and re.fullmatch('[0-9a-f]{64}', compiler), 'Missing declared compiler byte identity')
    sources = doc.get('sources', [])
    require(isinstance(sources, list) and len(sources) >= 2, 'Host composition needs original independent build witnesses')
    contracts, variants = zip(*(_record(source, compiler) for source in sources))
    require(len({v['leg'] for v in variants}) == len(variants), 'Duplicate original leg')
    require(all(c == contracts[0] for c in contracts), 'Host source/compiler/flags/index/runtime contract differs')
    require(doc.get('host_contract_sha256') == digest(contracts[0]), 'Host contract digest changed')
    scope = doc.get('host_format_scope', {})
    inventory = dict(contracts[0]['source_inventory'])
    require(set(scope) == set(HOST_FORMAT_SOURCES), 'Missing host CUDA-format scope witness')
    for name, record in scope.items():
        code = raw(record)
        require(record['sha256'] == inventory.get(name), 'Host flag-scope source differs from frozen inventory')
        require(b'MOJOLEARN_CUDA_CODE_FORMAT' not in code,
                'Changed host builder may consume inherited CUDA format; explicit support is required')
    require(doc.get('variants') == list(variants), 'Original host variants changed')
    require(doc.get('selected_leg') in {v['leg'] for v in variants}, 'Selected host has no original witness')
    selected = next(v for v in variants if v['leg'] == doc['selected_leg'])
    require(doc.get('selected_sha256') == selected['host_sha256'], 'Selected host digest differs from its original proof')
    return contracts[0], selected


def validate_assembly(doc, sets, proof_paths, source_root=None):
    contract, selected = validate_document(doc)
    sources = {s['leg']: s for s in doc['sources']}
    require(set(sets) == set(sources), 'Assembly set coverage differs from composition')
    proofs = {file_sha(p) for p in proof_paths}
    require(len(proof_paths) == len(proofs) and proofs == {s['proof']['sha256'] for s in sources.values()},
            'Pack proofs differ from original composition witnesses')
    if source_root is not None:
        for name, wanted in contract['source_inventory']:
            require(file_sha(Path(source_root) / relative(name)) == wanted, 'Current source differs from original build: ' + name)
    for leg, root in sets.items():
        root = Path(root)
        source = sources[leg]
        require(file_sha(root / 'manifest.json') == source['manifest']['sha256'], 'Original stage manifest was replaced')
        require(file_sha(root / 'readback.txt') == source['readback']['sha256'], 'Original readback was replaced')
        require(file_sha(root / 'host' / INDEX) == source['index']['sha256'], 'Merged host index changed')
        require(all(file_sha(root / 'host' / f) == v for f, v in selected['host_sha256'].items()),
                'Assembly does not contain the selected retained host bytes')
        require({p.name for p in (root / '.libs').iterdir() if p.is_file()} == set(contract['runtime_libraries']),
                'Runtime library inventory changed')
        for name, value in contract['runtime_libraries'].items():
            require(file_sha(root / '.libs' / name) == value, 'Runtime dependency changed: ' + name)
        proof = decoded(source['proof'])
        prefix = 'mojolearn/' + leg + '/'
        for path, value in proof['extensions'].items():
            require(path.startswith(prefix) and file_sha(root / relative(path[len(prefix):])) == value,
                    'Device artifact changed during host composition')
    return dict(schema=SCHEMA, selected_leg=selected['leg'], selected_sha256=selected['host_sha256'],
        selected_proof_sha256=selected['proof_sha256'], source_commit=contract['source_commit'],
        compiler_sha256=contract['compiler_sha256'], host_contract_sha256=doc['host_contract_sha256'],
        receipt_sha256=digest(doc), variants=doc['variants'],
        qualification='Retained build provenance only; installed-wheel and numerical qualification remain required')


def inspect_build(root):
    root = Path(root).resolve()
    proof = json.loads((root / 'build-provenance.json').read_text())
    capture = json.loads((root / 'compile-inputs.json').read_text())
    leg = capture['vendor'] + '/' + capture['arch']
    setdir = root / 'sets' / leg
    files = host_files()
    caches = {f: proof['bincache']['mojolearn/' + leg + '/host/' + f] for f in files}
    keys = {f: json.loads((root / 'bincache/keys' / (c['key'] + '.json')).read_text()) for f, c in caches.items()}
    text = json.dumps(keys, sort_keys=True, indent=2) + '\n'
    source = dict(leg=leg, original_build_directory=str(root),
        proof=witness(root / 'build-provenance.json'), capture=witness(root / 'compile-inputs.json'),
        inventory=witness(root / 'source-inventory.json'), manifest=witness(setdir / 'manifest.json'),
        index=witness(setdir / 'host' / INDEX), readback=witness(setdir / 'readback.txt'),
        # every host group's cache key, {file: key}, as one witness
        host_cache_key={'sha256': hashlib.sha256(text.encode()).hexdigest(), 'utf8': text})
    gha = shard_evidence_root(root)
    if gha is not None:
        origins = {}
        for f, cache in caches.items():
            if cache.get('outcome') != 'hot-hit':
                continue
            token = 'host:merged_build.sh:' + f[:-3]
            matches = [p for p in (gha/'shards').glob('*/shard.tsv')
                       if any(line.split('\t')[0] == token for line in p.read_text().splitlines())]
            require(len(matches) == 1, 'Missing or ambiguous original host shard')
            shard = matches[0].parent
            origins[f] = {name:witness(path) for name,path in dict(shard=shard/'shard.tsv',
                provenance=shard/'bincache/provenance.tsv', key=shard/'bincache/keys'/(cache['key']+'.json'),
                archive=shard/'hot_sha256.tsv', assembly_archive=gha/'assemble/hot_sha256.tsv').items()}
        if origins:
            source['host_shard_origins'] = origins
    require(all(file_sha(setdir / 'host' / f) == proof['host_extension']['mojolearn/' + leg + '/host/' + f] for f in files),
            'Retained original host bytes differ from original proof')
    return source, setdir


def prepare(builds, selected_leg, compiler, output=None, source_root=None):
    require(source_root is not None, 'Frozen source root is required for host flag-scope proof')
    inspected = [inspect_build(root) for root in builds]
    sources = [row[0] for row in inspected]
    records = [_record(source, compiler) for source in sources]
    variants = [r[1] for r in records]
    selected = next((v for v in variants if v['leg'] == selected_leg), None)
    require(selected is not None, 'Selected host leg is absent')
    doc = dict(schema=SCHEMA, action='select-retained-host', expected_compiler_sha256=compiler,
        selected_leg=selected_leg, selected_sha256=selected['host_sha256'],
        host_contract_sha256=digest(records[0][0]), variants=variants, sources=sources,
        host_format_scope={name: witness(Path(source_root) / name) for name in HOST_FORMAT_SOURCES})
    contract, _ = validate_document(doc)
    # Validate original physical inputs before copying anything; no caller can
    # turn a missing/changed retained artifact into a successful composition.
    for source, setdir in inspected:
        for name, value in contract['runtime_libraries'].items():
            require(file_sha(setdir / '.libs' / name) == value, 'Original runtime dependency changed')
    if source_root is not None:
        for name, value in contract['source_inventory']:
            require(file_sha(Path(source_root) / relative(name)) == value, 'Current source differs from original build: ' + name)
    if output is None:
        return doc
    output = Path(output).resolve()
    require(not output.exists(), 'Assembly destination already exists; originals are never replaced')
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=output.name + '.pending-', dir=output.parent))
    try:
        canonical_set = next(p for s, p in inspected if s['leg'] == selected_leg)
        for source, setdir in inspected:
            destination = temporary / 'sets' / source['leg']
            shutil.copytree(setdir, destination, ignore=lambda path, names: ['host'] if Path(path) == setdir else [])
            shutil.copytree(canonical_set / 'host', destination / 'host')
        sets = {source['leg']: temporary / 'sets' / source['leg'] for source in sources}
        validate_assembly(doc, sets, [Path(s['original_build_directory']) / 'build-provenance.json' for s in sources], source_root)
        (temporary / 'host-composition.json').write_text(json.dumps(doc, sort_keys=True, indent=2) + '\n')
        temporary.rename(output)
    except BaseException:
        shutil.rmtree(temporary)
        raise
    return doc


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--build', action='append', required=True)
    parser.add_argument('--select', required=True, help='vendor/arch of the retained host to ship')
    parser.add_argument('--compiler-sha256', required=True, help='declared frozen Linux compiler hash')
    parser.add_argument('--source-root', type=Path, required=True)
    parser.add_argument('--out', type=Path, help='new assembly directory; omission validates only')
    args = parser.parse_args(argv)
    doc = prepare(args.build, args.select, args.compiler_sha256, args.out, args.source_root)
    print(json.dumps({k: doc[k] for k in ('schema', 'selected_leg', 'selected_sha256', 'host_contract_sha256', 'variants')}, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
