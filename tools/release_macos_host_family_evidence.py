"""Reconcile retained individual Mac host-family builds, without compiling.

Producer: --producer receipt=/absolute/path/to/original/receipt.json. The source
archive, manifest, actual raw binary and complete compile log must remain. This
adapter accepts both the family-controller and tokenizer-controller receipts;
relative default artifact names are those original controllers' fixed layout.
An absent original Apple auxiliary-toolchain witness remains unqualified. A
later machine snapshot or another job's toolchain is not substituted for it.
"""
import hashlib
from pathlib import Path, PurePosixPath
import re
import tarfile

import release_retained_compile_evidence as common
from release_ledger_admission import load, reference, require

NAME = 'mac-host-family-build'
ROUTING = {'MOJOLEARN_NUMERIC_MODE', 'MOJOLEARN_TARGET_COLUMN', 'MOJOLEARN_COMPILE_JOBS'}
KNOWN_ENV = ROUTING | {'MOJOLEARN_HOST_OUTDIR', 'MOJOLEARN_MOJO_BUILD_FLAGS'}


def supports_requirement(wanted):
    binding = wanted.get('inputs', {}).get('binding', {})
    return (common.supports_requirement(wanted) and binding.get('target') == common.MAC
            and binding.get('tier') == 'host')


def archive_sources(path, expected):
    """Hash regular archive members without extracting arbitrary producer paths."""
    found = {}
    with tarfile.open(path) as archive:
        for member in archive:
            name = member.name
            require(member.isfile() and name not in found and
                    not PurePosixPath(name).is_absolute() and '..' not in PurePosixPath(name).parts,
                    'source archive contains an unsafe, duplicate or non-file member: ' + name)
            stream = archive.extractfile(member)
            require(stream is not None, 'source archive member is unreadable: ' + name)
            found[name] = hashlib.file_digest(stream, 'sha256').hexdigest()
    require(found == expected, 'source archive bytes differ from the captured manifest')
    return found


def _facts(root, producer):
    path = Path(producer['receipt']).resolve()
    receipt = load(path)
    command = receipt.get('argv', receipt.get('command'))
    require(isinstance(command, list) and len(command) == 5 and
            command[:4] == ['bash', 'tools/compile_slot.sh', 'sh', 'bindings/build_host_family.sh'],
            'not the standard compile-slot host-family invocation')
    family = command[4]
    require(isinstance(family, str) and re.fullmatch('[a-z_]+', family), 'invalid host family')
    require(receipt.get('family', family) == family, 'receipt family differs from invocation')
    require(type(receipt.get('exit_code')) is int and receipt['exit_code'] == 0 and
            receipt.get('sources_changed_during_build') == [], 'compile failed or source changed during build')
    revision = receipt.get('source_commit')
    require(isinstance(revision, str) and common.HEX40.fullmatch(revision), 'missing original source revision')
    require('macOS' in receipt.get('target', '') and 'arm64' in receipt['target'], 'not a Mac arm64 host build')
    environment = receipt.get('environment', {})
    require(isinstance(environment, dict) and set(environment) <= KNOWN_ENV and
            all(isinstance(v, str) for v in environment.values()), 'unrecognized build environment')
    require(environment.get('MOJOLEARN_NUMERIC_MODE') == 'identical' and
            environment.get('MOJOLEARN_TARGET_COLUMN') == 'cpu' and
            re.fullmatch('[1-9][0-9]*', environment.get('MOJOLEARN_COMPILE_JOBS', '')),
            'missing exact identical CPU routing or compiler worker allocation')
    require(environment.get('MOJOLEARN_MOJO_BUILD_FLAGS') in ('', '--optimization-level 0'),
            'unsupported or uncaptured host optimization flags')
    compiler = receipt.get('compiler_sha256')
    require(isinstance(compiler, str) and common.HEX64.fullmatch(compiler), 'missing original compiler hash')
    manifest_path = Path(receipt.get('source_manifest', path.parent / 'source-manifest.json'))
    archive_path = Path(receipt.get('source_archive', path.parent / 'source.tar.gz'))
    sources = load(manifest_path)
    require(isinstance(sources, dict) and len(sources) == receipt.get('source_count') and
            all(isinstance(v, str) and common.HEX64.fullmatch(v) for v in sources.values()),
            'invalid captured source manifest')
    archive_ref = reference(archive_path)
    require(archive_ref['sha256'] == receipt.get('source_archive_sha256'), 'source archive hash differs')
    archive_sources(archive_path, sources)
    tree = common.Tree.at(root, revision)
    for rel, sha in sources.items():
        if rel in common.GENERATED:
            expected = common.generated_sha(tree, rel, receipt.get('cwd', root))
        else:
            expected = tree.sha256(rel)
        require(expected == sha, 'captured source differs from original revision: ' + rel)
    required = {'bindings/build_host_family.sh', 'bindings/build_defines.sh', 'tools/compile_slot.sh',
                'pixi.lock', 'pixi.toml', 'bindings/_mojolearn_' + family + '_host.mojo',
                'bindings/entry/entry__mojolearn_' + family + '_host.mojo'}
    require(required <= sources.keys(), 'captured source lacks required entry/build inputs')
    binary = reference(receipt['binary'])
    require(binary['sha256'] == receipt.get('binary_sha256') and
            Path(binary['path']).name == '_mojolearn_' + family + '_host.so' and
            Path(binary['path']).stat().st_size == receipt.get('binary_bytes'), 'raw binary differs from receipt')
    log = reference(receipt.get('log', path.parent / 'compile.log'))
    require(Path(log['path']).stat().st_size > 0, 'complete compiler log is empty')
    apple = receipt.get('apple_toolchain')
    apple_keys = {'build', 'developer_dir', 'metal', 'product', 'sdk', 'xcode'}
    require(apple is None or isinstance(apple, dict) and apple_keys <= apple.keys() and
            all(isinstance(apple[k], str) and apple[k] for k in apple_keys),
            'incomplete original Apple toolchain witness')
    require(apple is None or receipt.get('sdk', apple['sdk']) == apple['sdk'],
            'original SDK witnesses disagree')
    return dict(producer=NAME, source_root=str(Path(root).resolve()), source_revision=revision,
                family=family, environment=environment, compiler_sha256=compiler,
                source_files=sources, host_toolchain=apple, sdk=receipt.get('sdk'),
                limitations=[] if apple else ['Original Apple auxiliary-toolchain witness was not captured'],
                evidence=[reference(path), reference(manifest_path), archive_ref, binary, log])


def _collect(root, document, producer):
    facts = _facts(root, producer)
    key = common.MAC + '/host/-/host/_mojolearn_' + facts['family'] + '_host'
    specs = ['compile-apple']
    if (document or {}).get('configuration', {}).get('dependency_context', {}).get('cpu_target') == common.MAC:
        specs.append('compile-cpu')
    flags = common.derived_build_flags(facts['environment'], ROUTING)
    context = dict(cpu_target=common.MAC, apple_toolchain=facts['host_toolchain'],
                   compiler_sha256={common.MAC: facts['compiler_sha256']},
                   selected_binding_compile_jobs={key: facts['environment']['MOJOLEARN_COMPILE_JOBS']},
                   selected_binding_build_flags={key: flags})
    return dict(source_revision=facts['source_revision'], facts=facts,
                units={spec + '/' + key: dict(binding_key=key) for spec in specs}, spec_ids=specs,
                configuration=dict(release_modes=['identical'], build_flags={}, dependency_context=context))


def _check(requirement_id, wanted, facts, unit):
    require(supports_requirement(wanted), 'unsupported host compile requirement')
    binding, declared = wanted['inputs']['binding'], wanted['inputs']['unit']
    key = common.MAC + '/host/-/host/_mojolearn_' + facts['family'] + '_host'
    require(unit['binding_key'] == key == common.binding_key(binding) and requirement_id.endswith('/' + key),
            'admitted unit names another binding')
    require(facts['host_toolchain'] is not None, 'missing original Apple auxiliary-toolchain witness')
    require(binding.get('host_toolchain') == facts['host_toolchain'], 'Apple host toolchain differs')
    require(declared.get('compiler_sha256') == facts['compiler_sha256'], 'compiler executable differs')
    expected_flags = dict(numeric_mode='identical', target_column='cpu', gpu_archs=None,
                          compile_jobs=facts['environment']['MOJOLEARN_COMPILE_JOBS'], package_byte_lm='1')
    # package_byte_lm selects included families in release packaging. This direct
    # invocation already compiled the named family; build_host_family.sh never
    # reads that packaging selector. All compiler-affecting flags stay exact.
    tree = common.Tree.at(facts['source_root'], facts['source_revision'])
    require('MOJOLEARN_PACKAGE_BYTE_LM' not in tree.text('bindings/build_host_family.sh'),
            'host recipe now consumes an uncaptured packaging flag')
    require(binding.get('flags') == expected_flags, 'host routing or worker allocation differs')
    semantic = common.derived_build_flags(facts['environment'], ROUTING)
    require(declared.get('build_flags') == semantic and binding.get('selected_build_flags') == semantic,
            'host compiler flags differ')
    require(binding.get('toolchain', {}).get('pixi_lock_sha256') == facts['source_files']['pixi.lock'],
            'pinned compiler package lock differs')
    for rel in common.require_closure(tree, wanted):
        expected = (common.generated_sha(tree, rel, facts['source_root']) if rel in common.GENERATED
                    else tree.sha256(rel))
        require(facts['source_files'].get(rel) == expected,
                'required source was not captured at its tested bytes: ' + rel)
    return common.coverage_rows(wanted), list(facts['evidence'])


ADAPTERS = {NAME: common._Adapter(_collect, _facts, _check)}
