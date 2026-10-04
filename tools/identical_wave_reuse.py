"""Fail-closed import of audited native build receipts into a fresh wave tree."""
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess

# This producer fixes the numeric mode, arm flags, architecture, host override,
# and single compiler job in its environment. Other producers need a new audit.
AUDITED_PRODUCER = 'e948a3db1f580e2bd0935abb7231b070aee34072a96b3378cf57a7f36f275b69'


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def require(condition, reason):
    if not condition:
        raise ValueError('native reuse refused: ' + reason)


def clean_source(source, sha):
    head = subprocess.check_output(['git', '-C', str(source), 'rev-parse', 'HEAD'], text=True).strip()
    require(head == sha, 'source HEAD differs')
    require(subprocess.run(['git', '-C', str(source), 'diff', '--quiet', 'HEAD', '--']).returncode == 0,
            'tracked source changed')


def checked_file(record, path, allow_alias=False):
    path = Path(path)
    require(path.is_file(), 'missing file: ' + str(path))
    require((Path(record['path']).resolve() == path.resolve()) if allow_alias else
            (Path(record['path']).absolute() == path.absolute()), 'recorded file path differs')
    require(Path(record['resolved_path']) == path.resolve(), 'resolved file path differs')
    require(record['sha256'] == digest(path), 'file hash differs: ' + str(path))


def dependency_fingerprint(source, binding):
    """Conservative repository closure for the reviewed initializer repair.

    Every tracked input is retained except other standalone binding entrypoints
    and the explicitly reviewed external DART diagnostic. Future unrelated
    source changes therefore refuse reuse rather than silently broadening it.
    """
    module = {'base': '_mojolearn', 'base_host': '_mojolearn_core_host'}.get(binding, '_mojolearn_' + binding)
    own = 'bindings/' + module + '.mojo'
    probe = subprocess.run(['git', '-C', str(source), 'grep', '-n', '-E',
                            r'bindings\._mojolearn_|(from|import)[[:space:]].*_mojolearn_',
                            'HEAD', '--', '*.mojo'], capture_output=True)
    require(probe.returncode == 1, 'binding entrypoints are imported or import audit failed')
    tree = subprocess.check_output(['git', '-C', str(source), 'ls-tree', '-r', '-z', 'HEAD'])
    kept = []
    for entry in tree.split(b'\0'):
        if not entry:
            continue
        name = entry.split(b'\t', 1)[1].decode()
        if name == 'tools/identical_dart_bit_diagnostic.py':
            continue
        if re.fullmatch(r'bindings/_mojolearn_[a-z0-9_]+\.mojo', name) and name != own:
            continue
        kept.append(entry)
    require(any(e.endswith(b'\t' + own.encode()) for e in kept), 'binding entrypoint absent')
    return hashlib.sha256(b'\0'.join(kept) + b'\0').hexdigest()


def validate_receipt(receipt, *, sha, vendor, arch, arm, builders, environment, python, pixi,
                     _allow_failed_donor=False, _depth=0):
    """Validate every source artifact before returning the requested subset.

    No files are copied here. The producer attestation was captured during the
    build; the completed receipt, clean source and toolchain are rechecked now.
    """
    receipt = Path(receipt).resolve()
    attestation = receipt.parent / 'producer-attestation.json'
    raw = receipt.read_bytes()
    doc = json.loads(raw)
    att = json.loads(attestation.read_bytes())
    source = receipt.parent / 'source'
    expected = dict(sha=sha, vendor=vendor, arch=arch, arm=arm, mode='identical')
    require(_depth < 3, 'excessive or cyclic donor chain')
    require(doc.get('status') == 'PASS' or (_allow_failed_donor and doc.get('status') == 'FAILED'),
            'receipt is not complete PASS')
    require(all(doc.get(k) == v for k, v in expected.items()), 'source/vendor/arch/mode/arm differs')
    require(att.get('schema') == 1 and att.get('source_clean') is True, 'missing clean-source attestation')
    require(Path(att['source_path']).resolve() == source.resolve(), 'attested source path differs')
    require(att.get('source_sha') == sha, 'attested source SHA differs')
    require(all(att.get(k) == v for k, v in dict(vendor=vendor, gpu_arch=arch, arm=arm, mode='identical', jobs=1).items()),
            'attested build configuration differs')
    flags = '-D MOJOLEARN_IDN_ALL_OFF=1' if arm == 'off' else ''
    require(att.get('mojo_build_flags') == flags, 'arm flags differ')
    require(att.get('compile_environment') == dict(MOJOLEARN_NUMERIC_MODE='identical',
            MOJOLEARN_COMPILE_JOBS='1', MOJOLEARN_GPU_ARCHS=arch, MOJOLEARN_TARGET_COLUMN=vendor),
            'compile environment differs')
    require(att.get('host_override') == dict(MOJOLEARN_GPU_ARCHS=None, MOJOLEARN_TARGET_COLUMN='cpu'),
            'host override differs')
    clean_source(source, sha)
    checked_file(att['pixi_lock'], source / 'pixi.lock')
    require(digest(source / 'pixi.lock') == digest(Path(environment) / 'pixi.lock'), 'environment lock differs')
    checked_file(att['compiler'], Path(environment) / '.pixi/envs/default/bin/mojo')
    require(bool(att['compiler'].get('version')), 'compiler version absent')
    checked_file(att['python'], Path(python), allow_alias=True)
    checked_file(att['pixi'], Path(pixi))
    checked_file(att['producer'], receipt.parent / 'native-builder-producer.py')
    require(att['producer']['sha256'] == AUDITED_PRODUCER, 'producer has not been audited')
    require(digest(source / 'tools/identical_wave_native_build.py') == AUDITED_PRODUCER,
            'producer does not match frozen source')
    expected_modules = doc.get('expected_builders', [])
    modules = doc.get('modules', {})
    require(expected_modules and len(set(expected_modules)) == len(expected_modules)
            and set(modules) == set(expected_modules), 'partial or duplicate module inventory')
    require(doc.get('bootstrap') and all(r.get('rc') == 0 for r in doc['bootstrap']), 'bootstrap incomplete')
    inventory_path = receipt.parent / 'artifact-inventory.json'
    inventory = json.loads(inventory_path.read_bytes())
    actual_inventory = {str(p.relative_to(source)): digest(p)
                        for p in (source / 'python/mojolearn').rglob('*.so')}
    require(inventory and actual_inventory == inventory, 'complete artifact inventory differs')
    selected, all_paths = {}, set()
    for binding, row in modules.items():
        builder = {'base': 'build.sh', 'base_host': 'build_core_host.sh'}.get(binding, 'build_' + binding + '.sh')
        if not _allow_failed_donor or builder in builders:
            require(row.get('status') == 'PASS' and row.get('rc') == 0
                    and row.get('import_smoke', {}).get('rc') == 0, 'module did not pass: ' + binding)
        module = {'base': '_mojolearn', 'base_host': '_mojolearn_core_host'}.get(binding, '_mojolearn_' + binding)
        relative = Path('python/mojolearn') / ('host' if binding.endswith('_host') else 'identical') / (module + '.so')
        artifact = source / relative
        require(Path(row['artifact']).absolute() == artifact.absolute(), 'artifact path differs: ' + binding)
        require(artifact.is_file() and not artifact.is_symlink() and artifact.resolve().is_relative_to(source.resolve()),
                'artifact missing or escapes source: ' + binding)
        require(digest(artifact) == row.get('sha256'), 'artifact hash differs: ' + binding)
        all_paths.add(relative)
        if builder in builders:
            selected[builder] = dict(path=str(artifact), relative=str(relative), sha256=row['sha256'])
            if row.get('reused_from'):
                prior = row['reused_from']
                require(prior.get('module') == binding, 'donor module differs')
                require(digest(prior['receipt']) == prior['receipt_sha256'], 'donor receipt changed')
                require(digest(prior['dependency_proof']) == prior['dependency_proof_sha256'], 'dependency proof changed')
                donor = validate_receipt(prior['receipt'], sha=prior['source_sha'], vendor=vendor,
                    arch=arch, arm=arm, builders=[builder], environment=environment, python=python, pixi=pixi,
                    _allow_failed_donor=True, _depth=_depth + 1)
                require(donor['builders'][builder]['sha256'] == row['sha256'], 'copied module differs from donor')
                before = dependency_fingerprint(donor['source'], binding)
                after = dependency_fingerprint(source, binding)
                require(before == after, 'compiled dependency closure differs: ' + binding)
                proof = json.loads(Path(prior['dependency_proof']).read_bytes())
                require(proof.get('old_source_sha') == prior['source_sha']
                        and proof.get('new_source_sha') == sha and proof.get('module') == binding
                        and proof.get('identical_closure_sha256') == after, 'dependency proof metadata differs')
                selected[builder]['dependency_equivalence'] = dict(closure_sha256=after,
                    rule='conservative-repository-v1', donor=donor,
                    dependency_proof_sha256=prior['dependency_proof_sha256'])
    helper = Path('python/mojolearn/.libs/libMojolearnMath.so')
    require((source / helper).is_file() and not (source / helper).is_symlink(), 'portable math helper absent or symlinked')
    all_paths.add(helper)
    actual = {p.relative_to(source) for p in (source / 'python/mojolearn').rglob('*.so')}
    require(actual == all_paths, 'unexpected or missing source .so artifacts')
    require(selected, 'receipt supplies no requested builders')
    return dict(receipt=str(receipt), receipt_sha256=hashlib.sha256(raw).hexdigest(),
                attestation=str(attestation), attestation_sha256=digest(attestation),
                artifact_inventory_sha256=digest(inventory_path),
                compiler_sha256=att['compiler']['sha256'], pixi_lock_sha256=att['pixi_lock']['sha256'],
                source=str(source), builders=selected,
                helper=dict(path=str(source / helper), relative=str(helper), sha256=digest(source / helper)))


def import_artifacts(validated, destination):
    """Copy into a new source tree; verify both ends and never mutate donor files."""
    destination = Path(destination)
    items = [*validated['builders'].values(), validated['helper']]
    for item in items:
        original = Path(item['path'])
        target = destination / item['relative']
        require(digest(original) == item['sha256'], 'donor changed after validation')
        require(target.resolve().is_relative_to(destination.resolve()), 'destination escapes source')
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            require(not target.is_symlink() and digest(target) == item['sha256'], 'destination collision')
        else:
            with original.open('rb') as src, target.open('xb') as dst:
                shutil.copyfileobj(src, dst)
            target.chmod(original.stat().st_mode & 0o777)
        require(digest(target) == item['sha256'], 'copied artifact hash differs')
