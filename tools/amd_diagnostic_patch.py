"""Strict, diagnostic-only provenance for one patched AMD OOB binding."""
import hashlib
import json
from pathlib import Path
import re

MEMBER = 'mojolearn/hip_native/gfx942/identical/_mojolearn_x_trees.so'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate(manifest, proof, source, *, binding=None, wheels=None):
    doc = json.loads(Path(manifest).read_text())
    if (doc.get('schema') != 'mojolearn.amd-oob-patch.v1'
            or doc.get('archive_path') != MEMBER or doc.get('base_source_commit') != source
            or doc.get('experimental') is not True or doc.get('release_qualified') is not False):
        raise ValueError('Invalid diagnostic patch scope/source/qualification')
    for key, length in [('base_source_commit', 40), ('patch_source_commit', 40),
                        ('original_sha256', 64), ('patched_sha256', 64),
                        ('original_wheel_sha256', 64), ('build_proof_sha256', 64)]:
        if not re.fullmatch('[0-9a-f]{' + str(length) + '}', doc.get(key, '')):
            raise ValueError('Invalid patch digest: ' + key)
    if sha(proof) != doc['build_proof_sha256']:
        raise ValueError('Patch build proof hash differs')
    build = json.loads(Path(proof).read_text())
    if build.get('source_commit') != doc['patch_source_commit'] or build.get('complete') is not True:
        raise ValueError('Patch build proof source/completeness differs')
    if build.get('schema') == 'mojolearn.amd-diagnostic-patch-build.v1':
        if (build.get('architecture') != 'gfx942' or build.get('numeric_mode') != 'identical'
                or build.get('archive_path') != MEMBER):
            raise ValueError('Patch build target differs')
        artifact = build.get('sha256')
    elif build.get('schema') == 'mojolearn.linux.build-provenance.v1':
        extensions = build.get('extensions', {})
        artifact = extensions.get(MEMBER, extensions.get(MEMBER.replace('/hip_native/', '/hip/')))
    else:
        raise ValueError('Unsupported patch build proof schema')
    if artifact != doc['patched_sha256']:
        raise ValueError('Patched bytes differ from build proof')
    if binding is not None and sha(binding) != doc['patched_sha256']:
        raise ValueError('Patched binding hash differs')
    if wheels is not None:
        rows = [r for r in wheels if r['distribution'] == 'mojolearn-amd-gfx942']
        if len(rows) != 1 or rows[0]['sha256'] != doc['original_wheel_sha256']:
            raise ValueError('Original wheel hash differs')
        import zipfile
        with zipfile.ZipFile(rows[0]['path']) as archive:
            if hashlib.sha256(archive.read(MEMBER)).hexdigest() != doc['original_sha256']:
                raise ValueError('Original binding hash differs')
    return doc


def apply(manifest, proof, binding, package_parent, source, wheels):
    doc = validate(manifest, proof, source, binding=binding, wheels=wheels)
    parent = Path(package_parent).resolve()
    dest = parent / MEMBER
    if dest.is_symlink() or not dest.resolve().is_relative_to(parent) or sha(dest) != doc['original_sha256']:
        raise ValueError('Installed original binding differs or escapes package')
    dest.write_bytes(Path(binding).read_bytes())
    if sha(dest) != doc['patched_sha256']:
        raise ValueError('Installed patch write differs')
    return dict(manifest=doc, loaded_path=str(dest), qualification=False)
