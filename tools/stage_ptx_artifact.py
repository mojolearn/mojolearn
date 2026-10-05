#!/usr/bin/env python3
"""Retain experimental build evidence without traversing container virtualenvs."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate(root, source):
    base = root / 'ptx-baseline'
    proof = json.loads((base / 'experimental-build.json').read_text())
    tree = base / 'build/sets/cuda/sm_80'
    manifest = json.loads((tree / 'PTX_BASELINE.json').read_text())
    for doc in (proof, manifest):
        if doc.get('source_commit') != source or doc.get('source_dirty') is not False:
            raise ValueError('Artifact source differs or is dirty')
        if doc.get('code_format') != 'ptx-baseline' or doc.get('identical_qualified') is not False:
            raise ValueError('Not an unqualified experimental PTX artifact')
    pairs = {'manifest_sha256': tree / 'PTX_BASELINE.json', 'readback_sha256': tree / 'readback.txt',
             'runtime_manifest_sha256': tree / 'manifest.json', 'portable_math_sha256': base / 'libMojolearnMath.so'}
    for key, path in pairs.items():
        if sha(path) != proof.get(key):
            raise ValueError('Build proof hash differs: ' + key)
    runtime = json.loads((tree / 'manifest.json').read_text())
    expected = [(row['file'], row['sha256']) for row in manifest['files']]
    expected += [(row['path'], row['sha256']) for row in runtime['extensions']]
    expected += [('.libs/' + row['name'], row['sha256']) for row in runtime['staged_libs']]
    for name, digest in expected:
        rel = Path(name)
        if rel.is_absolute() or '..' in rel.parts or sha(tree / rel) != digest:
            raise ValueError('Missing or altered artifact member: ' + name)
    return len(manifest['files'])


def stage(source, output, commit, require_complete=False):
    source, output = Path(source), Path(output)
    output.mkdir(parents=True, exist_ok=False)
    kept = {}
    # Never walk tools/, venv/, or arbitrary build scratch directories.
    roots = ['bootstrap', 'ptx-baseline/build/sets']
    def copy(path):
        if path.is_symlink() or not path.is_file():
            return
        rel = path.relative_to(source)
        dest = output / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, dest)
        kept[rel.as_posix()] = sha(dest)
    for name in roots:
        start = source / name
        if start.is_symlink() or not start.exists():
            continue
        for directory, dirs, files in os.walk(start, followlinks=False):
            dirs[:] = [d for d in dirs if not (Path(directory) / d).is_symlink()]
            for name in files:
                copy(Path(directory) / name)
    base = source / 'ptx-baseline'
    if base.is_dir() and not base.is_symlink():
        for path in base.iterdir():
            if path.suffix in ('.json', '.log', '.txt', '.env') or path.name == 'libMojolearnMath.so':
                copy(path)
    report = dict(schema='mojolearn.ptx-artifact-retention.v1', source_commit=commit,
                  complete=False, files=kept, qualification=False)
    error = None
    try:
        report['gpu_binding_files'] = validate(output, commit)
        report['complete'] = True
    except (OSError, ValueError, KeyError, TypeError) as exc:
        error = exc
        report['error'] = str(exc)
    (output / 'retention.json').write_text(json.dumps(report, indent=2) + '\n')
    if require_complete and error:
        raise ValueError('Successful build artifact is incomplete: ' + str(error)) from error
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--source-commit', required=True)
    parser.add_argument('--require-complete', action='store_true')
    args = parser.parse_args()
    report = stage(args.source, args.output, args.source_commit, args.require_complete)
    print(json.dumps({k: v for k, v in report.items() if k != 'files'}))


if __name__ == '__main__':
    main()
