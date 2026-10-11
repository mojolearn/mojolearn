#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Audit the PTX set (cuda_ptx/sm_80) of mojolearn-nvidia and write its manifest.

Andrew 2026-10-10: PTX is a normal target; no flag. This report is the PTX
set's build-format witness (target, no embedded cubin, IDENTICAL arithmetic
rounding-pinned, approximate instructions inventoried, final file hashes).
Its numerical identity is decided like every native set's: by its column in
the reference table (python/mojolearn/verify_reference/table.json), whose
digests must equal the NVIDIA == AMD digests. Native cubin gates remain
independent.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess

from ptx_contract import APPROX, modules, plain_ops
from device_glue import DEVICE_GLUE

FORMAT = 'ptx'
SCHEMA = 'mojolearn.ptx-set.v2'
TARGET = 'sm_80'
_VERSION = re.compile(rb'^\s*\.version\s+(\d+\.\d+)\s*$', re.M)
_TARGET = re.compile(rb'^\s*\.target\s+([^\r\n]+)', re.M)
_FATBIN = b'\x50\xed\x55\xba'


def validate_config(code_format, arch):
    if code_format not in ('native', FORMAT):
        raise ValueError(f'unsupported CUDA code format: {code_format}')
    if code_format == FORMAT and arch != TARGET:
        raise ValueError('the PTX set requires explicit MOJOLEARN_GPU_ARCHS=sm_80')


def audit_binary(data, identical=True):
    errors, rows = [], []
    if _FATBIN in data:
        errors.append('native fatbin present in portable PTX artifact')
    # Reject embedded CUDA ELF code objects, including ones outside a fatbin.
    for m in re.finditer(b'\x7fELF', data):
        offset = m.start()
        if len(data) >= offset + 20 and data[offset + 5] == 1:
            if int.from_bytes(data[offset + 18:offset + 20], 'little') == 190:
                errors.append('native CUDA ELF code object present')
    spans = modules(data)
    if len(re.findall(rb'\.version \d+\.\d+', data)) != len(spans):
        errors.append('unrecognized or malformed PTX module header')
    for start, end in spans:
        body = data[start:end]
        versions, targets = _VERSION.findall(body), _TARGET.findall(body)
        if len(versions) != 1 or len(targets) != 1:
            errors.append('PTX module must have exactly one version and target')
        target = targets[0].strip().decode('ascii', 'replace') if targets else ''
        if target != TARGET:
            errors.append(f'non-baseline PTX target: {target}')
        if not re.search(rb'^\s*\.address_size\s+64\s*$', body, re.M):
            errors.append('PTX module requires address_size 64')
        if identical and plain_ops(body):
            errors.append('IDENTICAL PTX contains unpinned floating arithmetic')
        approx = {}
        for op in APPROX.findall(body):
            name = op.decode('ascii')
            approx[name] = approx.get(name, 0) + 1
        rows.append(dict(sha256=hashlib.sha256(body).hexdigest(), target=target,
                         ptx_isa=versions[0].decode() if versions else '', approx=approx))
    return rows, errors


def audit_tree(root, source_commit, mojo_version, source_dirty=False):
    if not re.fullmatch(r'[0-9a-f]{40}', source_commit):
        raise ValueError('source_commit must be a full git SHA')
    if not mojo_version.strip():
        raise ValueError('Mojo toolchain version must be recorded')
    root = Path(root)
    rows, errors = [], []
    for path in sorted(root.rglob('*.so')):
        rel = path.relative_to(root)
        if any(part in ('host', '.libs') for part in rel.parts):
            continue
        mode = rel.parts[0] if rel.parts[0] in ('identical', 'deterministic') else 'fast'
        data = path.read_bytes()
        mods, failures = audit_binary(data, identical=mode == 'identical')
        errors.extend(f'{rel.as_posix()}: {error}' for error in failures)
        rows.append(dict(file=rel.as_posix(), sha256=hashlib.sha256(data).hexdigest(),
                         numeric_mode=mode, ptx_modules=mods))
    indexed = {(row['numeric_mode'], Path(row['file']).stem): row for row in rows}
    for row in rows:
        if row['ptx_modules']:
            continue
        delegates = DEVICE_GLUE.get((row['numeric_mode'], Path(row['file']).stem))
        if not delegates:
            errors.append(f"{row['file']}: unregistered binary without PTX modules")
            continue
        witnesses = [indexed.get((row['numeric_mode'], name)) for name in delegates]
        if not all(witness and witness['ptx_modules'] for witness in witnesses):
            errors.append(f"{row['file']}: missing PTX device delegates")
        else:
            row['delegates'] = [dict(file=w['file'], sha256=w['sha256']) for w in witnesses]
    if not any(row['ptx_modules'] for row in rows):
        errors.append('PTX set contains no PTX modules')
    return dict(schema=SCHEMA, code_format=FORMAT, vendor='cuda', target=TARGET,
                min_compute_capability=[8, 0], source_commit=source_commit,
                source_dirty=source_dirty, mojo_version=mojo_version.strip(),
                files=rows, errors=errors)


def source_witness(repo):
    """The commit the set was built from, and whether its tracked source was dirty.

    A checkout answers through git. The release build route (the GitHub Actions
    container, the cpu build boxes) builds an exported archive of the frozen
    commit: no .git, no git binary, and commit.txt as its witness, the same rule
    as the route's preflight (tools/release061_remote_build.sh). An archive is
    clean by construction. MOJOLEARN_COMMIT, when set, must agree.
    """
    repo = Path(repo)
    env_sha = os.environ.get('MOJOLEARN_COMMIT', '').strip()
    if (repo / '.git').exists():
        sha = subprocess.check_output(['git', '-C', str(repo), 'rev-parse', 'HEAD'], text=True).strip()
        dirty = bool(subprocess.check_output(['git', '-C', str(repo), 'status', '--porcelain',
                                             '--untracked-files=no'], text=True).strip())
    else:
        witness = repo / 'commit.txt'
        if not witness.is_file():
            raise SystemExit('ptx_baseline: %s is neither a git checkout nor an archive with commit.txt' % repo)
        sha, dirty = witness.read_text().strip(), False
    if env_sha and env_sha != sha:
        raise SystemExit('ptx_baseline: MOJOLEARN_COMMIT %s differs from the source witness %s'
                         % (env_sha[:12], sha[:12]))
    return sha, dirty


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    config = sub.add_parser('validate-config')
    config.add_argument('--code-format', required=True)
    config.add_argument('--arch', default='')
    audit = sub.add_parser('audit')
    audit.add_argument('root', type=Path)
    audit.add_argument('--repo', required=True, type=Path)
    audit.add_argument('--mojo-version', required=True)
    audit.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    if args.command == 'validate-config':
        try:
            validate_config(args.code_format, args.arch)
        except ValueError as error:
            parser.error(str(error))
        return 0
    sha, dirty = source_witness(args.repo)
    report = audit_tree(args.root, sha, args.mojo_version, dirty)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + '\n')
    for error in report['errors']:
        print(error)
    return int(bool(report['errors']))


if __name__ == '__main__':
    raise SystemExit(main())
