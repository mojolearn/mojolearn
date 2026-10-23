#!/usr/bin/env python3
"""Build/validate the eight supported autograd gate executables.

Only the release orchestrator executes `build`, once per vendor and freeze.
This uses autograd_smoke's existing supported Mojo build commands unchanged
except an explicitly requested compiler worker count. Linux x86-64 checks use
the same x86-64-v3 host baseline as the wheels. Outputs/logs/receipt must
live under a retained /root/results* directory on a build box.
"""
import argparse
import hashlib
import json
import os
import platform
from pathlib import Path
import shutil
import subprocess
import time

import autograd_smoke as AS

ROOT = Path(__file__).resolve().parents[1]
SCHEMA = 'mojolearn.autograd-gate-artifacts/1'


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()


def expected(vendor):
    return {AS.binary('.', row, vendor).name: row['source'] for row in AS.ENTRIES if 'source' in row}


def validate(manifest_path, vendor, source_commit, directory=None):
    manifest_path = Path(manifest_path)
    doc = json.loads(manifest_path.read_text())
    directory = Path(directory) if directory is not None else manifest_path.parent
    want = expected(vendor)
    if (doc.get('schema') != SCHEMA or doc.get('state') != 'built' or doc.get('vendor') != vendor or doc.get('source_commit') != source_commit
            or not doc.get('target') or not doc.get('compiler', {}).get('sha256')
            or len(doc.get('files', [])) != len(want) or {r['file'] for r in doc['files']} != set(want)):
        raise ValueError('incomplete/wrong-source autograd artifact receipt')
    target_cpu = AS.HOST_TARGET_CPU.get(vendor)
    if not target_cpu or doc.get('target_cpu') != target_cpu:
        raise ValueError('autograd artifact receipt lacks the portable Linux host CPU baseline')
    for row in doc['files']:
        path = directory/row['file']
        command = row.get('command') or []
        if (not isinstance(command, list) or command.count('--target-cpu') != 1
                or command.index('--target-cpu') + 1 >= len(command)
                or command[command.index('--target-cpu') + 1] != target_cpu):
            raise ValueError('autograd compile command lacks the portable host CPU baseline: '+row['file'])
        if (row.get('exit_code') != 0 or row.get('source') != want[row['file']] or not row.get('source_sha256')
                or not row.get('command') or not path.is_file() or not os.access(path, os.X_OK)
                or sha(path) != row['sha256']):
            raise ValueError('autograd binary/provenance differs: '+row['file'])
    return doc


def build(args):
    if platform.system() != 'Linux' or platform.machine().lower() not in ('x86_64', 'amd64'):
        raise RuntimeError('NVIDIA/AMD gate artifacts require a Linux x86-64 build host')
    def git(*parts):
        return subprocess.check_output(['git', '-C', str(ROOT), *parts], text=True).strip()
    if git('rev-parse', 'HEAD') != args.source_commit or git('status', '--porcelain', '--untracked-files=no'):
        raise RuntimeError('build autograd artifacts from the clean, exact candidate source commit')
    out = Path(args.out).resolve()
    if out.exists() and not args.resume:
        raise RuntimeError('use a fresh artifact output directory')
    out.mkdir(parents=True, exist_ok=args.resume)
    compiler = Path(shutil.which(args.mojo) or args.mojo).resolve()
    version = subprocess.check_output([str(compiler), '--version'], text=True).strip()
    doc = dict(schema=SCHEMA, source_commit=args.source_commit, vendor=args.vendor, target=args.target,
               target_cpu=AS.HOST_TARGET_CPU[args.vendor],
               compiler=dict(path=str(compiler), sha256=sha(compiler), version=version), files=[], state='building')
    receipt = out/'build-manifest.json'
    prior = {}
    if args.resume:
        previous = json.loads(receipt.read_text())
        if any(previous.get(key) != doc[key] for key in ('schema', 'source_commit', 'vendor', 'target', 'target_cpu', 'compiler')):
            raise RuntimeError('resume requires the same source, GPU/CPU targets and compiler')
        history = [row for attempt in previous.get('prior_attempts', []) for row in attempt]
        prior = {row['file']: row for row in history+previous.get('files', [])}
        doc['prior_attempts'] = previous.get('prior_attempts', []) + [previous.get('files', [])]
    def retain():
        temporary = receipt.with_suffix('.tmp')
        temporary.write_text(json.dumps(doc, indent=2)+'\n')
        temporary.replace(receipt)
    retain()
    commands = AS.build_commands(args.vendor, str(compiler), str(out), args.target)
    try:
        for command, (name, source) in zip(commands, expected(args.vendor).items()):
            if args.compile_jobs is not None:
                command[command.index('-j')+1] = str(args.compile_jobs)
            before = sha(ROOT/source)
            old = prior.get(name)
            if old and old.get('exit_code') == 0 and old.get('sha256'):
                if (old.get('command') != command or old.get('source_sha256') != before
                        or not (out/name).is_file() or sha(out/name) != old['sha256']):
                    raise RuntimeError('completed autograd artifact changed; use a new output directory: '+name)
                doc['files'].append(old)
                retain()
                continue
            log = out/(name+'-'+str(time.time_ns())+'.log')
            with log.open('w') as output:
                code = subprocess.run(command, cwd=ROOT, stdout=output, stderr=subprocess.STDOUT).returncode
            row = dict(file=name, source=source, source_sha256=before, command=command,
                       exit_code=code, log=log.name)
            doc['files'].append(row)
            if code or not (out/name).is_file():
                raise RuntimeError('autograd compile failed: '+name)
            row['sha256'] = sha(out/name)
            if sha(ROOT/source) != before or git('rev-parse', 'HEAD') != args.source_commit or git('status', '--porcelain', '--untracked-files=no'):
                raise RuntimeError('candidate source changed during compilation')
            retain()
        if (git('rev-parse', 'HEAD') != args.source_commit or git('status', '--porcelain', '--untracked-files=no')
                or sha(compiler) != doc['compiler']['sha256']):
            raise RuntimeError('source/compiler changed before completing the artifact set')
        doc['state'] = 'built'
    except BaseException as error:
        doc.update(state='failed', error=str(error))
        raise
    finally:
        retain()
    validate(receipt, args.vendor, args.source_commit)
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('build',))
    parser.add_argument('--vendor', choices=('nvidia', 'amd'), required=True)
    for name in ('source-commit', 'target', 'mojo', 'out'):
        parser.add_argument('--'+name, required=True)
    parser.add_argument('--compile-jobs', type=int)
    parser.add_argument('--resume', action='store_true', help='reuse completed unchanged artifacts from the same freeze/compiler/target; retain old logs')
    args = parser.parse_args()
    if args.compile_jobs is not None and args.compile_jobs < 1:
        parser.error('compile-jobs must be positive')
    return build(args)


if __name__ == '__main__':
    raise SystemExit(main())
