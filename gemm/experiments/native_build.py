#!/usr/bin/env python3
"""Compile a native Mojo experiment for its declared device column.

Process orchestration only. A build never executes the GPU harness. Native
recipes must select the accelerator and column together: the local host's
default accelerator cannot stand in for NVIDIA or AMD qualification.
The outer experiment runner owns the compile semaphore and build receipt.
"""
import argparse
import os
from pathlib import Path
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', type=Path)
    parser.add_argument('--vendor', choices=['nvidia', 'amd', 'apple', 'host'], required=True)
    parser.add_argument('--mode', choices=['identical', 'fast'], required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--source-sha')
    parser.add_argument('--mojo', default='mojo')
    parser.add_argument('--accelerator')
    parser.add_argument('--define', action='append', default=[])
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[2]
    sha = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=repo, text=True).strip()
    if args.source_sha and sha != args.source_sha:
        parser.error('requested source SHA does not match the checked-out build source')
    if subprocess.check_output(['git', 'diff', '--name-only', 'HEAD'], cwd=repo, text=True).strip():
        parser.error('freeze and commit source before building qualification artifacts')
    source = args.source.resolve()
    if not source.is_relative_to(repo) or not source.is_file():
        parser.error('source must be an existing file in this frozen worktree')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    command = [args.mojo, 'build', '-j1', '-I', str(repo)]
    if args.mode == 'identical':
        command += ['-D', 'MOJOLEARN_NUMERIC_IDENTICAL=1']
    column = 'CPU' if args.vendor == 'host' else args.vendor.upper()
    command += ['-D', 'MOJOLEARN_COLUMN_' + column + '=1']
    accelerator = args.accelerator or {'nvidia': 'sm_89', 'amd': 'gfx942', 'apple': 'metal:1'}.get(args.vendor)
    if args.vendor == 'apple':
        command += ['--target-cpu', 'apple-m1']
    if accelerator:
        command += ['--target-accelerator', accelerator]
    for define in args.define:
        if define.startswith('MOJOLEARN_COLUMN_') or define.startswith('MOJOLEARN_NUMERIC_'):
            parser.error('column and numeric mode must be selected through their explicit arguments')
        command += ['-D', define]
    command += [str(source), '-o', str(args.output)]
    return subprocess.run(command, cwd=repo, env=os.environ | {'MOJOLEARN_COMPILE_JOBS': '1'}).returncode


if __name__ == '__main__':
    sys.exit(main())
