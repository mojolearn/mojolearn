#!/usr/bin/env python3
"""Bind a root-observed guard exit to retained command/log/result bytes.

File-only helper: never executes the command or a model. Root must call this
after the actual guard has exited, passing its captured shell return status.
The receipt is provenance, not a proof against a malicious artifact author.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path


def retained(path, parent, maximum):
    path = Path(path)
    if path.is_symlink() or not path.is_file():
        raise ValueError('Expected retained regular file: ' + str(path))
    resolved = path.resolve(strict=True)
    try:
        relative = resolved.relative_to(parent)
    except ValueError as exc:
        raise ValueError('All receipt artifacts must be under its parent directory') from exc
    with resolved.open('rb') as stream:
        raw = stream.read(maximum + 1)
    if len(raw) > maximum:
        raise ValueError('Retained artifact exceeds bound: ' + str(path))
    return dict(file=relative.as_posix(), sha256=hashlib.sha256(raw).hexdigest(), bytes=len(raw)), raw


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--vendor', choices=('cuda', 'hip'), required=True)
    parser.add_argument('--exit-code', type=int, required=True)
    parser.add_argument('--job-kind', choices=('capture', 'oracle'), required=True)
    parser.add_argument('--command-file', type=Path, required=True)
    parser.add_argument('--guard-log', type=Path, required=True)
    parser.add_argument('--result', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if not 0 <= args.exit_code <= 255:
        parser.error('Expected the actual shell return status in [0,255]')
    parent = args.output.parent.resolve(strict=True)
    command, command_raw = retained(args.command_file, parent, 16384)
    log, log_raw = retained(args.guard_log, parent, 8 * 1024 * 1024)
    result, _ = retained(args.result, parent, 2 * 1024 * 1024)
    if not command_raw.strip():
        raise ValueError('Empty retained command')
    expected = {'cuda': 'nvidia-root-serial-v1', 'hip': 'amd-root-serial-v1'}[args.vendor]
    lines = log_raw.decode('utf-8', errors='strict').splitlines()
    terminal = json.loads(next(line for line in reversed(lines) if line.strip()))
    if terminal.get('guard') != expected:
        raise ValueError('Terminal guard record does not match vendor')
    if args.exit_code == 0:
        if terminal.get('reason') is not None or terminal.get('returncode') != 0:
            raise ValueError('A failing guard cannot receive a successful receipt')
        cores = str(terminal.get('cpu_affinity', '')).split(',')
        if not 1 <= len(cores) <= 2 or not all(core.isdigit() for core in cores):
            raise ValueError('Missing bounded CPU-affinity witness')
        if terminal.get('thread_limit') != 2:
            raise ValueError('Missing two-thread guard witness')
    receipt = dict(schema='mojolearn.root-job-receipt.v1', vendor=args.vendor,
                   exit_code=args.exit_code, job_kind=args.job_kind,
                   command=command, guard_log=log, result=result,
                   guard_terminal=terminal,
                   boundary='Root-observed guard return status, including cleanup; file-only receipt writer')
    with args.output.open('x') as stream:
        json.dump(receipt, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())


if __name__ == '__main__':
    main()
