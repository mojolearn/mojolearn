#!/usr/bin/env python3
"""Bind a root-observed guard exit to retained command/log/result bytes.

File-only helper: never executes the command or a model. Root must call this
after the actual guard has exited, passing its captured shell return status.
The receipt is provenance, not a proof against a malicious artifact author.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path


def validate_guard_terminal(terminal, vendor, exit_code, job_kind='capture'):
    """Shared file-only admission, with distinct honest Metal safety evidence."""
    if not isinstance(terminal, dict):
        raise ValueError('Missing terminal guard object')
    names = {'cuda': 'nvidia-root-serial-v1', 'hip': 'amd-root-serial-v1',
             'metal': 'macos-root-serial-v1'}
    if vendor not in names or terminal.get('guard') != names[vendor]:
        raise ValueError('Terminal guard record does not match vendor')
    if vendor == 'metal' and job_kind != 'capture':
        raise ValueError('Metal oracle receipts are not supported')
    if type(exit_code) is not int or not 0 <= exit_code <= 255:
        raise ValueError('Invalid root exit status')
    if exit_code != 0:
        return  # retain failed evidence without admitting success
    if (terminal.get('reason') is not None or type(terminal.get('returncode')) is not int
            or terminal['returncode'] != 0 or type(terminal.get('thread_limit')) is not int
            or terminal['thread_limit'] != 2):
        raise ValueError('Unsuccessful guard or missing two-thread witness')
    if vendor != 'metal':
        cores = str(terminal.get('cpu_affinity', '')).split(',')
        if not 1 <= len(cores) <= 2 or not all(core.isdigit() for core in cores):
            raise ValueError('Missing bounded CPU-affinity witness')
        return
    if ('error' not in terminal or terminal.get('error') is not None or terminal.get('cpu_affinity', 'missing') is not None
            or terminal.get('cpu_enforcement') != 'sampled, not hard affinity'
            or terminal.get('gpu_memory_accounting') != 'system unified-memory reserve/pressure; no per-process Metal VRAM counter'
            or terminal.get('watchdog_errors') != [] or terminal.get('watchdog_expired') is not False):
        raise ValueError('Metal requires honest accounting and clean watchdog evidence')
    cleanup = terminal.get('cleanup')
    if (not isinstance(cleanup, dict) or cleanup.get('verified') is not True
            or cleanup.get('quarantined') is not False or cleanup.get('errors') != []):
        raise ValueError('Metal cleanup must be verified without quarantine/errors')
    policy = terminal.get('policy_limits')
    fixed = dict(entry_reserve_bytes=4 * 2**30, runtime_reserve_bytes=2 * 2**30,
                 normal_pressure_level=1, swap_growth_bytes=128 * 2**20,
                 compressed_growth_bytes=256 * 2**20, sampled_cpu_limit=3,
                 cpu_grace_seconds=4, sample_interval_seconds=2.0)
    if (not isinstance(policy, dict) or set(policy) != set(fixed) | {'deadline_seconds', 'rss_bytes'}
            or any(type(policy[k]) not in (int, float) or policy[k] != v for k, v in fixed.items())
            or type(policy['deadline_seconds']) is not int or not 1 <= policy['deadline_seconds'] <= 180
            or type(policy['rss_bytes']) is not int or not 2**30 <= policy['rss_bytes'] <= 4 * 2**30):
        raise ValueError('Missing or weakened Metal policy limits')
    def memory(state, reserve):
        keys = ('total_bytes', 'conservative_reserve_bytes', 'pressure_level',
                'swap_used_bytes', 'compressed_bytes')
        if (not isinstance(state, dict) or any(type(state.get(k)) is not int for k in keys)
                or state['total_bytes'] <= 0 or state['pressure_level'] != 1
                or not reserve <= state['conservative_reserve_bytes'] <= state['total_bytes']
                or not 0 <= state['compressed_bytes'] <= state['total_bytes'] or state['swap_used_bytes'] < 0):
            raise ValueError('Invalid Metal unified-memory reserve evidence')
    initial = terminal.get('initial_memory')
    memory(initial, policy['entry_reserve_bytes'])
    samples = terminal.get('samples')
    if not isinstance(samples, list) or not 1 <= len(samples) <= 256:
        raise ValueError('Missing or unbounded Metal samples')
    previous, over_since = -1.0, None
    for sample in samples:
        memory(sample, policy['runtime_reserve_bytes'])
        elapsed, cpu = sample.get('elapsed_seconds'), sample.get('sampled_cpu_cores')
        if (type(elapsed) not in (int, float) or not math.isfinite(elapsed)
                or not previous <= elapsed < policy['deadline_seconds'] or elapsed < 0
                or type(cpu) not in (int, float) or not math.isfinite(cpu) or cpu < 0
                or type(sample.get('rss_bytes')) is not int or not 0 <= sample['rss_bytes'] <= policy['rss_bytes']
                or sample['total_bytes'] != initial['total_bytes']
                or sample['swap_used_bytes'] > initial['swap_used_bytes'] + policy['swap_growth_bytes']
                or sample['compressed_bytes'] > initial['compressed_bytes'] + policy['compressed_growth_bytes']):
            raise ValueError('Metal sample violates declared policy')
        over_since = (elapsed if over_since is None else over_since) if cpu > 3 else None
        if over_since is not None and elapsed - over_since >= 4:
            raise ValueError('Metal sustained CPU sample violates policy')
        previous = elapsed


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
    parser.add_argument('--vendor', choices=('cuda', 'hip', 'metal'), required=True)
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
    lines = log_raw.decode('utf-8', errors='strict').splitlines()
    terminal = json.loads(next(line for line in reversed(lines) if line.strip()))
    validate_guard_terminal(terminal, args.vendor, args.exit_code, args.job_kind)
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
