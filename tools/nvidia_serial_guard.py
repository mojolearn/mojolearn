#!/usr/bin/env python3
"""Run one root-owned NVIDIA job with bounded time, CPU and memory.

This guard performs no compilation or model work itself. Linux/RunPod only.
The process-group RSS limit includes compiler and benchmark worker children;
the GPU limit includes all contexts on the selected, otherwise idle GPU.
"""
import argparse
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time


def gpu_memory():
    result = subprocess.run(
        ['nvidia-smi', '--query-gpu=memory.used,memory.total',
         '--format=csv,noheader,nounits'], capture_output=True, text=True,
        check=True, timeout=5)
    rows = [tuple(map(int, row.split(','))) for row in result.stdout.splitlines()]
    if len(rows) != 1:
        raise RuntimeError('Requires exactly one NVIDIA GPU')
    return rows[0]


def memory(group):
    rss = 0
    for entry in Path('/proc').iterdir():
        if not entry.name.isdigit():
            continue
        try:
            # comm can contain spaces and parentheses; fields after its last
            # closing parenthesis begin with state (field 3).
            fields = (entry / 'stat').read_text().rsplit(')', 1)[1].split()
            if int(fields[2]) == group:
                rss += int(fields[21]) * os.sysconf('SC_PAGE_SIZE')
        except (FileNotFoundError, ProcessLookupError, PermissionError):
            continue
    available = next(int(line.split()[1]) * 1024
                     for line in Path('/proc/meminfo').read_text().splitlines()
                     if line.startswith('MemAvailable:'))
    return rss, available


def stop_group(group):
    try:
        os.killpg(group, signal.SIGTERM)
    except ProcessLookupError:
        return
    time.sleep(2)
    try:
        os.killpg(group, signal.SIGKILL)
    except ProcessLookupError:
        pass


def run(args):
    if sys.platform != 'linux' or not Path('/proc').is_dir():
        raise RuntimeError('Refusing local work: this guard requires Linux NVIDIA')
    if not args.command or args.seconds < 1 or not 1 <= args.rss_gib <= 16:
        raise ValueError('Require command, positive deadline and RSS cap of 1..16 GiB')
    lock = open('/tmp/mojolearn-nvidia-root-job.lock', 'a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    used, total = gpu_memory()
    if used > 512:
        raise RuntimeError('NVIDIA GPU is occupied; refusing overlapping work')
    _, available = memory(-1)
    if available < 4 * 2**30:
        raise RuntimeError('Less than 4 GiB available host memory')
    env = dict(os.environ)
    for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
                'NUMEXPR_NUM_THREADS', 'NUMBA_NUM_THREADS', 'MAX_JOBS', 'CMAKE_BUILD_PARALLEL_LEVEL',
                'MOJOLEARN_CPU_THREADS'):
        env[key] = '2'
    cores = ','.join(map(str, sorted(os.sched_getaffinity(0))[:2]))
    def cancelled(signum, frame):
        raise InterruptedError('NVIDIA job guard received signal ' + str(signum))
    for signum in (signal.SIGTERM, signal.SIGHUP, signal.SIGINT):
        signal.signal(signum, cancelled)
    proc = subprocess.Popen(['taskset', '-c', cores, *args.command],
                            env=env, start_new_session=True)
    started = time.monotonic()
    reason = None
    try:
        while proc.poll() is None:
            rss, available = memory(proc.pid)
            used, total = gpu_memory()
            if time.monotonic() - started > args.seconds:
                reason = 'deadline exceeded'
            elif rss > args.rss_gib * 2**30:
                reason = 'process-group RSS cap exceeded'
            elif available < 2 * 2**30:
                reason = 'available host memory below 2 GiB'
            elif used > total * .85:
                reason = 'GPU memory exceeds 85 percent'
            if reason:
                break
            time.sleep(1)
    finally:
        for signum in (signal.SIGTERM, signal.SIGHUP, signal.SIGINT):
            signal.signal(signum, signal.SIG_IGN)
        # Also remove descendants when the leader exits or monitoring fails.
        stop_group(proc.pid)
        proc.wait()
        lock.close()
    print(json.dumps({'guard': 'nvidia-root-serial-v1', 'reason': reason,
                      'cpu_affinity': cores, 'thread_limit': 2,
                      'returncode': proc.returncode}), flush=True)
    return 124 if reason else proc.returncode


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seconds', type=int, default=600)
    parser.add_argument('--rss-gib', type=int, default=12)
    parser.add_argument('command', nargs=argparse.REMAINDER)
    arguments = parser.parse_args()
    if arguments.command[:1] == ['--']:
        arguments.command.pop(0)
    sys.exit(run(arguments))
