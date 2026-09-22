#!/usr/bin/env python3
"""Run one root-owned job on a CPU-ONLY Linux build box with bounded time,
CPU and memory (the release CPU build route, tools/release_linux_build.sh).

The same contract as tools/nvidia_serial_guard.py (one job at a time under a
host lock, taskset to the first N allowed cores, a process-group RSS cap, a
deadline, thread-count variables pinned to two, the whole process group
stopped when the leader exits) with the GPU half removed: there is no device
on this box, and the guard REFUSES if one is visible, because a build that
can see a GPU is not the build this route claims (a device read could leak
into a binary and nothing would say so). The RSS cap is wider than the GPU
guards' 16 GiB because a CPU build box runs MOJOLEARN_BUILD_JOBS extension
builds at once; the release route sizes it at 16 GiB per job.
"""
import argparse
import fcntl
import glob
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from nvidia_serial_guard import memory, stop_group


def visible_devices():
    """Device nodes a GPU build could read. Empty on a CPU build box."""
    found = []
    for pattern in ('/dev/nvidia[0-9]*', '/dev/nvidiactl', '/dev/kfd', '/dev/dri/renderD*'):
        found.extend(glob.glob(pattern))
    return sorted(found)


def run(args):
    if sys.platform != 'linux' or not Path('/proc').is_dir():
        raise RuntimeError('Refusing local work: this guard requires the Linux build box')
    if not args.command or args.seconds < 1 or not 1 <= args.rss_gib <= 512:
        raise ValueError('Require command, positive deadline and RSS cap of 1..512 GiB')
    if not 1 <= args.cores <= 128:
        raise ValueError('Require a CPU core count of 1..128')
    devices = visible_devices()
    if devices:
        raise RuntimeError('A GPU device is visible on the CPU build box: ' + ', '.join(devices))
    lock = open('/tmp/mojolearn-cpu-build-box-job.lock', 'a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    _, available = memory(-1)
    if available < 4 * 2**30:
        raise RuntimeError('Less than 4 GiB available host memory')
    env = dict(os.environ)
    for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
                'NUMEXPR_NUM_THREADS', 'NUMBA_NUM_THREADS', 'MAX_JOBS', 'CMAKE_BUILD_PARALLEL_LEVEL',
                'MOJOLEARN_CPU_THREADS'):
        env[key] = '2'
    cores = ','.join(map(str, sorted(os.sched_getaffinity(0))[:args.cores]))

    def cancelled(signum, frame):
        raise InterruptedError('CPU build guard received signal ' + str(signum))
    for signum in (signal.SIGTERM, signal.SIGHUP, signal.SIGINT):
        signal.signal(signum, cancelled)
    proc = subprocess.Popen(['taskset', '-c', cores, *args.command],
                            env=env, start_new_session=True)
    started = time.monotonic()
    reason = None
    peak = 0
    try:
        while proc.poll() is None:
            rss, available = memory(proc.pid)
            peak = max(peak, rss)
            if time.monotonic() - started > args.seconds:
                reason = 'deadline exceeded'
            elif rss > args.rss_gib * 2**30:
                reason = 'process-group RSS cap exceeded'
            elif available < 2 * 2**30:
                reason = 'available host memory below 2 GiB'
            if reason:
                break
            time.sleep(1)
    finally:
        for signum in (signal.SIGTERM, signal.SIGHUP, signal.SIGINT):
            signal.signal(signum, signal.SIG_IGN)
        stop_group(proc.pid)
        proc.wait()
        lock.close()
    print(json.dumps({'guard': 'cpu-build-box-v1', 'reason': reason, 'cpu_affinity': cores,
                      'thread_limit': 2, 'rss_cap_gib': args.rss_gib,
                      'peak_group_rss_gib': round(peak / 2**30, 2),
                      'seconds': round(time.monotonic() - started, 1),
                      'returncode': proc.returncode}), flush=True)
    return 124 if reason else proc.returncode


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seconds', type=int, default=600)
    parser.add_argument('--rss-gib', type=int, default=12)
    parser.add_argument('--cores', type=int, default=2)
    parser.add_argument('command', nargs=argparse.REMAINDER)
    arguments = parser.parse_args()
    if arguments.command[:1] == ['--']:
        arguments.command.pop(0)
    sys.exit(run(arguments))
