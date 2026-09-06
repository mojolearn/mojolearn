#!/usr/bin/env python3
"""Root-only serial remote Linux AMD jobs with bounded CPU, RSS and VRAM.

Reads AMD DRM card sysfs counters without loading HIP or running a model.
Requires a visible /dev/kfd and exactly one distinct AMD PCI device exposing
mem_info_vram_used/total. Render-node aliases are ignored. This is a device
and memory witness, not proof that a particular GPU architecture supports HIP.

The legacy NVIDIA lock is also held, so an unchanged nvidia_serial_guard.py
cannot overlap on the same host. Locks do not coordinate different rentals.
Polling cannot prevent instantaneous allocation spikes or track descendants
that deliberately escape the job's process group. Use small workloads and an
independent rental watchdog; no job is launched outside the root thread.
"""
import argparse
import contextlib
import fcntl
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time

from nvidia_serial_guard import memory, stop_group

GIB = 2**30
LOCK_PATHS = ('/tmp/mojolearn-root-gpu-job.lock',
              '/tmp/mojolearn-nvidia-root-job.lock')


def gpu_memory(device):
    """Return byte counters; missing/inconsistent sysfs fails closed."""
    device = Path(device)
    used = int((device / 'mem_info_vram_used').read_text().strip())
    total = int((device / 'mem_info_vram_total').read_text().strip())
    if total <= 0 or used < 0 or used > total:
        raise RuntimeError('Invalid AMD VRAM counters')
    return used, total


def amd_device(root=Path('/sys/class/drm')):
    devices = set()
    for card in Path(root).iterdir():
        # card0-DP-1 and renderD128 are aliases, not additional devices.
        if not re.fullmatch(r'card[0-9]+', card.name):
            continue
        device = card / 'device'
        vendor = device / 'vendor'
        if not vendor.is_file():
            continue
        if vendor.read_text().strip().lower() != '0x1002':
            continue
        device = device.resolve(strict=True)
        gpu_memory(device)
        devices.add(device)
    if len(devices) != 1:
        raise RuntimeError('Requires exactly one AMD GPU with sysfs VRAM counters')
    return next(iter(devices))


def run(args):
    if sys.platform != 'linux' or not Path('/proc').is_dir():
        raise RuntimeError('Refusing local work: this guard requires remote Linux AMD')
    if not args.command or args.seconds < 1 or not 1 <= args.rss_gib <= 12:
        raise ValueError('Require command, positive deadline and RSS cap of 1..12 GiB')
    if not Path('/dev/kfd').exists():
        raise RuntimeError('AMD HIP requires a visible /dev/kfd device')
    with contextlib.ExitStack() as stack:
        for path in LOCK_PATHS:
            lock = stack.enter_context(open(path, 'a'))
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        device = amd_device()
        used, total = gpu_memory(device)
        if used > 512 * 2**20:
            raise RuntimeError('AMD GPU is occupied; refusing overlapping work')
        _, available = memory(-1)
        if available < 4 * GIB:
            raise RuntimeError('Less than 4 GiB available host memory')
        env = dict(os.environ)
        for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
                    'NUMEXPR_NUM_THREADS', 'NUMBA_NUM_THREADS', 'MAX_JOBS',
                    'CMAKE_BUILD_PARALLEL_LEVEL', 'MOJOLEARN_CPU_THREADS'):
            env[key] = '2'
        cores = ','.join(map(str, sorted(os.sched_getaffinity(0))[:2]))
        if not cores:
            raise RuntimeError('No permitted CPU cores')
        def cancelled(signum, frame):
            raise InterruptedError('AMD job guard received signal ' + str(signum))
        signals = (signal.SIGTERM, signal.SIGHUP, signal.SIGINT)
        previous = {signum: signal.getsignal(signum) for signum in signals}
        proc = None
        reason = None
        try:
            for signum in signals:
                signal.signal(signum, cancelled)
            proc = subprocess.Popen(['taskset', '-c', cores, *args.command],
                                    env=env, start_new_session=True)
            started = time.monotonic()
            while proc.poll() is None:
                rss, available = memory(proc.pid)
                used, observed_total = gpu_memory(device)
                if observed_total != total:
                    raise RuntimeError('AMD device VRAM total changed during job')
                if time.monotonic() - started > args.seconds:
                    reason = 'deadline exceeded'
                elif rss > args.rss_gib * GIB:
                    reason = 'process-group RSS cap exceeded'
                elif available < 2 * GIB:
                    reason = 'available host memory below 2 GiB'
                elif used > total * .85:
                    reason = 'GPU memory exceeds 85 percent'
                if reason:
                    break
                time.sleep(1)
        finally:
            for signum in signals:
                signal.signal(signum, signal.SIG_IGN)
            try:
                if proc is not None:
                    stop_group(proc.pid)
                    proc.wait()
            finally:
                for signum, handler in previous.items():
                    signal.signal(signum, handler)
        print(json.dumps({'guard': 'amd-root-serial-v1', 'reason': reason,
                          'cpu_affinity': cores, 'thread_limit': 2,
                          'device_sysfs': str(device), 'vram_total_bytes': total,
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
