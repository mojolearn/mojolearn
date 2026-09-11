#!/usr/bin/env python3
"""Root-only serial remote Linux AMD jobs with bounded CPU, RSS and VRAM.

Reads AMD DRM card sysfs counters without loading HIP or running a model.
Requires a visible /dev/kfd and exactly one accessible AMD render device whose
character-device major/minor matches sysfs, exposing mem_info_vram_used/total.
Host sysfs may list unallocated GPUs; only visible /dev/dri render nodes count.
This is a device
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
import stat
import subprocess
import sys
import time

from nvidia_serial_guard import memory, stop_group

GIB = 2**30
LOCK_PATHS = ('/tmp/mojolearn-root-gpu-job.lock',
              '/tmp/mojolearn-nvidia-root-job.lock')
PCI_SUBSYSTEM = Path('/sys/bus/pci')
PLATFORM_SUBSYSTEM = Path('/sys/bus/platform')
PLATFORM_DEVICES = Path('/sys/devices/platform')
MAX_DEVICE_ANCESTORS = 16


def gpu_memory(device):
    """Return byte counters; missing/inconsistent sysfs fails closed."""
    device = Path(device)
    used = int((device / 'mem_info_vram_used').read_text().strip())
    total = int((device / 'mem_info_vram_total').read_text().strip())
    if total <= 0 or used < 0 or used > total:
        raise RuntimeError('Invalid AMD VRAM counters')
    return used, total


def render_number(node):
    info = node.stat()
    if not stat.S_ISCHR(info.st_mode) or not os.access(node, os.R_OK | os.W_OK):
        raise RuntimeError('AMD render node must be an accessible character device')
    return str(os.major(info.st_rdev)) + ':' + str(os.minor(info.st_rdev))


def pci_identity(device):
    """Resolve only the nearest PCI ancestor; never search siblings/other GPUs.

    A virtual render device can lack vendor at its immediate device target.
    Identity requires the kernel PCI subsystem symlink, not a path name that
    happens to resemble a PCI address. The traversal and retained vendor list
    are bounded. Missing/malformed/contradictory identity fails closed.
    """
    current = Path(device).resolve(strict=True)
    pci = PCI_SUBSYSTEM.resolve(strict=True)
    witnessed = []
    for _ in range(MAX_DEVICE_ANCESTORS):
        vendor_path = current / 'vendor'
        vendor = None
        if vendor_path.exists() or vendor_path.is_symlink():
            with vendor_path.open() as source:
                encoded_vendor = source.read(33)
            vendor = encoded_vendor.strip().lower()
            if len(encoded_vendor) > 32 or not re.fullmatch(r'0x[0-9a-f]{4}', vendor):
                raise RuntimeError('Visible render node has malformed vendor identity')
            witnessed.append(vendor)
        subsystem = current / 'subsystem'
        if subsystem.is_symlink() and subsystem.resolve(strict=True) == pci:
            if vendor is None:
                raise RuntimeError('Nearest PCI device lacks vendor identity')
            if any(value != vendor for value in witnessed):
                raise RuntimeError('Visible render node has contradictory vendor identity')
            return current, vendor
        if current.parent == current:
            break
        current = current.parent
    raise RuntimeError('Visible render node lacks bounded PCI vendor identity')


def is_observed_xcp_placeholder(device):
    """Recognize only DO run3's kernel platform XCP placeholders (0..6).

    These render nodes are not independent physical GPU memory witnesses.
    Linux registers them as platform devices, outside PCI ancestry. Never
    infer AMD identity from a generic platform node or from the name alone.
    Future numbering/topology or partition-counter arrangements require review.
    """
    device = Path(device).resolve(strict=True)
    if not re.fullmatch(r'amdgpu_xcp_[0-6]', device.name):
        return False
    if device.parent != PLATFORM_DEVICES.resolve(strict=True):
        return False
    subsystem = device / 'subsystem'
    if not subsystem.is_symlink() or subsystem.resolve(strict=True) != PLATFORM_SUBSYSTEM.resolve(strict=True):
        return False
    # Do not silently discard conflicting identity or actual memory counters.
    for name in ('vendor', 'device', 'mem_info_vram_used', 'mem_info_vram_total'):
        attribute = device / name
        if attribute.exists() or attribute.is_symlink():
            raise RuntimeError('XCP placeholder unexpectedly exposes identity or VRAM counters')
    return True


def amd_device(root=Path('/sys/class/drm'), nodes=Path('/dev/dri')):
    devices = set()
    for node in Path(nodes).iterdir():
        if not re.fullmatch(r'renderD[0-9]+', node.name):
            continue
        render = Path(root) / node.name
        if (render / 'dev').read_text().strip() != render_number(node):
            raise RuntimeError('AMD render node device number differs from sysfs')
        if is_observed_xcp_placeholder(render / 'device'):
            continue
        device, vendor = pci_identity(render / 'device')
        if vendor != '0x1002':
            continue
        gpu_memory(device)
        devices.add(device)
    if len(devices) != 1:
        raise RuntimeError('Requires exactly one visible AMD render GPU with sysfs VRAM counters')
    return next(iter(devices))


def run(args):
    """Guard one job; terminal telemetry contains bounded observed samples.

    initial_vram_bytes is the prelaunch admission reading; initial_rss_bytes
    is the first child-group sample (None if none was observed). Peaks are
    sampled maxima, not continuous high-water marks. Last values are before
    cleanup, not a claim that VRAM was released afterward. Monitor exceptions
    retain the existing raise/cleanup behavior rather than emitting success.
    """
    if sys.platform != 'linux' or not Path('/proc').is_dir():
        raise RuntimeError('Refusing local work: this guard requires remote Linux AMD')
    if not args.command or args.seconds < 1 or not 1 <= args.rss_gib <= 12:
        raise ValueError('Require command, positive deadline and RSS cap of 1..12 GiB')
    core_count = getattr(args, 'cores', 2)
    if not 1 <= core_count <= 64:
        raise ValueError('Require a CPU core count of 1..64')
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
        cores = ','.join(map(str, sorted(os.sched_getaffinity(0))[:core_count]))
        if not cores:
            raise RuntimeError('No permitted CPU cores')
        def cancelled(signum, frame):
            raise InterruptedError('AMD job guard received signal ' + str(signum))
        signals = (signal.SIGTERM, signal.SIGHUP, signal.SIGINT)
        previous = {signum: signal.getsignal(signum) for signum in signals}
        proc = None
        reason = None
        # Bounded diagnostic state only: reuse the admission reading and each
        # existing monitor sample. No extra device queries or post-cleanup
        # measurement. RSS is the child process group, VRAM the whole device.
        telemetry = dict(initial_vram_bytes=used, peak_vram_bytes=used,
                         last_vram_bytes=used, initial_rss_bytes=None,
                         peak_rss_bytes=None, last_rss_bytes=None,
                         initial_host_available_bytes=available,
                         last_host_available_bytes=available,
                         samples=0, last_elapsed_seconds=None,
                         crossing_sample=None)
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
                elapsed = time.monotonic() - started
                telemetry['samples'] += 1
                if telemetry['initial_rss_bytes'] is None:
                    telemetry['initial_rss_bytes'] = rss
                telemetry['peak_rss_bytes'] = max(telemetry['peak_rss_bytes'] or 0, rss)
                telemetry['last_rss_bytes'] = rss
                telemetry['peak_vram_bytes'] = max(telemetry['peak_vram_bytes'], used)
                telemetry['last_vram_bytes'] = used
                telemetry['last_host_available_bytes'] = available
                telemetry['last_elapsed_seconds'] = elapsed
                if elapsed > args.seconds:
                    reason = 'deadline exceeded'
                elif rss > args.rss_gib * GIB:
                    reason = 'process-group RSS cap exceeded'
                elif available < 2 * GIB:
                    reason = 'available host memory below 2 GiB'
                elif used > total * .85:
                    reason = 'GPU memory exceeds 85 percent'
                if reason:
                    telemetry['crossing_sample'] = dict(
                        reason=reason, elapsed_seconds=elapsed,
                        vram_bytes=used, rss_bytes=rss,
                        host_available_bytes=available)
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
                          'telemetry': telemetry,
                          'returncode': proc.returncode}), flush=True)
        return 124 if reason else proc.returncode


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seconds', type=int, default=600)
    parser.add_argument('--rss-gib', type=int, default=12)
    # DEVIATION 2501: parallel wheel builds. Default 2 keeps every other job
    # serial; the release build passes 2 x MOJOLEARN_BUILD_JOBS.
    parser.add_argument('--cores', type=int, default=2)
    parser.add_argument('command', nargs=argparse.REMAINDER)
    arguments = parser.parse_args()
    if arguments.command[:1] == ['--']:
        arguments.command.pop(0)
    sys.exit(run(arguments))
