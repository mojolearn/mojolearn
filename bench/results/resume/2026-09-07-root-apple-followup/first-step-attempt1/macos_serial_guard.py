#!/usr/bin/env python3
"""Root-only, fail-closed Darwin supervisor for a separately authorized tiny job.

No hard CPU affinity or per-process Metal VRAM accounting is available here.
Two-thread environment limits are requests, not a two-core guarantee. Sampled
process-group CPU time/RSS and system unified-memory pressure bound the run.
This does not exclude unrelated applications from the GPU; locks coordinate
only cooperating mojolearn jobs. Root must review before any local execution.
"""
import argparse
import contextlib
import fcntl
import json
import math
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import tempfile
import time
import threading

GIB = 2**30
LOCKS = ('/tmp/mojolearn-root-gpu-job.lock', '/tmp/mojolearn-nvidia-root-job.lock',
         '/tmp/mojolearn-macos-root-job.lock', '/tmp/cbsym-build.lock')
INTERVAL = 2.0


def telemetry(command):
    """One bounded read-only OS query, with bounded retained output."""
    with tempfile.TemporaryFile() as output:
        subprocess.run(command, stdout=output, stderr=subprocess.DEVNULL,
                       timeout=1.5, check=True, env={'PATH': '/usr/bin:/bin:/usr/sbin:/sbin', 'LC_ALL': 'C'})
        output.seek(0)
        raw = output.read(1024 * 1024 + 1)
    if len(raw) > 1024 * 1024:
        raise RuntimeError('OS telemetry exceeds one MiB bound')
    return raw.decode('ascii')


def memory_state():
    system = telemetry(['/usr/sbin/sysctl', 'hw.memsize',
                        'kern.memorystatus_vm_pressure_level', 'vm.swapusage'])
    total = re.search(r'^hw.memsize:\s*([0-9]+)$', system, re.M)
    pressure = re.search(r'^kern.memorystatus_vm_pressure_level:\s*([0-9]+)$', system, re.M)
    swap = re.search(r'^vm.swapusage:.*?used\s*=\s*([0-9.]+)M', system, re.M)
    vm = telemetry(['/usr/bin/vm_stat'])
    page = re.search(r'page size of ([0-9]+) bytes', vm)
    def pages(label):
        match = re.search(r'^' + re.escape(label) + r':\s*([0-9]+)\.', vm, re.M)
        if not match:
            raise RuntimeError('Missing VM counter: ' + label)
        return int(match.group(1))
    if not all((total, pressure, swap, page)):
        raise RuntimeError('Required macOS memory/pressure/swap telemetry unavailable')
    total, page_size = int(total.group(1)), int(page.group(1))
    # Conservative reserve: no claim that inactive/compressed memory is free.
    reserve = (pages('Pages free') + pages('Pages speculative')) * page_size
    compressed = pages('Pages occupied by compressor') * page_size
    swap_used = float(swap.group(1)) * 2**20
    if (total <= 0 or page_size not in (4096, 16384) or not 0 <= reserve <= total
            or not 0 <= compressed <= total or not math.isfinite(swap_used) or swap_used < 0):
        raise RuntimeError('Inconsistent macOS memory telemetry')
    return dict(total_bytes=total, conservative_reserve_bytes=reserve,
                pressure_level=int(pressure.group(1)), swap_used_bytes=int(swap_used),
                compressed_bytes=compressed)


def cpu_seconds(value):
    days = 0
    if '-' in value:
        day, value = value.split('-', 1)
        days = int(day)
    pieces = value.split(':')
    if len(pieces) not in (2, 3):
        raise RuntimeError('Unsupported ps CPU-time format')
    seconds = float(pieces[-1]) + 60 * int(pieces[-2])
    if len(pieces) == 3:
        seconds += 3600 * int(pieces[0])
    seconds += 86400 * days
    if not math.isfinite(seconds) or seconds < 0:
        raise RuntimeError('Invalid ps CPU time')
    return seconds


def process_table():
    # lstart distinguishes observed PID lifetimes at ps's one-second precision.
    # This is not a kernel process handle; same-second PID reuse remains a gap.
    raw = telemetry(['/bin/ps', '-axo', 'pid=,ppid=,pgid=,rss=,time=,stat=,lstart='])
    table = {}
    for line in raw.splitlines():
        fields = line.split()
        if len(fields) != 11:
            raise RuntimeError('Incomplete ps process telemetry')
        pid, parent, pgid, size = map(int, fields[:4])
        if pid <= 0 or size < 0 or pid in table:
            raise RuntimeError('Invalid process telemetry')
        table[pid] = dict(parent=parent, group=pgid, rss=size * 1024,
                          cpu=cpu_seconds(fields[4]), zombie=fields[5].startswith('Z'),
                          identity=' '.join(fields[6:]))
    return table


class Descendants:
    def __init__(self, group):
        self.group = group
        self.known = {}
        self.mutex = threading.Lock()

    def observe(self, table):
        with self.mutex:
            # Remember identities after reparenting or a process-group change.
            found = {pid for pid, row in table.items()
                     if self.known.get(pid) == row['identity'] or row['group'] == self.group}
            while True:
                more = {pid for pid, row in table.items() if row['parent'] in found}
                if more <= found:
                    break
                found |= more
            for pid in found:
                self.known[pid] = table[pid]['identity']
            return {pid: table[pid] for pid in found if not table[pid]['zombie']}

    def signal(self, signum):
        # Signal the original group immediately, independently of telemetry.
        try:
            os.killpg(self.group, signum)
        except ProcessLookupError:
            pass
        # Revalidate observed escaped PIDs before signaling; never kill a PID
        # solely because it once belonged to the child tree.
        live = self.observe(process_table())
        for pid in live:
            try:
                os.kill(pid, signum)
            except ProcessLookupError:
                pass


class WallWatchdog:
    """Separate timer thread; blocked telemetry cannot defer group termination."""
    def __init__(self, descendants, deadline):
        self.descendants, self.deadline = descendants, deadline
        self.cancel = threading.Event()
        self.expired = threading.Event()
        self.errors = []
        self.thread = threading.Thread(target=self.watch, daemon=True)

    def watch(self):
        if self.cancel.wait(max(0, self.deadline - time.monotonic())):
            return
        self.expired.set()
        # Original group TERM/KILL does not wait for any telemetry query.
        for signum in (signal.SIGTERM, signal.SIGKILL):
            try:
                os.killpg(self.descendants.group, signum)
            except ProcessLookupError:
                pass
            except Exception as exc:
                self.errors.append(repr(exc))
            if signum == signal.SIGTERM:
                time.sleep(2)
        try:
            # Also kill escaped observed children, after current identity check.
            self.descendants.signal(signal.SIGKILL)
        except Exception as exc:
            self.errors.append(repr(exc))

    def stop(self):
        self.cancel.set()
        self.thread.join()


def cleanup(descendants, proc):
    errors = []
    for signum in (signal.SIGTERM, signal.SIGKILL):
        try:
            descendants.signal(signum)
        except Exception as exc:
            errors.append(repr(exc))
        if signum == signal.SIGTERM:
            time.sleep(2)
    # Do not release coordination locks while known live children or unknown
    # cleanup state remain. Payload is killed; supervisory quarantine may need
    # operator intervention if OS telemetry is permanently unavailable.
    quarantined = False
    while True:
        proc.poll()  # reap leader when possible
        try:
            live = descendants.observe(process_table())
            if proc.poll() is not None and not live:
                return dict(verified=True, quarantined=quarantined, errors=errors)
            descendants.signal(signal.SIGKILL)
        except Exception as exc:
            errors.append(repr(exc))
            errors = errors[-16:]
        if not quarantined:
            print('CLEANUP_UNVERIFIED: holding all locks until observed children exit and telemetry recovers',
                  file=sys.stderr, flush=True)
        quarantined = True
        time.sleep(2)


def run(args):
    if sys.platform != 'darwin':
        raise RuntimeError('This supervisor requires Darwin')
    if not args.command or not 1 <= args.seconds <= 180 or not 1 <= args.rss_gib <= 4:
        raise ValueError('Tiny job requires deadline 1..180 seconds and RSS cap 1..4 GiB')
    report = None
    with contextlib.ExitStack() as stack:
        for name in LOCKS:
            lock = stack.enter_context(open(name, 'a'))
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if args.report:
            report = stack.enter_context(args.report.open('x'))
        initial = memory_state()
        if initial['pressure_level'] != 1 or initial['conservative_reserve_bytes'] < 4 * GIB:
            raise RuntimeError('Require normal memory pressure and at least four GiB free/speculative reserve')
        process_table()  # Confirm process counters work before launch.
        env = dict(os.environ)
        for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
                    'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS', 'NUMBA_NUM_THREADS',
                    'MAX_JOBS', 'CMAKE_BUILD_PARALLEL_LEVEL', 'MOJOLEARN_COMPILE_JOBS',
                    'MOJOLEARN_CPU_THREADS'):
            env[key] = '2'
        # Set the wrapper bypass only after actually acquiring cbsym-build.lock;
        # an inherited mark never causes this guard to skip lock acquisition.
        env['MOJOLEARN_BUILD_LOCK_HELD'] = '1'
        proc, reason, error = None, None, None
        descendants, watchdog, cleanup_result = None, None, None
        samples, previous = [], {}
        over_cpu_since = None
        signals = (signal.SIGTERM, signal.SIGHUP, signal.SIGINT)
        previous_handlers = {s: signal.getsignal(s) for s in signals}
        def cancelled(signum, frame):
            raise InterruptedError('Supervisor received signal ' + str(signum))
        try:
            for signum in signals:
                signal.signal(signum, cancelled)
            started = last = time.monotonic()
            proc = subprocess.Popen(args.command, env=env, start_new_session=True)
            descendants = Descendants(proc.pid)
            watchdog = WallWatchdog(descendants, started + args.seconds)
            watchdog.thread.start()
            while proc.poll() is None:
                state = memory_state()
                live = descendants.observe(process_table())
                rss = sum(row['rss'] for row in live.values())
                current = {(pid, row['identity']): row['cpu'] for pid, row in live.items()}
                now = time.monotonic()
                if not current and proc.poll() is None:
                    raise RuntimeError('Live process group absent from safety telemetry')
                elapsed = now - last
                used = sum(max(0.0, value - previous.get(pid, 0.0)) for pid, value in current.items())
                cores = used / elapsed if elapsed > 0 else 0.0
                previous, last = current, now
                samples.append(dict(elapsed_seconds=now - started, rss_bytes=rss,
                                    sampled_cpu_cores=cores, **state))
                if cores > 3:
                    if over_cpu_since is None:
                        over_cpu_since = now
                else:
                    over_cpu_since = None
                if watchdog.expired.is_set() or now - started >= args.seconds:
                    reason = 'deadline exceeded'
                elif rss > args.rss_gib * GIB:
                    reason = 'process-group RSS cap exceeded'
                elif state['total_bytes'] != initial['total_bytes'] or state['pressure_level'] != 1:
                    reason = 'memory pressure or inconsistent host memory'
                elif state['conservative_reserve_bytes'] < 2 * GIB:
                    reason = 'unified-memory reserve below two GiB'
                elif state['swap_used_bytes'] > initial['swap_used_bytes'] + 128 * 2**20:
                    reason = 'swap usage grew by more than 128 MiB'
                elif state['compressed_bytes'] > initial['compressed_bytes'] + 256 * 2**20:
                    reason = 'compressed memory grew by more than 256 MiB'
                elif over_cpu_since is not None and now - over_cpu_since >= 4:
                    reason = 'sampled CPU usage above three cores sustained for four seconds'
                if reason:
                    break
                time.sleep(min(INTERVAL, max(0, args.seconds - (now - started))))
        except Exception as exc:
            reason, error = 'safety monitoring or launch failed', repr(exc)
        finally:
            for signum in signals:
                signal.signal(signum, signal.SIG_IGN)
            try:
                if proc is not None:
                    if descendants is None:
                        descendants = Descendants(proc.pid)
                    try:
                        lingering = descendants.observe(process_table())
                        if proc.poll() is not None and lingering and reason is None:
                            reason = 'leader exited with unfinished descendants'
                    except Exception as exc:
                        reason, error = 'cleanup telemetry failed', repr(exc)
                    cleanup_result = cleanup(descendants, proc)
                    if (cleanup_result['quarantined'] or cleanup_result['errors']) and reason is None:
                        reason = 'cleanup required quarantine or encountered telemetry errors'
                if watchdog is not None:
                    watchdog.stop()
                    if watchdog.expired.is_set():
                        reason = reason or 'independent wall deadline exceeded'
                    if watchdog.errors:
                        reason = reason or 'wall watchdog telemetry failed'
            finally:
                for signum, handler in previous_handlers.items():
                    signal.signal(signum, handler)
        result = dict(guard='macos-root-serial-v1', reason=reason, error=error,
                      returncode=proc.returncode if proc is not None else None,
                      thread_limit=2, cpu_affinity=None, cpu_enforcement='sampled, not hard affinity',
                      gpu_memory_accounting='system unified-memory reserve/pressure; no per-process Metal VRAM counter',
                      initial_memory=initial, samples=samples, cleanup=cleanup_result,
                      policy_limits=dict(deadline_seconds=args.seconds, rss_bytes=args.rss_gib * GIB,
                          entry_reserve_bytes=4 * GIB, runtime_reserve_bytes=2 * GIB,
                          normal_pressure_level=1, swap_growth_bytes=128 * 2**20,
                          compressed_growth_bytes=256 * 2**20, sampled_cpu_limit=3,
                          cpu_grace_seconds=4, sample_interval_seconds=INTERVAL),
                      watchdog_expired=watchdog.expired.is_set() if watchdog is not None else None,
                      watchdog_errors=watchdog.errors if watchdog is not None else [],
                      descendant_accounting='observed ancestry and PGID, ps start-time identities; sampling gaps remain')
        if report:
            json.dump(result, report, sort_keys=True, indent=2)
            report.write('\n')
            report.flush()
            os.fsync(report.fileno())
        print(json.dumps(result), flush=True)
        return 124 if reason else proc.returncode


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seconds', type=int, default=60)
    parser.add_argument('--rss-gib', type=int, default=2)
    parser.add_argument('--report', type=Path, help='optional new exclusive JSON report')
    parser.add_argument('command', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if args.command[:1] == ['--']:
        args.command.pop(0)
    raise SystemExit(run(args))
