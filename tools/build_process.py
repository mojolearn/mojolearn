"""Bound an active build process and its descendants; no background service."""
from contextlib import contextmanager
import json
import math
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import threading
import time

_ACTIVE = set()
_LOCK = threading.Lock()
_CANCELLED = threading.Event()
_SCOPE_ACTIVE = False


def cancelled():
    """Let an orchestrator refuse cache publication after cancellation."""
    return _CANCELLED.is_set()


def _signal_group(pid, sig):
    """False when nothing is left to signal.

    ESRCH: the group is gone. EPERM: on macOS killpg fails with EPERM once
    the group holds only zombies (the leader is reaped, a child not yet), and
    on any platform for a group we no longer own. Neither is a reason to
    replace the build's real verdict (2026-10-24, M4 development build: the
    second killpg after an RSS-limit SIGTERM raised EPERM, and the group's
    recorded reason became "[Errno 1] Operation not permitted")."""
    try:
        os.killpg(pid, sig)
        return True
    except (ProcessLookupError, PermissionError):
        return False


def stop_group(pid, grace=0.2):
    """The leader may already have exited while compiler children remain."""
    previous = {}
    if threading.current_thread() is threading.main_thread():
        for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
            previous[sig] = signal.signal(sig, signal.SIG_IGN)
    try:
        if not _signal_group(pid, signal.SIGTERM):
            return
        time.sleep(grace)
        _signal_group(pid, signal.SIGKILL)
    finally:
        for sig, handler in previous.items():
            signal.signal(sig, handler)


def cancel_active():
    _CANCELLED.set()
    with _LOCK:
        groups = list(_ACTIVE)
    for pid in groups:
        stop_group(pid)


@contextmanager
def cancellation_scope():
    """The orchestrating thread also cancels builds executing in worker threads."""
    global _SCOPE_ACTIVE
    if threading.current_thread() is not threading.main_thread() or _SCOPE_ACTIVE:
        yield
        return
    _SCOPE_ACTIVE = True
    _CANCELLED.clear()
    previous = {}

    def cancelled(signum, frame):
        cancel_active()
        raise KeyboardInterrupt('build controller received signal '+str(signum))

    try:
        for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
            previous[sig] = signal.signal(sig, cancelled)
        yield
    finally:
        for sig, handler in previous.items():
            signal.signal(sig, handler)
        _SCOPE_ACTIVE = False


def available_memory():
    if sys.platform == 'linux':
        from build_sizing import box_resources
        return box_resources()[1]
    if sys.platform == 'darwin':
        raw = subprocess.check_output(['vm_stat'], text=True, timeout=10)
        page = int(re.search(r'page size of (\d+) bytes', raw).group(1))
        values = dict(re.findall(r'^([^:]+):\s+(\d+)\.', raw, re.M))
        return page*(int(values.get('Pages free', 0))+int(values.get('Pages inactive', 0)))
    raise RuntimeError('build resource guard requires Linux or macOS')


def _cpu_seconds(raw):
    days, clock = raw.split('-', 1) if '-' in raw else ('0', raw)
    value = 0.0
    for part in clock.split(':'):
        value = 60*value+float(part)
    return 86400*int(days)+value


def sample_group(group, seen):
    # Retain each child's accumulated CPU after it exits. Summing only live
    # children can misclassify a busy next compiler phase as a CPU stall.
    raw = subprocess.check_output(['ps', '-axo', 'pid=,pgid=,time=,rss='], text=True, timeout=10)
    rss = 0
    for line in raw.splitlines():
        pid, pgid, cpu, kib = line.split()
        if int(pgid) == group:
            seen[int(pid)] = max(seen.get(int(pid), 0), _cpu_seconds(cpu))
            rss += int(kib)*1024
    return sum(seen.values()), rss, available_memory()


def run(argv, *, cwd, env, log, seconds, stall_seconds=0, rss_bytes=None,
        min_available_bytes=0, poll=1.0, probe=sample_group):
    """Return a retained result after success, failure, interruption or limits."""
    if (not argv or not math.isfinite(seconds) or seconds <= 0
            or not math.isfinite(stall_seconds) or stall_seconds < 0
            or not math.isfinite(poll) or poll <= 0
            or (rss_bytes is not None and (not math.isfinite(rss_bytes) or rss_bytes <= 0))
            or not math.isfinite(min_available_bytes) or min_available_bytes < 0):
        raise ValueError('build limits must be finite and positive; stall/floor may be zero')
    path = Path(log)
    path.parent.mkdir(parents=True, exist_ok=True)
    receipt = path.with_suffix(path.suffix+'.execution.json')
    result = dict(command=list(map(str, argv)), cwd=str(cwd), complete=False,
                  status='starting', exit_code=None, deadline_seconds=seconds,
                  stall_seconds=stall_seconds, rss_limit_bytes=rss_bytes,
                  peak_rss_bytes=0, min_available_bytes=None)

    def save():
        temporary = receipt.with_suffix(receipt.suffix+'.tmp')
        temporary.write_text(json.dumps(result, indent=2)+'\n')
        os.replace(temporary, receipt)

    save()
    proc = None
    start = time.monotonic()
    last_move, last_cpu, seen = start, 0.0, {}
    last_save = start
    gib = 2**30
    with path.open('w') as stream, cancellation_scope():
        try:
            if _CANCELLED.is_set():
                raise KeyboardInterrupt('build cancelled before launch')
            proc = subprocess.Popen(argv, cwd=cwd, env=env, stdout=stream,
                                    stderr=subprocess.STDOUT, start_new_session=True)
            with _LOCK:
                _ACTIVE.add(proc.pid)
            result.update(status='running', pid=proc.pid)
            save()
            while proc.poll() is None:
                now = time.monotonic()
                if _CANCELLED.is_set():
                    result.update(status='interrupted', exit_code=130)
                    break
                if now-start >= seconds:
                    result.update(status='timeout', exit_code=124,
                                  reason='deadline %gs reached; peak rss %d bytes' % (seconds, result['peak_rss_bytes']))
                    break
                cpu, rss, available = probe(proc.pid, seen)
                result['peak_rss_bytes'] = max(result['peak_rss_bytes'], rss)
                prior = result['min_available_bytes']
                result['min_available_bytes'] = available if prior is None else min(prior, available)
                if cpu > last_cpu:
                    last_cpu, last_move = cpu, now
                if rss_bytes is not None and rss > rss_bytes:
                    result.update(status='rss-limit', exit_code=124,
                                  reason='rss limit %d bytes (%.2f GiB) exceeded: process group at %d bytes (%.2f GiB)'
                                  % (rss_bytes, rss_bytes/gib, rss, rss/gib))
                    break
                if available < min_available_bytes:
                    result.update(status='memory-pressure', exit_code=124,
                                  reason='available memory %d bytes fell below the %d-byte floor; peak rss %d bytes'
                                  % (available, min_available_bytes, result['peak_rss_bytes']))
                    break
                if stall_seconds and now-last_move >= stall_seconds:
                    result.update(status='stalled', exit_code=124,
                                  reason='no CPU progress for %gs; peak rss %d bytes'
                                  % (stall_seconds, result['peak_rss_bytes']))
                    break
                if now-last_save >= 10:
                    # Peaks stay visible even if this controller dies.
                    save()
                    last_save = now
                time.sleep(min(poll, max(0, seconds-(time.monotonic()-start))))
            if result['exit_code'] is None:
                result.update(exit_code=proc.returncode,
                              status='passed' if proc.returncode == 0 else 'failed')
        except KeyboardInterrupt as error:
            result.update(status='interrupted', exit_code=130, error=str(error))
        except Exception as error:
            result.update(status='controller-error', exit_code=125,
                          error=type(error).__name__+': '+str(error))
        finally:
            if proc is not None:
                try:
                    stop_group(proc.pid)
                except OSError as error:
                    # Stopping must never overwrite the verdict it enforces.
                    result['stop_error'] = type(error).__name__+': '+str(error)
                proc.wait()
                with _LOCK:
                    _ACTIVE.discard(proc.pid)
                result['process_returncode'] = proc.returncode
            result.update(complete=True, elapsed_seconds=time.monotonic()-start)
            save()
    return result
