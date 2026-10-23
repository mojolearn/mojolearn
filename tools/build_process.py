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


def stop_group(pid, grace=0.2):
    """The leader may already have exited while compiler children remain."""
    previous = {}
    if threading.current_thread() is threading.main_thread():
        for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
            previous[sig] = signal.signal(sig, signal.SIG_IGN)
    try:
        try:
            os.killpg(pid, signal.SIGTERM)
        except ProcessLookupError:
            return
        time.sleep(grace)
        try:
            os.killpg(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
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
    if threading.current_thread() is not threading.main_thread():
        yield
        return
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
                    result.update(status='timeout', exit_code=124)
                    break
                cpu, rss, available = probe(proc.pid, seen)
                result['peak_rss_bytes'] = max(result['peak_rss_bytes'], rss)
                prior = result['min_available_bytes']
                result['min_available_bytes'] = available if prior is None else min(prior, available)
                if cpu > last_cpu:
                    last_cpu, last_move = cpu, now
                if rss_bytes is not None and rss > rss_bytes:
                    result.update(status='rss-limit', exit_code=124)
                    break
                if available < min_available_bytes:
                    result.update(status='memory-pressure', exit_code=124)
                    break
                if stall_seconds and now-last_move >= stall_seconds:
                    result.update(status='stalled', exit_code=124)
                    break
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
                stop_group(proc.pid)
                proc.wait()
                with _LOCK:
                    _ACTIVE.discard(proc.pid)
                result['process_returncode'] = proc.returncode
            result.update(complete=True, elapsed_seconds=time.monotonic()-start)
            save()
    return result
