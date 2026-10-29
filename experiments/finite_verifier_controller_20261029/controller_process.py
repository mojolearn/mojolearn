"""Run one owned producer process group with a finite wall-clock budget."""
import os
import signal
import subprocess
import time

def run_process(argv, *, cwd, log, env=None, timeout=None, kill_grace=2.0):
    if timeout is not None and timeout <= 0:
        raise ValueError('producer timeout must be positive')
    start=time.monotonic()
    timed_out=False
    with log.open('w') as stream:
        process=subprocess.Popen([str(x) for x in argv],cwd=cwd,env=env,stdout=stream,
                                 stderr=subprocess.STDOUT,start_new_session=True)
        grace=min(kill_grace,timeout/4) if timeout is not None else kill_grace
        try:
            process.wait(timeout=None if timeout is None else timeout-grace)
        except subprocess.TimeoutExpired:
            timed_out=True
            try:os.killpg(process.pid,signal.SIGTERM)
            except ProcessLookupError:pass
            try:process.wait(timeout=grace)
            except subprocess.TimeoutExpired:pass
            # The leader can exit while descendants ignore TERM; kill the entire
            # owned group even when wait() already returned for the leader.
            try:os.killpg(process.pid,signal.SIGKILL)
            except ProcessLookupError:pass
            process.wait()
    return dict(exit_code=124 if timed_out else process.returncode,
                process_exit_code=process.returncode,timed_out=timed_out,
                timeout_seconds=timeout,elapsed_seconds=time.monotonic()-start,
                process_group=process.pid)
