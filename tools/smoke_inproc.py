"""In-process parameter probes for the campaign smoke (tools/smoke_worker.py).

Andrew (2026-10-14): LOAD EVERYTHING ONCE, then run every recipe's check in one long-lived process.
The board (tools/bench_board.py) and the family drivers keep their own code paths; this module only
supplies the seams they switch on when they run inside the smoke worker:

- `in_process()`: true only in the smoke worker process itself (the variable holds its pid, so a
  subprocess that inherits the environment is never mistaken for it).
- `run_driver()`: the board's --params-only driver call as a function call (the driver's own main(argv)
  under the driver process's environment, output into the probe log), instead of a subprocess.
- `InProcessWorker`: a driver's per-arm worker, constructed by the driver's OWN ready stage
  (`probe_ready(argv)`, the same function the worker subprocess runs) under that worker's environment,
  instead of a subprocess. Parameter probe only: never a warm-up or a timed round.
- `scoped_env()`: one call's environment, applied exactly and restored afterwards, with a record of
  what it changed and which differences a process-wide import cannot honour.
- `ArmServers`: an arm whose worker needs another interpreter or another loader environment runs in
  a second long-lived process for that environment (loaded once, `smoke_worker.py arm-serve`).

Nothing here fits, warms up or times anything.
"""
import contextlib
import importlib.util
import json
import os
from pathlib import Path
import select
import signal
import subprocess
import sys
import time

TOOLS = Path(__file__).resolve().parent
ROOT = TOOLS.parent
ENV_FLAG = "MOJOLEARN_SMOKE_IN_PROCESS"

#: read once at library load (thread pools are sized at import); still applied to os.environ, recorded
LOAD_TIME_ENV = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS",
                 "VECLIB_MAXIMUM_THREADS", "BLIS_NUM_THREADS", "GOTO_NUM_THREADS", "NUMBA_NUM_THREADS",
                 "OMP_THREAD_LIMIT", "NUMEXPR_MAX_THREADS", "LOKY_MAX_CPU_COUNT", "MOJOLEARN_BENCH_THREADS",
                 "MOJOLEARN_CPU_THREADS")
#: read by the dynamic loader or the interpreter at process start: a difference needs another process
PROCESS_START_ENV = ("LD_LIBRARY_PATH", "LD_PRELOAD", "DYLD_LIBRARY_PATH", "DYLD_INSERT_LIBRARIES",
                     "DYLD_FALLBACK_LIBRARY_PATH", "PYTHONPATH", "PYTHONHOME", "PYTHONNOUSERSITE",
                     "PYTHONSAFEPATH", "PYTHONHASHSEED", "PYTHONOPTIMIZE")
#: interpreter-start settings that do not change what a probe reads back
PROCESS_START_HARMLESS = ("PYTHONUNBUFFERED", "PYTHONDONTWRITEBYTECODE", "PYTHONIOENCODING")


def in_process():
    """True only inside the smoke worker process (tools/smoke_worker.py serve)."""
    return os.environ.get(ENV_FLAG) == str(os.getpid())


def mark_process():
    os.environ[ENV_FLAG] = str(os.getpid())


_ONCE = {}


def once(key, fn):
    """fn() once per smoke worker process for `key` (box facts: the box fingerprint, the repo commit, a
    data file's sha256, the worker-observed GPU receipt). Exceptions are not cached."""
    if key not in _ONCE:
        _ONCE[key] = fn()
    return _ONCE[key]


def clear_once():
    _ONCE.clear()


# ---------------------------------------------------------------------------
# environment and output of one call
# ---------------------------------------------------------------------------

def env_difference(target, current=None):
    """How `target` differs from `current` (default os.environ), classified:
    {"set": {k: v}, "unset": [k], "load_time_only": [k], "process_start": [k]}."""
    current = dict(os.environ if current is None else current)
    target = {str(k): str(v) for k, v in (target or {}).items()}
    changed = {k: v for k, v in target.items() if current.get(k) != v}
    unset = sorted(k for k in current if k not in target)
    keys = set(changed) | set(unset)
    return dict(set={k: changed[k] for k in sorted(changed)}, unset=unset,
                load_time_only=sorted(k for k in keys if k in LOAD_TIME_ENV),
                process_start=sorted(k for k in keys if k in PROCESS_START_ENV and k != ENV_FLAG))


def _set_numeric_mode(mode):
    """mojolearn's in-process tier switch; (previous, None) or (None, error text)."""
    mj = sys.modules.get("mojolearn")
    if mj is None:
        return None, None
    try:
        return mj.set_numeric_mode(mode), None
    except Exception as exc:  # noqa: BLE001 - reported as the arm's error
        return None, repr(exc)[:400]


@contextlib.contextmanager
def scoped_env(target, record=None):
    """Run the block under exactly `target` as os.environ (restored afterwards, whatever the block added).
    MOJOLEARN_NUMERIC_MODE also goes through mojolearn.set_numeric_mode when mojolearn is loaded (its
    import read the variable once). `record` (a dict) receives what was applied."""
    if target is None:
        yield record
        return
    record = {} if record is None else record
    diff = env_difference(target)
    record.update(applied=sorted(diff["set"]), unset=diff["unset"], load_time_only=diff["load_time_only"],
                  process_start=diff["process_start"])
    saved = dict(os.environ)
    flag = os.environ.get(ENV_FLAG)
    want = str(target.get("MOJOLEARN_NUMERIC_MODE") or "identical").strip().lower()
    have = str(saved.get("MOJOLEARN_NUMERIC_MODE") or "identical").strip().lower()
    previous = None
    os.environ.clear()
    os.environ.update({str(k): str(v) for k, v in target.items()})
    if flag is not None:
        os.environ[ENV_FLAG] = flag
    try:
        mj = sys.modules.get("mojolearn")
        if mj is not None:
            try:
                current = mj._backend.default_mode()
            except Exception:  # noqa: BLE001
                current = have
            if current != want:
                previous, err = _set_numeric_mode(want)
                record["numeric_mode"] = dict(requested=want, previous=previous or current, via="set_numeric_mode",
                                              error=err)
                if err:
                    raise RuntimeError("mojolearn.set_numeric_mode(%r) failed: %s" % (want, err))
            else:
                record["numeric_mode"] = dict(requested=want, via="already the process default")
        else:
            record["numeric_mode"] = dict(requested=want, via="environment (mojolearn not loaded yet)")
        yield record
    finally:
        if previous is not None:
            _set_numeric_mode(previous)
        os.environ.clear()
        os.environ.update(saved)


@contextlib.contextmanager
def redirect_output(path, mode="a"):
    """fds 1 and 2 (C-level output too) and sys.stdout/sys.stderr into `path` for the block."""
    for s in (sys.stdout, sys.stderr):
        try:
            s.flush()
        except Exception:  # noqa: BLE001
            pass
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fh = open(path, mode + "b", buffering=0)
    saved = [os.dup(1), os.dup(2)]
    try:
        os.dup2(fh.fileno(), 1)
        os.dup2(fh.fileno(), 2)
        yield path
    finally:
        for s in (sys.stdout, sys.stderr):
            try:
                s.flush()
            except Exception:  # noqa: BLE001
                pass
        os.dup2(saved[0], 1)
        os.dup2(saved[1], 2)
        for fd in saved:
            os.close(fd)
        fh.close()


@contextlib.contextmanager
def working_directory(path):
    old = os.getcwd()
    os.chdir(str(path))
    try:
        yield
    finally:
        os.chdir(old)


# ---------------------------------------------------------------------------
# drivers as modules
# ---------------------------------------------------------------------------

_DRIVERS = {}


def driver_module(path):
    """A driver script as a module, loaded once per process (tools/ and bench/speed/ on sys.path, as when it
    runs as a script)."""
    path = os.path.abspath(str(path))
    if path not in _DRIVERS:
        here = os.path.dirname(path)
        if here not in sys.path:
            sys.path.insert(0, here)
        name = "smoke_driver_" + os.path.splitext(os.path.basename(path))[0].replace("-", "_")
        spec = importlib.util.spec_from_file_location(name, path)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[name] = mod
        try:
            spec.loader.exec_module(mod)
        except BaseException:
            sys.modules.pop(name, None)
            raise
        _DRIVERS[path] = mod
    return _DRIVERS[path]


def split_command(cmd):
    """(interpreter words, script path, argv) of `[python..., [-u], script.py, args...]`."""
    cmd = list(cmd)
    for i, word in enumerate(cmd):
        if word.endswith(".py"):
            interp = [w for w in cmd[:i] if w != "-u"]
            return interp, word, cmd[i + 1:]
    raise ValueError("no script in command: %r" % (cmd[:6],))


def same_interpreter(words):
    """True when `words` start the interpreter this process runs."""
    return len(words) == 1 and os.path.abspath(words[0]) == os.path.abspath(sys.executable)


def can_run_here(cmd, env):
    """True when `cmd` under `env` can run as a function call in this process: the same interpreter and no
    difference in the settings a process reads only at its start."""
    try:
        interp = split_command(cmd)[0]
    except ValueError:
        return False
    return same_interpreter(interp) and not env_difference(env)["process_start"]


def run_driver(cmd, env, log, cwd=ROOT):
    """The board's driver subprocess, as a function call: the driver's main(argv) under `env`, its output
    appended to `log`. Returns (returncode, environment record). A driver that needs another interpreter
    or loader environment raises (the board's subprocess path is the honest fallback there)."""
    interp, script, argv = split_command(cmd)
    record = {}
    diff = env_difference(env)
    if not same_interpreter(interp):
        raise RuntimeError("driver interpreter %r is not this process's %s" % (" ".join(interp), sys.executable))
    if diff["process_start"]:
        raise RuntimeError("driver environment differs in process-start settings %s" % ",".join(diff["process_start"]))
    mod = driver_module(script)
    with redirect_output(log), working_directory(cwd), scoped_env(env, record):
        try:
            rc = mod.main(argv)
        except SystemExit as exc:
            code = exc.code
            if code is not None and not isinstance(code, int):
                print(code, flush=True)
                code = 1
            rc = code or 0
        except Exception:  # noqa: BLE001 - a driver crash is its returncode, as in a subprocess
            import traceback
            traceback.print_exc()
            rc = 1
    record["process"] = "in-process (smoke worker pid %d)" % os.getpid()
    return (rc or 0), record


# ---------------------------------------------------------------------------
# a driver's per-arm worker, in process
# ---------------------------------------------------------------------------

class InProcessWorker:
    """Stands in for classical_two_datasets.Worker on the --params-only path inside the smoke worker. The
    first read() constructs the arm through the driver's own `probe_ready(argv)` (the worker subprocess's
    ready stage) under the arm's worker environment and returns its ready message (or its error/skipped
    message). The arm's log is the same per-arm log file the subprocess would have written."""

    def __init__(self, arm, cmd, env, log_path, cwd, driver=None):
        self.arm, self.cmd, self.env, self.cwd = arm, list(cmd), dict(env or {}), cwd
        self.log_path = log_path
        Path(log_path).parent.mkdir(parents=True, exist_ok=True)
        self.log = open(log_path, "a")
        self.alive = True
        self.status = "ok"
        self.error = None
        self.info = None
        self.noise = []
        self.msg = None
        self.environment = {}
        interp, script, argv = split_command(self.cmd)
        self.interp, self.script, self.argv = interp, driver or script, argv
        self.proc = None

    def _ready(self):
        diff = env_difference(self.env)
        if not same_interpreter(self.interp) or diff["process_start"]:
            # another interpreter or loader environment: its own long-lived process (loaded once)
            why = ("interpreter %s" % " ".join(self.interp) if not same_interpreter(self.interp)
                   else "process-start settings %s" % ",".join(diff["process_start"]))
            self.environment = dict(process="arm server (%s)" % why)
            msg = ARM_SERVERS.ready(self.interp, self.env, self.script, self.argv, self.log_path)
            self.environment.update(msg.pop("probe_environment", None) or {})
            return msg
        mod = driver_module(self.script)
        self.log.flush()
        with redirect_output(self.log_path), working_directory(self.cwd), scoped_env(self.env, self.environment):
            self.environment["process"] = "in-process (smoke worker pid %d)" % os.getpid()
            try:
                return mod.probe_ready(self.argv)
            except Exception as exc:  # noqa: BLE001 - the worker subprocess would have said so
                import traceback
                traceback.print_exc()
                return {"event": "error", "stage": "ready", "error": repr(exc)[:2000]}

    def read(self, seconds=None):
        if self.msg is not None or not self.alive:
            return None
        msg = self._ready()
        if isinstance(msg, dict):
            # the environment this arm's record was taken under rides in its info (the board's probe keeps it
            # as probe_process); the parameter record itself is untouched
            msg = dict(msg, pid=os.getpid())
            if isinstance(msg.get("info"), dict):
                msg["info"] = dict(msg["info"], probe_process=self.environment)
            else:
                msg["probe_process"] = self.environment
        self.msg = msg
        return msg

    def send(self, text):
        raise RuntimeError("in-process arm %s serves the parameter probe only (asked %r)" % (self.arm, text))

    def kill(self, status, error):
        self.alive = False
        self.status = status
        self.error = error

    def close(self):
        self.alive = False
        try:
            self.log.close()
        except Exception:  # noqa: BLE001
            pass


# ---------------------------------------------------------------------------
# long-lived processes for arms that need another environment
# ---------------------------------------------------------------------------

def read_json_line(proc, buf, deadline):
    """(object or None, rest) from proc.stdout before `deadline` (monotonic)."""
    fd = proc.stdout.fileno()
    while b"\n" not in buf:
        left = deadline - time.monotonic()
        if left <= 0:
            return None, buf
        ready, _, _ = select.select([fd], [], [], min(left, 1.0))
        if not ready:
            if proc.poll() is not None:
                return None, buf
            continue
        chunk = os.read(fd, 65536)
        if not chunk:
            return None, buf
        buf += chunk
    line, buf = buf.split(b"\n", 1)
    text = line.decode(errors="replace").strip()
    try:
        return (json.loads(text) if text.startswith("{") else {"event": "noise", "text": text[:300]}), buf
    except ValueError:
        return {"event": "noise", "text": text[:300]}, buf


class ArmServer:
    """`smoke_worker.py arm-serve` under one interpreter and loader environment: loads every driver and
    library once, then answers probe_ready requests one at a time."""

    def __init__(self, interp, env, load_timeout, log_dir):
        self.interp, self.env = list(interp), dict(env)
        self.load_timeout = load_timeout
        self.log_dir = Path(log_dir)
        self.proc, self.buf, self.busy = None, b"", False
        self.starts = []

    def start(self):
        self.log_dir.mkdir(parents=True, exist_ok=True)
        log = open(self.log_dir / ("arm-server-%d.log" % len(ARM_SERVERS.starts)), "ab")
        env = dict(self.env)
        env.pop(ENV_FLAG, None)
        t0 = time.time()
        self.proc = subprocess.Popen(self.interp + [str(TOOLS / "smoke_worker.py"), "arm-serve"], cwd=str(ROOT),
                                     env=env, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=log,
                                     start_new_session=True)
        log.close()
        self.buf = b""
        start = dict(interpreter=" ".join(self.interp), pid=self.proc.pid, started=t0)
        deadline = time.monotonic() + self.load_timeout
        while True:
            msg, self.buf = read_json_line(self.proc, self.buf, deadline)
            if msg is None or msg.get("event") == "ready":
                break
        start["wall_seconds"] = round(time.time() - t0, 2)
        if msg is None:
            start["problem"] = "no load report within %ss (rc %s)" % (self.load_timeout, self.proc.poll())
            self.stop()
        else:
            start["report"] = msg.get("report")
        self.starts.append(start)
        ARM_SERVERS.starts.append(start)
        return self.alive()

    def alive(self):
        return self.proc is not None and self.proc.poll() is None

    def stop(self):
        if self.proc is not None and self.proc.poll() is None:
            try:
                os.killpg(self.proc.pid, signal.SIGKILL)
            except OSError:
                pass
            try:
                self.proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                pass
        self.busy = False

    def ready(self, script, argv, env, log, timeout):
        if self.busy:          # an earlier load or request was cut off (its recipe timed out): start afresh
            self.stop()
        self.busy = True       # stays set if this recipe's timer cuts the load or the request off
        if not self.alive() and not self.start():
            self.busy = False
            return {"event": "error", "stage": "ready", "error": "arm server for %s did not load: %s" % (
                " ".join(self.interp), self.starts[-1].get("problem"))}
        req = dict(driver=str(script), argv=list(argv), env=dict(env), log=str(log))
        self.proc.stdin.write((json.dumps(req) + "\n").encode())
        self.proc.stdin.flush()
        deadline = time.monotonic() + timeout
        while True:
            msg, self.buf = read_json_line(self.proc, self.buf, deadline)
            if msg is None or msg.get("event") not in ("noise", "ready"):
                break
        self.busy = False
        if msg is None:
            self.stop()
            return {"event": "error", "stage": "ready", "error": "arm server gave no answer within %ss" % timeout}
        return msg.get("msg") or {"event": "error", "stage": "ready", "error": "arm server reply without msg"}


class _ArmServers:
    def __init__(self):
        self.servers, self.starts = {}, []
        self.load_timeout = 1800.0
        self.request_timeout = 3600.0
        self.log_dir = Path(os.environ.get("TMPDIR") or "/tmp")

    def server(self, interp, env):
        key = (tuple(interp), tuple(sorted((k, env.get(k)) for k in PROCESS_START_ENV)))
        if key not in self.servers:
            self.servers[key] = ArmServer(interp, env, self.load_timeout, self.log_dir)
        return self.servers[key]

    def ready(self, interp, env, script, argv, log):
        return self.server(interp, env).ready(script, argv, env, log, self.request_timeout)

    def prestart(self, interp, env):
        """Load an arm server before the per-recipe loop (outside every recipe's timer). Returns its start."""
        server = self.server(interp, env)
        if not server.alive():
            server.start()
        return (server.starts or [{}])[-1]

    def report(self):
        return dict(servers=len(self.servers), starts=list(self.starts))

    def stop(self):
        for s in self.servers.values():
            s.stop()


ARM_SERVERS = _ArmServers()


def arm_serve(stdin=None, stdout=None):
    """`smoke_worker.py arm-serve`: answer {"driver", "argv", "env", "log"} requests with the driver's
    probe_ready message, one JSON line each."""
    stdin = stdin or sys.stdin
    proto = stdout or os.fdopen(os.dup(1), "w", buffering=1)
    os.dup2(2, 1)
    sys.stdout = sys.stderr
    mark_process()
    for line in stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
        except ValueError:
            continue
        record = {"process": "arm server pid %d" % os.getpid()}
        try:
            mod = driver_module(req["driver"])
            with redirect_output(req["log"]), working_directory(ROOT), scoped_env(req.get("env"), record):
                msg = mod.probe_ready(list(req.get("argv") or []))
        except Exception as exc:  # noqa: BLE001
            msg = {"event": "error", "stage": "ready", "error": repr(exc)[:2000]}
        if isinstance(msg, dict):
            msg["probe_environment"] = record
        proto.write(json.dumps({"msg": msg}, sort_keys=True, default=str) + "\n")
        proto.flush()
    return 0


def start_worker(arm, cmd, env, log_path, cwd, params_only, subprocess_worker):
    """The drivers' worker factory: the in-process stand-in inside the smoke worker (parameter probe only),
    else `subprocess_worker(arm, cmd, env, log_path, cwd)` (classical_two_datasets.Worker) as always."""
    if in_process():
        if not params_only:
            raise RuntimeError("in-process arm workers serve --params-only probes only; a timed race always "
                               "starts worker subprocesses")
        return InProcessWorker(arm, cmd, env, log_path, cwd)
    return subprocess_worker(arm, cmd, env, log_path, cwd)
