#!/usr/bin/env python3
"""The smoke's warm process: load everything once, then fork every Python the smoke starts.

Andrew (2026-10-14): "load everything once, up front", then give each recipe its own 5 s for its
own check. A recipe's check is three levels of Python processes (the board, its driver's
--params-only probe, one worker per arm), and each cold interpreter re-imports torch, cuML, cupy
and the rest, which alone takes more than 5 s on a cold box. So the smoke starts ONE warm server
per box (`serve`) that imports every driver and every heavy library once, WITHOUT touching a GPU,
and hands out a launcher (`<warm>/bin/python`) that the smoke passes as the board's --python-env.
Every `<launcher> script args` / `-c code` / `-m module` at any level connects to the server, which
forks a child that already holds the loaded libraries; the child takes the caller's stdin, stdout,
stderr, environment, working directory and argv, and runs the script as __main__. The launcher
relays the exit status and forwards signals. A recipe's 5 s timer therefore starts with
everything already loaded.

Why fork and not threads or SIGALRM in one process: each recipe's check is a tree of processes with
their own environments (per-arm thread quotas, numeric mode, device masks) and must stay killable as
a whole at its timeout; one exception or hang must not take the next recipe down. A forked child per
process keeps all of that exactly as a cold run has it.

Fork safety: the server never initializes a GPU. Libraries are imported only; on NVIDIA mojolearn's
import would call cuInit to pick its PTX set, so the server imports it with MOJOLEARN_GPU_ARCH set to
the arch a separate cold process reported (children whose own environment differs re-import
mojolearn fresh). After the preload the server checks the CUDA driver is still uninitialized
(cuCtxGetCurrent returns CUDA_ERROR_NOT_INITIALIZED); if a preload initialized it, the server
refuses to serve and every launch falls back to a cold interpreter (recorded, never silent).

A launch the server cannot reproduce faithfully runs COLD (the launcher execs the real interpreter):
unknown interpreter flags, a PYTHONPATH that would shadow a preloaded package, an LD_LIBRARY_PATH
or LD_PRELOAD entry the server did not start with, or another PYTHON* setting read at interpreter
start. Every cold launch is counted with its reason in smoke.json.

Linux only (NVIDIA and AMD boxes). macOS forks are not safe once the Objective-C runtime and Metal
are loaded, so the Apple smoke keeps one cold process per recipe.
"""
import argparse
import json
import os
from pathlib import Path
import select
import signal
import socket
import struct
import sys
import time

ROOT = Path(__file__).resolve().parent.parent
TOOLS = ROOT / "tools"

#: Every library any recipe's drivers import (tools/bench_board*.py, classical_two_datasets.py,
#: speed_gbdt_arm.py, bench/speed/forest_speed_arm.py). A library that is not installed on this box is
#: recorded as absent, not a failure: the recipe that needs it fails on its own. hipVS imports as `cuvs`.
LIBRARIES = ("numpy", "scipy", "scipy.linalg", "scipy.sparse", "sklearn", "sklearn.cluster",
             "sklearn.decomposition", "sklearn.neighbors", "sklearn.linear_model", "sklearn.metrics",
             "threadpoolctl", "statsmodels", "statsmodels.tsa.arima.model", "statsmodels.tsa.holtwinters",
             "torch", "torch._dynamo", "torch._inductor", "torch._inductor.compile_fx", "triton",
             "torchvision", "torch_geometric", "gpytorch", "transformers",
             "transformers.models.llama.modeling_llama", "transformers.models.mixtral.modeling_mixtral",
             "mamba_ssm", "causal_conv1d", "cupy", "rmm", "cudf", "cuml", "cuml.cluster",
             "cuml.decomposition", "cuml.manifold", "cuml.neighbors", "cuvs", "cuvs.neighbors", "cugraph",
             "faiss", "umap", "xgboost", "lightgbm", "catboost")
#: Our drivers, imported as modules (their scripts then run as __main__ in each child, as cold).
DRIVERS = ("bench_board", "bench_board_evidence", "bench_board_params", "bench_board_algos",
           "bench_board_neural", "bench_board_more", "classical_two_datasets", "race_scorer_preflight",
           "neural_fixtures", "speed_gbdt_arm", "forest_speed_arm")
#: mojolearn reads these once at import (tier, vendor, binary set); a child whose value differs from
#: the server's re-imports mojolearn in its own environment.
MOJOLEARN_IMPORT_ENV = ("MOJOLEARN_NUMERIC_MODE", "MOJOLEARN_VENDOR", "MOJOLEARN_VENDOR_FORCE",
                        "MOJOLEARN_GPU_ARCH")
#: Interpreter-start PYTHON* settings a forked child would not honor. Anything else differing -> cold.
PY_ENV_OK = ("PYTHONPATH", "PYTHONUNBUFFERED", "PYTHONDONTWRITEBYTECODE", "PYTHONIOENCODING")
CUDA_ERROR_NOT_INITIALIZED = 3


# ---------------------------------------------------------------------------
# launcher (`client`): stdlib only, runs under `python -I -S`
# ---------------------------------------------------------------------------

def _send_request(sock, payload, fds):
    data = json.dumps(payload).encode()
    import array
    sock.sendmsg([struct.pack("!Q", len(data))],
                 [(socket.SOL_SOCKET, socket.SCM_RIGHTS, array.array("i", fds))] if fds else [])
    sock.sendall(data)


def _read_line(sock, buf=b""):
    while b"\n" not in buf:
        try:
            chunk = sock.recv(65536)
        except InterruptedError:
            continue
        if not chunk:
            return None, buf
        buf += chunk
    line, buf = buf.split(b"\n", 1)
    return json.loads(line.decode()), buf


def client(real, sock_path, argv):
    """The launcher: ask the server to fork `argv`; exec the real interpreter when it cannot."""
    def cold(_why=None):
        os.execv(real, [real] + argv)
    try:
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        sock.connect(sock_path)
    except OSError:
        cold()
    fds = []
    for fd in (0, 1, 2):
        try:
            os.fstat(fd)
            fds.append(fd)
        except OSError:
            pass
    try:
        umask = os.umask(0o022)
        os.umask(umask)
        _send_request(sock, dict(argv=argv, env=dict(os.environ), cwd=os.getcwd(), umask=umask, fds=fds), fds)
        reply, buf = _read_line(sock)
    except (OSError, ValueError):
        sock.close()
        cold()
    if not reply or "pid" not in reply:
        sock.close()
        cold()
    pid = reply["pid"]

    def forward(sig, _frame):
        try:
            os.killpg(pid, sig)
        except OSError:
            pass
    for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP, signal.SIGQUIT):
        signal.signal(sig, forward)
    try:
        done, buf = _read_line(sock, buf)
    except (OSError, ValueError):
        done = None
    if not done:
        forward(signal.SIGKILL, None)
        os._exit(1)
    if done.get("signal"):
        sig = int(done["signal"])
        signal.signal(sig, signal.SIG_DFL)
        os.kill(os.getpid(), sig)
        os._exit(128 + sig)
    os._exit(int(done.get("exit") or 0))


def control(sock_path, op, timeout=60, **extra):
    """One control request to the server ({"op": "end"} / "stats" / "shutdown"); its reply or None."""
    try:
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        sock.settimeout(timeout)
        sock.connect(str(sock_path))
        _send_request(sock, dict(extra, op=op), [])
        reply, _ = _read_line(sock)
        sock.close()
        return reply
    except (OSError, ValueError):
        return None


def write_launcher(warm, real, sock_path):
    """<warm>/bin/python: the launcher, under a prefix whose lib/ is the real environment's lib/, so code
    that finds package-local native libraries from the interpreter's prefix (bench_board.child_env's
    AMD lib64 glob) finds the same directories."""
    warm = Path(warm)
    (warm / "bin").mkdir(parents=True, exist_ok=True)
    real_prefix = Path(os.path.dirname(os.path.dirname(os.path.abspath(real))))
    lib = warm / "lib"
    if not lib.exists() and (real_prefix / "lib").is_dir():
        lib.symlink_to(real_prefix / "lib")
    launcher = warm / "bin" / "python"
    launcher.write_text("#!/bin/sh\nexec %s -I -S %s client %s %s -- \"$@\"\n" % (
        _q(real), _q(str(Path(__file__).resolve())), _q(real), _q(str(sock_path))))
    launcher.chmod(0o755)
    return launcher


def _q(text):
    return "'" + str(text).replace("'", "'\\''") + "'"


# ---------------------------------------------------------------------------
# server (`serve`)
# ---------------------------------------------------------------------------

def cuda_driver_state():
    """'not-loaded', 'uninitialized' or 'initialized' for the CUDA driver in this process (Linux)."""
    import ctypes
    try:
        lib = ctypes.CDLL("libcuda.so.1", mode=getattr(os, "RTLD_NOLOAD", 4) | ctypes.RTLD_GLOBAL)
    except OSError:
        return "not-loaded"
    ctx = ctypes.c_void_p()
    rc = lib.cuCtxGetCurrent(ctypes.byref(ctx))
    return "uninitialized" if rc == CUDA_ERROR_NOT_INITIALIZED else "initialized"


def gpu_initialized():
    """The name of what initialized a GPU runtime in this process, or None."""
    torch = sys.modules.get("torch")
    try:
        if torch is not None and torch.cuda.is_initialized():
            return "torch.cuda"
    except Exception:  # noqa: BLE001
        pass
    if sys.platform.startswith("linux") and cuda_driver_state() == "initialized":
        return "cuda driver (cuInit ran)"
    return None


def _version(mod):
    v = getattr(mod, "__version__", None)
    return str(v) if v is not None else None


def preload(report, mojolearn_arch=None):
    """Import every driver and library once. Stops importing (keeping what loaded) the moment one
    initializes a GPU runtime: the report names it and the server will not serve."""
    import importlib
    sys.path[:0] = [str(TOOLS), str(ROOT / "bench" / "speed")]
    try:
        for kind, names in (("library", LIBRARIES), ("driver", DRIVERS), ("mojolearn", ("mojolearn",))):
            for name in names:
                t0 = time.time()
                rec = dict(kind=kind)
                saved = None
                if name == "mojolearn" and not mojolearn_arch:
                    report["preloads"][name] = dict(kind=kind, status="skipped", seconds=0.0, error=(
                        "no GPU arch named: importing mojolearn would pick its set with cuInit; each child imports it"))
                    continue
                try:
                    if name == "mojolearn" and mojolearn_arch and not os.environ.get("MOJOLEARN_GPU_ARCH"):
                        # mojolearn's import picks its set with cuInit unless the arch is named; name the arch a
                        # cold process reported so the server stays fork safe.
                        saved = True
                        os.environ["MOJOLEARN_GPU_ARCH"] = mojolearn_arch
                    mod = importlib.import_module(name)
                    rec.update(status="ok", version=_version(mod), file=getattr(mod, "__file__", None))
                except ImportError as exc:
                    # absent = the package is not installed; an installed package that fails to import (a
                    # missing system library such as hipVS's libopenblas.so.0) is a failure
                    absent = isinstance(exc, ModuleNotFoundError) and (exc.name or "") == name.split(".")[0]
                    rec.update(status="absent" if kind == "library" and absent else "failed",
                               error=repr(exc)[:400])
                except BaseException as exc:  # noqa: BLE001 - recorded; a library may raise anything
                    rec.update(status="failed", error=repr(exc)[:400])
                finally:
                    if saved:
                        os.environ.pop("MOJOLEARN_GPU_ARCH", None)
                rec["seconds"] = round(time.time() - t0, 3)
                report["preloads"][name] = rec
                why = gpu_initialized()
                if why:
                    report["gpu_initialized_by"] = "%s (%s)" % (name, why)
                    return
    finally:
        for p in (str(TOOLS), str(ROOT / "bench" / "speed")):
            while p in sys.path:
                sys.path.remove(p)
    if "mojolearn" in sys.modules:
        report["mojolearn_import_env"] = {k: os.environ.get(k) for k in MOJOLEARN_IMPORT_ENV}
        if mojolearn_arch:
            report["mojolearn_import_env"]["MOJOLEARN_GPU_ARCH"] = mojolearn_arch


def parse_argv(argv):
    """(kind, target, rest, flags) for `[flags] script|-c code|-m module args`, or (None, why, ...)."""
    flags = set()
    i = 0
    while i < len(argv) and argv[i].startswith("-") and argv[i] not in ("-c", "-m", "-"):
        if argv[i] not in ("-u", "-B", "-s", "-E"):
            return None, "interpreter flag %s" % argv[i], None, None
        flags.add(argv[i])
        i += 1
    if i >= len(argv) or argv[i] == "-":
        return None, "interactive or stdin program", None, None
    if argv[i] in ("-c", "-m"):
        if i + 1 >= len(argv):
            return None, "%s without an argument" % argv[i], None, None
        return argv[i], argv[i + 1], argv[i + 2:], flags
    return "script", argv[i], argv[i + 1:], flags


def _paths(text):
    return [os.path.realpath(p) for p in (text or "").split(os.pathsep) if p]


class Server:
    def __init__(self, sock_path, report):
        self.sock_path = str(sock_path)
        self.report = report
        self.env0 = dict(os.environ)
        self.ld0 = set(_paths(self.env0.get("LD_LIBRARY_PATH")))
        self.base_path = list(sys.path[1:])           # the server script's own dir is not a child's path
        self.launches = dict(warm=0, cold=[])
        self.children = {}                            # pid -> conn
        self.pending = {}                             # conn -> bytes buffered before a full request

    # --- cold decisions -------------------------------------------------
    def cold_reason(self, req, kind, target, flags):
        if kind is None:
            return target
        env = req["env"]
        for k in set(env) | set(self.env0):
            if k.startswith("PYTHON") and k not in PY_ENV_OK and env.get(k) != self.env0.get(k):
                return "interpreter setting %s differs" % k
        if env.get("LD_PRELOAD") != self.env0.get("LD_PRELOAD"):
            return "LD_PRELOAD differs"
        extra = [p for p in _paths(env.get("LD_LIBRARY_PATH")) if p not in self.ld0]
        if extra:
            return "LD_LIBRARY_PATH adds %s" % extra[0]
        if "-E" not in flags:
            for entry in (env.get("PYTHONPATH") or "").split(os.pathsep):
                shadow = self._shadow(entry)
                if shadow:
                    return "PYTHONPATH %s shadows the preloaded %s" % (entry, shadow)
        return None

    def _shadow(self, entry):
        if not entry or not os.path.isdir(entry):
            return None
        names = os.listdir(entry)
        for name, mod in list(sys.modules.items()):
            if "." in name or mod is None:
                continue
            f = getattr(mod, "__file__", None)
            if not f:
                continue
            hit = name in names or (name + ".py") in names or any(
                n.startswith(name + ".") and n.endswith(".so") for n in names)
            if hit and not os.path.realpath(f).startswith(os.path.realpath(entry) + os.sep):
                return name
        return None

    # --- the loop ---------------------------------------------------------
    def serve(self):
        try:
            os.unlink(self.sock_path)
        except OSError:
            pass
        lst = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        lst.bind(self.sock_path)
        lst.listen(64)
        self.lst = lst
        Path(self.sock_path + ".ready").write_text(json.dumps(self.report))
        while True:
            conns = [lst] + list(self.pending) + list(self.children.values())
            try:
                ready, _, _ = select.select(conns, [], [], 0.1)
            except InterruptedError:
                ready = []
            for s in ready:
                if s is lst:
                    conn, _ = lst.accept()
                    self.pending[conn] = b""
                elif s in self.pending:
                    if self.take(s) == "shutdown":
                        self.kill_all()
                        return 0
                else:
                    # the launcher closed its end (killed): kill that child's whole group now
                    pid = next(p for p, c in self.children.items() if c is s)
                    try:
                        if not s.recv(1):
                            os.killpg(pid, signal.SIGKILL)
                    except OSError:
                        pass
            self.reap()

    def take(self, conn):
        """Read one request (length header + fds, then JSON) and answer it."""
        import array
        try:
            fdsz = array.array("i")
            msg, anc, _, _ = conn.recvmsg(8, socket.CMSG_SPACE(3 * fdsz.itemsize))
            for level, typ, data in anc:
                if level == socket.SOL_SOCKET and typ == socket.SCM_RIGHTS:
                    fdsz.frombytes(data[:len(data) - (len(data) % fdsz.itemsize)])
            if len(msg) < 8:
                raise ValueError("short header")
            size = struct.unpack("!Q", msg)[0]
            conn.settimeout(30)
            data = b""
            while len(data) < size:
                chunk = conn.recv(size - len(data))
                if not chunk:
                    raise ValueError("short request")
                data += chunk
            conn.settimeout(None)
            req = json.loads(data.decode())
        except (OSError, ValueError) as exc:
            self.pending.pop(conn, None)
            conn.close()
            self.launches["cold"].append("bad request: %r" % (exc,))
            return None
        self.pending.pop(conn, None)
        if "op" in req:
            return self.op(conn, req)
        kind, target, rest, flags = parse_argv(req["argv"])
        why = self.cold_reason(req, kind, target, flags or set())
        if why:
            self.launches["cold"].append("%s: %s" % (why, " ".join(req["argv"][:3])[:200]))
            for fd in fdsz:
                os.close(fd)
            conn.sendall((json.dumps(dict(cold=why)) + "\n").encode())
            conn.close()
            return None
        sys.stdout.flush()
        sys.stderr.flush()
        pid = os.fork()
        if pid == 0:
            self.child(conn, req, list(fdsz), kind, target, rest, flags)   # never returns
        for fd in fdsz:
            os.close(fd)
        self.children[pid] = conn
        self.launches["warm"] += 1
        conn.sendall((json.dumps(dict(pid=pid)) + "\n").encode())
        return None

    def op(self, conn, req):
        reply = dict(ok=True)
        if req["op"] in ("end", "stats"):
            reply.update(warm=self.launches["warm"], cold=list(self.launches["cold"]),
                         live=len(self.children))
            if req["op"] == "end":
                reply["killed"] = self.kill_all()
                self.launches = dict(warm=0, cold=[])
        try:
            conn.sendall((json.dumps(reply) + "\n").encode())
        except OSError:
            pass
        conn.close()
        return req["op"] if req["op"] == "shutdown" else None

    def kill_all(self):
        """SIGKILL every live child's process group (a recipe's leftovers at its end or timeout)."""
        killed = 0
        for pid in list(self.children):
            try:
                os.killpg(pid, signal.SIGKILL)
                killed += 1
            except OSError:
                pass
        deadline = time.time() + 10
        while self.children and time.time() < deadline:
            self.reap()
            time.sleep(0.02)
        return killed

    def reap(self):
        while self.children:
            try:
                pid, status = os.waitpid(-1, os.WNOHANG)
            except ChildProcessError:
                return
            if pid == 0:
                return
            conn = self.children.pop(pid, None)
            if conn is None:
                continue
            done = dict(signal=os.WTERMSIG(status)) if os.WIFSIGNALED(status) else \
                dict(exit=os.WEXITSTATUS(status))
            try:
                conn.sendall((json.dumps(done) + "\n").encode())
            except OSError:
                pass
            conn.close()

    # --- the forked child -------------------------------------------------
    def child(self, conn, req, fds, kind, target, rest, flags):
        code = 1
        try:
            os.setsid()
            for s in [self.lst] + list(self.pending) + list(self.children.values()) + [conn]:
                try:
                    s.close()
                except OSError:
                    pass
            for want, fd in zip(req.get("fds") or [], fds):
                os.dup2(fd, want)
            for fd in fds:
                if fd not in (0, 1, 2):
                    os.close(fd)
            signal.signal(signal.SIGCHLD, signal.SIG_DFL)
            signal.signal(signal.SIGTERM, signal.SIG_DFL)
            signal.signal(signal.SIGHUP, signal.SIG_DFL)
            signal.signal(signal.SIGINT, signal.default_int_handler)
            env = req["env"]
            os.environ.clear()
            os.environ.update(env)
            os.chdir(req["cwd"])
            os.umask(int(req.get("umask", 0o022)))
            self.purge_mojolearn(env)
            self.thread_limits(env)
            first = os.path.dirname(os.path.realpath(target)) if kind == "script" else \
                ("" if kind == "-c" else os.getcwd())
            extra = [] if "-E" in flags else [p for p in (env.get("PYTHONPATH") or "").split(os.pathsep) if p]
            sys.path[:] = [first] + extra + self.base_path
            sys.argv = ([target] if kind == "script" else (["-c"] if kind == "-c" else [target])) + list(rest)
            sys.executable = self.report["launcher"]
            if "-B" in flags or env.get("PYTHONDONTWRITEBYTECODE"):
                sys.dont_write_bytecode = True
            if "-u" in flags or env.get("PYTHONUNBUFFERED"):
                for stream in (sys.stdout, sys.stderr):
                    try:
                        stream.reconfigure(write_through=True)
                    except (AttributeError, ValueError):
                        pass
        except BaseException:  # noqa: BLE001
            import traceback
            os.write(2, ("smoke-warm: child setup failed\n" + traceback.format_exc()).encode())
            os._exit(1)
        code = run_main(kind, target)
        os._exit(code)

    def purge_mojolearn(self, env):
        want = self.report.get("mojolearn_import_env")
        if want is None:
            return
        if all(env.get(k) == want.get(k) or (k == "MOJOLEARN_GPU_ARCH" and env.get(k) is None)
               for k in MOJOLEARN_IMPORT_ENV):
            return
        for name in [n for n in sys.modules if n == "mojolearn" or n.startswith(("mojolearn.", "_mojolearn"))]:
            del sys.modules[name]

    @staticmethod
    def thread_limits(env):
        """Apply the child's thread quota to libraries that read it at load (they loaded in the server)."""
        try:
            n = int(env.get("OMP_NUM_THREADS") or 0)
        except ValueError:
            n = 0
        if n <= 0:
            return
        torch = sys.modules.get("torch")
        if torch is not None:
            try:
                torch.set_num_threads(n)
            except Exception:  # noqa: BLE001
                pass
        if "threadpoolctl" in sys.modules:
            try:
                Server._limits = sys.modules["threadpoolctl"].threadpool_limits(n)
            except Exception:  # noqa: BLE001
                pass


def run_main(kind, target):
    """Run the program as __main__ and return the exit code a cold interpreter would."""
    import atexit
    import runpy
    import threading
    import traceback
    code = 0
    try:
        if kind == "script":
            runpy.run_path(target, run_name="__main__")
        elif kind == "-m":
            runpy.run_module(target, run_name="__main__", alter_sys=True)
        else:
            exec(compile(target, "<string>", "exec"), {"__name__": "__main__", "__builtins__": __builtins__})
    except SystemExit as exc:
        if exc.code is None:
            code = 0
        elif isinstance(exc.code, int):
            code = exc.code
        else:
            print(exc.code, file=sys.stderr)
            code = 1
    except KeyboardInterrupt:
        code = 128 + signal.SIGINT
    except BaseException:  # noqa: BLE001
        traceback.print_exc()
        code = 1
    try:
        threading._shutdown()          # a cold interpreter waits for its non-daemon threads
    except Exception:  # noqa: BLE001
        pass
    try:
        atexit._run_exitfuncs()
    except Exception:  # noqa: BLE001
        pass
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.flush()
        except Exception:  # noqa: BLE001
            pass
    return code & 0xFF


def serve_main(args):
    report = dict(schema="mojolearn-smoke-warm/1", pid=os.getpid(), python=sys.executable,
                  launcher=args.launcher, started=time.time(), preloads={}, gpu_initialized_by=None)
    if args.no_preload:
        report["preloads"] = {}
    else:
        preload(report, mojolearn_arch=args.mojolearn_arch)
    report["import_seconds"] = round(time.time() - report["started"], 3)
    report["threads_after_preload"] = __import__("threading").active_count()
    if sys.platform.startswith("linux"):
        report["cuda_driver"] = cuda_driver_state()
        try:
            report["os_threads_after_preload"] = len(os.listdir("/proc/self/task"))
        except OSError:
            pass
    if report["gpu_initialized_by"]:
        # not fork safe: refuse to serve; every launch then runs cold (the launcher cannot connect)
        Path(args.sock + ".ready").write_text(json.dumps(dict(report, serving=False)))
        return 3
    report["serving"] = True
    return Server(args.sock, report).serve()


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if argv[:1] == ["client"]:
        real, sock_path = argv[1], argv[2]
        rest = argv[4:] if argv[3:4] == ["--"] else argv[3:]
        return client(real, sock_path, rest)
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("serve")
    s.add_argument("--sock", required=True)
    s.add_argument("--launcher", required=True)
    s.add_argument("--mojolearn-arch", help="name mojolearn's GPU arch for the server's own import (no cuInit)")
    s.add_argument("--no-preload", action="store_true", help="serve without importing anything (tests)")
    args = ap.parse_args(argv)
    return serve_main(args)


if __name__ == "__main__":
    sys.exit(main())
