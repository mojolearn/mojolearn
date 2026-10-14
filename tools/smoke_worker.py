#!/usr/bin/env python3
"""The campaign smoke's one long-lived worker: load everything once, then check every recipe in process.

Andrew (2026-10-14), exact: LOAD EVERYTHING ONCE, up front. `serve` imports every driver and every library
any recipe needs (torch and its compile backend, cuML/cuVS/cuGraph, cupy, gpytorch, xgboost, catboost,
lightgbm, transformers, mamba-ssm, hipVS on AMD (it imports as cuvs), torch_geometric, our mojolearn bindings),
records each import's seconds and status, initializes each GPU runtime ONCE (torch CUDA/ROCm/MPS, cupy,
mojolearn) and records that cost separately, then serves requests one at a time as JSON lines (stdin in,
a private copy of fd 1 out; everything a library prints goes to the recipe's log):

- {"op": "warm-box", "argv": [...], "paths": [...]}: the box facts every recipe's check reads (the box
  fingerprint, the repo commit, the harness hash, the driver version, each data file's sha256), once,
  before the per-recipe loop, outside any recipe's timer.
- {"op": "recipe", "key", "argv", "log", "timeout"}: tools/bench_board.py main(argv) (a --preflight-only
  --preflight-scorer check of one race) IN THIS PROCESS: the board's parameter probe calls each family
  driver and constructs ours and every opponent arm as function calls (tools/smoke_inproc.py), audits the
  records and runs the light scorer preflight. The recipe gets its own SIGALRM timer
  (setitimer(ITIMER_REAL, timeout)), started when its check starts. On the alarm the check stops where it is
  and the reply says "timeout" (the campaign records STARTED with how far it got); an exception is recorded
  as "error"; either way the worker goes on to the next request. A C call that never returns to Python is
  the campaign's outer watchdog's to handle (it kills and restarts this worker).
- {"op": "quit"}.

`arm-serve` is the second kind of long-lived process: an arm whose worker needs another interpreter or
loader environment (tools/smoke_inproc.py ArmServers) is constructed there, loaded once per environment.

No fits, no warm-ups, no timed work: the parameter records are the drivers' own (same builders, same
read-back) and the parameter-check rules and quality margins are unchanged.
"""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback

TOOLS = Path(__file__).resolve().parent
ROOT = TOOLS.parent
SPEED = ROOT / "bench" / "speed"

#: Every library any recipe's drivers import. A library that is not installed on this box is listed as
#: absent, not a failure (the recipe that needs it fails on its own); an installed one that fails to import
#: is a failure. hipVS imports as `cuvs`. The torch.compile backend is torch._inductor.
LIBRARIES = ("numpy", "scipy", "scipy.linalg", "scipy.sparse", "sklearn", "sklearn.cluster",
             "sklearn.decomposition", "sklearn.neighbors", "sklearn.linear_model", "sklearn.metrics",
             "threadpoolctl", "statsmodels", "statsmodels.tsa.arima.model", "statsmodels.tsa.holtwinters",
             "torch", "torch._dynamo", "torch._inductor", "torch._inductor.compile_fx", "triton",
             "torchvision", "torch_geometric", "gpytorch", "transformers",
             "transformers.models.llama.modeling_llama", "transformers.models.mixtral.modeling_mixtral",
             "mamba_ssm", "causal_conv1d", "cupy", "rmm", "cudf", "cuml", "cuml.cluster",
             "cuml.decomposition", "cuml.manifold", "cuml.neighbors", "cuvs", "cuvs.neighbors", "cugraph",
             "faiss", "umap", "xgboost", "lightgbm", "catboost", "implicit")
#: Our harness modules, imported by name.
MODULES = ("bench_board", "bench_board_evidence", "bench_board_params", "bench_board_probe",
           "race_scorer_preflight", "race_fairness", "neural_fixtures", "speed_gbdt_arm")
#: The family drivers the board calls (--params-only), loaded once at the paths the board passes them.
DRIVER_SCRIPTS = (TOOLS / "classical_two_datasets.py", TOOLS / "bench_board_more.py",
                  TOOLS / "bench_board_algos.py", TOOLS / "bench_board_neural.py",
                  SPEED / "forest_speed_arm.py")


def _version(mod):
    v = getattr(mod, "__version__", None)
    return str(v) if v is not None else None


def classify_import_error(name, exc, kind="library"):
    """'absent' (the top-level package is not installed; libraries only) or 'failed' (installed but broken,
    e.g. a missing system library, or any harness/driver module)."""
    absent = isinstance(exc, ModuleNotFoundError) and (getattr(exc, "name", None) or "") == name.split(".")[0]
    return "absent" if kind == "library" and absent else "failed"


def import_one(name, kind, importer=None):
    """{kind, status ok|absent|failed, seconds, version, error}."""
    import importlib
    importer = importer or importlib.import_module
    t0 = time.time()
    rec = dict(kind=kind)
    try:
        mod = importer(name)
        rec.update(status="ok", version=_version(mod))
    except ImportError as exc:
        rec.update(status=classify_import_error(name, exc, kind), error=repr(exc)[:400])
    except BaseException as exc:  # noqa: BLE001 - a library may raise anything at import
        if isinstance(exc, (KeyboardInterrupt, SystemExit)) and kind != "library":
            raise
        rec.update(status="failed", error=repr(exc)[:400])
    rec["seconds"] = round(time.time() - t0, 3)
    return rec


def preload(report, libraries=LIBRARIES, modules=MODULES, drivers=DRIVER_SCRIPTS):
    """Import every library, harness module, family driver and mojolearn once; per-module seconds and status
    into report["imports"], the total into report["import_seconds"]."""
    for p in (str(SPEED), str(TOOLS)):
        if p not in sys.path:
            sys.path.insert(0, p)
    import smoke_inproc
    t0 = time.time()
    imports = report.setdefault("imports", {})
    for name in libraries:
        imports[name] = import_one(name, "library")
    imports["mojolearn"] = import_one("mojolearn", "mojolearn")
    for name in modules:
        imports[name] = import_one(name, "module")
    for path in drivers:
        imports[Path(path).name] = import_one(str(path), "driver", importer=smoke_inproc.driver_module)
    report["import_seconds"] = round(time.time() - t0, 2)
    return report


def gpu_init(report, vendor):
    """Initialize each GPU runtime once (a tiny allocation and a synchronize), recorded separately:
    report["gpu_init"] = {runtime: {status, seconds, device|error}}, report["gpu_init_seconds"]."""
    out = report.setdefault("gpu_init", {})
    t_all = time.time()

    def step(name, fn):
        t0 = time.time()
        try:
            got = fn()
            out[name] = dict(status="ok" if got is not None else "absent", device=got)
        except Exception as exc:  # noqa: BLE001 - recorded
            out[name] = dict(status="failed", error=repr(exc)[:400])
        out[name]["seconds"] = round(time.time() - t0, 3)

    def torch_init():
        torch = sys.modules.get("torch")
        if torch is None:
            return None
        if vendor == "apple":
            if not torch.backends.mps.is_available():
                raise RuntimeError("torch sees no MPS device")
            torch.zeros(1, device="mps")
            torch.mps.synchronize()
            return "mps"
        if not torch.cuda.is_available():
            raise RuntimeError("torch sees no CUDA/ROCm device")
        torch.cuda.init()
        torch.zeros(1, device="cuda")
        torch.cuda.synchronize()
        return torch.cuda.get_device_name(0)

    def cupy_init():
        cupy = sys.modules.get("cupy")
        if cupy is None or vendor == "apple":
            return None
        cupy.zeros(1)
        cupy.cuda.Device(0).synchronize()
        name = cupy.cuda.runtime.getDeviceProperties(0)["name"]
        return name.decode() if isinstance(name, bytes) else str(name)

    def mojolearn_init():
        mj = sys.modules.get("mojolearn")
        if mj is None:
            return None
        got = {}
        for attr in ("vendor", "gpu_arch", "numeric_mode"):
            v = getattr(mj, attr, None)
            try:
                got[attr] = v() if callable(v) else v
            except Exception as exc:  # noqa: BLE001
                got[attr] = "error: %r" % (exc,)
        return json.loads(json.dumps(got, default=str))

    step("torch", torch_init)
    step("cupy", cupy_init)
    step("mojolearn", mojolearn_init)
    report["gpu_init_seconds"] = round(time.time() - t_all, 2)
    return report


# ---------------------------------------------------------------------------
# one recipe under its own alarm
# ---------------------------------------------------------------------------

class RecipeTimeout(BaseException):
    """The recipe's own timer fired (BaseException, so a driver's `except Exception` cannot swallow it)."""


_ARMED = [False]


def _on_alarm(signum, frame):
    if _ARMED[0]:
        raise RecipeTimeout()


def arm_alarm(timeout, interval=0.25):
    """Start the recipe's timer. It keeps firing every `interval` after the first expiry, so a handler that
    caught one RecipeTimeout gets the next one."""
    signal.signal(signal.SIGALRM, _on_alarm)
    _ARMED[0] = True
    signal.setitimer(signal.ITIMER_REAL, max(float(timeout), 0.001), interval)


def disarm_alarm():
    _ARMED[0] = False
    signal.setitimer(signal.ITIMER_REAL, 0)


def descendants(pid=None, keep=()):
    """Every live descendant pid of `pid` (default this process), from `ps`; the subtrees of `keep` are
    left out."""
    pid = os.getpid() if pid is None else pid
    try:
        out = subprocess.run(["ps", "-A", "-o", "pid=,ppid="], capture_output=True, text=True, timeout=10).stdout
    except (OSError, subprocess.SubprocessError):
        return []
    children = {}
    for line in out.splitlines():
        parts = line.split()
        if len(parts) == 2 and parts[0].isdigit() and parts[1].isdigit():
            children.setdefault(int(parts[1]), []).append(int(parts[0]))
    found, todo = [], [pid]
    while todo:
        for c in children.get(todo.pop(), []):
            if c not in found and c not in keep:
                found.append(c)
                todo.append(c)
    return found


def reap_children(keep=()):
    """Kill whatever a recipe left running below this worker (arm servers in `keep` stay). Returns the pids."""
    killed = []
    for pid in descendants(keep=set(keep)):
        try:
            os.kill(pid, signal.SIGKILL)
            killed.append(pid)
        except OSError:
            pass
    for pid in killed:
        try:
            os.waitpid(pid, os.WNOHANG)
        except OSError:
            pass
    return killed


def run_recipe(fn, argv, log, timeout):
    """fn(argv) (bench_board.main) under `timeout` seconds of SIGALRM, its output in `log`.
    Returns {status done|timeout|error, rc, seconds, error, children_killed}."""
    import smoke_inproc
    t0 = time.time()
    reply = dict(status="error", rc=None, error=None)
    finished = False
    try:
        with smoke_inproc.redirect_output(log):
            print("=== smoke worker pid %d: recipe check in process, %ss timer" % (os.getpid(), timeout), flush=True)
            try:
                try:
                    arm_alarm(timeout)
                    rc = fn(argv)
                    finished = True
                    reply.update(status="done", rc=rc if isinstance(rc, int) else (0 if rc is None else 1))
                except SystemExit as exc:
                    finished = True
                    code = exc.code
                    if code is not None and not isinstance(code, int):
                        print(code, flush=True)
                        code = 1
                    reply.update(status="done", rc=code or 0)
                finally:
                    disarm_alarm()
            except RecipeTimeout:
                if not finished:
                    reply.update(status="timeout", rc=None)
                    print("=== smoke worker: TIMER FIRED at %ss; stopped here:\n%s" % (
                        timeout, "".join(traceback.format_stack(limit=12))[-3000:]), flush=True)
            except Exception as exc:  # noqa: BLE001 - recorded, the worker goes on
                traceback.print_exc()
                reply.update(status="error", rc=None, error=repr(exc)[:1200])
    except RecipeTimeout:
        disarm_alarm()
        if not finished:
            reply.update(status="timeout", rc=None)
    finally:
        disarm_alarm()
    # an arm server cut off mid-request stays (it is restarted on its next use: smoke_inproc.ArmServer.busy)
    keep = {s.proc.pid for s in smoke_inproc.ARM_SERVERS.servers.values() if s.alive()}
    reply["children_killed"] = reap_children(keep)
    reply["seconds"] = round(time.time() - t0, 2)
    return reply


def warm_box(argv, paths):
    """The box facts every recipe's check reads, once, outside any recipe's timer. Returns per-fact seconds."""
    import bench_board as B
    got = {}

    def step(name, fn):
        t0 = time.time()
        try:
            fn()
            got[name] = dict(status="ok")
        except Exception as exc:  # noqa: BLE001 - recorded; the recipe pays it instead
            got[name] = dict(status="failed", error=repr(exc)[:400])
        got[name]["seconds"] = round(time.time() - t0, 2)

    args = B.build_parser().parse_args(list(argv))
    vendor = args.vendor
    step("repo_commit", B.repo_commit)
    step("harness_sha256", B.smoke_files_sha256)
    if vendor == "nvidia":
        step("nvidia_driver", lambda: B._box_once(("nvidia-driver",), lambda: B.capture(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"])))
    ctx = {"python": args.python_env, "vendor": vendor, "commit": B.repo_commit(),
           "mojolearn_version": args.mojolearn_version, "wheel": None}
    step("box_fingerprint", lambda: B.box_fingerprint(ctx))
    t0 = time.time()
    hashed, missing = 0, []
    for p in paths or []:
        if os.path.isfile(p):
            B.sha256_file(p)
            hashed += 1
        else:
            missing.append(p)
    got["data_sha256"] = dict(status="ok", files=hashed, missing=missing, seconds=round(time.time() - t0, 2))
    return got


def serve(vendor, out):
    """The worker loop (see the module docstring)."""
    for p in (str(SPEED), str(TOOLS)):
        if p not in sys.path:
            sys.path.insert(0, p)
    proto = os.fdopen(os.dup(1), "w", buffering=1)
    os.dup2(2, 1)
    sys.stdout = sys.stderr

    def say(obj):
        proto.write(json.dumps(obj, sort_keys=True, default=str) + "\n")
        proto.flush()

    import smoke_inproc
    smoke_inproc.mark_process()
    smoke_inproc.ARM_SERVERS.log_dir = Path(out)
    report = dict(pid=os.getpid(), python=sys.executable, vendor=vendor, started=time.time())
    t0 = time.time()
    preload(report)
    gpu_init(report, vendor)
    report["wall_seconds"] = round(time.time() - t0, 2)
    say({"event": "ready", "report": report})
    import bench_board as B
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
        except ValueError:
            say({"event": "error", "error": "unreadable request"})
            continue
        op = req.get("op")
        if op == "quit":
            smoke_inproc.ARM_SERVERS.stop()
            say({"event": "bye"})
            return 0
        if op == "warm-box":
            say({"event": "warm-box", "key": req.get("key"),
                 "facts": warm_box(req.get("argv") or [], req.get("paths") or [])})
            continue
        if op == "recipe":
            reply = run_recipe(B.main, list(req["argv"]), req["log"], float(req["timeout"]))
            reply.update(event="recipe", key=req.get("key"), arm_servers=smoke_inproc.ARM_SERVERS.report())
            say(reply)
            continue
        say({"event": "error", "error": "unknown op %r" % (op,)})
    smoke_inproc.ARM_SERVERS.stop()
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("serve")
    s.add_argument("--vendor", choices=("nvidia", "amd", "apple"), required=True)
    s.add_argument("--out", required=True, help="where arm-server logs go")
    sub.add_parser("arm-serve")
    args = ap.parse_args(argv)
    if args.cmd == "serve":
        return serve(args.vendor, args.out)
    for p in (str(SPEED), str(TOOLS)):
        if p not in sys.path:
            sys.path.insert(0, p)
    import smoke_inproc
    proto = os.fdopen(os.dup(1), "w", buffering=1)
    os.dup2(2, 1)                      # what an import prints goes to the server's log, not the protocol
    sys.stdout = sys.stderr
    report = {}
    preload(report)
    proto.write(json.dumps({"event": "ready", "report": report}, sort_keys=True, default=str) + "\n")
    proto.flush()
    return smoke_inproc.arm_serve(stdout=proto)


if __name__ == "__main__":
    sys.exit(main())
