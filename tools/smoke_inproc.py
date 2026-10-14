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
# staged inputs: each dataset block and generated synthetic input once per process
# ---------------------------------------------------------------------------
# smoke-fixes r12 (2026-10-14; the STARTED recipes: optimizer lanes regenerating 11 x 2^24 float32 seed-7 arrays
# and sha256-hashing them once PER ARM; istella/text lanes re-reading a ~1 GB npz (zip crc32) once for the
# conductor and once PER ARM; cholesky rebuilding its 8192 x 8192 system per arm). Andrew's design: everything
# is loaded ONCE up front and each recipe's 5 s is for its own check. The drivers' params-only paths ask this
# cache for the very values their builders make (same builder functions, same seeds, same bytes), so the
# parameter records are the ones the timed path reads back; a timed worker never reaches it (in_process() is
# true only in the smoke worker and its arm servers, which serve the parameter probe only).
# The cached arrays are SHARED between the arms of a recipe and across recipes: the builders treat their
# inputs as read-only (the timed worker already rebuilds every arm from the same D each round, so a builder
# that wrote into its inputs would change its own later rounds).

#: the staged-input budget of one process, in GiB: an explicit override. Unset, the budget is sized from the
#: box's free memory when the cache first stores something (free_memory_bytes x STAGE_FREE_FRACTION).
STAGE_BUDGET_ENV = "MOJOLEARN_SMOKE_STAGE_GB"
STAGE_BUDGET_GB = 8.0          # the budget where the free memory cannot be read
STAGE_FREE_FRACTION = 0.5      # the share of the free memory the staged inputs may hold (the arms need the rest)
STAGE_LRU_FLOOR = 1 << 30      # the LRU keeps at least this much (at most a quarter of the budget) beside the pins


def free_memory_bytes():
    """(bytes, source): the memory this process can still take. Linux: MemAvailable, lowered to the cgroup
    memory.max headroom when one is set (a RunPod container sees the host's /proc/meminfo; the same rule as
    tools/build_sizing.py box_resources). Elsewhere half the physical memory (macOS has no MemAvailable).
    (None, why) when neither can be read."""
    try:
        import build_sizing
        available = build_sizing.mem_available()
        headroom = build_sizing.cgroup_memory_headroom()
        if headroom is not None and headroom < available:
            return int(headroom), "cgroup memory.max headroom"
        return int(available), "MemAvailable"
    except Exception:  # noqa: BLE001 - not Linux, or no /proc/meminfo
        pass
    try:
        return int(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")) // 2, "half the physical memory"
    except (AttributeError, OSError, ValueError):
        return None, "free memory unreadable"


def nbytes_of(value, seen=None):
    """The bytes of the arrays held by `value` (NumPy arrays and torch tensors, in dicts, lists, tuples, scipy
    sparse matrices and plain objects' attributes), each array once."""
    seen = set() if seen is None else seen
    if id(value) in seen:
        return 0
    seen.add(id(value))
    if hasattr(value, "nbytes") and hasattr(value, "dtype"):
        return int(value.nbytes)
    if isinstance(value, dict):
        return sum(nbytes_of(v, seen) for v in value.values())
    if isinstance(value, (list, tuple)):
        return sum(nbytes_of(v, seen) for v in value)
    if all(hasattr(value, k) for k in ("data", "indices", "indptr")):          # scipy CSR/CSC
        return sum(nbytes_of(getattr(value, k), seen) for k in ("data", "indices", "indptr"))
    if hasattr(value, "memory_usage") and hasattr(value, "columns"):           # a pandas DataFrame
        try:
            return int(value.memory_usage(index=True, deep=False).sum())
        except Exception:  # noqa: BLE001
            return 0
    if hasattr(value, "__dict__") and not isinstance(value, type) and not callable(value):
        return sum(nbytes_of(v, seen) for v in vars(value).values())
    return 0


class StagedInputs:
    """An in-process cache of staged inputs: PINNED entries (dataset blocks read from files, kept for the whole
    smoke) and an LRU of the rest under the remaining budget, with hit/miss/eviction counts.

    smoke-fixes r14 (2026-10-14; NVIDIA in-process smoke, 32 STARTED at 5 s, up from 3): under the fixed 8 GB
    budget the LRU evicted 85-96 times, so the istella/taxi blocks were read again inside later recipes'
    timers, and every family re-read its npz there. Now the file-backed blocks are pinned (never evicted)
    while they fit the budget, and the budget is sized from the box's free memory (free_memory_bytes x
    STAGE_FREE_FRACTION) unless MOJOLEARN_SMOKE_STAGE_GB names it; report() records the budget, its source,
    the pinned bytes and the evictions."""

    def __init__(self, budget_bytes=None):
        import collections
        self.budget_source = None
        self.free_bytes = None
        if budget_bytes is None and os.environ.get(STAGE_BUDGET_ENV):
            try:
                budget_bytes = float(os.environ[STAGE_BUDGET_ENV]) * (1 << 30)
                self.budget_source = STAGE_BUDGET_ENV
            except ValueError:
                budget_bytes = None
        self.budget = None if budget_bytes is None else int(budget_bytes)
        if self.budget is not None and self.budget_source is None:
            self.budget_source = "explicit"
        self.entries = collections.OrderedDict()     # key -> (value, nbytes), least recently used first
        self.pinned = {}                             # key -> (value, nbytes), never evicted
        self.bytes = 0                               # LRU bytes
        self.pinned_bytes = 0
        self.hits = self.misses = self.evictions = self.pin_declined = 0
        self.build_seconds = 0.0

    def _size(self):
        """The budget, sized from the free memory on first use (not at import: the worker loads its libraries
        first)."""
        if self.budget is None:
            free, source = free_memory_bytes()
            self.free_bytes = free
            if free is None:
                self.budget, self.budget_source = int(STAGE_BUDGET_GB * (1 << 30)), source + ": the default"
            else:
                self.budget = int(free * STAGE_FREE_FRACTION)
                self.budget_source = "%s x %g at first use" % (source, STAGE_FREE_FRACTION)
        return self.budget

    def lru_budget(self):
        return max(self._size() - self.pinned_bytes, min(STAGE_LRU_FLOOR, self._size() // 4))

    def get(self, key, build, nbytes=None, pin=False):
        if key in self.pinned:
            self.hits += 1
            return self.pinned[key][0]
        if key in self.entries:
            self.entries.move_to_end(key)
            self.hits += 1
            return self.entries[key][0]
        self.misses += 1
        t0 = time.time()
        value = build()
        self.build_seconds += time.time() - t0
        size = nbytes_of(value) if nbytes is None else int(nbytes)   # int() of a _Deferred reads it after build
        if pin:
            if self.pinned_bytes + size <= self._size():
                self.pinned[key] = (value, size)
                self.pinned_bytes += size
                self._evict(None)
                return value
            self.pin_declined += 1                   # over the budget: an ordinary LRU entry instead
        self.entries[key] = (value, size)
        self.bytes += size
        self._evict(key)
        return value

    def _evict(self, keep):
        limit = self.lru_budget()
        while self.bytes > limit and self.entries:
            old_key = next(iter(self.entries))
            if old_key == keep:
                break
            self.bytes -= self.entries.pop(old_key)[1]
            self.evictions += 1

    def clear(self):
        self.entries.clear()
        self.pinned.clear()
        self.bytes = self.pinned_bytes = 0

    def report(self):
        return dict(entries=len(self.entries), bytes=self.bytes, budget=self.budget,
                    budget_source=self.budget_source, free_bytes_at_sizing=self.free_bytes,
                    lru_budget=self.lru_budget() if self.budget is not None else None,
                    pinned=len(self.pinned), pinned_bytes=self.pinned_bytes, pin_declined=self.pin_declined,
                    hits=self.hits, misses=self.misses, evictions=self.evictions,
                    build_seconds=round(self.build_seconds, 2))


STAGED = StagedInputs()


def staged(key, build, nbytes=None, pin=False):
    """build() once per smoke-worker process for `key` (a tuple naming the exact file stamp or seed and size);
    outside the smoke worker it is simply build(). pin=True (a dataset block read from files) keeps it for the
    whole smoke, outside the LRU, while the pinned blocks fit the budget."""
    if not in_process():
        return build()
    return STAGED.get(key, build, nbytes=nbytes, pin=pin)


class _Deferred:
    """An nbytes known only after build() ran (memo_on: the derived value's bytes, not its inputs')."""

    def __init__(self, box):
        self.box = box

    def __int__(self):
        return int(self.box.get("n", 0))


def file_stamp(path):
    """(absolute path, size, mtime_ns) of a file: the key part that names its exact bytes."""
    st = os.stat(path)
    return (os.path.abspath(path), st.st_size, st.st_mtime_ns)


def staged_npz(path, keys=None, contiguous=True, pin=True):
    """{name: array} of an npz file, read once per smoke-worker process and pinned (keyed by its stamp). The
    arrays are what `{k: np.ascontiguousarray(z[k]) for k in z.files}` (contiguous=True) or
    `{k: z[k] ...}` gives: the same values every reader of that file computes from. `keys` limits the arrays
    read (None: all). Outside the smoke worker it simply reads the file."""
    def read():
        import numpy as np
        with np.load(path) as z:
            names = [k for k in z.files if keys is None or k in keys]
            return {k: (np.ascontiguousarray(z[k]) if contiguous else z[k]) for k in names}
    if not in_process():
        return read()
    key = ("npz",) + file_stamp(path) + (tuple(sorted(keys)) if keys is not None else None, bool(contiguous))
    return dict(STAGED.get(key, read, pin=pin))


def memo_on(tag, inputs, build, extra=(), nbytes=None, pin=False):
    """build() once per smoke-worker process for the same live input objects (and `extra`, a hashable that
    names everything else build reads): a value derived from staged inputs (a CSR of a staged block, a
    DataFrame of a staged matrix). The entry holds the inputs, so their ids cannot be reused while it lives.
    Outside the smoke worker it is simply build()."""
    if not in_process():
        return build()
    inputs = tuple(inputs)
    key = ("memo-on", tag, tuple((id(a), _array_stamp(a)) for a in inputs), extra)
    sized = {}

    def make():
        value = build()
        sized["n"] = nbytes_of(value) if nbytes is None else int(nbytes)      # the inputs are counted already
        return inputs, value
    got = STAGED.get(key, make, nbytes=_Deferred(sized), pin=pin)
    if any(x is not y for x, y in zip(got[0], inputs)):        # cannot happen while the entry holds them
        STAGED.entries.pop(key, None)
        STAGED.pinned.pop(key, None)
        return build()
    return got[1]


class _HashMemo:
    """sha256 of an array's C-order bytes, once per live memory (weak references: an entry goes with the object
    that owns the memory). The same bytes as hashlib.sha256(np.ascontiguousarray(a).tobytes()).

    smoke-fixes r14 (2026-10-14): keyed by the object that OWNS the memory (the end of the array's .base
    chain: an ndarray, or the torch tensor behind `t.numpy()`) plus the view's data pointer, dtype, shape and
    strides, not by the view object: every torch layer arm records `x_cpu.numpy()`, a new view object of the
    same staged tensor, and re-hashed its 0.2-0.3 GB per arm. While the owner lives its memory cannot be
    reused, so the same pointer, dtype, shape and strides are the same bytes."""

    def __init__(self):
        self.memo = {}                # id(owner) -> (weakref to owner, {stamp: digest})
        self.hits = self.misses = 0

    def _drop(self, key):
        self.memo.pop(key, None)

    @staticmethod
    def _owner(a):
        owner = a
        while getattr(owner, "base", None) is not None:
            owner = owner.base
        return owner

    def sha256(self, a, compute):
        import weakref
        stamp = _array_stamp(a)
        owner = self._owner(a)
        for cand in (owner, a):
            got = self.memo.get(id(cand))
            if got is not None and got[0]() is cand and stamp in got[1]:
                self.hits += 1
                return got[1][stamp]
        self.misses += 1
        digest = compute(a)
        for cand in (owner, a):       # an owner that is not weak-referenceable (bytes): the view itself
            key = id(cand)
            got = self.memo.get(key)
            if got is not None and got[0]() is cand:
                got[1][stamp] = digest
                return digest
            try:
                ref = weakref.ref(cand, lambda _r, k=key: self._drop(k))
            except TypeError:
                continue
            self.memo[key] = (ref, {stamp: digest})
            return digest
        return digest                 # nothing weak-referenceable: not memoized

    def report(self):
        return dict(live=len(self.memo), hits=self.hits, misses=self.misses)


def _array_stamp(a):
    iface = getattr(a, "__array_interface__", None) or {}
    return (str(getattr(a, "dtype", "")), tuple(getattr(a, "shape", ()) or ()), (iface.get("data") or (None,))[0],
            iface.get("strides"))


HASHES = _HashMemo()


def _sha256_bytes(a):
    import hashlib
    import numpy as np
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def sha256_array(a):
    """hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest(), memoized per live array object inside the
    smoke worker (an array staged once is hashed once, however many arms and recipes record it)."""
    if not in_process():
        return _sha256_bytes(a)
    return HASHES.sha256(a, _sha256_bytes)


def staging_report():
    return dict(staged=STAGED.report(), hashes=HASHES.report())


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
    # sys.stdout/sys.stderr too: they need not write to fds 1 and 2 (a replaced stream, e.g. under pytest)
    text = open(os.dup(fh.fileno()), "w", buffering=1, encoding="utf-8", errors="replace")
    streams = (sys.stdout, sys.stderr)
    try:
        os.dup2(fh.fileno(), 1)
        os.dup2(fh.fileno(), 2)
        sys.stdout = sys.stderr = text
        yield path
    finally:
        for s in (text,) + streams:
            try:
                s.flush()
            except Exception:  # noqa: BLE001
                pass
        sys.stdout, sys.stderr = streams
        os.dup2(saved[0], 1)
        os.dup2(saved[1], 2)
        for fd in saved:
            os.close(fd)
        text.close()
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

def _wanted_mode(env):
    return str((env or {}).get("MOJOLEARN_NUMERIC_MODE") or "identical").strip().lower()


def single_tier_conflict(env):
    """The type-registering mojolearn bindings loaded in this process under another tier than `env`'s
    numeric mode ([] when there is none, or the mode is the process's own).
    # smoke-fixes (2026-10-14; neural/lm-train-step Apple fast: "foreach is unset on torch-eager-fp32 and unset
    # on torch-compile-fp32", "missing parameter/version probe: ours-fast"): the identical recipes loaded
    # _mojolearn_byte_lm's identical tier in the smoke worker; ours-fast then switched the tier in process
    # (scoped_env -> set_numeric_mode) and its import refused (python/mojolearn/_backend.py _TYPE_REGISTERING
    # and _load_member: a binding that registers Mojo-owned Python types holds one tier per process, "Use one
    # numeric_mode per process for this binding"). ours-fast had no record, so a torch arm became the
    # reference and its foreach/fused (torch-only implementation choices) were compared torch to torch.
    # Such an arm now goes to an arm server started in its own numeric mode (ArmServers keys on the mode)."""
    mj = sys.modules.get("mojolearn")
    backend = getattr(mj, "_backend", None) if mj is not None else None
    if backend is None:
        return []
    want = _wanted_mode(env)
    try:
        current = str(backend.default_mode()).strip().lower()
    except Exception:  # noqa: BLE001
        return []
    if current == want:
        return []
    names = getattr(backend, "_TYPE_REGISTERING", ()) or ()
    loaded = []
    for name in sorted(names):
        if ("mojolearn." + name) in sys.modules or any(
                k.startswith("mojolearn._sets.") and k.endswith("." + name)
                and k != "mojolearn._sets.%s.%s" % (want, name) for k in list(sys.modules)):
            loaded.append(name)
    return loaded


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
        tiers = single_tier_conflict(self.env)
        if not same_interpreter(self.interp) or diff["process_start"] or tiers:
            # another interpreter or loader environment: its own long-lived process (loaded once)
            why = ("interpreter %s" % " ".join(self.interp) if not same_interpreter(self.interp)
                   else "process-start settings %s" % ",".join(diff["process_start"]) if diff["process_start"]
                   else "numeric mode %s while %s of another tier is loaded here (one tier per process)"
                   % (_wanted_mode(self.env), ",".join(tiers)))
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

    def stage(self, items, timeout):
        """Stage the next recipe's inputs in this arm server (stage_items), outside every recipe's timer.
        Returns its reply, or a failure record (the server is then restarted on its next use)."""
        if self.busy:
            self.stop()
        if not self.alive() and not self.start():
            return dict(status="failed", problem="arm server did not load: %s" % self.starts[-1].get("problem"))
        self.busy = True
        self.proc.stdin.write((json.dumps(dict(op="stage", items=list(items))) + "\n").encode())
        self.proc.stdin.flush()
        deadline = time.monotonic() + timeout
        while True:
            msg, self.buf = read_json_line(self.proc, self.buf, deadline)
            if msg is None or msg.get("event") != "noise":
                break
        self.busy = False
        if msg is None:
            self.stop()
            return dict(status="failed", problem="arm server gave no staging answer within %ss" % timeout)
        return msg.get("staged") or dict(status="failed", problem="arm server reply without staged")


def stage_items(items):
    """Stage each item's inputs in this process ({"driver": path, "lane", "dataset", "data"}: the driver's
    stage_inputs(lane, dataset, data)), outside every recipe's timer. Returns per-item status and seconds plus
    the cache report. An item that fails is recorded; the recipe then builds its inputs inside its own check."""
    out = []
    for item in items or []:
        t0 = time.time()
        rec = {k: item.get(k) for k in ("lane", "dataset")}
        try:
            mod = driver_module(item["driver"])
            fn = getattr(mod, "stage_inputs", None)
            if fn is None:
                rec.update(status="absent", detail="driver has no stage_inputs")
            else:
                with redirect_output(item["log"]) if item.get("log") else contextlib.nullcontext():
                    rec.update(status="ok", staged=fn(item["lane"], item["dataset"], item["data"]))
        except Exception as exc:  # noqa: BLE001 - recorded; the recipe pays it inside its own check
            rec.update(status="failed", error=repr(exc)[:400])
        rec["seconds"] = round(time.time() - t0, 2)
        out.append(rec)
    return dict(items=out, report=staging_report())


class _ArmServers:
    def __init__(self):
        self.servers, self.starts = {}, []
        self.load_timeout = 1800.0
        self.request_timeout = 3600.0
        self.log_dir = Path(os.environ.get("TMPDIR") or "/tmp")

    def server(self, interp, env):
        # the numeric mode is part of the key: a server loads one tier of the type-registering bindings
        # (single_tier_conflict)
        key = (tuple(interp), tuple(sorted((k, env.get(k)) for k in PROCESS_START_ENV)), _wanted_mode(env))
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
        if req.get("op") == "stage":
            # smoke-fixes r12 (2026-10-14): the next recipe's inputs, staged in this arm server too, outside
            # every recipe's timer (stage_items)
            proto.write(json.dumps({"staged": stage_items(req.get("items") or [])}, sort_keys=True,
                                   default=str) + "\n")
            proto.flush()
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
