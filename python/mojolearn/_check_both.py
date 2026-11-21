# SPDX-License-Identifier: Apache-2.0
"""One operation, two sides (GPU and CPU), one output: the shared check.

`python -m mojolearn verify --cross-check` runs this on the fixtures shipped
in the wheel; `python -m mojolearn verify --check MODEL --inputs X` and
`_verify_check.check(estimator, X)` run it on a user's estimator and data.
Both go through the same four functions, so neither has compare or print
code of its own:

    run_side(side, call)        run one call, time it, describe its outputs
    side_environment(ml, side)  what answered on that side (device, binaries)
    compare(gpu, cpu)           MATCH, or MISMATCH with the first differing
                                element
    format_*                    the lines of the report

`identical` mode only. `fast` makes no promise about bits, so a process or an
estimator in `fast` is refused before anything runs.

This is verification tooling: it runs around product calls and never inside
one. It imports nothing from the package at module level (it is handed the
imported package as `ml`), so the tool tests load it by path without a built
binding. NumPy is imported when the first output is described.

What a side record holds: the wall time of the call, per output its shape,
dtype, SHA-256 and first few values, and the ACTIVITY RECORD of the call:
what the runtime itself counted while the side ran (`activity_begin` /
`activity_end` below). The two sides print different work and the same
bytes. The counts come from the native runtime, at the shared doors it has:

    device launches, by site name   the named launch sites (the forest path)
    bytes to and from the device    the shared transfer entries
    host routines, by name          the named host inference entries
    host thread splits and tasks    the one host thread split

A count is what passed its door. A family whose device work passes none of
them reads "not counted on this path"; a number that was not counted is
never printed, and an install whose binaries do not have the record prints
no activity lines at all.

`check_estimator(..., fit_both=True)` fits on EACH side: the GPU side in this
process, the CPU side in a child process started with `MOJOLEARN_VENDOR=cpu`
(`python -m mojolearn._check_both DIRECTORY`, `child_main` below), for the
estimators whose fitted model cannot be handed to the CPU side.

AN ESTIMATOR THAT TAKES `device=` (`takes_device`: Ridge and LinearRegression
so far) needs neither a saved file nor a child process. Its two sides are the
same estimator constructed on each device in this process (`on_device`): the
fitted-model form hands the fitted attributes to the other device's estimator
in memory, and `fit_both` fits one estimator per device here. The side record
names the route under `route`.
"""
import hashlib
import json
import os
import subprocess
import sys
import time

FORMAT = "mojolearn.check-both.v1"
SIDES = ("gpu", "cpu")
#: values shown per output
HEAD_VALUES = 4
#: save method and file suffix an estimator may offer, in the order tried
SAVE_FORMS = (("save", ".npz"), ("save_checkpoint", ".json"))
#: the method checked when none is named: the first of these the estimator has
DEFAULT_METHODS = ("predict", "transform")

#: the host binding whose `activity_begin` / `activity_end` switch and read the
#: runtime's process-wide activity record (core/activity_record.mojo)
ACTIVITY_BINDING = "_mojolearn_core_host"
ACTIVITY_FORMAT = "mojolearn-activity-record/1"
#: names shown per activity line
TOP_NAMES = 4
#: fit_both without a row cap warns from this many training rows: the CPU fit
#: can take minutes there
FIT_BOTH_WARN_ROWS = 100_000
CHILD_FORMAT = "mojolearn.check-both-child.v1"

NO_GPU = ("no GPU in this installation, so there is no second piece of hardware to compare "
          "against. This is not a pass and not a failure: the check did not run.")


class NotChecked(Exception):
    """The check could not run; the message says why. Never a verdict."""


# --------------------------------------------------------------------------
# one side
# --------------------------------------------------------------------------

def _np():
    import numpy
    return numpy


def as_arrays(result):
    """The outputs of one call as a list of arrays: one array, or each member
    of a tuple or list of arrays (`kneighbors` answers two)."""
    np = _np()
    members = result if isinstance(result, (tuple, list)) else (result,)
    arrays = []
    for member in members:
        a = np.asarray(member)
        arrays.append(a if a.dtype == object else np.ascontiguousarray(a))
    return arrays


def _bytes(a):
    if a.dtype == object:
        # an object array's bytes are addresses; its values are what was answered
        return "\n".join(repr(x) for x in a.ravel().tolist()).encode("utf-8")
    return a.tobytes()


def describe_output(a):
    """shape, dtype, SHA-256 of the bytes and the first few values of one output."""
    flat = a.ravel()
    return dict(shape=[int(n) for n in a.shape], dtype=str(a.dtype),
                sha256=hashlib.sha256(_bytes(a)).hexdigest(),
                head=[repr(v) for v in flat[:HEAD_VALUES].tolist()])


def outputs_digest(arrays):
    """SHA-256 over the dtype, shape and bytes of every output, in order."""
    m = hashlib.sha256()
    for a in arrays:
        m.update(str(a.dtype).encode())
        m.update(str(tuple(a.shape)).encode())
        m.update(_bytes(a))
    return m.hexdigest()


# --------------------------------------------------------------------------
# the activity record: what the runtime counted while a side ran
# --------------------------------------------------------------------------

_ACTIVITY_COUNTS = ("launches", "upload_calls", "upload_bytes", "download_calls", "download_bytes",
                    "host_splits", "host_tasks", "host_entries", "names_dropped")
_DOORS = {}


def activity_door(ml):
    """The object whose `activity_begin()` / `activity_end()` switch and read
    the runtime's activity record, or None when this install has none (an
    older binary set). Found once per package."""
    key = id(ml)
    if key in _DOORS and _DOORS[key][2] is ml:
        return _DOORS[key][0]
    door, opened_here = None, False
    own = getattr(ml, "activity_door", None)       # a stub install hands its own
    try:
        if callable(own):
            door = own()
        else:
            full = "mojolearn._host." + ACTIVITY_BINDING
            opened_here = full not in sys.modules
            door = ml._backend.load_host_module(ACTIVITY_BINDING)
    except Exception:
        door = None
    if not (callable(getattr(door, "activity_begin", None)) and callable(getattr(door, "activity_end", None))):
        door = None
    # the package is held beside its door, so an id is never read for another
    _DOORS[key] = (door, opened_here and door is not None, ml)
    return door


def door_opened_here(ml):
    """True when `activity_door` itself loaded the binding that holds the
    door: that load says nothing about which binary answered a side."""
    entry = _DOORS.get(id(ml))
    return bool(entry and entry[2] is ml and entry[1])


def activity_begin(ml):
    """Start the runtime's activity record. False when this install has none."""
    door = activity_door(ml)
    if door is None:
        return False
    try:
        door.activity_begin()
    except Exception:
        return False
    return True


def _named(rows):
    out = []
    for row in rows or ():
        out.append([str(row["name"]), int(row["count"])])
    return sorted(out, key=lambda r: (-r[1], r[0]))


def activity_end(ml):
    """Stop the record and return what the runtime counted since
    `activity_begin`, or None when this install has none or the read failed."""
    door = activity_door(ml)
    if door is None:
        return None
    try:
        raw = door.activity_end()
        if raw.get("format") != ACTIVITY_FORMAT:
            return None
        record = {name: int(raw[name]) for name in _ACTIVITY_COUNTS}
        record["host_tasks_widest"] = int(raw["host_tasks_widest"])
        record["launch_names"] = _named(raw.get("launch_names"))
        record["host_names"] = _named(raw.get("host_names"))
    except Exception:
        return None
    return record


def activity_merge(total, more):
    """The sum of two activity records (either may be None)."""
    if not more:
        return total
    if not total:
        return dict(more, launch_names=[list(r) for r in more["launch_names"]],
                    host_names=[list(r) for r in more["host_names"]])
    merged = {name: total[name] + more[name] for name in _ACTIVITY_COUNTS}
    merged["host_tasks_widest"] = max(total["host_tasks_widest"], more["host_tasks_widest"])
    for key in ("launch_names", "host_names"):
        counts = {}
        for name, count in list(total[key]) + list(more[key]):
            counts[name] = counts.get(name, 0) + count
        merged[key] = sorted(([n, c] for n, c in counts.items()), key=lambda r: (-r[1], r[0]))
    return merged


def _size(n):
    for unit, scale in (("GB", 1 << 30), ("MB", 1 << 20), ("KB", 1 << 10)):
        if n >= scale:
            return f"{n / scale:.1f} {unit}"
    return f"{n} B"


def _top(rows):
    shown = ", ".join(f"{name} x{count:,}" for name, count in rows[:TOP_NAMES])
    more = len(rows) - TOP_NAMES
    return shown + (f" (+{more} more)" if more > 0 else "")


def activity_lines(side, activity):
    """The "what ran" lines of one side, at most four, from the runtime's
    activity record. None (an install without the record) gives no line."""
    if not activity:
        return []
    a, lines = activity, []
    if a["launches"]:
        lines.append(f"  launches  {a['launches']:,} device launches at named sites: {_top(a['launch_names'])}")
    if a["upload_calls"] or a["download_calls"]:
        lines.append(f"  copies    {_size(a['upload_bytes'])} to the device in {a['upload_calls']:,}, "
                     f"{_size(a['download_bytes'])} back in {a['download_calls']:,} (shared transfer entries)")
    if not (a["launches"] or a["upload_calls"] or a["download_calls"]):
        lines.append("  launches  not counted on this path (its device work passes no counted door)" if side == "gpu"
                     else "  launches  0 device launches, 0 bytes to the device, 0 bytes back")
    if a["host_entries"]:
        lines.append(f"  host      routines: {_top(a['host_names'])}")
    if a["host_splits"]:
        lines.append(f"  threads   {a['host_splits']:,} host thread splits, {a['host_tasks']:,} tasks, "
                     f"widest {a['host_tasks_widest']:,}")
    elif side == "cpu" and not a["host_entries"]:
        lines.append("  host      no counted host routine or thread split")
    return lines


def activity_line(side, activity):
    """One line for one lane of a table of them (verify's lanes)."""
    return "; ".join(line.split(None, 1)[1] for line in activity_lines(side, activity))


def run_side(side, call, digest=None, ml=None):
    """Run `call` once on the named side. Returns `(record, arrays)`.

    With `ml` (the imported package), the runtime's activity record is
    switched on around the call and what it counted is the record's
    `activity` (None when this install has no such record).

    `call` answers an array, a tuple of arrays, or a string. A string is a
    value the caller already reduced (verify's batch part answers one hash for
    many calls); it becomes the record's digest and there are no arrays.
    `digest`, when given, reduces the arrays to the record's digest (verify
    passes its harness's, so a cell keeps the value its records carry); the
    default is `outputs_digest`. The arrays are returned beside the record,
    for `compare` to find the first differing element; they are not part of
    the record."""
    if side not in SIDES:
        raise ValueError(f"side must be one of {SIDES}, not {side!r}")
    counting = ml is not None and activity_begin(ml)
    try:
        started = time.perf_counter()
        result = call()
        wall = time.perf_counter() - started
    finally:
        activity = activity_end(ml) if counting else None
    extra = dict(activity=activity) if ml is not None else {}
    if isinstance(result, str):
        return dict(side=side, wall_s=round(wall, 6), digest=result, outputs=[], **extra), None
    arrays = as_arrays(result)
    value = digest(*arrays) if digest is not None else outputs_digest(arrays)
    return dict(side=side, wall_s=round(wall, 6), digest=value,
                outputs=[describe_output(a) for a in arrays], **extra), arrays


def _binary_set(report, side):
    """The directory the side's loaded binaries came from, relative to the
    package, and how many of them are loaded."""
    rows = report.get("bindings") or {}
    host = side == "cpu"
    paths = sorted({row["path"] for name, row in rows.items()
                    if row.get("loaded") and row.get("path")
                    and name.startswith("mojolearn._host.") == host})
    base = report.get("package_dir") or ""
    dirs = sorted({os.path.dirname(p) for p in paths})
    shown = [os.path.relpath(d, base) if base and d.startswith(base) else d for d in dirs]
    return dict(directories=shown, loaded=len(paths))


def side_environment(ml, side, describe_device=None, cpu_model=None):
    """What answers on `side` in this process: the device (name and driver for
    the GPU; processor model and thread setting for the CPU), the wheel
    version, the numeric mode, and the binary set loaded for that side. Read
    it AFTER the side has run: a binding loads when it is first used."""
    if side not in SIDES:
        raise ValueError(f"side must be one of {SIDES}, not {side!r}")
    report = ml.runtime_report()
    env = dict(side=side, version=report.get("version"), mode=report.get("numeric_mode"),
               binaries=_binary_set(report, side))
    if side == "gpu":
        if describe_device is None:
            from ._verify import describe_device
        env.update(vendor=report.get("vendor"), gpu_arch=report.get("gpu_arch"),
                   code_kind=report.get("code_kind"), device=describe_device())
    else:
        if cpu_model is None:
            try:
                from ._verify_all import load_harness
                cpu_model = load_harness().cpu_model
            except Exception:
                # an install without the harness still names the side; the
                # model string is a label and nothing in the verdict reads it
                cpu_model = lambda: None
        asked = os.environ.get("MOJOLEARN_CPU_THREADS", "").strip()
        try:
            available = len(os.sched_getaffinity(0))
        except (AttributeError, OSError):
            available = os.cpu_count()
        env.update(device=cpu_model() or "unidentified processor",
                   cpu_threads_setting=asked or None, cpus_available=available)
    return env


# --------------------------------------------------------------------------
# two sides
# --------------------------------------------------------------------------

def _first_difference(gpu_arrays, cpu_arrays):
    np = _np()
    if len(gpu_arrays) != len(cpu_arrays):
        return dict(why="count", gpu=len(gpu_arrays), cpu=len(cpu_arrays))
    for k, (g, c) in enumerate(zip(gpu_arrays, cpu_arrays)):
        if g.dtype != c.dtype:
            return dict(why="dtype", output=k, gpu=str(g.dtype), cpu=str(c.dtype))
        if g.shape != c.shape:
            return dict(why="shape", output=k, gpu=[int(n) for n in g.shape], cpu=[int(n) for n in c.shape])
        if g.dtype == object:
            unequal = np.array([repr(a) != repr(b) for a, b in zip(g.ravel().tolist(), c.ravel().tolist())],
                               dtype=bool)
        elif g.dtype.itemsize == 0 or g.size == 0:
            continue
        else:
            # bytes, not values: -0.0 against 0.0 and two NaN payloads are differences
            size = g.dtype.itemsize
            unequal = (np.frombuffer(g.tobytes(), dtype=np.uint8).reshape(-1, size)
                       != np.frombuffer(c.tobytes(), dtype=np.uint8).reshape(-1, size)).any(axis=1)
        if unequal.any():
            flat = int(np.argmax(unequal))
            index = [int(i) for i in np.unravel_index(flat, g.shape)] if g.ndim else []
            return dict(why="value", output=k, flat=flat, index=index,
                        gpu=repr(g.ravel()[flat].item()), cpu=repr(c.ravel()[flat].item()))
    return None


def compare(gpu, cpu, gpu_arrays=None, cpu_arrays=None):
    """The verdict over two side records: MATCH when the two digests are the
    same string, else MISMATCH. With the arrays of both sides, a MISMATCH names
    the first differing element (output number, index, the two values), or the
    count, dtype or shape that differs."""
    if gpu.get("side") != "gpu" or cpu.get("side") != "cpu":
        raise ValueError("compare takes the gpu record, then the cpu record")
    match = isinstance(gpu.get("digest"), str) and gpu.get("digest") == cpu.get("digest")
    first = None
    if not match and gpu_arrays is not None and cpu_arrays is not None:
        first = _first_difference(gpu_arrays, cpu_arrays)
    elements = sum(_elements(o) for o in gpu.get("outputs", ()))
    return dict(verdict="MATCH" if match else "MISMATCH", match=match,
                outputs=len(gpu.get("outputs", ())), elements=elements, first=first)


def _elements(output):
    n = 1
    for d in output["shape"]:
        n *= d
    return n


def loaded_cpu_binaries(ml):
    """How many CPU binaries this process has loaded. A CPU side that ran with
    none loaded did not run on the CPU binaries, and comparing it would
    compare a thing to itself."""
    return _binary_set(ml.runtime_report(), "cpu")["loaded"]


def save_form(est):
    """(method name, file suffix) of the save an estimator offers, or None."""
    for name, suffix in SAVE_FORMS:
        if callable(getattr(est, name, None)):
            return name, suffix
    return None


def takes_device(est):
    """True when the estimator's class takes `device=` (it declares
    `_mojolearn_device`): its other side is the same estimator on the other
    device, in this process. The explicit host classes (`host_model`) are the
    CPU side already and are not."""
    cls = type(est)
    return bool(getattr(cls, "_mojolearn_device", False)) and not getattr(cls, "_HOST_INFERENCE_ONLY", False)


def on_device(est, device, fitted=True):
    """The same estimator on `device` ('gpu' or 'cpu'): `est` itself when it
    already runs there, else a new estimator of its class with its constructor
    parameters and `device=device`. With `fitted`, the new estimator is handed
    the fitted attributes of `est` (the same objects, in memory), so both
    answer from one fitted model.

    `device=None` on `est` is the installation's device. The check runs only
    where the installation has a GPU (a CPU-only one is refused before this),
    so None is the GPU here."""
    name = type(est).__name__
    try:
        params = dict(est.get_params(deep=False))
    except TypeError:
        params = dict(est.get_params())
    if "device" not in params:
        raise NotChecked(f"{name} declares device= but get_params does not return it")
    if (params["device"] or "gpu") == device:
        return est
    params["device"] = device
    try:
        twin = type(est)(**params)
    except Exception as exc:
        raise NotChecked(f"{name}(device={device!r}) cannot be built here: "
                         f"{type(exc).__name__}: {exc}"[:400]) from exc
    if fitted:
        # the instance dictionary is the estimator's state; object's own
        # lookup reads it without the estimator's attribute door
        state = object.__getattribute__(est, "__dict__")
        target = object.__getattribute__(twin, "__dict__")
        for key, value in state.items():
            if key not in params:
                target[key] = value
    return twin


def to_cpu(ml, est, directory, name="model", form=None):
    """The fitted model on the CPU side. For an estimator that takes
    `device=`: the same estimator constructed with `device="cpu"` and handed
    the fitted attributes in memory (no file; the returned path is None).
    For every other estimator: save it and load the file with
    `mojolearn.host_model`; an estimator with no save, or a file `host_model`
    does not serve, cannot cross and raises NotChecked with the reason.
    Returns `(cpu model, path)`."""
    if takes_device(est):
        return on_device(est, "cpu"), None
    form = form or save_form(est)
    if form is None:
        raise NotChecked(f"{type(est).__name__} has no save, so the CPU side has nothing to load")
    save, suffix = form
    path = os.path.join(directory, name + suffix)
    getattr(est, save)(path)
    return from_saved(ml, path), path


def from_saved(ml, path):
    """The CPU model for a saved file, through `mojolearn.host_model`."""
    try:
        return ml.host_model(path)
    except Exception as exc:
        raise NotChecked(f"the CPU side cannot load this model: {type(exc).__name__}: {exc}"[:400]) from exc


def refuse_fast(ml, est=None):
    """Raise NotChecked unless the process, and the estimator when it names a
    mode, are in `identical`."""
    loaded = ml.numeric_mode()
    if loaded != "identical":
        raise NotChecked(f"this process loaded the {loaded!r} binaries, which make no promise about "
                         "bits; the check runs in identical mode only")
    mode = None
    if est is not None:
        try:
            mode = getattr(est, "numeric_mode", None)
        except Exception:
            mode = None     # an estimator that does not name a mode runs in the process's
        if not isinstance(mode, str):
            mode = None
    if mode is not None and mode != "identical":
        raise NotChecked(f"{type(est).__name__} is set to numeric_mode={mode!r}, which makes no promise "
                         "about bits; the check runs in identical mode only")


def _cpu_answered(ml, cpu):
    """True when a CPU binary demonstrably answered the CPU side: one is
    loaded, and it is not only the binding this module loaded itself to read
    the activity record (unless that record shows host work)."""
    loaded = loaded_cpu_binaries(ml)
    if door_opened_here(ml):
        activity = cpu.get("activity") or {}
        if not (activity.get("host_splits") or activity.get("host_entries")):
            loaded -= 1
    return loaded > 0


def _route_of(est):
    """What the estimator itself says it runs on: its `device` parameter as
    given (None: this installation's device) and the vendor of the compiled
    binary it uses (`vendor_used`)."""
    try:
        used = est.vendor_used()
    except Exception as exc:
        used = f"unknown ({type(exc).__name__})"
    try:
        device = est.get_params(deep=False).get("device")
    except Exception:
        device = None
    return dict(device=device, vendor_used=used)


def _require_two_devices(ml, gpu, cpu):
    """Refuse a check whose two sides ran on one device: the CPU side must
    report the CPU binary and the GPU side must not, and a CPU binary must be
    loaded. Otherwise the comparison would compare a thing to itself."""
    g, c = gpu["route"]["vendor_used"], cpu["route"]["vendor_used"]
    if c != "cpu" or g == "cpu":
        raise NotChecked(f"the two sides did not run on two devices (gpu side: {g}, cpu side: {c}); "
                         "this would compare a thing to itself")
    if not _cpu_answered(ml, cpu):
        raise NotChecked("no CPU binary is loaded after the CPU side ran, so which binary answered "
                         "cannot be established; this would compare a thing to itself")


def _timed_fit_and_call(est, fit, method, X, walls):
    """`est.fit(*fit)` when asked, then `est.<method>(X)`; the two wall times
    go into `walls`."""
    if fit is not None:
        started = time.perf_counter()
        est.fit(*fit)
        walls["fit"] = round(time.perf_counter() - started, 6)
    started = time.perf_counter()
    result = getattr(est, method)(X)
    walls["call"] = round(time.perf_counter() - started, 6)
    return result


def estimator_spec(est):
    """What the CPU process needs to build the same unfitted estimator: the
    class name and the constructor parameters (`get_params`)."""
    name = type(est).__name__
    get = getattr(est, "get_params", None)
    if not callable(get):
        raise NotChecked(f"{name} has no get_params, so the CPU process cannot build the same estimator")
    try:
        params = get(deep=False)
    except TypeError:
        params = get()
    return dict(cls=name, params=dict(params))


def _cap_rows(fit, max_rows):
    """The fit arguments cut to their first `max_rows` rows (every array
    argument that has the first argument's row count)."""
    rows = len(fit[0])
    if not max_rows or rows <= max_rows:
        return fit, rows, rows
    return (tuple(a[:max_rows] if hasattr(a, "__len__") and len(a) == rows else a for a in fit),
            max_rows, rows)


def spawn_cpu_child(directory):
    """Run the CPU side of a fit-on-both-sides check in a child process that
    selects the CPU binaries (`MOJOLEARN_VENDOR=cpu`). Returns the exit code
    and the tail of what the child printed."""
    env = dict(os.environ, MOJOLEARN_VENDOR="cpu")
    proc = subprocess.run([sys.executable, "-m", "mojolearn._check_both", directory], env=env,
                          stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    return proc.returncode, (proc.stdout or "")[-600:]


def _cpu_side_in_child(spec, method, fit, X, spawn):
    """The CPU side record and arrays of a fit-on-both-sides check, from a
    child process. Raises NotChecked with the reason the side did not run
    (the product's own refusal message when the class has no CPU path)."""
    import tempfile
    np = _np()
    with tempfile.TemporaryDirectory(prefix="mojolearn_check_fit_") as tmp:
        try:
            for k, a in enumerate(fit):
                np.save(os.path.join(tmp, f"fit_{k}.npy"), np.asarray(a), allow_pickle=False)
            np.save(os.path.join(tmp, "X.npy"), np.asarray(X), allow_pickle=False)
        except Exception as exc:
            raise NotChecked("the fit arguments and the rows must be plain arrays to reach the CPU "
                             f"process: {type(exc).__name__}: {exc}"[:400]) from exc
        try:
            # numbers, strings, booleans, None and lists of them; a NumPy
            # scalar goes as its Python value
            request = json.dumps(dict(format=CHILD_FORMAT, cls=spec["cls"], method=method, fit_arrays=len(fit),
                                      params=spec["params"]), default=lambda v: v.item())
        except Exception as exc:
            raise NotChecked(f"the constructor parameters of {spec['cls']} cannot be sent to the CPU process "
                             f"(plain values only): {type(exc).__name__}: {exc}"[:400]) from exc
        with open(os.path.join(tmp, "request.json"), "w", encoding="utf-8") as stream:
            stream.write(request)
        started = time.perf_counter()
        code, tail = (spawn or spawn_cpu_child)(tmp)
        process_wall = round(time.perf_counter() - started, 3)
        path = os.path.join(tmp, "result.json")
        if not os.path.exists(path):
            raise NotChecked(f"the CPU process ended (exit {code}) without a result: {tail.strip()[-300:]}")
        with open(path, encoding="utf-8") as stream:
            answer = json.load(stream)
        if answer.get("format") != CHILD_FORMAT:
            raise NotChecked("the CPU process answered in another format (two versions of mojolearn?)")
        if not answer.get("ran"):
            raise NotChecked("the CPU side did not run: " + str(answer.get("reason")))
        record = answer["record"]
        record["process_wall_s"] = process_wall
        arrays = [np.load(os.path.join(tmp, f"out_{k}.npy"), allow_pickle=True)
                  for k in range(answer["outputs"])]
    return record, arrays


def child_main(directory, ml=None):
    """The CPU side of a fit-on-both-sides check, in its own process: build
    the estimator from the request, fit it, answer, and write the side record
    and the outputs back. Whatever the product raises (a class with no CPU
    path refuses by name) is written as the reason the side did not run."""
    np = _np()
    answer = dict(format=CHILD_FORMAT, ran=False)
    try:
        with open(os.path.join(directory, "request.json"), encoding="utf-8") as stream:
            request = json.load(stream)
        if ml is None:
            import mojolearn as ml
        if ml.vendor() != "cpu":
            raise NotChecked(f"the CPU process loaded the {ml.vendor()!r} binaries, not the CPU ones")
        refuse_fast(ml)
        params = request["params"]
        fit = tuple(np.load(os.path.join(directory, f"fit_{k}.npy"), allow_pickle=False)
                    for k in range(request["fit_arrays"]))
        X = np.load(os.path.join(directory, "X.npy"), allow_pickle=False)
        cls = getattr(ml, request["cls"], None)
        if cls is None:
            raise NotChecked(f"mojolearn has no class {request['cls']}")
        est = cls(**params)
        refuse_fast(ml, est)
        method, walls = request["method"], {}
        if not callable(getattr(est, method, None)):
            raise NotChecked(f"{request['cls']} has no method {method!r}")
        record, arrays = run_side("cpu", lambda: _timed_fit_and_call(est, fit, method, X, walls), ml=ml)
        record.update(wall_s=walls["call"], fit_wall_s=walls["fit"])
        record.update(side_environment(ml, "cpu"))
        for k, a in enumerate(arrays):
            np.save(os.path.join(directory, f"out_{k}.npy"), a)
        answer.update(ran=True, record=record, outputs=len(arrays))
    except Exception as exc:
        answer["reason"] = f"{type(exc).__name__}: {exc}"[:600] if not isinstance(exc, NotChecked) else str(exc)
    temporary = os.path.join(directory, "result.json.tmp")
    with open(temporary, "w", encoding="utf-8") as stream:
        json.dump(answer, stream)
    os.replace(temporary, os.path.join(directory, "result.json"))
    return 0


def check_estimator(ml, est, X, method=None, fit=None, saved=None, describe_device=None, cpu_model=None,
                    fit_both=False, max_rows=None, spawn=None):
    """Run `est.<method>(X)` on the GPU and on the CPU and compare the bits.

    The GPU side is the estimator as given. Two forms give the CPU side:

    - the fitted-model form (the default): the CPU side is the same fitted
      model, saved and loaded with `mojolearn.host_model` (or loaded from
      `saved`, a file the estimator was loaded from). `fit`, when given, is
      the tuple of arguments of `est.fit`: the model is fitted on the GPU
      first and the fitted model is what both sides answer from. Without
      `fit` the estimator must already be fitted.
    - `fit_both=True` (needs `fit`): the same fit runs once per side, the
      GPU side here and the CPU side in a child process that selects the CPU
      binaries, and each side answers from its own fitted model. This is the
      form for an estimator whose fitted model cannot be handed to the CPU
      side. `max_rows` cuts the training rows of both sides to their first
      `max_rows`; without it, a large fit carries a warning (the CPU fit can
      be slow).

    Returns a dictionary: `ran` False with a `reason` when nothing was
    compared (fast mode, no GPU, a model that cannot cross, a class the CPU
    side refuses); else the two side records under `sides` and the verdict
    under `verdict`."""
    import tempfile
    name = type(est).__name__
    result = dict(format=FORMAT, ran=False, estimator=name, method=method,
                  form="fit-both" if fit_both else "fitted-model")
    try:
        refuse_fast(ml, est)
    except NotChecked as exc:
        result.update(refused="fast", reason=str(exc))
        return result
    if ml.vendor() == "cpu":
        result.update(reason=NO_GPU)
        return result
    if method is None:
        method = next((m for m in DEFAULT_METHODS if callable(getattr(est, m, None))), None)
        result["method"] = method
    if method is None or not callable(getattr(est, method, None)):
        result.update(reason=f"{name} has no method {method!r}" if method else
                      f"{name} has neither predict nor transform; name the method to check")
        return result
    if fit_both and fit is None:
        result.update(reason="fit_both needs the fit arguments: there is nothing to fit on each side")
        return result
    device = takes_device(est)
    if device:
        result["transfer"] = "device"
    try:
        walls = {}
        if fit_both:
            spec = estimator_spec(est)       # the parameters as given, before the GPU fit
            fit, rows, given = _cap_rows(tuple(fit), max_rows)
            result.update(fit_rows=rows, fit_rows_given=given)
            if rows > FIT_BOTH_WARN_ROWS:
                result["warning"] = (f"fitting {rows:,} rows on the CPU as well; this can take long. "
                                     "max_rows (--fit-rows) fits both sides on the first rows only")
            if device:
                # one estimator per device, both fitted in this process
                gpu_est, cpu_est = on_device(est, "gpu", fitted=False), on_device(est, "cpu", fitted=False)
                gpu, gpu_arrays = run_side("gpu", lambda: _timed_fit_and_call(gpu_est, fit, method, X, walls),
                                           ml=ml)
                gpu.update(wall_s=walls["call"], fit_wall_s=walls["fit"], route=_route_of(gpu_est))
                cpu_walls = {}
                cpu, cpu_arrays = run_side("cpu", lambda: _timed_fit_and_call(cpu_est, fit, method, X, cpu_walls),
                                           ml=ml)
                cpu.update(wall_s=cpu_walls["call"], fit_wall_s=cpu_walls["fit"], route=_route_of(cpu_est))
                _require_two_devices(ml, gpu, cpu)
                cpu.update(side_environment(ml, "cpu", cpu_model=cpu_model))
            else:
                gpu, gpu_arrays = run_side("gpu", lambda: _timed_fit_and_call(est, fit, method, X, walls), ml=ml)
                gpu.update(wall_s=walls["call"], fit_wall_s=walls["fit"])
                cpu, cpu_arrays = _cpu_side_in_child(spec, method, fit, X, spawn)
        elif device:
            # the fitted-model form of an estimator that takes device=: the
            # fitted attributes are handed to the same estimator on the other
            # device, in memory; no file and no host_model
            if fit is not None:
                started = time.perf_counter()
                est.fit(*fit)
                walls["fit"] = round(time.perf_counter() - started, 6)
            gpu_est, cpu_est = on_device(est, "gpu"), on_device(est, "cpu")
            gpu, gpu_arrays = run_side("gpu", lambda: getattr(gpu_est, method)(X), ml=ml)
            gpu["route"] = _route_of(gpu_est)
            if fit is not None:
                gpu["fit_wall_s"] = walls["fit"]
            cpu, cpu_arrays = run_side("cpu", lambda: getattr(cpu_est, method)(X), ml=ml)
            cpu["route"] = _route_of(cpu_est)
            _require_two_devices(ml, gpu, cpu)
            cpu.update(side_environment(ml, "cpu", cpu_model=cpu_model))
        else:
            # the fit, when asked for, is not part of the compared call: the
            # activity record and the wall time are the method's on both sides
            if fit is not None:
                started = time.perf_counter()
                est.fit(*fit)
                walls["fit"] = round(time.perf_counter() - started, 6)
            gpu, gpu_arrays = run_side("gpu", lambda: getattr(est, method)(X), ml=ml)
            if fit is not None:
                gpu["fit_wall_s"] = walls["fit"]
            with tempfile.TemporaryDirectory(prefix="mojolearn_check_") as tmp:
                try:
                    host = from_saved(ml, saved) if saved else to_cpu(ml, est, tmp)[0]
                except NotChecked as exc:
                    raise NotChecked(str(exc) + ". fit_both=True (--fit-both) fits on each side instead "
                                     "and needs no saved model") from exc
                if not callable(getattr(host, method, None)):
                    raise NotChecked(f"the CPU model for {name} ({type(host).__name__}) has no {method}")
                cpu, cpu_arrays = run_side("cpu", lambda: getattr(host, method)(X), ml=ml)
            if not _cpu_answered(ml, cpu):
                raise NotChecked("no CPU binary is loaded after the CPU side ran, so which binary answered "
                                 "cannot be established; this would compare a thing to itself")
            cpu.update(side_environment(ml, "cpu", cpu_model=cpu_model))
    except NotChecked as exc:
        result.update(reason=str(exc))
        return result
    gpu.update(side_environment(ml, "gpu", describe_device=describe_device))
    result.update(ran=True, fitted_here=fit is not None, sides=dict(gpu=gpu, cpu=cpu),
                  verdict=compare(gpu, cpu, gpu_arrays, cpu_arrays))
    return result


# --------------------------------------------------------------------------
# the lines
# --------------------------------------------------------------------------

def environment_lines(env):
    """The lines that say what answered on one side."""
    binaries = env.get("binaries") or {}
    where = ", ".join(binaries.get("directories") or ()) or "none loaded"
    lines = [f"side      {env['side']}"]
    if env["side"] == "gpu":
        arch = "/".join(str(x) for x in (env.get("vendor"), env.get("gpu_arch"), env.get("code_kind")) if x)
        lines.append(f"  device    {env.get('device')} ({arch})")
    else:
        setting = env.get("cpu_threads_setting")
        threads = (f"MOJOLEARN_CPU_THREADS={setting}" if setting else
                   f"MOJOLEARN_CPU_THREADS unset, {env.get('cpus_available')} CPUs available")
        lines.append(f"  device    {env.get('device')} ({threads})")
    route = env.get("route")
    if route:
        given = route.get("device")
        shown = "device=None (this installation's device)" if given is None else f"device={given!r}"
        lines.append(f"  route     {shown} in this process, binary vendor {route.get('vendor_used')}")
    lines.append(f"  wheel     mojolearn {env.get('version')}, {env.get('mode')} mode")
    lines.append(f"  binaries  {where} ({binaries.get('loaded', 0)} loaded)")
    return lines


def output_lines(record, values=True):
    """The wall time and the outputs of one side record. `values=False`
    leaves the first values out, for a report that prints them side by side
    (`side_by_side_lines`)."""
    lines = []
    if "fit_wall_s" in record:
        lines.append(f"  fit       {record['fit_wall_s']:.6f} s")
    lines.append(f"  wall      {record['wall_s']:.6f} s")
    if "process_wall_s" in record:
        lines.append(f"  process   {record['process_wall_s']:.3f} s (the child process, start and import included)")
    for k, o in enumerate(record.get("outputs", ())):
        shape = "x".join(str(n) for n in o["shape"]) or "scalar"
        lines.append(f"  output {k}  shape {shape}  dtype {o['dtype']}  sha256 {o['sha256']}")
        if values:
            lines.append(f"            first values {', '.join(o['head'])}")
    if not record.get("outputs"):
        lines.append(f"  value     {record.get('digest')}")
    return lines


def side_by_side_lines(gpu, cpu):
    """The first values of every output, the GPU's line above the CPU's."""
    lines = ["first values, side by side"]
    for k in range(max(len(gpu.get("outputs", ())), len(cpu.get("outputs", ())))):
        label = f"  output {k}"
        for side, record in (("gpu", gpu), ("cpu", cpu)):
            outputs = record.get("outputs", ())
            head = ", ".join(outputs[k]["head"]) if k < len(outputs) else "(no such output)"
            lines.append(f"{label:<10}  {side}  {head}")
            label = ""
    return lines


def verdict_line(verdict):
    """MATCH, or MISMATCH with the first differing element."""
    if verdict["match"]:
        return (f"MATCH: {verdict['outputs']} output(s), {verdict['elements']} element(s), "
                "the same bytes on both sides")
    first = verdict.get("first")
    if not first:
        return "MISMATCH: the two sides answered different bytes"
    if first["why"] == "count":
        return f"MISMATCH: the sides answered a different number of outputs: gpu={first['gpu']} cpu={first['cpu']}"
    if first["why"] in ("dtype", "shape"):
        return (f"MISMATCH: output {first['output']} has a different {first['why']}: "
                f"gpu={first['gpu']} cpu={first['cpu']}")
    index = ", ".join(str(i) for i in first["index"])
    return (f"MISMATCH: output {first['output']}, first differing element at index [{index}] "
            f"(flat {first['flat']}): gpu={first['gpu']} cpu={first['cpu']}")


def pair_line(label, gpu, cpu, match, na=False):
    """One line for one compared value of a table of them (verify's cells)."""
    if na:
        return f"  {label} n/a  {gpu}"
    return f"  {label} gpu={gpu}  cpu={cpu}  {'agree' if match else 'DIFFER'}"


def sides_lines(sides):
    """The environment lines of both sides, GPU first."""
    lines = []
    for side in SIDES:
        if sides.get(side):
            lines.extend(environment_lines(sides[side]))
            lines.extend(activity_lines(side, sides[side].get("activity")))
    return lines


def format_check(result):
    """The report of `check_estimator`: the two sides, then one verdict line."""
    what = ".".join(str(x) for x in (result.get("estimator"), result.get("method")) if x)
    title = f"# mojolearn check: {what} on the GPU and on the CPU"
    if not result.get("ran"):
        head = "REFUSED" if result.get("refused") else "NOT RUN"
        return "\n".join([title, "", f"{head}: {result.get('reason')}"])
    lines = [title, ""]
    if result.get("form") == "fit-both":
        rows = (f" on the first {result['fit_rows']:,} of {result['fit_rows_given']:,} rows"
                if result.get("fit_rows") != result.get("fit_rows_given") else "")
        where = ("one estimator per device (this installation's GPU, device='cpu'), both in this process"
                 if result.get("transfer") == "device" else
                 "on the GPU in this process, on the CPU in a child process (MOJOLEARN_VENDOR=cpu)")
        lines += [f"Fitted on each side{rows}: {where}. Each side answers from its own fitted model.", ""]
        if result.get("warning"):
            lines += ["WARNING: " + result["warning"], ""]
    elif result.get("transfer") == "device":
        lines += [("Fitted on the given estimator; " if result.get("fitted_here") else "") +
                  "The same estimator was constructed on each device (this installation's GPU, device='cpu') in this "
                  "process and both answer from the one fitted model.", ""]
    elif result.get("fitted_here"):
        lines += ["Fitted on the GPU; the fitted model was saved and loaded on the CPU.", ""]
    for side in SIDES:
        record = result["sides"][side]
        lines.extend(environment_lines(record))
        lines.extend(activity_lines(side, record.get("activity")))
        lines.extend(output_lines(record, values=False))
    lines += [""] + side_by_side_lines(result["sides"]["gpu"], result["sides"]["cpu"])
    lines += ["", verdict_line(result["verdict"])]
    return "\n".join(lines)


if __name__ == "__main__":
    # the CPU side of a fit-on-both-sides check: `python -m mojolearn._check_both DIRECTORY`
    sys.exit(child_main(sys.argv[1]))
