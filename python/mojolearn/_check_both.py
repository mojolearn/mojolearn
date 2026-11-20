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

What a side record holds, and nothing more: the wall time of the call and,
per output, its shape, dtype, SHA-256 and first few values. There is no
per-iteration or per-kernel line. Device work (bytes uploaded, launches,
bytes downloaded) is NOT in the record: the runtime has no process-wide
counter for it (the only counters are the autograd program session's
`execution_profile_begin`/`_end`, sequence/execution_profile.mojo, which
cover program sessions and not the estimators), and a number that was not
measured is never printed.
"""
import hashlib
import os
import time

FORMAT = "mojolearn.check-both.v1"
SIDES = ("gpu", "cpu")
#: values shown per output
HEAD_VALUES = 4
#: save method and file suffix an estimator may offer, in the order tried
SAVE_FORMS = (("save", ".npz"), ("save_checkpoint", ".json"))
#: the method checked when none is named: the first of these the estimator has
DEFAULT_METHODS = ("predict", "transform")

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


def run_side(side, call, digest=None):
    """Run `call` once on the named side. Returns `(record, arrays)`.

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
    started = time.perf_counter()
    result = call()
    wall = time.perf_counter() - started
    if isinstance(result, str):
        return dict(side=side, wall_s=round(wall, 6), digest=result, outputs=[]), None
    arrays = as_arrays(result)
    value = digest(*arrays) if digest is not None else outputs_digest(arrays)
    return dict(side=side, wall_s=round(wall, 6), digest=value,
                outputs=[describe_output(a) for a in arrays]), arrays


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


def to_cpu(ml, est, directory, name="model", form=None):
    """The fitted model on the CPU side: save it and load the file with
    `mojolearn.host_model`. This is the state transfer between the sides; an
    estimator with no save, or a file `host_model` does not serve, cannot
    cross and raises NotChecked with the reason. Returns `(cpu model, path)`."""
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


def check_estimator(ml, est, X, method=None, fit=None, saved=None, describe_device=None, cpu_model=None):
    """Run `est.<method>(X)` on the GPU and on the CPU and compare the bits.

    The GPU side is the estimator as given. The CPU side is the same fitted
    model, saved and loaded with `mojolearn.host_model` (or loaded from
    `saved`, a file the estimator was loaded from). `fit`, when given, is the
    tuple of arguments of `est.fit`: the model is fitted on the GPU first and
    the fitted model is what both sides answer from. Without `fit` the
    estimator must already be fitted.

    Returns a dictionary: `ran` False with a `reason` when nothing was
    compared (fast mode, no GPU, a model that cannot cross); else the two side
    records under `sides` and the verdict under `verdict`."""
    import tempfile
    name = type(est).__name__
    result = dict(format=FORMAT, ran=False, estimator=name, method=method)
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
    try:
        fit_wall = None
        if fit is not None:
            started = time.perf_counter()
            est.fit(*fit)
            fit_wall = round(time.perf_counter() - started, 6)
        gpu, gpu_arrays = run_side("gpu", lambda: getattr(est, method)(X))
        if fit_wall is not None:
            gpu["fit_wall_s"] = fit_wall
        with tempfile.TemporaryDirectory(prefix="mojolearn_check_") as tmp:
            host = from_saved(ml, saved) if saved else to_cpu(ml, est, tmp)[0]
            if not callable(getattr(host, method, None)):
                raise NotChecked(f"the CPU model for {name} ({type(host).__name__}) has no {method}")
            cpu, cpu_arrays = run_side("cpu", lambda: getattr(host, method)(X))
        if not loaded_cpu_binaries(ml):
            raise NotChecked("no CPU binary is loaded after the CPU side ran, so which binary answered "
                             "cannot be established; this would compare a thing to itself")
    except NotChecked as exc:
        result.update(reason=str(exc))
        return result
    gpu.update(side_environment(ml, "gpu", describe_device=describe_device))
    cpu.update(side_environment(ml, "cpu", cpu_model=cpu_model))
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
    lines.append(f"  wheel     mojolearn {env.get('version')}, {env.get('mode')} mode")
    lines.append(f"  binaries  {where} ({binaries.get('loaded', 0)} loaded)")
    return lines


def output_lines(record):
    """The wall time and the outputs of one side record."""
    lines = []
    if "fit_wall_s" in record:
        lines.append(f"  fit       {record['fit_wall_s']:.6f} s")
    lines.append(f"  wall      {record['wall_s']:.6f} s")
    for k, o in enumerate(record.get("outputs", ())):
        shape = "x".join(str(n) for n in o["shape"]) or "scalar"
        lines.append(f"  output {k}  shape {shape}  dtype {o['dtype']}  sha256 {o['sha256']}")
        lines.append(f"            first values {', '.join(o['head'])}")
    if not record.get("outputs"):
        lines.append(f"  value     {record.get('digest')}")
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
    return lines


def format_check(result):
    """The report of `check_estimator`: the two sides, then one verdict line."""
    what = ".".join(str(x) for x in (result.get("estimator"), result.get("method")) if x)
    title = f"# mojolearn check: {what} on the GPU and on the CPU"
    if not result.get("ran"):
        head = "REFUSED" if result.get("refused") else "NOT RUN"
        return "\n".join([title, "", f"{head}: {result.get('reason')}"])
    lines = [title, ""]
    if result.get("fitted_here"):
        lines += ["Fitted on the GPU; the fitted model was saved and loaded on the CPU.", ""]
    for side in SIDES:
        record = result["sides"][side]
        lines.extend(environment_lines(record))
        lines.extend(output_lines(record))
    lines += ["", verdict_line(result["verdict"])]
    return "\n".join(lines)
