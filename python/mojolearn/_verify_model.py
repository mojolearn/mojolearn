# SPDX-License-Identifier: Apache-2.0
"""``python -m mojolearn verify-model``: a model's verification report.

A report names the install that ran the model (mojolearn version, numeric
mode, vendor, device), the model file (its SHA-256 and description hash),
every input (shape, dtype, digest) and every output (shape, digest). With an
expected report written on another install, every output is judged MATCH or
DIFFER: save a model on one GPU, verify it on another.

``Program.save``, ``Sequential.save_checkpoint`` and ``Sequential.export``
write the producing install's report beside the model as
``<file>.verify.json``, so the model file and its report are all a receiver
needs.
"""
# Engineering note: every number of a report comes from Mojo
# (autograd/model_verify.mojo through the x_sequence binding): the FNV-1a64
# identity digest of each input and output, the file's SHA-256, the model's
# kind and description hash, and the seeded inputs (Philox4x32-10, exact in
# float32). Python loads the model, passes buffers and formats the JSON.
from __future__ import annotations

import json
import os
import sys

FORMAT = "mojolearn-model-verify/1"
SIDECAR_SUFFIX = ".verify.json"
DEFAULT_SEED = 0
# rows of the seeded input of a Sequential model's report
SEQUENTIAL_ROWS = 16
# autograd/model_verify.mojo VERIFY_KIND_*
_PROGRAM = "program"
_SEQUENTIAL = ("sequential", "sequential-export")
# Sequential._forward's raw output (autograd/model.mojo OUT_RAW)
_OUT_RAW = 0

EXIT_MATCH, EXIT_DIFFER, EXIT_USAGE, EXIT_CANNOT_RUN = 0, 1, 2, 4


def sidecar_path(path):
    """The report path written beside a saved model."""
    return os.fspath(path) + SIDECAR_SUFFIX


def _np():
    from ._optional_numpy import require_numpy
    return require_numpy("mojolearn verify-model")


def _binding(numeric_mode=None):
    from . import _backend
    return _backend.binding("_mojolearn_x_sequence", numeric_mode)


def _digest(b, a, name):
    if not a.flags.c_contiguous:
        raise ValueError(f"{name} must be a C-contiguous array")
    # Mojo folds the array's bytes (FNV-1a64, the identity digest)
    return str(b.verify_digest(a.ctypes.data, a.nbytes))


def _seeded(b, shapes, seed):
    np = _np()
    arrays = []
    for k, shape in enumerate(shapes):  # glue: one buffer per model input
        a = np.empty(tuple(shape), dtype=np.float32)
        # Mojo writes the seeded values (input k is Philox stream k)
        b.verify_fill(a.ctypes.data, a.size, int(seed), k)
        arrays.append(a)
    return arrays


def _install(numeric_mode):
    from . import _backend
    from ._verify import describe_device
    try:
        from ._version import __version__ as version
    except Exception:  # pragma: no cover
        version = "unknown"
    try:
        vendor = _backend.vendor()
    except Exception as exc:
        vendor = f"unknown ({type(exc).__name__})"
    import platform
    return {
        "mojolearn_version": version,
        "numeric_mode": numeric_mode or _backend.default_mode(),
        "vendor": vendor or "unknown",
        "device": describe_device(),
        "platform": f"{platform.system()} {platform.machine()}",
    }


def _model_entry(b, path):
    sha, size = b.verify_file_sha256(os.fspath(path))  # cpu-route: model file hashed on the host
    kind, desc = b.verify_model_info(os.fspath(path))  # cpu-route: model file read on the host
    return {"file": os.path.basename(os.fspath(path)), "kind": str(kind), "sha256": str(sha),
            "bytes": int(size), "description_hash": str(desc)}


def _report(b, path, numeric_mode, names_in, arrays, source, seed, names_out, outs):
    inputs = []
    for k, (name, a) in enumerate(zip(names_in, arrays)):  # glue: one entry per model input
        inputs.append({"name": name, "shape": list(a.shape), "dtype": str(a.dtype),
                       "digest": _digest(b, a, f"input {name!r}")})
    outputs = []
    for name, y in zip(names_out, outs):  # glue: one entry per model output
        outputs.append({"name": name, "shape": list(y.shape), "digest": _digest(b, y, f"output {name!r}")})
    rep = {"format": FORMAT, "install": _install(numeric_mode), "model": _model_entry(b, path),
           "input_source": source, "inputs": inputs, "outputs": outputs}
    if source == "seeded":
        rep["seed"] = int(seed)
    return rep


def _write(rep, path):
    with open(path, "w") as fh:  # cpu-route: report file written on the host
        json.dump(rep, fh, indent=2, sort_keys=True)
        fh.write("\n")


def _names(entries, n, stem):
    if entries and len(entries) == n:
        return [str(e) for e in entries]  # glue: the model's input or output names
    return [f"{stem}{k}" for k in range(n)]  # glue: one label per model input or output


# ------------------------------------------------------------ at save time

def record_program(program, path):
    """Run ``program`` on the seeded inputs and write its report beside the
    file ``path`` that ``Program.save`` just wrote."""
    s = program._s
    b = s._b
    arrays = _seeded(b, program.input_shapes, DEFAULT_SEED)
    outs = s.run(program, *arrays)
    outs = outs if isinstance(outs, tuple) else (outs,)
    names_in = _names([e[0] for e in getattr(program, "inputs", ())], len(arrays), "input")  # glue: the model's input names
    names_out = _names([e[0] for e in getattr(program, "outputs", ())], len(outs), "output")  # glue: the model's output names
    rep = _report(b, path, s.numeric_mode, names_in, arrays, "seeded", DEFAULT_SEED, names_out, outs)
    _write(rep, sidecar_path(path))
    return rep


def record_sequential(model, path):
    """Run a fitted ``Sequential`` on the seeded inputs and write its report
    beside the file ``path`` that ``save_checkpoint`` or ``export`` just
    wrote."""
    b = model._binding()
    (X,) = _seeded(b, [(SEQUENTIAL_ROWS, model.n_features_in_)], DEFAULT_SEED)
    out = model._forward(X, _OUT_RAW)
    rep = _report(b, path, model.numeric_mode, ["X"], [X], "seeded", DEFAULT_SEED, ["output"], [out])
    _write(rep, sidecar_path(path))
    return rep


# ------------------------------------------------------------ at verify time

def _load_inputs(path, want_names, n):
    np = _np()
    with np.load(os.fspath(path), allow_pickle=False) as z:  # cpu-route: input file read on the host
        files = list(z.files)
        if len(files) != n:
            raise ValueError(f"{os.fspath(path)} holds {len(files)} arrays; the model takes {n} inputs")
        order = want_names if want_names and sorted(want_names) == sorted(files) else files
        return list(order), [z[k] for k in order]  # glue: one array per model input


def verify(path, *, inputs=None, expect=None, numeric_mode=None):
    """The report of the saved model at ``path`` run on this install.

    ``inputs`` is an ``.npz`` file (one array per model input) or None for
    the seeded inputs; ``expect`` a report dict whose seed, input shapes and
    names are reused when ``inputs`` is None."""
    from .ops import Session
    from ._x_sequence_model import Sequential
    b = _binding(numeric_mode)
    kind, _ = b.verify_model_info(os.fspath(path))  # cpu-route: model file read on the host
    kind = str(kind)
    exp_in = (expect or {}).get("inputs") or []
    exp_names_in = [e.get("name") for e in exp_in]  # glue: the expected report's input names
    exp_names_out = [e.get("name") for e in (expect or {}).get("outputs") or []]  # glue: the expected report's output names
    seed = int((expect or {}).get("seed", DEFAULT_SEED))
    if inputs is None and expect is not None and expect.get("input_source") != "seeded":
        raise ValueError("the expected report ran on inputs from a file: pass the same file with --inputs")
    if kind == _PROGRAM:
        with Session(numeric_mode=numeric_mode) as s:
            prog = s.load_program(path)
            n = len(prog.input_shapes)
            if inputs is None:
                arrays, names_in, source = _seeded(s._b, prog.input_shapes, seed), _names(exp_names_in, n, "input"), "seeded"
            else:
                names_in, arrays = _load_inputs(inputs, exp_names_in, n)
                source = "file"
            outs = prog.run(*arrays)
            outs = outs if isinstance(outs, tuple) else (outs,)
            names_out = _names(exp_names_out, len(outs), "output")
            return _report(s._b, path, numeric_mode, names_in, arrays, source, seed, names_out, outs)
    if kind in _SEQUENTIAL:
        model = Sequential.load(path, numeric_mode=numeric_mode)
        if inputs is None:
            rows = int(exp_in[0]["shape"][0]) if exp_in else SEQUENTIAL_ROWS
            arrays, names_in, source = _seeded(b, [(rows, model.n_features_in_)], seed), ["X"], "seeded"
        else:
            names_in, arrays = _load_inputs(inputs, exp_names_in, 1)
            source = "file"
        out = model._forward(arrays[0], _OUT_RAW)
        return _report(b, path, numeric_mode, names_in, arrays, source, seed, ["output"], [out])
    raise ValueError(f"{os.fspath(path)} is not a saved model")


def compare(report, expect):
    """``(verdict, rows, notes)``: each output's MATCH or DIFFER against
    ``expect``, and the reasons the comparison itself differs (another model
    file, other inputs). The verdict is MATCH only when the model file, every
    input and every output are the expected ones."""
    if expect.get("format") != FORMAT:
        raise ValueError(f"the expected report is not a {FORMAT} report")
    notes = []
    m, em = report["model"], expect.get("model", {})
    if m["sha256"] != em.get("sha256"):
        notes.append("the model file differs from the expected report's (sha256 "
                     f"{m['sha256'][:16]}... vs {str(em.get('sha256'))[:16]}...)")
    if m["description_hash"] != em.get("description_hash"):
        notes.append("the model description differs from the expected report's")
    ins, eins = report["inputs"], expect.get("inputs", [])
    if len(ins) != len(eins):
        notes.append(f"the model takes {len(ins)} inputs; the expected report has {len(eins)}")
    for k, (a, e) in enumerate(zip(ins, eins)):  # glue: one check per model input
        if a["shape"] != e.get("shape") or a["dtype"] != e.get("dtype") or a["digest"] != e.get("digest"):
            notes.append(f"input {a['name']!r} differs from the expected report's")
    rows = []
    outs, eouts = report["outputs"], expect.get("outputs", [])
    if len(outs) != len(eouts):
        notes.append(f"the model has {len(outs)} outputs; the expected report has {len(eouts)}")
    for k, y in enumerate(outs):  # glue: one verdict per model output
        e = eouts[k] if k < len(eouts) else {}
        same = y["shape"] == e.get("shape") and y["digest"] == e.get("digest")
        rows.append({"name": y["name"], "digest": y["digest"], "expected": e.get("digest"),
                     "verdict": "MATCH" if same else "DIFFER"})
    ok = not notes and all(r["verdict"] == "MATCH" for r in rows)  # glue: the per-output verdicts
    return ("MATCH" if ok else "DIFFER"), rows, notes


def _shape_text(shape):
    return "x".join(str(d) for d in shape)  # glue: format a shape


def _lines(rep, result, expect, expect_from):
    inst, m = rep["install"], rep["model"]
    out = [f"model    {m['file']}  kind={m['kind']}  sha256={m['sha256']}",
           f"         description={m['description_hash']}  bytes={m['bytes']}",
           f"install  mojolearn {inst['mojolearn_version']}  numeric_mode={inst['numeric_mode']}  "
           f"vendor={inst['vendor']}  device={inst['device']}"]
    src = f"seeded, seed {rep['seed']}" if rep["input_source"] == "seeded" else "from file"
    for a in rep["inputs"]:  # glue: one line per model input
        out.append(f"input    {a['name']}  shape={_shape_text(a['shape'])}  dtype={a['dtype']}  "
                   f"digest={a['digest']}  ({src})")
    verdicts = {} if result is None else {r["name"]: r for r in result[1]}  # glue: index the verdict rows
    for y in rep["outputs"]:  # glue: one line per model output
        line = f"output   {y['name']}  shape={_shape_text(y['shape'])}  digest={y['digest']}"
        r = verdicts.get(y["name"])
        if r is not None:
            line += f"  {r['verdict']}" + ("" if r["verdict"] == "MATCH" else f" (expected {r['expected']})")
        out.append(line)
    if result is not None:
        verdict, _, notes = result
        einst = expect.get("install", {})
        out.append(f"expected {expect_from}: mojolearn {einst.get('mojolearn_version')}  "
                   f"vendor={einst.get('vendor')}  device={einst.get('device')}")
        for note in notes:  # glue: one line per note
            out.append(f"note     {note}")
        out.append(f"verdict  {verdict}")
    return out


def cmd_verify_model(args):
    """`python -m mojolearn verify-model MODEL [--inputs F] [--expect R]`."""
    path = args.model
    if not os.path.isfile(path):
        print(f"verify-model: no file {path}", file=sys.stderr)
        return EXIT_USAGE
    expect_from = args.expect
    if expect_from is None and not args.no_sidecar and os.path.isfile(sidecar_path(path)):
        expect_from = sidecar_path(path)
    expect = None
    if expect_from is not None:
        try:
            with open(expect_from) as fh:  # cpu-route: report file read on the host
                expect = json.load(fh)
        except (OSError, ValueError) as exc:
            print(f"verify-model: cannot read the expected report {expect_from}: {exc}", file=sys.stderr)
            return EXIT_USAGE
    try:
        rep = verify(path, inputs=args.inputs, expect=expect, numeric_mode=args.numeric_mode)
        result = None if expect is None else compare(rep, expect)
    except (ValueError, TypeError) as exc:
        print(f"verify-model: {exc}", file=sys.stderr)
        return EXIT_USAGE
    except (RuntimeError, OSError) as exc:
        print(f"verify-model: cannot run the model: {exc}", file=sys.stderr)
        return EXIT_CANNOT_RUN
    if args.out:
        _write(rep, args.out)
    if args.json:
        payload = dict(rep)
        if result is not None:
            payload["comparison"] = {"expected": expect_from, "verdict": result[0], "outputs": result[1],
                                     "notes": result[2]}
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print("\n".join(_lines(rep, result, expect, expect_from)))
    if result is None:
        return EXIT_MATCH
    return EXIT_MATCH if result[0] == "MATCH" else EXIT_DIFFER


def add_parser(sub):
    p = sub.add_parser(
        "verify-model",
        help="run a saved model on this install and report (or check) its output bits",
        description=(
            "Load a saved Program, Sequential checkpoint or Sequential export, run it on this "
            "install and print its report: mojolearn version, numeric mode, vendor and device, the "
            "model file's SHA-256 and description hash, and the shape, dtype and digest of every "
            "input and output. With an expected report (--expect, or the <model>.verify.json the "
            "save wrote beside the model), every output is judged MATCH or DIFFER. "
            "Exit codes: 0 match (or no expected report), 1 differ, 2 usage, 4 cannot run."))
    p.add_argument("model", help="the saved model file")
    p.add_argument("--inputs", metavar="FILE.npz", default=None,
                   help="one array per model input (default: the seeded inputs of the expected report)")
    p.add_argument("--expect", metavar="REPORT.json", default=None,
                   help="a report written on another install (default: <model>.verify.json when present)")
    p.add_argument("--no-sidecar", action="store_true",
                   help="do not compare against <model>.verify.json")
    p.add_argument("--out", metavar="REPORT.json", default=None, help="also write this install's report here")
    p.add_argument("--json", action="store_true", help="print the report as JSON")
    p.add_argument("--numeric-mode", dest="numeric_mode", choices=("identical", "fast"), default=None,
                   help="numeric mode (default: the library default, identical)")
    p.set_defaults(func=cmd_verify_model)
    return p
