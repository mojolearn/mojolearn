#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""How close an imported model's outputs are to its source framework's (docs/plans/MODEL_IMPORT.md section 5).

A reference report, never a gate: torch and onnxruntime sum in another order, fuse differently and use
their own erf/exp/tanh, so their bits are not ours. The identity of an import is decided by the
IMPORT_DIGEST / PUBLIC_DIGEST lines across CPU, Apple, NVIDIA and AMD, not here. A result that looks like a
bug (a whole output off by orders of magnitude, the wrong class on every row) is a lowering error to fix,
not a tolerance to widen.

The tool runs the source model and mojolearn.import_model's Program on the same seeded inputs and prints,
per output, one line:

    IMPORT_COMPARE <output> shape=<s> max_abs=<a> max_rel=<r> beyond_1e-5=<n>/<total> first_diff=<i|none>
                   [argmax_agree=<fraction>]

`argmax_agree` (rows whose largest entry along the last axis is the same) is printed for a rank-2 output.

    # an ONNX file, onnxruntime as the reference
    python3 tools/import_compare.py model.onnx --shape x=8,3,32,32 [--out report.json]
    # a torch.export program: FILE.py defines build() -> (ExportedProgram, {input name: shape})
    python3 tools/import_compare.py --torch FILE.py [--out report.json]

Inputs: float inputs are seeded standard normals (float32); integer inputs (token ids) are seeded
integers in [0, --ids-high). torch and onnxruntime are optional and imported only here; the package and
its tests never import them. Exit status 0 whenever both sides ran, whatever the distances; 2 when a side
could not run (the reason printed).
"""
import argparse
import importlib.util
import json
import sys

REL_LIMIT = 1e-5


def _parse_shape(text):
    name, _, dims = text.partition("=")
    if not name or not dims:
        raise argparse.ArgumentTypeError(f"--shape takes NAME=D0,D1,...; got {text!r}")
    return name, tuple(int(d) for d in dims.split(","))


def _inputs(model, seed, ids_high):
    """Seeded arrays for the Program's inputs, by name, in its order."""
    import numpy as np
    rng = np.random.default_rng(seed)
    feeds = {}
    for name, shape, dtype in model.inputs:  # one array per model input
        if str(dtype).startswith(("int", "uint")):
            feeds[name] = rng.integers(0, ids_high, shape).astype(np.int64)
        else:
            feeds[name] = rng.standard_normal(shape, dtype=np.float32)
    return feeds


def _as_tuple(out):
    return out if isinstance(out, tuple) else (out,)


def compare(name, ours, ref):
    """Closeness of one output: a reference report (section 5), nothing asserted."""
    import numpy as np
    ours = np.asarray(ours, dtype=np.float32)
    ref = np.asarray(ref)
    row = {"output": name, "shape": list(ours.shape), "reference_shape": list(ref.shape)}
    if ours.shape != ref.shape:
        row["error"] = "shape differs"
        return row
    ref32 = ref.astype(np.float32)
    diff = np.abs(ours.astype(np.float64) - ref32.astype(np.float64))
    scale = np.maximum(np.abs(ref32.astype(np.float64)), np.finfo(np.float32).tiny)
    rel = diff / scale
    differs = np.flatnonzero(ours.reshape(-1).view(np.uint32) != ref32.reshape(-1).view(np.uint32))
    row.update(max_abs=float(diff.max(initial=0.0)), max_rel=float(rel.max(initial=0.0)),
               beyond=int((rel > REL_LIMIT).sum()), total=int(ours.size),
               first_diff=int(differs[0]) if differs.size else None)
    if ours.ndim == 2 and ours.shape[1] > 1:
        row["argmax_agree"] = float((ours.argmax(axis=1) == ref32.argmax(axis=1)).mean())
    return row


def _line(row):
    if "error" in row:
        return (f"IMPORT_COMPARE {row['output']} shape={row['shape']} reference_shape={row['reference_shape']}"
                f" error={row['error']!r}")
    first = "none" if row["first_diff"] is None else row["first_diff"]
    text = (f"IMPORT_COMPARE {row['output']} shape={'x'.join(map(str, row['shape']))} max_abs={row['max_abs']:.3e}"
            f" max_rel={row['max_rel']:.3e} beyond_1e-5={row['beyond']}/{row['total']} first_diff={first}")
    if "argmax_agree" in row:
        text += f" argmax_agree={row['argmax_agree']:.4f}"
    return text


def run_onnx(path, shapes, seed, ids_high):
    import mojolearn
    try:
        import onnxruntime as ort
    except ImportError:
        return None, "onnxruntime is not installed (pip install onnxruntime)"
    model = mojolearn.import_model(path, shapes=shapes)
    feeds = _inputs(model, seed, ids_high)
    ours = _as_tuple(model.run(*[feeds[n] for n, _, _ in model.inputs]))
    sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    names = [n for n, _, _ in model.outputs]
    refs = sess.run(names, feeds)
    return [compare(n, o, r) for n, o, r in zip(names, ours, refs)], None


def run_torch(file, seed, ids_high):
    import mojolearn
    try:
        import torch
    except ImportError:
        return None, "torch is not installed"
    spec = importlib.util.spec_from_file_location("import_compare_source", file)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    ep, shapes = mod.build()
    model = mojolearn.import_model(ep, shapes=shapes)
    feeds = _inputs(model, seed, ids_high)
    arrays = [feeds[n] for n, _, _ in model.inputs]
    ours = _as_tuple(model.run(*arrays))
    with torch.no_grad():
        refs = ep.module()(*[torch.from_numpy(a) for a in arrays])
    refs = refs if isinstance(refs, (tuple, list)) else (refs,)
    names = [n for n, _, _ in model.outputs]
    return [compare(n, o, r.detach().cpu().numpy()) for n, o, r in zip(names, ours, refs)], None


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("model", nargs="?", help="an ONNX file (onnxruntime is the reference)")
    ap.add_argument("--torch", metavar="FILE.py", help="a file whose build() returns (ExportedProgram, shapes)")
    ap.add_argument("--shape", action="append", type=_parse_shape, default=[],
                    help="NAME=D0,D1,... for an input axis the model leaves open (repeatable)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--ids-high", type=int, default=2, help="integer inputs are drawn from [0, ids-high)")
    ap.add_argument("--out", help="also write the rows as JSON here")
    a = ap.parse_args(argv)
    if bool(a.model) == bool(a.torch):
        ap.error("give an ONNX file or --torch FILE.py, not both")
    if a.torch:
        rows, why = run_torch(a.torch, a.seed, a.ids_high)
        source, reference = a.torch, "torch"
    else:
        rows, why = run_onnx(a.model, dict(a.shape) or None, a.seed, a.ids_high)
        source, reference = a.model, "onnxruntime"
    if rows is None:
        print(f"IMPORT_COMPARE not run: {why}")
        return 2
    import mojolearn
    for row in rows:  # one line per output
        print(_line(row))
    if a.out:
        doc = {"schema": "mojolearn-import-compare/1", "source": source, "reference": reference,
               "seed": a.seed, "vendor": mojolearn.vendor(), "numeric_mode": mojolearn.numeric_mode(),
               "rel_limit": REL_LIMIT, "gate": False, "rows": rows}
        with open(a.out, "w") as fh:
            json.dump(doc, fh, indent=1, sort_keys=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
