#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""One readable call line per race arm (lane harness-evidence, 2026-10-14).

From what a race recorded about each arm -- `measurement.invocation.implementation`
(module, name) with `effective_arguments`, or the cell's `params.implementation`
with `params.raw_params` -- render the call the arm made, for example

    cuml.KMeans(max_iter=300, n_clusters=8, tol=0.0001, ...)
    mojolearn.Adafactor(beta2_decay=-0.8, lr=0.001, p0=float32[16777216], ...)

Arrays are shown as dtype[shape], never their values. Display text only: the
call line is derived from the recorded invocation and is never an observation
(tools/race_fairness.py `pairing` ignores it). Standard library only.

    cell["call"] = call_line_of(cell)        # or set_call(cell)
"""
import json
import math

#: the longest call line rendered; longer argument lists end with ", ..."
MAX_CHARS = 240
#: the longest rendering of one argument value
MAX_VALUE_CHARS = 48


def _is_array_record(v):
    return isinstance(v, dict) and "dtype" in v and "shape" in v


def _shape_text(shape):
    try:
        return "x".join(str(int(s)) for s in shape) or "scalar"
    except (TypeError, ValueError):
        return str(shape)


def _array_text(dtype, shape):
    return "%s[%s]" % (str(dtype).replace("<", "").replace("'", ""), _shape_text(shape))


def format_value(v):
    """Readable text of one recorded argument. Arrays (records or live ndarrays/tensors) as dtype[shape]."""
    if _is_array_record(v):
        return _array_text(v.get("dtype"), v.get("shape") or [])
    shape, dtype = getattr(v, "shape", None), getattr(v, "dtype", None)
    if shape is not None and dtype is not None and not isinstance(v, (str, bytes)):
        try:
            if len(tuple(shape)) > 0:
                return _array_text(dtype, tuple(shape))
        except TypeError:
            pass
    if isinstance(v, bool) or v is None:
        return repr(v)
    if isinstance(v, float):
        if math.isfinite(v):
            return "%.6g" % v
        return repr(v)
    if isinstance(v, int):
        return str(v)
    if isinstance(v, str):
        text = repr(v)
    elif isinstance(v, (list, tuple)):
        if v and all(_is_array_record(x) for x in v):
            text = "[%s]" % ", ".join(format_value(x) for x in v)
        elif len(v) > 8:
            text = "%s[%d]" % (type(v).__name__, len(v))
        else:
            text = "[%s]" % ", ".join(format_value(x) for x in v)
            if isinstance(v, tuple):
                text = "(%s%s)" % (text[1:-1], "," if len(v) == 1 else "")
    elif isinstance(v, dict):
        if len(v) > 6:
            text = "{%d keys}" % len(v)
        else:
            text = "{%s}" % ", ".join("%s: %s" % (k, format_value(x)) for k, x in sorted(v.items(), key=str))
    else:
        text = str(v)
    if len(text) > MAX_VALUE_CHARS:
        text = text[:MAX_VALUE_CHARS - 3] + "..."
    return text


def _arguments(arguments):
    """[(name, text)]: scalar settings first (sorted), then the input arrays the call was handed."""
    if not isinstance(arguments, dict):
        return []
    settings, arrays = [], []
    for k in sorted(arguments, key=str):
        v = arguments[k]
        if k == "input_arrays":
            items = (list(v.items()) if isinstance(v, dict)
                     else [(x.get("name", "arg%d" % i), x) for i, x in enumerate(v or [])
                           if isinstance(x, dict)])
            arrays.extend((str(n), format_value(x)) for n, x in items)
            continue
        if v is None:
            continue
        (arrays if _is_array_record(v) else settings).append((str(k), format_value(v)))
    return settings + arrays


def _callee(implementation, library=None):
    impl = implementation or {}
    module, name = str(impl.get("module") or ""), str(impl.get("name") or "")
    if not name and not module:
        return None
    root = module.split(".")[0] if module else (library or "")
    root = root.lstrip("_") if root.startswith("_mojolearn") else root
    return ("%s.%s" % (root, name)) if root and name else (name or module)


def call_line(implementation, arguments, library=None, max_chars=MAX_CHARS):
    """`package.Name(k=v, ...)`, or None when no implementation was recorded."""
    callee = _callee(implementation, library)
    if not callee:
        return None
    parts, text = [], callee + "("
    for k, v in _arguments(arguments):
        piece = "%s=%s" % (k, v)
        if len(text) + len(", ".join(parts + [piece])) + 1 > max_chars:
            parts.append("...")
            break
        parts.append(piece)
    return text + ", ".join(parts) + ")"


def call_line_of(cell):
    """(call, source) of one board cell from its recorded invocation, or (None, None)."""
    m = (cell or {}).get("measurement") or {}
    inv = m.get("invocation") if isinstance(m, dict) else None
    if isinstance(inv, dict) and inv.get("implementation"):
        line = call_line(inv["implementation"], inv.get("effective_arguments"), cell.get("library"))
        if line:
            return line, "measurement.invocation"
    p = (cell or {}).get("params") or {}
    if isinstance(p, dict) and p.get("implementation"):
        args = p.get("raw_params") if isinstance(p.get("raw_params"), dict) else p.get("params")
        line = call_line(p["implementation"], args, p.get("library") or cell.get("library"))
        if line:
            return line, "params"
    return None, None


def set_call(cell, overwrite=False):
    """Store the call line on a cell: `call` on a fit cell, `constructor_call` on an inference cell (whose
    `call` already names what its clock ran). Existing values stay unless `overwrite`. Returns the line."""
    line, source = call_line_of(cell)
    if not line:
        return None
    key = "constructor_call" if cell.get("phase") == "infer" else "call"
    if overwrite or not cell.get(key):
        cell[key] = line
        cell["call_source"] = source
    return cell.get(key)


def arm_call_text(cell):
    """The text a board renders for one arm: the constructor call, then the inference call when it has one."""
    if cell.get("phase") == "infer":
        ctor, inf = cell.get("constructor_call"), cell.get("call")
        if ctor and inf:
            return "%s; %s" % (ctor, inf)
        return ctor or inf
    return cell.get("call")


def annotate_board(board):
    """set_call on every cell of every race of a board.json (in place); returns how many cells gained a line."""
    n = 0
    for section in ("races", "extra_races"):
        for rr in ((board or {}).get(section) or {}).values():
            for c in (rr.get("cells") or []) + (rr.get("infer_cells") or []):
                key = "constructor_call" if c.get("phase") == "infer" else "call"
                had = c.get(key)
                if set_call(c) and not had:
                    n += 1
    return n


if __name__ == "__main__":
    import sys
    for path in sys.argv[1:]:
        with open(path) as fh:
            b = json.load(fh)
        for section in ("races", "extra_races"):
            for rid, rr in sorted((b.get(section) or {}).items()):
                for c in rr.get("cells") or []:
                    line, _ = call_line_of(c)
                    print("%-48s %-22s %s" % (rid[:48], c.get("arm"), line))
