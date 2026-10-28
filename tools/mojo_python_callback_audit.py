#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Static triage of Python hooks Mojo/native code can invoke on objects it is handed.

AGENTS.md "No Python in the runtime" (Andrew, 2026-10-20): native product code must not call
back into Python, including property/descriptor hooks and input protocols. A PythonObject
operation such as `x.tolist()`, `x.__array_interface__`, `x.shape`, `len(x)`, `sorted(x)`,
`getattr(x, ...)`, iteration or a `numbers.*` ABC check runs whatever Python the object's
class defines, unless the object was first admitted as an exact builtin/NumPy type (or was
created by the package itself).

This is a line-level source scan of bindings/, providers/ and core/ (*.mojo). It is a
triage list, not a proof in either direction:

  allowed_exact_type      the receiver is checked against an exact type earlier in the same
                          function (`py_type(x) is ...`, `type(x) is ...`, `*_exact(x)`,
                          `*CheckExact(x)`, `admit*(x)`)
  allowed_package_object  the receiver comes from the package's own context/tables or is
                          created natively in the function (context[...], Python.list(), ...)
  prohibited_user_object  the receiver comes from the entry's arguments (args[...],
                          from_borrowed=...) and no exact-type admission precedes the hook
  review_caller_admission the receiver is a helper parameter: whether it was admitted depends
                          on every caller
  review_helper_result    the receiver is what a package helper returned for the caller's object
                          (normally a checked or converted array; the helper is the place to audit)
  review                  anything else (numbers ABC references, unknown origin)

  python3 tools/mojo_python_callback_audit.py --out CALLBACKS.tsv [--summary]
"""
from __future__ import annotations

import argparse
import re
import sys
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
ROOTS = ("bindings", "providers", "core")

NAME = r"([A-Za-z_]\w*)"
OPS = [
    ("tolist", re.compile(NAME + r"(?:\[[^\]\n]*\])?\.tolist\(")),
    ("array_protocol", re.compile(NAME + r"(?:\[[^\]\n]*\])?\.__array__\b")),
    ("array_interface", re.compile(NAME + r"(?:\[[^\]\n]*\])?\.__array_interface__\b")),
    ("astype", re.compile(NAME + r"(?:\[[^\]\n]*\])?\.astype\(")),
    ("array_attribute", re.compile(NAME + r"\.(?:shape|dtype|ndim|strides|flags|itemsize)\b")),
    ("item", re.compile(NAME + r"\.item\(")),
    ("len", re.compile(r"\b(?:len|py_len|Python\.len)\(\s*" + NAME + r"\s*\)")),
    ("sorted", re.compile(r"\bsorted\(\s*" + NAME)),
    ("getattr", re.compile(r"(?<![\w.])(?:_?getattr|hasattr|py_getattr|py_hasattr)\(\s*" + NAME)),
    ("hook_by_name", re.compile(NAME + r"\s*,\s*\"(?:tolist|__array__|__array_interface__|__len__|__iter__|__getitem__|"
                                r"__index__|__float__|astype|item)\"")),
    ("iteration", re.compile(r"^\s*for\s+\w+\s+in\s+" + NAME + r"\s*:")),
    ("dunder_protocol", re.compile(NAME + r"\.(?:__len__|__iter__|__getitem__|__index__|__float__|__bool__)\(")),
    ("numbers_abc", re.compile(r"\bnumbers\.(?:Integral|Real|Number|Complex|Rational)\b()")),
]
# Ops that exist only on Python objects: reported whatever the receiver's inferred type.
PYTHON_ONLY = frozenset(("tolist", "array_protocol", "array_interface", "astype", "getattr", "hook_by_name",
                         "dunder_protocol", "numbers_abc"))
# Attribute-style ops on these receivers are Mojo values, not Python objects.
PY_TYPE = re.compile(r"\bPythonObject\b|\bPyObjectPtr\b")
DEF = re.compile(r"^(\s*)(?:def|fn)\s+(\w+)")
ASSIGN = re.compile(r"^\s*(?:var\s+|ref\s+)?(\w+)\s*(?::\s*[\w\[\], ]+)?\s*=\s*(.+)$")
PARAM = re.compile(r"(\w+)\s*:\s*(?:mut\s+|owned\s+|var\s+)?(PythonObject|PyObjectPtr)\b")
USER_SOURCE = re.compile(r"\bargs\b\s*\[|from_borrowed\s*=|args_ptr|kwargs\b")
PACKAGE_SOURCE = re.compile(r"\b(?:context|ctx|tables?|admission|_CONTEXT|shared)\b\s*[\[.]|Python\.(?:list|dict|tuple|import_module)\(|"
                            r"native_loaded_module\(|native_numpy_type\(|Python\(\)|PythonObject\(\s*(?:\d|\"|'|True|False|None)")
# `x = classical_check_x(args[0], ...)`: a package helper's result (checked, converted or
# rebuilt natively), not the caller's raw object
HELPER_CALL = re.compile(r"^\s*(?!py_call|py_get|getattr|Python\b|PythonObject\b|len\b|b\.|builtins\.)[A-Za-z_]\w*\(")
# the CPython METH_VARARGS argument tuple is an exact tuple built by the interpreter
ARGS_TUPLE = re.compile(r"PythonObject\(\s*from_borrowed\s*=\s*args(?:_ptr)?\s*\)")
# a builtin constructor or type slot called natively returns an exact builtin object
# (`py_call(b.list, x)`, `py_call(b.type.__dict__["__mro__"].__get__, ...)`); a hook the call
# itself runs on its argument is reported at that call, not on the result
EXACT_RESULT = re.compile(r"^\s*py_call\(\s*[\w\[\]().]*?\.(?:list|tuple|dict|set|frozenset|sorted|str|int|float|bool|len)\s*[,)]|"
                          r"type\.__dict__\[")
PY_SOURCE = re.compile(r"\bPythonObject\(|\bpy_\w+\(|\bPython\.|\bargs\b\s*\[|from_borrowed=")


def admitted(name, text):
    n = re.escape(name)
    # `py_type(x)` taken natively (compared directly or through an alias such as
    # `var cls = py_type(x); if cls is bi.list ...`), an *exact*/*CheckExact* or *admit* helper
    return re.search(r"py_type\(\s*" + n + r"\s*\)|\btype\(\s*" + n + r"\s*\)\s+is\b|"
                     r"\w*exact\w*\(\s*" + n + r"\b|\w*CheckExact\(\s*" + n + r"\b|\w*admit\w*\(\s*" + n + r"\b|"
                     r"\b" + n + r"\s+is\s+(?:None|True|False)\b", text) is not None


def owner_of(path):
    name = path.name.lower()
    rules = (
        (("token", "causal", "safetensor", "catalog", "model_import", "onnx", "importer", "byte_lm", "lm_",
          "sequence", "x_sequence", "autograd", "program", "session", "neural", "mlp", "rnn",
          "optim", "embedding", "samba", "mamba"), "peer3/peer4 (neural/sequence/ops)"),
        (("prep", "scaler", "encode", "impute", "spline", "lda", "minmax", "maxabs", "robust", "nb_", "_nb",
          "decomp", "pca", "tsvd", "manifold", "umap", "tsne", "mds", "isomap", "locally_linear", "sparse_pca", "dict_learn",
          "nmf", "fastica", "factor_analysis"), "peer2 (prep/decomp/manifold)"),
        (("neighbor", "knn", "ann", "ivf", "cagra", "svm", "svc", "density", "dbscan", "hdbscan", "hierarchy",
          "agglom", "spectral", "kmeans", "cluster", "lof"), "peer1 (neighbors/svm/cluster)"),
        (("tree", "forest", "gbdt", "boost", "bagging", "iforest", "gp_", "gpc", "explain", "catboost"),
         "peer5 (trees/ensembles/gp)"),
        (("arima", "ets", "theta", "croston", "prophet", "stl", "var_", "tsa", "resample", "metrics", "mx_"),
         "peer6 (time series/metrics)"),
        (("mode", "backend", "array", "numeric", "archive", "classical_attr", "public_attr", "params"),
         "peer4 (mode/backend/array/attributes)"),
        (("linear", "ridge", "lasso", "solver", "logistic", "sgd", "mcls"), "linear/classical owner (unassigned)"),
    )
    for keys, owner in rules:
        if any(k in name for k in keys):
            return owner
    return "unassigned"


def code_lines(text):
    """Lines with comments and docstrings blanked (line numbers kept)."""
    out, in_doc = [], None
    for line in text.splitlines():
        stripped = line.strip()
        if in_doc:
            out.append("")
            if in_doc in stripped:
                in_doc = None
            continue
        if stripped.startswith(('"""', "'''")):
            quote = stripped[:3]
            if stripped.count(quote) == 1 or stripped == quote:
                in_doc = quote
            out.append("")
            continue
        if stripped.startswith("#"):
            out.append("")
            continue
        # drop a trailing comment (good enough: string literals rarely carry '#  ')
        out.append(re.sub(r"\s+#\s.*$", "", line))
    return out


def scan_file(path, root):
    text = path.read_text(errors="replace")
    lines = code_lines(text)
    rows = []
    functions = []   # (start, end, name, indent)
    for i, line in enumerate(lines):
        m = DEF.match(line)
        if m:
            functions.append([i, len(lines), m.group(2), len(m.group(1))])
    for k, fn in enumerate(functions):
        for later in functions[k + 1:]:
            if later[3] <= fn[3]:
                fn[1] = later[0]
                break
    rel = path.relative_to(root).as_posix()
    owner = owner_of(path)
    for start, end, fname, _ in functions:
        body = lines[start:end]
        header = []
        j = 0
        while j < len(body):   # the signature can span lines
            header.append(body[j])
            if body[j].rstrip().endswith(":") and ")" in body[j]:
                break
            j += 1
        signature = " ".join(header)
        is_entry = 'abi("C")' in signature
        pointers = re.findall(r"(\w+)\s*:\s*PyObjectPtr", signature)
        args_tuple = re.compile(r"PythonObject\(\s*from_borrowed\s*=\s*(?:" +
                                "|".join(map(re.escape, pointers[1:2] or ["args_ptr"])) + r"|args|args_ptr)\s*\)")
        py_names, origin = {}, {}
        for m in PARAM.finditer(signature):
            py_names[m.group(1)] = True
            origin[m.group(1)] = "user" if is_entry else "param"
        changed = True
        while changed:
            changed = False
            for line in body[len(header):]:
                m = ASSIGN.match(line)
                if not m or m.group(1) in py_names:
                    continue
                name, value = m.group(1), m.group(2)
                sources = [n for n in py_names if re.search(r"\b" + re.escape(n) + r"\b", value)]
                if not (sources or PY_SOURCE.search(value) or PY_TYPE.search(line)):
                    continue
                py_names[name] = True
                if args_tuple.search(value) and is_entry:
                    origin[name] = "args_tuple"
                elif HELPER_CALL.match(value) and USER_SOURCE.search(value) or (
                        HELPER_CALL.match(value) and any(origin.get(s) in ("user", "param") for s in sources)):
                    origin[name] = "helper_result"
                elif EXACT_RESULT.search(value):
                    origin[name] = "package"
                elif USER_SOURCE.search(value):
                    origin[name] = "user"
                elif PACKAGE_SOURCE.search(value):
                    origin[name] = "package"
                elif sources:
                    kinds = {origin.get(s, "unknown") for s in sources}
                    origin[name] = ("user" if kinds & {"user", "args_tuple"} else "param" if "param" in kinds
                                    else "helper_result" if "helper_result" in kinds
                                    else "package" if kinds == {"package"} else "unknown")
                else:
                    origin[name] = "unknown"
                changed = True
        for offset, line in enumerate(body):
            if not line.strip():
                continue
            lineno = start + offset + 1
            for op, pattern in OPS:
                for m in pattern.finditer(line):
                    receiver = m.group(1)
                    if op == "numbers_abc":
                        receiver, where = "", "abc"
                    elif receiver not in py_names and op not in PYTHON_ONLY:
                        continue
                    else:
                        where = origin.get(receiver, "unknown")
                    before = "\n".join(body[:offset + 1])
                    ok = bool(receiver) and admitted(receiver, before)
                    if ok:
                        verdict = "allowed_exact_type"
                    elif where == "package":
                        verdict = "allowed_package_object"
                    elif where == "args_tuple" and op == "len":
                        verdict = "allowed_exact_type"
                    elif where == "helper_result":
                        verdict = "review_helper_result"
                    elif where == "user":
                        verdict = "prohibited_user_object"
                    elif where == "param":
                        verdict = "review_caller_admission"
                    else:
                        verdict = "review"
                    rows.append((rel, lineno, fname, op, receiver, where, "yes" if ok else "no", verdict, owner,
                                 line.strip()[:140]))
    return rows


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", type=Path, default=REPO)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--summary", action="store_true", help="print counts by verdict and owner")
    args = parser.parse_args(argv)
    rows = []
    for top in ROOTS:
        for path in sorted((args.root / top).rglob("*.mojo")):
            rows.extend(scan_file(path, args.root))
    header = ("file", "line", "function", "op", "receiver", "origin", "exact_admission", "classification",
              "owner", "source")
    text = "\t".join(header) + "\n" + "".join("\t".join(map(str, r)) + "\n" for r in rows)
    if args.out:
        args.out.write_text(text)
    else:
        sys.stdout.write(text)
    if args.summary or args.out:
        by_verdict = Counter(r[7] for r in rows)
        print("callback-audit:", len(rows), "rows;", dict(sorted(by_verdict.items())), file=sys.stderr)
        if args.summary:
            for (owner, verdict), n in sorted(Counter((r[8], r[7]) for r in rows).items()):
                print(f"  {owner}\t{verdict}\t{n}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
