#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn
"""The plain form of a boundary file that reads through held name keys.

A converted door (bindings/py_keys.mojo) writes

    py_item(fields, nk[].k_alpha)          where it wrote   fields["alpha"]
    py_get(context, nk[].k_ndarray)                          py_get(context, "ndarray")
    operation == TSA_OPN_HW_FIT                              operation == "hw_fit"
    var operation = py_op_code(nk[].ops, call[1])            var operation = String(py=call[1])
    var operation = classical_operation_leaf(request[1])     var operation = String(py=request[1])
    def owner(model, operation: Int, ...)                    def owner(model, operation: String, ...)

The tools that read Mojo SOURCE to learn what a door reaches (tools/lane_select.py: the dispatcher's
branches per lane and the entry names a branch calls; tools/verify_coverage_audit.py: the entries an
operation's branch names; tools/native_caller_audit.py: dispatch arms and their callers) read the right-hand
column. `plain(path, text)` gives them that column back, line for line (no line is added or removed), so a
conversion changes no lane's closure, no branch's entries and no audit's arms, and the tools hold ONE
notion of a door's text. It is a reading aid for analysis only: nothing compiles the plain form.

Rules it relies on (the codemod and the recipe keep them):
  * a held key is always spelled `nk[].k_<name>`, and `<name>` is the Python key itself;
  * an operation code constant is `<FAMILY>_OPN_<NAME>` with `<NAME>` the operation name in upper case
    (operation names are lower-case identifiers), and the variable holding a code keeps the name the
    door's chain compared (`operation`, or `op`);
  * a family's table file is `bindings/<family>_keys_unpack.mojo`; it is returned unchanged (its fill lists
    every name of the family, so readers treat it as a manifest, never as a caller);
  * a family owner's operation-name file is `bindings/operation_names_<owner>.mojo` (the names and leaf codes
    `bindings/native_operation_codes.mojo` registers for an owner reached through a forwarding dispatcher,
    lane bw2-core): a manifest too, returned unchanged. Its constants are `<FAMILY>_OPN_<FULL NAME upper>`
    (`GP_OPN_FAMILY_GP_GPR_FIT` reads back as `"family_gp_gpr_fit"`).
Import statements are left as written.
"""
from __future__ import annotations

import re

KEYS_FILE = re.compile(r"(?:^|/)bindings/(?:[a-z0-9_]+_keys_unpack|operation_names_[a-z0-9_]+)\.mojo$")
_KEY = re.compile(r"\bnk\[\]\.k_(\w+)")
_ITEM = re.compile(r"\bpy_item\(")
_OPN = re.compile(r"\b[A-Z][A-Z0-9_]*?_OPN_([A-Z0-9_]+)\b")
_OP_CODE = re.compile(r"\bpy_op_code\(")
# A door's leaf of the caller's str by the operation table (bindings/classical_native_pack.mojo
# `classical_operation_leaf(name)`, lane bw2-classical): one argument, the caller's str.
_OP_LEAF = re.compile(r"\b\w*_operation_leaf\(")
_OP_PARAM = re.compile(r"\b(operation|op)\s*:\s*Int\b")


def is_keys_file(path):
    """True for a family's key table or an owner's operation-name file (manifests of names for the source
    readers)."""
    return bool(KEYS_FILE.search(path.replace("\\", "/")))


def _close(line, start):
    """Index of the `)` closing the `(` at line[start], or -1 (strings respected)."""
    depth, quote, k = 0, None, start
    while k < len(line):
        c = line[k]
        if quote:
            if c == "\\":
                k += 1
            elif c == quote:
                quote = None
        elif c in "\"'":
            quote = c
        elif c in "([{":
            depth += 1
        elif c in ")]}":
            depth -= 1
            if depth == 0:
                return k
        k += 1
    return -1


def _split(text):
    out, depth, quote, cur = [], 0, None, []
    k = 0
    while k < len(text):
        c = text[k]
        if quote:
            cur.append(c)
            if c == "\\" and k + 1 < len(text):
                k += 1
                cur.append(text[k])
            elif c == quote:
                quote = None
        elif c in "\"'":
            quote = c
            cur.append(c)
        elif c in "([{":
            depth += 1
            cur.append(c)
        elif c in ")]}":
            depth -= 1
            cur.append(c)
        elif c == "," and depth == 0:
            out.append("".join(cur))
            cur = []
        else:
            cur.append(c)
        k += 1
    out.append("".join(cur))
    return out


def _line(line):
    # py_item(X, nk[].k_name) -> X["name"], innermost first
    pos = 0
    while True:
        hit = None
        for m in _ITEM.finditer(line, pos):
            end = _close(line, m.end() - 1)
            if end < 0:
                continue
            args = _split(line[m.end():end])
            if len(args) == 2 and _KEY.fullmatch(args[1].strip()) and not _ITEM.search(args[0]):
                hit = (m.start(), end, args)
                break
        if hit is None:
            break
        start, end, args = hit
        name = _KEY.fullmatch(args[1].strip()).group(1)
        line = line[:start] + args[0].strip() + '["' + name + '"]' + line[end + 1:]
    line = _KEY.sub(lambda m: '"' + m.group(1) + '"', line)
    # var operation = py_op_code(nk[].ops, call[1]) -> var operation = String(py=call[1])
    while True:
        m = _OP_CODE.search(line)
        if not m:
            break
        end = _close(line, m.end() - 1)
        if end < 0:
            break
        args = _split(line[m.end():end])
        if len(args) != 2:
            break
        line = line[:m.start()] + "String(py=" + args[1].strip() + ")" + line[end + 1:]
    # var operation = classical_operation_leaf(request[1]) -> var operation = String(py=request[1])
    while True:
        m = _OP_LEAF.search(line)
        if not m or line[:m.start()].rstrip().endswith("def"):
            break
        end = _close(line, m.end() - 1)
        if end < 0:
            break
        args = _split(line[m.end():end])
        if len(args) != 1:
            break
        line = line[:m.start()] + "String(py=" + args[0].strip() + ")" + line[end + 1:]
    line = _OPN.sub(lambda m: '"' + m.group(1).lower() + '"', line)
    return _OP_PARAM.sub(lambda m: m.group(1) + ": String", line)


def plain(path, text):
    """`text` of the Mojo file `path` with held-key reads and integer operation codes written back as the
    by-name forms. Text that uses neither, a non-Mojo path and a key table are returned as they are."""
    if text is None or not path.endswith(".mojo") or is_keys_file(path):
        return text
    if "nk[].k_" not in text and "_OPN_" not in text and "py_op_code(" not in text and "_operation_leaf(" not in text:
        return text
    coded = "_OPN_" in text or "py_op_code(" in text or "_operation_leaf(" in text
    out, importing = [], False
    for line in text.split("\n"):
        stripped = line.lstrip()
        if importing:
            out.append(line)
            if ")" in line:
                importing = False
            continue
        if stripped.startswith(("from ", "import ")) and line[:1] not in " \t":
            out.append(line)
            importing = "(" in line and ")" not in line
            continue
        if "nk[].k_" in line or (coded and ("_OPN_" in line or "py_op_code(" in line or "_operation_leaf(" in line
                                             or ": Int" in line)):
            line = _line(line)
        out.append(line)
    return "\n".join(out)
