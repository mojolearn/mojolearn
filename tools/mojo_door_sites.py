#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Where the Mojo door layer touches Python, and what a public method's native entry reaches.

The rebuilt classifier of the 2026-11-09 static audit (py-audit-a-imports, method A),
kept in the repo so the public runtime inventory can follow a public method THROUGH its
native door instead of stopping at the first native call. Source only: nothing is built,
imported or run.

The door layer is the Mojo that may touch Python: bindings/ and providers/python/ (the
LAYERS table of tools/hooks/no_python_in_kernels.py; kernel files touch no Python, which
that checker enforces, so a walk stops at them).

Every door line that touches Python gets one class:

  a          object layer: argument, attribute and buffer reads, result building, C builtins
             and C-type methods (list.append, dict.get, ...)
  a-native   a compiled Mojo entry called through the CPython call protocol
             (`py_call(entries["x"], ...)`, `_native_function(...)`, `*_entry(...)`)
  b          a call into Python code: a Python-coded module (`numpy`, `os`, `mojolearn._mode`,
             ...), a numpy function, a value read off the caller's object and called (a user
             callable such as KMeans' `init`), a method on a caller-provided PythonObject
             parameter, a table entry the Python side fills with a Python def/lambda/class
  b?         a call into a callable this static reading cannot resolve (a local of unknown origin,
             a descriptor's `__get__`, a stored constructor): usually C, never proved either way
  b-table    `py_call(context["k"], ...)` whose key the Python side fills only with C builtins
             or compiled entries (or never: filled in Mojo)
  c-inline   Python iteration inside one expression (`b.sorted`, `b.map`, `.tolist`, `b.zip`,
             `PyObject_GetIter`/`PyIter_Next`, itertools)

and every `for`/`while` loop whose body touches Python is a (c) loop, with its tag
(driver / small / cpu-route / untagged), whether it iterates a Python object (pyiter) or a
Mojo range (mojoiter), and a data-sized hint (its header or tag names rows, labels, values,
elements, records, tokens, ...).

The index also records each module's exported entries (`def_function[fn]("name")`,
`def_py_c_function(fn, "name")`, `def_py_function[fn]("name")`), the string-keyed
operation blocks (`if operation == "family_x_fit":`), and each function's bare-name callees
(resolved through the file's own defs, then its `from bindings.x import y` imports, then a
unique door-wide name), so `reach(entry, key)` can walk from an entry into everything it
calls in the door layer.

  python3 tools/mojo_door_sites.py --summary            # per-class counts over the door layer
  python3 tools/mojo_door_sites.py --reach NAME[:KEY]   # what one entry (and operation key) reaches
"""
from __future__ import annotations

import argparse
import ast
import fnmatch
import os
import re
import sys
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DOOR_PATTERNS = ("bindings/*.mojo", "providers/python/*.mojo")
TOOLING = ("*/checks/*", "*/tests/*", "*/test_*.mojo")

# Modules implemented in C (or holding only C objects): reading them runs no Python code.
C_MODULES = frozenset({
    "builtins", "operator", "_operator", "sys", "array", "itertools", "math", "posix", "_thread",
    "mmap", "_struct", "_codecs", "_json", "_io", "gc", "time", "errno", "_functools", "_collections",
    "zlib", "binascii", "_random", "_hashlib", "_sha256", "_bisect", "_heapq", "cmath", "marshal",
    "_weakref", "atexit", "_signal", "select", "fcntl", "_locale", "_string", "_abc", "_warnings",
    # Python-coded, but it only holds C type objects (FunctionType, SimpleNamespace, ...): reading
    # it runs no Python code (no_runtime_import_module.py still counts it as a module read)
    "types",
})
# Methods of the C builtin types (list, dict, tuple, str, bytes, set, int, float, memoryview,
# array.array, slice): calling them is object-layer traffic, not Python code. This is the one
# name-based judgement here; it is limited to receivers that are not a known Python-coded object.
C_METHODS = frozenset({
    "append", "extend", "insert", "pop", "remove", "clear", "copy", "index", "count", "reverse", "sort",
    "get", "keys", "values", "items", "update", "setdefault", "popitem", "fromkeys",
    "join", "split", "rsplit", "strip", "lstrip", "rstrip", "partition", "rpartition", "startswith", "endswith",
    "encode", "decode", "replace", "find", "rfind", "lower", "upper", "format", "isdigit", "isidentifier",
    "add", "discard", "union", "intersection", "difference", "issubset", "issuperset",
    "bit_length", "to_bytes", "from_bytes", "is_integer", "hex", "tobytes", "frombytes", "cast", "release",
    "tolist", "itemsize", "fileno", "readinto", "read", "write", "seek", "tell", "close", "flush",
    "__getitem__", "__setitem__", "__len__", "__contains__", "__getattribute__", "__getattr__", "__index__",
    "__int__", "__float__", "__bool__", "__and__", "__or__", "__rshift__", "__lshift__", "__eq__", "__ne__",
    "__hash__", "__repr__", "__str__", "__reduce__", "__class__", "acquire", "release", "indices", "buffer_info",
    # Mojo-level methods of the PythonObject struct itself (no Python call at all)
    "steal_data", "unsafe_as_py_object_ptr", "downcast_value_ptr", "unchecked_downcast_value_ptr", "py_object_ptr",
})
# Names of C builtins and C operator functions that a builtins / operator handle exposes.
BUILTIN_NAMES = frozenset({
    "list", "dict", "tuple", "set", "frozenset", "int", "float", "complex", "str", "bytes", "bytearray", "bool",
    "slice", "memoryview", "range", "type", "object", "isinstance", "issubclass", "getattr", "setattr", "hasattr",
    "delattr", "len", "id", "hash", "repr", "abs", "min", "max", "sum", "divmod", "pow", "round", "iter", "next",
    "callable", "vars", "dir", "super", "property", "staticmethod", "classmethod", "ord", "chr", "format", "print",
    "eq", "ne", "lt", "le", "gt", "ge", "neg", "index", "add", "sub", "mul", "truediv", "floordiv", "mod",
    "getitem", "setitem", "contains", "itemgetter", "attrgetter", "methodcaller", "not_", "is_", "and_", "or_",
    "array", "SimpleNamespace", "BuiltinFunctionType", "MappingProxyType", "ModuleType", "Ellipsis", "NotImplemented",
    "__new__", "__init__", "__get__", "copy", "fromkeys", "frombytes", "zeros",
})
# A handle of a C module: `b`, `bi`, `_b()`, `msel_b()`, `p[].bi`, `env.b`, `xcnn_builtins()`,
# `mods.module("builtins")`, `native_loaded_module("operator")`, `op`.
BUILTIN_RECEIVER = re.compile(r"(?:^|[\].)\s])(?:_?b|bi|bt|op|ops|builtins|operator|\w+_b|\w*builtins|\w*_op)(?:\([\w, ]*\))?$|"
                              r"(?:module|native_loaded_module)\(\s*\"(?:%s|types)\"\s*\)$|"
                              r"^py_type\(\s*(?:Python\.(?:list|tuple|dict)\(\)|PythonObject\([^)]*\))\s*\)(?:\.__dict__\[\"\w+\"\])?$|"
                              r"^PythonObject\(\s*\"[^\"]*\"\s*\)$" % "|".join(map(re.escape, sorted(C_MODULES))))
# Python names that are C builtins (the `b.<name>` handles of native_loaded_module("builtins")).
BUILTIN_HANDLES = re.compile(r"^(?:b|bi|bt|builtins|_b|bs|trees_builtins\(\)|native_builtins\(\)|"
                             r"native_loaded_module\(\s*\"(?:%s)\"\s*\)|Python)\." % "|".join(map(re.escape, sorted(C_MODULES))))
# A compiled entry reached through the CPython call protocol (how audit A spelled them).
COMPILED_ENTRY = re.compile(r"^(?:[\w\[\]]+\.)?(?:entries\s*\[|entry\b|_entry\s*\(|\w+_entry\s*\(|\w+_entry\b|_native_function\s*\(|"
                            r"_x\s*\(|trees_helper\s*\(|trees_need\s*\(|callable\._obj_ptr|\w+_native_entry\b|\w*entries\s*\[)")
NUMPY = re.compile(r"\b(?:np|numpy)\s*\.\s*\w+\s*\(")
TABLE = re.compile(r"^(?P<table>\w+)\s*\[\s*\"(?P<key>[^\"]+)\"\s*\]")
PY_TOUCH = re.compile(r"\bPythonObject\b|\bPython\s*\.|\bpy_\w+\s*\(|\bnative_loaded_module\b|\bPyObject_\w+|"
                      r"\bPy[A-Z][A-Za-z]*_\w+|\bcpython\b|\bPythonModuleBuilder\b|\.to_python_object\b")
INLINE = re.compile(r"\b(?:b|bi|builtins|trees_builtins\(\))\s*\.\s*(?:sorted|map|zip|filter|enumerate|reversed)\b|"
                    r"\)\s*\.\s*(?:map|sorted|zip)\b|\.\s*tolist\b|\"tolist\"|\bPyObject_GetIter\b|\bPyIter_Next\b|"
                    r"\bitertools\b|\"(?:repeat|chain|starmap)\"|\b(?:b|bi|builtins)\s*\.\s*(?:max|min|all|any|sum)\b")
IMPORT_MODULE = re.compile(r"\bPython\s*\.\s*import_module\s*\(\s*\"(?P<m>[^\"]+)\"")
LOADED_MODULE = re.compile(r"\bnative_loaded_module\s*\(\s*(?P<arg>\"[^\"]*\"|[^)]*)\)")
PY_CALL = re.compile(r"\bpy_call\s*\(\s*")
# numpy's C types: constructing or reading them runs no Python code (numpy functions do).
NUMPY_C_TYPES = frozenset({"ndarray", "dtype", "generic", "number", "integer", "floating", "bool_", "str_",
                           "float32", "float64", "int32", "int64", "uint8", "int8", "int16", "uint16", "uint32", "uint64"})
DEF = re.compile(r"^(?P<indent>[ \t]*)(?:@\w+[^\n]*\n[ \t]*)*(?:def|fn)\s+(?P<name>\w+)")
LOOP = re.compile(r"^(?P<indent>[ \t]*)(?P<kw>for|while)\b(?P<head>.*)$")
IF_KEY = re.compile(r"^(?P<indent>[ \t]*)(?:el)?if\b(?P<cond>.*):\s*$")
KEY_EQ = re.compile(r"\b\w+\s*==\s*\"(?P<key>[A-Za-z_][\w.:-]*)\"")
IMPORT = re.compile(r"^\s*from\s+(?P<mod>(?:bindings|providers)(?:\.\w+)+)\s+import\s+(?P<names>.*)$")
REGISTER = re.compile(r"\.def_function\s*\[\s*(?P<fn>[\w.]+)(?:\[[^\]]*\])?\s*\]\s*\(\s*\"(?P<name>[^\"]+)\"|"
                      r"\.def_py_function\s*\[\s*(?P<fn2>[\w.]+)\s*\]\s*\(\s*\"(?P<name2>[^\"]+)\"|"
                      r"\.def_py_c_function\s*\(\s*(?P<fn3>[\w.]+)\s*,\s*\"(?P<name3>[^\"]+)\"")
CALLEE = re.compile(r"(?<![\w.\]])(?P<name>[A-Za-z_]\w*)\s*(?:\[[^\]\n]*\])?\s*\(")
TYPED = re.compile(r"\b(?P<name>\w+)\s*:\s*(?:mut\s+|var\s+|read\s+|owned\s+)?PythonObject\b")
ASSIGNED = re.compile(r"^\s*(?:var|ref)\s+(?P<name>\w+)\s*=\s*(?P<rhs>.+)$")
# A right-hand side that yields a PythonObject (used to type a local without an annotation).
PY_RHS = re.compile(r"^(?:py_(?:get|call|attr|fields|type|getattr|tuple|list|dict)\w*\s*\(|native_loaded_module\s*\(|"
                    r"Python\s*\.\s*(?:import_module|list|dict|tuple|none|evaluate)\s*\(|PythonObject\s*\()")
STRING = re.compile(r'"(?:\\.|[^"\\])*"')
TAG_DRIVER = re.compile(r"#\s*driver-loop\b")
TAG_SMALL = re.compile(r"#\s*small-loop\b")
TAG_CPU = re.compile(r"#\s*cpu-route\b")
# Words that make a loop data-sized: it iterates per row, label, value, ... of the input.
DATA_WORDS = re.compile(r"\b(?:rows?|samples?|labels?|values?|elements?|records?|tokens?|chars?|characters?|"
                        r"points?|observations?|targets?|items?|entries|coordinates?|segments?|strings?|"
                        r"n_samples|n_rows|n_labels|shape\[0\]|per[ -](?:row|label|element|value|record|token|item))\b",
                        re.IGNORECASE)
MOJO_KEYWORDS = frozenset({
    "if", "elif", "while", "for", "return", "raise", "and", "or", "not", "in", "is", "print", "var", "ref",
    "def", "fn", "struct", "trait", "with", "assert", "comptime", "alias", "try", "except", "else", "pass",
    "Int", "UInt", "Float32", "Float64", "Int32", "Int64", "UInt8", "UInt32", "UInt64", "Bool", "String",
    "List", "Dict", "Optional", "Span", "SIMD", "Scalar", "UnsafePointer", "Error", "abort", "len", "range",
    "PythonObject", "Python", "rebind", "materialize", "Pointer", "StringSlice", "StaticString", "Tuple",
})
LEVELS = ("native", "glue-heavy", "python-on-path")
GLUE_OBJECT_LINES = 400   # object-layer lines on one reached path above which a door is glue-heavy


def is_door(path):
    return any(fnmatch.fnmatchcase(path, p) for p in DOOR_PATTERNS) and not any(
        fnmatch.fnmatchcase(path, p) for p in TOOLING)


# ---------------------------------------------------------------- one file

def _code_and_comment(line):
    """(code without the trailing comment, comment text), keeping string literals in the code."""
    if "#" not in line:
        return line, ""
    if '"' not in line and "'" not in line:
        i = line.index("#")
        return line[:i], line[i:]
    quote, i = None, 0
    while i < len(line):
        c = line[i]
        if quote:
            if c == "\\":
                i += 2
                continue
            if c == quote:
                quote = None
        elif c in "\"'":
            quote = c
        elif c == "#":
            return line[:i], line[i:]
        i += 1
    return line, ""


def _logical_lines(text):
    """[(line number, raw line, code, comment)], docstring / triple-quoted lines blanked."""
    out, in_doc = [], None
    for n, raw in enumerate(text.splitlines(), 1):
        s = raw.strip()
        if in_doc:
            if in_doc in s:
                in_doc = None
            out.append((n, raw, "", ""))
            continue
        q = '"""' if '"""' in s else ("'''" if "'''" in s else None)
        if q and (s.startswith(q) or s.startswith(("r" + q, "b" + q))):
            if s.count(q) % 2 == 1:
                in_doc = q
            out.append((n, raw, "", ""))
            continue
        code, comment = _code_and_comment(raw)
        out.append((n, raw, code, comment))
    return out


def _indent(raw):
    return len(raw) - len(raw.lstrip(" \t"))


def _call_target(code, start):
    """The text of the first argument of a call whose '(' ends at `start`."""
    depth, i = 0, start
    while i < len(code):
        c = code[i]
        if c in "([{":
            depth += 1
        elif c in ")]}":
            if depth == 0:
                return code[start:i].strip()
            depth -= 1
        elif c == "," and depth == 0:
            return code[start:i].strip()
        elif c == '"':
            j = code.find('"', i + 1)
            i = j if j > 0 else len(code)
        i += 1
    return code[start:].strip()


# Locals that hold the caller's own objects: a value read off them and called is user code.
CALLER_NAMES = frozenset({"model", "estimator", "self_", "obj", "user", "callback", "fields", "params", "py_self"})
CALLER_READ = re.compile(r"^(?:py_get|py_getattr|py_attr|native_fitted_get|py_get_item)\s*\(\s*(?P<recv>\w+)\b|"
                         r"^(?P<recv2>\w+)\s*(?:\[\s*\"|\.\s*(?!__)\w+\s*$)")


_NAMES_RE = {}
_IDENT_USE = re.compile(r"(?<![\w.])(\w+)\s*[.\[(]")
_IN_NAME = re.compile(r"\bin\s+(\w+)\b")


def _names_regex(names, template):
    key = (names, template)
    r = _NAMES_RE.get(key)
    if r is None:
        r = _NAMES_RE[key] = re.compile(template % "|".join(sorted(map(re.escape, names))))
    return r


class Names:
    """The PythonObject-typed names of one function, with one compiled alternation."""
    __slots__ = ("names", "_re", "params")

    def __init__(self, names=()):
        self.names, self._re = set(names), None
        self.params = frozenset(names)

    def add(self, *names):
        new = [n for n in names if n not in self.names]
        if new:
            self.names.update(new)
            self._re = None

    def regex(self):
        if self._re is None:
            self._re = _names_regex(frozenset(self.names), r"(?<![\w.])(?:%s)\s*[.\[(]") if self.names else False
        return self._re

    def in_regex(self):
        return _names_regex(frozenset(self.names), r"\bin\s+(?:%s)\b") if self.names else None

    def used(self, code):
        if not self.names:
            return False
        return any(m.group(1) in self.names for m in _IDENT_USE.finditer(code))

    def iterated(self, head):
        return any(m.group(1) in self.names for m in _IN_NAME.finditer(head))


def classify_line(code, names=None, python_tables=None, origins=None, call_re=None):
    """[(class, what)] for one door code line (comments stripped). Empty when it touches no Python.
    `origins` maps a local name to the (class, what) of the expression it was bound to, so a
    call through a local (`var list_type = py_type(Python.list())`) is classed by its origin."""
    names = names if names is not None else Names()
    bare = STRING.sub('""', code)          # string contents never read as code
    sites = []
    if not (PY_TOUCH.search(bare) or names.used(bare) or INLINE.search(code) or NUMPY.search(bare)
            or (call_re is not None and call_re.search(bare))):
        return sites
    for m in IMPORT_MODULE.finditer(code):
        if m["m"] not in C_MODULES:
            sites.append(("b", f"imports Python module {m['m']}"))
    for m in LOADED_MODULE.finditer(code):
        arg = m["arg"].strip()
        name = arg[1:-1] if arg[:1] == '"' else None
        if name is None:
            sites.append(("b", f"native_loaded_module({arg[:40]}) of a module named at run time"))
        elif name not in C_MODULES:
            sites.append(("b", f"reads Python module {name}"))
    m = NUMPY.search(bare)
    if m:
        sites.append(("b", "numpy callable: " + m.group(0)[:40]))
    m = INLINE.search(code)
    if m:
        sites.append(("c-inline", "inline Python iteration: " + m.group(0)[:40]))
    calls = [mm.end() for mm in (call_re or PY_CALL).finditer(code)]
    for start in calls:
        target = _call_target(code, start)
        if origins and target in origins:
            sites.append(origins[target])
        else:
            sites.append(_classify_target(target, python_tables))
    if names.names:
        for mm in re.finditer(r"(?<![\w.])(?P<recv>\w+)\s*\.\s*(?P<attr>\w+)\s*\(", bare):
            if (mm["recv"] in names.names and mm["attr"] not in C_METHODS
                    and not BUILTIN_HANDLES.match(mm["recv"] + ".")):
                caller = mm["recv"] in getattr(names, "params", ()) or mm["recv"] in CALLER_NAMES
                sites.append(("b" if caller else "b?",
                              f"method call {mm['recv']}.{mm['attr']}(...) on a "
                              f"{'caller-provided ' if caller else ''}PythonObject"))
    if not sites:
        sites.append(("a", "object layer"))
    return sites


def _classify_target(target, python_tables=None):
    t = target.strip()
    if BUILTIN_HANDLES.match(t):
        return ("a", "C builtin " + t[:40])
    chain = re.sub(r'\.__dict__\["(\w+)"\]', r'.\1', t)
    pa = re.match(r'^py_attr\(\s*(?P<recv>[^,]+?)\s*,\s*"(?P<attr>\w+)"\s*\)$', chain)
    if pa:
        chain = pa["recv"] + "." + pa["attr"]
    head, dot, attr = chain.rpartition(".")
    while dot and (attr in BUILTIN_NAMES or attr in C_METHODS or re.fullmatch(r"__\w+__", attr)):
        if BUILTIN_RECEIVER.search(head.strip()):
            return ("a", "C builtin " + t[:40])
        head, dot, attr = head.rpartition(".")
    if BUILTIN_RECEIVER.search(t) and t.startswith("py_type("):
        return ("a", "C builtin type " + t[:40])
    if t.startswith("itertools.") or ".itertools." in t:
        return ("c-inline", "per-element Python iteration via " + t[:40])
    if COMPILED_ENTRY.match(t):
        return ("a-native", "compiled entry via CPython call: " + t[:40])
    if re.match(r"^native_numpy_(?:type|dtype)\s*\(", t):
        return ("a", "numpy C type " + t[:40])
    m = TABLE.match(t)
    if m:
        if m["key"] in NUMPY_C_TYPES:
            return ("a", f"numpy C type {m['table']}[\"{m['key']}\"]")
        if m["key"] in ("np", "numpy"):
            return ("b", f"numpy callable: {m['table']}[\"{m['key']}\"]")
        if python_tables is not None and m["key"] in python_tables:
            return ("b", f"table entry {m['key']} filled with Python code ({python_tables[m['key']]})")
        return ("b-table", f"table entry {m['key']} (C builtin, compiled entry or Mojo-filled)")
    dotted = re.match(r"^(?P<recv>[\w\[\]\".()]+?)\s*\.\s*(?P<attr>\w+)$", t)
    if dotted and dotted["attr"] in C_METHODS:
        return ("a", "C-type method " + dotted["attr"])
    return ("b?", "py_call into " + t[:50])


class Func:
    __slots__ = ("path", "name", "start", "end", "sites", "loops", "callees", "blocks", "lines")

    def __init__(self, path, name, start):
        self.path, self.name, self.start, self.end = path, name, start, start
        self.sites = []      # (line, class, what)
        self.loops = []      # (line, tag, pyiter, data_sized, header)
        self.callees = []    # (line, name)
        self.blocks = []     # (start, end, frozenset(keys))
        self.lines = 0

    @property
    def key(self):
        return (self.path, self.name)


CALL_HELPER = re.compile(r"^[ \t]*(?:def|fn)\s+(?P<name>\w+)\s*\(\s*(?P<param>\w+)\s*:\s*(?:read\s+|owned\s+)?PythonObject\b",
                         re.MULTILINE)


def call_helpers(text):
    """Door functions that call their first PythonObject parameter (`family_call(callable, *args)`):
    a call to one is classed like `py_call(callable, ...)`."""
    found = set()
    for m in CALL_HELPER.finditer(text):
        body = text[m.end():m.end() + 1500]
        nxt = re.search(r"^(?:def|fn|struct)\s", body, re.MULTILINE)
        body = body[:nxt.start()] if nxt else body
        if re.search(r"(?:\bpy_call\w*|PyObject_Call\w*|PyObject_Vectorcall\w*)\s*\(\s*%s\b" % re.escape(m["param"]), body):
            found.add(m["name"])
    return found


def call_regex(helpers):
    return _names_regex(frozenset({"py_call"} | set(helpers)), r"(?<![\w.])(?:%s)\s*\(\s*")


def scan_file(path, text, python_tables=None, call_re=None):
    """(funcs, imports {alias: (module path, name)}, registrations [(python name, fn name)])."""
    lines = _logical_lines(text)
    imports, registrations = {}, []
    flat = re.sub(r"\s*\n\s*", " ", "\n".join(l[2] for l in lines))
    for m in REGISTER.finditer(flat):
        fn = m["fn"] or m["fn2"] or m["fn3"]
        name = m["name"] or m["name2"] or m["name3"]
        registrations.append((name, fn.rsplit(".", 1)[-1]))
    for _, _, code, _ in lines:
        m = IMPORT.match(code)
        if not m:
            continue
        mod = m["mod"].replace(".", "/") + ".mojo"
        for part in m["names"].strip().strip("()").split(","):
            bits = part.split()
            if not bits:
                continue
            alias = bits[2] if len(bits) == 3 and bits[1] == "as" else bits[0]
            imports[alias] = (mod, bits[0])
    funcs, stack = [], []   # stack of (indent, Func)
    py_names_of, origins_of = {}, {}
    for idx, (n, raw, code, comment) in enumerate(lines):
        if not code.strip():
            continue
        ind = _indent(raw)
        while stack and stack[-1][0] >= ind:
            stack.pop()
        m = re.match(r"^(?P<indent>[ \t]*)(?:def|fn)\s+(?P<name>\w+)", code)
        if m:
            f = Func(path, m["name"], n)
            funcs.append(f)
            stack.append((ind, f))
            py_names_of[id(f)] = Names(TYPED_NAMES(code))
            origins_of[id(f)] = {}
            continue
        if not stack:
            continue
        for _ind, f in stack:
            f.end = n
            f.lines += 1
        f = stack[-1][1]
        names = py_names_of[id(f)]
        origins = origins_of[id(f)]
        names.add(*TYPED_NAMES(code))
        a = ASSIGNED.match(code) or re.match(r"^\s*(?P<name>\w+)\s*=\s*(?P<rhs>[^=].*)$", code)
        if a:
            rhs = a["rhs"].strip()
            cr = CALLER_READ.match(rhs)
            recv = cr and (cr["recv"] or cr["recv2"])
            if recv and (recv in names.params or recv in CALLER_NAMES) and recv not in ("context", "ctx", "mods"):
                origins[a["name"]] = ("b", f"calls a value read off the caller's object ({rhs[:40]})")
            elif not rhs.startswith(("py_call(", "py_get(")) and not re.search(r"\)\s*$", rhs) or rhs.startswith(
                    ("py_type(", "native_numpy_type(")):
                origin = _classify_target(rhs, python_tables)
                if origin[0] == "b?":
                    origins.pop(a["name"], None)
                else:
                    origins[a["name"]] = origin
            else:
                origins.pop(a["name"], None)
            head = re.match(r"(?P<recv>\w+)\s*[.\[]", rhs)
            if PY_RHS.match(rhs) or (head and head["recv"] in names.names and not re.match(r"\w+\s*\.\s*__\w+__", rhs)):
                names.add(a["name"])
        for cls, what in classify_line(code, names, python_tables, origins, call_re):
            f.sites.append((n, cls, what))
        for c in CALLEE.finditer(code):
            name = c["name"]
            if name not in MOJO_KEYWORDS and not name[0].isupper():
                f.callees.append((n, name))
        lm = LOOP.match(code)
        if lm:
            end, touches = n, False
            prev_comment = lines[idx - 1][3] if idx else ""
            for n2, raw2, code2, _ in lines[idx + 1:]:
                if not code2.strip():
                    continue
                if _indent(raw2) <= ind:
                    break
                end = n2
                if not touches and (PY_TOUCH.search(code2) or names.used(code2) or INLINE.search(code2)):
                    touches = True
            if touches:
                tags = comment + " " + prev_comment
                tag = ("driver" if TAG_DRIVER.search(tags) else "small" if TAG_SMALL.search(tags)
                       else "cpu-route" if TAG_CPU.search(tags) else "untagged")
                head = lm["head"]
                pyiter = bool(PY_TOUCH.search(head)) or names.used(head + "(") or names.iterated(head)
                data = bool(DATA_WORDS.search(head + " " + tags))
                f.loops.append((n, tag, pyiter, data, code.strip()[:100]))
        im = IF_KEY.match(code)
        if im:
            keys = frozenset(k["key"] for k in KEY_EQ.finditer(im["cond"]))
            if keys:
                end = n
                for n2, raw2, code2, _ in lines[idx + 1:]:
                    if not code2.strip():
                        continue
                    if _indent(raw2) <= ind:
                        break
                    end = n2
                f.blocks.append((n, end, keys))
    return funcs, imports, registrations


def TYPED_NAMES(code):
    return [m["name"] for m in TYPED.finditer(code)]


def _scan_one(job):
    path, text, tables, helpers = job
    return path, scan_file(path, text, tables, call_regex(helpers))


# ---------------------------------------------------------------- the index

def _table_entries_of(job):
    """The python_table_entries rows of one Python file."""
    path, rel = job
    try:
        tree = ast.parse(Path(path).read_text(encoding="utf-8", errors="replace"))
    except SyntaxError:
        return {}
    defined = {n.name for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))}

    def python_value(v):
        if isinstance(v, ast.Lambda):
            return "lambda"
        if isinstance(v, ast.Name) and v.id in defined:
            return v.id
        return None

    out = {}
    for node in ast.walk(tree):
        pairs = []
        if isinstance(node, ast.Dict):
            pairs = [(k.value, v) for k, v in zip(node.keys, node.values)
                     if isinstance(k, ast.Constant) and isinstance(k.value, str)]
        elif isinstance(node, ast.Call) and (
                (isinstance(node.func, ast.Name) and node.func.id == "dict") or
                (isinstance(node.func, ast.Attribute) and node.func.attr in ("update", "tables"))):
            pairs = [(kw.arg, kw.value) for kw in node.keywords if kw.arg]
        elif isinstance(node, ast.Assign):
            for t in node.targets:
                if isinstance(t, ast.Subscript) and isinstance(t.slice, ast.Constant) and isinstance(t.slice.value, str):
                    pairs.append((t.slice.value, node.value))
        for key, value in pairs:
            what = python_value(value)
            if what:
                out.setdefault(key, f"{rel}:{value.lineno} {what}")
    return out


def python_table_entries(root, pool=None):
    """{key: 'module:line name'} of string keys the Python package fills with a Python
    def, lambda or class of its own (dict literals, dict(k=v), .update(k=v), .tables(k=v),
    x["k"] = v). One task per file, across every core."""
    base = Path(root) / "python" / "mojolearn"
    jobs = [(str(p), p.relative_to(root).as_posix()) for p in sorted(base.rglob("*.py")) if "/tests/" not in p.as_posix()]
    results = pool.map(_table_entries_of, jobs, chunksize=8) if pool else map(_table_entries_of, jobs)
    out = {}
    for part in results:
        for k, v in part.items():
            out.setdefault(k, v)
    return out


class DoorIndex:
    """Every door-layer Mojo function, its Python sites, loops, callees and operation blocks."""

    def __init__(self, root=REPO, files=None, workers=None, python_tables=None):
        self.root = Path(root)
        if files is None:
            files = {}
            for pattern in ("bindings", "providers/python"):
                for p in sorted((self.root / pattern).rglob("*.mojo")):
                    rel = p.relative_to(self.root).as_posix()
                    if is_door(rel):
                        files[rel] = p.read_text(encoding="utf-8", errors="replace")
        workers = workers or os.cpu_count() or 1
        pool = ProcessPoolExecutor(max_workers=workers) if workers > 1 and len(files) > 8 else None
        try:
            self._build(files, python_tables, pool)
        finally:
            if pool:
                pool.shutdown()

    def _build(self, files, python_tables, pool):
        self.tables = python_table_entries(self.root, pool) if python_tables is None else python_tables
        helpers = set()
        texts = [files[p] for p in sorted(files)]
        for found in (pool.map(call_helpers, texts, chunksize=8) if pool else map(call_helpers, texts)):
            helpers |= found
        self.call_helpers = sorted(helpers - {"py_call"})
        jobs = [(p, t, self.tables, frozenset(helpers)) for p, t in sorted(files.items())]
        if pool:
            # one file per task, across every core (files are independent)
            results = list(pool.map(_scan_one, jobs, chunksize=4))
        else:
            results = [_scan_one(j) for j in jobs]
        self.funcs = {}                       # (path, name) -> Func (first definition)
        self.by_name = defaultdict(list)      # name -> [Func]
        self.imports = {}                     # path -> {alias: (path, name)}
        self.entries = defaultdict(list)      # python entry name -> [Func]
        self.keys = defaultdict(list)         # operation key -> [(Func, start, end)]
        self.files = sorted(files)
        registrations = []
        for path, (funcs, imports, regs) in results:
            self.imports[path] = imports
            for f in funcs:
                self.funcs.setdefault(f.key, f)
                self.by_name[f.name].append(f)
                for start, end, keys in f.blocks:
                    for k in keys:
                        self.keys[k].append((f, start, end))
            registrations.extend((path, name, fn) for name, fn in regs)
        for path, name, fn in registrations:
            target = self._resolve(path, fn)
            if target:
                self.entries[name].extend(t for t in target if t not in self.entries[name])
        self.unregistered = sorted({name for _, name, fn in registrations if not self.entries.get(name)})
        self._callee_cache = {}

    def _resolve(self, path, name):
        """Door functions a bare call `name(` in `path` reaches."""
        f = self.funcs.get((path, name))
        if f:
            return [f]
        imp = self.imports.get(path, {}).get(name)
        if imp:
            f = self.funcs.get((imp[0], imp[1]))
            if f:
                return [f]
            return []          # imported from a kernel (K) file or a struct: no Python there
        cands = self.by_name.get(name, [])
        return cands if len(cands) <= 2 else []

    def is_compiled_entry(self, name):
        return bool(self.entries.get(name))

    def _callees(self, f):
        if f.key not in self._callee_cache:
            self._callee_cache[f.key] = [(n, g) for n, name in f.callees
                                         for g in self._resolve(f.path, name) if g is not f]
        return self._callee_cache[f.key]

    def reach(self, entry, key=None, known_keys=frozenset()):
        """Aggregate the door code an entry (and its operation key) reaches.

        A function holding operation blocks for keys in `known_keys` is walked as a
        dispatcher: with a key, only its lines outside every such block plus the blocks
        for that key; without one, whole. Returns a Reach or None when the entry is not
        a registered compiled entry."""
        starts = self.entries.get(entry)
        if not starts:
            return None
        r = Reach(entry, key)
        stack = [(f, None) for f in starts]
        if key and key in self.keys:
            # the operation's own blocks, and the functions holding them walked with the key
            # filter (their shared preamble); not the whole router chain from the entry, whose
            # other operations would otherwise all count
            stack = [(f, (s, e)) for f, s, e in self.keys[key]] + [(f, None) for f, _, _ in self.keys[key]]
        seen = set()
        while stack:
            f, window = stack.pop()
            mark = (f.key, window)
            if mark in seen:
                continue
            seen.add(mark)
            if window is None:
                excluded = [(s, e) for s, e, ks in f.blocks if key and (ks & known_keys) and key not in ks]
                inside = lambda n: not any(s <= n <= e for s, e in excluded)
            else:
                inside = lambda n, w=window: w[0] <= n <= w[1]
            r.functions.add(f.key)
            for n, cls, what in f.sites:
                if inside(n):
                    r.add_site(f, n, cls, what)
            for n, tag, pyiter, data, head in f.loops:
                if inside(n):
                    r.add_loop(f, n, tag, pyiter, data, head)
            for n, g in self._callees(f):
                if inside(n) and (g.key, None) not in seen:
                    stack.append((g, None))
        return r


class Reach:
    __slots__ = ("entry", "key", "functions", "sites", "loops")

    def __init__(self, entry, key):
        self.entry, self.key = entry, key
        self.functions, self.sites, self.loops = set(), {}, {}

    def add_site(self, f, n, cls, what):
        self.sites.setdefault((f.path, n, cls, what), None)

    def add_loop(self, f, n, tag, pyiter, data, head):
        self.loops[(f.path, n)] = (tag, pyiter, data, head)

    def counts(self):
        c = Counter(cls for (_, _, cls, _) in self.sites)
        loops = list(self.loops.values())
        c["c-loops"] = len(loops)
        c["c-loops-data"] = sum(1 for t, p, d, h in loops if d or t in ("cpu-route", "untagged"))
        return c

    def level(self):
        c = self.counts()
        if c["b"]:
            return "python-on-path"
        if c["c-loops-data"] or c["c-inline"] or c["a"] >= GLUE_OBJECT_LINES:
            return "glue-heavy"
        return "native"

    def worst(self, limit=6):
        """The b sites first, then data-sized loops, as 'path:line what'."""
        out = [f"{p}:{n} {w}" for (p, n, cls, w) in sorted(self.sites) if cls == "b"]
        out += [f"{p}:{n} unresolved: {w}" for (p, n, cls, w) in sorted(self.sites) if cls == "b?"][:2]
        out += [f"{p}:{n} {t}-loop{' data' if d else ''}: {h}" for (p, n), (t, py, d, h) in sorted(self.loops.items())
                if d or t in ("cpu-route", "untagged")]
        return out[:limit]


def summarize(index):
    per = Counter()
    loops = Counter()
    for f in index.funcs.values():
        for _, cls, _ in f.sites:
            per[cls] += 1
        for _, tag, pyiter, data, _ in f.loops:
            loops[f"c-{'pyiter' if pyiter else 'mojoiter'}-{tag}"] += 1
            loops["data-sized"] += data
    lines = [f"door files {len(index.files)}, functions {len(index.funcs)}, compiled entries {len(index.entries)}, "
             f"operation keys {len(index.keys)}"]
    lines += [f"  {k:<24}{v:>8}" for k, v in sorted(per.items())]
    lines += [f"  {k:<24}{v:>8}" for k, v in sorted(loops.items())]
    return "\n".join(lines) + "\n"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--root", type=Path, default=REPO)
    ap.add_argument("--summary", action="store_true")
    ap.add_argument("--reach", metavar="ENTRY[:KEY]")
    args = ap.parse_args(argv)
    index = DoorIndex(args.root)
    if args.reach:
        entry, _, key = args.reach.partition(":")
        r = index.reach(entry, key or None, frozenset(index.keys))
        if r is None:
            print(f"{entry}: not a registered compiled entry (def_function / def_py_c_function)")
            return 1
        print(f"{entry}{':' + key if key else ''}: {r.level()} {dict(r.counts())} functions={len(r.functions)}")
        for w in r.worst(40):
            print("  " + w)
        return 0
    sys.stdout.write(summarize(index))
    return 0


if __name__ == "__main__":
    sys.exit(main())
