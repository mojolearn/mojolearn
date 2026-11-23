#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn
"""LAST-USE AUDIT: a local whose memory a pointer still reads after Mojo has freed the local.

    python3 tools/mojo_lastuse_audit.py [--commit <sha>] [--root DIR] [--all-kinds]

Mojo destroys a local at its LAST NAMED USE. A pointer derived from it keeps it alive only when the pointer's origin
says so. Mojo 1.2.0 stdlib, `std/origin/__init__.mojo` (tag mojo/v1.2.0):
  * a tracked origin (`x.unsafe_ptr()`, `Pointer(to=x)`, `Span(x)`, `.bitcast`) is `origin_of(x)`: x lives as long
    as the pointer does;
  * `MutAnyOrigin`/`ImmutAnyOrigin` are `#lit.any.origin`, the same attribute as `UnsafeAnyOrigin`, whose docstring
    says "Every other value in scope is kept alive for as long as the reference is live ... effectively halting ASAP
    destruction" (also the manual, mojolang.org/docs/manual/values/lifetimes, "Wildcard origins"): x lives as long as
    the pointer does;
  * `MutUntrackedOrigin`/`ImmUntrackedOrigin` are the EMPTY origin ("nothing for the lifetime checker to track or
    extend"), and an address (`Int(x.unsafe_ptr())`, `unsafe_from_address=`, a helper returning `Int(arg.unsafe_ptr())`)
    has no origin at all: x dies at its last named use, whatever the pointer still does.
So the audit fails on an UNTRACKED derivation of a local X where
  * the derivation is X's last named use and sits inside a call's arguments (X is freed before the call reads it:
    xtrees/calibrated_apply.mojo, the five native check suites of 2026-11-16), or is returned; or
  * it is assigned to P and P (or a name assigned from P) is used after X's last named use.
Any-origin and tracked derivations are counted (`--all-kinds` lists them) and never fail.

LAST USE ON THE PATH, AND PYTHON DATA ADDRESSES (2026-11-23, lane lastuse-addr-arg). Two shapes passed the rule
above and read freed memory:
  * the last named use is per PATH, not per text: `if case == 0: ... f(Int(y.unsafe_ptr()), n); return` frees y
    at that argument even when a later branch names y again (checks/native_tests/gbdt_public_fits.mojo
    host_ordered, y and coded, fixed in 90c52f3a7a). So "used after" counts only the statements reachable from the
    derivation: a `return`/`raise` of its block or an enclosing one ends the path, and the `elif`/`else` siblings
    of a branch it sits in are skipped;
  * a Python data address helper (a call named `*_address(` / `*_addr(` whose one argument is a local or one of
    its fields, `classical_address(labels.codes)`) is an untracked derivation of that local, whatever pointer is
    built around it: the address has no origin, and an any-origin `I32P(unsafe_from_address=...)` around it did
    not keep `labels` alive (bindings/prep_classify_common_door.mojo partial_walk, fixed in a1bbc6c2ac).
    What keeps the owner alive, and so passes: the owner itself is an argument of the same call (the callee
    borrows it: `KmArray(value, Int(py=classical_address(value)))`); the address is only compared; the helper's
    first parameter is a slot index or other non-owner (`_slot_address(index: Int)`, by name across files); the
    helper is a method of another object (`program.const_address(r)`); and, for data addresses only, a local
    that aliases a borrowed parameter's item (`var v = fields["_w"]`, never rebound) or that a holder takes on
    every path (`py_call(keep.append, x)`, `door.keep(x)`, `py_setattr(a, "_store", x)`, `pin.owner = x`).
Fix either with `_ = owner` (or `_ = owner^`) after the call or the pointer's last use.

ANY-ORIGIN AROUND AN ADDRESS (2026-11-23, lane gbdt-leaf-body). The same holds for a `.unsafe_ptr()` address: an
any-origin pointer built from `Int(x.unsafe_ptr())` (`MutPointer[T, MutAnyOrigin](unsafe_from_address=
Int(x.unsafe_ptr()))`, or an any-origin alias such as `F32P(unsafe_from_address=...)`) is an UNTRACKED derivation of
x, not an any-origin one: the Int carries no origin for the wildcard to extend, as partial_walk showed. It fails like
any untracked derivation when x has no later named use on the path (the Python-holder exceptions above are for data
addresses only). A cast that keeps the pointer (`x.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin]()`, `rebind`) is
still an any-origin derivation and passes.

FREED BEFORE THE DRAIN (2026-11-18, lane fx-lifetime). `ctx.enqueue_copy(dst_buf=d, src_ptr=h.unsafe_ptr())` is a
tracked derivation, so the rule above passes it, but the copy is asynchronous: the driver reads h after the call
returns, and no origin says so. A freed pinned block goes straight back to the caching allocator, so the next
`enqueue_create_host_buffer` on the thread is handed it and its fill overwrites bytes the copy has not read yet
(hist2_check and partitions_reduce_check failed on the L40S this way). So the audit also fails (kind `queued`) on a
local X declared as a pinned host buffer (`= ...enqueue_create_host_buffer`) or a List (`= List[`, `= [`) where
  * X is used by an `enqueue_copy` statement (a List: as `src_ptr=X...`; either: through a pointer `var p =
    X.unsafe_ptr()` of the same function, `src_ptr=p` / `dst_ptr=p`), and
  * no DRAIN lies between the latest such copy and X's last named use (extended over an enclosing loop's back
    edge), and X is never moved (`X^` other than `_ = X^`).
A DRAIN is a statement with `synchronize(` or a call (`name(` or `name[..](`) to a function that drains: its body
has `synchronize(` or calls a function that drains (any scanned file, by name; `main` excluded). Fix: `_ = X^` AFTER the drain. A keep-alive placed
before the drain is the same bug and fails.

A CHECK ENQUEUES NO HOST-TO-DEVICE COPY ITSELF (2026-11-18, lane fx-check-kit). The 587 findings of the rule above
had one cause: every check wrote its own upload sequence. So a file under checks/native_tests/ other than the kit
(checks/native_tests/kit.mojo) fails (kind `own-copy`) on any `enqueue_copy` with `src_ptr=`, or with `src_buf=` naming
a pinned host buffer of the function: it calls the kit (`upload`, `upload_into`, `upload_host_into`, ...), which
holds the host memory and drains before it returns. A check whose subject is the queue itself uses the kit's
`..._undrained_keep` forms or gets an allow line with its reason.

Fix: name the local again after the pointer's last use (`_ = x^`; free). A true false alarm (a stack array whose
slot outlives the name, a value the caller owns) gets one line in tools/mojo_lastuse_allow.txt:
`<path>:<function>:<local>  # <reason>`. Unlisted findings fail; allow lines that match nothing are printed as STALE.

Every tracked .mojo file (git ls-files, or the blobs of --commit) is scanned, split across os.cpu_count() worker
processes. tools/prebuild_check.py runs it in its `audits` part (seconds).
"""
import argparse
import concurrent.futures
import functools
import os
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ALLOW = "tools/mojo_lastuse_allow.txt"
_DEF = re.compile(r"^(\s*)def\s+(\w+)")
_VAR = re.compile(r"(?<![\w.])var\s+(\w+)\b")
_PARAM_VAR = re.compile(r"[(,]\s*(?:var|owned)\s+(\w+)\s*:")
_ASSIGN = re.compile(r"^\s*(?:var\s+)?(\w+)\s*(?::[^=]*)?=(?!=)")
_HELPER_DEF = re.compile(r"^\s*def\s+(\w+)\s*\(([^)]*)\)\s*(?:raises\s*)?->\s*(\w+)")
_HELPER_RET = r"return\s+(?:U?Int\(\s*{p}\.unsafe_ptr\(\)|{p}\.unsafe_ptr\(\)[^\n]*UntrackedOrigin)"
#: wrappers that turn their argument into an address or pointer; not calls that consume the pointer
_CONVERT = {"Int", "UInt", "UnsafePointer", "Pointer", "MutPointer", "ImmPointer", "OpaquePointer", "rebind"}
_IDENT = re.compile(r"(?<![\w.])([A-Za-z_]\w*)\b(?!=[^=])")
_HOT = re.compile(r"unsafe_ptr|unsafe_origin_cast|as_unsafe_any_origin|\bto=|_addr(?:ess)?\s*\(")
#: a Python data address helper call wrapping the owner (`classical_address(` before `labels.codes)`)
#: a free function, not a method (`program.const_address(r)` is an address into program, not into r)
_PYADDR = re.compile(r"(?<![\w.])(\w+_addr(?:ess)?)\(\s*$")
_PYADDR_ARG = re.compile(r"(?:\.\w+)*\s*\)")
_END = re.compile(r"^\s*(?:return|raise)\b")
_SIBLING = re.compile(r"^\s*(?:elif\b|else\s*:)")
_TRY = re.compile(r"^\s*try\s*:")
_FIELD_STORE = re.compile(r"^\s*\w+(?:\.\w+)+\s*=\s*(?:Optional\[[^\]]*\]\(\s*)?(\w+)\s*\)?\s*$")
_HOLDER = re.compile(r"\b(?:append|extend|insert|keep|py_setattr|py_setitem|set_store)\b")
#: the innermost call a value handed to a holder sits in (`py_call(keep.append, Python.tuple(d, x))`)
_HOLDS = {"append", "extend", "insert", "keep", "py_setattr", "py_setitem", "set_store", "py_call", "tuple", "list",
          "dict"}
_BARE_ARG = re.compile(r"(?:[(,]|[^=!<>]=)\s*([A-Za-z_]\w*)\s*(?=[,)])")
#: first parameter types of a `*_address`/`*_addr` helper that make it no data address of an owner (a slot index,
#: a borrowed C object pointer, the device context it allocates through)
_NOT_OWNER = {"Int", "UInt", "Int64", "UInt64", "Int32", "UInt32", "PyObjectPtr", "DeviceContext"}
_ADDR_DEF = re.compile(r"^\s*def\s+(\w+_addr(?:ess)?)\s*(?:\[[^\]\n]*\])?\(\s*(?:(?:mut|read|var|ref|owned)\s+)?\w+"
                       r"\s*:\s*([\w.]+)", re.M)
_READ = re.compile(r"\.(?:unsafe_ptr|unsafe_origin_cast\[[^\]]*\]|bitcast\[[^\]]*\])\(\)\s*(?:\[|\.(?:unsafe_load|load|unsafe_get)\b)")
_KEEP = re.compile(r"^\s*_\s*=\s*\w+\^?\s*$")
_LOOP = re.compile(r"^\s*(?:for|while)\b")
_PIN = re.compile(r"(?<![\w.])var\s+(\w+)\s*(?::[^=]*)?=\s*[\w.\[\]]*enqueue_create_host_buffer")
_LST = re.compile(r"(?<![\w.])var\s+(\w+)\s*(?::\s*List\[[^=]*)?=\s*(?:List\[|\[)")
_PTR_OF = re.compile(r"^\s*var\s+(\w+)\s*(?::[^=]*)?=\s*(\w+)\.unsafe_ptr\(\)\s*$")
#: the checks that take their transfers through the kit, and the kit
KIT_DIR = "checks/native_tests/"
KIT = "checks/native_tests/kit.mojo"
_OWN_PTR = re.compile(r"\bsrc_ptr\s*=\s*([^,)]*)")
_OWN_BUF = re.compile(r"\bsrc_buf\s*=\s*(\w+)\b")
_SYNC = re.compile(r"\bsynchronize\s*\(")
_CALLEE = re.compile(r"\b([A-Za-z_]\w*)\s*(?:\[[^\[\]]*\])?\s*\(")
_STRING = re.compile(r'"(?:\\.|[^"\\])*"|\'(?:\\.|[^\'\\])*\'')


def _strip(line):
    """LINE without string literals and its comment."""
    line = _STRING.sub('""', line)
    cut = line.find("#")
    return line if cut < 0 else line[:cut]


def statements(lines):
    """[(first line index, last line index, code)] of LINES: comments, strings and docstrings removed, a statement
    continued over lines while a bracket is open."""
    out, buf, start, depth, in_doc = [], [], None, 0, None
    for i, raw in enumerate(lines):
        if in_doc:
            if in_doc in raw:
                in_doc = None
            continue
        s = raw.strip()
        doc = next((q for q in ('"""', "'''") if s.startswith(q) and not buf), None)
        if doc:  # a docstring (or a bare string block): skipped, closed on its line or a later one
            if s.count(doc) == 1:
                in_doc = doc
            continue
        code = _strip(raw)
        if not code.strip() and not buf:
            continue
        if start is None:
            start = i
        buf.append(code)
        depth += sum(code.count(c) for c in "([{") - sum(code.count(c) for c in ")]}")
        if depth <= 0:
            out.append((start, i, " ".join(b.strip() for b in buf)))
            buf, start, depth = [], None, 0
    if buf:
        out.append((start, len(lines) - 1, " ".join(b.strip() for b in buf)))
    return out


def functions(lines):
    """[(name, def line index, end line index exclusive)] for every def, nested ones included (an outer body
    contains its nested defs: a closure's capture is a use)."""
    defs = []
    for i, line in enumerate(lines):
        m = _DEF.match(line)
        if not m:
            continue
        indent, end = len(m.group(1)), len(lines)
        j = i + 1
        # skip the rest of a multi-line signature, a parameter list (`def f[\n    n: Int\n](`) included: counting
        # only parentheses ended `scan_run[` at its `](` line, so its body (and its synchronize) was never seen
        def opened(t):
            t = _strip(t)
            return sum(t.count(c) for c in "([{") - sum(t.count(c) for c in ")]}")
        depth = opened(line)
        while depth > 0 and j < len(lines):
            depth += opened(lines[j])
            j += 1
        for k in range(j, len(lines)):
            t = lines[k]
            if t.strip() and not t.lstrip().startswith("#") and len(t) - len(t.lstrip()) <= indent:
                end = k
                break
        defs.append((m.group(2), i, end))
    return defs


def helpers_of(text):
    """{name: 'untracked'} for the defs in TEXT that return an address or untracked pointer of a parameter, and
    {'alias:NAME': 'any'} for its pointer aliases with a wildcard origin (`comptime _AP = MutPointer[T, MutAnyOrigin]`)."""
    found = {}
    for m in _ADDR_DEF.finditer(text):
        if m.group(2).split(".")[-1] in _NOT_OWNER:
            found["noaddr:" + m.group(1)] = "noaddr"
    for m in re.finditer(r"^comptime\s+(\w+)\s*=\s*(?:Mut|Imm)?(?:Unsafe)?Pointer\[[^\n]*AnyOrigin", text, re.M):
        found["alias:" + m.group(1)] = "any"
    lines = text.splitlines()
    for i, line in enumerate(lines):
        m = _HELPER_DEF.match(line)
        if not m:
            continue
        params = [p.split(":")[0].split()[-1] for p in m.group(2).split(",") if ":" in p]
        body = "\n".join(lines[i + 1:i + 8])
        if any(re.search(_HELPER_RET.format(p=re.escape(p)), body) for p in params):
            found[m.group(1)] = "untracked"
    calls = {}  # for the queued rule: functions that drain directly (`drain:NAME`) and every def's callees
    for fname, a, b in functions(lines):
        if fname == "main":
            continue
        if any(_SYNC.search(t) for t in lines[a:b]):
            found["drain:" + fname] = "drain"
        else:
            calls.setdefault(fname, set()).update(_CALLEE.findall("\n".join(_strip(t) for t in lines[a + 1:b])))
    found["__calls__"] = calls
    return found


def close_drains(direct, call_maps):
    """DIRECT (names whose body synchronizes) closed over CALL_MAPS ({name: callees}): a function that calls a
    draining function drains. By name across files, like the helper pass."""
    drains, grew = set(direct), True
    while grew:
        grew = False
        for calls in call_maps:
            for fname, callees in calls.items():
                if fname not in drains and not callees.isdisjoint(drains):
                    drains.add(fname)
                    grew = True
    return frozenset(drains)


@functools.lru_cache(maxsize=8)
def _drain_call(drains):
    """A regex for a call to one of DRAINS (a frozenset of function names), parametric calls included."""
    if not drains:
        return None
    return re.compile(r"\b(?:" + "|".join(sorted(map(re.escape, drains))) + r")\s*(?:\[[^\[\]]*\])?\s*\(")


def _chain_end(s, k):
    """End of the postfix chain (.name, [..], (..)) that starts at S[k]."""
    while k < len(s):
        if s[k] == "." and k + 1 < len(s) and (s[k + 1].isalpha() or s[k + 1] == "_"):
            k += 1
            while k < len(s) and (s[k].isalnum() or s[k] == "_"):
                k += 1
        elif s[k] in "[(":
            close, depth = {"[": "]", "(": ")"}[s[k]], 0
            open_ = s[k]
            while k < len(s):
                depth += (s[k] == open_) - (s[k] == close)
                k += 1
                if depth == 0:
                    break
        else:
            break
    return k


def _enclosing_calls(s, p, at=False):
    """Names before each open '(' that encloses position P (innermost first). A pointer constructor
    (`T(unsafe_from_address=...)`, T an alias) is named `UnsafePointer`. AT: (position of the '(', name) pairs."""
    names, depth = [], 0
    for k in range(p - 1, -1, -1):
        c = s[k]
        if c in ")]":
            depth += 1
        elif c in "([":
            if depth:
                depth -= 1
                continue
            if c == "(":
                m = re.search(r"([\w.]+)(?:\[[^\[\]]*\])?\s*$", s[:k])
                ctor = s[k + 1:].lstrip().startswith("unsafe_from_address=")
                name = "UnsafePointer" if ctor else (m.group(1).split(".")[-1] if m else "")
                names.append((k, name) if at else name)
    return names


def _held_by_the_call(s, name, start):
    """Whether NAME itself (bare, positional or keyword) is an argument of a call that encloses the derivation at
    START: the callee borrows NAME, so it lives until that call returns (`Python.tuple(store, addr(store))`,
    `KmArray(value, Int(py=classical_address(value)))`)."""
    opens = {k for k, c in _enclosing_calls(s, start, at=True) if c not in _CONVERT}
    if not opens:
        return False
    for m in re.finditer(r"(?<![\w.])" + re.escape(name) + r"\b(?=\s*[,)])", s):
        if not re.search(r"(?:[(,]|[^=!<>]=)\s*$", s[:m.start()]):
            continue
        inner = next(((k, c) for k, c in _enclosing_calls(s, m.start(), at=True) if c not in _CONVERT), None)
        if inner and inner[0] in opens:
            return True
    return False


def _arg_span(s, p):
    """The text of the argument (of the innermost enclosing call that is not a conversion) that holds position P:
    an any-origin pointer built around an address (`MutPointer[T, MutAnyOrigin](unsafe_from_address=Int(..))`)
    keeps the value alive like any other any-origin pointer."""
    lo, depth = 0, 0
    for k in range(p - 1, -1, -1):
        c = s[k]
        if c in ")]":
            depth += 1
        elif c in "([":
            if depth:
                depth -= 1
                continue
            m = re.search(r"([\w.]+)(?:\[[^\[\]]*\])?\s*$", s[:k]) if c == "(" else None
            ctor = s[k + 1:].lstrip().startswith("unsafe_from_address=")
            if c == "(" and not ctor and (m.group(1).split(".")[-1] if m else "") not in _CONVERT:
                lo = k + 1
                break
        elif c == "," and depth == 0:
            lo = k + 1
            break
    hi, depth = len(s), 0
    for k in range(p, len(s)):
        c = s[k]
        if c in "([":
            depth += 1
        elif c in ")]":
            if depth == 0:
                hi = k
                break
            depth -= 1
        elif c == "," and depth == 0:
            hi = k
            break
    return s[lo:hi]


def derivation(s, name, helpers):
    """(kind, sink, via) of the first pointer derivation of NAME in statement S, or None. kind: untracked, any,
    tracked; sink: call, held (NAME itself is an argument of the call), return, assign; via: pyaddr (a data
    address helper) or ptr. A pointer read on the spot (`x.unsafe_ptr()[i]`, `.unsafe_load(`) and a
    pointer into what X points at (`x[].field.unsafe_ptr()`: X is a pointer, not the owner) are not derivations."""
    for m in re.finditer(r"(?<![\w.])" + re.escape(name) + r"\b(?!=)", s):
        p, kind, via = m.start(), None, "ptr"
        before = s[:p]
        if s[m.end():].startswith("[]"):
            continue
        hm = re.search(r"\b(\w+)\(\s*$", before)
        am = _PYADDR.search(before)
        if hm and helpers.get(hm.group(1)) == "untracked" and s[m.end():].lstrip().startswith(")"):
            kind, start = helpers[hm.group(1)], hm.start(1)
        elif am and am.group(1) not in helpers.get("__noaddr__", ()) and _PYADDR_ARG.match(s, m.end()):
            # a data address of X (or of a field of X): no origin, whatever pointer is built around it
            kind, start, via = "untracked", am.start(1), "pyaddr"
        else:
            to = re.search(r"\b(?:Unsafe)?Pointer(?:\[[^\[\]]*\])?\(\s*to=$", before)
            start = to.start() if to else p
            end = _chain_end(s, s.index(")", m.end()) + 1) if to and ")" in s[m.end():] else _chain_end(s, m.end())
            chain = s[start:end]
            # `x.unsafe_origin_cast[...]` alone: X is already a pointer and owns nothing
            if not to and not re.search(r"\.unsafe_ptr\b", chain):
                continue
            if _READ.search(chain):
                continue
            arg = _arg_span(s, start)
            for alias, akind in helpers.items():
                if alias.startswith("alias:") and re.search(r"\b" + alias[6:] + r"\b", arg):
                    arg += " " + ("AnyOrigin" if akind == "any" else "")
            if "UntrackedOrigin" in chain:
                kind = "untracked"
            elif re.search(r"(?:\bU?Int\(|unsafe_from_address=)\s*$", s[:start]):
                # an ADDRESS of X (`Int(x.unsafe_ptr())`) has no origin, so an any-origin pointer built around it
                # (`MutPointer[T, MutAnyOrigin](unsafe_from_address=Int(x.unsafe_ptr()))`) keeps nothing alive
                # either: the same shape as the Python data address of partial_walk (module docstring, ANY-ORIGIN
                # AROUND AN ADDRESS)
                kind = "untracked"
                if re.search(r"AnyOrigin|as_unsafe_any_origin", arg):
                    via = "anyaddr"
            elif re.search(r"AnyOrigin|as_unsafe_any_origin", arg):
                kind = "any"
            else:
                kind = "tracked"
        if re.search(r"[=!<>]=|\bis\b", _arg_span(s, start)):
            continue  # an address compared (`a == Int(py=classical_address(view))`): nothing reads through it
        calls = [c for c in _enclosing_calls(s, start) if c not in _CONVERT]
        if calls:
            sink = "held" if _held_by_the_call(s, name, start) else "call"
        elif re.match(r"^\s*return\b", s):
            sink = "return"
        else:
            sink = "assign"
        return kind, sink, via
    return None


def scan_text(path, text, helpers):
    """[finding dict] of one file."""
    lines = text.splitlines()
    stmts = statements(lines)
    out, seen = [], set()
    drains = helpers.get("__drains__")
    if drains is None:
        drains = close_drains((k[6:] for k in helpers if k.startswith("drain:")), [helpers.get("__calls__", {})])
    drain_call = _drain_call(drains) if "enqueue_copy" in text else None
    if "__noaddr__" not in helpers:
        helpers = dict(helpers, __noaddr__=frozenset(k[7:] for k in helpers if k.startswith("noaddr:")))
    for fname, a, b in functions(lines):
        body = [(i, j, s) for (i, j, s) in stmts if a < i < b or (i == a)]
        if not body:
            continue
        indent = [len(lines[i]) - len(lines[i].lstrip()) for (i, _, _) in body]
        n_body = len(body)

        def block_end(k):
            """First statement after K's block (the statements indented deeper than K that follow it)."""
            for q in range(k + 1, n_body):
                if indent[q] <= indent[k]:
                    return q
            return n_body

        def scope_end(k):
            """End of the scope a `var` declared at K lives in: the first later statement indented less."""
            for q in range(k + 1, n_body):
                if indent[q] < indent[k]:
                    return q
            return n_body

        idents = [set(_IDENT.findall(s)) if k else set() for k, (_, _, s) in enumerate(body)]
        calls = [h for h, v in helpers.items() if v == "untracked"]
        hot = [bool(_HOT.search(s)) or any(h + "(" in s for h in calls) for (_, _, s) in body]
        loops = [k for k, (_, _, s) in enumerate(body) if _LOOP.match(s)]
        decls = {}  # name -> [(declaration statement, scope end)]
        for v in _PARAM_VAR.findall(body[0][2]):
            decls.setdefault(v, []).append((0, n_body))
        for k in range(1, n_body):
            for v in _VAR.findall(body[k][2]):
                decls.setdefault(v, []).append((k, scope_end(k)))
        decls.pop("self", None)
        # PythonObject locals that only alias what a borrowed parameter holds (`var v = fields["_w"]`,
        # `py_item(fields, k)`): the caller's object keeps the data, so a data address of v outlives v
        sig = body[0][2]
        borrowed = {n for n in re.findall(r"[(,]\s*(?:(?:mut|read|ref)\s+)?(\w+)\s*:", sig)} - set(
            _PARAM_VAR.findall(sig)) - {"self"}
        aliases = set()
        if borrowed:
            looked = r"(?:(?:{p})\s*(?:\[|\.\w+\s*$)|py_(?:item|get|getattr)\(\s*(?:{p})\s*,)".format(
                p="|".join(map(re.escape, sorted(borrowed))))
            for v, scopes in decls.items():
                if len(scopes) == 1 and scopes[0][0] > 0 and re.search(
                        r"(?<![\w.])var\s+" + re.escape(v) + r"\s*(?::[^=]*)?=\s*" + looked, body[scopes[0][0]][2]) \
                        and not any(re.match(r"^\s*" + re.escape(v) + r"\s*=(?!=)", t) for (_, _, t) in body):
                    aliases.add(v)
        # ... and PythonObject locals the function hands to a holder (`py_call(keep.append, x)`, `door.keep(x)`,
        # `py_setattr(array, "_store", x)`, `py_call(set_store, result, x)`): the holder keeps the data
        holders = {}  # name -> statements that hand it to a holder
        for q in range(1, n_body):
            t = body[q][2]
            fm = _FIELD_STORE.match(t)
            if fm:  # `pin.owner = codes`: the field holds it
                holders.setdefault(fm.group(1), []).append(q)
            if _HOLDER.search(t):
                for m in _BARE_ARG.finditer(t):
                    inner = _enclosing_calls(t, m.start(1))
                    if inner and inner[0] in _HOLDS:
                        holders.setdefault(m.group(1), []).append(q)

        def held(x, k, upto):
            """Whether a holder statement for X runs on every path to K (in a block that encloses K, X not rebound
            in between), or runs unconditionally after K and before UPTO (the pointer's last use): X lives until
            the holder takes it, and the holder (a function-scope list, field or object) keeps it after."""
            for q in holders.get(x, ()):
                if q < k and all(indent[r] >= indent[q] for r in range(q + 1, k + 1)) and not any(
                        re.match(r"^\s*" + re.escape(x) + r"\s*=(?!=)", body[r][2]) for r in range(q + 1, k)):
                    return True
                if k < q <= upto and q in reachable(k) and indent[q] <= indent[k] and all(
                        indent[r] >= indent[q] for r in range(k, q + 1)):
                    return True
            return False

        def uses(n, lo, hi):
            return [k for k in range(max(lo, 1), hi) if n in idents[k]]

        def scope_of(n, k):
            """(declaration, scope end) of the NAME visible at statement K, or None."""
            best = None
            for d, e in decls.get(n, ()):
                if d <= k < e and (best is None or d >= best[0]):
                    best = (d, e)
            return best

        def effective_last(d, last):
            """LAST extended to the end of every loop that encloses it and began after the declaration D: the
            value lives across the back edge."""
            eff = last
            for h in loops:
                if d < h <= last < block_end(h):
                    eff = max(eff, block_end(h) - 1)
            return eff

        reach_memo = {}

        def reachable(k):
            """The statements a run can reach after K without leaving the function: a `return`/`raise` in K's
            block or an enclosing one ends the path, the `elif`/`else` siblings of a branch K is in are skipped
            (module docstring, LAST USE ON THE PATH). Loops are left to effective_last."""
            if k in reach_memo:
                return reach_memo[k]
            tries, low = [], indent[k]  # indents of the `try:` headers around K: a `raise` inside goes to except
            for q in range(k - 1, 0, -1):
                if indent[q] < low:
                    low = indent[q]
                    if _TRY.match(body[q][2]):
                        tries.append(low)
            got, cur, q = set(), indent[k], k + 1
            while q < n_body:
                ind = indent[q]
                if ind < cur:
                    cur = ind
                    if _SIBLING.match(body[q][2]):
                        while q < n_body and indent[q] == cur and _SIBLING.match(body[q][2]):
                            q = block_end(q)
                        continue
                got.add(q)
                t = body[q][2]
                if ind <= cur and _END.match(t) and not (t.lstrip().startswith("raise") and
                                                         any(ti < cur for ti in tries)):
                    break
                q += 1
            reach_memo[k] = got
            return got

        for x, scopes in sorted(decls.items()):
            for d, e in scopes:
                ux = uses(x, d, e)
                if not ux:
                    continue
                for k in ux:
                    if not hot[k]:
                        continue
                    s = body[k][2]
                    found = derivation(s, x, helpers)
                    if not found:
                        continue
                    kind, sink, via = found
                    on_path = reachable(k)
                    later = [q for q in ux if q > k and q in on_path]
                    last_x = later[-1] if later else k
                    moved = re.search(r"(?<![\w.])" + re.escape(x) + r"\^", body[last_x][2]) and not re.match(
                        r"^\s*_\s*=\s*" + re.escape(x) + r"\^\s*$", body[last_x][2])
                    eff_x = effective_last(d, last_x)
                    bad, why, upto = False, "", k
                    if sink == "assign":
                        am = _ASSIGN.match(s)
                        if am and am.group(1) != x:
                            names, frontier, p_last = {am.group(1)}, [am.group(1)], k
                            while frontier:  # names assigned from the pointer after the derivation
                                n = frontier.pop()
                                sc = scope_of(n, k) or (d, e)
                                for q in uses(n, k + 1, sc[1]):
                                    if q not in on_path or _KEEP.match(body[q][2]):
                                        continue  # `_ = p`: a keep-alive, not a read through the pointer
                                    p_last = max(p_last, q)
                                    bm = _ASSIGN.match(body[q][2])
                                    if bm and bm.group(1) not in names and bm.group(1) not in (n, "_") and \
                                            _carries(body[q][2], n):
                                        names.add(bm.group(1))
                                        frontier.append(bm.group(1))
                            upto = p_last
                            if p_last > eff_x:
                                bad, why = True, f"pointer {am.group(1)} used at line {body[p_last][0] + 1}"
                    elif sink in ("call", "return") and k == last_x and eff_x == last_x:
                        bad, why = True, ("freed before the call reads it" if sink == "call" else "returned")
                    if kind != "untracked" or moved or (via == "pyaddr" and (x in aliases or held(x, k, upto))):
                        bad = False
                    key = (x, body[k][0])
                    if key in seen:
                        continue
                    seen.add(key)
                    out.append(dict(path=path, line=body[k][0] + 1, function=fname, local=x, kind=kind, sink=sink,
                                    last_use=body[last_x][0] + 1, fails=bad, why=why))
        if "enqueue_copy" in text:
            out.extend(_queued(path, fname, body, idents, effective_last, drain_call, seen))
            if path.startswith(KIT_DIR) and path != KIT:
                out.extend(_own_copies(path, fname, body))
    return out


def _queued(path, fname, body, idents, effective_last, drain_call, seen):
    """The FREED BEFORE THE DRAIN findings of one function (module docstring)."""
    out, n_body = [], len(body)
    copies = [q for q in range(1, n_body) if "enqueue_copy" in body[q][2]]
    if not copies:
        return out
    drain = [bool(_SYNC.search(s)) or bool(drain_call and drain_call.search(s)) for (_, _, s) in body]
    ptrs = {}  # owner -> pointer names (`var p = x.unsafe_ptr()`)
    for q in range(1, n_body):
        pm = _PTR_OF.match(body[q][2])
        if pm:
            ptrs.setdefault(pm.group(2), set()).add(pm.group(1))
    for d in range(1, n_body):
        s = body[d][2]
        for rx, kind in ((_PIN, "pinned"), (_LST, "list")):
            m = rx.search(s)
            if not m:
                continue
            x = m.group(1)
            uses = [q for q in range(d + 1, n_body) if x in idents[q]]
            if not uses:
                continue
            if any(re.search(r"(?<![\w.])" + re.escape(x) + r"\^", body[q][2]) and not _KEEP.match(body[q][2])
                   for q in uses):
                continue  # moved into a holder, a field or the return value: the new owner keeps it
            aliases = sorted(ptrs.get(x, ()))
            alias_rx = ("|".join(map(re.escape, aliases))) if aliases else None
            queued = []
            for q in copies:
                if q <= d:
                    continue
                t = body[q][2]
                if alias_rx and re.search(r"\b(?:src|dst)_ptr\s*=\s*(?:" + alias_rx + r")\b", t):
                    queued.append(q)
                elif kind == "pinned" and x in idents[q]:
                    queued.append(q)
                elif kind == "list" and re.search(r"\bsrc_ptr\s*=\s*" + re.escape(x) + r"\b", t):
                    queued.append(q)
            if not queued:
                continue
            alias_uses = [q for p in aliases for q in range(d + 1, n_body) if p in idents[q]]
            last = effective_last(d, max(uses + alias_uses))
            first = queued[-1]
            if any(drain[q] for q in range(first, last + 1)):
                continue
            nxt = next((q for q in range(last + 1, n_body) if drain[q]), None)
            # its own key: the pointer pass above records (x, line) for the same enqueue_copy line as a tracked,
            # passing finding, and sharing that key dropped every queued finding
            key = (x, body[first][0], "queued")
            if key in seen:
                continue
            seen.add(key)
            where = f"the drain at line {body[nxt][0] + 1}" if nxt is not None else "a drain (none follows in it)"
            out.append(dict(path=path, line=body[first][0] + 1, function=fname, local=x, kind="queued",
                            sink="enqueue_copy", last_use=body[last][0] + 1, fails=True,
                            why=f"the enqueue_copy at line {body[first][0] + 1} still reads it; put `_ = {x}^` "
                                f"after {where}"))
    return out


def _own_copies(path, fname, body):
    """The OWN-COPY findings of one function of a check under checks/native_tests/ (module docstring)."""
    out = []
    pinned = {m.group(1) for (_, _, s) in body[1:] for m in [_PIN.search(s)] if m}
    for q in range(1, len(body)):
        s = body[q][2]
        if "enqueue_copy" not in s:
            continue
        m = _OWN_PTR.search(s)
        if m:
            source = (_IDENT.findall(m.group(1)) or ["host"])[-1]
        else:
            b = _OWN_BUF.search(s)
            if not b or b.group(1) not in pinned:
                continue
            source = b.group(1)
        out.append(dict(path=path, line=body[q][0] + 1, function=fname, local=source, kind="own-copy",
                        sink="enqueue_copy", last_use=body[q][0] + 1, fails=True,
                        why="a check enqueues no host-to-device copy itself: call " + KIT))
    return out


def _carries(s, n):
    """Whether assignment S copies the pointer N into its target (`q = n + 4`, `[n, m]`,
    `MutPointer(unsafe_from_address=n)`), not a value a call computed from it."""
    rhs = s[_ASSIGN.match(s).end():]
    off = len(s) - len(rhs)
    for m in re.finditer(r"(?<![\w.])" + re.escape(n) + r"\b", rhs):
        if re.match(r"\s*(?:\[|\.(?:unsafe_load|load|unsafe_get)\b)", rhs[m.end():]):
            continue  # a value read through the pointer, not the pointer
        if not [c for c in _enclosing_calls(s, off + m.start()) if c not in _CONVERT]:
            return True
    return False


def _scan_many(items):
    return [f for path, text, helpers in items for f in scan_text(path, text, helpers)]


def _helper_pass(items):
    return [(path, helpers_of(text)) for path, text in items]


def read_files(root=ROOT, commit=None):
    """[(path, text)] of every tracked .mojo file in ROOT's work tree, or of COMMIT's blobs."""
    if commit is None:
        names = subprocess.run(["git", "ls-files", "-z", "--", "*.mojo"], cwd=str(root), capture_output=True,
                               check=True).stdout.decode().split("\0")
        return [(n, (Path(root) / n).read_text(errors="replace")) for n in names if n and (Path(root) / n).is_file()]
    listing = subprocess.run(["git", "ls-tree", "-r", "-z", commit], cwd=str(root), capture_output=True,
                             check=True).stdout.decode().split("\0")
    rows = [r.split("\t", 1) for r in listing if r.endswith(".mojo") and " blob " in r]
    shas = [(meta.split()[2], path) for meta, path in rows]
    proc = subprocess.run(["git", "cat-file", "--batch"], cwd=str(root), check=True, capture_output=True,
                          input="".join(f"{sha}\n" for sha, _ in shas).encode())
    data, pos, out = proc.stdout, 0, []
    for _, path in shas:
        header_end = data.index(b"\n", pos)
        size = int(data[pos:header_end].split()[2])
        out.append((path, data[header_end + 1:header_end + 1 + size].decode(errors="replace")))
        pos = header_end + 1 + size + 1
    return out


def _module_file(path, module):
    """The repo path of MODULE imported from PATH (`a.b` -> a/b.mojo; `.b` relative to PATH's directory)."""
    if module.startswith("."):
        base = Path(path).parent
        dots = len(module) - len(module.lstrip("."))
        for _ in range(dots - 1):
            base = base.parent
        rel = base / module.lstrip(".").replace(".", "/")
    else:
        rel = Path(module.replace(".", "/"))
    return rel.as_posix() + ".mojo"


def _chunks(items, n):
    return [items[i::n] for i in range(n) if items[i::n]]


def scan(files, workers=None):
    """[finding] over FILES [(path, text)], split across WORKERS processes (default os.cpu_count())."""
    workers = max(1, workers or os.cpu_count() or 1)
    with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as pool:
        per_file = dict(h for part in pool.map(_helper_pass, _chunks(files, workers)) for h in part)
        drains = close_drains((k[6:] for h in per_file.values() for k in h if k.startswith("drain:")),
                              [h.pop("__calls__", {}) for h in per_file.values()])
        # by name across files, like the drains: a `*_address` helper of a slot index is no owner's data address
        noaddr = frozenset(k[7:] for h in per_file.values() for k in h if k.startswith("noaddr:"))
        items = []
        for path, text in files:
            local = dict(per_file.get(path, {}))
            for m in re.finditer(r"^\s*from\s+(\S+)\s+import\s+([^\n]+)", text, re.M):
                source = per_file.get(_module_file(path, m.group(1)), {})
                for name in re.findall(r"\w+", m.group(2)):
                    if source.get(name) == "untracked":
                        local.setdefault(name, "untracked")
                    if "alias:" + name in source:
                        local.setdefault("alias:" + name, source["alias:" + name])
            local["__drains__"] = drains
            local["__noaddr__"] = noaddr
            items.append((path, text, local))
        found = [f for part in pool.map(_scan_many, _chunks(items, workers)) for f in part]
    return sorted(found, key=lambda f: (f["path"], f["line"], f["local"]))


def load_allow(text):
    """{(path, function, local): reason} of an allow-list; a line without a reason is an error."""
    allow, errors = {}, []
    for n, line in enumerate(text.splitlines(), 1):
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        key, _, reason = line.partition("#")
        parts = key.strip().split(":")
        if len(parts) != 3 or not reason.strip():
            errors.append(f"{ALLOW}:{n}: want '<path>:<function>:<local>  # <reason>'")
            continue
        allow[tuple(parts)] = reason.strip()
    return allow, errors


def verdict(found, allow):
    """(failing findings, allow-listed findings, stale allow keys)."""
    failing, allowed, used = [], [], set()
    for f in found:
        if not f["fails"]:
            continue
        key = (f["path"], f["function"], f["local"])
        if key in allow:
            allowed.append(f)
            used.add(key)
        else:
            failing.append(f)
    return failing, allowed, sorted(set(allow) - used)


def audit(root=ROOT, commit=None, workers=None, all_kinds=False, out=print):
    """Run the audit; print lines; return the exit code."""
    files = read_files(root, commit)
    if commit is None:
        p = Path(root) / ALLOW
        allow_text = p.read_text() if p.is_file() else ""
    else:
        r = subprocess.run(["git", "show", f"{commit}:{ALLOW}"], cwd=str(root), capture_output=True, text=True)
        allow_text = r.stdout if r.returncode == 0 else ""
    allow, errors = load_allow(allow_text)
    found = scan(files, workers)
    failing, allowed, stale = verdict(found, allow)
    kinds = {}
    for f in found:
        kinds[f["kind"]] = kinds.get(f["kind"], 0) + 1
    for e in errors:
        out(f"LASTUSE FAIL {e}")
    for f in failing:
        if f["kind"] == "own-copy":
            out(f"LASTUSE FAIL {f['path']}:{f['line']}: {f['local']} in {f['function']} (own-copy): {f['why']}")
            continue
        out(f"LASTUSE FAIL {f['path']}:{f['line']}: {f['local']} in {f['function']} ({f['kind']} pointer, "
            f"{f['sink']}): last named use line {f['last_use']}, {f['why']}; keep it alive (`_ = {f['local']}^`)")
    for key in stale:
        out(f"LASTUSE STALE allow line matches nothing: {':'.join(key)}")
    if all_kinds:
        for f in found:
            if not f["fails"]:
                out(f"LASTUSE INFO {f['path']}:{f['line']}: {f['local']} ({f['kind']}, {f['sink']})")
    bad = len(failing) + len(errors)
    out(f"LASTUSE {'FAIL' if bad else 'PASS'} {len(files)} .mojo files, derivations {dict(sorted(kinds.items()))}, "
        f"{len(failing)} unfixed, {len(allowed)} allow-listed, {len(stale)} stale allow line(s)")
    return 1 if bad else 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--commit", default=None, help="scan this commit's blobs instead of the work tree")
    ap.add_argument("--root", default=str(ROOT))
    ap.add_argument("--workers", type=int, default=None, help="default os.cpu_count()")
    ap.add_argument("--all-kinds", action="store_true", help="also list the tracked and any-origin derivations")
    a = ap.parse_args(argv)
    return audit(Path(a.root), a.commit, a.workers, a.all_kinds)


if __name__ == "__main__":
    sys.exit(main())
