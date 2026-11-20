#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Check bindings/cpython_layout.mojo against live CPython objects (ctypes).

    python3 tools/cpython_layout_check.py                  # the running interpreter
    python3 tools/cpython_layout_check.py PYTHON...         # these interpreters, in parallel
    python3 tools/cpython_layout_check.py --installed       # every python3.10 .. python3.14 on PATH

For each interpreter it prints one TSV line per check: version, check, the
offset the table holds, the offset found on the live object, verdict. A
group the interpreter's version row lists must pass every one of its checks
(exit 1 otherwise); a group the row does not list is reported (`info`) and
never fails. The public layouts the Mojo code also reads (Py_buffer's ten
words, PyObject.ob_type at +8) are checked on every interpreter.

Parallel plan: one subprocess per interpreter, as many workers as
interpreters up to os.cpu_count() (each is a few ms of ctypes reads).
"""
from __future__ import annotations

import concurrent.futures
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TABLE = ROOT / "bindings" / "cpython_layout.mojo"
GROUP_NAMES = {"LAYOUT_DICT_VERSION": "dict", "LAYOUT_TYPE_VERSION": "type", "LAYOUT_ARRAY_BLOCK": "array"}


def parse_table(path: Path = TABLE) -> tuple[dict[str, int], dict[int, set[str]]]:
    """The offsets (`comptime NAME = <int>`) and the version rows
    (`if minor == N:` then `return A | B`) of the Mojo table."""
    text = path.read_text()
    offsets = {m.group(1): int(m.group(2)) for m in re.finditer(r"^comptime ((?:DICT|TYPE|ARRAY)_[A-Z_]+) = (\d+)$", text, re.M)}
    rows: dict[int, set[str]] = {}
    for m in re.finditer(r"^\s+if minor == (\d+):\n\s+return ([A-Z_ |]+)$", text, re.M):
        rows[int(m.group(1))] = {GROUP_NAMES[g.strip()] for g in m.group(2).split("|")}
    return offsets, rows


# ------------------------------------------------------------- measurement
# Runs inside the interpreter under test (no imports beyond the standard
# library; offsets passed in as JSON).
PROBE = r'''
import array, ctypes, json, sys, sysconfig
off = json.loads(sys.argv[1])
api = ctypes.pythonapi
def w64(a, o): return ctypes.c_uint64.from_address(a + o).value
def w32(a, o): return ctypes.c_uint32.from_address(a + o).value
out = []
def put(check, table, found, ok): out.append([check, table, found, bool(ok)])
def scan(a, limit, pred, size=8):
    for o in range(0, limit, size):
        if pred(o): return o
    return -1
VALID = 1 << 19

# PyTypeObject.tp_flags: equals PyType_GetFlags and type.__flags__
api.PyType_GetFlags.restype = ctypes.c_ulong
api.PyType_GetFlags.argtypes = [ctypes.py_object]
cls = type("_MojolearnLayoutProbe", (), {})
a = id(cls)
flags = api.PyType_GetFlags(cls)
found = scan(a, 512, lambda o: w64(a, o) == flags == cls.__flags__)
put("type.tp_flags", off["TYPE_TP_FLAGS"], found, found == off["TYPE_TP_FLAGS"])
# PyTypeObject.tp_version_tag: a lookup assigns a nonzero tag with the valid
# flag; an attribute assignment clears the flag; the next lookup assigns a
# different tag (the same proof bindings/py_version_tags.mojo runs).
fo, vo = off["TYPE_TP_FLAGS"], off["TYPE_TP_VERSION_TAG"]
cls.__init__
f1, t1 = w64(a, fo), w32(a, vo)
before = {o: w32(a, o) for o in range(0, 512, 4)}
cls.mojolearn_probe = 1
f2 = w64(a, fo)
cls.__init__
f3, t3 = w64(a, fo), w32(a, vo)
after = {o: w32(a, o) for o in range(0, 512, 4)}
moved = [o for o in range(0, 512, 4) if before[o] != after[o] and before[o] and after[o]]
ok = bool(f1 & VALID) and t1 != 0 and not (f2 & VALID) and bool(f3 & VALID) and t3 not in (0, t1)
put("type.tp_version_tag", vo, ",".join(map(str, moved)) or "-", ok)

# PyDictObject.ma_used and ma_version_tag
d, e = {}, {}
a = id(d)
used0, tag0 = w64(a, off["DICT_MA_USED"]), w64(a, off["DICT_MA_VERSION_TAG"])
tag_e = w64(id(e), off["DICT_MA_VERSION_TAG"])
d["k"] = 1
used1, tag1 = w64(a, off["DICT_MA_USED"]), w64(a, off["DICT_MA_VERSION_TAG"])
d["k"] = 2
tag2 = w64(a, off["DICT_MA_VERSION_TAG"])
found = scan(a, 64, lambda o: w64(a, o) == 1 and o >= 16)
put("dict.ma_used", off["DICT_MA_USED"], found, used0 == 0 and used1 == 1 and found == off["DICT_MA_USED"])
put("dict.ma_version_tag", off["DICT_MA_VERSION_TAG"], off["DICT_MA_VERSION_TAG"],
    tag0 != 0 and tag_e != tag0 and len({tag0, tag1, tag2, tag_e}) == 4)

# array.array arrayobject: ob_size, ob_item, allocated
x = array.array("d", [0.0] * 5)
a = id(x)
item = x.buffer_info()[0]
put("array.ob_size", off["ARRAY_OB_SIZE"], scan(a, 64, lambda o: o >= 16 and w64(a, o) == 5), w64(a, off["ARRAY_OB_SIZE"]) == 5)
put("array.ob_item", off["ARRAY_OB_ITEM"], scan(a, 64, lambda o: w64(a, o) == item), w64(a, off["ARRAY_OB_ITEM"]) == item)
put("array.allocated", off["ARRAY_ALLOCATED"], off["ARRAY_ALLOCATED"], w64(a, off["ARRAY_ALLOCATED"]) >= 5)
y = array.array("d")
b = id(y)
put("array.empty_words", 0, 0, w64(b, off["ARRAY_OB_SIZE"]) == 0 and w64(b, off["ARRAY_OB_ITEM"]) == 0 and w64(b, off["ARRAY_ALLOCATED"]) == 0)

# PUBLIC: PyObject.ob_type at +8 (stdlib Py_TYPE)
put("public.ob_type", 8, scan(id(x), 32, lambda o: w64(id(x), o) == id(array.array)), w64(id(x), 8) == id(array.array))
# PUBLIC: Py_buffer, ten words (core/boundary_layout.mojo); word 10 is a guard
src = array.array("d", range(6))
view = memoryview(src).cast("B").cast("d", [2, 3])
words = (ctypes.c_ssize_t * 11)()
words[10] = 0x5A5A
api.PyObject_GetBuffer.argtypes = [ctypes.py_object, ctypes.c_void_p, ctypes.c_int]
api.PyBuffer_Release.argtypes = [ctypes.c_void_p]
rc = api.PyObject_GetBuffer(view, ctypes.addressof(words), 0x11C)
packed = words[4] & 0xFFFFFFFFFFFFFFFF
ok = (rc == 0 and words[0] == src.buffer_info()[0] and words[1] == id(view) and words[2] == 48 and words[3] == 8
      and (packed & 0xFFFFFFFF) == 0 and (packed >> 32) == 2 and ctypes.string_at(words[5]) == b"d"
      and ctypes.c_ssize_t.from_address(words[6]).value == 2 and ctypes.c_ssize_t.from_address(words[6] + 8).value == 3
      and ctypes.c_ssize_t.from_address(words[7]).value == 24 and ctypes.c_ssize_t.from_address(words[7] + 8).value == 8
      and words[8] == 0 and words[10] == 0x5A5A)
api.PyBuffer_Release(ctypes.addressof(words))
put("public.Py_buffer_words", 10, 10, ok)

gil = not sysconfig.get_config_var("Py_GIL_DISABLED")
print(json.dumps({"version": sys.version.split()[0], "minor": sys.version_info[1], "free_threaded": not gil,
                  "pointer": ctypes.sizeof(ctypes.c_void_p), "checks": out}))
'''

CHECK_GROUP = {"type": "type", "dict": "dict", "array": "array", "public": "public"}


def measure(python: str, offsets: dict[str, int]) -> dict:
    run = subprocess.run([python, "-I", "-c", PROBE, json.dumps(offsets)], capture_output=True, text=True, timeout=60)
    if run.returncode != 0:
        return {"python": python, "error": run.stderr.strip().splitlines()[-1:] or ["exit %d" % run.returncode]}
    result = json.loads(run.stdout)
    result["python"] = python
    return result


def verdicts(result: dict, rows: dict[int, set[str]]) -> tuple[list[list[str]], bool]:
    """TSV rows and whether every check the version row requires passed."""
    if "error" in result:
        return [[result["python"], "probe", "-", "-", "ERROR " + " ".join(result["error"])]], False
    row = set() if result["free_threaded"] or result["pointer"] != 8 else rows.get(result["minor"], set())
    lines, good = [], True
    for check, table, found, ok in result["checks"]:
        group = check.split(".")[0]
        required = group == "public" or group in row
        verdict = ("verified" if ok else "WRONG") if required else ("info:matches" if ok else "info:differs (not read on this version)")
        good = good and (ok or not required)
        lines.append([result["version"] + ("t" if result["free_threaded"] else ""), check, str(table), str(found), verdict])
    return lines, good


def installed() -> list[str]:
    found = []
    for minor in range(10, 15):
        path = shutil.which("python3.%d" % minor)
        if path:
            found.append(path)
    return found


def main(argv: list[str]) -> int:
    pythons = [p for p in argv if p != "--installed"]
    if "--installed" in argv:
        pythons = installed() + pythons
    pythons = pythons or [sys.executable]
    offsets, rows = parse_table()
    workers = max(1, min(len(pythons), os.cpu_count() or 1))
    with concurrent.futures.ThreadPoolExecutor(workers) as pool:
        results = list(pool.map(lambda p: measure(p, offsets), pythons))
    status = 0
    print("version\tcheck\ttable_offset\tfound_offset\tverdict")
    for result in results:
        lines, good = verdicts(result, rows)
        for line in lines:
            print("\t".join(line))
        status |= 0 if good else 1
    return status


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
