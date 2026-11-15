#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""How big the Python boundary is: a read-only report, never a gate (AGENTS.md "The Python source guards are retired").

    python3 tools/python_boundary_report.py [--ref REF] [--json OUT] [--top 20]

The boundary files are the ones tools/python_free_dirs.sh admits: its `boundary_allow` patterns (read from the
script itself, so this report holds no second copy) under bindings/ and providers/python/, plus the `allow:` lines
of tools/python_free_doors.txt. It prints the boundary files and lines per pattern and, for every `*_door.mojo`,
the code lines outside its unpack/call/pack, with the largest listed (a door that computes is an incomplete
migration). That count is a static estimate: top-level definitions that name no Python (native work living in a
door) plus the bodies of `for`/`while` loops not tagged `# driver-loop` inside the Python ones.

Files come from git at REF (default HEAD), so a sparse or dirty worktree is fine. One read of the tree; it takes
about a second, so it is not split across cores.
"""
import argparse
import fnmatch
import json
import re
import subprocess
import sys

FENCE = "tools/python_free_dirs.sh"
DOORS = "tools/python_free_doors.txt"
ROOTS = ("bindings/", "providers/python/")   # the directories the fence's boundary check scans
PY_TOKEN = re.compile(r"PythonObject|\bPython\b|PyObject|GILReleased|\bpy_[A-Za-z_]+\(|_unpack\b|_pack\b|"
                      r"downcast_value_ptr|PythonModuleBuilder|lookup_py_type_object|ConvertibleFromPython|"
                      r"ConvertibleToPython|PythonConvertible")
TOP_DEF = re.compile(r"^(?:@\w+.*\n)*(def|fn|struct|trait)\s", re.M)
LOOP = re.compile(r"^(\s*)(for|while)\b")


def git(*args, inp=None):
    r = subprocess.run(["git", *args], capture_output=True, input=inp)
    if r.returncode:
        raise SystemExit(f"python_boundary_report: git {' '.join(args[:3])} failed: {r.stderr.decode().strip()}")
    return r.stdout


def mojo_texts(ref):
    blobs = {}
    for ln in git("ls-tree", "-r", "--full-tree", ref).decode("utf-8", "replace").splitlines():
        meta, _, path = ln.partition("\t")
        bits = meta.split()
        if len(bits) == 3 and bits[1] == "blob" and path.endswith(".mojo") and path.startswith(ROOTS):
            blobs[path] = bits[2]
    need = list(dict.fromkeys(blobs.values()))
    out = git("cat-file", "--batch", inp="".join(s + "\n" for s in need).encode())
    texts, i = {}, 0
    for sha in need:
        nl = out.index(b"\n", i)
        head = out[i:nl].split()
        i = nl + 1
        size = int(head[2]) if len(head) >= 3 and head[1] != b"missing" else 0
        texts[sha] = out[i:i + size].decode("utf-8", "replace")
        i += size + (1 if size or len(head) >= 3 else 0)
    return {p: texts[s] for p, s in blobs.items()}


def patterns(ref):
    """(the fence's boundary patterns, the allow: lines), as tools/python_free_dirs.sh reads them at REF."""
    fence = git("show", f"{ref}:{FENCE}").decode()
    m = re.search(r"^boundary_allow=\((.*?)\)", fence, re.M | re.S)
    if not m:
        raise SystemExit(f"python_boundary_report: no boundary_allow=( ... ) in {FENCE} at {ref}")
    pats = re.findall(r"'([^']+)'", m.group(1))
    allowed = []
    for ln in git("show", f"{ref}:{DOORS}").decode().splitlines():
        ln = ln.split("#", 1)[0].strip()
        if ln.startswith("allow:"):
            allowed.append(ln[len("allow:"):].strip())
    return pats, allowed


def matches(path, pat):
    return fnmatch.fnmatchcase(path if "/" in pat else path.rsplit("/", 1)[-1], pat)


def code_lines(text):
    return sum(1 for ln in text.split("\n") if ln.strip() and not ln.lstrip().startswith("#"))


def door_outside(text):
    """Code lines of one door outside its unpack/call/pack (static estimate; see the module docstring)."""
    starts = [m.start() for m in TOP_DEF.finditer(text)]
    native = loops = 0
    for k, start in enumerate(starts):
        block = text[start:starts[k + 1] if k + 1 < len(starts) else len(text)]
        body, opened = [], False
        for ln in block.split("\n"):
            if opened and ln and not ln[0].isspace() and not ln.startswith(("#", ")", "]")):
                break
            opened = opened or ln.startswith(("def ", "fn ", "struct ", "trait "))
            body.append(ln)
        if not PY_TOKEN.search("\n".join(body)):
            native += code_lines("\n".join(body))
            continue
        i = 0
        while i < len(body):
            m = LOOP.match(body[i])
            if not m or "driver-loop" in body[i]:
                i += 1
                continue
            indent, j = len(m.group(1)), i + 1
            while j < len(body) and (not body[j].strip() or len(body[j]) - len(body[j].lstrip()) > indent):
                j += 1
            loops += code_lines("\n".join(body[i:j]))
            i = j
    return native + loops


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--ref", default="HEAD")
    ap.add_argument("--json", default="")
    ap.add_argument("--top", type=int, default=20)
    a = ap.parse_args(argv)
    sha = git("rev-parse", "--verify", a.ref + "^{commit}").decode().strip()
    pats, allowed = patterns(sha)
    by_kind, doors = {}, []
    for path, text in sorted(mojo_texts(sha).items()):
        kind = "allow:" if any(matches(path, p) for p in allowed) else next(
            (p for p in pats if matches(path, p)), None)
        if kind is None:
            continue
        row = by_kind.setdefault(kind, {"files": 0, "lines": 0, "code_lines": 0})
        row["files"] += 1
        row["lines"] += text.count("\n") + 1
        row["code_lines"] += code_lines(text)
        if kind == "*_door.mojo":
            doors.append({"path": path, "code_lines": code_lines(text), "outside": door_outside(text)})
    doors.sort(key=lambda r: (-r["outside"], r["path"]))
    files = sum(r["files"] for r in by_kind.values())
    lines = sum(r["lines"] for r in by_kind.values())
    outside = sum(r["outside"] for r in doors)
    print(f"PYTHON_BOUNDARY {sha[:10]} files={files} lines={lines} doors={len(doors)} "
          f"door_lines_outside_unpack_call_pack={outside} (a number, never a gate)")
    for kind, row in sorted(by_kind.items(), key=lambda kv: -kv[1]["lines"]):
        print(f"  {kind:28s} files={row['files']:4d} lines={row['lines']:7d} code_lines={row['code_lines']:7d}")
    print(f"  largest doors by lines outside unpack/call/pack (outside / code lines):")
    for r in doors[:a.top]:
        print(f"  {r['outside']:5d} / {r['code_lines']:5d}  {r['path']}")
    if a.json:
        with open(a.json, "w") as f:
            json.dump({"commit": sha, "files": files, "lines": lines, "by_pattern": by_kind,
                       "door_lines_outside_unpack_call_pack": outside, "doors": doors}, f, indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
