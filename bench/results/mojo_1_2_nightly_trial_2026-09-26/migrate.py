#!/usr/bin/env python3
"""Mechanical Mojo 1.0 -> 1.2-nightly renames over every tracked .mojo file.

  1. std.gpu -> max.gpu                    (std.gpu became private in Mojo 1.1)
  2. memcpy/memset/memset_zero -> unsafe_* (renamed in Mojo 1.1)
  3. InlineArray -> Array                  (alias removed in Mojo 1.1)
  4. `@parameter` + `if`/`for` -> `comptime if`/`comptime for` (removed in 1.1)
Prints per-rule counts. Usage: migrate.py <repo root>
"""
import re
import subprocess
import sys

root = sys.argv[1]
files = subprocess.run(["git", "-C", root, "ls-files", "*.mojo"], capture_output=True, text=True).stdout.split()
if not files:  # not a git checkout (an archive copy): walk it
    import os
    files = [os.path.relpath(os.path.join(d, f), root) for d, _, fs in os.walk(root)
             if "/.pixi" not in d for f in fs if f.endswith(".mojo")]
counts = dict(gpu=0, mem=0, inline=0, param=0, files=0)
MEM = ("memcpy", "memset_zero", "memset")


def fix_param(text):
    lines = text.split("\n")
    out = []
    i = 0
    n = 0
    while i < len(lines):
        m = re.match(r"^(\s*)@parameter\s*$", lines[i])
        if m:
            j = i + 1
            while j < len(lines) and lines[j].strip() == "":
                j += 1
            if j < len(lines):
                m2 = re.match(r"^(\s*)(if|for)\b(.*)$", lines[j])
                if m2 and m2.group(1) == m.group(1):
                    out.append(m.group(1) + "comptime " + m2.group(2) + m2.group(3))
                    n += 1
                    i = j + 1
                    continue
        out.append(lines[i])
        i += 1
    return "\n".join(out), n


for rel in files:
    path = f"{root}/{rel}"
    try:
        src = open(path).read()
    except OSError:
        continue
    t = src
    t, k = re.subn(r"\bstd\.gpu\b", "max.gpu", t)
    counts["gpu"] += k
    if re.search(r"from std\.memory import[^\n]*\b(memcpy|memset|memset_zero)\b", t) or \
            re.search(r"from std\.memory import \([^)]*\b(memcpy|memset|memset_zero)\b", t):
        for name in MEM:
            t, k = re.subn(r"(?<![\w.])" + name + r"\b", "unsafe_" + name, t)
            counts["mem"] += k
    t, k = re.subn(r"\bInlineArray\b", "Array", t)
    counts["inline"] += k
    t, k = fix_param(t)
    counts["param"] += k
    if t != src:
        counts["files"] += 1
        open(path, "w").write(t)
print(counts)
