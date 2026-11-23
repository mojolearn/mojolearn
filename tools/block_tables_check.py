#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""The per-layer tables of autograd/model_blocks.mojo agree, for EVERY block layer code (lane mamba2-nan, 2026-11-23).

A block layer's parameters are described four times, in four functions that the interpreter reads separately:
block_param_sizes (the element counts, the flat vector's layout), block_param_names (the checkpoint names),
block_init_bounds (the seeded init of each tensor) and block_tensor_dims (each tensor's shape for import and
checkpoints). Merge 9e745eb1a8 dropped three lines of the ssd_scan rows (an empty block_tensor_dims branch, the
second name `.D`, the D init) and nothing failed until a fit ran off the end of the init list. This check reads the
source and, for every code from BLOCK_FIRST to BLOCK_LAST:
  - the number of tensors in block_param_sizes, block_param_names and block_init_bounds is the same;
  - block_tensor_dims has a branch exactly when the layer has parameters;
  - no branch of the four tables is empty (a header with no `out.append`);
  - block_shape and block_forward name the code (every layer has a shape rule and a forward).
A `for _ in range(K):` around an append counts K appends. Source reading only: no build, no binding.

    python3 tools/block_tables_check.py [path/to/model_blocks.mojo]     exit 0 when consistent, 1 with the list
"""
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "autograd/model_blocks.mojo"
TABLES = ("block_param_sizes", "block_param_names", "block_init_bounds", "block_tensor_dims")
NAMED = ("block_shape", "block_forward")
_CODE = re.compile(r"\bcode == (LAYER_[A-Z0-9_]+)")
_HEADER = re.compile(r"^\s*(?:el)?if\b.*\bcode == LAYER_")
_RANGE = re.compile(r"^(\s*)for _ in range\((\d+)\)")


def constants(text):
    """{name: value} of the module's `comptime LAYER_* = <int>` lines, and BLOCK_FIRST / BLOCK_LAST."""
    out = {}
    for m in re.finditer(r"^comptime ((?:LAYER|BLOCK)_[A-Z0-9_]+) = (\d+)\s*$", text, re.M):
        out[m.group(1)] = int(m.group(2))
    return out


def function_body(text, name):
    """The lines of `def name(...)` up to the next top-level statement."""
    lines = text.splitlines()
    for i, line in enumerate(lines):
        if line.startswith(f"def {name}(") or line.startswith(f"def {name}["):
            body = []
            for nxt in lines[i + 1:]:
                # a multi-line signature closes with `) raises -> T:` at column 0
                if nxt and not nxt[0].isspace() and not nxt.startswith(("#", ")")):
                    break
                body.append(nxt)
            return body
    raise SystemExit(f"block_tables_check: {name} not found in the source")


def branches(body):
    """[(codes, appends)] per `if/elif code == LAYER_*` branch of a table function, in order. A branch ends at the
    next line indented at or left of its header (another header, an else, the return)."""
    out = []
    i = 0
    while i < len(body):
        line = body[i]
        if not _HEADER.match(line):
            i += 1
            continue
        indent = len(line) - len(line.lstrip())
        codes = _CODE.findall(line)
        appends = 0
        loops = []  # (indent, count) of enclosing `for _ in range(K)`
        j = i + 1
        while j < len(body):
            nxt = body[j]
            stripped = nxt.strip()
            if stripped and not stripped.startswith("#"):
                ind = len(nxt) - len(nxt.lstrip())
                if ind <= indent:
                    break
                while loops and ind <= loops[-1][0]:
                    loops.pop()
                m = _RANGE.match(nxt)
                if m:
                    loops.append((len(m.group(1)), int(m.group(2))))
                elif "out.append(" in nxt:
                    k = 1
                    for _, n in loops:
                        k *= n
                    appends += k
            j += 1
        out.append((codes, appends))
        i = j
    return out


def check(path=SOURCE):
    """The list of disagreements (empty: consistent) and the per-code table {code name: (sizes, names, inits, dims)}."""
    text = Path(path).read_text(encoding="utf-8")
    values = constants(text)
    first, last = values.get("BLOCK_FIRST"), values.get("BLOCK_LAST")
    if first is None or last is None:
        return ["BLOCK_FIRST / BLOCK_LAST not found"], {}
    names = {v: k for k, v in values.items() if k.startswith("LAYER_") and first <= v <= last}
    problems = []
    for v in range(first, last + 1):
        if v not in names:
            problems.append(f"code {v}: no LAYER_ constant between BLOCK_FIRST and BLOCK_LAST")
    counts = {name: [0, 0, 0, 0] for name in names.values()}
    seen = {name: [0, 0, 0, 0] for name in names.values()}
    for t, table in enumerate(TABLES):
        for codes, appends in branches(function_body(text, table)):
            for code in codes:
                if code not in counts:
                    problems.append(f"{table}: {code} is not a block layer code")
                    continue
                seen[code][t] += 1
                counts[code][t] = appends
                if appends == 0:
                    problems.append(f"{table}: the {code} branch is empty")
    for code, (n_sizes, n_names, n_inits, n_dims) in counts.items():
        for t, table in enumerate(TABLES):
            if seen[code][t] > 1:
                problems.append(f"{table}: {code} has {seen[code][t]} branches")
        if not (n_sizes == n_names == n_inits):
            problems.append(f"{code}: {n_sizes} parameter sizes, {n_names} names and {n_inits} inits")
        if (n_sizes > 0) != (seen[code][3] > 0):
            problems.append(f"{code}: {n_sizes} parameter tensors but "
                            f"{'a' if seen[code][3] else 'no'} block_tensor_dims branch")
    for fn in NAMED:
        named = set(_CODE.findall("\n".join(function_body(text, fn))))
        for code in counts:
            if code not in named:
                problems.append(f"{fn}: no rule for {code}")
    return problems, {c: tuple(n) for c, n in counts.items()}


def main(argv):
    path = Path(argv[1]) if len(argv) > 1 else SOURCE
    problems, table = check(path)
    for code, (s, n, i, d) in sorted(table.items()):
        print(f"{code:<26} sizes {s} names {n} inits {i} dims {'yes' if d else 'no'}")
    for p in problems:
        print("DISAGREE", p)
    print(f"BLOCK_TABLES {'consistent' if not problems else 'INCONSISTENT'}: {len(table)} codes, {len(problems)} problems")
    return 0 if not problems else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv))
