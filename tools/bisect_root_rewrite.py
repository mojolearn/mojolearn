#!/usr/bin/env python3
"""Rewrite one merged-group member root for tools/bisect_hang_group.sh --root-ddmin (build tooling, never runtime).

    python3 tools/bisect_root_rewrite.py list ROOT
    python3 tools/bisect_root_rewrite.py rewrite ROOT --mode keep|none|loo|loi [--index I] --out FILE

The candidate imports are the single-line `from bindings.X import ...` and `from providers.X import ...` lines of ROOT,
numbered from 0 in file order (`list` prints index, the names each one binds, and the line). Every other import (std,
core, checks) is infrastructure and always kept.

An ITEM is a top-level block (a def/fn/struct with its decorators and indented body, or any other top-level
statement with its continuation lines) or, inside the module builder (the top-level function that calls
PythonModuleBuilder), one body statement (with its continuation lines; comment lines stay). Given the set K of kept
candidate imports, an item is dropped when it mentions (whole word) a name bound by a candidate import outside K, or
the name of a dropped top-level function; this repeats until nothing more drops.

  keep  K = every import: the root as it is (re-emitted through the same code path; the control)
  none  K = no import: only the infrastructure items remain
  loo   leave one out: K = every import except I
  loi   leave one in: K = {I}, closed transitively: an item that mentions a name of K is kept, and every
        candidate import whose names a kept item mentions joins K

`rewrite` writes FILE and prints one line: kept=<indexes> dropped_items=<n> kept_items=<n>.
"""
import argparse
import re
import sys

IMPORT_RE = re.compile(r"^from\s+((?:bindings|providers)\.[\w.]+)\s+import\s+(.+?)\s*(#.*)?$")
TOP_DEF_RE = re.compile(r"^(?:def|fn|struct|trait)\s+(\w+)")


def parse_imports(lines):
    imports = []  # (line number, names, text)
    for n, line in enumerate(lines):
        m = IMPORT_RE.match(line)
        if not m:
            continue
        if "(" in m.group(2) or m.group(2).rstrip().endswith("\\"):
            sys.exit(f"bisect_root_rewrite: multi-line import at line {n + 1} is not supported: {line.strip()}")
        names = []
        for part in m.group(2).split(","):
            words = part.split()
            if not words:
                continue
            names.append(words[-1] if len(words) == 3 and words[1] == "as" else words[0])
        imports.append((n, names, line.rstrip("\n")))
    return imports


def paren_depth(text):
    # Rough statement grouping: brackets outside string literals and comments.
    depth, quote = 0, None
    for i, ch in enumerate(text):
        if quote:
            if ch == quote and text[i - 1] != "\\":
                quote = None
        elif ch in "\"'":
            quote = ch
        elif ch == "#":
            break
        elif ch in "([{":
            depth += 1
        elif ch in ")]}":
            depth -= 1
    return depth


def group_statements(lines, start, end):
    """[start, end) -> list of (first, last+1) statement spans, joined while brackets are open."""
    spans, i = [], start
    while i < end:
        j, depth = i, paren_depth(lines[i])
        while depth > 0 and j + 1 < end:
            j += 1
            depth += paren_depth(lines[j])
        spans.append((i, j + 1))
        i = j + 1
    return spans


def parse_items(lines, import_lines):
    """Top-level blocks and builder-body statements: dicts with span, kind, name, text."""
    items, n, total = [], 0, len(lines)
    while n < total:
        line = lines[n]
        if n in import_lines or not line.strip() or line.startswith((" ", "\t")):
            n += 1
            continue
        start = n
        if line.lstrip().startswith("#"):
            n += 1
            continue
        while lines[n].startswith("@") and n + 1 < total:
            n += 1
        m = TOP_DEF_RE.match(lines[n])
        if m:
            end = n + 1
            while end < total and (not lines[end].strip() or lines[end].startswith((" ", "\t"))):
                end += 1
            while end > n + 1 and not lines[end - 1].strip():
                end -= 1
            body = "".join(lines[start:end])
            if "PythonModuleBuilder(" in body:
                # the builder: header lines up to the first body statement are kept; each body statement is an item
                items.append({"span": (start, n + 1), "kind": "builder-head", "name": m.group(1), "text": ""})
                for s, e in group_statements(lines, n + 1, end):
                    text = "".join(lines[s:e])
                    if text.lstrip().startswith("#") or not text.strip():
                        items.append({"span": (s, e), "kind": "keep", "name": None, "text": ""})
                    else:
                        items.append({"span": (s, e), "kind": "stmt", "name": None, "text": text})
            else:
                items.append({"span": (start, end), "kind": "def", "name": m.group(1), "text": body})
            n = end
        else:
            s, e = group_statements(lines, n, total)[0]
            items.append({"span": (s, e), "kind": "top", "name": None, "text": "".join(lines[s:e])})
            n = e
    return items


def mentions(text, names):
    return any(re.search(r"(?<![\w.])" + re.escape(x) + r"\b", text) for x in names)


def solve(imports, items, kept):
    """Drop every item that mentions a name of an import outside kept or a dropped def, to a fixpoint."""
    kept = set(kept)
    outside = [x for i, (_, names, _) in enumerate(imports) if i not in kept for x in names]
    dropped_defs, dropped = [], set()
    changed = True
    while changed:
        changed = False
        for k, item in enumerate(items):
            if k in dropped or item["kind"] not in ("def", "stmt", "top"):
                continue
            if mentions(item["text"], outside + dropped_defs):
                dropped.add(k)
                if item["kind"] == "def":
                    dropped_defs.append(item["name"])
                changed = True
    return kept, dropped


def close_loi(imports, items, index):
    """{index} plus every import a kept item needs: an item mentioning a kept name (or a kept def) pulls in
    every import it mentions, transitively."""
    kept, defs = {index}, []
    while True:
        kept_names = [x for i in kept for x in imports[i][1]] + defs
        grow, grow_defs = set(kept), list(defs)
        for item in items:
            if item["kind"] in ("def", "stmt", "top") and mentions(item["text"], kept_names):
                grow |= {i for i, (_, names, _) in enumerate(imports) if mentions(item["text"], names)}
                if item["kind"] == "def" and item["name"] not in grow_defs:
                    grow_defs.append(item["name"])
        if grow == kept and grow_defs == defs:
            return kept
        kept, defs = grow, grow_defs


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=("list", "rewrite"))
    ap.add_argument("root")
    ap.add_argument("--mode", choices=("keep", "none", "loo", "loi"))
    ap.add_argument("--index", type=int)
    ap.add_argument("--out")
    args = ap.parse_args(argv)
    with open(args.root) as handle:
        lines = handle.readlines()
    imports = parse_imports(lines)
    if not imports:
        sys.exit("bisect_root_rewrite: no `from bindings./providers. import` lines in " + args.root)
    if args.command == "list":
        for i, (_, names, text) in enumerate(imports):
            print(f"{i}\t{','.join(names)}\t{text}")
        return 0
    if not args.mode or not args.out:
        sys.exit("bisect_root_rewrite: rewrite needs --mode and --out")
    if args.mode in ("loo", "loi") and (args.index is None or not 0 <= args.index < len(imports)):
        sys.exit(f"bisect_root_rewrite: --index must be 0..{len(imports) - 1}")
    items = parse_items(lines, {n for n, _, _ in imports})
    every = set(range(len(imports)))
    kept = {"keep": every, "none": set(), "loo": every - {args.index}}.get(args.mode)
    if args.mode == "loi":
        kept = close_loi(imports, items, args.index)
    kept, dropped = solve(imports, items, kept)
    drop_lines = {n for i, (n, _, _) in enumerate(imports) if i not in kept}
    for k in dropped:
        drop_lines.update(range(*items[k]["span"]))
    with open(args.out, "w") as handle:
        handle.writelines(line for n, line in enumerate(lines) if n not in drop_lines)
    live = sum(1 for k, item in enumerate(items) if item["kind"] in ("def", "stmt", "top") and k not in dropped)
    print(f"kept={','.join(map(str, sorted(kept))) or '-'} dropped_items={len(dropped)} kept_items={live}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
