"""Two identity_break columns, every `par-*` cell, part by part.

    python3 compare_columns.py <before.json> <after.json> [--changed-only]

A cell's hashes and parts are LISTS, one entry per repeat, never strings: an
`isinstance(v, str)` diff silently skips every part and reads as success (CELL
HASHES ARE LISTS, NOT STRINGS, 2026-09-20). Everything here compares lists, and
`selfcheck()` perturbs a value and REQUIRES the comparison to notice, and feeds
it a string-valued column and requires a refusal, before any real answer is
printed.

`--changed-only` prints the cells that moved and the totals, which is the shape
a regression check wants ("0 verdict changes, 0 moved hashes").
"""
import json
import sys


def load(path):
    with open(path) as stream:
        return json.load(stream)


def parts_of(cell):
    """{part: [value per repeat]} from a cell's `parts` LIST of dicts."""
    out = {}
    for repeat in cell.get("parts") or []:
        if not isinstance(repeat, dict):
            raise ValueError("parts entry is not a dict: %r" % (repeat,))
        for key, value in repeat.items():
            out.setdefault(key, []).append(value)
    return out


def diff_cell(a, b):
    """(divergent, unmoved, missing) part names between two cells."""
    pa, pb = parts_of(a), parts_of(b)
    divergent, unmoved, missing = [], [], []
    for key in sorted(set(pa) | set(pb)):
        va, vb = pa.get(key), pb.get(key)
        if va is None or vb is None:
            missing.append(key)
        elif va != vb:                      # LIST against LIST
            divergent.append(key)
        else:
            unmoved.append(key)
    return divergent, unmoved, missing


def cell_moved(a, b):
    """True when the train hash LIST differs, or a verdict differs."""
    return (a.get("hashes") != b.get("hashes")
            or a.get("verdict") != b.get("verdict"))


def selfcheck():
    base = dict(verdict="STABLE", hashes=["aa", "aa"],
                parts=[dict(x="aa", y="bb"), dict(x="aa", y="bb")])
    same = dict(verdict="STABLE", hashes=["aa", "aa"],
                parts=[dict(x="aa", y="bb"), dict(x="aa", y="bb")])
    d, u, m = diff_cell(base, same)
    assert (d, u, m) == ([], ["x", "y"], []), (d, u, m)
    assert not cell_moved(base, same)
    moved = dict(verdict="STABLE", hashes=["aa", "aa"],
                 parts=[dict(x="aa", y="cc"), dict(x="aa", y="bb")])
    d, u, m = diff_cell(base, moved)
    assert (d, u) == (["y"], ["x"]), (d, u)
    hashed = dict(verdict="STABLE", hashes=["zz", "zz"],
                  parts=[dict(x="aa", y="bb"), dict(x="aa", y="bb")])
    assert cell_moved(base, hashed)
    refused = dict(verdict="REFUSED", hashes=["aa", "aa"],
                   parts=[dict(x="aa", y="bb"), dict(x="aa", y="bb")])
    assert cell_moved(base, refused)
    try:
        diff_cell(dict(parts=["aa"]), same)
    except ValueError:
        pass
    else:
        raise AssertionError("a string-valued parts column was accepted")
    print("selfcheck: a moved part, a moved cell hash, a changed verdict, a clean pair "
          "and a malformed column are each classified correctly\n")


def column_note(tag, column):
    families = (column.get("host") or {}).get("families") or {}
    armed = sorted(n for n, f in families.items() if isinstance(f, dict) and f.get("sabotage"))
    flags = sorted(k for k, v in column.items() if k.endswith("_sabotage") and v)
    print("%-8s commit=%s  par_devices=%s  repeats=%s  complete=%s"
          % (tag, str(column.get("commit"))[:9],
             (column.get("package") or {}).get("par_devices"),
             column.get("repeats"), column.get("complete")))
    print("%-8s %d host families, binding sabotage read-back: %s; column flags: %s"
          % ("", len(families), ", ".join(armed) or "(none)", ", ".join(flags) or "(none)"))


def main():
    selfcheck()
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    changed_only = "--changed-only" in sys.argv
    before, after = load(args[0]), load(args[1])
    column_note("before", before)
    column_note("after", after)
    print()
    ca_all = before.get("cells") or {}
    cb_all = after.get("cells") or {}
    keys = sorted(k for k in set(ca_all) | set(cb_all) if k.startswith("par-"))
    total_div = moved_cells = verdict_changes = 0
    for key in keys:
        a, b = ca_all.get(key), cb_all.get(key)
        if a is None or b is None:
            print("%-32s MISSING (before=%s after=%s)" % (key, a is not None, b is not None))
            continue
        d, u, m = diff_cell(a, b)
        total_div += len(d)
        moved = cell_moved(a, b)
        moved_cells += bool(moved)
        verdict_changes += a.get("verdict") != b.get("verdict")
        if changed_only and not moved and not d:
            continue
        print("== %s" % key)
        print("   verdict      %s -> %s" % (a.get("verdict"), b.get("verdict")))
        print("   cell hashes  %s -> %s" % (a.get("hashes"), b.get("hashes")))
        print("   DIVERGENT parts: %d of %d" % (len(d), len(d) + len(u) + len(m)))
        pa, pb = parts_of(a), parts_of(b)
        for k in sorted(set(pa) | set(pb)):
            tag = "DIVERGENT" if k in d else ("missing  " if k in m else "unmoved  ")
            print("     %-14s %s %s -> %s" % (k, tag, pa.get(k), pb.get(k)))
        for side, col in (("before", a), ("after", b)):
            if col.get("oracle_errors"):
                print("   %s oracle_errors: %s" % (side, col["oracle_errors"][0]))
        if b.get("error"):
            print("   after error: %s" % str(b["error"])[:200])
        print()
    print("cells compared: %d   cells whose hash list or verdict MOVED: %d   "
          "verdict changes: %d   DIVERGENT parts: %d"
          % (len(keys), moved_cells, verdict_changes, total_div))


main()
