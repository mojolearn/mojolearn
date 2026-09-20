"""Before/after comparison of two identity_break columns, part by part.

The hashes a cell records are LISTS, one per repeat, never strings: an
`isinstance(v, str)` diff silently skips every part and reads as success
(CELL HASHES ARE LISTS, NOT STRINGS). Everything here compares lists, and
the self-check at the bottom perturbs a value and requires the comparison
to notice before any real answer is printed.
"""
import json, sys

LANES = ("par-ivf", "par-queries-nn", "par-rbf-sampler", "par-forecast-arima",
         "par-arima", "par-holtwinters", "par-queries-knn", "par-queries-radius")


def load(p):
    return json.load(open(p))


def parts_of(cell):
    """{part: [value per repeat]} from a cell's `parts` LIST of dicts."""
    out = {}
    for rep in cell.get("parts") or []:
        if not isinstance(rep, dict):
            raise ValueError("parts entry is not a dict: %r" % (rep,))
        for k, v in rep.items():
            out.setdefault(k, []).append(v)
    return out


def diff_cell(a, b):
    """(divergent, unmoved, missing) part names between two cells."""
    pa, pb = parts_of(a), parts_of(b)
    keys = sorted(set(pa) | set(pb))
    divergent, unmoved, missing = [], [], []
    for k in keys:
        va, vb = pa.get(k), pb.get(k)
        if va is None or vb is None:
            missing.append(k)
        elif va != vb:                      # LIST against LIST
            divergent.append(k)
        else:
            unmoved.append(k)
    return divergent, unmoved, missing


def selfcheck():
    """A comparison that cannot fail is not a comparison. Perturb one value
    and require diff_cell to see it; perturb none and require silence."""
    base = dict(parts=[dict(x="aa", y="bb"), dict(x="aa", y="bb")])
    same = dict(parts=[dict(x="aa", y="bb"), dict(x="aa", y="bb")])
    d, u, m = diff_cell(base, same)
    assert d == [] and u == ["x", "y"] and m == [], (d, u, m)
    moved = dict(parts=[dict(x="aa", y="cc"), dict(x="aa", y="bb")])
    d, u, m = diff_cell(base, moved)
    assert d == ["y"] and u == ["x"], (d, u, m)
    try:
        diff_cell(dict(parts=["aa"]), same)
    except ValueError:
        pass
    else:
        raise AssertionError("string parts were accepted")
    print("selfcheck: the comparison sees a moved part, a clean pair and a malformed column")


def main():
    selfcheck()
    before, after = load(sys.argv[1]), load(sys.argv[2])
    total_div = 0
    for lane in LANES:
        key = lane + "/base"
        ca, cb = (before.get("cells") or {}).get(key), (after.get("cells") or {}).get(key)
        if ca is None or cb is None:
            print("%-22s MISSING (before=%s after=%s)" % (lane, ca is not None, cb is not None))
            continue
        d, u, m = diff_cell(ca, cb)
        total_div += len(d)
        print("\n== %s" % lane)
        print("   verdict      %s -> %s" % (ca.get("verdict"), cb.get("verdict")))
        print("   cell hashes  %s -> %s" % (ca.get("hashes"), cb.get("hashes")))
        print("   DIVERGENT parts: %d of %d" % (len(d), len(d) + len(u) + len(m)))
        pa, pb = parts_of(ca), parts_of(cb)
        for k in sorted(set(pa) | set(pb)):
            tag = "DIVERGENT" if k in d else ("missing  " if k in m else "unmoved  ")
            print("     %-11s %s %s -> %s" % (k, tag, pa.get(k), pb.get(k)))
        if cb.get("oracle_errors"):
            print("   oracle_errors: %s" % cb["oracle_errors"][0])
    print("\nTOTAL DIVERGENT parts: %d" % total_div)
    for tag, col in (("before", before), ("after", after)):
        fams = (col.get("host") or {}).get("families") or {}
        sab = sorted(n for n, f in fams.items() if isinstance(f, dict) and f.get("sabotage"))
        print("%-6s column: %d host families, sabotage-flagged: %s"
              % (tag, len(fams), ", ".join(sab) or "(none)"))


main()
