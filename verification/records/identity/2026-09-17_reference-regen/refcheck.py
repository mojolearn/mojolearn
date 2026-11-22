"""Every cell part where a NEW column disagrees with the CURRENTLY SHIPPED
reference. Run before regenerating: a newest-wins rebuild would otherwise
silently replace a multi-vendor reference with this one column's value."""
import importlib.util, json, os, sys
ROOT = os.path.expanduser("~/mojolearn-wt/reference-regen")
def load(n, p):
    s = importlib.util.spec_from_file_location(n, p); m = importlib.util.module_from_spec(s)
    sys.modules[n] = m; s.loader.exec_module(m); return m
vref = load("vref", os.path.join(ROOT, "python/mojolearn/_verify_reference.py"))
table = json.load(open(os.path.join(ROOT, "python/mojolearn/verify_reference/table.json")))
parts = vref.PARTS + vref.OPTIONAL_PARTS
rows = []
counts = dict(compared=0, agree=0, differ=0, new=0)
for path in sys.argv[1:]:
    j = json.load(open(path))
    for key, cell in j["cells"].items():
        for part in parts:
            v = vref._part_value(cell, part, min_repeats=2)
            if v is None or v.startswith("n/a:"):
                continue
            lane, _, fixture = key.partition("/")
            ent = vref.entry(table, lane, fixture, part)
            ref = None if ent is None else ent.get("ref")
            if ref is None or str(ref).startswith("n/a"):
                counts["new"] += 1
                continue
            counts["compared"] += 1
            if v == ref:
                counts["agree"] += 1
            else:
                counts["differ"] += 1
                cols = sorted((ent.get("cols") or {}).keys())
                rows.append((key, part, ref, v, ",".join(cols)))
print(json.dumps(counts, sort_keys=True))
for r in sorted(rows):
    print("DIFFERS\t%s\t%s\tshipped=%s\tnew=%s\tshipped_classes=%s" % r)
print("total differing cell parts:", len(rows))
