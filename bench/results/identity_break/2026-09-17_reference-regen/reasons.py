"""The reason PUBLIC_PENDING_LANES should carry for each held-back lane,
derived from the table and the harness the same way test_host_surface checks
it. Prints lane, current reason, derived reason."""
import importlib.util, json, os, sys, types
ROOT = os.path.expanduser("~/mojolearn-wt/reference-regen"); PKG = os.path.join(ROOT, "python/mojolearn")
pkg = types.ModuleType("mlpkg"); pkg.__path__ = [PKG]; sys.modules["mlpkg"] = pkg
def sub(n):
    s = importlib.util.spec_from_file_location("mlpkg." + n, os.path.join(PKG, n + ".py"))
    m = importlib.util.module_from_spec(s); sys.modules["mlpkg." + n] = m; setattr(pkg, n, m)
    s.loader.exec_module(m); return m
hs = sub("host_surface")
s = importlib.util.spec_from_file_location("idb", os.path.join(ROOT, "tools/identity_break.py"))
h = importlib.util.module_from_spec(s); sys.modules["idb"] = h; s.loader.exec_module(h)
table = json.load(open(sys.argv[1] if len(sys.argv) > 1 else os.path.join(PKG, "verify_reference/table.json")))
with_cells = {k.partition("/")[0] for k in table["cells"]}
revs = h.LANE_REVISIONS
trevs = table.get("lane_revisions") or {}
stale = {l for l, r in revs.items() if l in with_cells and trevs.get(l) != r}
def classes(lane):
    out = set()
    for key, cell in table["cells"].items():
        if key.partition("/")[0] != lane:
            continue
        for e in cell.values():
            if e.get("ref") is None or e.get("conflict"):
                continue
            out.update(e.get("cols", {}))
    return out
for lane, why in hs.PUBLIC_PENDING_LANES.items():
    if why == "own record":
        derived = "own record"
    elif lane not in with_cells:
        derived = "no reference"
    elif lane in stale:
        derived = "stale reference"
    else:
        cl = classes(lane)
        derived = "one column" if len(cl) < 2 else "unwatched"
    mark = "  " if derived == why else "->"
    print(f"{mark} {lane:38s} now={why:24s} derived={derived:16s} classes={','.join(sorted(classes(lane))) or '-'}")
