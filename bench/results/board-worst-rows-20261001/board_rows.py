import json, sys

# board_rows.py <board.json> <box>: one TSV row per race: ours IDENTICAL median vs the fastest opponent median.
b = json.load(open(sys.argv[1]))
box = sys.argv[2]
wheel = None
for name, race in b["races"].items():
    cells = race.get("cells") or []
    ours = [c for c in cells if c.get("library") == "mojolearn" and c.get("mode") == "identical" and c.get("median_ms")]
    opp = [c for c in cells if c.get("library") != "mojolearn" and c.get("median_ms")]
    if not ours or not opp:
        continue
    o = ours[0]
    wheel = o.get("library_version")
    best = min(opp, key=lambda c: c["median_ms"])
    print("\t".join([box, o.get("family", ""), o.get("lane", ""), o.get("dataset", ""), str(o.get("rows_tag", "")),
                     "%.3f" % o["median_ms"], str(best.get("arm")), "%.3f" % best["median_ms"],
                     "%.2f" % (o["median_ms"] / best["median_ms"]), o.get("device", ""), str(wheel)]))
