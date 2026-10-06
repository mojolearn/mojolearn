#!/usr/bin/env python3
"""Select a frozen RF/ET A/B batch; emits a plan, never launches or rebuilds.

Example: python tools/select_identical_ab.py --families rf --candidates rf-k2
Necessary baselines are included automatically. Full model state is captured
in the measurement run; Apple timings never vote on IDENTICAL defaults.
"""
import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--families", default="rf,et", help="Comma-separated rf,et")
    parser.add_argument("--candidates", help="Comma-separated candidate names; defaults to all selected families")
    parser.add_argument("--vendors", default="nvidia,amd")
    args = parser.parse_args()
    catalog = json.loads((Path(__file__).resolve().parents[1] / "experiments/identical_speed/selected-batch.json").read_text())
    families = list(dict.fromkeys(args.families.split(",")))
    vendors = list(dict.fromkeys(args.vendors.split(",")))
    if not set(families) <= catalog["families"].keys():
        parser.error("families must be rf and/or et")
    if not set(vendors) <= {"nvidia", "amd"}:
        parser.error("vendors must be nvidia and/or amd")
    allowed = {c for f in families for c in catalog["families"][f]["candidates"]}
    candidates = set(args.candidates.split(",")) if args.candidates else allowed
    if not candidates or not candidates <= allowed:
        parser.error("candidate must belong to a selected family: " + ", ".join(sorted(allowed)))
    cells = []
    for vendor in vendors:
        for family in families:
            spec = catalog["families"][family]
            chosen = [c for c in spec["candidates"] if c in candidates]
            if not chosen:
                continue
            for dataset in spec["datasets"]:
                for profile in [spec["baseline"], *chosen]:
                    cells.append(dict(vendor=vendor, family=family, dataset=dataset, profile=profile))
    print(json.dumps(dict(schema=1, cells=cells, expected_cells=len(cells),
                         measurement=catalog["measurement"], retained_binaries=catalog["retained_binaries"],
                         launch=False, promotion=False), indent=2))


if __name__ == "__main__":
    main()
