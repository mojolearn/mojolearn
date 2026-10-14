#!/usr/bin/env python3
"""Attach recovered transform quality to the same retained Xq inference outputs.

This reuses verified quality receipts. It runs no model, GPU work or timer.
"""
import argparse
import copy
import json
from pathlib import Path
import tarfile

import bench_board_evidence as E
import race_fairness as F
from recover_transform_quality import verify_observations


def attach(race, source):
    fit = {c["arm"]: c for c in race["cells"]}
    changed = 0
    for cell in race.get("infer_cells", []):
        own = fit.get(cell["arm"], {})
        prior = (own.get("measurement") or {}).get("quality_recovery") or {}
        # FAIRNESS FIX (2026-10-13 audit): transform recovery originally
        # recognized batch='test', but algos records the query batch as Xq.
        # The saved output digest binds this inference to the scored arrays
        # used for quality. A different batch/output is never assigned it.
        if (cell.get("status") != "ok" or cell.get("batch") != "Xq"
                or not cell.get("hash") or cell["hash"] != own.get("hash")
                or (prior.get("artifact") or {}).get("status") != "verified"
                or any(k != "output_shape" for k in cell.get("quality") or {})):
            continue
        cell["quality_superseded"] = copy.deepcopy(cell.get("quality"))
        cell["quality"] = copy.deepcopy(own["quality"])
        cell["quality_basis"] = "same retained scored Xq output digest as the verified transform quality receipt"
        record = cell["measurement"]
        previous_id = record.pop("id")
        record["inference_quality_recovery"] = dict(previous_id=previous_id,
            fit_measurement=own["measurement"]["id"], artifact=prior["artifact"], source=source)
        record["id"] = E.digest(E.encoded(record))
        changed += 1
    return changed


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input-root", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    root = Path(args.out)
    root.mkdir(parents=True, exist_ok=True)
    source, _ = E.source_snapshot(root / "source")
    source = E.public_ref(E.upload(source))
    rows = []
    for path in sorted(Path(args.input_root).glob("*/board.json")):
        board = json.loads(path.read_text())
        race = next(iter(board["races"].values()))
        dest = root / path.parent.name
        dest.mkdir()
        oldref = race["evidence_receipt"]
        E.fetch(oldref, dest / "original-receipt.tar.gz")
        with tarfile.open(dest / "original-receipt.tar.gz") as archive:
            verified = json.load(archive.extractfile("result.json"))
        verify_observations(race, verified)
        changed = attach(race, source)
        race["inference_quality_previous_receipt"] = oldref
        receipt = E.upload(E.archive(dest / "receipt.tar.gz", {"result.json": E.encoded(race)}))
        race["evidence_receipt"] = E.public_ref(receipt)
        F.apply(board)
        E.write(dest / "board.json", board)
        rows.append(dict(id=race["id"], inference_cells_recovered=changed, board=str(dest / "board.json")))
        E.write(root / "summary.json", rows)
        print(json.dumps(rows[-1]), flush=True)


if __name__ == "__main__":
    main()
