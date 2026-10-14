#!/usr/bin/env python3
"""Derive missing transform quality from verified retained scored arrays.

No GPU call or timing is repeated. Original observations and receipts stay in
the new receipt chain. Only deterministic sklearn-reference transform lanes
are supported; other missing evidence remains unresolved.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import tarfile

import numpy as np

import bench_board_algos as A
import bench_board_evidence as E
import race_fairness as F


def verify_inputs(arrays, metadata):
    for name, expected in metadata["arrays"].items():
        a = np.ascontiguousarray(arrays[name])
        if (list(a.shape) != expected["shape"] or str(a.dtype) != expected["dtype"]
                or A._tool("classical_two_datasets").sha256_array(a) != expected["sha256"]):
            raise ValueError("prepared input differs from scored recipe: " + name)


def verify_outputs(outputs, scored_digest):
    h = hashlib.sha256()
    for key in sorted(outputs):
        h.update(key.encode())
        h.update(np.ascontiguousarray(outputs[key]).data)
    if h.hexdigest()[:16] != scored_digest:
        raise ValueError("retained output differs from the scored execution")


def verify_observations(race, original):
    previous = {c["measurement"]["id"]: c for field in ("cells", "infer_cells")
                for c in original.get(field, []) if c.get("measurement")}
    for field in ("cells", "infer_cells"):
        for cell in race.get(field, []):
            old = previous.get((cell.get("measurement") or {}).get("id"))
            if old is None or any(old.get(k) != cell.get(k) for k in (
                    "measurement", "params", "quality", "median_ms", "times_ms", "warmup_ms",
                    "status", "fitted_parameters", "iterations", "comparability", "hash")):
                raise ValueError("original verified receipt differs from recovery observations")


def retained_tree(receipts, directory):
    """Verify whole retained archives, then materialize selected regular files."""
    root = Path(directory)
    inventory = {}
    for item in receipts:
        archive = Path(item["path"])
        ref = item["reference"]
        if ref.get("status") != "verified" or E.file_hash(archive) != ref["sha256"]:
            raise ValueError("retained archive checksum or verification is missing")
        with tarfile.open(archive, mode="r|gz") as stream:
            for member in stream:
                if not member.isfile() or not member.name.startswith((
                        "root/fairness-cache/algos-data/", "root/fairness-evidence/measured/algos.")):
                    continue
                rel = Path(member.name)
                if rel.is_absolute() or ".." in rel.parts:
                    raise ValueError("unsafe archive member")
                data = stream.extractfile(member).read()
                target = root / rel
                target.parent.mkdir(parents=True, exist_ok=True)
                if target.exists() and E.file_hash(target) != E.digest(data):
                    raise ValueError("local retained copy changed: " + str(rel))
                if not target.exists():
                    target.write_bytes(data)
                inventory[str(target)] = dict(archive=E.public_ref(ref), member=member.name,
                                             sha256=E.digest(data))
    return inventory


def recover(board_path, verified, inventory, destination, source):
    path, destination = Path(board_path), Path(destination)
    if str(path) not in inventory:
        raise ValueError("board is not in the verified original archive")
    board = json.loads(path.read_text())
    if len(board["races"]) != 1:
        raise ValueError("one original recipe per recovery")
    rid, race = next(iter(board["races"].items()))
    lane, dataset = race["lane"], race["dataset"]
    if A.quality_kind(lane) not in ("vs-sklearn", "labels", "multilabel"):
        raise ValueError("unsupported reference-quality recovery lane")
    rawpath = path.parent / race["race_json"]
    if str(rawpath) not in inventory:
        raise ValueError("original worker result not retained")
    raw = json.loads(rawpath.read_text())
    if json.loads(json.dumps(A.lane_config(lane))) != raw["lane_config"]:
        raise ValueError("recipe configuration changed")
    destination.mkdir(parents=True, exist_ok=True)
    if (destination / "board.json").exists():
        raise ValueError("recovery already exists; do not overwrite it")
    oldref = race["evidence_receipt"]
    receipt_file = destination / "original-receipt.tar.gz"
    E.fetch(oldref, receipt_file)
    with tarfile.open(receipt_file) as archive:
        original = json.load(archive.extractfile("result.json"))
    verify_observations(race, original)
    datadir = Path(verified) / "root/fairness-cache/algos-data/rows-full"
    blockpath = datadir / (A.block_file(lane, dataset) + ".npz")
    if str(blockpath) not in inventory:
        raise ValueError("prepared input bytes are not retained")
    block, meta = A._load_block(lane, dataset, str(datadir))
    if meta != raw["block"]:
        raise ValueError("prepared block metadata differs")
    verify_inputs(block, meta)
    outputs, proofs = {}, {}
    for arm, filename in raw["saved_outputs"].items():
        output = Path(verified) / filename.lstrip("/")
        if str(output) not in inventory:
            raise ValueError("scored output bytes not retained: " + arm)
        with np.load(output, allow_pickle=False) as z:
            outputs[arm] = {k: z[k] for k in z.files}
        scored = raw["arms"][arm]["digests"][-1]
        verify_outputs(outputs[arm], scored)
        proofs[arm] = dict(inventory[str(output)], scored_digest=scored)
    inputs = A.lane_arrays(lane, block)
    reference = A.transform_quality_reference(lane, inputs, outputs, destination / "oracle")
    qualities = A.quality(lane, inputs, outputs)
    provenance = dict(schema="mojolearn.retained-transform-quality.v1", source=source,
                      original_receipt=oldref, prepared_input=inventory[str(blockpath)],
                      scored_outputs=proofs, oracle=reference, quality=qualities,
                      scope="quality derived from retained scored arrays; all timings unchanged")
    artifact = E.upload(E.archive(destination / "quality.tar.gz", {
        "provenance.json": E.encoded(provenance),
        "reference.npz": (destination / "oracle/reference.npz").read_bytes()}))
    for field in ("cells", "infer_cells"):
        for cell in race.get(field, []):
            arm = cell.get("arm")
            if cell.get("status") != "ok" or arm not in qualities:
                continue
            if field == "infer_cells":
                # FAIRNESS FIX (2026-10-13 audit): algos names its query batch
                # Xq. Recover only the same retained scored output digest.
                if (cell.get("batch") != (raw.get("infer") or {}).get("batch")
                        or cell.get("hash") != proofs[arm]["scored_digest"]):
                    continue
            old_quality = copy.deepcopy(cell.get("quality"))
            # Preserve any original numerical quality or error. This tool only
            # fills shapes-only records and never replaces an observed failure.
            if any(k != "output_shape" for k in old_quality or {}):
                continue
            cell["quality_superseded"] = old_quality
            cell["quality"] = qualities[arm]
            record = cell["measurement"]
            previous_id = record.pop("id")
            record["quality_recovery"] = dict(previous_id=previous_id,
                original_receipt=oldref, artifact=E.public_ref(artifact), source=source)
            record["id"] = E.digest(E.encoded(record))
    receipt = E.upload(E.archive(destination / "receipt.tar.gz", {"result.json": E.encoded(race)}))
    race["evidence_receipt"] = E.public_ref(receipt)
    F.apply(board)
    E.write(destination / "board.json", board)
    E.write(destination / "recovery.json", provenance)
    return dict(id=rid, board=str(destination / "board.json"), quality=qualities, audit=race.get("audit"))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--archive-receipts", required=True)
    ap.add_argument("--verified-tree", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    root = Path(args.out)
    root.mkdir(parents=True, exist_ok=True)
    source, _ = E.source_snapshot(root / "source")
    source = E.public_ref(E.upload(source))
    inventory = retained_tree(json.loads(Path(args.archive_receipts).read_text()), args.verified_tree)
    E.write(root / "verified-inventory.json", inventory)
    rows = []
    for p in sorted(Path(args.verified_tree).glob("root/fairness-evidence/measured/algos.*/board.json")):
        board = json.loads(p.read_text())
        race = next(iter(board["races"].values()))
        if A.quality_kind(race["lane"]) not in ("vs-sklearn", "labels", "multilabel"):
            continue
        rid = race["id"]
        try:
            row = recover(p, args.verified_tree, inventory, root / rid.replace("/", "."), source)
            row["status"] = "recovered"
        except Exception as exc:
            row = dict(id=rid, status="failed", error=repr(exc))
        rows.append(row)
        E.write(root / "summary.json", rows)
        print(json.dumps({k: row[k] for k in ("id", "status", "error") if k in row}), flush=True)
    return 1 if any(r["status"] == "failed" for r in rows) else 0


if __name__ == "__main__":
    raise SystemExit(main())
