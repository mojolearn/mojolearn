#!/usr/bin/env python3
"""Retain deterministic matrix inputs by matching original worker byte hashes.

This is an untimed evidence operation, never a product or benchmark execution.
It changes no timing, output, quality metric or board measurement.
"""
import argparse
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import sys
import time


def generated_inputs(rec, algos):
    """Reconstruct only inputs whose complete original byte identity is known."""
    import numpy as np
    if rec.get("family") != "algos" or rec.get("lane") not in (
            "cholesky", "eigh", "lu-solve", "lu-factor"):
        return []
    expected = None
    for cell in rec.get("cells", []):
        if cell.get("status") != "ok":
            continue
        params = cell.get("params") or cell.get("measurement", {}).get("parameters") or {}
        raw = params.get("raw_params") or params.get("params") or {}
        arrays = raw.get("input_arrays")
        if not arrays or raw.get("seed") != algos.SEED:
            raise ValueError("missing recorded input bytes/seed identity")
        if expected is not None and expected != arrays:
            raise ValueError("workers recorded different input arrays")
        expected = arrays
    if not expected:
        raise ValueError("no successful worker with recorded input identity")
    shape = expected[0].get("shape")
    if (not isinstance(shape, list) or len(shape) != 2 or shape[0] != shape[1]
            or type(shape[0]) is not int or shape[0] <= 0):
        raise ValueError("invalid recorded system shape")
    arrays = (algos.lu_system(shape[0]) if rec["lane"] in ("lu-solve", "lu-factor")
              else (algos.sym_system(shape[0]),))
    if len(arrays) != len(expected):
        raise ValueError("recorded input count differs from reconstruction")
    result = []
    for array, original in zip(arrays, expected):
        array = np.ascontiguousarray(array)
        actual = dict(shape=list(array.shape), dtype=str(array.dtype),
                      sha256=hashlib.sha256(array.data).hexdigest())
        if any(original.get(k) != value for k, value in actual.items()):
            raise ValueError("reconstructed bytes differ from original worker input")
        result.append((actual, array))
    return result


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--board", action="append", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tools-dir", default=str(Path(__file__).resolve().parent))
    args = ap.parse_args(argv)
    sys.path.insert(0, args.tools_dir)
    import bench_board_algos as algos
    import bench_board_evidence as evidence
    import numpy as np
    root = Path(args.out).resolve()
    root.mkdir(parents=True, exist_ok=False)
    inputs = root / "inputs"
    inputs.mkdir()
    # FAIRNESS FIX (2026-10-13 audit): generated matrix inputs previously
    # had source/seed/hashes but no explicit retained array bytes. Snapshot
    # the recovery code and original records before reconstructing anything.
    source, _ = evidence.source_snapshot(root / "source",
        drivers=[str(Path(__file__).resolve()), *args.board])
    source = evidence.upload(source)
    report = dict(started=time.time(), source=evidence.public_ref(source), inputs={}, recipes={},
        role="untimed exact input reconstruction; original worker SHA256 must match; no timing rerun",
        versions=dict(python=platform.python_version(), numpy=np.__version__),
        hardware=platform.uname()._asdict())
    for filename in args.board:
        path = Path(filename)
        board = json.loads(path.read_text())
        for rid, rec in board.get("races", {}).items():
            materialized = generated_inputs(rec, algos)
            if not materialized:
                continue
            hashes = []
            for metadata, array in materialized:
                sha = metadata["sha256"]
                dest = inputs / (sha + ".npy")
                if sha not in report["inputs"]:
                    np.save(dest, array, allow_pickle=False)
                    report["inputs"][sha] = dict(metadata, member="inputs/" + dest.name,
                        npy_sha256=evidence.file_hash(dest), bytes=dest.stat().st_size)
                hashes.append(sha)
            report["recipes"][rid] = dict(board=str(path), board_sha256=evidence.file_hash(path),
                input_sha256=hashes, measurement_ids=[c.get("measurement", {}).get("id")
                    for c in rec.get("cells", []) if c.get("status") == "ok"])
    if not report["inputs"]:
        raise ValueError("no generated inputs selected")
    evidence.write(root / "receipt.json", report)
    archive = root / "generated-inputs.tar.gz"
    with (root / "archive.log").open("w") as stream:
        subprocess.run(["tar", "-I", "gzip -1", "-cf", str(archive), "-C", str(root),
                        "inputs", "receipt.json"], stdout=stream, stderr=stream, check=True)
    ref = evidence.upload(dict(local_path=str(archive), sha256=evidence.file_hash(archive),
                               bytes=archive.stat().st_size))
    report.update(finished=time.time(), artifact=evidence.public_ref(ref))
    evidence.write(root / "retention.json", report)
    print(json.dumps(dict(recipes=len(report["recipes"]), arrays=len(report["inputs"]),
                          artifact_status=ref["status"], bytes=ref["bytes"])))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
