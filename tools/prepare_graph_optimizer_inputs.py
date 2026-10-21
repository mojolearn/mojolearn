#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Prepare saved full-workload inputs; never run or qualify an optimizer arm."""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import platform
import subprocess
import sys


def digest(path):
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for part in iter(lambda: stream.read(1 << 22), b""):
            result.update(part)
    return result.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("recipe", choices=("program-run", "transformer-forward", "sequential-predict", "tape-user-model"))
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--higgs", type=Path)
    args = parser.parse_args()
    import numpy as np

    root = Path(__file__).resolve().parents[1]
    source = root / "tools/bench_board_neural.py"
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    if args.out.exists():
        parser.error("output already exists; use a fresh immutable evidence directory")
    original_data = None
    if args.recipe == "tape-user-model":
        if args.higgs is None:
            parser.error("tape-user-model requires --higgs with the manifest-pinned complete NPZ")
        key = "gbm-bench/higgs/higgs_speed.npz"
        manifest = root / "bench/results/dataset_store/manifest.tsv"
        fields = next(line.split("\t") for line in manifest.read_text().splitlines() if line.startswith(key + "\t"))
        actual = digest(args.higgs)
        if args.higgs.stat().st_size != int(fields[1]) or actual != fields[2]:
            parser.error("HIGGS size/hash differs from the retained dataset-store manifest")
        os.environ["MOJOLEARN_BENCH_HIGGS"] = str(args.higgs.resolve())
        original_data = {"key": key, "bytes": int(fields[1]), "sha256": actual,
                         "manifest_sha256": digest(manifest),
                         "coverage": "saved board recipe takes 1,000,000 train and 100,000 test rows; this does not prove full source-dataset coverage"}
    spec = importlib.util.spec_from_file_location("graph_recipe_board", source)
    board = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = board
    spec.loader.exec_module(board)
    args.out.mkdir(parents=True)
    artifact = args.out / "inputs.npz"
    record = board.make_inputs(args.recipe, "full", 2, str(artifact))
    arrays = {}
    with np.load(artifact, allow_pickle=False) as archive:
        for name in sorted(archive.files):
            value = np.ascontiguousarray(archive[name])
            arrays[name] = {"shape": list(value.shape), "dtype": value.dtype.str,
                            "bytes": value.nbytes, "sha256": hashlib.sha256(value.tobytes()).hexdigest()}
    receipt = {"schema": "mojolearn.graph-optimizer-input-recipe.v1", "status": "inputs_prepared_only",
               "recipe": args.recipe, "shape": "full", "seed": board.SEED,
               "source_commit": revision, "generator": str(source.relative_to(root)),
               "generator_sha256": digest(source), "python": platform.python_version(), "numpy": np.__version__,
               "source_dataset": original_data, "saved_recipe": record,
               "artifact": {"path": "inputs.npz", "bytes": artifact.stat().st_size, "sha256": digest(artifact)},
               "arrays": arrays, "performance_measurements": [],
               "pending": ["native captured graph artifact and exact model equivalence", "pass activation report",
                           "full-operation A/B on NVIDIA and AMD", "cross-vendor identity and quality"]}
    (args.out / "recipe.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": receipt["status"], "recipe": args.recipe,
                      "output": str(args.out), "artifact_sha256": receipt["artifact"]["sha256"],
                      "array_count": len(arrays), "performance_measurements": 0}))


if __name__ == "__main__":
    main()
