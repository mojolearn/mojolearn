#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""CAGRA FAST quality dump and main/candidate comparison (lane/apple-fast-w2-cagra).

  dump DATASET OUT.npz   fit CagraIndex with the board's parameters on the
                         board's ann block (tools/knn_datasets.real_block:
                         400,000 index rows, 4,000 queries, raw), search the
                         queries, and score recall@10 against the board's
                         float64 brute force (bench_board_algos._recall, the
                         same stable tie order as the board cell)
  compare A.npz B.npz    PASS when, on every dataset dumped, B's recall@10 >=
                         A's recall@10 (no tolerance: the build and search are
                         deterministic and the brief asks for recall not below
                         main's), and, for istella (d = 220, where
                         CAGRA_FAST_IVFG_LOWD must change nothing), with
                         --istella-identical, B's graph and neighbor ids are
                         byte-identical to A's (not for the _SEEDS4 arm,
                         whose search moves on every dataset).

Quality work only (NumPy brute force outside the measured operation)."""
import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
PARAMS = dict(graph_degree=32, intermediate_graph_degree=64, itopk_size=64, n_neighbors=10, random_state=7)


def dump(dataset, path):
    assert os.environ.get("MOJOLEARN_NUMERIC_MODE") == "fast"
    import mojolearn as ml
    from knn_datasets import real_block
    import bench_board_algos as bba
    D = real_block(dataset)
    est = ml.CagraIndex(**PARAMS)
    binding = est._bind()
    assert str(binding.x_ann_vendor()) == "metal", binding.x_ann_vendor()
    assert int(binding.x_ann_numeric_mode()) == 0
    est.fit(D["index"])
    dist, ind = est.search(D["queries"])
    ind = np.asarray(ind, dtype=np.int64)
    graph = np.asarray(est.graph_, dtype=np.int32)
    recall = float(bba._recall(D, ind))
    meta = dict(dataset=dataset, binding_sha256=hashlib.sha256(Path(binding.__file__).read_bytes()).hexdigest(),
                sha256_index=D["sha256_index"], sha256_queries=D["sha256_queries"], recall_at_10=recall,
                graph_sha256=hashlib.sha256(graph.tobytes()).hexdigest(),
                ind_sha256=hashlib.sha256(ind.tobytes()).hexdigest())
    np.savez(path, graph=graph, ind=ind, dist=np.asarray(dist, dtype=np.float32),
             meta=np.array(json.dumps(meta, sort_keys=True)))
    print("CAGRA-LOWD-CAPTURE " + json.dumps(meta, sort_keys=True))


def compare(pa, pb, istella_identical):
    a, b = np.load(pa), np.load(pb)
    ma, mb = json.loads(str(a["meta"])), json.loads(str(b["meta"]))
    assert ma["dataset"] == mb["dataset"]
    assert (ma["sha256_index"], ma["sha256_queries"]) == (mb["sha256_index"], mb["sha256_queries"])
    ok = mb["recall_at_10"] >= ma["recall_at_10"]
    same = a["graph"].tobytes() == b["graph"].tobytes() and a["ind"].tobytes() == b["ind"].tobytes()
    if ma["dataset"] == "istella" and istella_identical:
        ok = ok and same
    overlap = float(np.mean([len(set(x[:10]) & set(y[:10])) / 10.0 for x, y in zip(a["ind"], b["ind"])]))
    print("CAGRA-LOWD-AB dataset=%s status=%s recall_A=%.6f recall_B=%.6f identical=%s overlap_AB=%.6f" % (
        ma["dataset"], "PASS" if ok else "FAIL", ma["recall_at_10"], mb["recall_at_10"], same, overlap))
    return ok


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("action", choices=["dump", "compare"])
    p.add_argument("first")
    p.add_argument("second")
    p.add_argument("--istella-identical", action="store_true")
    args = p.parse_args()
    if args.action == "dump":
        assert args.first in ("taxi", "istella")
        dump(args.first, args.second)
    else:
        sys.exit(0 if compare(args.first, args.second, args.istella_identical) else 1)


if __name__ == "__main__":
    main()
