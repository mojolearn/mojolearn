#!/usr/bin/env python3
"""cuML brute-force NearestNeighbors at the k-NN reference grid; one JSON out.

The opponent's FAST arm for `bench/knn_reference_price_main.mojo`: same
dyadic-v1 fixture (the `_coordinate` mixer, index salt 0, query salt 593),
same shapes, same two regions:
  request   host numpy in, host numpy out (cuML uploads and downloads inside
            `kneighbors`; the brute index has no build, `fit` is a copy);
  device    cupy in, cupy out, `kneighbors` + stream synchronize.
Two warmups, `--rounds` timed rounds, medians. With `--ours-dump DIR` the
UInt32 neighbour lists `bench/knn_reference_price_main.mojo` wrote there
are compared row by row (as sets, and as ordered lists).

`--dataset higgs` (DEVIATION 2524; ENGINEERING_RULES.md section 9) prices
the same opponent on the SECOND KIND, the real HIGGS prefix that
`tools/knn_selection_gate.py`'s `higgs` fixture uses: the same
`tools/knn_datasets.py::higgs_block` call, so the bytes are the gate's by
construction (index = prefix rows [0, --index), queries = prefix rows
[400,000, 400,000 + --queries), 28 raw float32 features, no shuffle, no
scaling). The JSON names the dataset, the sha256 of the whole 404,000-row
block and of the index and query parts, and the row ranges. That row is
measured ONCE per (GPU, driver, cuML version, dataset) and cached in
`bench/OPPONENT_REFERENCE.md`; later rounds run ours alone against it.
`--dataset dyadic` (the default) is the unchanged behavior. A missing
HIGGS.csv.gz is fetched (2.6 GB) before any timing, as its own untimed
step, recorded under `environment.dataset_source`.

`--out` is a JSON file path, or a directory (an existing one, or a path
without a `.json` suffix), in which case the file is
`<out>/cuml-reference-<dataset>.json`.

Only the main lane runs this, on the rented box.
"""
import argparse
import json
import os
import statistics
import subprocess
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import knn_datasets  # noqa: E402  (tools/knn_datasets.py, the shared real-data loader)


def coordinate_block(n_rows, n_features, salt):
    """`bench/knn_smallk_dispatch_fixture._coordinate`, vectorized in uint32."""
    rows = np.arange(n_rows, dtype=np.uint32).reshape(-1, 1)
    feats = np.arange(n_features, dtype=np.uint32).reshape(1, -1)
    with np.errstate(over="ignore"):
        v = (rows + np.uint32(1)) * np.uint32(747796405) + (
            feats * np.uint32(131) + np.uint32(salt)
        )
        v = (v ^ (v >> np.uint32(16))) * np.uint32(2246822519)
        v = (v ^ (v >> np.uint32(13))) * np.uint32(3266489917)
        v = v ^ (v >> np.uint32(16))
    return ((v % np.uint32(2048)).astype(np.int64) - 1024).astype(np.float32) / np.float32(1024)


def fnv_indices(idx):
    h = 1469598103934665603
    for v in idx.reshape(-1).tolist():
        h = ((h ^ int(v)) * 1099511628211) & 0xFFFFFFFFFFFFFFFF
    return h


def nvidia_smi():
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=20, check=True,
        ).stdout.strip()
        name, driver = [s.strip() for s in out.splitlines()[0].split(",")]
        return name, driver
    except Exception as exc:  # noqa: BLE001
        return "unknown", "unknown (%r)" % (exc,)


def fixture_blocks(dataset, n_index, n_queries, d, data_root=None):
    """(xi, xq, record fields) for one shape. `dyadic` is the generator
    (unchanged); `higgs` is the shared real prefix, whose feature count is
    the data's (28), so `--features` is ignored for it and recorded."""
    if dataset == "dyadic":
        xi = np.ascontiguousarray(coordinate_block(n_index, d, 0))
        xq = np.ascontiguousarray(coordinate_block(n_queries, d, 593))
        return xi, xq, {"fixture": "dyadic-v1", "dataset": "dyadic"}
    if dataset == "higgs":
        block = knn_datasets.higgs_block(n_index, n_queries, data_root=data_root)
        fields = {
            "fixture": block["fixture"], "dataset": "higgs",
            "index_rows": block["index_rows"], "query_rows": block["query_rows"],
            "sha256_block": block["sha256_block"], "sha256_index": block["sha256_index"],
            "sha256_queries": block["sha256_queries"],
            "note_dataset": "REAL data, the second kind (DEVIATION 2524): HIGGS prefix, 28 raw float32 features, no shuffle, no scaling, no deduplication; the same bytes tools/knn_selection_gate.py's higgs fixture measures",
        }
        return block["index"], block["queries"], fields
    raise ValueError("unknown dataset %r" % (dataset,))


def one_shape(n_index, n_queries, k, d, rounds, ours_dump, dataset="dyadic", data_root=None):
    import cupy as cp
    from cuml.neighbors import NearestNeighbors

    xi, xq, fields = fixture_blocks(dataset, n_index, n_queries, d, data_root)
    d = int(xi.shape[1])

    def sync():
        cp.cuda.Stream.null.synchronize()
        cp.cuda.runtime.deviceSynchronize()

    record = {"index": n_index, "queries": n_queries, "k": k, "features": d,
              "rounds": rounds}
    record.update(fields)

    # request: host arrays both ways
    nn = NearestNeighbors(n_neighbors=k, algorithm="brute", metric="euclidean",
                          output_type="numpy")
    nn.fit(xi)
    sync()
    req = []
    ref_idx = None
    for r in range(rounds + 2):
        t0 = time.perf_counter()
        dist, idx = nn.kneighbors(xq)
        sync()
        ms = (time.perf_counter() - t0) * 1000.0
        idx = np.asarray(idx)
        if ref_idx is None:
            ref_idx = idx.copy()
        elif not np.array_equal(ref_idx, idx):
            record.setdefault("notes", []).append("request indices moved between rounds %d" % r)
        if r >= 2:
            req.append(ms)
    record["request_ms"] = req
    record["request_median_ms"] = statistics.median(req)
    record["index_fnv1a64"] = fnv_indices(ref_idx)
    record["row0"] = [[int(ref_idx[0, s]), float(np.asarray(dist)[0, s])] for s in range(k)]

    # device: cupy both ways
    gi = cp.asarray(xi)
    gq = cp.asarray(xq)
    nn2 = NearestNeighbors(n_neighbors=k, algorithm="brute", metric="euclidean",
                           output_type="cupy")
    nn2.fit(gi)
    sync()
    dev = []
    for r in range(rounds + 2):
        sync()
        t0 = time.perf_counter()
        dist2, idx2 = nn2.kneighbors(gq)
        sync()
        ms = (time.perf_counter() - t0) * 1000.0
        if r >= 2:
            dev.append(ms)
    record["device_ms"] = dev
    record["device_median_ms"] = statistics.median(dev)
    record["device_vs_request_equal_indices"] = bool(np.array_equal(cp.asnumpy(idx2), ref_idx))

    if ours_dump:
        path = os.path.join(ours_dump, "ours-%d-%d-%d.u32" % (n_index, n_queries, k))
        if os.path.exists(path):
            ours = np.fromfile(path, dtype="<u4").reshape(n_queries, k)
            ordered_equal_rows = int(np.sum(np.all(ours == ref_idx, axis=1)))
            set_equal_rows = int(sum(
                1 for q in range(n_queries) if set(ours[q].tolist()) == set(ref_idx[q].tolist())
            ))
            record["ours_dump"] = path
            record["rows_equal_ordered"] = ordered_equal_rows
            record["rows_equal_as_set"] = set_equal_rows
            record["rows_total"] = n_queries
        else:
            record["ours_dump_missing"] = path
    del gi, gq, nn, nn2
    return record


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", type=int, nargs="+", default=[100000, 400000])
    ap.add_argument("--queries", type=int, nargs="+", default=[32, 128, 1000, 4000])
    ap.add_argument("--k", type=int, nargs="+", default=[10, 15])
    ap.add_argument("--features", type=int, default=32)
    ap.add_argument("--rounds", type=int, default=7)
    ap.add_argument("--ours-dump", default="")
    ap.add_argument("--dataset", choices=("dyadic", "higgs"), default="dyadic", help="dyadic: the dyadic-v1 generator (unchanged default); higgs: the real HIGGS prefix shared with tools/knn_selection_gate.py (DEVIATION 2524; --index <= 400000, --queries <= 4000, features fixed at 28)")
    ap.add_argument("--data-root", default=None, help="higgs only: where HIGGS lives (default GBM_BENCH_DATA or ~/datasets/gbm-bench)")
    ap.add_argument("--out", required=True, help="JSON file, or a directory (then <out>/cuml-reference-<dataset>.json)")
    args = ap.parse_args()

    out_path = args.out
    if os.path.isdir(out_path) or not out_path.endswith(".json"):
        os.makedirs(out_path, exist_ok=True)
        out_path = os.path.join(out_path, "cuml-reference-%s.json" % args.dataset)

    dataset_source = None
    if args.dataset == "higgs":
        # The fetch and the decode happen here, ONCE, before any GPU work
        # or timing; they are recorded, never timed as part of a request.
        t0 = time.perf_counter()
        _x, _y, dataset_source = knn_datasets.load_higgs_prefix(args.data_root)
        dataset_source = dict(dataset_source)
        dataset_source["prefetch_seconds"] = time.perf_counter() - t0
        dataset_source["index_row_range_rule"] = "index = prefix rows [0, --index); queries = prefix rows [400000, 400000 + --queries)"
        del _x, _y

    import cupy as cp
    import cuml

    name, driver = nvidia_smi()
    env = {
        "gpu": name, "driver": driver, "cuml": cuml.__version__,
        "cupy": cp.__version__, "numpy": np.__version__,
        "cuda_runtime": int(cp.cuda.runtime.runtimeGetVersion()),
        "cuda_driver": int(cp.cuda.runtime.driverGetVersion()),
        "python": sys.version.split()[0],
        "arm": "cuml-brute-force-NearestNeighbors-fast",
        "note": "cuML's FAST arm: no deterministic configuration exists for brute kNN; distances are cuML's own (TF32-capable GEMM), so only the neighbour lists are compared",
        "dataset": args.dataset,
        "dataset_source": dataset_source,
    }
    results = []
    for n_index in args.index:
        for n_queries in args.queries:
            for k in args.k:
                rec = one_shape(n_index, n_queries, k, args.features, args.rounds, args.ours_dump,
                                dataset=args.dataset, data_root=args.data_root)
                results.append(rec)
                print("CUML_REF index=%d queries=%d k=%d request_median_ms=%.3f device_median_ms=%.3f dataset=%s" % (
                    n_index, n_queries, k, rec["request_median_ms"], rec["device_median_ms"], args.dataset), flush=True)
                if "rows_equal_as_set" in rec:
                    print("CUML_REF_VS_OURS index=%d queries=%d k=%d set_equal=%d ordered_equal=%d of %d" % (
                        n_index, n_queries, k, rec["rows_equal_as_set"], rec["rows_equal_ordered"], rec["rows_total"]), flush=True)
    with open(out_path, "w") as fh:
        json.dump({"environment": env, "results": results}, fh, indent=1, default=str)
    print("CUML REFERENCE DONE", out_path)


if __name__ == "__main__":
    main()
