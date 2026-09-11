#!/usr/bin/env python3
"""Real-data fixtures shared by the kNN harnesses (DEVIATION 2524).

ENGINEERING_RULES.md section 9: a non-tree classical speed claim is timed
on TWO datasets that differ in KIND at the lane's large shape, one real
beside one generator. The kNN selection gate's `dyadic` and `large`
fixtures are both 400,000 x 32 synthetic generators, so a selection win
timed on both was timed on one kind. This module is the second kind, and
it is ONE module on purpose: `tools/knn_selection_gate.py` (ours) and
`tools/knn_cuml_reference.py` (the opponent) both call `higgs_block`, so
the bytes the two tools measure cannot drift apart.

THE HIGGS PREFIX
----------------
HIGGS (UCI 00280, 11,000,000 x 29 CSV, gzip 2.6 GB): column 0 is the
label, columns 1..28 are the kinematic features. This module reads ONLY
the first `HIGGS_PREFIX_ROWS` = 404,000 lines of the gzip STREAM
(`gzip.open` plus `itertools.islice`; the file is never decompressed
whole and never read into memory whole), parses them with numpy, and
keeps the 28 features as float32: rows 0..399,999 are the index block,
rows 400,000..403,999 are the queries. No shuffle, no scaling, no
deduplication: the raw features, several of which take few distinct
values (the b-tag columns are three-valued, the jet multiplicity columns
are near-discrete), are what makes this a different kind from the two
generators, ties included. Duplicate rows, if the prefix has any, are
COUNTED and reported by the caller, never dropped: a harness that chose
rows would be choosing its fixture.

The parsed block is cached as an `.npz` under the same data root the
trees lane uses (`GBM_BENCH_DATA`, default `~/datasets/gbm-bench`, folder
`higgs/`), beside the trees lane's `HIGGS.csv.gz`, which is reused when it
is already there. A cache entry that exists is not a cache entry that
loads (the trees lane's zero-byte stub lesson, `tools/speed_gbdt_arm.py::
load_higgs`): the cache is used only if it is non-empty, loads, and has
exactly the prefix shape; otherwise it is re-decoded from the gzip beside
it. A missing gzip is downloaded with urllib to that path, into a `.part`
file renamed only on completion, and the download is returned as its own
timed record so the caller can report it as a separate item that is
never part of a timed request.

    python3 tools/knn_datasets.py --prefetch higgs [--data-root DIR]

does the download and the decode once, outside any timed run, and prints
the record as JSON; the two harnesses then find the cache.
"""
import argparse
import gzip
import hashlib
import itertools
import json
import os
import sys
import time
import urllib.request

import numpy as np

HIGGS_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/00280/HIGGS.csv.gz"
HIGGS_GZ_NAME = "HIGGS.csv.gz"
HIGGS_TOTAL_ROWS = 11_000_000
HIGGS_FEATURES = 28
HIGGS_INDEX_ROWS = 400_000
HIGGS_QUERY_ROWS = 4_000
HIGGS_PREFIX_ROWS = HIGGS_INDEX_ROWS + HIGGS_QUERY_ROWS
HIGGS_CACHE_NAME = "higgs_knn_prefix_%d.npz" % HIGGS_PREFIX_ROWS
HIGGS_FIXTURE_NAME = "higgs-prefix-%d" % HIGGS_PREFIX_ROWS


def data_root():
    """The trees lane's store (`tools/speed_gbdt_arm.py::data_root`): the
    `GBM_BENCH_DATA` environment variable, default `~/datasets/gbm-bench`,
    so a box that already has the trees lane's HIGGS fetches nothing."""
    return os.environ.get("GBM_BENCH_DATA", os.path.join(os.path.expanduser("~"), "datasets", "gbm-bench"))


def higgs_paths(root=None):
    folder = os.path.join(root or data_root(), "higgs")
    return folder, os.path.join(folder, HIGGS_GZ_NAME), os.path.join(folder, HIGGS_CACHE_NAME)


def sha256_array(a):
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def _say(log, msg):
    if log is not None:
        log(msg)
    else:
        print(msg, flush=True)


def download_higgs(gz_path, log=None):
    """Fetch HIGGS.csv.gz to `gz_path`. A separate, timed, named step; the
    record it returns is reported beside the fixture, never inside a timed
    request. Writes to `<gz_path>.part` and renames on completion so an
    interrupted fetch never leaves a truncated file under the real name."""
    os.makedirs(os.path.dirname(gz_path), exist_ok=True)
    part = gz_path + ".part"
    _say(log, "downloading %s -> %s (about 2.6 GB; a separate untimed step)" % (HIGGS_URL, gz_path))
    t0 = time.perf_counter()
    urllib.request.urlretrieve(HIGGS_URL, part)
    os.replace(part, gz_path)
    seconds = time.perf_counter() - t0
    size = os.path.getsize(gz_path)
    _say(log, "downloaded %d bytes in %.1f s" % (size, seconds))
    return {"url": HIGGS_URL, "path": gz_path, "bytes": size, "seconds": seconds}


def parse_higgs_prefix(gz_path, n_rows=HIGGS_PREFIX_ROWS):
    """The first `n_rows` lines of the gzip stream as (x float32 [n_rows,
    28], y float32 [n_rows]). Parsed by numpy as float64 and cast to
    float32 (the same rounding the trees lane's pandas parse performs).
    Raises if the stream ends early: a short read is a truncated download,
    and the message names the file."""
    with gzip.open(gz_path, "rt", encoding="ascii", newline="") as fh:
        block = np.loadtxt(itertools.islice(fh, n_rows), delimiter=",", dtype=np.float64, ndmin=2)
    if block.shape != (n_rows, HIGGS_FEATURES + 1):
        raise RuntimeError(
            "HIGGS prefix parse: %s yielded shape %s, want (%d, %d); a short read is a "
            "truncated download, remove the file and fetch it again"
            % (gz_path, block.shape, n_rows, HIGGS_FEATURES + 1))
    x = np.ascontiguousarray(block[:, 1:].astype(np.float32))
    y = np.ascontiguousarray(block[:, 0].astype(np.float32))
    return x, y


def load_higgs_prefix(root=None, download="auto", log=None):
    """The cached 404,000 x 28 float32 prefix, decoded (and fetched) on
    demand. `download`: `auto` fetches a missing gzip, `never` raises on
    one. Returns (x, y, record); the record carries the cache decision, the
    parse time and, when it happened, the download."""
    folder, gz_path, npz_path = higgs_paths(root)
    record = {"data_root": root or data_root(), "gz_path": gz_path, "cache_path": npz_path,
              "url": HIGGS_URL, "prefix_rows": HIGGS_PREFIX_ROWS, "features": HIGGS_FEATURES,
              "download": None, "cache_hit": False, "parse_seconds": None,
              "parse": "gzip stream, first %d lines, numpy loadtxt float64 cast to float32; no shuffle, no scaling, no deduplication" % HIGGS_PREFIX_ROWS}
    x = y = None
    if os.path.isfile(npz_path) and os.path.getsize(npz_path) > 0:
        try:
            with np.load(npz_path) as z:
                x, y = z["x"], z["y"]
            if x.shape != (HIGGS_PREFIX_ROWS, HIGGS_FEATURES) or x.dtype != np.float32 or y.shape != (HIGGS_PREFIX_ROWS,):
                _say(log, "knn_datasets: %s has shape %s dtype %s, not the prefix; re-decoding" % (npz_path, x.shape, x.dtype))
                x = y = None
            else:
                record["cache_hit"] = True
        except Exception as exc:  # noqa: BLE001
            _say(log, "knn_datasets: %s is unreadable (%r); re-decoding from the gzip beside it" % (npz_path, exc))
            x = y = None
    if x is None:
        if not (os.path.isfile(gz_path) and os.path.getsize(gz_path) > 0):
            if download != "auto":
                raise RuntimeError(
                    "HIGGS is not downloaded: %s is missing and download is %r. Run "
                    "`python3 tools/knn_datasets.py --prefetch higgs` (2.6 GB) outside the timed run." % (gz_path, download))
            record["download"] = download_higgs(gz_path, log)
        else:
            record["download"] = "already present (%d bytes)" % os.path.getsize(gz_path)
        t0 = time.perf_counter()
        x, y = parse_higgs_prefix(gz_path)
        record["parse_seconds"] = time.perf_counter() - t0
        os.makedirs(folder, exist_ok=True)
        tmp = npz_path + ".part.npz"
        np.savez(tmp, x=x, y=y)
        os.replace(tmp, npz_path)
        _say(log, "knn_datasets: decoded the %d-row HIGGS prefix in %.1f s, cached at %s" % (HIGGS_PREFIX_ROWS, record["parse_seconds"], npz_path))
    record["gz_bytes"] = os.path.getsize(gz_path) if os.path.isfile(gz_path) else None
    record["sha256_block"] = sha256_array(x)
    return x, y, record


def higgs_present(root=None):
    """True when either the cache or the gzip is on this box (no fetch)."""
    _, gz_path, npz_path = higgs_paths(root)
    return ((os.path.isfile(npz_path) and os.path.getsize(npz_path) > 0)
            or (os.path.isfile(gz_path) and os.path.getsize(gz_path) > 0))


def higgs_block(n_index=HIGGS_INDEX_ROWS, n_queries=HIGGS_QUERY_ROWS, data_root=None, download="auto", log=None):
    """The HIGGS kNN fixture: index = prefix rows [0, n_index), queries =
    prefix rows [400,000, 400,000 + n_queries). The queries always come
    from the query region, so a smaller `n_index` (a `--quick` smoke) never
    turns a query row into an index row. Returns a dict with the two
    float32 blocks, the row ranges, the sha256 of the whole prefix block
    and of each part, and the load record (cache, parse, download)."""
    if not (0 < n_index <= HIGGS_INDEX_ROWS):
        raise ValueError("n_index must be in 1..%d, got %d" % (HIGGS_INDEX_ROWS, n_index))
    if not (0 < n_queries <= HIGGS_QUERY_ROWS):
        raise ValueError("n_queries must be in 1..%d, got %d" % (HIGGS_QUERY_ROWS, n_queries))
    x, _y, record = load_higgs_prefix(data_root, download, log)
    index = np.ascontiguousarray(x[:n_index])
    queries = np.ascontiguousarray(x[HIGGS_INDEX_ROWS: HIGGS_INDEX_ROWS + n_queries])
    return {
        "dataset": "higgs", "fixture": HIGGS_FIXTURE_NAME,
        "index": index, "queries": queries, "d": HIGGS_FEATURES,
        "index_rows": [0, n_index], "query_rows": [HIGGS_INDEX_ROWS, HIGGS_INDEX_ROWS + n_queries],
        "sha256_block": record["sha256_block"], "sha256_index": sha256_array(index), "sha256_queries": sha256_array(queries),
        "source": record,
    }


def distinct_row_count(x):
    v = np.ascontiguousarray(x).view(np.dtype((np.void, x.dtype.itemsize * x.shape[1])))
    return int(np.unique(v).shape[0])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--prefetch", choices=("higgs",), required=True, help="download (if absent) and decode the fixture's prefix into the cache, outside any timed run")
    ap.add_argument("--data-root", default=None, help="override GBM_BENCH_DATA / ~/datasets/gbm-bench")
    args = ap.parse_args()
    t0 = time.perf_counter()
    block = higgs_block(data_root=args.data_root)
    out = {k: v for k, v in block.items() if k not in ("index", "queries")}
    out["distinct_index_rows"] = distinct_row_count(block["index"])
    out["duplicate_index_rows"] = block["index"].shape[0] - out["distinct_index_rows"]
    out["prefetch_seconds"] = time.perf_counter() - t0
    json.dump(out, sys.stdout, indent=1, default=str)
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
