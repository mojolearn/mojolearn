#!/usr/bin/env python3
"""Real-data fixtures shared by the kNN harnesses (DEVIATION 2524).

ENGINEERING_RULES.md section 9 (rewritten 2026-09-11): a non-tree classical
speed claim runs on THE SAME TWO real datasets, which differ in structure
and look like ordinary tables: NYC TLC yellow taxi trips (`taxi`, the
narrow, skewed, mixed-type business table; a classical lane takes its 11
numeric columns) and Istella-S LETOR (`istella`, the wide numeric table,
220 features whose scales span seven orders of magnitude). The kNN
selection gate's `dyadic` and `large` fixtures are both 400,000 x 32
synthetic generators, so a selection win timed on both was timed on one
kind; the generators stay as correctness fixtures and a timing or a default
flip quotes the two real datasets only. This module is where both kNN
tools get those bytes, and it is ONE module on purpose:
`tools/knn_selection_gate.py` (ours) and `tools/knn_cuml_reference.py` (the
opponent) both call `real_block`, so the bytes the two tools measure
cannot drift apart.

THE TWO REAL BLOCKS (`real_block`)
----------------------------------
Both come from the trees harness's NumPy caches, the ones
`tools/speed_gbdt_arm.py --download taxi` and `--download istella` build
once, untimed, under the shared data root (`GBM_BENCH_DATA`, default
`~/datasets/gbm-bench`):

  taxi     `taxi/taxi_speed.npz`, key `x` ([~5.75M, 16] float32, every
           plausible trip of January and February 2024, card and cash
           alike). The kNN rows are the `TAXI_NUMERIC` columns (11 of the
           16: passengers, distance, pickup hour/weekday/day, duration,
           extra, mta_tax, tolls, congestion, airport fee; -1 marks a
           missing value), so d = 11.
  istella  `istella/istella_speed.npz`, key `x_train` ([2,043,304, 220]
           float32, the training split's rows in file order), d = 220.

Rows 0..399,999 of each are the index block and rows 400,000..403,999 the
queries: the same layout the retired HIGGS prefix used, so `n_index` and
`n_queries` mean the same thing in every JSON that names a fixture. No
shuffle, no scaling, no deduplication: the raw columns, with their small
integers (taxi's hour, weekday, passenger count), their -1 sentinels and
Istella's near-constant columns, are what makes each a different kind from
the generators, ties included. Duplicate rows, if the prefix has any, are
COUNTED and reported by the caller, never dropped: a harness that chose
rows would be choosing its fixture. Fixture names: `taxi-prefix-404000`,
`istella-prefix-404000`.

Only the leading 404,000 rows are read from the cache. `np.savez` stores
its members uncompressed, so the `.npy` header is parsed from the zip
member's stream and exactly the prefix bytes are read after it: a taxi
block is 26 MB, an Istella block 355 MB, and the 1.8 GB `x_train` member
is never loaded whole (a member that is not C ordered falls back to a full
load). THIS MODULE NEVER DOWNLOADS: the trees harness owns the downloads
(and the parquet/LETOR decodes), and a missing cache raises with the exact
`python tools/speed_gbdt_arm.py --download <name>` command to run first.

    python3 tools/knn_datasets.py --prefetch taxi|istella [--data-root DIR]

checks that the cache is there, reads the block and prints its record
(shapes, d, sha256 of the block and of each part) as JSON; the two
harnesses then find the same bytes. Under the shell script this is the
`prefetch-<dataset>` status row.

HIGGS (RETIRED 2026-09-11)
--------------------------
`higgs_block`, `load_higgs_prefix`, `download_higgs`, `parse_higgs_prefix`,
`higgs_present`, `higgs_paths` and `--prefetch higgs` are the first
version of this module (the 404,000-line prefix of UCI 00280, 28 kinematic
features). HIGGS is RETIRED as a benchmark dataset for trees and classical
lanes (ENGINEERING_RULES.md section 9): its opponent rows in
`bench/OPPONENT_REFERENCE.md` are history and a HIGGS ratio is never
quoted as a result again. The functions stay, unchanged, so a JSON that
names `higgs-prefix-404000` can still be re-read and re-derived; nothing
new should call them.
"""
import argparse
import gzip
import hashlib
import importlib.util
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

#: The two real datasets (ENGINEERING_RULES.md section 9). Same row layout
#: HIGGS used: index rows [0, 400,000), query rows [400,000, 404,000).
REAL_DATASETS = ("taxi", "istella")
REAL_INDEX_ROWS = 400_000
REAL_QUERY_ROWS = 4_000
REAL_PREFIX_ROWS = REAL_INDEX_ROWS + REAL_QUERY_ROWS
#: dataset -> (cache folder under the data root, cache file, npz key,
#: features for the kNN rows: None = every column of the member, else the
#: names of the columns to take).
REAL_CACHES = {
    "taxi": ("taxi", "taxi_speed.npz", "x", "TAXI_NUMERIC"),
    "istella": ("istella", "istella_speed.npz", "x_train", None),
}
_HERE = os.path.dirname(os.path.abspath(__file__))


def data_root():
    """The trees lane's store (`tools/speed_gbdt_arm.py::data_root`): the
    `GBM_BENCH_DATA` environment variable, default `~/datasets/gbm-bench`,
    so the kNN tools read the caches `--download taxi` and `--download
    istella` built and never fetch anything themselves."""
    return os.environ.get("GBM_BENCH_DATA", os.path.join(os.path.expanduser("~"), "datasets", "gbm-bench"))


def higgs_paths(root=None):
    """RETIRED (HIGGS, 2026-09-11); kept so old JSON can be re-read."""
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
    """RETIRED (HIGGS, 2026-09-11; ENGINEERING_RULES.md section 9): kept so
    old evidence can be re-derived, never called by a new lane.

    Fetch HIGGS.csv.gz to `gz_path`. A separate, timed, named step; the
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
    """RETIRED (HIGGS, 2026-09-11); kept so old evidence can be re-derived.

    The first `n_rows` lines of the gzip stream as (x float32 [n_rows,
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
    """RETIRED (HIGGS, 2026-09-11; ENGINEERING_RULES.md section 9): the
    loader stays so a JSON naming `higgs-prefix-404000` can be re-read;
    new lanes call `real_block`.

    The cached 404,000 x 28 float32 prefix, decoded (and fetched) on
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
    """RETIRED (HIGGS, 2026-09-11). True when either the cache or the gzip
    is on this box (no fetch)."""
    _, gz_path, npz_path = higgs_paths(root)
    return ((os.path.isfile(npz_path) and os.path.getsize(npz_path) > 0)
            or (os.path.isfile(gz_path) and os.path.getsize(gz_path) > 0))


def higgs_block(n_index=HIGGS_INDEX_ROWS, n_queries=HIGGS_QUERY_ROWS, data_root=None, download="auto", log=None):
    """RETIRED (HIGGS, 2026-09-11; ENGINEERING_RULES.md section 9). Kept so
    old JSON can be re-read; `real_block("taxi" | "istella", ...)` is the
    fixture a lane uses now.

    The HIGGS kNN fixture: index = prefix rows [0, n_index), queries =
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


def _trees_harness():
    """`tools/speed_gbdt_arm.py`, imported from its file (stdlib + numpy
    only at import time), so `TAXI_NUMERIC`, `TAXI_FEATURES` and
    `data_root` are the trees harness's own definitions and the column
    selection cannot drift from the lane that built the cache."""
    path = os.path.join(_HERE, "speed_gbdt_arm.py")
    spec = importlib.util.spec_from_file_location("speed_gbdt_arm", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def real_paths(dataset, root=None):
    """(folder, cache path) of one real dataset's trees-harness cache."""
    if dataset not in REAL_CACHES:
        raise ValueError("unknown real dataset %r; choose from %s" % (dataset, REAL_DATASETS))
    folder_name, cache_name, _key, _cols = REAL_CACHES[dataset]
    folder = os.path.join(root or data_root(), folder_name)
    return folder, os.path.join(folder, cache_name)


def real_present(dataset, root=None):
    """True when the trees harness's cache for `dataset` is on this box
    and non-empty (a zero-byte stub is not a cache; no fetch, ever)."""
    _folder, npz_path = real_paths(dataset, root)
    return os.path.isfile(npz_path) and os.path.getsize(npz_path) > 0


def _missing_cache_error(dataset, npz_path):
    return RuntimeError(
        "%s is not downloaded: %s is missing or empty. Run "
        "`python tools/speed_gbdt_arm.py --download %s` first (the trees harness owns the "
        "download and the decode), OUTSIDE the timed run; tools/knn_datasets.py never downloads."
        % (dataset, npz_path, dataset))


def npz_member_prefix(npz_path, key, n_rows):
    """The first `n_rows` rows of member `key` of an `np.savez` archive
    (C contiguous, the member's dtype), reading only those bytes.

    `np.savez` stores members with ZIP_STORED, so the member is a plain
    `.npy` inside the zip: the header gives shape, dtype and order, and
    the rows follow in C order. Exactly `n_rows * row_bytes` are read
    after the header; the rest of the member (1.4 GB for Istella's
    `x_train`) is never touched. A member that is Fortran ordered or
    compressed falls back to `np.load` of the whole member and slices.
    Returns (rows, total_rows); raises if the member has fewer than
    `n_rows` rows, because a short cache is a broken cache, not a smaller
    fixture."""
    import zipfile
    with zipfile.ZipFile(npz_path) as z:
        name = key + ".npy"
        info = z.getinfo(name)
        with z.open(name) as fh:
            version = np.lib.format.read_magic(fh)
            if version == (1, 0):
                shape, fortran, dtype = np.lib.format.read_array_header_1_0(fh)
            else:
                shape, fortran, dtype = np.lib.format.read_array_header_2_0(fh)
            if len(shape) != 2:
                raise RuntimeError("%s[%s]: want a 2-d member, got shape %s" % (npz_path, key, shape))
            total_rows, n_cols = int(shape[0]), int(shape[1])
            if total_rows < n_rows:
                raise RuntimeError(
                    "%s[%s] has %d rows, fewer than the %d the kNN prefix needs; the cache is short, "
                    "remove it and rebuild it with `python tools/speed_gbdt_arm.py --download`"
                    % (npz_path, key, total_rows, n_rows))
            if fortran or info.compress_type != zipfile.ZIP_STORED or dtype.hasobject:
                arr = np.load(npz_path)[key]
                return np.ascontiguousarray(arr[:n_rows]), total_rows
            row_bytes = n_cols * dtype.itemsize
            want = n_rows * row_bytes
            buf = bytearray(want)
            view = memoryview(buf)
            got = 0
            while got < want:
                n = fh.readinto(view[got:])
                if not n:
                    break
                got += n
            if got != want:
                raise RuntimeError("%s[%s]: short read, %d of %d bytes" % (npz_path, key, got, want))
    rows = np.frombuffer(buf, dtype=dtype).reshape(n_rows, n_cols)
    return np.ascontiguousarray(rows), total_rows


def load_real_prefix(dataset, root=None, log=None):
    """The 404,000 x d float32 prefix of one real dataset from the trees
    harness's cache, and a load record (cache path, member, columns,
    total rows, read seconds). Raises with the download command when the
    cache is missing; never downloads and never decodes the raw files."""
    folder, npz_path = real_paths(dataset, root)
    _folder_name, _cache_name, key, cols = REAL_CACHES[dataset]
    if not real_present(dataset, root):
        raise _missing_cache_error(dataset, npz_path)
    harness = _trees_harness()
    t0 = time.perf_counter()
    rows, total_rows = npz_member_prefix(npz_path, key, REAL_PREFIX_ROWS)
    column_names = None
    if cols is not None:
        names = tuple(getattr(harness, cols))
        all_names = tuple(harness.TAXI_FEATURES)
        take = [all_names.index(c) for c in names]
        rows = np.ascontiguousarray(rows[:, take])
        column_names = list(names)
    if rows.dtype != np.float32:
        rows = np.ascontiguousarray(rows.astype(np.float32))
    read_seconds = time.perf_counter() - t0
    _say(log, "knn_datasets: read the %d-row %s prefix (%d columns) from %s in %.2f s"
         % (REAL_PREFIX_ROWS, dataset, rows.shape[1], npz_path, read_seconds))
    record = {"data_root": root or data_root(), "cache_path": npz_path, "npz_key": key,
              "cache_bytes": os.path.getsize(npz_path), "cache_total_rows": total_rows,
              "columns": column_names, "prefix_rows": REAL_PREFIX_ROWS, "features": int(rows.shape[1]),
              "download": None, "cache_hit": True, "parse_seconds": None, "read_seconds": read_seconds,
              "built_by": "python tools/speed_gbdt_arm.py --download %s" % dataset,
              "parse": "leading %d rows of the trees harness's NumPy cache%s; float32, C contiguous; no shuffle, no scaling, no deduplication"
              % (REAL_PREFIX_ROWS, ", TAXI_NUMERIC columns" if cols else "")}
    record["sha256_block"] = sha256_array(rows)
    return rows, record


def real_block(dataset, n_index=REAL_INDEX_ROWS, n_queries=REAL_QUERY_ROWS, data_root=None, log=None):
    """The kNN fixture of one real dataset (`taxi` or `istella`): index =
    prefix rows [0, n_index), queries = prefix rows [400,000, 400,000 +
    n_queries). The queries always come from the query region, so a
    smaller `n_index` (a `--quick` smoke) never turns a query row into an
    index row. Same dict shape as the retired `higgs_block`: the two
    float32 blocks, `d` from the data (11 or 220), the row ranges, the
    sha256 of the whole prefix block and of each part, and the load record
    under `source`. Never downloads (see `load_real_prefix`)."""
    if dataset not in REAL_CACHES:
        raise ValueError("unknown real dataset %r; choose from %s" % (dataset, REAL_DATASETS))
    if not (0 < n_index <= REAL_INDEX_ROWS):
        raise ValueError("n_index must be in 1..%d, got %d" % (REAL_INDEX_ROWS, n_index))
    if not (0 < n_queries <= REAL_QUERY_ROWS):
        raise ValueError("n_queries must be in 1..%d, got %d" % (REAL_QUERY_ROWS, n_queries))
    x, record = load_real_prefix(dataset, data_root, log)
    index = np.ascontiguousarray(x[:n_index])
    queries = np.ascontiguousarray(x[REAL_INDEX_ROWS: REAL_INDEX_ROWS + n_queries])
    return {
        "dataset": dataset, "fixture": "%s-prefix-%d" % (dataset, REAL_PREFIX_ROWS),
        "index": index, "queries": queries, "d": int(x.shape[1]),
        "index_rows": [0, n_index], "query_rows": [REAL_INDEX_ROWS, REAL_INDEX_ROWS + n_queries],
        "sha256_block": record["sha256_block"], "sha256_index": sha256_array(index), "sha256_queries": sha256_array(queries),
        "source": record,
    }


def distinct_row_count(x):
    v = np.ascontiguousarray(x).view(np.dtype((np.void, x.dtype.itemsize * x.shape[1])))
    return int(np.unique(v).shape[0])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--prefetch", choices=("taxi", "istella", "higgs"), required=True,
                    help="taxi | istella: check the trees harness's cache is present, read the 404,000-row prefix and print its record (shapes, d, sha256); never downloads. higgs: RETIRED (2026-09-11), download (if absent) and decode the HIGGS prefix; only to re-read old evidence")
    ap.add_argument("--data-root", default=None, help="override GBM_BENCH_DATA / ~/datasets/gbm-bench")
    args = ap.parse_args()
    t0 = time.perf_counter()
    if args.prefetch == "higgs":
        block = higgs_block(data_root=args.data_root)
    else:
        block = real_block(args.prefetch, data_root=args.data_root)
    out = {k: v for k, v in block.items() if k not in ("index", "queries")}
    out["index_shape"] = list(block["index"].shape)
    out["queries_shape"] = list(block["queries"].shape)
    out["dtype"] = str(block["index"].dtype)
    out["distinct_index_rows"] = distinct_row_count(block["index"])
    out["duplicate_index_rows"] = block["index"].shape[0] - out["distinct_index_rows"]
    out["prefetch_seconds"] = time.perf_counter() - t0
    json.dump(out, sys.stdout, indent=1, default=str)
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
