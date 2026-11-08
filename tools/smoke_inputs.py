#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""THE MEASUREMENT SMOKE'S INPUTS: one fixture set for the identity verification and the smoke (Andrew 2026-11-08:
"the smoke should be light"; "maybe the smoke fixtures should be the same as the verify, for those algos").

The smoke judges completion and the fields the board ingests, never speed. Until this file every board driver built
its blocks from the board DATASETS (istella / taxi / higgs .npz, the corpora), so even a smoke staged gigabytes. Under
`bench_board.py --smoke --inputs verify-fixtures` (set by tools/release_measurement_smoke.py) every recipe takes one
of three inputs (docs/measurements/SMOKE_INPUTS.md lists them recipe by recipe):

  verify fixture   the identity verification's own fixture `base` (tools/identity_break.py `fixture("base")`: the
                   20,000 x 16 float32 draw of seed 0, its two labels, and `heldout("base")`, the seed-1 draw the
                   lanes' infer column reads). A driver's prep takes its arrays from here (`prep_sources`) instead
                   of the dataset; nothing is staged for it. A recipe with no data file keeps the driver's own
                   seeded inputs (it stages nothing either).
  sliced file      a recipe whose purpose is a dataset FILE loader (the trees family's speed_gbdt_arm loaders) or a
                   corpus reader: the same loader code reads a SLICE of the pinned file (SLICE_TRAIN head rows +
                   SLICE_TEST tail rows of a gbm-bench npz, CORPUS_HEAD_BYTES of a corpus), made once on the Mac
                   from the full file (`release_measurement_smoke.py make-smoke-slices`) and pinned with its
                   sha256 in bench/results/dataset_store/manifest.tsv like every other staged file.
  NO VERIFY LANE   no identity lane verifies what the board measures: the recipe keeps its current inputs at the
                   smoke size, and a file it reads is the slice (listed for Andrew's decision).

The mode is the environment variable MODE_ENV, which bench_board's child_env sets for every driver in the smoke;
without it (every timed run) nothing here is reached: the measured arms and their inputs are untouched.

Numbers here are bench-side input preparation before every clock (tools, never the runtime).
"""
import importlib.util
import os
import types

HERE = os.path.dirname(os.path.abspath(__file__))

#: the variable that selects the smoke's inputs (bench_board.py --inputs sets it in every driver's environment)
MODE_ENV = "MOJOLEARN_BOARD_INPUTS"
DATASETS_MODE, FIXTURES_MODE = "datasets", "verify-fixtures"
MODES = (DATASETS_MODE, FIXTURES_MODE)
#: the identity verification's fixture the smoke runs on (identity_break.FIXTURES[0])
FIXTURE = "base"

#: THE SLICE SIZE (the loader-side smoke size). Head rows: the neural fit lanes' `small` shape reads 4,096 train +
#: 1,024 held-out rows of HIGGS from the front of the file (bench_board_neural.FIT_ROWS["small"]), the trees smoke
#: caps its train rows at bench_board.SMOKE_ROWS (2,000), so 5,120 head rows serve both. Tail rows: the loaders' test
#: split is a fixed tail of the file (500,000 rows at full size); the slice keeps 1,024 of it, whole queries for the
#: ranking file.
SLICE_TRAIN = 5120
SLICE_TEST = 1024
#: a corpus slice: the first 2 MB (the text lanes read 2,048-byte documents; bench_board_extra reads up to 4 MiB of
#: enwik8 and takes what the file holds)
CORPUS_HEAD_BYTES = 2_000_000
#: a taxi slice scans at most this many raw rows from each end for its card-paid rows (the classification task
#: keeps card trips only; about 70 % of the trips): a file that does not hold them there is refused by name
TAXI_SCAN_ROWS = 4 * SLICE_TRAIN
#: the marker array a gbm-bench slice carries: True on the rows of the file's test tail (the loaders take their test
#: rows from it, after their own row filter)
TAIL = "smoke_tail"
#: the infix of a slice's key and file name: gbm-bench/taxi/taxi_speed.npz -> gbm-bench/taxi/taxi_speed.smoke.npz
SLICE_INFIX = ".smoke"
#: where make-smoke-slices writes the slices on the Mac (tools/dataset_store.sh catalog); on a box they land where the
#: loaders look: datasets/<key> for gbm-bench files, r2-stage/<key> for corpora
SLICE_HOME = os.path.join("datasets", "smoke-slices")


def mode(env=None):
    env = os.environ if env is None else env
    got = (env.get(MODE_ENV) or DATASETS_MODE).strip()
    if got not in MODES:
        raise SystemExit("smoke_inputs: %s=%r; one of %s" % (MODE_ENV, got, ", ".join(MODES)))
    return got


def slices_on(env=None):
    """True in the smoke's verify-fixtures mode: every dataset file a loader opens is its slice."""
    return mode(env) == FIXTURES_MODE


def slice_name(name):
    """'taxi_speed.npz' -> 'taxi_speed.smoke.npz' (a key, a relative path or an absolute path)."""
    head, tail = os.path.split(name)
    stem, ext = os.path.splitext(tail)
    if stem.endswith(SLICE_INFIX):
        return name
    return os.path.join(head, stem + SLICE_INFIX + ext) if head else stem + SLICE_INFIX + ext


def loader_path(path, env=None):
    """The file a loader opens: `path`, or its slice beside it in the smoke's verify-fixtures mode."""
    return slice_name(path) if slices_on(env) else path


# ---------------------------------------------------------------------------------------------- the fixture
_IB = []


def _identity_break():
    """tools/identity_break.py, loaded once (stdlib + numpy at import; the lanes need the installed wheel only when
    one runs, and none runs here)."""
    if not _IB:
        spec = importlib.util.spec_from_file_location("smoke_inputs_identity_break",
                                                      os.path.join(HERE, "identity_break.py"))
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        _IB.append(mod)
    return _IB[0]


_BASE = {}


def base():
    """(X, y_clf, y_reg, Xh, yh_clf, yh_reg): fixture `base` and its held-out draw, exactly as the identity lanes
    receive them (identity_break.fixture / heldout; the held-out labels by the same labels_for rule)."""
    if not _BASE:
        ib = _identity_break()
        X, yc, yr = ib.fixture(FIXTURE)
        Xh, yhc, yhr = ib.fixture(FIXTURE, seed=ib.HELDOUT_SEED)
        _BASE["v"] = (X, yc, yr, Xh, yhc, yhr)
    return _BASE["v"]


#: the label each task reads. clf: labels_for's signed rule; reg: its fixed-order linear target; mc: three classes,
#: the rule tools/identity_lanes/neighbors.py gives its multiclass lanes (y_clf + (X[:, 5] > 0)); cat: y_clf.
TASKS = ("clf", "reg", "mc", "cat")


def _np():
    import numpy as np
    return np


def _labels(task, X, yc, yr):
    np = _np()
    if task == "reg":
        return np.asarray(yr, dtype=np.float32)
    if task == "mc":
        return (np.asarray(yc, dtype=np.int32) + (X[:, 5] > 0).astype(np.int32)).astype(np.float32)
    return np.asarray(yc, dtype=np.float32)


def _codes(X, ref, k=8):
    """Each column of X as k integer codes by the quantiles of `ref`'s column (dense 0..k-1 on `ref`)."""
    np = _np()
    out = np.empty(X.shape, dtype=np.float32)
    for j in range(X.shape[1]):
        edges = np.quantile(ref[:, j].astype(np.float64), np.linspace(0, 1, k + 1)[1:-1])
        out[:, j] = np.searchsorted(edges, X[:, j], side="right")
    return out


def for_recipe(family, lane, dataset, rows=None, task="clf"):
    """(X, y, meta) a recipe's prep takes in the smoke: the fit rows of fixture `base` (the first `rows`, all 20,000
    when None), the task's label, and meta = {Xq, yq: the held-out draw (as many rows), fixture, task, rows, source}.
    `family`, `lane` and `dataset` name the recipe in the record; the arrays are the verification's own, the same for
    every recipe (a smoke judges completion, not the data)."""
    if task not in TASKS:
        raise ValueError("smoke_inputs: task %r; one of %s" % (task, ", ".join(TASKS)))
    np = _np()
    X, yc, yr, Xh, yhc, yhr = base()
    n = X.shape[0] if rows is None else max(1, min(int(rows), X.shape[0]))
    Xf, Xq = X[:n], Xh[:n]
    y, yq = _labels(task, Xf, yc[:n], yr[:n]), _labels(task, Xq, yhc[:n], yhr[:n])
    if task == "cat":
        Xf, Xq = _codes(Xf, Xf), _codes(Xq, Xf)
    meta = dict(Xq=np.ascontiguousarray(Xq, dtype=np.float32), yq=yq, fixture=FIXTURE, task=task, rows=n,
                recipe="%s/%s/%s" % (family, lane, dataset),
                source="tools/identity_break.py fixture(%r) rows [0, %d) and heldout(%r) rows [0, %d)%s"
                       % (FIXTURE, n, FIXTURE, n, ", quantile-coded to 8 codes" if task == "cat" else ""))
    return np.ascontiguousarray(Xf, dtype=np.float32), y, meta


# ---------------------------------------------------------------------------------------------- the stand-ins
#: the fixture's 16 columns under one name each: the stand-in's TAXI_FEATURES and TAXI_NUMERIC are this tuple, so
#: classical_two_datasets._taxi_numeric (and every column pick by name) selects the fixture's columns as they are
FIXTURE_COLUMNS = tuple("f%d" % j for j in range(16))


class FixtureHarness(object):
    """The part of tools/speed_gbdt_arm.py the drivers' prep reads (load_taxi, load_istella, data_root, the taxi
    column names), answered from fixture `base`: X_train / y_train the fit draw, X_test / y_test the held-out draw.
    A driver's derivations (stride samples, sentinel cleaning, standardization, quantile codes) run on it unchanged."""

    TAXI_FEATURES = FIXTURE_COLUMNS
    TAXI_NUMERIC = FIXTURE_COLUMNS

    def __init__(self, family="smoke"):
        self.family = family

    def data_root(self):
        return "verify-fixture:%s" % FIXTURE

    def _data(self, dataset, task):
        X, y, meta = for_recipe(self.family, "prep", dataset, None, task)
        return types.SimpleNamespace(X_train=X, X_test=meta["Xq"], y_train=y, y_test=meta["yq"],
                                     name="%s-fixture-%s" % (dataset, task), task=task, source=meta["source"])

    def load_taxi(self, size, rows_cap=None, regression=False, multiclass=False, categorical=False):
        return self._data("taxi", "reg" if regression else "mc" if multiclass else "cat" if categorical else "clf")

    def load_istella(self, size, rows_cap=None, regression=False, multiclass=False):
        return self._data("istella", "reg" if regression else "mc" if multiclass else "clf")


class FixtureKnn(object):
    """tools/knn_datasets.real_block's shape, answered from fixture `base`: index = the fit draw's first rows,
    queries = the held-out draw's first rows (never an index row)."""

    @staticmethod
    def real_block(dataset, n_index=None, n_queries=None, data_root=None, log=None):
        np = _np()
        X, _, _, Xh, _, _ = base()
        n_index = min(int(n_index or X.shape[0]), X.shape[0])
        n_queries = min(int(n_queries or Xh.shape[0]), Xh.shape[0])
        index, queries = np.ascontiguousarray(X[:n_index]), np.ascontiguousarray(Xh[:n_queries])
        src = dict(fixture=FIXTURE, rule="tools/identity_break.py fixture/heldout")
        return {"dataset": dataset, "fixture": "verify-%s" % FIXTURE, "index": index, "queries": queries,
                "d": int(X.shape[1]), "index_rows": [0, n_index], "query_rows": ["heldout", n_queries],
                "sha256_block": _sha(np.concatenate([index, queries])), "sha256_index": _sha(index),
                "sha256_queries": _sha(queries), "source": src}


def _sha(a):
    import hashlib
    np = _np()
    a = np.ascontiguousarray(a)
    h = hashlib.sha256()
    h.update(str(a.dtype).encode())
    h.update(str(a.shape).encode())
    h.update(a.data)
    return h.hexdigest()


def prep_sources(inputs, harness, knn):
    """THE ONE CALL a driver's prep makes: (harness, knn module) to read its blocks from. `inputs` is the prep's
    --inputs value: 'datasets' returns the real ones unchanged; 'verify-fixtures' the fixture stand-ins."""
    if (inputs or DATASETS_MODE) == DATASETS_MODE:
        return harness, knn
    if inputs != FIXTURES_MODE:
        raise SystemExit("smoke_inputs: --inputs %r; one of %s" % (inputs, ", ".join(MODES)))
    return FixtureHarness(), FixtureKnn()


def series(kind, n_series, n_obs):
    """(n_series, n_obs) float32 series from fixture `base` the way the identity sequence lanes build theirs
    (tools/identity_lanes/sequence.py): column j of the fit draw, then of the held-out draw, as series j (the draws
    are 20,000 rows, so n_obs <= 20,000). kind 'seasonal' (ts): a period-24 wave plus the column, positive level;
    'intermittent' (tsi): the column where it exceeds 0.8 plus 1, else 0 (sequence-croston's rule); 'garch' (tsr):
    the bounded shocks 1.4 z / (1 + |z|) (sequence-garch's rule)."""
    np = _np()
    X, _, _, Xh, _, _ = base()
    cols = np.concatenate([X, Xh], axis=1)
    n_obs = min(int(n_obs), cols.shape[0])
    Z = np.ascontiguousarray(cols[:n_obs, np.arange(n_series) % cols.shape[1]].T, dtype=np.float32)
    if kind == "intermittent":
        return np.where(Z > np.float32(0.8), Z + np.float32(1.0), np.float32(0.0)).astype(np.float32)
    if kind == "garch":
        return (np.float32(1.4) * Z / (np.float32(1.0) + np.abs(Z))).astype(np.float32)
    t = np.arange(n_obs, dtype=np.float32)
    wave = (np.sin(t * np.float32(2 * np.pi / 24)) * np.float32(3.0)).astype(np.float32)
    return (Z + wave[None, :] + np.float32(10.0)).astype(np.float32)


def zone_matrix(rows):
    """A nonnegative count-like matrix from fixture `base`: the positive part of the first `rows` fit rows, as the
    identity ALS lane builds its implicit-feedback matrix (tools/identity_lanes/decomp.py x-decomp-als)."""
    np = _np()
    X = base()[0]
    return np.ascontiguousarray(np.maximum(X[:max(1, min(int(rows), X.shape[0]))], np.float32(0)), dtype=np.float32)


# ---------------------------------------------------------------------------------------------- the slices
#: the gbm-bench files a slice is made of: key -> (form, the arrays it holds). 'tail': one array set whose test rows
#: are the file's last rows (higgs x/y; taxi x/fare/tip/card), sliced to head + tail with the TAIL marker; 'split':
#: separate train/test arrays (istella_speed x_train/r_train/x_test/r_test); 'rank': istella_rank's train qids for the
#: head rows and whole queries of its test split.
SLICE_FORMS = {"gbm-bench/higgs/higgs_speed.npz": "tail", "gbm-bench/taxi/taxi_speed.npz": "taxi",
               "gbm-bench/istella/istella_speed.npz": "split", "gbm-bench/istella/istella_rank.npz": "rank"}
#: the declared width of each file's feature matrix (for the plan-time byte bound only; the slice records its own)
SLICE_WIDTH = {"gbm-bench/higgs/higgs_speed.npz": 28, "gbm-bench/taxi/taxi_speed.npz": 16,
               "gbm-bench/istella/istella_speed.npz": 220, "gbm-bench/istella/istella_rank.npz": 220}


def slice_key(key):
    return slice_name(key)


def sliceable(key):
    return key in SLICE_FORMS or key.startswith("corpus/")


def slice_bound_bytes(key):
    """An upper bound of the slice's bytes from its declared shape (plan time, before any slice exists): every row
    it can hold x (width + 4 side columns) x 4 bytes, plus the npz framing; a corpus slice is CORPUS_HEAD_BYTES."""
    if key.startswith("corpus/"):
        return CORPUS_HEAD_BYTES
    rows = 2 * TAXI_SCAN_ROWS if SLICE_FORMS[key] == "taxi" else SLICE_TRAIN + 2 * SLICE_TEST
    return rows * (SLICE_WIDTH[key] + 4) * 4 + 64 * 1024


def _take_tail(arrays, head, tail):
    np = _np()
    n = next(iter(arrays.values())).shape[0]
    if head + tail > n:
        raise SystemExit("smoke_inputs: the file has %d rows; the slice needs %d head + %d tail" % (n, head, tail))
    out = {k: np.ascontiguousarray(np.concatenate([v[:head], v[n - tail:]])) for k, v in arrays.items()}
    out[TAIL] = np.concatenate([np.zeros(head, dtype=bool), np.ones(tail, dtype=bool)])
    return out


def _count_rows(mask, want):
    """The fewest leading rows of `mask` holding `want` True rows (refused when it holds fewer)."""
    np = _np()
    c = np.cumsum(mask)
    if c.shape[0] == 0 or c[-1] < want:
        raise SystemExit("smoke_inputs: %d rows hold %d card trips; the slice needs %d"
                         % (mask.shape[0], int(c[-1]) if c.shape[0] else 0, want))
    return int(np.searchsorted(c, want) + 1)


def make_slice(key, src, dst):
    """Write the slice of the full file `src` (R2 key `key`) to `dst`. Returns a record of what it holds."""
    np = _np()
    os.makedirs(os.path.dirname(os.path.abspath(dst)), exist_ok=True)
    tmp = dst + ".part"
    if key.startswith("corpus/"):
        with open(src, "rb") as fh:
            head = fh.read(CORPUS_HEAD_BYTES)
        with open(tmp, "wb") as fh:
            fh.write(head)
        os.replace(tmp, dst)
        return dict(key=slice_key(key), source=key, head_bytes=len(head))
    form = SLICE_FORMS[key]
    with np.load(src) as z:
        if form == "tail":
            out = _take_tail({k: z[k] for k in ("x", "y")}, SLICE_TRAIN, SLICE_TEST)
        elif form == "taxi":
            card = np.asarray(z["card"])
            n = card.shape[0]
            # head: enough raw rows for SLICE_TRAIN rows of every task (the card-paid ones are the fewer); tail: the
            # raw rows holding the file's last SLICE_TEST card trips (the classification test rows)
            head = max(SLICE_TRAIN, _count_rows(card[:TAXI_SCAN_ROWS], SLICE_TRAIN))
            tail = max(SLICE_TEST, _count_rows(card[::-1][:TAXI_SCAN_ROWS], SLICE_TEST))
            if head + tail > n:
                raise SystemExit("smoke_inputs: %s has %d rows; the slice needs %d head + %d tail" % (key, n, head, tail))
            out = _take_tail({k: z[k] for k in ("x", "fare", "tip", "card")}, head, tail)
        elif form == "split":
            out = {"x_train": z["x_train"][:SLICE_TRAIN], "r_train": z["r_train"][:SLICE_TRAIN],
                   "x_test": z["x_test"][:SLICE_TEST], "r_test": z["r_test"][:SLICE_TEST]}
        else:
            qid = np.asarray(z["qid_test"])
            starts = np.flatnonzero(np.diff(qid) != 0) + 1
            cut = starts[starts <= SLICE_TEST]
            m = int(cut[-1]) if cut.size else min(SLICE_TEST, qid.shape[0])
            out = {"qid_train": z["qid_train"][:SLICE_TRAIN], "x_test": z["x_test"][:m],
                   "r_test": z["r_test"][:m], "qid_test": qid[:m]}
            for k in z.files:
                if k not in out:
                    raise SystemExit("smoke_inputs: %s holds %r, which the slice does not know" % (key, k))
    with open(tmp, "wb") as fh:
        np.savez(fh, **{k: np.ascontiguousarray(v) for k, v in out.items()})
    os.replace(tmp, dst)
    return dict(key=slice_key(key), source=key, arrays={k: list(v.shape) for k, v in out.items()})


def tail_rows(cached, keep=None):
    """A slice's test rows after the loader's own row filter `keep` (None: every row)."""
    np = _np()
    tail = np.asarray(cached[TAIL])
    return int(tail[keep].sum() if keep is not None else tail.sum())
