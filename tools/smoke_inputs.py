#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""THE MEASUREMENT SMOKE'S INPUTS: one fixture set for the identity verification and the smoke (Andrew 2026-11-08:
"the smoke should be light"; "maybe the smoke fixtures should be the same as the verify, for those algos").

The smoke judges completion and the fields the board ingests, never speed. Until this file every board driver built
its blocks from the board DATASETS (istella / taxi / higgs .npz, the corpora), so even a smoke staged gigabytes. Under
`bench_board.py --smoke --inputs synthetic` (set by tools/release_measurement_smoke.py) THE SMOKE STAGES NO DATA AT
ALL (Andrew 2026-11-08): every recipe's inputs are generated on the box (docs/measurements/SMOKE_INPUTS.md lists them
recipe by recipe):

  verify fixture   the identity verification's own fixture `base` (tools/identity_break.py `fixture("base")`: the
                   20,000 x 16 float32 draw of seed 0, its two labels, and `heldout("base")`, the seed-1 draw the
                   lanes' infer column reads). A driver's prep takes its arrays from here (`prep_sources`) instead
                   of the dataset. A recipe with no data file keeps the driver's own seeded inputs.
  synthetic        a recipe whose purpose is a dataset FILE loader (the trees family's speed_gbdt_arm loaders) or a
                   corpus reader: the same loader code opens a SYNTHETIC FILE in the full file's own schema (the same
                   keys, dtypes and layout) at the loader-side smoke size (SMOKE_TRAIN_ROWS head + SMOKE_TEST_ROWS
                   tail rows; CORPUS_SYNTH_BYTES of generated text), written on the box when the loader opens it and
                   deleted with the race's scratch.
  NO VERIFY LANE   no identity lane verifies what the board measures: the recipe keeps its own inputs at the smoke
                   size, and a file it reads is synthetic too (listed for Andrew's decision).

The mode is the environment variable MODE_ENV, which bench_board sets for every driver in the smoke; without it
(every timed run) nothing here is reached: the measured arms and their inputs are untouched.

Numbers here are bench-side input preparation before every clock (tools, never the runtime).
"""
import importlib.util
import os
import types

HERE = os.path.dirname(os.path.abspath(__file__))

#: the variable that selects the smoke's inputs (bench_board.py --inputs sets it in every driver's environment)
MODE_ENV = "MOJOLEARN_BOARD_INPUTS"
#: synthetic: the release measurement smoke (bench_board.py --smoke --inputs synthetic); verify-fixtures: the drivers'
#: prep flag for the recipes on the verify fixture
DATASETS_MODE, SYNTHETIC_MODE, FIXTURES_MODE = "datasets", "synthetic", "verify-fixtures"
MODES = (DATASETS_MODE, SYNTHETIC_MODE)
#: the identity verification's fixture the smoke runs on (identity_break.FIXTURES[0])
FIXTURE = "base"

#: THE LOADER-SIDE SMOKE SIZE (the synthetic files' rows). Head rows: the neural fit lanes' `small` shape reads 4,096
#: train + 1,024 held-out rows of HIGGS from the front of the file (bench_board_neural.FIT_ROWS["small"]), the trees
#: smoke caps its train rows at bench_board.SMOKE_ROWS (2,000), so 5,120 head rows serve both. Tail rows: the loaders'
#: test split is a fixed tail of the file (500,000 rows at full size); the synthetic file has 1,024 (whole queries in
#: the ranking file).
SMOKE_TRAIN_ROWS = 5120
SMOKE_TEST_ROWS = 1024
#: the marker array a synthetic gbm-bench file carries where the loader's test rows are the file's tail: True on them
#: (the loaders count their test rows from it, after their own row filter)
TAIL = "smoke_tail"


def mode(env=None):
    env = os.environ if env is None else env
    got = (env.get(MODE_ENV) or DATASETS_MODE).strip()
    if got not in MODES:
        raise SystemExit("smoke_inputs: %s=%r; one of %s" % (MODE_ENV, got, ", ".join(MODES)))
    return got


def synthetic_on(env=None):
    """True in the smoke's synthetic mode: every dataset file a loader opens is generated on the box."""
    return mode(env) == SYNTHETIC_MODE


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
    --inputs value: 'datasets' returns the real ones unchanged (in the smoke's synthetic mode their loaders open the
    synthetic files); 'verify-fixtures' the fixture stand-ins."""
    if (inputs or DATASETS_MODE) == DATASETS_MODE:
        return harness, knn
    if inputs != FIXTURES_MODE:
        raise SystemExit("smoke_inputs: prep --inputs %r; datasets or %s" % (inputs, FIXTURES_MODE))
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


# ---------------------------------------------------------------------------------------------- the synthetic files
# A loader or corpus recipe (and a NO VERIFY LANE recipe that reads a file) runs on a SYNTHETIC FILE in the loader's
# own schema: the same keys, dtypes and layout the full file has, at the loader-side smoke size, generated on the box
# the moment the loader opens it (synthetic_path), into SYNTH_DIR_ENV (bench_board puts it under the race's work/,
# which --drop-race-scratch deletes once the record is written; a prep-only process's under its own out, which the
# smoke deletes with its group). Nothing is staged, downloaded or pinned. Deterministic: seeded by the schema name.
#: the directory the synthetic files are written to (bench_board.set_inputs sets it); required in synthetic mode
SYNTH_DIR_ENV = "MOJOLEARN_SMOKE_SYNTH_DIR"
#: the files a synthetic schema stands for, by the last two components of the path the loader opens
SCHEMAS = {"higgs/higgs_speed.npz": "higgs", "taxi/taxi_speed.npz": "taxi", "istella/istella_speed.npz": "istella",
           "istella/istella_rank.npz": "istella-rank", "enwik8/input.txt": "enwik8",
           "pile_github/input.txt": "pile_github"}
#: the feature width of each npz schema (the full files': HIGGS 28, taxi 16, Istella-S 220)
SCHEMA_WIDTH = {"higgs": 28, "taxi": 16, "istella": 220, "istella-rank": 220}
#: a synthetic corpus: 64 KiB, 32 documents of the text lanes' 2,048 bytes, so their 1-in-10 held-out split holds
#: rows of both corpora (a few KB would leave it empty)
CORPUS_SYNTH_BYTES = 64 * 1024
#: Istella-S ranking queries in the synthetic file: this many rows per query
RANK_QUERY_ROWS = 32
_WORDS = {"enwik8": ("the", "of", "and", "in", "was", "history", "city", "[[", "]]", "language", "century", "war",
                     "river", "known", "first", "{{", "}}", "people", "north", "music"),
          "pile_github": ("def", "return", "if", "else", "for", "in", "import", "self", "(", ")", ":", "{", "}",
                          ";", "int", "const", "=", "==", "#include", "void")}


def _schema_of(path):
    rel = "/".join(os.path.normpath(str(path)).split(os.sep)[-2:])
    if rel not in SCHEMAS:
        raise SystemExit("smoke_inputs: no synthetic schema for %s (known: %s)" % (path, ", ".join(sorted(SCHEMAS))))
    return rel


def synthetic_dir(env=None):
    env = os.environ if env is None else env
    d = (env.get(SYNTH_DIR_ENV) or "").strip()
    if not d:
        raise SystemExit("smoke_inputs: synthetic inputs need %s (bench_board.py --inputs synthetic sets it)"
                         % SYNTH_DIR_ENV)
    return d


def synthetic_path(path, env=None):
    """The synthetic file standing for `path` (written now when it is not there yet); idempotent on its own result."""
    rel = _schema_of(path)
    dst = os.path.join(synthetic_dir(env), rel)
    if not os.path.isfile(dst):
        write_synthetic(rel, dst)
    return dst


def loader_path(path, env=None):
    """The file a loader opens: `path`, or its synthetic stand-in in the smoke's synthetic mode."""
    return synthetic_path(path, env) if synthetic_on(env) else path


def _rng(rel):
    import hashlib
    return _np().random.default_rng(int.from_bytes(hashlib.sha256(rel.encode()).digest()[:8], "little"))


def synthetic_arrays(rel):
    """{key: array} of the synthetic npz for schema `rel`: the full file's keys and dtypes, SMOKE_TRAIN_ROWS head
    rows and SMOKE_TEST_ROWS tail rows (TAIL marks the tail where the loader's test split is the file's end)."""
    np = _np()
    rng = _rng(rel)
    name = SCHEMAS[rel]
    w = SCHEMA_WIDTH[name]
    n, m = SMOKE_TRAIN_ROWS, SMOKE_TEST_ROWS
    if name == "higgs":
        x = rng.standard_normal((n + m, w)).astype(np.float32)
        return {"x": x, "y": (x[:, 0] + 0.5 * x[:, 1] > 0).astype(np.float32), TAIL: _tail(n, m)}
    if name == "taxi":
        # the decode's 16 columns (speed_gbdt_arm.TAXI_FEATURES): integer ids and small counts where the real file
        # has them (vendor, ratecode, store_fwd, zones, hour, weekday, day), positives elsewhere; fare > 0, a tip on
        # every trip, about 70 % card-paid. The card rows of the head and the tail hold at least the smoke size.
        k = 2 * (n + m)
        x = np.abs(rng.standard_normal((k, w))).astype(np.float32)
        ids = {0: 3, 3: 6, 4: 2, 5: 264, 6: 264, 7: 24, 8: 7, 9: 29}
        for j, hi in ids.items():
            x[:, j] = rng.integers(0, hi, k).astype(np.float32)
        fare = (rng.random(k) * 40 + 3).astype(np.float32)
        tip = (fare * rng.random(k) * 0.4).astype(np.float32)
        card = rng.random(k) < 0.7
        card[: n] = True                              # every task's head rows reach the smoke size
        card[-m:] = True                              # the classification tail holds SMOKE_TEST_ROWS card trips
        return {"x": x, "fare": fare, "tip": tip, "card": card, TAIL: _tail(k - m, m)}
    if name == "istella":
        xt = rng.standard_normal((n, w)).astype(np.float32)
        xe = rng.standard_normal((m, w)).astype(np.float32)
        grade = lambda X: np.clip(np.floor(X[:, 0] + X[:, 1] + 2), 0, 4).astype(np.float32)  # noqa: E731
        return {"x_train": xt, "r_train": grade(xt), "x_test": xe, "r_test": grade(xe)}
    if name == "istella-rank":
        q = RANK_QUERY_ROWS
        mt = (m // q) * q
        xe = rng.standard_normal((mt, w)).astype(np.float32)
        return {"qid_train": np.repeat(np.arange(n // q + 1, dtype=np.int64), q)[:n],
                "qid_test": np.repeat(np.arange(mt // q, dtype=np.int64), q), "x_test": xe,
                "r_test": np.clip(np.floor(xe[:, 0] + 2), 0, 4).astype(np.float32)}
    raise ValueError(rel)


def synthetic_text(rel, nbytes=CORPUS_SYNTH_BYTES):
    """Generated text of the corpus's kind (prose for enwik8, code for pile_github), `nbytes` bytes."""
    rng = _rng(rel)
    words = _WORDS[SCHEMAS[rel]]
    out = bytearray()
    while len(out) < nbytes:
        line = " ".join(words[i] for i in rng.integers(0, len(words), 12))
        out += (line + "\n").encode()
    return bytes(out[:nbytes])


def _tail(head, tail):
    np = _np()
    return np.concatenate([np.zeros(head, dtype=bool), np.ones(tail, dtype=bool)])


def write_synthetic(rel, dst):
    """Write schema `rel`'s synthetic file to `dst` (atomically: another process of the race may open it)."""
    np = _np()
    os.makedirs(os.path.dirname(os.path.abspath(dst)), exist_ok=True)
    tmp = "%s.%d.part" % (dst, os.getpid())
    with open(tmp, "wb") as fh:
        if rel.endswith(".txt"):
            fh.write(synthetic_text(rel))
        else:
            np.savez(fh, **synthetic_arrays(rel))
    os.replace(tmp, dst)
    return dst


def synthetic_bound_bytes(rel):
    """An upper bound of a synthetic file's bytes (plan time): rows x (width + 4 side columns) x 4, plus framing."""
    if rel.endswith(".txt"):
        return CORPUS_SYNTH_BYTES
    rows = 2 * (SMOKE_TRAIN_ROWS + SMOKE_TEST_ROWS)
    return rows * (SCHEMA_WIDTH[SCHEMAS[rel]] + 4) * 4 + 64 * 1024


def schema_of_key(key):
    """The synthetic schema an R2 key's file is generated in ('taxi/taxi_speed.npz'), or None."""
    rel = "/".join(key.split("/")[-2:])
    return rel if rel in SCHEMAS else None


def tail_rows(cached, keep=None):
    """A synthetic file's test-tail rows after the loader's own row filter `keep` (None: every row)."""
    np = _np()
    tail = np.asarray(cached[TAIL])
    return int(tail[keep].sum() if keep is not None else tail.sum())
