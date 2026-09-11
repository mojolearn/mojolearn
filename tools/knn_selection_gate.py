#!/usr/bin/env python3
"""kNN selection gate: baseline versus candidate selector arm, IDENTICAL only.

DEVIATION 2496 (lane brief: docs/lanes/BRIEF_knn_selection_2026-09-10.md).

WHAT THIS IS
------------
One process, one GPU box, the public `mojolearn.neighbors.NearestNeighbors`
request (host in, host out, the same boundary the cached cuML row in
`bench/OPPONENT_REFERENCE.md` was priced at). It builds four fixtures,
runs every selector arm on each of them, and refuses to bless a candidate
unless ALL of the following hold:

  1. complete bitwise equality of distances AND indices between the
     baseline arm, every candidate arm, and the build's DEFAULT (env unset);
  2. the pinned order in every output row: distances non-decreasing, equal
     distances ordered by ascending index, no index repeated, all in range;
  3. planted exact matches sit at rank 0 (by index; whether their distance
     bits are exactly zero is reported, not asserted, see `check_planted`),
     and planted duplicate-row groups (equal distance by construction: same
     bytes, same FMA chain) come out with equal bits, ascending by index;
  4. an independent float64 host oracle agrees on the neighbor lists
     (complete on the tie fixture, sampled on the 400k fixtures), except
     where the oracle itself cannot separate two candidates beyond float32
     resolution, which is counted and reported, never silently accepted;
  5. REACH: for every arm, the sabotage switch flips the output, and a
     clean call afterwards restores the clean bits. An arm whose sabotage
     does not flip is an arm this process never ran, and the gate FAILS.
     On a build without the trial hook, sabotage cannot flip anything, so
     this gate fails by design until the hook lands: a gate that cannot
     fail is not a gate.

Then it prices ordinary requests: EVERY later `--arms` arm is paired with
the first one (DEVIATION 2522; before it only the second arm was timed):
per pair, one warmup per arm, then `--pairs` alternating timed pairs in
order (A, B) and `--pairs` more in order (B, A). Every sample is retained,
both orders are reported separately and pooled, phase medians are per arm,
and the output of every timed request is compared, outside the timed
region, against the clean reference of that fixture. One `timing` row per
fixture, k and pair. The whole process runs under a hard deadline
(`--deadline`, default 300 s); on expiry the partial JSON is written and
the exit code is 3.

THE SWITCHES (runtime, read by the native side per request)
-----------------------------------------------------------
  MOJOLEARN_KNN_SELECT=<name>       explicit arm: `baseline` (the
                                    2026-09-09 kernel), `uniform` (C4,
                                    DEVIATION 2497), `headbound` (C4 +
                                    C1, DEVIATION 2498; NEGATIVE on the
                                    H100 2026-09-11), `warpbound` (C4 +
                                    C2, DEVIATION 2515: warp-scope group
                                    bound, shuffles only; NEGATIVE on the
                                    H100 2026-09-11) or `deferred` (C4 +
                                    DEVIATION 2517: admitted keys queue in
                                    four registers per lane and the
                                    K-chain runs only at warp-uniform
                                    drains; NEGATIVE on the H100
                                    2026-09-11), `capk` (DEVIATION 2521:
                                    the K-specialized kernel instantiated
                                    with CAP = K, a list of exactly k
                                    slots; raises on the generic bucket)
                                    or `capk_selp` (capk plus a
                                    branch-free min/max carry chain),
                                    `voteguard` (DEVIATION 2522: the
                                    uniform arm with the K-chain behind a
                                    warp-uniform `vote` guard, output
                                    valid, bit-identical by construction).
                                    Unset = the build's default.
                                    Unknown names RAISE on the native side,
                                    never fall back.
  MOJOLEARN_KNN_SELECT_SABOTAGE=1   deliberately perturbs the selected
                                    arm's own code path so a reached arm
                                    cannot return clean bits: baseline and
                                    uniform flip bit 0 of the index half of
                                    the composite key for the first element
                                    of every unrolled batch, inside their
                                    own loop form; headbound takes the
                                    block minimum head as the bound
                                    instead of the k-th, inside the refresh;
                                    warpbound clears bit 63 of its reduced
                                    bound inside its refresh, so every
                                    non-negative-distance key after the
                                    first 4,096 columns is rejected;
                                    deferred skips the newest queued key
                                    at every drain, inside its drain;
                                    capk and capk_selp carry the uniform
                                    flip (their scan is the uniform loop
                                    with a k-deep list); voteguard carries
                                    the uniform flip INSIDE its guarded
                                    and admitted path, so a flip proves
                                    the guarded body ran.

Both are honored only by a binding built with
`-D MOJOLEARN_KNN_SELECT_TRIAL=1` (the hook the brief specifies; not on any
shipped build). The names are overridable with `--arm-env` /
`--sabotage-env` if the follow-on lane chooses differently.

TIMING-ONLY ARMS (`--timing-only-arms`, DEVIATION 2516)
--------------------------------------------------------
`skiprank`, `skipscan` and `scanonly1` are selector arms whose OUTPUT IS
INVALID BY CONSTRUCTION: they run one phase of the shipped selector kernel
(the scan, or the rank phase) and skip the other, so the two phases can be
priced separately on the box instead of modeled. `noshift` (DEVIATION
2522) is the same kind: the uniform scan with its K-deep list and its
admission compare, but an admitted key overwrites the threshold slot
instead of running the carry chain, so select_ms(uniform) minus
select_ms(noshift) is the chain's own cost. They are EXCLUDED from the
correctness, oracle and reach sections. In the timing block each one is
paired with the first `--arms` arm exactly like a candidate (one warmup
each, `--pairs` pairs in order (A, B) then (B, A), every sample kept) and
reported under a separate `timing_only` table labeled "output invalid;
phase cost only". The only assertion on such an arm is that its output
DIFFERS from the clean reference (an arm whose output equals the reference
did not run its own body); the baseline samples in those pairs must still
equal the reference.

`votecount` (DEVIATION 2522) is listed under `--timing-only-arms` too but
its OUTPUT IS VALID (it is `voteguard` plus a per-block admission counter
whose launcher synchronizes and reads back every launch, so its time is
not a price): for the arms in TIMING_ONLY_VALID_OUTPUT the assertion is
the opposite one, equality with the reference, and the row says
`output_valid` True for both arms. Under the phase-timer build the
binding prints `KNN_ADMIT_RATE warp_steps N any_admit M` once per launch;
the harness sums the lines of a request and records `admit_warp_steps`,
`admit_any_admit` and `admit_rate` (M / N: the fraction of warp-steps on
which some lane of the warp admitted a key, which is the fraction on
which the guarded chain runs) beside the phase milliseconds, so the
per-arm phase medians carry `admit_rate`.

PHASE TIMERS (`--phase-timers`)
-------------------------------
A binding built with `-D MOJOLEARN_KNN_PHASE_TIMERS=1` (the profile phase's
switch: a BUILD define, not an environment variable; the shell script adds
it to the binding build under MOJOLEARN_KNN_SELECTION_PHASE_TIMERS=1)
synchronizes after every launch class and prints one
`KNN_PHASE_TIMERS distance_ms ... select_ms ... merge_ms ...` line per
request to file descriptor 1 from inside the native call. The harness
redirects fd 1 into a scratch file around every request and parses that
line, so every timed sample carries the request's distance / select /
merge milliseconds and every arm's select_ms is READ, not inferred from
request deltas. The synchronizations serialize the launch queue, so on
such a build the request medians are slower than an untimed request and
are NOT comparable to the qualified numbers; the JSON says so
(`phase_timers.serialized`). `auto` (default) records the lines if the
build prints them, `require` fails when it does not, `off` never
redirects.

FIXTURES (all from `--seed`, all recorded by sha256 in the JSON)
----------------------------------------------------------------
  large          400,000 x 32 index, 4,000 x 32 queries, HASHED non-uniform
                 values (splitmix64 over (row, feature, seed), log-scaled,
                 per-feature scale, clustered offsets); every index row is
                 distinct (checked); 16 queries are exact copies of index
                 rows at partition-boundary positions. Timed at k10/k15.
  dyadic         the opponent's fixture, `tools/knn_cuml_reference.py::
                 coordinate_block` (dyadic-v1, salts 0/593), same shape.
                 Timed at k10/k15; this is the only fixture whose cached
                 cuML row may be quoted, and only as a cached-reference
                 ratio (`--cached-opponent`), never as a paired opponent.
  ties           131,079 x 32 (= 2 x 65,536 + 7: the remainder is shorter
                 than k, so the index-axis loop carves a k-wide last
                 partition), 700 queries (two query batches, the second
                 ragged). 40 anchor rows are each planted at three
                 positions (one straddling 65,535/65,536, one inside the
                 carved tail); 40 queries equal an anchor (exact match,
                 three-way zero-distance tie), 40 sit at a small offset
                 (three-way nonzero tie). Complete float64 oracle.
  divergent_tail 397,156 x 32 (= 6 x 65,536 + 3,940): the last partition's
                 length makes the CURRENT eight-deep unrolled scan loop
                 run a different trip count in different threads of one
                 block (see the brief), which is exactly the shape a
                 warp-collective inside that loop would hang or garble
                 on. 600 queries. Sampled oracle.

`--quick` shrinks every fixture (a smoke on any GPU; the JSON says
`quick: true` and nothing from it is a gate result).

Only existing public APIs are used: `NearestNeighbors(n_neighbors=k)`,
`.fit(X)`, `.kneighbors(Q)`; the returned `mojolearn.Array` objects are
read through their `__array_interface__`.
"""
import argparse
import ctypes
import hashlib
import importlib.util
import json
import os
import platform
import signal
import statistics
import subprocess
import sys
import tempfile
import threading
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)

MASK64 = np.uint64(0xFFFFFFFFFFFFFFFF)


class Deadline(Exception):
    pass


class GateFailure(Exception):
    pass


# ----------------------------------------------------------------- hashing


def mix64(x):
    """splitmix64 finalizer over a uint64 array; wraps mod 2**64."""
    x = np.asarray(x, dtype=np.uint64)
    with np.errstate(over="ignore"):
        z = (x + np.uint64(0x9E3779B97F4A7C15)) & MASK64
        z = ((z ^ (z >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)) & MASK64
        z = ((z ^ (z >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)) & MASK64
        z = z ^ (z >> np.uint64(31))
    return z


def unit(h):
    """uint64 hash -> float64 in [0, 1) with 53 random mantissa bits."""
    return (h >> np.uint64(11)).astype(np.float64) * (1.0 / 9007199254740992.0)


def hashed_block(n_rows, n_features, seed, salt):
    """Non-uniform float32 rows: every cell a fresh 53-bit hash, log-scaled
    magnitudes, a per-feature scale over four octaves, and one of twelve
    cluster offsets per row, so no two cells share a value lattice and no
    two rows share a magnitude profile. Nothing here is dyadic."""
    rows = np.arange(n_rows, dtype=np.uint64).reshape(-1, 1)
    feats = np.arange(n_features, dtype=np.uint64).reshape(1, -1)
    with np.errstate(over="ignore"):
        base = (np.uint64(seed) * np.uint64(0x2545F4914F6CDD1D) + np.uint64(salt)) & MASK64
        cell = (rows * np.uint64(0x9E3779B97F4A7C15) + feats * np.uint64(0xD1B54A32D192ED03) + base) & MASK64
    h1 = mix64(cell)
    h2 = mix64(h1 ^ np.uint64(0xA5A5A5A5A5A5A5A5))
    u1 = unit(h1)
    u2 = unit(h2)
    sign = np.where((h2 & np.uint64(1)) == 0, 1.0, -1.0)
    magnitude = np.exp(2.5 * (u1 - 0.5))  # log-uniform over about e**-1.25 .. e**1.25
    feature_scale = np.ldexp(1.0, (mix64(feats + np.uint64(salt) * np.uint64(7)) % np.uint64(7)).astype(np.int64) - 3)
    cluster = (mix64(rows + np.uint64(salt) * np.uint64(13)) % np.uint64(12)).astype(np.int64)
    offsets = unit(mix64(np.arange(12 * n_features, dtype=np.uint64) + np.uint64(salt) * np.uint64(17))).reshape(12, n_features) * 8.0 - 4.0
    values = sign * magnitude * feature_scale + offsets[cluster[:, 0]] + 0.37 * (u2 - 0.5)
    return np.ascontiguousarray(values.astype(np.float32))


def rows_distinct(x):
    v = np.ascontiguousarray(x).view(np.dtype((np.void, x.dtype.itemsize * x.shape[1])))
    return np.unique(v).shape[0] == x.shape[0]


def sha256_bytes(*arrays):
    h = hashlib.sha256()
    for a in arrays:
        h.update(np.ascontiguousarray(a).tobytes())
    return h.hexdigest()


def load_dyadic_generator():
    """`coordinate_block` from tools/knn_cuml_reference.py, imported from the
    file so the dyadic-v1 bytes cannot drift from the opponent script's."""
    path = os.path.join(HERE, "knn_cuml_reference.py")
    spec = importlib.util.spec_from_file_location("knn_cuml_reference", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.coordinate_block


# ---------------------------------------------------------------- fixtures


def make_large(seed, quick):
    n_index = 40_000 if quick else 400_000
    n_queries = 400 if quick else 4_000
    d = 32
    index = hashed_block(n_index, d, seed, 1)
    queries = hashed_block(n_queries, d, seed, 2)
    # Exact matches at boundary positions: first/last column of the first
    # partition, both sides of 65,535/65,536, deep interior, and the last
    # rows of the ragged final partition.
    positions = [0, 1, 255, 256, 2047, 2048, 65535, 65536, 65537, 131071, 131072,
                 200000, 393215, 393216, n_index - 2, n_index - 1]
    positions = [p % n_index for p in positions]
    planted = []
    for j, p in enumerate(positions):
        q = 100 + j if n_queries > 116 else j
        queries[q] = index[p]
        planted.append((q, p))
    return {
        "name": "large", "index": index, "queries": queries,
        "n_index": n_index, "n_queries": n_queries, "d": d,
        "exact": planted, "dup_groups": [], "offset_groups": [],
        "oracle": "sampled", "timed": True,
        "description": "hashed non-uniform 400k x 32 / 4k x 32 with 16 exact-match queries",
    }


def make_dyadic(quick):
    coordinate_block = load_dyadic_generator()
    n_index = 40_000 if quick else 400_000
    n_queries = 400 if quick else 4_000
    d = 32
    index = np.ascontiguousarray(coordinate_block(n_index, d, 0))
    queries = np.ascontiguousarray(coordinate_block(n_queries, d, 593))
    return {
        "name": "dyadic", "index": index, "queries": queries,
        "n_index": n_index, "n_queries": n_queries, "d": d,
        "exact": [], "dup_groups": [], "offset_groups": [],
        "oracle": "sampled", "timed": True,
        "description": "dyadic-v1, the cached cuML opponent fixture (salts 0/593)",
    }


def make_ties(seed, quick, k_max):
    part = 65_536
    n_index = (2 * part + 7) if not quick else (4_096 + 7)
    n_queries = 700 if not quick else 140
    d = 32
    index = hashed_block(n_index, d, seed, 3)
    queries = hashed_block(n_queries, d, seed, 4)
    assert (n_index % part) < k_max or quick, "the tie fixture must exercise the carved last partition"
    n_anchor = 40 if not quick else 8
    rng = np.random.default_rng(seed)
    # A pool of distinct positions: boundary straddlers, the carved tail,
    # then random interior ones. Every anchor gets three.
    tail = list(range(n_index - 7, n_index))
    special = [part - 2, part - 1, part, part + 1, 0, 1, 2 * part - 1, 2 * part] if not quick else [0, 1, 2]
    special = [p for p in special if p < n_index]
    pool = list(dict.fromkeys(special + tail))
    used = set(pool)
    while len(pool) < 3 * n_anchor:
        p = int(rng.integers(0, n_index))
        if p not in used:
            used.add(p)
            pool.append(p)
    rng.shuffle(pool)
    dup_groups = []
    exact = []
    offset_groups = []
    for a in range(n_anchor):
        positions = sorted(pool[3 * a: 3 * a + 3])
        anchor = index[positions[0]].copy()
        for p in positions[1:]:
            index[p] = anchor
        dup_groups.append(positions)
        # exact-match query
        qe = a
        queries[qe] = anchor
        exact.append((qe, positions))
        # offset query: same nonzero distance to all three copies
        qo = n_anchor + a
        delta = (unit(mix64(np.arange(d, dtype=np.uint64) + np.uint64(1000 + a))) - 0.5).astype(np.float32) * np.float32(0.05)
        queries[qo] = anchor + delta
        offset_groups.append((qo, positions))
    return {
        "name": "ties", "index": index, "queries": queries,
        "n_index": n_index, "n_queries": n_queries, "d": d,
        "exact": exact, "dup_groups": dup_groups, "offset_groups": offset_groups,
        "oracle": "complete", "timed": False,
        "description": "carved last partition (2 x 65536 + 7), planted three-way exact and offset ties",
    }


def make_divergent_tail(seed, quick):
    part = 65_536
    n_index = (6 * part + 3_940) if not quick else (2_048 + 3_940)
    n_queries = 600 if not quick else 120
    d = 32
    index = hashed_block(n_index, d, seed, 5)
    queries = hashed_block(n_queries, d, seed, 6)
    return {
        "name": "divergent_tail", "index": index, "queries": queries,
        "n_index": n_index, "n_queries": n_queries, "d": d,
        "exact": [], "dup_groups": [], "offset_groups": [],
        "oracle": "sampled", "timed": False,
        "description": "last partition 3,940 columns: per-thread trip-count divergence in the unrolled scan",
    }


# ------------------------------------------------------------------ oracle


def oracle_topk(index, queries, k, rows):
    """float64 (distance, index)-ordered top-k for the given query rows.
    Returns (idx [len(rows), k], d64 sorted [len(rows), k], gap [len(rows)]),
    where gap is the float64 distance between rank k-1 and rank k."""
    xi = index.astype(np.float64)
    xq = queries[rows].astype(np.float64)
    ni = xi.shape[0]
    yn = np.einsum("ij,ij->i", xi, xi)
    out_idx = np.empty((len(rows), k), dtype=np.int64)
    out_d = np.empty((len(rows), k), dtype=np.float64)
    gap = np.empty(len(rows), dtype=np.float64)
    step = max(1, min(len(rows), int(5.0e7 // max(ni, 1))))  # about 400 MB of float64 per chunk
    for s in range(0, len(rows), step):
        q = xq[s: s + step]
        qn = np.einsum("ij,ij->i", q, q)
        d2 = qn[:, None] + yn[None, :] - 2.0 * (q @ xi.T)
        np.maximum(d2, 0.0, out=d2)
        for r in range(d2.shape[0]):
            row = d2[r]
            part = np.argpartition(row, min(k, ni - 1))[: k + 1] if ni > k else np.arange(ni)
            order = part[np.lexsort((part, row[part]))]
            out_idx[s + r] = order[:k]
            out_d[s + r] = np.sqrt(row[order[:k]])
            gap[s + r] = (np.sqrt(row[order[k]]) - out_d[s + r, -1]) if len(order) > k else np.inf
    return out_idx, out_d, gap


# ------------------------------------------------------------ the request


PHASE_LINE = "KNN_PHASE_TIMERS"
ADMIT_LINE = "KNN_ADMIT_RATE"
# Keys medianed per arm from the per-request phase record: the three phase
# milliseconds, and the admission counts the `votecount` arm's launcher
# prints (summed over a request's launches; `admit_rate` = any_admit /
# warp_steps).
PHASE_KEYS = ("distance_ms", "select_ms", "merge_ms", "admit_warp_steps", "admit_any_admit", "admit_rate")
# `--timing-only-arms` entries whose output is VALID (equality with the
# reference is asserted instead of difference): counter arms whose time is
# not a price because their launcher synchronizes per launch.
TIMING_ONLY_VALID_OUTPUT = ("votecount",)


class PhaseCapture:
    """Redirect file descriptor 1 into a scratch file around one native
    call and parse the `KNN_PHASE_TIMERS distance_ms ...` line the binding
    prints when built with -D MOJOLEARN_KNN_PHASE_TIMERS=1. The redirect is
    at the descriptor level because the print happens inside the extension,
    not through `sys.stdout`; C stdio is flushed before the descriptor is
    restored in case the runtime buffers. Disabled, it does nothing."""

    def __init__(self, enabled):
        self.enabled = enabled
        self.scratch = tempfile.TemporaryFile(mode="w+b") if enabled else None
        self.saved = None
        try:
            self.libc = ctypes.CDLL(None) if enabled else None
        except OSError:
            self.libc = None

    def begin(self):
        if not self.enabled:
            return
        sys.stdout.flush()
        sys.stderr.flush()
        self.scratch.seek(0)
        self.scratch.truncate(0)
        self.saved = os.dup(1)
        os.dup2(self.scratch.fileno(), 1)

    def end(self):
        """Restore fd 1; return the parsed phase dict (None when no line)."""
        if not self.enabled:
            return None
        if self.libc is not None:
            try:
                self.libc.fflush(None)
            except Exception:  # noqa: BLE001
                pass
        os.dup2(self.saved, 1)
        os.close(self.saved)
        self.saved = None
        self.scratch.seek(0)
        text = self.scratch.read().decode("utf-8", "replace")
        phases = None
        admit_steps = 0
        admit_any = 0
        admit_lines = 0
        for line in text.splitlines():
            words = line.split()
            if words[:2] == [PHASE_LINE, "distance_ms"]:
                rec = {}
                for i in range(1, len(words) - 1, 2):
                    try:
                        rec[words[i]] = float(words[i + 1])
                    except ValueError:
                        rec[words[i]] = words[i + 1]
                phases = rec  # the last line of the request wins
            elif words[:2] == [ADMIT_LINE, "warp_steps"]:
                # One line per selector launch (DEVIATION 2522, the
                # `votecount` arm); a request is the sum of its launches.
                fields = {}
                for i in range(1, len(words) - 1, 2):
                    try:
                        fields[words[i]] = int(words[i + 1])
                    except ValueError:
                        pass
                admit_steps += fields.get("warp_steps", 0)
                admit_any += fields.get("any_admit", 0)
                admit_lines += 1
        if admit_lines:
            if phases is None:
                phases = {}
            phases["admit_warp_steps"] = float(admit_steps)
            phases["admit_any_admit"] = float(admit_any)
            phases["admit_rate"] = (float(admit_any) / float(admit_steps)) if admit_steps else float("nan")
            phases["admit_lines"] = float(admit_lines)
        return phases


class Runner:
    def __init__(self, arm_env, sabotage_env, log, capture=None):
        self.arm_env = arm_env
        self.sabotage_env = sabotage_env
        self.log = log
        self.calls = 0
        self.capture = capture or PhaseCapture(False)
        self.last_phases = None
        self.phase_lines_seen = 0

    def set_arm(self, arm, sabotage):
        if arm is None:
            os.environ.pop(self.arm_env, None)
        else:
            os.environ[self.arm_env] = arm
        if sabotage:
            os.environ[self.sabotage_env] = "1"
        else:
            os.environ.pop(self.sabotage_env, None)

    def search(self, model, queries, arm, sabotage=False):
        """One public request; returns (elapsed_ns, dist float32 [n, k], idx int64 [n, k]).
        The request's phase-timer line, if the build prints one, is left in
        `self.last_phases` (the redirect happens outside the timed region)."""
        self.set_arm(arm, sabotage)
        self.capture.begin()
        try:
            t0 = time.perf_counter_ns()
            dist, idx = model.kneighbors(queries)
            t1 = time.perf_counter_ns()
        finally:
            self.last_phases = self.capture.end()
        if self.last_phases is not None:
            self.phase_lines_seen += 1
        self.calls += 1
        d = np.array(np.asarray(dist), dtype=np.float32, copy=True)
        i = np.array(np.asarray(idx), dtype=np.int64, copy=True)
        return t1 - t0, d, i


def bits_equal(a, b):
    da, ia = a
    db, ib = b
    return (da.shape == db.shape and ia.shape == ib.shape
            and np.array_equal(da.view(np.uint32), db.view(np.uint32))
            and np.array_equal(ia, ib))


def count_diff(a, b):
    da, ia = a
    db, ib = b
    if da.shape != db.shape:
        return int(da.size + db.size)
    return int(np.count_nonzero(da.view(np.uint32) != db.view(np.uint32)) + np.count_nonzero(ia != ib))


# ------------------------------------------------------------------ checks


def check_row_order(dist, idx, n_index):
    """The pinned contract of every output row."""
    problems = []
    if idx.min() < 0 or idx.max() >= n_index:
        problems.append("index out of range")
    if not np.all(np.diff(dist.astype(np.float64), axis=1) >= 0):
        problems.append("distance not non-decreasing")
    eq = dist[:, 1:] == dist[:, :-1]
    if np.any(eq & ~(idx[:, 1:] > idx[:, :-1])):
        problems.append("equal distances not ascending by index")
    srt = np.sort(idx, axis=1)
    if np.any(srt[:, 1:] == srt[:, :-1]):
        problems.append("repeated index in a row")
    if np.any(np.isnan(dist)):
        problems.append("NaN distance")
    return problems


def check_planted(fx, dist, idx, k):
    """Planted expectations. Exact matches are asserted by INDEX: the pinned
    expanded form `||q||^2 + ||y||^2 - 2 q.y` sums the norm through a block
    tree (`core/row_norms.mojo`) and the product through a serial chain
    (`pinned_distance_tile.mojo`), so an exact match need not come out as
    distance bits 0 on non-dyadic data; the number that did is reported as
    `exact_zero_bits`, not asserted. Duplicate-row groups are asserted by
    equal bits (same bytes, same chain) and ascending index."""
    problems = []
    dbits = dist.view(np.uint32)
    zero = 0
    for q, p in fx["exact"]:
        positions = p if isinstance(p, list) else [p]
        n = min(len(positions), k)
        for r, pos in enumerate(positions[:n]):
            if idx[q, r] != pos:
                problems.append(f"exact query {q}: rank {r} index {idx[q, r]}, want {pos}")
        if len(set(dbits[q, :n].tolist())) != 1:
            problems.append(f"exact query {q}: group distance bits differ {[hex(b) for b in dbits[q, :n]]}")
        if dbits[q, 0] == 0:
            zero += 1
    fx["exact_zero_bits"] = zero
    for q, positions in fx["offset_groups"]:
        n = min(len(positions), k)
        got = idx[q, :n].tolist()
        if got != positions[:n]:
            problems.append(f"offset query {q}: first {n} indices {got}, want {positions[:n]}")
        if len(set(dbits[q, :n].tolist())) != 1:
            problems.append(f"offset query {q}: tie group distance bits differ {[hex(b) for b in dbits[q, :n]]}")
        if dbits[q, 0] == 0:
            problems.append(f"offset query {q}: expected a nonzero tie distance")
    return problems


def check_oracle(fx, dist, idx, k, rows, log):
    o_idx, o_d, gap = oracle_topk(fx["index"], fx["queries"], k, rows)
    hard = []
    ambiguous = 0
    dist_bad = []
    for r, q in enumerate(rows):
        ours = idx[q]
        if not np.array_equal(ours, o_idx[r]):
            # A float64 oracle cannot separate two candidates the pinned
            # float32 chain sees as one distance; those disagreements are
            # counted as ambiguous. Anything else is a wrong answer.
            scale = max(1.0, float(o_d[r, -1]))
            tol = 1e-5 * scale
            if set(ours.tolist()) != set(o_idx[r].tolist()):
                if gap[r] <= tol:
                    ambiguous += 1
                else:
                    hard.append(int(q))
            else:
                # same set, different order: ambiguous only where the
                # reordered neighbors' float64 distances coincide
                pos = {int(v): j for j, v in enumerate(o_idx[r])}
                worst = max(abs(float(o_d[r, pos[int(v)]]) - float(o_d[r, j])) for j, v in enumerate(ours))
                if worst <= tol:
                    ambiguous += 1
                else:
                    hard.append(int(q))
        # distances: ours (float32 euclid) vs sqrt of the float64 distance at OUR index
        d64 = np.sqrt(np.maximum(((fx["queries"][q].astype(np.float64) - fx["index"][ours].astype(np.float64)) ** 2).sum(axis=1), 0.0))
        # The kernel's distance is the float32 Gram form |q|^2 + |y|^2 - 2 q.y;
        # at a true distance of 0 (an exact match) the cancellation leaves a
        # residual of order sqrt(eps32 * |x|^2), about 1e-3 for these fixtures,
        # so the floor is that residual, not 1e-3 absolute (first H100 run,
        # 2026-09-11: every "mismatch" row was a planted exact match, sets and
        # order agreed). The exact-match rows' bits are reported by
        # check_planted; this check keeps to the rows the oracle can separate.
        norms = np.maximum(np.einsum("ij,ij->i", fx["index"][ours].astype(np.float64), fx["index"][ours].astype(np.float64)),
                           float(np.dot(fx["queries"][q].astype(np.float64), fx["queries"][q].astype(np.float64))))
        cancellation = 4.0 * np.sqrt(np.finfo(np.float32).eps * norms)
        tol = np.maximum(1e-3 * np.maximum(1.0, d64), cancellation)
        if np.any(np.abs(dist[q].astype(np.float64) - d64) > tol):
            dist_bad.append(int(q))
    return {"rows": len(rows), "hard_mismatch_rows": hard, "ambiguous_rows": ambiguous, "distance_mismatch_rows": dist_bad}


# ---------------------------------------------------------------- the gate


def median(xs):
    return statistics.median(xs) if xs else None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True, help="output directory (JSON, summary)")
    ap.add_argument("--arms", default="baseline,headbound", help="explicit arms, comma separated; every later arm is timed as a pair against the first")
    ap.add_argument("--timing-only-arms", default="", help="comma separated arms whose OUTPUT IS INVALID by construction (skiprank, skipscan, scanonly1, noshift), plus the counter arm votecount (valid output, time not a price); excluded from correctness, oracle and reach; each is timed against the first --arms arm and reported under `timing_only`")
    ap.add_argument("--phase-timers", choices=("auto", "require", "off"), default="auto", help="read the binding's KNN_PHASE_TIMERS line per request (needs a build with -D MOJOLEARN_KNN_PHASE_TIMERS=1): auto records it when present, require fails without it, off never redirects fd 1")
    ap.add_argument("--ks", default="10,15")
    ap.add_argument("--pairs", type=int, default=3, help="timed pairs per order (3 = the protocol's initial count; 5 where affordable)")
    ap.add_argument("--deadline", type=int, default=300, help="hard process deadline in seconds")
    ap.add_argument("--seed", type=int, default=20260910)
    ap.add_argument("--arm-env", default="MOJOLEARN_KNN_SELECT")
    ap.add_argument("--sabotage-env", default="MOJOLEARN_KNN_SELECT_SABOTAGE")
    ap.add_argument("--cached-opponent", default="", help="k10=ms,k15=ms from bench/OPPONENT_REFERENCE.md (dyadic-v1 tuple only); reported as a cached-reference ratio")
    ap.add_argument("--fixtures", default="large,dyadic,ties,divergent_tail")
    ap.add_argument("--time-fixtures", default="dyadic,large")
    ap.add_argument("--skip-timing", action="store_true")
    ap.add_argument("--skip-reach", action="store_true", help="only for a build known to lack the trial hook; the JSON records reach as NOT PROVEN")
    ap.add_argument("--oracle-sample", type=int, default=48)
    ap.add_argument("--quick", action="store_true", help="tiny shapes; a smoke, never a gate result")
    ap.add_argument("--selftest", action="store_true", help="no GPU: exercise the checker against a numpy stand-in search")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    json_path = os.path.join(args.out, "knn_selection_gate.json")
    summary_path = os.path.join(args.out, "summary.txt")
    log_lines = []

    def log(msg):
        line = f"[{time.strftime('%H:%M:%S')}] {msg}"
        print(line, flush=True)
        log_lines.append(line)

    report = {
        "deviation": 2496, "tool": "tools/knn_selection_gate.py", "status": "running",
        "quick": bool(args.quick), "selftest": bool(args.selftest),
        "seed": args.seed, "arm_env": args.arm_env, "sabotage_env": args.sabotage_env,
        "arms": args.arms.split(","), "ks": [int(k) for k in args.ks.split(",")],
        "timing_only_arms": [a for a in args.timing_only_arms.split(",") if a],
        "timing_only_note": "output invalid; phase cost only. These arms skip one phase of the selector kernel on purpose; never a correctness, oracle, reach or promotion input.",
        "phase_timers": {"mode": args.phase_timers, "available": None, "serialized": None,
                         "note": "from the binding's KNN_PHASE_TIMERS line (build define MOJOLEARN_KNN_PHASE_TIMERS); when available the request medians are serialized by per-class synchronizations and are not comparable to untimed requests; the split is the measurement"},
        "pairs_per_order": args.pairs, "deadline_s": args.deadline,
        "python": sys.version, "platform": platform.platform(), "numpy": np.__version__,
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "environment": {k: v for k, v in os.environ.items() if k.startswith("MOJOLEARN_")},
        "fixtures": {}, "correctness": [], "reach": [], "timing": [], "timing_only": [], "failures": [], "log": log_lines,
    }
    try:
        report["git_commit"] = subprocess.run(["git", "-C", REPO, "rev-parse", "HEAD"], capture_output=True, text=True, timeout=10).stdout.strip() or "unknown"
    except Exception as exc:  # noqa: BLE001
        report["git_commit"] = f"unknown ({exc!r})"
    try:
        report["nvidia_smi"] = subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version,uuid", "--format=csv,noheader"], capture_output=True, text=True, timeout=20).stdout.strip()
    except Exception as exc:  # noqa: BLE001
        report["nvidia_smi"] = f"unavailable ({exc!r})"

    def dump(status):
        report["status"] = status
        report["finished_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        with open(json_path, "w") as f:
            json.dump(report, f, indent=1, default=str)
        with open(summary_path, "w") as f:
            f.write(f"status: {status}\n")
            if report["phase_timers"].get("available"):
                f.write("phase timers: READ from the binding (build with MOJOLEARN_KNN_PHASE_TIMERS); request medians below are SERIALIZED by the per-class synchronizations and are not comparable to untimed requests\n")
            for row in report["timing"]:
                f.write(f"timing {row['fixture']} k{row['k']} {row['arm_b']} vs {row['arm_a']}: " + ", ".join(
                    f"{arm} median {row['median_ms'][arm]:.6f} ms (min {row['min_ms'][arm]:.6f})" for arm in row["median_ms"]) + "\n")
                if row.get("phase_ms_median"):
                    f.write("  phases (median ms): " + "; ".join(
                        f"{arm} " + " ".join(f"{key} {val:.3f}" for key, val in ph.items())
                        for arm, ph in row["phase_ms_median"].items()) + "\n")
                if row.get("cached_opponent_ratio"):
                    f.write("  cached-reference ratios (NOT a paired opponent): " + ", ".join(
                        f"{arm} {r:.3f}x" for arm, r in row["cached_opponent_ratio"].items()) + "\n")
            if report["timing_only"]:
                f.write("timing_only: OUTPUT INVALID; PHASE COST ONLY (never a correctness, reach or promotion input)\n")
            for row in report["timing_only"]:
                if row.get("output_valid", {}).get(row["arm_b"]):
                    tail = f"; {row['arm_b']} output VALID (equal to the reference); its time is not a price"
                    label = "output valid; counter arm; time not a price"
                else:
                    tail = f"; {row['arm_b']} output differs from the reference in {row['differing_cells_from_reference']} cells"
                    label = "output invalid; phase cost only"
                f.write(f"timing_only {row['fixture']} k{row['k']} {row['arm_b']} vs {row['arm_a']} ({label}): " + ", ".join(
                    f"{arm} median {row['median_ms'][arm]:.6f} ms (min {row['min_ms'][arm]:.6f})" for arm in row["median_ms"])
                    + tail + "\n")
                if row.get("phase_ms_median"):
                    f.write("  phases (median ms): " + "; ".join(
                        f"{arm} " + " ".join(f"{key} {val:.3f}" for key, val in ph.items())
                        for arm, ph in row["phase_ms_median"].items()) + "\n")
                if row.get("select_ms_ratio_b_over_a") is not None:
                    f.write(f"  select_ms {row['arm_b']}/{row['arm_a']} = {row['select_ms_ratio_b_over_a']:.4f} (paired median)\n")
            for failure in report["failures"]:
                f.write(f"FAIL {failure}\n")

    # THE HARD DEADLINE: SIGALRM raises inside Python; a daemon timer is the
    # backstop if a native call is holding the interpreter when it fires.
    def on_alarm(signum, frame):
        raise Deadline()

    def hard_exit():
        log("hard deadline backstop fired; writing partial report")
        report["failures"].append("hard deadline backstop fired")
        dump("deadline")
        os._exit(3)

    if hasattr(signal, "SIGALRM"):
        signal.signal(signal.SIGALRM, on_alarm)
        signal.alarm(args.deadline)
    backstop = threading.Timer(args.deadline + 15, hard_exit)
    backstop.daemon = True
    backstop.start()

    try:
        run_gate(args, report, log, dump)
    except Deadline:
        report["failures"].append(f"process deadline of {args.deadline} s exceeded")
        dump("deadline")
        log("DEADLINE")
        return 3
    except GateFailure as exc:
        report["failures"].append(str(exc))
        dump("failed")
        log(f"FAILED: {exc}")
        return 1
    finally:
        backstop.cancel()
    if report["failures"]:
        dump("failed")
        log("FAILED (see failures)")
        return 1
    dump("passed")
    log("PASSED")
    return 0


def run_gate(args, report, log, dump):
    ks = [int(k) for k in args.ks.split(",")]
    arms = [a for a in args.arms.split(",") if a]
    if len(arms) < 1:
        raise GateFailure("--arms needs at least one arm name")
    timing_only = [a for a in args.timing_only_arms.split(",") if a]
    overlap = sorted(set(arms) & set(timing_only))
    if overlap:
        raise GateFailure(f"--timing-only-arms {overlap} also in --arms; a timing-only arm never enters a correctness section")

    # ---- the library -------------------------------------------------
    if args.selftest:
        model_factory, describe = selftest_backend(log)
    else:
        os.environ.setdefault("MOJOLEARN_NUMERIC_MODE", "identical")
        import mojolearn  # noqa: PLC0415
        from mojolearn.neighbors import NearestNeighbors  # noqa: PLC0415

        def model_factory(k):
            return NearestNeighbors(n_neighbors=k)

        probe = NearestNeighbors(n_neighbors=1)
        info = {"mojolearn_version": getattr(mojolearn, "__version__", "unknown")}
        for name in ("numeric_mode_used", "vendor_used"):
            fn = getattr(probe, name, None)
            try:
                info[name] = fn() if callable(fn) else None
            except Exception as exc:  # noqa: BLE001
                info[name] = f"unavailable ({exc!r})"
        try:
            module = probe._bind("_mojolearn")
            so = getattr(module, "__file__", None)
            info["binding_path"] = so
            if so and os.path.exists(so):
                with open(so, "rb") as f:
                    info["binding_sha256"] = hashlib.sha256(f.read()).hexdigest()
        except Exception as exc:  # noqa: BLE001
            info["binding_path"] = f"unavailable ({exc!r})"
        describe = info
    report["library"] = describe
    if not args.selftest and describe.get("numeric_mode_used") not in (None, "identical"):
        raise GateFailure(f"numeric mode is {describe.get('numeric_mode_used')!r}; this gate runs IDENTICAL only")
    log(f"library: {describe}")

    runner = Runner(args.arm_env, args.sabotage_env, log, PhaseCapture(args.phase_timers != "off"))

    # ---- fixtures ----------------------------------------------------
    wanted = args.fixtures.split(",")
    fixtures = []
    t0 = time.perf_counter()
    if "large" in wanted:
        fixtures.append(make_large(args.seed, args.quick))
    if "dyadic" in wanted:
        fixtures.append(make_dyadic(args.quick))
    if "ties" in wanted:
        fixtures.append(make_ties(args.seed, args.quick, max(ks)))
    if "divergent_tail" in wanted:
        fixtures.append(make_divergent_tail(args.seed, args.quick))
    for fx in fixtures:
        fx["sha256_index"] = sha256_bytes(fx["index"])
        fx["sha256_queries"] = sha256_bytes(fx["queries"])
        fx["distinct_rows"] = bool(rows_distinct(fx["index"])) if fx["name"] != "ties" else None
        report["fixtures"][fx["name"]] = {k: v for k, v in fx.items() if k not in ("index", "queries")}
        if fx["name"] in ("large", "divergent_tail") and not fx["distinct_rows"]:
            raise GateFailure(f"fixture {fx['name']}: index rows are not all distinct")
    log(f"fixtures built in {time.perf_counter() - t0:.1f} s: " + ", ".join(f"{fx['name']} {fx['n_index']}x{fx['d']}/{fx['n_queries']}" for fx in fixtures))

    # ---- correctness and reach ---------------------------------------
    references = {}
    for fx in fixtures:
        for k in ks:
            model = model_factory(k).fit(fx["index"])
            outputs = {}
            for arm in arms + [None]:
                label = arm or "default"
                ns, d, i = runner.search(model, fx["queries"], arm)
                outputs[label] = (d, i)
                log(f"{fx['name']} k{k} arm {label}: {ns / 1e6:.3f} ms (untimed correctness call)"
                    + (f"; phases {runner.last_phases}" if runner.last_phases else ""))
                if report["phase_timers"]["available"] is None:
                    # Decided on the first request of the process.
                    available = runner.last_phases is not None
                    report["phase_timers"]["available"] = available
                    report["phase_timers"]["serialized"] = available
                    if args.phase_timers == "require" and not available:
                        raise GateFailure("--phase-timers require: the binding printed no KNN_PHASE_TIMERS line (build it with -D MOJOLEARN_KNN_PHASE_TIMERS=1)")
                    log(f"phase timers {'READ from the binding' if available else 'not printed by this build'} (mode {args.phase_timers})")
            ref = outputs[arms[0]]
            references[(fx["name"], k)] = ref
            entry = {"fixture": fx["name"], "k": k, "arms": {}, "row_order": [], "planted": [], "oracle": None}
            for label, out in outputs.items():
                eq = bits_equal(ref, out)
                entry["arms"][label] = {"equal_to_" + arms[0]: bool(eq), "differing_cells": count_diff(ref, out)}
                if not eq:
                    report["failures"].append(f"{fx['name']} k{k}: arm {label} differs from {arms[0]} in {count_diff(ref, out)} cells")
            entry["row_order"] = check_row_order(ref[0], ref[1], fx["n_index"])
            for p in entry["row_order"]:
                report["failures"].append(f"{fx['name']} k{k}: row order: {p}")
            entry["planted"] = check_planted(fx, ref[0], ref[1], k)
            entry["exact_zero_bits"] = fx.get("exact_zero_bits")
            for p in entry["planted"]:
                report["failures"].append(f"{fx['name']} k{k}: planted: {p}")
            if fx["oracle"] == "complete":
                rows = np.arange(fx["n_queries"])
            else:
                rng = np.random.default_rng(args.seed + k)
                rows = np.unique(np.concatenate([
                    np.array([q for q, _ in fx["exact"]], dtype=np.int64),
                    rng.integers(0, fx["n_queries"], size=max(1, args.oracle_sample)),
                ]))
            entry["oracle"] = check_oracle(fx, ref[0], ref[1], k, rows, log)
            if entry["oracle"]["hard_mismatch_rows"]:
                report["failures"].append(f"{fx['name']} k{k}: oracle hard mismatch rows {entry['oracle']['hard_mismatch_rows'][:10]}")
            if entry["oracle"]["distance_mismatch_rows"]:
                report["failures"].append(f"{fx['name']} k{k}: oracle distance mismatch rows {entry['oracle']['distance_mismatch_rows'][:10]}")
            log(f"{fx['name']} k{k}: arms equal={all(v['equal_to_' + arms[0]] for v in entry['arms'].values())} order={entry['row_order'] or 'ok'} planted={entry['planted'] or 'ok'} oracle={entry['oracle']}")
            report["correctness"].append(entry)

            # REACH, per arm, including the default.
            if not args.skip_reach:
                for arm in arms + [None]:
                    label = arm or "default"
                    _, ds, i_s = runner.search(model, fx["queries"], arm, sabotage=True)
                    flipped = count_diff(outputs[label], (ds, i_s))
                    _, dc, ic = runner.search(model, fx["queries"], arm, sabotage=False)
                    restored = bits_equal(outputs[label], (dc, ic))
                    rec = {"fixture": fx["name"], "k": k, "arm": label, "sabotage_flipped_cells": flipped, "clean_restored": bool(restored)}
                    report["reach"].append(rec)
                    if flipped == 0:
                        report["failures"].append(f"{fx['name']} k{k}: REACH NOT PROVEN for arm {label}: sabotage left every cell unchanged (build lacks -D MOJOLEARN_KNN_SELECT_TRIAL=1, or the arm is not wired)")
                    if not restored:
                        report["failures"].append(f"{fx['name']} k{k}: arm {label} did not restore clean bits after sabotage (sticky state)")
                    log(f"{fx['name']} k{k} reach {label}: sabotage flipped {flipped} cells, restored={restored}")
            else:
                report["reach"].append({"fixture": fx["name"], "k": k, "arm": "all", "note": "reach NOT PROVEN (--skip-reach)"})
            del model
            dump("running")

    # ---- timing --------------------------------------------------------
    if args.skip_timing:
        return
    cached = {}
    for item in [s for s in args.cached_opponent.split(",") if s]:
        key, val = item.split("=")
        cached[int(key.lstrip("k"))] = float(val)
    timed_fixtures = [fx for fx in fixtures if fx["name"] in args.time_fixtures.split(",") and fx["timed"]]
    if len(arms) < 2:
        log("timing needs two arms; the arm pair is skipped")
    else:
        # EVERY later arm against the first (DEVIATION 2522): one row per
        # fixture, k and pair; the first arm is re-timed inside every pair
        # so each candidate's numbers sit beside a same-window control.
        a = arms[0]
        for b in arms[1:]:
            for fx in timed_fixtures:
                for k in ks:
                    model = model_factory(k).fit(fx["index"])
                    row = time_pair(args, runner, log, model, fx, k, references[(fx["name"], k)], a, b, invalid_b=False)
                    if fx["name"] == "dyadic" and k in cached and not args.quick:
                        row["cached_opponent_ms"] = cached[k]
                        row["cached_opponent_ratio"] = {arm: row["median_ms"][arm] / cached[k] for arm in (a, b)}
                        row["cached_opponent_note"] = "cached cuML row from bench/OPPONENT_REFERENCE.md (H100 80GB HBM3, driver 580.126.09, dyadic-v1); admissible only if this box matches that tuple; not a paired opponent measurement"
                        if report["phase_timers"].get("available"):
                            row["cached_opponent_note"] += "; this build's requests are serialized by the phase timers, so the ratio is inflated and not admissible"
                    report["timing"].append(row)
                    del model
                    dump("running")

    # ---- timing-only arms: OUTPUT INVALID; phase cost only ---------------
    # Each timing-only arm is paired with arms[0] under the same protocol.
    # Its output is never checked for correctness; the only assertion is
    # that it DIFFERS from the reference (the arm's own body ran), while the
    # baseline samples in the same pairs must still equal the reference.
    # The exception is TIMING_ONLY_VALID_OUTPUT (counter arms, DEVIATION
    # 2522): valid output, so equality is asserted; time not a price.
    for t_arm in timing_only:
        valid = t_arm in TIMING_ONLY_VALID_OUTPUT
        for fx in timed_fixtures:
            for k in ks:
                model = model_factory(k).fit(fx["index"])
                row = time_pair(args, runner, log, model, fx, k, references[(fx["name"], k)], arms[0], t_arm, invalid_b=not valid, counter_b=valid)
                row["note"] = "output valid; counter arm; its launcher synchronizes per launch so its time is not a price; the admit_rate in its phase medians is the measurement" if valid else "output invalid; phase cost only"
                report["timing_only"].append(row)
                del model
                dump("running")


def phase_medians(samples, arm):
    """Median of every phase key over an arm's samples; None when no sample
    carried a phase line."""
    recs = [ph for _, _, ph in samples[arm] if ph]
    if not recs:
        return None
    return {key: median([ph[key] for ph in recs if isinstance(ph.get(key), float)]) for key in PHASE_KEYS if any(isinstance(ph.get(key), float) for ph in recs)}


def time_pair(args, runner, log, model, fx, k, ref, a, b, invalid_b, counter_b=False):
    """One warmup per arm, `--pairs` pairs in order (a, b) then (b, a), every
    sample kept with its phase line when the build prints one. `invalid_b`
    marks a timing-only arm: its output must DIFFER from the reference
    instead of equaling it. `counter_b` marks a counter arm (valid output,
    equality asserted; when the build prints phase lines every one of its
    samples must also carry the admission counts, or the arm's launcher
    did not run)."""
    samples = {a: [], b: []}
    orders = []
    differing = None

    def check(arm, out, where):
        nonlocal differing
        if arm == b and invalid_b:
            n = count_diff(ref, out)
            if n == 0:
                raise GateFailure(f"{fx['name']} k{k}: timing-only arm {b} returned the reference bits at {where}; its body did not run (output invalid by design, so equality means the arm was not reached)")
            differing = n if differing is None else min(differing, n)
        elif not bits_equal(ref, out):
            raise GateFailure(f"{fx['name']} k{k}: {arm} output moved from the correctness reference at {where}")
        if arm == b and counter_b and runner.last_phases is not None and not isinstance(runner.last_phases.get("admit_rate"), float):
            raise GateFailure(f"{fx['name']} k{k}: counter arm {b} printed no {ADMIT_LINE} line at {where} on a build that prints phase lines; its launcher did not run")

    for arm in (a, b):
        ns, d, i = runner.search(model, fx["queries"], arm)
        log(f"{fx['name']} k{k} warmup {arm}: {ns / 1e6:.3f} ms" + (f"; phases {runner.last_phases}" if runner.last_phases else ""))
        check(arm, (d, i), "warmup")
    for order_id, order in enumerate([(a, b), (b, a)]):
        for pair in range(args.pairs):
            rec = {"order": order_id, "pair": pair, "ms": {}, "phases": {}}
            for arm in order:
                ns, d, i = runner.search(model, fx["queries"], arm)
                check(arm, (d, i), f"order {order_id}, pair {pair}")
                rec["ms"][arm] = ns / 1e6
                rec["phases"][arm] = runner.last_phases
                samples[arm].append((order_id, ns / 1e6, runner.last_phases))
            rec["ratio_b_over_a"] = rec["ms"][b] / rec["ms"][a]
            pa, pb = rec["phases"][a], rec["phases"][b]
            if pa and pb and isinstance(pa.get("select_ms"), float) and isinstance(pb.get("select_ms"), float) and pa["select_ms"] > 0:
                rec["select_ms_ratio_b_over_a"] = pb["select_ms"] / pa["select_ms"]
            orders.append(rec)
            log(f"{fx['name']} k{k} order{order_id} pair{pair}: {a} {rec['ms'][a]:.3f} ms, {b} {rec['ms'][b]:.3f} ms, {b}/{a} {rec['ratio_b_over_a']:.4f}"
                + (f", select_ms {pa['select_ms']:.3f} vs {pb['select_ms']:.3f}" if "select_ms_ratio_b_over_a" in rec else ""))
    row = {
        "fixture": fx["name"], "k": k, "n_index": fx["n_index"], "n_queries": fx["n_queries"], "d": fx["d"],
        "arm_a": a, "arm_b": b, "pairs": orders,
        "median_ms": {arm: median([ms for _, ms, _ in samples[arm]]) for arm in (a, b)},
        "min_ms": {arm: min(ms for _, ms, _ in samples[arm]) for arm in (a, b)},
        "median_ms_by_order": {arm: {str(o): median([ms for oo, ms, _ in samples[arm] if oo == o]) for o in (0, 1)} for arm in (a, b)},
        "paired_ratio_median_b_over_a": median([r["ratio_b_over_a"] for r in orders]),
        "pairs_favoring_b": sum(1 for r in orders if r["ratio_b_over_a"] < 1.0),
        "spread_pct": {arm: 100.0 * (max(ms for _, ms, _ in samples[arm]) - min(ms for _, ms, _ in samples[arm])) / median([ms for _, ms, _ in samples[arm]]) for arm in (a, b)},
        "boundary": "public NearestNeighbors.kneighbors: host in, host out, includes query conversion, upload, norms, transpose, distance, selection, merge, download, index widening to int64",
    }
    phases = {arm: phase_medians(samples, arm) for arm in (a, b)}
    if any(phases.values()):
        row["phase_ms_median"] = {arm: ph for arm, ph in phases.items() if ph}
        row["phase_ms_median_by_order"] = {
            arm: {str(o): {key: median([ph[key] for oo, _, ph in samples[arm] if oo == o and ph and isinstance(ph.get(key), float)]) for key in PHASE_KEYS
                           if any(oo == o and ph and isinstance(ph.get(key), float) for oo, _, ph in samples[arm])} for o in (0, 1)}
            for arm in (a, b)}
        ratios = [r["select_ms_ratio_b_over_a"] for r in orders if "select_ms_ratio_b_over_a" in r]
        row["select_ms_ratio_b_over_a"] = median(ratios) if ratios else None
        row["phase_note"] = "phase milliseconds from the binding's KNN_PHASE_TIMERS line (serializing); request ms above include those synchronizations"
    if invalid_b:
        row["output_valid"] = {a: True, b: False}
        row["differing_cells_from_reference"] = differing
    elif counter_b:
        row["output_valid"] = {a: True, b: True}
        row["differing_cells_from_reference"] = 0
        row["time_is_a_price"] = {a: True, b: False}
    msg = f"{fx['name']} k{k}: {a} median {row['median_ms'][a]:.3f} ms, {b} median {row['median_ms'][b]:.3f} ms, paired median {b}/{a} {row['paired_ratio_median_b_over_a']:.4f}, {row['pairs_favoring_b']}/{len(orders)} pairs favor {b}"
    if row.get("phase_ms_median"):
        msg += "; select_ms medians " + ", ".join(f"{arm} {ph['select_ms']:.3f}" for arm, ph in row["phase_ms_median"].items() if "select_ms" in ph)
        rates = {arm: ph["admit_rate"] for arm, ph in row["phase_ms_median"].items() if isinstance(ph.get("admit_rate"), float)}
        if rates:
            msg += "; admit_rate medians " + ", ".join(f"{arm} {r:.4f}" for arm, r in rates.items())
    if invalid_b:
        msg += f" [OUTPUT INVALID; phase cost only; {b} differs from the reference in {differing} cells]"
    elif counter_b:
        msg += f" [counter arm; output valid; {b} time is not a price]"
    log(msg)
    return row


# --------------------------------------------------------------- selftest


def selftest_backend(log):
    """A numpy stand-in for the native search, honoring the pinned tie rule,
    the arm/sabotage switches and the float32 output types, so the checker
    itself can be exercised without a GPU. NOT the library."""
    arm_env = "MOJOLEARN_KNN_SELECT"
    sab_env = "MOJOLEARN_KNN_SELECT_SABOTAGE"

    class Model:
        def __init__(self, k):
            self.k = k
            self.x = None

        def fit(self, x):
            self.x = np.asarray(x, dtype=np.float32)
            return self

        def kneighbors(self, q):
            arm = os.environ.get(arm_env)
            timing_only = ("skiprank", "skipscan", "scanonly1", "noshift")
            counter_arms = ("votecount",)
            if arm not in (None, "baseline", "uniform", "headbound", "warpbound", "deferred", "capk", "capk_selp", "voteguard") + timing_only + counter_arms:
                raise ValueError(f"unknown arm {arm!r}")
            if arm in timing_only + counter_arms and os.environ.get(sab_env) == "1":
                raise ValueError("timing-only arms carry no sabotage")
            xi = self.x.astype(np.float64)
            xq = np.asarray(q, dtype=np.float32).astype(np.float64)
            d2 = np.einsum("ij,ij->i", xq, xq)[:, None] + np.einsum("ij,ij->i", xi, xi)[None, :] - 2.0 * (xq @ xi.T)
            np.maximum(d2, 0.0, out=d2)
            d32 = np.sqrt(d2).astype(np.float32)
            idx = np.empty((xq.shape[0], self.k), dtype=np.int64)
            dist = np.empty((xq.shape[0], self.k), dtype=np.float32)
            for r in range(xq.shape[0]):
                order = np.lexsort((np.arange(xi.shape[0]), d32[r]))[: self.k]
                idx[r] = order
                dist[r] = d32[r, order]
            if os.environ.get(sab_env) == "1":
                idx[:, 0] ^= 1
                dist[:, 0] = d32[np.arange(xq.shape[0]), idx[:, 0]]
            if arm in timing_only:
                # Output invalid by construction, like the native arms:
                # sentinel index and value bits in every slot but the first.
                idx[:, 1:] = 0xFFFFFFFF
                dist[:, 1:] = np.float32(np.nan)
            # The phase line the instrumented binding prints to fd 1, so the
            # harness's descriptor-level capture is exercised here too.
            select = {"skiprank": 0.4, "skipscan": 0.6, "scanonly1": 0.3, "noshift": 0.5}.get(arm, 1.0)
            if arm in counter_arms:
                # The counter arm's launcher prints one line per launch;
                # two here so the harness's per-request sum is exercised
                # (3 of 8 warp-steps admit: admit_rate 0.375).
                os.write(1, b"KNN_ADMIT_RATE warp_steps 5 any_admit 2 rows 4 length 100 k 3\n")
                os.write(1, b"KNN_ADMIT_RATE warp_steps 3 any_admit 1 rows 4 length 100 k 3\n")
            os.write(1, (f"KNN_PHASE_TIMERS distance_ms 1.5 select_ms {select} merge_ms 0.05 "
                         f"distance_launches 1 select_launches 1 merge_launches 0 k {self.k}\n").encode())
            return dist, idx

    log("SELFTEST: numpy stand-in search, no GPU, no mojolearn import")
    return (lambda k: Model(k)), {"selftest": True}


if __name__ == "__main__":
    sys.exit(main())
