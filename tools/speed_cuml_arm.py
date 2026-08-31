#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The NVIDIA-native opponent for `bench/speed/classical_speed_main.mojo`.

    MOJOLEARN_SPEED_LANE=kmeans python3 tools/speed_cuml_arm.py
    MOJOLEARN_SPEED_DUMP=/tmp/speedfix MOJOLEARN_SPEED_LANE=kde \\
        python3 tools/speed_cuml_arm.py

ONE LANE PER PROCESS, selected by the SAME environment variable the Mojo
driver uses, for the same reason: most of these opponents are optional and
several of them will not import at all on a given RAPIDS build. A lane whose
opponent is missing prints `FSPEED-REFUSED` and exits zero. Nothing here may
take another lane down.

WHAT AN ARM LABEL MEANS, AND IT MEANS EXACTLY WHAT IT SAYS
==========================================================
`arm=` is the library and the device that actually ran:

    cuml-gpu          a RAPIDS cuML estimator on the GPU
    cuvs-gpu          a cuVS index on the GPU
    torch-gpu         PyTorch, which is cuSOLVER / cuBLAS underneath
    sklearn-cpu       scikit-learn on the host CPU
    scipy-cpu         SciPy on the host CPU
    statsmodels-cpu   statsmodels on the host CPU

An arm labeled `cuml-gpu` IS a cuml call. Nine of these lanes have NO RAPIDS
counterpart at any version -- there is no GPU Gaussian mixture, no GPU
Gaussian process, no GPU Nystroem, no GPU random Fourier features, no GPU
bootstrap and no GPU spectral clustering in RAPIDS -- and those arms fall
back to the strongest thing that genuinely runs on the box and SAY SO in the
label. `bench/speed/CLASSICAL_SPEED.md` lists them in one table so nobody has
to infer it from a log.

THE TWO SIDES RUN ON THE SAME BYTES
===================================
Two mechanisms, because the lanes divide into two kinds.

1. The five lanes that came from `bench/bench_main.mojo` (kmeans, dbscan,
   pca, ols, knn) generate their data from a splitmix64 recurrence.
   `bench/bench_sklearn.py` already holds the vectorized twin of it, proven
   bit-identical, so this file LOADS `u01` OUT OF THAT FILE rather than
   writing a third copy. The row-indexed variant this file does add
   (`u01_at`, for the k-means initial centroids, whose rows are `c * 7919`
   and not `0..n-1`) is CHECKED against the imported `u01` at startup on
   every run, so the two cannot drift apart silently.

2. Every other lane's fixture is a Mojo builder with no Python twin. Writing
   one would be writing a SECOND fixture that agrees today and drifts on the
   first edit. Instead the Mojo driver dumps its inputs:

       MOJOLEARN_SPEED_DUMP=/tmp/speedfix MOJOLEARN_SPEED_LANE=kde \\
           pixi run mojo run -I . bench/speed/classical_speed_main.mojo

   and this file reads `/tmp/speedfix/<lane>.fixture`. Floats travel as their
   BITS, never as decimals, because `String(Float32)` does not round trip in
   the Mojo toolchain and a decimal dump would hand this side a different
   dataset while looking correct. DUMP FIRST, RACE AFTER: a lane in group 2
   with no dump present prints `FSPEED-REFUSED` rather than inventing data.

WHAT IS TIMED
=============
The call plus the synchronization that proves it finished, and nothing else.
Data generation, host-to-device transfer, index construction where the lane's
unit is a search, and model construction all happen before the clock starts.
One untimed warm-up round runs first and is printed as `FSPEED-WARMUP`.
`_sync()` calls `cupy.cuda.runtime.deviceSynchronize()` or
`torch.cuda.synchronize()` -- whichever the arm is using -- because an
unsynchronized GPU timing measures how fast Python can enqueue.

THE HASH
========
FNV-1a64 over the output array's bytes, byte at a time, little endian: the
same recurrence `core/identity_trace.mojo::fnv1a64_bytes` uses, so a number
printed here means the same KIND of thing as a number printed by the Mojo
side. It is a WITHIN-ARM determinism probe and not a cross-arm equality
check: two different implementations of k-means will not produce the same
bits and are not supposed to. Where an output is not a float32/int32 array of
the same shape as ours, or where the two estimators genuinely compute
different objects (Nystroem's basis sample, spectral's label numbering), the
line carries `hash=-` rather than a number that means nothing.

Pure-Python FNV is about a microsecond a byte, so outputs above
`MOJOLEARN_SPEED_HASH_MAX` bytes (default 262144) report `hash=-` and one
`FSPEED-NOTE`. Hashing four megabytes of k-means labels five times would cost
more than the benchmark.

ACCURACY IS NOT MEASURED HERE
=============================
`tools/fast_speed_table.py` understands an `FSPEED-ACC` line. This file emits
none. A speed harness that also scores accuracy invites a reader to trade one
against the other in a single table, and the lanes that need an accuracy
statement have gates that make it properly.

Environment:
    MOJOLEARN_SPEED_LANE      required
    MOJOLEARN_SPEED_ROUNDS    timed rounds (default 5)
    MOJOLEARN_SPEED_SIZE      `shipped` (default) or `smoke`
    MOJOLEARN_SPEED_DUMP      directory holding <lane>.fixture
    MOJOLEARN_SPEED_HASH_MAX  bytes hashed before giving up (default 262144)
"""

import importlib.util
import os
import sys
import shutil
import time

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

FAMILY = "classical"

FNV_OFFSET = 0xCBF29CE484222325
FNV_PRIME = 0x100000001B3
MASK64 = (1 << 64) - 1


# ---------------------------------------------------------------------------
# `u01`, imported from the file that already owns it.
# ---------------------------------------------------------------------------

def _load_bench_sklearn():
    """`bench/bench_sklearn.py` as a module, by path.

    By path rather than by `import bench.bench_sklearn` so this works from any
    working directory and so there is no doubt about WHICH file was loaded:
    the one in this repository, beside this one. Its `main()` is behind an
    `if __name__ == "__main__"` guard, so importing it runs nothing.
    """
    path = os.path.join(REPO, "bench", "bench_sklearn.py")
    spec = importlib.util.spec_from_file_location("_mojolearn_bench_sklearn", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_BS = _load_bench_sklearn()
u01 = _BS.u01

M1 = 0x9E3779B97F4A7C15
M2 = 0xBF58476D1CE4E5B9
M3 = 0x94D049BB133111EB


def u01_at(rows, cols, salt):
    """`u01` for an EXPLICIT vector of row indices instead of `arange(n)`.

    The k-means initial centroids in `bench/bench_main.mojo` are rows
    `c * 7919` for `c` in `0..63`, which reach row 498,897. Materializing
    `u01(498898, 32, 5)` to take one row of it would allocate 128 MB per
    centroid, so this variant exists. It is the same recurrence, and
    `_check_u01_variant` asserts on every run that it agrees with the
    imported `u01` on the rows they share -- a re-spelling that is checked
    against its original every time is a re-spelling that cannot drift.
    """
    r = np.asarray(rows, dtype=np.uint64)[:, None]
    k = np.arange(cols, dtype=np.uint64)[None, :]
    z = (r * np.uint64(M1) + (k + np.uint64(1)) * np.uint64(M2)
         + np.uint64(salt + 1) * np.uint64(M3))
    z = (z ^ (z >> np.uint64(30))) * np.uint64(M2)
    z = (z ^ (z >> np.uint64(27))) * np.uint64(M3)
    z = z ^ (z >> np.uint64(31))
    return (z >> np.uint64(11)).astype(np.float64) * (1.0 / 9007199254740992.0)


def _check_u01_variant():
    a = u01(37, 11, 3)
    b = u01_at(np.arange(37), 11, 3)
    if not np.array_equal(a, b):
        raise SystemExit(
            "u01_at disagrees with bench/bench_sklearn.py::u01. The opponent "
            "would be racing on a different dataset. Fix u01_at before "
            "quoting any number from this run."
        )


# ---------------------------------------------------------------------------
# The hash.
# ---------------------------------------------------------------------------

HASH_MAX = int(os.environ.get("MOJOLEARN_SPEED_HASH_MAX", "262144"))


def hash_arrays(lane, arm, *arrays):
    """FNV-1a64 over the concatenated raw bytes, or `-` when that is a lie.

    `-` in three cases, each of which is a real statement rather than a
    shrug: an array we could not bring back to the host, an array bigger than
    `HASH_MAX` (a note is printed naming the size), and an explicit `None`
    passed by a lane that knows its output is not comparable with ours.
    """
    total = 0
    chunks = []
    for a in arrays:
        if a is None:
            return "-"
        try:
            h = to_numpy(a)
        except Exception:
            return "-"
        h = np.ascontiguousarray(h)
        chunks.append(h)
        total += h.nbytes
    if total > HASH_MAX:
        note(lane, arm, "output is %d bytes, above MOJOLEARN_SPEED_HASH_MAX "
                        "(%d): hash omitted, not computed and discarded"
             % (total, HASH_MAX))
        return "-"
    h = FNV_OFFSET
    for c in chunks:
        for b in memoryview(c.tobytes()).cast("B"):
            h = ((h ^ b) * FNV_PRIME) & MASK64
    return "%016x" % h


def to_numpy(a):
    """A host numpy view of a cupy / cudf / torch / numpy object."""
    if isinstance(a, np.ndarray):
        return a
    for attr in ("get", "to_numpy", "to_output"):
        f = getattr(a, attr, None)
        if callable(f):
            try:
                out = f()
                if isinstance(out, np.ndarray):
                    return out
                return np.asarray(out)
            except Exception:
                pass
    cpu = getattr(a, "cpu", None)
    if callable(cpu):
        return cpu().detach().numpy()
    return np.asarray(a)


# ---------------------------------------------------------------------------
# The output contract.
# ---------------------------------------------------------------------------

def header(lane, arm, device, rounds, size):
    print("FSPEED-HEADER family=%s lane=%s arm=%s mode=FAST device=%s "
          "rounds=%d size=%s" % (FAMILY, lane, arm, device, rounds, size),
          flush=True)


def warmup(lane, arm, shape, ms):
    print("FSPEED-WARMUP lane=%s arm=%s shape=%s ms=%.6f"
          % (lane, arm, shape, ms), flush=True)


def emit(lane, arm, shape, idx, ms, h):
    print("FSPEED lane=%s arm=%s shape=%s round=%d ms=%.6f hash=%s"
          % (lane, arm, shape, idx, ms, h), flush=True)


def note(lane, arm, text):
    print("FSPEED-NOTE lane=%s arm=%s %s" % (lane, arm, text), flush=True)


def refuse(lane, arm, reason):
    print("FSPEED-REFUSED lane=%s arm=%s reason=%s"
          % (lane, arm, str(reason).replace("\n", " ")), flush=True)


# ---------------------------------------------------------------------------
# Devices and synchronization.
# ---------------------------------------------------------------------------

def _sync():
    """Drain whichever GPU runtime is loaded. A no-op for the CPU arms.

    Both are tried because an arm may be using cupy (every cuml estimator) or
    torch (the cholesky arm) and this file does not want to know which. An
    unsynchronized timing on either measures enqueue rate.
    """
    cp = sys.modules.get("cupy")
    if cp is not None:
        try:
            cp.cuda.runtime.deviceSynchronize()
        except Exception:
            pass
    torch = sys.modules.get("torch")
    if torch is not None:
        try:
            if torch.cuda.is_available():
                torch.cuda.synchronize()
        except Exception:
            pass


def gpu_device_name():
    try:
        import cupy as cp
        props = cp.cuda.runtime.getDeviceProperties(cp.cuda.runtime.getDevice())
        return props["name"].decode().replace(" ", "_")
    except Exception:
        pass
    try:
        import torch
        if torch.cuda.is_available():
            return torch.cuda.get_device_name(0).replace(" ", "_")
    except Exception:
        pass
    return "unknown-gpu"


def cpu_device_name():
    import platform
    n = os.cpu_count() or 0
    return ("%s-%dcores" % (platform.machine(), n)).replace(" ", "_")


# ---------------------------------------------------------------------------
# The fixture dump reader.
# ---------------------------------------------------------------------------

class Fixture:
    """`<dump-dir>/<lane>.fixture`, as written by the Mojo driver.

    Three line kinds and nothing else matters:

        PARAM    <name> <token>
        PARAMHEX <name> <8 hex digits>       a float32 by its BITS
        ARRAY    <name> <f32|i32> <count>    then `count` hex-word lines
    """

    def __init__(self, lane):
        d = os.environ.get("MOJOLEARN_SPEED_DUMP", "")
        if not d:
            raise RuntimeError(
                "MOJOLEARN_SPEED_DUMP is not set and this lane's fixture is a "
                "Mojo builder. Run the Mojo driver once with that variable "
                "set to write <lane>.fixture, then run this arm against it.")
        self.path = os.path.join(d, "%s.fixture" % lane)
        if not os.path.exists(self.path):
            raise RuntimeError(
                "no fixture dump at %s. Run: MOJOLEARN_SPEED_DUMP=%s "
                "MOJOLEARN_SPEED_LANE=%s pixi run mojo run -I . "
                "bench/speed/classical_speed_main.mojo" % (self.path, d, lane))
        self.params = {}
        self.arrays = {}
        with open(self.path) as fh:
            lines = fh.read().split("\n")
        i = 0
        while i < len(lines):
            line = lines[i].strip()
            i += 1
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if parts[0] == "PARAM":
                self.params[parts[1]] = " ".join(parts[2:])
            elif parts[0] == "PARAMHEX":
                bits = np.array([int(parts[2], 16)], dtype=np.uint32)
                self.params[parts[1]] = float(bits.view(np.float32)[0])
            elif parts[0] == "ARRAY":
                name, dtype, count = parts[1], parts[2], int(parts[3])
                words = np.empty(count, dtype=np.uint32)
                for j in range(count):
                    words[j] = int(lines[i + j].strip(), 16)
                i += count
                if dtype == "f32":
                    self.arrays[name] = words.view(np.float32).copy()
                else:
                    self.arrays[name] = words.view(np.int32).copy()

    def f32(self, name, *shape):
        a = self.arrays[name].astype(np.float32, copy=True)
        return a.reshape(*shape) if shape else a

    def i32(self, name, *shape):
        a = self.arrays[name].astype(np.int32, copy=True)
        return a.reshape(*shape) if shape else a

    def i(self, name):
        return int(self.params[name])

    def f(self, name):
        return float(self.params[name])

    def s(self, name):
        return str(self.params[name])


# ---------------------------------------------------------------------------
# The round loop. Every lane hands back a closure and a hash function.
# ---------------------------------------------------------------------------

def gpu_only():
    """True when this box is a GPU vendor's box and the vendor's CPU path is
    therefore ILLEGAL as an opponent.

    THE RULE, AND IT IS NOT A PREFERENCE. On NVIDIA and on AMD we compare
    against the vendor's GPU path ONLY. Their CPU path is for the MacBook,
    where it is the only path they have.

    A GPU-versus-CPU ratio is not the claim this project makes and it is
    not the claim a reader will take from it. `catboost-cpu` beside
    `catboost-gpu` on an H100 invites the table to be graded on the easy
    comparison, and the easy comparison is meaningless: the interesting
    number is ours against their own CUDA kernel on the same silicon.

    It is also not free. `lightgbm-cpu` took 89 SECONDS on 522,911 rows in
    the rf lane. At a 5,000,000-row rung that is most of a per-arm budget
    spent measuring something nobody asked about.

    Default ON wherever CUDA or ROCm is visible. `MOJOLEARN_SPEED_DEVICES`
    can say `cpu` to turn it off, which is what the Apple runs do, and the
    header line records which way it went so a table can never be read
    without knowing.
    """
    want = os.environ.get("MOJOLEARN_SPEED_DEVICES", "").strip().lower()
    if want:
        return "cpu" not in [w.strip() for w in want.split(",")]
    return _accel_visible()


def _accel_visible():
    try:
        import torch                                    # noqa: PLC0415
        if torch.cuda.is_available():
            return True
    except Exception:                                   # noqa: BLE001
        pass
    for var in ("CUDA_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES"):
        if os.environ.get(var, "").strip() not in ("", "-1"):
            return True
    return bool(shutil.which("nvidia-smi") or shutil.which("rocm-smi"))


def to_device(lane, arm, *arrays):
    """Move host arrays to the GPU BEFORE the clock starts.

    THIS EXISTS BECAUSE THE ARMS WERE NOT MEASURING THE SAME REGION, AND THE
    ASYMMETRY RAN IN OUR FAVOUR AT EXACTLY THE SHAPES THAT MATTER.

    Our Mojo arm uploads its fixture once, before the round loop, and times
    the fit alone. The cuML arms were handing `estimator.fit()` a HOST NUMPY
    ARRAY inside the timed region, so every round cuML paid a host-to-device
    transfer our arm had already paid outside the clock. At `ols` and
    `kmeans` that array is 4,000,000 x 32 float32 -- FIVE HUNDRED AND TWELVE
    MEGABYTES, every round, on their side only.

    Measured 2026-08-26, before this fix: ols ours 8.6 ms against cuML 176.9
    ms (20.6x) and kmeans ours 155.1 ms against 236.8 ms (1.53x). Those were
    the two largest-shape rows in the classical set and therefore the only
    two that were quotable at all, and a chunk of both was PCIe.

    `output_type="cupy"` is the other half: it keeps the result on the
    device so the device-to-host copy of, for instance, kmeans' 4,000,000
    int32 labels does not land inside the clock either. `_host_view` takes
    the copy afterwards, where our arm takes its own.

    IF CUPY IS NOT IMPORTABLE the arrays are returned unchanged and a NOTE
    says so on the row, because a silently host-fed arm is the bug this
    function exists to remove and it must not come back quietly.
    """
    try:
        import cupy as cp                               # noqa: PLC0415
    except Exception as e:                              # noqa: BLE001
        note(lane, arm, "cupy is not importable (%r), so this arm's inputs "
                        "stay on the HOST and every timed round pays a "
                        "transfer our arm pays once outside the clock. The "
                        "ratio on this row is NOT comparable." % (e,))
        return arrays if len(arrays) != 1 else arrays[0]
    out = tuple(cp.asarray(a) for a in arrays)
    cp.cuda.runtime.deviceSynchronize()
    return out if len(out) != 1 else out[0]


def race(lane, arm, shape, rounds, size, device, call):
    """One warm-up plus `rounds` timed calls of `call`, which returns the
    outputs to hash (a tuple, or `None` for a lane whose output is not
    comparable with ours)."""
    # THE VENDOR'S CPU PATH DOES NOT RUN ON THE VENDOR'S GPU BOX.
    #
    # Refused BY NAME rather than dropped, because "cuML has no GaussianMixture
    # and so this lane has no legal opponent on NVIDIA" is a finding about
    # their coverage. A lane that silently prints nothing reads as a lane
    # nobody ran.
    if arm.endswith("-cpu") and gpu_only():
        refuse(lane, arm, "GPU-PATH-ONLY: this box is a GPU vendor's box and "
                          "%s is their CPU path. On NVIDIA and AMD we compare "
                          "against the vendor's GPU arm only; the CPU arm is "
                          "the MacBook's. This lane therefore has no legal "
                          "opponent here, which is a fact about the vendor's "
                          "GPU coverage and not a failure of this run." % arm)
        return
    header(lane, arm, device, rounds, size)
    hashes = []
    for r in range(rounds + 1):
        t0 = time.perf_counter()
        out = call()
        _sync()
        ms = (time.perf_counter() - t0) * 1000.0
        if out is None:
            h = "-"
        else:
            h = hash_arrays(lane, arm, *out)
        if r == 0:
            warmup(lane, arm, shape, ms)
        else:
            emit(lane, arm, shape, r, ms, h)
            hashes.append(h)
    for h in hashes[1:]:
        if h != hashes[0] and "-" not in (h, hashes[0]):
            note(lane, arm, "hash moved across rounds: %s %s" % (hashes[0], h))
            break


# ===========================================================================
# GROUP 1: the five lanes whose data is the splitmix64 recurrence.
#
# SHAPES TRANSCRIBED from `bench/bench_main.mojo`, exactly as
# `bench/bench_sklearn.py` transcribes them and exactly as
# `bench/speed/classical_speed_main.mojo` transcribes them. Three copies of
# one table is two too many and it is what the current layout costs; if that
# file's shapes move, all three move.
# ===========================================================================

def shapes(smoke):
    if smoke:
        return dict(km_rows=40000, km_cols=32, km_k=64, km_iter=20,
                    knn_index=40000, knn_queries=400, knn_cols=32, knn_k=10,
                    pca_rows=40000, pca_cols=32, pca_comp=8,
                    db_rows=512, db_cols=16,
                    ols_rows=40000, ols_cols=32)
    return dict(km_rows=4000000, km_cols=32, km_k=64, km_iter=20,
                knn_index=400000, knn_queries=4000, knn_cols=32, knn_k=10,
                pca_rows=4000000, pca_cols=32, pca_comp=8,
                db_rows=4000, db_cols=16,
                ols_rows=4000000, ols_cols=32)


def lane_kmeans(rounds, size, smoke):
    lane, arm = "kmeans", "cuml-gpu"
    try:
        from cuml.cluster import KMeans
    except Exception as e:
        refuse(lane, arm, "import failed: %r" % (e,))
        return
    s = shapes(smoke)
    X = np.ascontiguousarray(
        u01(s["km_rows"], s["km_cols"], 0) * 10.0, dtype=np.float32)
    # THE INITIAL CENTROIDS ARE ROWS `c * 7919`, NOT ROW 0.
    # `bench/bench_main.mojo:106` seeds centroid `c` from `_u01(c * 7919, f, 5)`.
    # `bench/bench_sklearn.py:164` writes `u01(c * 7919 + 1, km_cols, 5)[0]`,
    # which indexes row ZERO of that block for every `c` and therefore hands
    # scikit-learn 64 IDENTICAL centroids. That is a defect in that file, it
    # is reported as DEVIATION 1810, and it is NOT reproduced here: this arm
    # takes the row the Mojo side actually uses.
    init = np.ascontiguousarray(
        u01_at(np.arange(s["km_k"]) * 7919, s["km_cols"], 5) * 10.0,
        dtype=np.float32)
    # ON THE DEVICE BEFORE THE CLOCK, matching our arm. At 4,000,000 x 32
    # this array is 512 MB and it was being transferred inside every timed
    # round on their side only.
    X, init = to_device(lane, arm, X, init)
    km = None

    def call():
        nonlocal km
        # Constructed inside the clock because cuml.KMeans does no device work
        # in __init__; fit is where everything happens. Matching our arm,
        # which also re-uploads its initial centroids each round.
        km = KMeans(n_clusters=s["km_k"], init=init, n_init=1,
                    max_iter=s["km_iter"], tol=1e-7, output_type="cupy")
        km.fit(X)
        return (km.cluster_centers_, km.labels_)

    race(lane, arm, "%dx%dk%di%d" % (s["km_rows"], s["km_cols"], s["km_k"],
                                     s["km_iter"]),
         rounds, size, gpu_device_name(), call)


def lane_dbscan(rounds, size, smoke):
    lane, arm = "dbscan", "cuml-gpu"
    try:
        from cuml.cluster import DBSCAN
    except Exception as e:
        refuse(lane, arm, "import failed: %r" % (e,))
        return
    s = shapes(smoke)
    X = np.ascontiguousarray(
        u01(s["db_rows"], s["db_cols"], 4) * 2.0, dtype=np.float32)
    note(lane, arm, "calc_core_sample_indices=False: our dbscan_fit_impl "
                    "returns labels only, and computing their core-sample "
                    "index array would be work our arm does not do")

    def call():
        # eps and min_samples are passed on both sides; nothing is left to a
        # default that could differ between the two libraries.
        db = DBSCAN(eps=0.35, min_samples=5, calc_core_sample_indices=False,
                    output_type="numpy")
        db.fit(X)
        return (db.labels_,)

    race(lane, arm, "%dx%d" % (s["db_rows"], s["db_cols"]), rounds, size,
         gpu_device_name(), call)


def lane_pca(rounds, size, smoke):
    lane, arm = "pca", "cuml-gpu"
    try:
        from cuml.decomposition import PCA
    except Exception as e:
        refuse(lane, arm, "import failed: %r" % (e,))
        return
    s = shapes(smoke)
    X = np.ascontiguousarray(
        u01(s["pca_rows"], s["pca_cols"], 3) * 4.0, dtype=np.float32)

    # OUR ROUTE IS THE COVARIANCE EIGENDECOMPOSITION. Whichever of cuML's
    # solvers matches it is what this arm asks for, and the solver that
    # actually ran is printed. `auto` is NOT used: a comparison that depends
    # on somebody's heuristic staying put is a comparison that stops being one
    # without telling you.
    # PROBE BY FITTING, NOT BY CONSTRUCTING. cuml.PCA's constructor does not
    # validate `svd_solver`, so the old probe "accepted" covariance_eigh and
    # the ValueError arrived at fit() -- inside the timed region, killing the
    # arm. On 2026-08-25 that is exactly what happened and the pca lane came
    # home with no opponent at all:
    #
    #   ValueError: Expected `svd_solver` to be one of
    #               ['auto', 'full', 'jacobi'], got 'covariance_eigh'
    #
    # `jacobi` is listed FIRST because it is the real algorithm match: cuML's
    # jacobi solver takes an iterative eigendecomposition, and our `pca_fit`
    # forms the covariance and runs `jacobi_eigh_device` on it. `full` is the
    # fallback and is a DIFFERENT decomposition (SVD of the data matrix), so
    # the note below says so when it is what ran.
    _probe = np.ascontiguousarray(
        u01(64, s["pca_cols"], 99), dtype=np.float32)
    solver = None
    for cand in ("jacobi", "full", "covariance_eigh"):
        try:
            PCA(n_components=s["pca_comp"], svd_solver=cand,
                output_type="numpy").fit(_probe)
            solver = cand
            break
        except Exception:
            continue
    if solver is None:
        refuse(lane, arm, "cuml.PCA accepted none of covariance_eigh/full/jacobi")
        return
    if solver == "jacobi":
        note(lane, arm, "svd_solver=jacobi, algorithm-matched: their "
                        "iterative eigendecomposition against our pca_fit, "
                        "which forms the covariance and runs "
                        "jacobi_eigh_device on it")
    elif solver == "covariance_eigh":
        note(lane, arm, "svd_solver=covariance_eigh, algorithm-matched")
    else:
        note(lane, arm, "svd_solver=%s: this is an SVD OF THE DATA MATRIX, a "
                        "DIFFERENT decomposition from our covariance-plus-"
                        "eigen route, so the ratio on this row is algorithm "
                        "plus device and not device alone" % solver)

    X = to_device(lane, arm, X)

    def call():
        p = PCA(n_components=s["pca_comp"], svd_solver=solver,
                output_type="cupy")
        p.fit(X)
        return (p.components_, p.explained_variance_, p.singular_values_)

    race(lane, arm, "%dx%dc%d" % (s["pca_rows"], s["pca_cols"], s["pca_comp"]),
         rounds, size, gpu_device_name(), call)


def lane_ols(rounds, size, smoke):
    lane, arm = "ols", "cuml-gpu"
    try:
        from cuml.linear_model import LinearRegression
    except Exception as e:
        refuse(lane, arm, "import failed: %r" % (e,))
        return
    s = shapes(smoke)
    A = np.ascontiguousarray(
        u01(s["ols_rows"], s["ols_cols"], 6) - 0.5, dtype=np.float32)
    w = 1.0 + 0.1 * np.arange(s["ols_cols"])
    b = np.ascontiguousarray(A @ w, dtype=np.float32)
    # `algorithm="eig"` is the normal equations through an eigendecomposition
    # of X^T X: the SAME algorithm class as `lstsq_eig`. It is cuML's default
    # for a tall matrix and it is passed explicitly anyway. `fit_intercept`
    # is False on both sides.
    note(lane, arm, "algorithm=eig fit_intercept=False, algorithm-matched "
                    "with lstsq_eig (normal equations, eigen route)")

    # ON THE DEVICE BEFORE THE CLOCK. `A` is 4,000,000 x 32 float32 = 512 MB
    # and was crossing PCIe inside every timed round on their side alone.
    A, b = to_device(lane, arm, A, b)

    def call():
        m = LinearRegression(algorithm="eig", fit_intercept=False,
                             output_type="cupy")
        m.fit(A, b)
        return (m.coef_,)

    race(lane, arm, "%dx%d" % (s["ols_rows"], s["ols_cols"]), rounds, size,
         gpu_device_name(), call)


def lane_knn(rounds, size, smoke):
    lane, arm = "knn", "cuml-gpu"
    try:
        from cuml.neighbors import NearestNeighbors
    except Exception as e:
        refuse(lane, arm, "import failed: %r" % (e,))
        return
    s = shapes(smoke)
    idx = np.ascontiguousarray(
        u01(s["knn_index"], s["knn_cols"], 1), dtype=np.float32)
    qry = np.ascontiguousarray(
        u01(s["knn_queries"], s["knn_cols"], 2), dtype=np.float32)
    # THE INDEX BUILD IS OUTSIDE THE CLOCK because our timed region is a
    # search, not a build: `brute_force_knn_impl` has no index to build. The
    # two `compute_norms` calls ARE inside ours, and cuML's `kneighbors`
    # computes its norms inside itself, so the two regions cover the same
    # work.
    # Symmetric with the other lanes even though the asymmetry here ran
    # AGAINST them: `qry` is only 4,000 x 32 and the transfer was inside
    # their clock, so our 21.6x loss on this row was if anything understated.
    idx, qry = to_device(lane, arm, idx, qry)
    nn = NearestNeighbors(n_neighbors=s["knn_k"], algorithm="brute",
                          metric="euclidean", output_type="cupy")
    nn.fit(idx)
    _sync()
    note(lane, arm, "cuML returns EUCLIDEAN distances and our arm returns "
                    "SQUARED ones (is_sqrt=False); sqrt is monotone so the "
                    "neighbor sets and their order agree, and the extra root "
                    "is one elementwise pass on their side")

    def call():
        d, i = nn.kneighbors(qry)
        return (d, i)

    race(lane, arm, "%dx%dq%dk%d" % (s["knn_index"], s["knn_cols"],
                                     s["knn_queries"], s["knn_k"]),
         rounds, size, gpu_device_name(), call)


# ===========================================================================
# GROUP 2: the dumped-fixture lanes.
# ===========================================================================

def lane_cd(rounds, size, smoke):
    lane, arm = "cd", "cuml-gpu"
    try:
        from cuml.linear_model import Lasso
        fx = Fixture(lane)
    except Exception as e:
        refuse(lane, arm, "%r" % (e,))
        return
    n, d = fx.i("n"), fx.i("d")
    X = fx.f32("x", n, d)
    y = fx.f32("y")
    alpha = fx.f("alpha")
    max_iter = fx.i("max_iter")
    tol = fx.f("tol")
    note(lane, arm, "alpha, max_iter, tol, fit_intercept and selection are "
                    "all passed explicitly; cuML's tol default is 1e-3 and "
                    "ours is the fixture's, and they are made equal here")

    def call():
        m = Lasso(alpha=alpha, fit_intercept=True, max_iter=max_iter,
                  tol=tol, selection="cyclic", output_type="numpy")
        m.fit(X, y)
        return (m.coef_,)

    race(lane, arm, "%dx%d" % (n, d), rounds, size, gpu_device_name(), call)


def lane_kde(rounds, size, smoke):
    lane, arm = "kde", "cuml-gpu"
    try:
        from cuml.neighbors import KernelDensity
        fx = Fixture(lane)
    except Exception as e:
        refuse(lane, arm, "%r" % (e,))
        return
    n_train, n_query, d = fx.i("n_train"), fx.i("n_query"), fx.i("d")
    train = fx.f32("train", n_train, d)
    query = fx.f32("query", n_query, d)
    weights = fx.f32("weights")
    bw = fx.f("bandwidth")
    # `fit` stores the training set; the SCORE is the timed call, matching our
    # `score_samples`, whose validation and upload also happen once.
    try:
        kd = KernelDensity(bandwidth=bw, kernel="gaussian", metric="euclidean")
        kd.fit(train, sample_weight=weights)
        _sync()
    except Exception as e:
        refuse(lane, arm, "cuml.KernelDensity fit failed: %r" % (e,))
        return

    def call():
        return (kd.score_samples(query),)

    race(lane, arm, "%dx%dx%d" % (n_train, n_query, d), rounds, size,
         gpu_device_name(), call)


def lane_linkage(rounds, size, smoke):
    lane, arm = "linkage", "cuml-gpu"
    try:
        from cuml.cluster import AgglomerativeClustering
        fx = Fixture(lane)
    except Exception as e:
        refuse(lane, arm, "%r" % (e,))
        return
    n, d, k = fx.i("n"), fx.i("d"), fx.i("n_clusters")
    X = fx.f32("x", n, d)
    note(lane, arm, "linkage=single connectivity=pairwise metric=euclidean; "
                    "cuML implements single linkage only, which is the arm "
                    "our single_linkage was ported from")

    # `metric=` replaced `affinity=` in cuML at some point between the
    # versions this repository has been built against, and which one the pod
    # has is not knowable from here. Both spellings mean euclidean; the one
    # that constructs is the one that runs, and it is named in a note.
    kw = dict(n_clusters=k, linkage="single", connectivity="pairwise",
              output_type="numpy")
    keyword = None
    for cand in ("metric", "affinity"):
        try:
            AgglomerativeClustering(**dict(kw, **{cand: "euclidean"}))
            keyword = cand
            break
        except Exception:
            continue
    if keyword is None:
        refuse(lane, arm, "cuml.AgglomerativeClustering accepted neither "
                          "metric= nor affinity=")
        return
    note(lane, arm, "distance keyword on this build is %s=euclidean" % keyword)
    kw[keyword] = "euclidean"

    def call():
        m = AgglomerativeClustering(**kw)
        m.fit(X)
        return (m.labels_,)

    race(lane, arm, "%s.%dx%d" % (fx.s("fixture"), n, d), rounds, size,
         gpu_device_name(), call)


def lane_svm(rounds, size, smoke):
    lane, arm = "svm", "cuml-gpu"
    try:
        from cuml.svm import SVC
        fx = Fixture(lane)
    except Exception as e:
        refuse(lane, arm, "%r" % (e,))
        return
    n, d = fx.i("n"), fx.i("d")
    X = fx.f32("x", n, d)
    y = fx.f32("y")
    C = float(fx.s("C"))
    gamma = float(fx.s("gamma"))
    tol = float(fx.s("tol"))
    note(lane, arm, "FIT ONLY. C, gamma, tol and nochange_steps are passed "
                    "explicitly; cache_size is 0 on our side and left at "
                    "cuML's default here because a cache size is a memory "
                    "policy, not a parameter of the answer")

    def call():
        m = SVC(kernel="rbf", C=C, gamma=gamma, tol=tol,
                nochange_steps=fx.i("nochange_steps"), output_type="numpy")
        m.fit(X, y)
        return (m.dual_coef_, m.support_)

    race(lane, arm, "%s.%dx%d" % (fx.s("fixture"), n, d), rounds, size,
         gpu_device_name(), call)


def lane_metrics(rounds, size, smoke):
    lane, arm = "metrics", "cuml-gpu"
    try:
        fx = Fixture(lane)
        from cuml.metrics import accuracy_score, kl_divergence, r2_score
        from cuml.metrics import trustworthiness
        from cuml.metrics.cluster import (
            adjusted_rand_score, completeness_score, entropy,
            homogeneity_score, mutual_info_score, silhouette_score,
            v_measure_score,
        )
    except Exception as e:
        refuse(lane, arm, "import failed (one of the eleven cuml metrics is "
                          "missing on this build): %r" % (e,))
        return
    n_lab = fx.i("n_labels")
    n_flt = fx.i("n_float")
    n_sil, d_sil, k_sil = fx.i("n_sil"), fx.i("d_sil"), fx.i("k_sil")
    n_tru, m_tru, d_tru, k_tru = (fx.i("n_trust"), fx.i("m_trust"),
                                  fx.i("d_trust"), fx.i("k_trust"))
    yt = fx.i32("y_true")
    yp = fx.i32("y_pred")
    y = fx.f32("y")
    yhat = fx.f32("y_hat")
    p = fx.f32("p")
    q = fx.f32("q")
    sil_x = fx.f32("sil_x", n_sil, d_sil)
    sil_l = fx.i32("sil_labels")
    tr_x = fx.f32("trust_x", n_tru, m_tru)
    tr_e = fx.f32("trust_emb", n_tru, d_tru)
    note(lane, arm, "ELEVEN metrics, the same eleven the Mojo lane times. "
                    "rand_index is in neither pass: cuML ships no plain Rand "
                    "index and an arm computing one more metric than the "
                    "other is not a comparison")
    note(lane, arm, "hash=- for this lane: the eleven return values are "
                    "Python floats of two different widths on the two sides "
                    "and folding them would compare formatting, not answers")

    def call():
        accuracy_score(yt, yp)
        adjusted_rand_score(yt, yp)
        entropy(yt)
        mutual_info_score(yt, yp)
        homogeneity_score(yt, yp)
        completeness_score(yt, yp)
        v_measure_score(yt, yp)
        r2_score(y, yhat)
        kl_divergence(p, q)
        silhouette_score(sil_x, sil_l)
        trustworthiness(tr_x, tr_e, n_neighbors=k_tru)
        return None

    race(lane, arm,
         "lab%d.flt%d.sil%dx%d.tru%dx%d" % (n_lab, n_flt, n_sil, d_sil,
                                            n_tru, m_tru),
         rounds, size, gpu_device_name(), call)


def lane_ivf(rounds, size, smoke):
    lane, arm = "ivf", "cuvs-gpu"
    try:
        import cupy as cp
        from cuvs.neighbors import ivf_flat
        fx = Fixture(lane)
    except Exception as e:
        refuse(lane, arm, "import failed: %r" % (e,))
        return
    n_rows, n_q, dim = fx.i("n_rows"), fx.i("n_queries"), fx.i("dim")
    n_lists, n_probes, k = fx.i("n_lists"), fx.i("n_probes"), fx.i("k")
    kiters = fx.i("kmeans_n_iters")
    dataset = cp.asarray(fx.f32("index", n_rows, dim))
    queries = cp.asarray(fx.f32("queries", n_q, dim))
    _sync()
    note(lane, arm, "ONE build plus ONE search inside the clock, matching "
                    "ivf_flat_build_and_search_host. metric=sqeuclidean, "
                    "kmeans_trainset_fraction=1.0, n_probes and n_lists "
                    "passed explicitly (cuVS defaults n_probes to 20)")

    def call():
        ip = ivf_flat.IndexParams(n_lists=n_lists, metric="sqeuclidean",
                                  kmeans_n_iters=kiters,
                                  kmeans_trainset_fraction=1.0)
        index = ivf_flat.build(ip, dataset)
        dist, ind = ivf_flat.search(ivf_flat.SearchParams(n_probes=n_probes),
                                    index, queries, k)
        return (cp.asarray(dist), cp.asarray(ind))

    race(lane, arm, "%dx%dq%dL%dp%dk%d" % (n_rows, dim, n_q, n_lists,
                                           n_probes, k),
         rounds, size, gpu_device_name(), call)


def lane_hdbscan(rounds, size, smoke):
    lane, arm = "hdbscan", "cuml-gpu"
    try:
        from cuml.cluster import HDBSCAN
        fx = Fixture(lane)
    except Exception as e:
        refuse(lane, arm, "%r" % (e,))
        return
    n, d = fx.i("n"), fx.i("d")
    X = fx.f32("x", n, d)
    note(lane, arm, "min_samples, min_cluster_size, metric and "
                    "cluster_selection_method are passed explicitly. At %d "
                    "rows both arms are dominated by launch latency; this is "
                    "the fixture the lane ships and it has no size knob" % n)

    def call():
        m = HDBSCAN(min_samples=fx.i("min_samples"),
                    min_cluster_size=fx.i("min_cluster_size"),
                    metric="euclidean", cluster_selection_method="eom",
                    output_type="numpy")
        m.fit(X)
        return (m.labels_,)

    race(lane, arm, "%s.%dx%d" % (fx.s("fixture"), n, d), rounds, size,
         gpu_device_name(), call)


def lane_cholesky(rounds, size, smoke):
    lane, arm = "cholesky", "torch-gpu"
    try:
        import torch
        fx = Fixture(lane)
    except Exception as e:
        refuse(lane, arm, "%r" % (e,))
        return
    if not torch.cuda.is_available():
        refuse(lane, arm, "torch reports no CUDA device")
        return
    n, nrhs = fx.i("n"), fx.i("nrhs")
    jitter = fx.f("jitter")
    A0 = torch.tensor(fx.f32("a", n, n), device="cuda")
    B0 = torch.tensor(fx.f32("b", n, nrhs), device="cuda")
    eye = torch.eye(n, device="cuda", dtype=torch.float32)
    torch.cuda.synchronize()
    # THE OPPONENT IS cuSOLVER AND NOT cuML. RAPIDS exposes no public Cholesky
    # estimator; torch.linalg.cholesky IS cuSOLVER's potrf, and
    # torch.cholesky_solve IS its potrs, which is exactly the pair our
    # potrf_lower / cho_solve were ported against.
    note(lane, arm, "torch.linalg.cholesky is cuSOLVER potrf and "
                    "torch.cholesky_solve is potrs; the ridge, the logdet and "
                    "the solve are all inside the clock on both sides")

    def call():
        A = A0 + jitter * eye
        L = torch.linalg.cholesky(A)
        logdet = 2.0 * torch.log(torch.diagonal(L)).sum()
        X = torch.cholesky_solve(B0, L, upper=False)
        return (L, X, logdet.reshape(1))

    race(lane, arm, "%s.%dx%dr%d" % (fx.s("fixture"), n, n, nrhs), rounds,
         size, gpu_device_name(), call)


def lane_gmm(rounds, size, smoke):
    lane, arm = "gmm", "sklearn-cpu"
    try:
        from sklearn.mixture import GaussianMixture
        fx = Fixture(lane)
    except Exception as e:
        refuse(lane, arm, "%r" % (e,))
        return
    # NO RAPIDS COUNTERPART. cuML ships no GaussianMixture at any version, so
    # there is no GPU arm to race and this one is labeled for what it is.
    refuse(lane, "cuml-gpu",
           "RAPIDS ships no GaussianMixture; the arm below is scikit-learn on "
           "the CPU and is labeled sklearn-cpu")
    n, d, k = fx.i("n"), fx.i("d"), fx.i("n_components")
    X = fx.f32("x", n, d)

    def call():
        m = GaussianMixture(n_components=k, covariance_type="full",
                            max_iter=fx.i("max_iter"), tol=fx.f("tol"),
                            reg_covar=fx.f("reg_covar"), init_params="kmeans",
                            n_init=1, random_state=fx.i("random_state"))
        m.fit(X)
        return (m.weights_.astype(np.float32), m.means_.astype(np.float32))

    race(lane, arm, "%s.%dx%dK%d" % (fx.s("fixture"), n, d, k), rounds, size,
         cpu_device_name(), call)


def lane_gp(rounds, size, smoke):
    lane, arm = "gp", "sklearn-cpu"
    try:
        from sklearn.gaussian_process import GaussianProcessRegressor
        from sklearn.gaussian_process.kernels import RBF
        fx = Fixture(lane)
    except Exception as e:
        refuse(lane, arm, "%r" % (e,))
        return
    refuse(lane, "cuml-gpu",
           "RAPIDS ships no Gaussian process regressor; the arm below is "
           "scikit-learn on the CPU and is labeled sklearn-cpu")
    n, d, ns = fx.i("n_train"), fx.i("d"), fx.i("n_star")
    X = fx.f32("x", n, d).astype(np.float64)
    y = fx.f32("y").astype(np.float64)
    Xs = fx.f32("x_star", ns, d).astype(np.float64)
    ls = fx.f32("length_scale").astype(np.float64)
    alpha = fx.f("alpha")
    note(lane, arm, "optimizer=None and normalize_y=False on both sides: our "
                    "gpr_fit_host implements no hyperparameter optimizer "
                    "(DEVIATION 1761), so an arm that optimized would be "
                    "doing different work")
    note(lane, arm, "scikit-learn works in float64 and we work in float32; "
                    "that is a real difference in the amount of arithmetic "
                    "and it cannot be turned off on their side")

    def call():
        m = GaussianProcessRegressor(kernel=RBF(length_scale=ls),
                                     alpha=alpha, optimizer=None,
                                     normalize_y=False)
        m.fit(X, y)
        mean, std = m.predict(Xs, return_std=True)
        return (mean.astype(np.float32), std.astype(np.float32))

    race(lane, arm, "%s.%dx%ds%d" % (fx.s("fixture"), n, d, ns), rounds, size,
         cpu_device_name(), call)


def lane_krr(rounds, size, smoke):
    lane, arm = "krr", "cuml-gpu"
    try:
        from cuml.kernel_ridge import KernelRidge
        fx = Fixture(lane)
    except Exception as e:
        refuse(lane, arm, "import failed: %r" % (e,))
        return
    n, d, nq = fx.i("n"), fx.i("d"), fx.i("n_query")
    X = fx.f32("x", n, d)
    y = fx.f32("y")
    Xq = fx.f32("x_query", nq, d)
    gamma = float(fx.s("gamma"))
    alpha = fx.f("alpha")
    note(lane, arm, "kernel=rbf, gamma and alpha passed explicitly; fit plus "
                    "predict inside the clock, matching our lane")

    def call():
        m = KernelRidge(alpha=alpha, kernel="rbf", gamma=gamma)
        m.fit(X, y)
        return (m.dual_coef_, m.predict(Xq))

    race(lane, arm, "%s.%dx%d" % (fx.s("fixture"), n, d), rounds, size,
         gpu_device_name(), call)


def lane_nystroem(rounds, size, smoke):
    lane, arm = "nystroem", "sklearn-cpu"
    try:
        from sklearn.kernel_approximation import Nystroem
        fx = Fixture(lane)
    except Exception as e:
        refuse(lane, arm, "%r" % (e,))
        return
    refuse(lane, "cuml-gpu",
           "RAPIDS ships no Nystroem; the arm below is scikit-learn on the "
           "CPU and is labeled sklearn-cpu")
    n, d, nq = fx.i("n"), fx.i("d"), fx.i("n_query")
    q = fx.i("n_components")
    X = fx.f32("x", n, d)
    Xq = fx.f32("x_query", nq, d)
    gamma = float(fx.s("gamma"))
    note(lane, arm, "hash=- : the BASIS SAMPLE differs. Ours permutes with a "
                    "pinned Philox stream and theirs with numpy RandomState, "
                    "so the two fit different rows. The WORK is the same "
                    "(sample q rows, form q x q, eigendecompose, scale, cross "
                    "kernel, matmul) and that is what is being timed")

    def call():
        m = Nystroem(kernel="rbf", gamma=gamma, n_components=q,
                     random_state=0)
        m.fit(X)
        m.transform(Xq)
        return None

    race(lane, arm, "%s.%dx%dq%d" % (fx.s("fixture"), n, d, q), rounds, size,
         cpu_device_name(), call)


def lane_rbfsampler(rounds, size, smoke):
    lane, arm = "rbfsampler", "sklearn-cpu"
    try:
        from sklearn.kernel_approximation import RBFSampler
        fx = Fixture(lane)
    except Exception as e:
        refuse(lane, arm, "%r" % (e,))
        return
    refuse(lane, "cuml-gpu",
           "RAPIDS ships no RBFSampler / random Fourier features; the arm "
           "below is scikit-learn on the CPU and is labeled sklearn-cpu")
    d, nq = fx.i("d"), fx.i("n_query")
    q = fx.i("n_components")
    Xq = fx.f32("x_query", nq, d)
    gamma = float(fx.s("gamma"))
    note(lane, arm, "hash=- : the random draws differ (a pinned Philox stream "
                    "against numpy RandomState.normal). Both arms draw a "
                    "d x q weight matrix and a q offset, then one matmul and "
                    "one cosine over %d rows" % nq)

    def call():
        m = RBFSampler(gamma=gamma, n_components=q, random_state=0)
        m.fit(np.zeros((1, d), dtype=np.float32))
        m.transform(Xq)
        return None

    race(lane, arm, "%dx%dq%d" % (nq, d, q), rounds, size, cpu_device_name(),
         call)


def lane_resample(rounds, size, smoke):
    lane, arm = "resample", "scipy-cpu"
    try:
        from scipy.stats import bootstrap
        fx = Fixture(lane)
    except Exception as e:
        refuse(lane, arm, "%r" % (e,))
        return
    refuse(lane, "cuml-gpu",
           "RAPIDS ships no bootstrap; the arm below is SciPy on the CPU and "
           "is labeled scipy-cpu")
    n, d = fx.i("n"), fx.i("d")
    n_res = fx.i("n_resamples")
    X = fx.f32("x", n, d)
    col0 = np.ascontiguousarray(X[:, 0].astype(np.float64))
    conf = fx.f("confidence_level")
    note(lane, arm, "statistic=mean of column 0, method=percentile, "
                    "n_resamples=%d, confidence_level passed explicitly. Our "
                    "bootstrap_host resamples ROWS of the two-column sample "
                    "(SciPy's paired=True shape) and computes the mean of "
                    "column 0, which is what this arm does" % n_res)
    note(lane, arm, "with_bca_diagnostics is False on our side, so neither "
                    "arm computes a jackknife")

    def call():
        r = bootstrap((col0,), np.mean, n_resamples=n_res,
                      confidence_level=conf, method="percentile",
                      vectorized=False, random_state=0)
        return (np.asarray([r.standard_error], dtype=np.float64),)

    race(lane, arm, "%dx%dr%d" % (n, d, n_res), rounds, size,
         cpu_device_name(), call)


def lane_spectral(rounds, size, smoke):
    lane, arm = "spectral", "sklearn-cpu"
    try:
        from sklearn.cluster import SpectralClustering
        fx = Fixture(lane)
    except Exception as e:
        refuse(lane, arm, "%r" % (e,))
        return
    refuse(lane, "cuml-gpu",
           "RAPIDS ships no SpectralClustering estimator; the arm below is "
           "scikit-learn on the CPU and is labeled sklearn-cpu")
    n, d = fx.i("n"), fx.i("d")
    X = fx.f32("x", n, d).astype(np.float64)
    note(lane, arm, "affinity=nearest_neighbors with the same n_neighbors, "
                    "same n_clusters, same n_components, same n_init: both "
                    "arms build a kNN affinity, take a Lanczos "
                    "eigendecomposition of the normalized Laplacian and run "
                    "k-means on the embedding. The eigensolvers are two "
                    "Lanczos implementations (ours restarts, theirs is ARPACK)")
    note(lane, arm, "hash=- : cluster label NUMBERING is arbitrary in both "
                    "and the two are at best a permutation of each other")

    def call():
        m = SpectralClustering(n_clusters=fx.i("n_clusters"),
                               n_components=fx.i("n_components"),
                               affinity="nearest_neighbors",
                               n_neighbors=fx.i("n_neighbors"),
                               eigen_solver="arpack",
                               n_init=fx.i("n_init"),
                               assign_labels="kmeans",
                               random_state=fx.i("seed"))
        m.fit(X)
        return None

    race(lane, arm, "%dx%dc%d" % (n, d, fx.i("n_clusters")), rounds, size,
         cpu_device_name(), call)


def lane_holtwinters(rounds, size, smoke):
    lane, arm = "holtwinters", "cuml-gpu"
    try:
        from cuml import ExponentialSmoothing
        fx = Fixture(lane)
    except Exception as e:
        refuse(lane, arm, "import failed: %r" % (e,))
        return
    n, batch, freq = fx.i("n"), fx.i("batch_size"), fx.i("frequency")
    y = fx.f32("y", batch, n)
    note(lane, arm, "seasonal=additive, seasonal_periods, start_periods, "
                    "ts_num and eps all passed explicitly. The dump is "
                    "series-major (batch_size x n), which is cuML's own "
                    "(ts_num, n) layout, so no transpose is needed")

    def call():
        m = ExponentialSmoothing(y, seasonal="additive",
                                 seasonal_periods=freq,
                                 start_periods=fx.i("start_periods"),
                                 ts_num=batch, eps=fx.f("eps"))
        m.fit()
        return (to_numpy(m.get_level()).astype(np.float32),)

    race(lane, arm, "%dx%df%d" % (batch, n, freq), rounds, size,
         gpu_device_name(), call)


def lane_kpss(rounds, size, smoke):
    lane = "kpss"
    try:
        fx = Fixture(lane)
    except Exception as e:
        refuse(lane, "cuml-gpu", "%r" % (e,))
        return
    n_obs, batch = fx.i("n_obs"), fx.i("batch_size")
    y = fx.f32("y", batch, n_obs)
    shape = "%dx%d" % (batch, n_obs)

    # PREFER cuML'S OWN. `cuml.tsa.stationarity` is where our port came from;
    # where the installed RAPIDS exposes it, that is the honest opponent and
    # the arm is `cuml-gpu`. Where it does not, statsmodels' single-series
    # KPSS on the CPU is the strongest thing that genuinely runs, and the arm
    # says `statsmodels-cpu` so nobody reads a CPU number as a GPU one.
    try:
        from cuml.tsa.stationarity import kpss_test as cuml_kpss
    except Exception:
        cuml_kpss = None

    if cuml_kpss is not None:
        arm = "cuml-gpu"
        note(lane, arm, "cuml.tsa.stationarity.kpss_test at d=1, D=0, s=0, "
                        "pval_threshold=0.05, the same batch of %d series"
             % batch)

        def call():
            return (np.asarray(to_numpy(cuml_kpss(y, d=1, D=0, s=0,
                                                  pval_threshold=0.05))),)

        race(lane, arm, shape, rounds, size, gpu_device_name(), call)
        return

    refuse(lane, "cuml-gpu",
           "this RAPIDS build exposes no cuml.tsa.stationarity.kpss_test; "
           "falling back to statsmodels on the CPU, labeled statsmodels-cpu")
    try:
        from statsmodels.tsa.stattools import kpss as sm_kpss
    except Exception as e:
        refuse(lane, "statsmodels-cpu", "import failed: %r" % (e,))
        return
    arm = "statsmodels-cpu"
    diffed = np.diff(y.astype(np.float64), axis=1)
    note(lane, arm, "ONE SERIES AT A TIME, %d of them, on the first "
                    "difference: statsmodels has no batched entry and ours is "
                    "batched, so this arm's number includes %d Python calls "
                    "that our single launch does not have" % (batch, batch))
    note(lane, arm, "hash=- : statsmodels returns a float64 statistic from a "
                    "different lag-truncation rule and the two are not the "
                    "same number")

    def call():
        for b in range(batch):
            sm_kpss(diffed[b], regression="c", nlags="legacy")
        return None

    race(lane, arm, shape, rounds, size, cpu_device_name(), call)


# ===========================================================================
# main
# ===========================================================================

LANES = {
    "kmeans": lane_kmeans,
    "dbscan": lane_dbscan,
    "pca": lane_pca,
    "ols": lane_ols,
    "knn": lane_knn,
    "cd": lane_cd,
    "kde": lane_kde,
    "linkage": lane_linkage,
    "svm": lane_svm,
    "metrics": lane_metrics,
    "ivf": lane_ivf,
    "hdbscan": lane_hdbscan,
    "cholesky": lane_cholesky,
    "gmm": lane_gmm,
    "gp": lane_gp,
    "krr": lane_krr,
    "nystroem": lane_nystroem,
    "rbfsampler": lane_rbfsampler,
    "resample": lane_resample,
    "spectral": lane_spectral,
    "holtwinters": lane_holtwinters,
    "kpss": lane_kpss,
}


def main():
    lane = os.environ.get("MOJOLEARN_SPEED_LANE", "")
    rounds = int(os.environ.get("MOJOLEARN_SPEED_ROUNDS", "5"))
    size = os.environ.get("MOJOLEARN_SPEED_SIZE", "shipped")
    if size not in ("shipped", "smoke"):
        raise SystemExit("MOJOLEARN_SPEED_SIZE must be shipped or smoke")
    if rounds < 1:
        raise SystemExit("MOJOLEARN_SPEED_ROUNDS must be >= 1")
    if lane not in LANES:
        raise SystemExit(
            "MOJOLEARN_SPEED_LANE must be one of: %s; got %r"
            % (" ".join(sorted(LANES)), lane))
    _check_u01_variant()
    try:
        LANES[lane](rounds, size, size == "smoke")
    except Exception as e:
        # A LANE THAT DIES IS A REFUSAL, NOT A CRASH. The box is rented and
        # the leg body runs one process per lane; a traceback that took the
        # exit code with it would look the same as a lane that was never run.
        refuse(lane, "unknown", "raised at run time: %r" % (e,))
    return 0


if __name__ == "__main__":
    sys.exit(main())
