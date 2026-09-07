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
label. `bench/speed/README.md` defines the comparison policy so nobody has
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

ACCURACY IS NOT MEASURED HERE, WITH ONE EXCEPTION
==================================================
`tools/fast_speed_table.py` understands an `FSPEED-ACC` line. This file emits
one only for the `umap` lane (DEVIATION 2136). A speed harness that also
scores accuracy invites a reader to trade one against the other in a single
table, and the lanes that need an accuracy statement have gates that make it
properly. UMAP is the exception because its two arms share NO bits (two
different optimizers, two different neighbor graphs) and the only statement
that ties the timed embedding to a sane one is a trustworthiness score,
computed by ONE helper (`trustworthiness_subsample`) on the same 10,000-row
subsample for every arm, ours included.

THE SECOND ARM SELECTOR: THE VENDOR'S DETERMINISTIC CONFIGURATION
==================================================================
DEVIATION 2131. `MOJOLEARN_SPEED_ARM=deterministic` runs the SAME lane in the
vendor's DOCUMENTED deterministic configuration, as arm
`<arm>-deterministic`, with every other parameter equal to the default arm.
Absent or empty (`default`) is the behaviour this file always had, byte for
byte. The NVIDIA identity-cost campaign (2026-09-07) needs both: what a
vendor charges for a repeatable answer on its own GPU, beside what we charge
for one that is bitwise the same on every vendor's.

A vendor that documents no such configuration is not given one. The table
`DETERMINISTIC_SIBLINGS` (keyed by lane, the same shape as
`tools/speed_gbdt_arm.py::NOT_OFFERED`, DEVIATION 1890) carries, per lane,
one of four verdicts and the vendor's OWN sentence with the URL it was read
from on 2026-09-07:

    LIVE                      the sibling runs (kmeans, umap, cholesky)
    NOT-OFFERED               the vendor documents that the switch does not
                              apply to what is timed (svm)
    DETERMINISTIC-BY-DEFAULT  the vendor documents the default as already
                              deterministic; the fast arm's per-round hash
                              IS the deterministic arm (hdbscan)
    NOT-DOCUMENTED            the vendor's page for that estimator states
                              nothing about determinism (the rest)

Every non-LIVE verdict is emitted as exactly one `FSPEED-REFUSED` line with
the verdict as the reason's prefix, so the leg's log reader can find it. No
sentence in that table is paraphrased into a stronger claim than the vendor
made, and a page that is silent is recorded as silent.

SIZES: shipped, smoke, large, wide
==================================
DEVIATION 2132. `MOJOLEARN_SPEED_SIZE` takes four values. `shipped` and
`smoke` are unchanged. `large` reproduces, per lane, EXACTLY the `large_v`
value `bench/speed/classical_speed_main.mojo` passes to its `_sz(size,
smoke_v, shipped_v, large_v)` helper, so the shape tag the Mojo driver prints
at `large` and the tag this arm prints are the same string. ONLY THREE LANES
HAVE A `large_v` THERE (cd, kde, metrics); every other lane's Mojo `large` is
its shipped fixture, and so is it here, because a tag that did not match
would be a row that could not be paired. `wide` (DEVIATION 2133) is the
512-feature variant of every lane, rows chosen so the fixture stays under
about 4 GB of float32 and a cuML fit takes seconds; it has NO Mojo-driver
twin and pairs only with `bench/speed/classical_py_speed_arm.py`, which
consumes the same `fixture()` and therefore the same bytes.

`fixture(lane, size)` (DEVIATION 2134) is the ONE importable entry every arm
consumes, ours and the vendor's; see its docstring for the key contract.

Environment:
    MOJOLEARN_SPEED_LANE      required
    MOJOLEARN_SPEED_ARM       `default` (absent) or `deterministic`
    MOJOLEARN_SPEED_ROUNDS    timed rounds (default 5)
    MOJOLEARN_SPEED_SIZE      `shipped` (default), `smoke`, `large`, `wide`
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


def hash_arrays(lane, arm, *arrays, hash_max=None):
    """FNV-1a64 over the concatenated raw bytes, or `-` when that is a lie.

    `-` in three cases, each of which is a real statement rather than a
    shrug: an array we could not bring back to the host, an array bigger than
    `HASH_MAX` (a note is printed naming the size), and an explicit `None`
    passed by a lane that knows its output is not comparable with ours.

    `hash_max` (DEVIATION 2139) lets ONE lane raise the ceiling: the umap
    embedding is 1.6 MB at 200,000 x 2 and the per-round hash is the only
    evidence the deterministic question has, so that lane pays the ~1 s of
    pure-Python FNV per round, outside the clock, rather than print `-`.
    """
    if hash_max is None:
        hash_max = HASH_MAX
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
    if total > hash_max:
        note(lane, arm, "output is %d bytes, above MOJOLEARN_SPEED_HASH_MAX "
                        "(%d): hash omitted, not computed and discarded"
             % (total, hash_max))
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

def header(lane, arm, device, rounds, size, mode="FAST"):
    """`mode=` is FAST for a vendor's default configuration (unchanged) and
    DETERMINISTIC for its documented deterministic sibling (DEVIATION 2131);
    the `ours` umap arm passes the tier it read back from its binary.
    `tools/fast_speed_table.py` gates the mode of `ours` rows only."""
    print("FSPEED-HEADER family=%s lane=%s arm=%s mode=%s device=%s "
          "rounds=%d size=%s" % (FAMILY, lane, arm, mode, device, rounds,
                                 size),
          flush=True)


def acc(lane, arm, metric, value):
    """DEVIATION 2136. Emitted by the umap lane only; see the module
    docstring for why every other lane emits none."""
    print("FSPEED-ACC lane=%s arm=%s metric=%s value=%.6f"
          % (lane, arm, metric, value), flush=True)


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

def _parse_hex_words(lines):
    """`count` lines of eight lowercase hex digits (the Mojo driver's
    `_hex8`) to a uint32 array.

    DEVIATION 2137. The per-line `int(s, 16)` loop this replaced is fine at
    2,048 rows and is forty seconds at the `large` cd dump (32,000,000
    words). The vectorized path decodes the nibbles with a lookup table and
    produces the SAME integers; any line that is not exactly eight
    characters sends the whole block back through the original loop, so a
    dump this file has never seen is parsed the old way rather than
    misread.
    """
    stripped = [s.strip() for s in lines]
    if stripped and all(len(s) == 8 for s in stripped):
        raw = np.frombuffer("".join(stripped).encode("ascii"), dtype=np.uint8)
        lut = np.full(256, 255, dtype=np.uint8)
        for ch in "0123456789abcdef":
            lut[ord(ch)] = int(ch, 16)
        for ch in "ABCDEF":
            lut[ord(ch)] = int(ch, 16)
        nib = lut[raw]
        if not (nib == 255).any():
            nib = nib.reshape(-1, 8).astype(np.uint32)
            words = np.zeros(nib.shape[0], dtype=np.uint32)
            for j in range(8):
                words = (words << np.uint32(4)) | nib[:, j]
            return words
    words = np.empty(len(stripped), dtype=np.uint32)
    for j, s in enumerate(stripped):
        words[j] = int(s, 16)
    return words


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
                words = _parse_hex_words(lines[i:i + count])
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


def race(lane, arm, shape, rounds, size, device, call, mode="FAST",
         hash_max=None):
    """One warm-up plus `rounds` timed calls of `call`, which returns the
    outputs to hash (a tuple, or `None` for a lane whose output is not
    comparable with ours). `mode` and `hash_max` are passed through to
    `header` and `hash_arrays` (DEVIATIONS 2131, 2139); the `ours` umap arm
    in `bench/speed/umap_speed_arm.py` runs its rounds through this same
    function so the two sides' loops cannot differ."""
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
    header(lane, arm, device, rounds, size, mode=mode)
    hashes = []
    for r in range(rounds + 1):
        t0 = time.perf_counter()
        out = call()
        _sync()
        ms = (time.perf_counter() - t0) * 1000.0
        if out is None:
            h = "-"
        else:
            h = hash_arrays(lane, arm, *out, hash_max=hash_max)
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
# The vendors' deterministic siblings (DEVIATION 2131).
# ===========================================================================

def arm_selector():
    """`default` or `deterministic`, from MOJOLEARN_SPEED_ARM."""
    v = os.environ.get("MOJOLEARN_SPEED_ARM", "").strip().lower()
    if v in ("", "default"):
        return "default"
    if v == "deterministic":
        return "deterministic"
    raise SystemExit("MOJOLEARN_SPEED_ARM must be default or deterministic; "
                     "got %r" % (v,))


def deterministic():
    return arm_selector() == "deterministic"


LIVE = "LIVE"
NOT_OFFERED_KIND = "NOT-OFFERED"
BY_DEFAULT = "DETERMINISTIC-BY-DEFAULT"
NOT_DOCUMENTED = "NOT-DOCUMENTED"

_CUML = "https://docs.nvidia.com/cuml/latest/api/generated/"


class Sibling(object):
    """One lane's `-deterministic` arm: its name, its verdict, and the
    vendor's own words. Not an arm; the lane function asks
    `deterministic_gate` whether to build one. Same shape and same reason
    as `tools/speed_gbdt_arm.py::NotOffered` (DEVIATION 1890): the refusal
    is emitted from ONE place, once per lane, with a fixed prefix, and it
    is a fact about the vendor's documentation, true on every box."""

    def __init__(self, name, kind, words, url, note=None):
        self.name = name
        self.kind = kind
        self.words = words
        self.url = url
        self.note = note

    def reason(self):
        return "%s: %s (%s)" % (self.kind, self.words, self.url)


def _not_documented(name, estimator, page, extra=""):
    return Sibling(name, NOT_DOCUMENTED,
                   "cuML's %s documentation states nothing about "
                   "determinism%s" % (estimator, extra), page)


def _no_estimator(name, what):
    return Sibling(name, NOT_OFFERED_KIND,
                   "RAPIDS ships no %s, so there is no GPU arm and no "
                   "deterministic sibling of one; the CPU fallback arm is "
                   "seeded by random_state in its default configuration "
                   "and is GPU-PATH-ONLY refused on a GPU vendor's box" % what,
                   "https://docs.nvidia.com/cuml/latest/api/")


#: Keyed by LANE. Every sentence in `words` was read from `url` on
#: 2026-09-07 and is quoted, not paraphrased. The leg's log reader finds a
#: refused sibling by its `FSPEED-REFUSED ... reason=<KIND>:` prefix.
DETERMINISTIC_SIBLINGS = {
    # DEVIATION 2142. The one knob KMeans documents. With `init=` an array
    # and `n_init=1` the seed selects nothing our fixture leaves to chance,
    # and the sentence promises repeatability across Python restarts, not
    # a summation order; the arm runs so the hash column can say what it
    # actually buys.
    "kmeans": Sibling(
        "cuml-gpu-deterministic", LIVE,
        "\"random_state int or None (default = None) If you want results "
        "to be the same when you restart Python, select a state.\"; the "
        "sibling passes random_state=7 and nothing else changes",
        _CUML + "cuml.cluster.KMeans/"),
    "dbscan": _not_documented("cuml-gpu-deterministic", "DBSCAN",
                              _CUML + "cuml.cluster.DBSCAN/"),
    "pca": _not_documented("cuml-gpu-deterministic", "PCA",
                           _CUML + "cuml.decomposition.PCA/"),
    # The source, linear_regression.pyx, carries the comment "Always use 2
    # streams to expose concurrency in the eig computation"; a code comment
    # is not documentation and is recorded as a note, not as the verdict.
    "ols": _not_documented("cuml-gpu-deterministic", "LinearRegression",
                           _CUML + "cuml.linear_model.LinearRegression/"),
    "knn": _not_documented("cuml-gpu-deterministic", "NearestNeighbors",
                           _CUML + "cuml.neighbors.NearestNeighbors/"),
    "cd": _not_documented("cuml-gpu-deterministic", "Lasso",
                          _CUML + "cuml.linear_model.Lasso/"),
    "kde": _not_documented(
        "cuml-gpu-deterministic", "KernelDensity",
        _CUML + "cuml.neighbors.KernelDensity/",
        "; its only random_state is documented on sample(), which is not "
        "what is timed"),
    "linkage": _not_documented("cuml-gpu-deterministic",
                               "AgglomerativeClustering",
                               _CUML + "cuml.cluster.AgglomerativeClustering/"),
    # DEVIATION 2143. The seed exists and the vendor restricts it to a
    # path this lane does not take: `probability=False`, fit only.
    "svm": Sibling(
        "cuml-gpu-deterministic", NOT_OFFERED_KIND,
        "cuML's SVC documents its seed as \"random_state: int (default = "
        "None) Seed for random number generator (used only when "
        "probability=True).\"; this lane fits with probability=False, so "
        "no deterministic configuration is documented for what is timed",
        _CUML + "cuml.svm.SVC/"),
    "metrics": _not_documented(
        "cuml-gpu-deterministic", "metrics",
        _CUML + "cuml.metrics.trustworthiness/",
        " (accuracy_score, trustworthiness, silhouette_score and "
        "kl_divergence pages read; none mentions it)"),
    "ivf": Sibling(
        "cuvs-gpu-deterministic", NOT_DOCUMENTED,
        "cuVS's IVF-Flat Python documentation (IndexParams: n_lists, "
        "metric, kmeans_n_iters, kmeans_trainset_fraction, "
        "add_data_on_build, adaptive_centers) states nothing about "
        "determinism and takes no seed",
        "https://docs.nvidia.com/cuvs/api-reference/"
        "python-api-neighbors-ivf-flat"),
    # DEVIATION 2144. The vendor's own description of its default; the
    # fast arm's per-round hash IS this lane's deterministic column.
    "hdbscan": Sibling(
        "cuml-gpu-deterministic", BY_DEFAULT,
        "\"Note that while the algorithm is generally deterministic and "
        "should provide matching results between RAPIDS and the "
        "Scikit-learn Contrib versions, the construction of the k-nearest "
        "neighbors graph and minimum spanning tree can introduce "
        "differences between the two algorithms, especially when several "
        "nearest neighbors around a point might have the same distance.\"",
        _CUML + "cuml.cluster.hdbscan.HDBSCAN/"),
    # DEVIATION 2145. torch's documented switch is global. Its per-op
    # lists ("will act deterministically" / "will throw a RuntimeError")
    # name neither torch.linalg.cholesky nor torch.cholesky_solve, so
    # under the switch the two either run (and the hash says whether they
    # repeat) or raise (and the arm is refused with torch's message). The
    # 2.14 pages read on 2026-09-07 do not mention CUBLAS_WORKSPACE_CONFIG,
    # so it is not set here.
    "cholesky": Sibling(
        "torch-gpu-deterministic", LIVE,
        "torch.use_deterministic_algorithms(True): \"Sets whether PyTorch "
        "operations must use \"deterministic\" algorithms. That is, "
        "algorithms which, given the same input, and when run on the same "
        "software and hardware, always produce the same output. When "
        "enabled, operations will use deterministic algorithms when "
        "available, and if only nondeterministic algorithms are available "
        "they will throw a RuntimeError when called.\" Neither "
        "torch.linalg.cholesky nor torch.cholesky_solve appears in either "
        "of that page's operation lists",
        "https://docs.pytorch.org/docs/2.14/generated/"
        "torch.use_deterministic_algorithms.html"),
    "gmm": _no_estimator("cuml-gpu-deterministic", "GaussianMixture"),
    "gp": _no_estimator("cuml-gpu-deterministic",
                        "Gaussian process regressor"),
    "krr": _not_documented("cuml-gpu-deterministic", "KernelRidge",
                           _CUML + "cuml.kernel_ridge.KernelRidge/"),
    "nystroem": _no_estimator("cuml-gpu-deterministic", "Nystroem"),
    "rbfsampler": _no_estimator("cuml-gpu-deterministic",
                                "RBFSampler / random Fourier features"),
    "resample": _no_estimator("cuml-gpu-deterministic", "bootstrap"),
    "spectral": _no_estimator("cuml-gpu-deterministic",
                              "SpectralClustering estimator"),
    "holtwinters": _not_documented("cuml-gpu-deterministic",
                                   "ExponentialSmoothing",
                                   _CUML + "cuml.tsa.ExponentialSmoothing/"),
    "kpss": Sibling(
        "cuml-gpu-deterministic", NOT_DOCUMENTED,
        "cuML's API reference has no page for kpss_test; the cuml.tsa page "
        "names it only in \"cuml.tsa is deprecated in cuML 26.08 and will "
        "be removed in the cuML 26.12 release. This includes ... "
        "cuml.tsa.stationarity.kpss_test ...\" and states nothing about "
        "determinism",
        "https://docs.nvidia.com/cuml/latest/api/cuml.tsa/"),
    # DEVIATION 2135. The vendor documents the switch AND its price.
    "umap": Sibling(
        "cuml-gpu-deterministic", LIVE,
        "\"random_state int, RandomState instance or None, optional "
        "(default=None) Seed used by the random number generator for "
        "embedding initialization and optimizer sampling. Setting a "
        "random_state enables reproducible embeddings, but at the cost of "
        "slower training and increased memory usage. This is because high "
        "parallelism during optimization involves non-deterministic "
        "floating-point addition ordering. Note: Explicitly setting "
        "build_algo='nn_descent' will break reproducibility, as NN Descent "
        "produces non-deterministic KNN graphs.\"; the sibling passes "
        "random_state=7 and nothing else changes",
        _CUML + "cuml.manifold.UMAP/"),
}


def deterministic_gate(lane, base_arm):
    """`(arm_name, run)`. Under the default selector: `(base_arm, True)`,
    nothing printed. Under `deterministic`: the sibling's name, and
    whether the lane should build it. A non-LIVE sibling is refused HERE,
    once, with the vendor's words, and the lane returns without touching
    the fixture; a LIVE one gets a note carrying the same words so the
    log says what configuration the arm ran in."""
    if not deterministic():
        return base_arm, True
    sib = DETERMINISTIC_SIBLINGS[lane]
    if sib.kind != LIVE:
        refuse(lane, sib.name, sib.reason())
        return sib.name, False
    note(lane, sib.name, "vendor's documented deterministic configuration: "
                         "%s" % sib.reason())
    return sib.name, True


# ===========================================================================
# GROUP 1: the five lanes whose data is the splitmix64 recurrence.
#
# SHAPES TRANSCRIBED from `bench/bench_main.mojo`, exactly as
# `bench/bench_sklearn.py` transcribes them and exactly as
# `bench/speed/classical_speed_main.mojo` transcribes them (now in
# `_gen_<lane>` above, one table per lane, all four sizes). If that file's
# shapes move, these move.
# ===========================================================================

SIZES = ("shipped", "smoke", "large", "wide")

#: The lanes whose fixture is the splitmix64 recurrence at EVERY size
#: (group 1 plus umap, DEVIATION 2135). Every other lane reads the Mojo
#: driver's dump at `shipped`/`smoke`, and at `large` where the driver has
#: no `large_v` for it (DEVIATION 2132).
GENERATED_LANES = ("kmeans", "dbscan", "pca", "ols", "knn", "umap")

#: The three lanes the Mojo driver sizes with `_sz(size, smoke, shipped,
#: large)`. At `large` these are generated here from the recurrence when
#: no dump is present (DEVIATION 2134); when the driver's dump IS present
#: it wins, so that all three arms (Mojo ours, Python ours, vendor) hold
#: one set of bytes.
SZ_LANES = ("cd", "kde", "metrics")


def _u01_f32(rows, cols, salt, mul=1.0, add=0.0, block=65536):
    """`np.ascontiguousarray(u01(rows, cols, salt) * mul + add,
    dtype=np.float32)`, computed `block` rows at a time.

    Elementwise-identical to the one-shot expression the group-1 lanes used
    to write (the recurrence is a pure function of `(row, k, salt)` and
    `u01_at` is asserted against `u01` at startup), and needed because
    `u01(1000000, 512, s)` would hold three 4 GB uint64 intermediates at
    once. `mul` and `add` are applied in float64 BEFORE the cast, exactly
    as the old expressions did, so the bits do not move at `shipped`.
    """
    out = np.empty((rows, cols), dtype=np.float32)
    for r0 in range(0, rows, block):
        r1 = min(rows, r0 + block)
        v = u01_at(np.arange(r0, r1), cols, salt)
        if mul != 1.0:
            v = v * mul
        if add != 0.0:
            v = v + add
        out[r0:r1] = v
    return out


def _blobs_f32(rows, cols, k, salt, spread=10.0):
    """`k` hashed centers in `[0, spread)^cols`, row `i` around center
    `i % k` with a uniform `[-0.5, 0.5)` offset. The wide-tier stand-in for
    the Mojo drivers' blob fixtures (linkage, hdbscan, gmm, spectral), whose
    builders take no size."""
    centers = _u01_f32(k, cols, salt, mul=spread)
    x = _u01_f32(rows, cols, salt + 1, add=-0.5)
    labels = np.arange(rows, dtype=np.int64) % k
    x += centers[labels]
    return np.ascontiguousarray(x, dtype=np.float32)


def _pick(size, smoke_v, shipped_v, large_v, wide_v):
    """The Python twin of the Mojo driver's `_sz`, plus the `wide` tier."""
    return {"smoke": smoke_v, "shipped": shipped_v, "large": large_v,
            "wide": wide_v}[size]


# ---- per-lane generators. Each returns (arrays, tag, params). ------------

def _gen_kmeans(size):
    rows = _pick(size, 40000, 4000000, 4000000, 1000000)
    cols = _pick(size, 32, 32, 32, 512)
    k, iters = 64, 20
    x = _u01_f32(rows, cols, 0, mul=10.0)
    # THE INITIAL CENTROIDS ARE ROWS `c * 7919`, NOT ROW 0.
    # `bench/bench_main.mojo:106` seeds centroid `c` from `_u01(c * 7919, f, 5)`.
    # `bench/bench_sklearn.py:164` writes `u01(c * 7919 + 1, km_cols, 5)[0]`,
    # which indexes row ZERO of that block for every `c` and therefore hands
    # scikit-learn 64 IDENTICAL centroids. That is a defect in that file, it
    # is reported as DEVIATION 1810, and it is NOT reproduced here: this arm
    # takes the row the Mojo side actually uses.
    init = np.ascontiguousarray(
        u01_at(np.arange(k) * 7919, cols, 5) * 10.0, dtype=np.float32)
    tag = "%dx%dk%di%d" % (rows, cols, k, iters)
    return ({"x": x, "init": init}, tag,
            {"n_clusters": k, "max_iter": iters, "tol": 1e-7, "n_init": 1,
             "rows": rows, "cols": cols})


def _gen_dbscan(size):
    rows = _pick(size, 512, 4000, 4000, 20000)
    cols = _pick(size, 16, 16, 16, 512)
    x = _u01_f32(rows, cols, 4, mul=2.0)
    # DEVIATION 2140. eps 0.35 is the shipped value in 16 dimensions. In 512
    # dimensions the pairwise distance of two points of this fixture is
    # 18.5 +- 0.5 (512 iid coordinate differences of variance 2/3), so at
    # 0.35 every point is noise and the BFS half of the algorithm never
    # runs. 17.0 is three standard deviations below the mean: about 0.13%
    # of pairs are neighbors, ~27 per point, so the expansion is exercised.
    eps = 0.35 if cols == 16 else 17.0
    return ({"x": x}, "%dx%d" % (rows, cols),
            {"eps": eps, "min_samples": 5, "rows": rows, "cols": cols})


def _gen_pca(size):
    rows = _pick(size, 40000, 4000000, 4000000, 1000000)
    cols = _pick(size, 32, 32, 32, 512)
    comp = 8
    x = _u01_f32(rows, cols, 3, mul=4.0)
    return ({"x": x}, "%dx%dc%d" % (rows, cols, comp),
            {"n_components": comp, "rows": rows, "cols": cols})


def _gen_ols(size):
    rows = _pick(size, 40000, 4000000, 4000000, 1000000)
    cols = _pick(size, 32, 32, 32, 512)
    x = _u01_f32(rows, cols, 6, add=-0.5)
    w = 1.0 + 0.1 * np.arange(cols)
    # The full-matrix product, not a chunked one: BLAS may sum a chunked
    # product in a different order and this line is required to keep the
    # `shipped` bytes exactly where they were.
    y = np.ascontiguousarray(x @ w, dtype=np.float32)
    return ({"x": x, "y": y}, "%dx%d" % (rows, cols),
            {"fit_intercept": 0, "algorithm": "eig", "rows": rows,
             "cols": cols})


def _gen_knn(size):
    index = _pick(size, 40000, 400000, 400000, 200000)
    queries = _pick(size, 400, 4000, 4000, 20000)
    cols = _pick(size, 32, 32, 32, 512)
    k = 10
    idx = _u01_f32(index, cols, 1)
    qry = _u01_f32(queries, cols, 2)
    return ({"index": idx, "queries": qry},
            "%dx%dq%dk%d" % (index, cols, queries, k),
            {"k": k, "n_index": index, "n_queries": queries, "cols": cols})


def _gen_umap(size):
    """DEVIATION 2135. `large` is the campaign's 200,000 x 64; `wide` is
    100,000 x 512; smoke 5,000 x 16. Salt 7 is new (no other lane uses it)
    so this fixture shares no bytes with any other lane's."""
    rows = _pick(size, 5000, 200000, 200000, 100000)
    cols = _pick(size, 16, 64, 64, 512)
    x = _u01_f32(rows, cols, 7)
    p = {"n_neighbors": 15, "n_components": 2, "min_dist": 0.1,
         "n_epochs": 200, "metric": "euclidean", "rows": rows, "cols": cols,
         "trust_rows": 10000, "deterministic_random_state": 7}
    tag = "%dx%dn%dc%de%d" % (rows, cols, p["n_neighbors"],
                              p["n_components"], p["n_epochs"])
    return ({"x": x}, tag, p)


def _gen_cd(size):
    """Lasso on a planted sparse signal. The Mojo `large_v` shape is
    1,000,000 x 32 (`fixture_planted_sparse(n, d, 610)`); the bytes here
    are this file's recurrence with salt 610, NOT that builder's, and the
    `_source` key says which one a run held."""
    n = _pick(size, 256, 2048, 1000000, 1000000)
    d = _pick(size, 4, 16, 32, 512)
    x = _u01_f32(n, d, 610, add=-0.5)
    w = np.zeros(d, dtype=np.float64)
    w[::4] = 1.0 + 0.25 * np.arange(0, d, 4)
    noise = _u01_f32(n, 1, 611, add=-0.5)[:, 0].astype(np.float64) * 0.01
    y = np.ascontiguousarray(x @ w + noise, dtype=np.float32)
    return ({"x": x, "y": y}, "%dx%d" % (n, d),
            {"n": n, "d": d, "alpha": 0.01, "l1_ratio": 1.0,
             "max_iter": 1000, "tol": 1e-3, "fit_intercept": 1})


def _gen_kde(size):
    n_train = _pick(size, 128, 1024, 200000, 100000)
    n_query = _pick(size, 32, 256, 20000, 10000)
    d = _pick(size, 8, 8, 8, 512)
    train = _u01_f32(n_train, d, 1)
    query = _u01_f32(n_query, d, 2)
    weights = np.ascontiguousarray(
        _u01_f32(n_train, 1, 3, add=0.5)[:, 0], dtype=np.float32)
    # DEVIATION 2141. 2.75 is the shipped bandwidth at d = 8; it is scaled
    # by sqrt(d / 8) so the kernel sees the same fraction of the typical
    # pairwise distance in 512 dimensions (2.75 * 8 = 22.0).
    bandwidth = 2.75 * float(np.sqrt(d / 8.0))
    return ({"train": train, "query": query, "weights": weights},
            "%dx%dx%d" % (n_train, n_query, d),
            {"n_train": n_train, "n_query": n_query, "d": d,
             "bandwidth": bandwidth, "kernel": "gaussian",
             "metric": "euclidean"})


def _gen_metrics(size):
    n_lab = _pick(size, 257, 2053, 2000000, 2000000)
    n_flt = _pick(size, 257, 2053, 2000000, 2000000)
    n_sil = _pick(size, 67, 521, 100000, 100000)
    n_tru = _pick(size, 61, 301, 50000, 50000)
    d_sil = _pick(size, 4, 4, 4, 512)
    m_tru = _pick(size, 6, 6, 6, 512)
    k_sil, d_tru, k_tru, n_true, n_pred = 5, 2, 5, 6, 5
    y_true = np.floor(_u01_f32(n_lab, 1, 4099)[:, 0].astype(np.float64)
                      * n_true).astype(np.int32)
    keep = _u01_f32(n_lab, 1, 4100)[:, 0] < 0.66
    rnd = np.floor(_u01_f32(n_lab, 1, 4101)[:, 0].astype(np.float64)
                   * n_pred).astype(np.int32)
    y_pred = np.where(keep, np.minimum(y_true, n_pred - 1), rnd).astype(np.int32)
    y = _u01_f32(n_flt, 1, 17, mul=12.0, add=-6.0)[:, 0].copy()
    res = _u01_f32(n_flt, 1, 18, mul=11.0, add=-6.0)[:, 0]
    y_hat = np.ascontiguousarray(y + res, dtype=np.float32)
    p = _u01_f32(n_flt, 1, 19, add=0.01)[:, 0].astype(np.float64)
    q = _u01_f32(n_flt, 1, 20, add=0.01)[:, 0].astype(np.float64)
    p = np.ascontiguousarray(p / p.sum(), dtype=np.float32)
    q = np.ascontiguousarray(q / q.sum(), dtype=np.float32)
    sil_x = _blobs_f32(n_sil, d_sil, k_sil, 23)
    sil_labels = (np.arange(n_sil, dtype=np.int64) % k_sil).astype(np.int32)
    trust_x = _blobs_f32(n_tru, m_tru, 4, 29)
    # The Mojo driver's own formula for the embedding: the first two
    # coordinates plus `(u01(i, qq, 38) - 0.5) * 0.8`.
    trust_emb = np.ascontiguousarray(
        trust_x[:, :d_tru] + _u01_f32(n_tru, d_tru, 38, mul=0.8, add=-0.4),
        dtype=np.float32)
    tag = "lab%d.flt%d.sil%dx%d.tru%dx%d" % (n_lab, n_flt, n_sil, d_sil,
                                             n_tru, m_tru)
    return ({"y_true": y_true, "y_pred": y_pred, "y": y, "y_hat": y_hat,
             "p": p, "q": q, "sil_x": sil_x, "sil_labels": sil_labels,
             "trust_x": trust_x, "trust_emb": trust_emb},
            tag,
            {"n_labels": n_lab, "n_float": n_flt, "n_sil": n_sil,
             "d_sil": d_sil, "k_sil": k_sil, "n_trust": n_tru,
             "m_trust": m_tru, "d_trust": d_tru, "k_trust": k_tru})


def _gen_ivf(size):
    n_rows, n_q, dim = 200000, 2000, 512
    n_lists, n_probes, k, iters = 256, 16, 10, 20
    return ({"index": _u01_f32(n_rows, dim, 1),
             "queries": _u01_f32(n_q, dim, 2)},
            "%dx%dq%dL%dp%dk%d" % (n_rows, dim, n_q, n_lists, n_probes, k),
            {"n_rows": n_rows, "n_queries": n_q, "dim": dim,
             "n_lists": n_lists, "n_probes": n_probes, "k": k,
             "kmeans_n_iters": iters, "metric": "sqeuclidean", "seed": 0})


def _gen_linkage(size):
    n, d, k = 10000, 512, 3
    return ({"x": _blobs_f32(n, d, k, 30)}, "wide.%dx%d" % (n, d),
            {"n": n, "d": d, "n_clusters": k, "fixture": "wide"})


def _gen_svm(size):
    n, d = 20000, 512
    x = _u01_f32(n, d, 40, add=-0.5)
    w = np.where(np.arange(d) % 2 == 0, 1.0, -1.0)
    y = np.where(x @ w >= 0.0, 1.0, -1.0).astype(np.float32)
    return ({"x": x, "y": y}, "wide.%dx%d" % (n, d),
            {"fixture": "wide", "n": n, "d": d, "C": 10.0,
             "gamma": 1.0 / d, "tol": 1e-3, "kernel": "rbf",
             "nochange_steps": 1000})


def _gen_hdbscan(size):
    n, d = 10000, 512
    return ({"x": _blobs_f32(n, d, 5, 45)}, "wide.%dx%d" % (n, d),
            {"fixture": "wide", "n": n, "d": d, "min_samples": 5,
             "min_cluster_size": 5, "metric": "euclidean",
             "cluster_selection_method": "eom"})


def _gen_cholesky(size):
    """An RBF Gram matrix of 4,096 hashed points in 8 dimensions, which is
    SPD for distinct points, plus a planted solution. n x n is the knob
    here, not a feature count."""
    n, nrhs, jitter = 4096, 4, 1e-3
    pts = _u01_f32(n, 8, 50, mul=4.0).astype(np.float64)
    sq_norm = (pts * pts).sum(axis=1)
    sq = np.maximum(sq_norm[:, None] + sq_norm[None, :] - 2.0 * (pts @ pts.T),
                    0.0)
    a = np.exp(-sq / 2.0)
    x_planted = _u01_f32(n, nrhs, 51, add=-0.5).astype(np.float64)
    b = (a + jitter * np.eye(n)) @ x_planted
    return ({"a": np.ascontiguousarray(a, dtype=np.float32),
             "b": np.ascontiguousarray(b, dtype=np.float32)},
            "wide.%dx%dr%d" % (n, n, nrhs),
            {"fixture": "wide", "n": n, "nrhs": nrhs, "jitter": jitter})


def _gen_gmm(size):
    n, d, k = 100000, 512, 3
    return ({"x": _blobs_f32(n, d, k, 55)}, "wide.%dx%dK%d" % (n, d, k),
            {"fixture": "wide", "n": n, "d": d, "n_components": k,
             "max_iter": 50, "tol": 1e-3, "reg_covar": 1e-6,
             "covariance_type": "full", "init_params": "kmeans",
             "random_state": 0})


def _gen_gp(size):
    n, d, ns = 8000, 512, 1000
    x = _u01_f32(n, d, 60)
    w = _u01_f32(d, 1, 61, add=-0.5)[:, 0].astype(np.float64)
    y = np.ascontiguousarray(x @ w, dtype=np.float32)
    xs = _u01_f32(ns, d, 62)
    ls = _u01_f32(d, 1, 63, mul=7.0, add=0.5)[:, 0].copy()
    return ({"x": x, "y": y, "x_star": xs, "length_scale": ls},
            "wide.%dx%ds%d" % (n, d, ns),
            {"fixture": "wide", "n_train": n, "n_star": ns, "d": d,
             "alpha": 0.1, "kernel": "rbf_ard"})


def _gen_km(lane, size):
    n = {"krr": 10000, "nystroem": 100000, "rbfsampler": 100000}[lane]
    d, nq, q = 512, {"krr": 1000, "nystroem": 10000,
                     "rbfsampler": 10000}[lane], 256
    x = _u01_f32(n, d, 70)
    w = _u01_f32(d, 1, 71, add=-0.5)[:, 0].astype(np.float64)
    y = np.ascontiguousarray(x @ w, dtype=np.float32)
    xq = _u01_f32(nq, d, 72)
    if lane == "krr":
        tag = "wide.%dx%d" % (n, d)
    elif lane == "nystroem":
        tag = "wide.%dx%dq%d" % (n, d, q)
    else:
        tag = "%dx%dq%d" % (nq, d, q)
    return ({"x": x, "y": y, "x_query": xq}, tag,
            {"fixture": "wide", "n": n, "d": d, "n_query": nq,
             "n_components": q, "gamma": 1.0 / d, "alpha": 0.5,
             "kernel": "rbf", "seed": 20260825})


def _gen_resample(size):
    n, d, n_res = 200000, 512, 10000
    return ({"x": _u01_f32(n, d, 80)}, "%dx%dr%d" % (n, d, n_res),
            {"n": n, "d": d, "n_resamples": n_res, "seed": 20260825,
             "confidence_level": 0.95, "statistic": "mean",
             "method": "percentile", "alternative": "two-sided"})


def _gen_spectral(size):
    n, d = 21000, 512
    return ({"x": _blobs_f32(n, d, 3, 85)}, "%dx%dc3" % (n, d),
            {"n": n, "d": d, "n_clusters": 3, "n_components": 3,
             "n_init": 10, "n_neighbors": 10, "tolerance": 1e-5, "seed": 7})


def _gen_holtwinters(size):
    batch, n, freq = 512, 1200, 12
    t = np.arange(n, dtype=np.float64)[None, :]
    y = (10.0 + 0.01 * t + 2.0 * np.sin(2.0 * np.pi * t / freq)
         + _u01_f32(batch, n, 90, add=-0.5).astype(np.float64))
    return ({"y": np.ascontiguousarray(y, dtype=np.float32)},
            "%dx%df%d" % (batch, n, freq),
            {"n": n, "batch_size": batch, "frequency": freq,
             "start_periods": 2, "seasonal": "additive", "eps": 2.24e-3,
             "layout": "series-major (batch_size x n)"})


def _gen_kpss(size):
    batch, n_obs = 512, 5200
    steps = _u01_f32(batch, n_obs, 95, add=-0.5).astype(np.float64)
    y = np.cumsum(steps, axis=1)
    return ({"y": np.ascontiguousarray(y, dtype=np.float32)},
            "%dx%d" % (batch, n_obs),
            {"n_obs": n_obs, "batch_size": batch, "d": 1, "D": 0, "s": 0,
             "pval_threshold": 0.05,
             "layout": "series-major (batch_size x n_obs)"})


GENERATORS = {
    "kmeans": _gen_kmeans, "dbscan": _gen_dbscan, "pca": _gen_pca,
    "ols": _gen_ols, "knn": _gen_knn, "umap": _gen_umap, "cd": _gen_cd,
    "kde": _gen_kde, "metrics": _gen_metrics, "ivf": _gen_ivf,
    "linkage": _gen_linkage, "svm": _gen_svm, "hdbscan": _gen_hdbscan,
    "cholesky": _gen_cholesky, "gmm": _gen_gmm, "gp": _gen_gp,
    "krr": lambda s: _gen_km("krr", s),
    "nystroem": lambda s: _gen_km("nystroem", s),
    "rbfsampler": lambda s: _gen_km("rbfsampler", s),
    "resample": _gen_resample, "spectral": _gen_spectral,
    "holtwinters": _gen_holtwinters, "kpss": _gen_kpss,
}

#: How a dump's arrays are reshaped and its params typed, per lane, so the
#: dump path and the generated path hand back the SAME keys with the same
#: dtypes and ranks. `("x", ("n", "d"))` reshapes array `x` to
#: `(params["n"], params["d"])`; a bare name stays one-dimensional.
DUMP_SHAPES = {
    "cd": [("x", ("n", "d")), "y"],
    "kde": [("train", ("n_train", "d")), ("query", ("n_query", "d")),
            "weights"],
    "linkage": [("x", ("n", "d"))],
    "svm": [("x", ("n", "d")), "y"],
    "metrics": ["y_true", "y_pred", "y", "y_hat", "p", "q",
                ("sil_x", ("n_sil", "d_sil")), "sil_labels",
                ("trust_x", ("n_trust", "m_trust")),
                ("trust_emb", ("n_trust", "d_trust"))],
    "ivf": [("index", ("n_rows", "dim")), ("queries", ("n_queries", "dim"))],
    "hdbscan": [("x", ("n", "d"))],
    "cholesky": [("a", ("n", "n")), ("b", ("n", "nrhs"))],
    "gmm": [("x", ("n", "d"))],
    "gp": [("x", ("n_train", "d")), "y", ("x_star", ("n_star", "d")),
           "length_scale"],
    "krr": [("x", ("n", "d")), "y", ("x_query", ("n_query", "d"))],
    "nystroem": [("x", ("n", "d")), "y", ("x_query", ("n_query", "d"))],
    "rbfsampler": [("x", ("n", "d")), "y", ("x_query", ("n_query", "d"))],
    "resample": [("x", ("n", "d"))],
    "spectral": [("x", ("n", "d"))],
    "holtwinters": [("y", ("batch_size", "n"))],
    "kpss": [("y", ("batch_size", "n_obs"))],
}

DUMP_INT_KEYS = ("n", "d", "n_train", "n_query", "n_star", "n_clusters",
                 "n_components", "n_labels", "n_float", "n_sil", "d_sil",
                 "k_sil", "n_trust", "m_trust", "d_trust", "k_trust",
                 "n_rows", "n_queries", "dim", "n_lists", "n_probes", "k",
                 "kmeans_n_iters", "seed", "nrhs", "max_iter",
                 "fit_intercept", "min_samples", "min_cluster_size",
                 "random_state", "n_resamples", "n_init", "n_neighbors",
                 "batch_size", "frequency", "start_periods", "n_obs", "D",
                 "s", "nochange_steps")
DUMP_FLOAT_KEYS = ("alpha", "l1_ratio", "tol", "bandwidth", "C", "gamma",
                   "jitter", "reg_covar", "confidence_level", "tolerance",
                   "eps", "pval_threshold")

#: The shape tag the Mojo driver prints for a dumped lane, from the dump's
#: own params, so the two arms print the same string by construction.
DUMP_TAGS = {
    "cd": lambda p: "%dx%d" % (p["n"], p["d"]),
    "kde": lambda p: "%dx%dx%d" % (p["n_train"], p["n_query"], p["d"]),
    "linkage": lambda p: "%s.%dx%d" % (p["fixture"], p["n"], p["d"]),
    "svm": lambda p: "%s.%dx%d" % (p["fixture"], p["n"], p["d"]),
    "metrics": lambda p: "lab%d.flt%d.sil%dx%d.tru%dx%d" % (
        p["n_labels"], p["n_float"], p["n_sil"], p["d_sil"], p["n_trust"],
        p["m_trust"]),
    "ivf": lambda p: "%dx%dq%dL%dp%dk%d" % (
        p["n_rows"], p["dim"], p["n_queries"], p["n_lists"], p["n_probes"],
        p["k"]),
    "hdbscan": lambda p: "%s.%dx%d" % (p["fixture"], p["n"], p["d"]),
    "cholesky": lambda p: "%s.%dx%dr%d" % (p["fixture"], p["n"], p["n"],
                                           p["nrhs"]),
    "gmm": lambda p: "%s.%dx%dK%d" % (p["fixture"], p["n"], p["d"],
                                      p["n_components"]),
    "gp": lambda p: "%s.%dx%ds%d" % (p["fixture"], p["n_train"], p["d"],
                                     p["n_star"]),
    "krr": lambda p: "%s.%dx%d" % (p["fixture"], p["n"], p["d"]),
    "nystroem": lambda p: "%s.%dx%dq%d" % (p["fixture"], p["n"], p["d"],
                                           p["n_components"]),
    "rbfsampler": lambda p: "%dx%dq%d" % (p["n_query"], p["d"],
                                          p["n_components"]),
    "resample": lambda p: "%dx%dr%d" % (p["n"], p["d"], p["n_resamples"]),
    "spectral": lambda p: "%dx%dc%d" % (p["n"], p["d"], p["n_clusters"]),
    "holtwinters": lambda p: "%dx%df%d" % (p["batch_size"], p["n"],
                                           p["frequency"]),
    "kpss": lambda p: "%dx%d" % (p["batch_size"], p["n_obs"]),
}


def _from_dump(lane):
    fx = Fixture(lane)
    params = {}
    for k, v in fx.params.items():
        if k in DUMP_INT_KEYS:
            params[k] = int(float(v))
        elif k in DUMP_FLOAT_KEYS:
            params[k] = float(v)
        else:
            params[k] = v
    arrays = {}
    for spec in DUMP_SHAPES[lane]:
        name, shape = (spec, None) if isinstance(spec, str) else spec
        a = fx.arrays[name]
        if shape is not None:
            a = a.reshape(*[params[s] for s in shape])
        arrays[name] = np.ascontiguousarray(a)
    params["_source"] = "mojo-dump"
    params["_path"] = fx.path
    return arrays, DUMP_TAGS[lane](params), params


def _dump_present(lane):
    d = os.environ.get("MOJOLEARN_SPEED_DUMP", "")
    return bool(d) and os.path.exists(os.path.join(d, "%s.fixture" % lane))


def fixture(lane, size):
    """THE ONE FIXTURE ENTRY, for every arm of every lane (DEVIATION 2134).

    Returns `(arrays, tag, params)`:

      arrays  dict of C-contiguous numpy arrays, float32 or int32, by the
              names the lane's Mojo dump already uses:
                kmeans      x (rows, cols), init (k, cols)
                dbscan      x (rows, cols)
                pca         x (rows, cols)
                ols         x (rows, cols), y (rows,)
                knn         index (n_index, cols), queries (n_queries, cols)
                umap        x (rows, cols)
                cd          x (n, d), y (n,)
                kde         train (n_train, d), query (n_query, d),
                            weights (n_train,)
                linkage     x (n, d)
                svm         x (n, d), y (n,) in {-1, +1} (wide) or the
                            dump's labels
                metrics     y_true, y_pred (int32, n_labels), y, y_hat
                            (n_float), p, q (n_float), sil_x (n_sil, d_sil),
                            sil_labels (int32), trust_x (n_trust, m_trust),
                            trust_emb (n_trust, d_trust)
                ivf         index (n_rows, dim), queries (n_queries, dim)
                hdbscan     x (n, d)
                cholesky    a (n, n), b (n, nrhs)
                gmm         x (n, d)
                gp          x (n_train, d), y, x_star (n_star, d),
                            length_scale (d,)
                krr, nystroem, rbfsampler
                            x (n, d), y (n,), x_query (n_query, d)
                resample    x (n, d)
                spectral    x (n, d)
                holtwinters y (batch_size, n), series-major
                kpss        y (batch_size, n_obs), series-major
      tag     the shape string EXACTLY as the vendor arm prints it in
              `shape=`, which at `shipped`/`smoke`/`large` is exactly what
              `bench/speed/classical_speed_main.mojo` prints for the lane.
      params  the hyperparameters the vendor arm passes, by the dump's
              names (see the generators above), plus two meta keys:
              `_source` in {"splitmix64", "mojo-dump"} and, for a dump,
              `_path`.

    WHERE THE BYTES COME FROM. kmeans, dbscan, pca, ols, knn and umap are
    generated from the splitmix64 recurrence at every size (the same
    recurrence, the same salts, as the Mojo driver). Every other lane reads
    the Mojo driver's dump at `shipped` and `smoke` (unchanged: no dump is
    a refusal, never invented data), and ALSO at `large` unless it is one
    of the three lanes (cd, kde, metrics) the driver sizes with `_sz`; for
    those three at `large`, and for every dumped lane at `wide`, the arrays
    are generated here. A present dump always wins over generation at
    `large`, so that a leg which ran the Mojo driver first hands all three
    arms one set of bytes; `_source` records which happened.

    Everything the other arm needs to be the same problem is in `params`;
    an arm that reads a number from anywhere else is not comparable.
    """
    if size not in SIZES:
        raise ValueError("size must be one of %s, got %r" % (SIZES, size))
    if lane not in GENERATORS:
        raise ValueError("unknown lane %r" % (lane,))
    # An importer of this function (bench/speed/umap_speed_arm.py,
    # bench/speed/classical_py_speed_arm.py) never passes through `main`,
    # so the recurrence self-check runs here too; it is one 37 x 11 block.
    _check_u01_variant()
    generate = (lane in GENERATED_LANES or size == "wide"
                or (size == "large" and lane in SZ_LANES
                    and not _dump_present(lane)))
    if generate:
        arrays, tag, params = GENERATORS[lane](size)
        params["_source"] = "splitmix64"
        return arrays, tag, params
    return _from_dump(lane)


def trustworthiness_subsample(x, embedding, n_neighbors, n_rows=10000):
    """DEVIATION 2136. `sklearn.manifold.trustworthiness` on the FIRST
    `n_rows` rows of the data and of the embedding, the same rows for every
    arm. Returns `(metric_name, value)` or `(None, reason)` when
    scikit-learn is not importable. The first rows rather than a random
    draw because the fixture is hashed (row order carries no structure) and
    because a subsample that needs a seed is a subsample two arms can
    disagree about. cuML's own `trustworthiness` is deliberately not the
    scorer: one scorer for both sides, or the accuracy column compares two
    scorers rather than two embeddings."""
    try:
        from sklearn.manifold import trustworthiness         # noqa: PLC0415
    except Exception as e:                                   # noqa: BLE001
        return None, "scikit-learn is not importable (%r)" % (e,)
    xs = np.ascontiguousarray(to_numpy(x)[:n_rows], dtype=np.float32)
    es = np.ascontiguousarray(to_numpy(embedding)[:n_rows], dtype=np.float32)
    return ("trustworthiness_k%d_n%d" % (n_neighbors, xs.shape[0]),
            float(trustworthiness(xs, es, n_neighbors=n_neighbors)))


def _mode():
    return "DETERMINISTIC" if deterministic() else "FAST"


def _source_note(lane, arm, size, tag, prm):
    """Printed at `large` and `wide` only, so the `shipped` and `smoke`
    output is byte for byte what it was (DEVIATION 2132)."""
    if size in ("large", "wide"):
        note(lane, arm, "size=%s fixture_source=%s shape=%s"
             % (size, prm["_source"], tag))


def lane_kmeans(rounds, size, smoke):
    lane = "kmeans"
    arm, run = deterministic_gate(lane, "cuml-gpu")
    if not run:
        return
    try:
        from cuml.cluster import KMeans
    except Exception as e:
        refuse(lane, arm, "import failed: %r" % (e,))
        return
    arrays, tag, prm = fixture(lane, size)
    _source_note(lane, arm, size, tag, prm)
    # ON THE DEVICE BEFORE THE CLOCK, matching our arm. At 4,000,000 x 32
    # this array is 512 MB and it was being transferred inside every timed
    # round on their side only.
    X, init = to_device(lane, arm, arrays["x"], arrays["init"])
    extra = {"random_state": 7} if deterministic() else {}
    km = None

    def call():
        nonlocal km
        # Constructed inside the clock because cuml.KMeans does no device work
        # in __init__; fit is where everything happens. Matching our arm,
        # which also re-uploads its initial centroids each round.
        km = KMeans(n_clusters=prm["n_clusters"], init=init,
                    n_init=prm["n_init"], max_iter=prm["max_iter"],
                    tol=prm["tol"], output_type="cupy", **extra)
        km.fit(X)
        return (km.cluster_centers_, km.labels_)

    race(lane, arm, tag, rounds, size, gpu_device_name(), call, mode=_mode())


def lane_dbscan(rounds, size, smoke):
    lane = "dbscan"
    arm, run = deterministic_gate(lane, "cuml-gpu")
    if not run:
        return
    try:
        from cuml.cluster import DBSCAN
    except Exception as e:
        refuse(lane, arm, "import failed: %r" % (e,))
        return
    arrays, tag, prm = fixture(lane, size)
    _source_note(lane, arm, size, tag, prm)
    X = arrays["x"]
    note(lane, arm, "calc_core_sample_indices=False: our dbscan_fit_impl "
                    "returns labels only, and computing their core-sample "
                    "index array would be work our arm does not do")

    def call():
        # eps and min_samples are passed on both sides; nothing is left to a
        # default that could differ between the two libraries.
        db = DBSCAN(eps=prm["eps"], min_samples=prm["min_samples"],
                    calc_core_sample_indices=False, output_type="numpy")
        db.fit(X)
        return (db.labels_,)

    race(lane, arm, tag, rounds, size, gpu_device_name(), call, mode=_mode())


def lane_pca(rounds, size, smoke):
    lane = "pca"
    arm, run = deterministic_gate(lane, "cuml-gpu")
    if not run:
        return
    try:
        from cuml.decomposition import PCA
    except Exception as e:
        refuse(lane, arm, "import failed: %r" % (e,))
        return
    arrays, tag, prm = fixture(lane, size)
    _source_note(lane, arm, size, tag, prm)
    X = arrays["x"]
    s = {"pca_cols": prm["cols"], "pca_comp": prm["n_components"]}

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

    race(lane, arm, tag, rounds, size, gpu_device_name(), call, mode=_mode())


def lane_ols(rounds, size, smoke):
    lane = "ols"
    arm, run = deterministic_gate(lane, "cuml-gpu")
    if not run:
        return
    try:
        from cuml.linear_model import LinearRegression
    except Exception as e:
        refuse(lane, arm, "import failed: %r" % (e,))
        return
    arrays, tag, prm = fixture(lane, size)
    _source_note(lane, arm, size, tag, prm)
    A, b = arrays["x"], arrays["y"]
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

    race(lane, arm, tag, rounds, size, gpu_device_name(), call, mode=_mode())


def lane_knn(rounds, size, smoke):
    lane = "knn"
    arm, run = deterministic_gate(lane, "cuml-gpu")
    if not run:
        return
    try:
        from cuml.neighbors import NearestNeighbors
    except Exception as e:
        refuse(lane, arm, "import failed: %r" % (e,))
        return
    arrays, tag, prm = fixture(lane, size)
    _source_note(lane, arm, size, tag, prm)
    idx, qry = arrays["index"], arrays["queries"]
    # THE INDEX BUILD IS OUTSIDE THE CLOCK because our timed region is a
    # search, not a build: `brute_force_knn_impl` has no index to build. The
    # two `compute_norms` calls ARE inside ours, and cuML's `kneighbors`
    # computes its norms inside itself, so the two regions cover the same
    # work.
    # Symmetric with the other lanes even though the asymmetry here ran
    # AGAINST them: `qry` is only 4,000 x 32 and the transfer was inside
    # their clock, so our 21.6x loss on this row was if anything understated.
    idx, qry = to_device(lane, arm, idx, qry)
    nn = NearestNeighbors(n_neighbors=prm["k"], algorithm="brute",
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

    race(lane, arm, tag, rounds, size, gpu_device_name(), call, mode=_mode())


# ===========================================================================
# GROUP 2: the dumped-fixture lanes.
# ===========================================================================

def lane_cd(rounds, size, smoke):
    lane = "cd"
    arm, run = deterministic_gate(lane, "cuml-gpu")
    if not run:
        return
    try:
        from cuml.linear_model import Lasso
        arrays, tag, prm = fixture(lane, size)
    except Exception as e:
        refuse(lane, arm, "%r" % (e,))
        return
    _source_note(lane, arm, size, tag, prm)
    X, y = arrays["x"], arrays["y"]
    alpha, max_iter, tol = prm["alpha"], prm["max_iter"], prm["tol"]
    note(lane, arm, "alpha, max_iter, tol, fit_intercept and selection are "
                    "all passed explicitly; cuML's tol default is 1e-3 and "
                    "ours is the fixture's, and they are made equal here")

    def call():
        m = Lasso(alpha=alpha, fit_intercept=True, max_iter=max_iter,
                  tol=tol, selection="cyclic", output_type="numpy")
        m.fit(X, y)
        return (m.coef_,)

    race(lane, arm, tag, rounds, size, gpu_device_name(), call, mode=_mode())


def lane_kde(rounds, size, smoke):
    lane = "kde"
    arm, run = deterministic_gate(lane, "cuml-gpu")
    if not run:
        return
    try:
        from cuml.neighbors import KernelDensity
        arrays, tag, prm = fixture(lane, size)
    except Exception as e:
        refuse(lane, arm, "%r" % (e,))
        return
    _source_note(lane, arm, size, tag, prm)
    train, query, weights = arrays["train"], arrays["query"], arrays["weights"]
    bw = prm["bandwidth"]
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

    race(lane, arm, tag, rounds, size, gpu_device_name(), call, mode=_mode())


def lane_linkage(rounds, size, smoke):
    lane = "linkage"
    arm, run = deterministic_gate(lane, "cuml-gpu")
    if not run:
        return
    try:
        from cuml.cluster import AgglomerativeClustering
        arrays, tag, prm = fixture(lane, size)
    except Exception as e:
        refuse(lane, arm, "%r" % (e,))
        return
    _source_note(lane, arm, size, tag, prm)
    k = prm["n_clusters"]
    X = arrays["x"]
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

    race(lane, arm, tag, rounds, size, gpu_device_name(), call, mode=_mode())


def lane_svm(rounds, size, smoke):
    lane = "svm"
    arm, run = deterministic_gate(lane, "cuml-gpu")
    if not run:
        return
    try:
        from cuml.svm import SVC
        arrays, tag, prm = fixture(lane, size)
    except Exception as e:
        refuse(lane, arm, "%r" % (e,))
        return
    _source_note(lane, arm, size, tag, prm)
    X, y = arrays["x"], arrays["y"]
    C, gamma, tol = prm["C"], prm["gamma"], prm["tol"]
    nochange = prm["nochange_steps"]
    note(lane, arm, "FIT ONLY. C, gamma, tol and nochange_steps are passed "
                    "explicitly; cache_size is 0 on our side and left at "
                    "cuML's default here because a cache size is a memory "
                    "policy, not a parameter of the answer")

    def call():
        m = SVC(kernel="rbf", C=C, gamma=gamma, tol=tol,
                nochange_steps=nochange, output_type="numpy")
        m.fit(X, y)
        return (m.dual_coef_, m.support_)

    race(lane, arm, tag, rounds, size, gpu_device_name(), call, mode=_mode())


def lane_metrics(rounds, size, smoke):
    lane = "metrics"
    arm, run = deterministic_gate(lane, "cuml-gpu")
    if not run:
        return
    try:
        arrays, tag, prm = fixture(lane, size)
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
    _source_note(lane, arm, size, tag, prm)
    k_tru = prm["k_trust"]
    yt, yp = arrays["y_true"], arrays["y_pred"]
    y, yhat = arrays["y"], arrays["y_hat"]
    p, q = arrays["p"], arrays["q"]
    sil_x, sil_l = arrays["sil_x"], arrays["sil_labels"]
    tr_x, tr_e = arrays["trust_x"], arrays["trust_emb"]
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

    race(lane, arm, tag, rounds, size, gpu_device_name(), call, mode=_mode())


def lane_ivf(rounds, size, smoke):
    lane = "ivf"
    arm, run = deterministic_gate(lane, "cuvs-gpu")
    if not run:
        return
    try:
        import cupy as cp
        from cuvs.neighbors import ivf_flat
        arrays, tag, prm = fixture(lane, size)
    except Exception as e:
        refuse(lane, arm, "import failed: %r" % (e,))
        return
    _source_note(lane, arm, size, tag, prm)
    n_lists, n_probes, k = prm["n_lists"], prm["n_probes"], prm["k"]
    kiters = prm["kmeans_n_iters"]
    dataset = cp.asarray(arrays["index"])
    queries = cp.asarray(arrays["queries"])
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

    race(lane, arm, tag, rounds, size, gpu_device_name(), call, mode=_mode())


def lane_hdbscan(rounds, size, smoke):
    lane = "hdbscan"
    arm, run = deterministic_gate(lane, "cuml-gpu")
    if not run:
        return
    try:
        from cuml.cluster import HDBSCAN
        arrays, tag, prm = fixture(lane, size)
    except Exception as e:
        refuse(lane, arm, "%r" % (e,))
        return
    _source_note(lane, arm, size, tag, prm)
    n = prm["n"]
    X = arrays["x"]
    note(lane, arm, "min_samples, min_cluster_size, metric and "
                    "cluster_selection_method are passed explicitly. At %d "
                    "rows both arms are dominated by launch latency; this is "
                    "the fixture the lane ships and it has no size knob" % n)

    def call():
        m = HDBSCAN(min_samples=prm["min_samples"],
                    min_cluster_size=prm["min_cluster_size"],
                    metric="euclidean", cluster_selection_method="eom",
                    output_type="numpy")
        m.fit(X)
        return (m.labels_,)

    race(lane, arm, tag, rounds, size, gpu_device_name(), call, mode=_mode())


def lane_cholesky(rounds, size, smoke):
    lane = "cholesky"
    arm, run = deterministic_gate(lane, "torch-gpu")
    if not run:
        return
    try:
        import torch
        arrays, tag, prm = fixture(lane, size)
    except Exception as e:
        refuse(lane, arm, "%r" % (e,))
        return
    if not torch.cuda.is_available():
        refuse(lane, arm, "torch reports no CUDA device")
        return
    _source_note(lane, arm, size, tag, prm)
    if deterministic():
        # DEVIATION 2145: the vendor's global switch, and nothing else. If
        # either op has no deterministic implementation torch raises inside
        # the first (warm-up) call and the lane is refused with its words.
        try:
            torch.use_deterministic_algorithms(True)
        except Exception as e:                            # noqa: BLE001
            refuse(lane, arm, "torch.use_deterministic_algorithms(True) "
                              "raised: %r" % (e,))
            return
    n, nrhs = prm["n"], prm["nrhs"]
    jitter = prm["jitter"]
    A0 = torch.tensor(arrays["a"], device="cuda")
    B0 = torch.tensor(arrays["b"], device="cuda")
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

    race(lane, arm, tag, rounds, size, gpu_device_name(), call, mode=_mode())


def _cpu_fallback_gate(lane, cpu_arm, what):
    """The six lanes with no RAPIDS estimator. Under the default selector
    the GPU arm is refused by name (as before) and the CPU arm proceeds.
    Under `deterministic` BOTH siblings are refused: the GPU one from the
    table (NOT-OFFERED), and the CPU one because its default configuration
    is already seeded and it is not the arm this campaign compares against.
    Returns `(arm, run)`."""
    if deterministic():
        sib = DETERMINISTIC_SIBLINGS[lane]
        refuse(lane, sib.name, sib.reason())
        refuse(lane, cpu_arm + "-deterministic",
               "NOT-APPLICABLE: the %s arm passes random_state in its "
               "default configuration already and is a CPU library; there "
               "is no separate deterministic configuration to time" % cpu_arm)
        return cpu_arm + "-deterministic", False
    refuse(lane, "cuml-gpu",
           "RAPIDS ships no %s; the arm below is %s and is labeled %s"
           % (what, {"sklearn-cpu": "scikit-learn on the CPU",
                     "scipy-cpu": "SciPy on the CPU"}[cpu_arm], cpu_arm))
    return cpu_arm, True


def lane_gmm(rounds, size, smoke):
    lane = "gmm"
    try:
        from sklearn.mixture import GaussianMixture
        arrays, tag, prm = fixture(lane, size)
    except Exception as e:
        refuse(lane, "sklearn-cpu", "%r" % (e,))
        return
    # NO RAPIDS COUNTERPART. cuML ships no GaussianMixture at any version, so
    # there is no GPU arm to race and this one is labeled for what it is.
    arm, run = _cpu_fallback_gate(lane, "sklearn-cpu", "GaussianMixture")
    if not run:
        return
    _source_note(lane, arm, size, tag, prm)
    k = prm["n_components"]
    X = arrays["x"]

    def call():
        m = GaussianMixture(n_components=k, covariance_type="full",
                            max_iter=prm["max_iter"], tol=prm["tol"],
                            reg_covar=prm["reg_covar"], init_params="kmeans",
                            n_init=1, random_state=prm["random_state"])
        m.fit(X)
        return (m.weights_.astype(np.float32), m.means_.astype(np.float32))

    race(lane, arm, tag, rounds, size, cpu_device_name(), call)


def lane_gp(rounds, size, smoke):
    lane = "gp"
    try:
        from sklearn.gaussian_process import GaussianProcessRegressor
        from sklearn.gaussian_process.kernels import RBF
        arrays, tag, prm = fixture(lane, size)
    except Exception as e:
        refuse(lane, "sklearn-cpu", "%r" % (e,))
        return
    arm, run = _cpu_fallback_gate(lane, "sklearn-cpu",
                                  "Gaussian process regressor")
    if not run:
        return
    _source_note(lane, arm, size, tag, prm)
    X = arrays["x"].astype(np.float64)
    y = arrays["y"].astype(np.float64)
    Xs = arrays["x_star"].astype(np.float64)
    ls = arrays["length_scale"].astype(np.float64)
    alpha = prm["alpha"]
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

    race(lane, arm, tag, rounds, size, cpu_device_name(), call)


def lane_krr(rounds, size, smoke):
    lane = "krr"
    arm, run = deterministic_gate(lane, "cuml-gpu")
    if not run:
        return
    try:
        from cuml.kernel_ridge import KernelRidge
        arrays, tag, prm = fixture(lane, size)
    except Exception as e:
        refuse(lane, arm, "import failed: %r" % (e,))
        return
    _source_note(lane, arm, size, tag, prm)
    X, y, Xq = arrays["x"], arrays["y"], arrays["x_query"]
    gamma, alpha = prm["gamma"], prm["alpha"]
    note(lane, arm, "kernel=rbf, gamma and alpha passed explicitly; fit plus "
                    "predict inside the clock, matching our lane")

    def call():
        m = KernelRidge(alpha=alpha, kernel="rbf", gamma=gamma)
        m.fit(X, y)
        return (m.dual_coef_, m.predict(Xq))

    race(lane, arm, tag, rounds, size, gpu_device_name(), call, mode=_mode())


def lane_nystroem(rounds, size, smoke):
    lane = "nystroem"
    try:
        from sklearn.kernel_approximation import Nystroem
        arrays, tag, prm = fixture(lane, size)
    except Exception as e:
        refuse(lane, "sklearn-cpu", "%r" % (e,))
        return
    arm, run = _cpu_fallback_gate(lane, "sklearn-cpu", "Nystroem")
    if not run:
        return
    _source_note(lane, arm, size, tag, prm)
    q = prm["n_components"]
    X, Xq = arrays["x"], arrays["x_query"]
    gamma = prm["gamma"]
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

    race(lane, arm, tag, rounds, size, cpu_device_name(), call)


def lane_rbfsampler(rounds, size, smoke):
    lane = "rbfsampler"
    try:
        from sklearn.kernel_approximation import RBFSampler
        arrays, tag, prm = fixture(lane, size)
    except Exception as e:
        refuse(lane, "sklearn-cpu", "%r" % (e,))
        return
    arm, run = _cpu_fallback_gate(lane, "sklearn-cpu",
                                  "RBFSampler / random Fourier features")
    if not run:
        return
    _source_note(lane, arm, size, tag, prm)
    d, nq, q = prm["d"], prm["n_query"], prm["n_components"]
    Xq = arrays["x_query"]
    gamma = prm["gamma"]
    note(lane, arm, "hash=- : the random draws differ (a pinned Philox stream "
                    "against numpy RandomState.normal). Both arms draw a "
                    "d x q weight matrix and a q offset, then one matmul and "
                    "one cosine over %d rows" % nq)

    def call():
        m = RBFSampler(gamma=gamma, n_components=q, random_state=0)
        m.fit(np.zeros((1, d), dtype=np.float32))
        m.transform(Xq)
        return None

    race(lane, arm, tag, rounds, size, cpu_device_name(), call)


def lane_resample(rounds, size, smoke):
    lane = "resample"
    try:
        from scipy.stats import bootstrap
        arrays, tag, prm = fixture(lane, size)
    except Exception as e:
        refuse(lane, "scipy-cpu", "%r" % (e,))
        return
    arm, run = _cpu_fallback_gate(lane, "scipy-cpu", "bootstrap")
    if not run:
        return
    _source_note(lane, arm, size, tag, prm)
    n_res = prm["n_resamples"]
    X = arrays["x"]
    col0 = np.ascontiguousarray(X[:, 0].astype(np.float64))
    conf = prm["confidence_level"]
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

    race(lane, arm, tag, rounds, size, cpu_device_name(), call)


def lane_spectral(rounds, size, smoke):
    lane = "spectral"
    try:
        from sklearn.cluster import SpectralClustering
        arrays, tag, prm = fixture(lane, size)
    except Exception as e:
        refuse(lane, "sklearn-cpu", "%r" % (e,))
        return
    arm, run = _cpu_fallback_gate(lane, "sklearn-cpu",
                                  "SpectralClustering estimator")
    if not run:
        return
    _source_note(lane, arm, size, tag, prm)
    X = arrays["x"].astype(np.float64)
    note(lane, arm, "affinity=nearest_neighbors with the same n_neighbors, "
                    "same n_clusters, same n_components, same n_init: both "
                    "arms build a kNN affinity, take a Lanczos "
                    "eigendecomposition of the normalized Laplacian and run "
                    "k-means on the embedding. The eigensolvers are two "
                    "Lanczos implementations (ours restarts, theirs is ARPACK)")
    note(lane, arm, "hash=- : cluster label NUMBERING is arbitrary in both "
                    "and the two are at best a permutation of each other")

    def call():
        m = SpectralClustering(n_clusters=prm["n_clusters"],
                               n_components=prm["n_components"],
                               affinity="nearest_neighbors",
                               n_neighbors=prm["n_neighbors"],
                               eigen_solver="arpack",
                               n_init=prm["n_init"],
                               assign_labels="kmeans",
                               random_state=prm["seed"])
        m.fit(X)
        return None

    race(lane, arm, tag, rounds, size, cpu_device_name(), call)


def lane_holtwinters(rounds, size, smoke):
    lane = "holtwinters"
    arm, run = deterministic_gate(lane, "cuml-gpu")
    if not run:
        return
    try:
        from cuml import ExponentialSmoothing
        arrays, tag, prm = fixture(lane, size)
    except Exception as e:
        refuse(lane, arm, "import failed: %r" % (e,))
        return
    _source_note(lane, arm, size, tag, prm)
    batch, freq = prm["batch_size"], prm["frequency"]
    y = arrays["y"]
    note(lane, arm, "seasonal=additive, seasonal_periods, start_periods, "
                    "ts_num and eps all passed explicitly. The dump is "
                    "series-major (batch_size x n), which is cuML's own "
                    "(ts_num, n) layout, so no transpose is needed")

    def call():
        m = ExponentialSmoothing(y, seasonal="additive",
                                 seasonal_periods=freq,
                                 start_periods=prm["start_periods"],
                                 ts_num=batch, eps=prm["eps"])
        m.fit()
        return (to_numpy(m.get_level()).astype(np.float32),)

    race(lane, arm, tag, rounds, size, gpu_device_name(), call, mode=_mode())


def lane_kpss(rounds, size, smoke):
    lane = "kpss"
    arm, run = deterministic_gate(lane, "cuml-gpu")
    if not run:
        return
    try:
        arrays, shape, prm = fixture(lane, size)
    except Exception as e:
        refuse(lane, arm, "%r" % (e,))
        return
    _source_note(lane, arm, size, shape, prm)
    batch = prm["batch_size"]
    y = arrays["y"]

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
        note(lane, arm, "cuml.tsa.stationarity.kpss_test at d=1, D=0, s=0, "
                        "pval_threshold=0.05, the same batch of %d series"
             % batch)

        def call():
            return (np.asarray(to_numpy(cuml_kpss(y, d=1, D=0, s=0,
                                                  pval_threshold=0.05))),)

        race(lane, arm, shape, rounds, size, gpu_device_name(), call,
             mode=_mode())
        return

    refuse(lane, arm,
           "this RAPIDS build exposes no cuml.tsa.stationarity.kpss_test; "
           "falling back to statsmodels on the CPU, labeled statsmodels-cpu")
    if deterministic():
        refuse(lane, "statsmodels-cpu-deterministic",
               "NOT-APPLICABLE: statsmodels' kpss draws nothing at random "
               "and is a CPU library; there is no separate deterministic "
               "configuration to time")
        return
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
# umap -- DEVIATION 2135, the lane with no Mojo-driver twin
# ===========================================================================

#: DEVIATION 2139: the embedding is n x 2 float32 (1.6 MB at 200,000 rows)
#: and its per-round hash is the whole deterministic question, so this lane
#: alone hashes up to 8 MB.
UMAP_HASH_MAX = 8 * 1024 * 1024


def lane_umap(rounds, size, smoke):
    """`cuml.manifold.UMAP(...).fit_transform` at the vendor's defaults
    (`random_state=None`, `build_algo='auto'`), and under `deterministic`
    the same call with `random_state=7`.

    WHAT `auto` MEANS ON EACH ARM, from cuML's source (manifold/umap/
    umap.pyx, `init_params`): with `random_state=None` and more than
    50,000 rows the neighbor graph is built by NN-descent; with a
    `random_state` set it is brute-force k-NN, and the source sets
    `params.deterministic = random_state is not None or n_rows < 300`. So
    the two arms differ in the graph builder AND the optimizer, exactly as
    the vendor's sentence says a user who asks for reproducibility pays
    for it. Both are the vendor's documented behaviour and neither is
    adjusted here.

    Our arm (`bench/speed/umap_speed_arm.py`) consumes the same `fixture()`
    and the same `trustworthiness_subsample` helper, so the bytes in and
    the accuracy scorer are shared by construction; the embeddings are
    NOT expected to share bits (different neighbor search, different
    optimizer, different init), and the `hash=` column is within-arm only,
    as everywhere else in this file.
    """
    lane = "umap"
    arm, run = deterministic_gate(lane, "cuml-gpu")
    if not run:
        return
    try:
        from cuml.manifold import UMAP
    except Exception as e:
        refuse(lane, arm, "import failed: %r" % (e,))
        return
    arrays, tag, prm = fixture(lane, size)
    _source_note(lane, arm, size, tag, prm)
    x_host = arrays["x"]
    X = to_device(lane, arm, x_host)
    kw = dict(n_neighbors=prm["n_neighbors"], n_components=prm["n_components"],
              min_dist=prm["min_dist"], n_epochs=prm["n_epochs"],
              metric=prm["metric"], init="spectral", output_type="cupy")
    if deterministic():
        kw["random_state"] = prm["deterministic_random_state"]
    note(lane, arm, "n_neighbors=%d n_components=%d min_dist=%g n_epochs=%d "
                    "metric=%s init=spectral random_state=%s build_algo=auto"
         % (prm["n_neighbors"], prm["n_components"], prm["min_dist"],
            prm["n_epochs"], prm["metric"], kw.get("random_state", "None")))
    last = {}

    def call():
        m = UMAP(**kw)
        emb = m.fit_transform(X)
        last["emb"] = emb
        return (emb,)

    race(lane, arm, tag, rounds, size, gpu_device_name(), call, mode=_mode(),
         hash_max=UMAP_HASH_MAX)
    if "emb" in last:
        name, value = trustworthiness_subsample(
            x_host, last["emb"], prm["n_neighbors"], prm["trust_rows"])
        if name is None:
            refuse(lane, arm, "FSPEED-ACC not emitted: %s" % value)
        else:
            acc(lane, arm, name, value)


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
    "umap": lane_umap,
}


def main():
    lane = os.environ.get("MOJOLEARN_SPEED_LANE", "")
    rounds = int(os.environ.get("MOJOLEARN_SPEED_ROUNDS", "5"))
    size = os.environ.get("MOJOLEARN_SPEED_SIZE", "shipped")
    if size not in SIZES:
        raise SystemExit("MOJOLEARN_SPEED_SIZE must be one of %s"
                         % " ".join(SIZES))
    if rounds < 1:
        raise SystemExit("MOJOLEARN_SPEED_ROUNDS must be >= 1")
    if lane not in LANES:
        raise SystemExit(
            "MOJOLEARN_SPEED_LANE must be one of: %s; got %r"
            % (" ".join(sorted(LANES)), lane))
    arm_selector()          # validates MOJOLEARN_SPEED_ARM before any work
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
