#!/usr/bin/env python3
"""Device memory accounting across repeated DBSCAN fits (lane/amd-dbscan-oom).

WHY THIS EXISTS. `lane/classical-host-recordings` took the AMD column for the
saved-model predict lanes on a RunPod MI300X and got
`cells=54 stable=31 moved=0 refused=23`, every refusal a `hipErrorOutOfMemory`
raised inside `dbscan_fit_core` while fitting 6000 rows of four columns on a
192 GB card. That lane claimed nothing about the cause and could not: it had no
allocator instrumentation and no replication. This file is the instrumentation.

WHAT IT MEASURES. The DEVICE's own used-memory figure, read from outside our
library, immediately before and immediately after every fit, continuing past an
exception so the whole trend is visible rather than only the point where it
stopped. A LEAK is a used-memory line that climbs with the fit index.
FRAGMENTATION is a line that does not climb while the fits fail anyway. "It
stopped crashing" separates neither and is not reported as evidence.

THE PROBE PROVES IT CAN MOVE BEFORE IT REPORTS ANYTHING. A flat reading from a
broken reader is indistinguishable from a flat reading from a clean allocator.
`--self-check` (on by default) reads the device before any GPU work, runs one
fit, reads again, and if the figure has not moved by at least `--move-floor`
MiB it prints PROBE CANNOT MOVE and exits 3 without running an arm.

ARMS, each one sequence of fits in one process, chosen to split the variables
the original observation confounds:

  seq      the reproduction: dbscan, dbscan-brute-l1, dbscan-weighted over all
           nine fixtures in the lane-major order tools/identity_break.py uses,
           `--repeats` times. 54 fits at the default.
  repeat   ONE fixture fitted N times. The shape never changes, so a climb here
           is a per-fit leak and cannot be a size threshold.
  shapes   every fixture once, ascending by neighbourhood size. A wall that
           tracks size rather than order is not a leak.
  arms     the same fixture with prediction_data on and off, rbc and brute,
           weighted and not, to name which path climbs.

The JSON is the evidence. Stdout is for watching a box.
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import subprocess
import sys
import time
import traceback

import numpy as np

# --------------------------------------------------------------- the fixtures
# COPIED from tools/identity_break.py rather than imported: that module builds a
# lane registry and imports mojolearn at module scope, and this probe has to run
# before and after a failing fit without either. `--fixture-digest` prints the
# sha256 of every array this file makes so a drift from identity_break's own
# bytes is a visible mismatch rather than an assumption.
N, D = 20000, 16
FIXTURES = ["base", "ties", "hashed", "wide", "denormal", "denormal_ftz",
            "dupes", "odd", "negative"]


def _hash_stream(seed_bytes, nbytes):
    stream = bytearray()
    blk = 0
    while len(stream) < nbytes:
        stream += hashlib.sha256(seed_bytes + blk.to_bytes(8, "little")).digest()
        blk += 1
    return bytes(stream[:nbytes])


def _hashed_uniform(n, d, seed):
    out = np.empty(n * d, dtype=np.float32)
    ctr = np.arange(n * d, dtype=np.uint64)
    dig = hashlib.sha256(ctr.tobytes() + str(seed).encode()).digest()
    u32 = np.frombuffer(_hash_stream(dig, n * d * 4), dtype=np.uint32)
    out[:] = (u32.astype(np.float64) / 2**32).astype(np.float32)
    return out.reshape(n, d)


def fixture(kind, n=N, d=D, seed=0):
    rng = np.random.default_rng(seed)
    if kind == "base":
        X = rng.standard_normal((n, d)).astype(np.float32)
    elif kind == "ties":
        X = rng.integers(0, 6, size=(n, d)).astype(np.float32)
    elif kind == "hashed":
        X = (_hashed_uniform(n, d, seed) * 4.0 - 2.0).astype(np.float32)
    elif kind == "wide":
        X = rng.standard_normal((n, d)).astype(np.float32)
        scale = np.logspace(-4, 4, d).astype(np.float32)
        X = (X * scale).astype(np.float32)
    elif kind == "denormal":
        X = rng.standard_normal((n, d)).astype(np.float32)
        X[: n // 4, :3] = (X[: n // 4, :3] * np.float32(1e-40)).astype(np.float32)
    elif kind == "denormal_ftz":
        X = fixture("denormal", n, d, seed)
        sub = (X != 0) & (np.abs(X) < np.finfo(np.float32).tiny)
        X = X.copy()
        X[sub] = np.copysign(np.float32(0.0), X[sub])
    elif kind == "dupes":
        X = rng.standard_normal((n, d)).astype(np.float32)
        X[n // 2:] = X[: n - n // 2]
        X[:, d - 2] = np.float32(3.5)
        X[:, d - 1] = np.float32(0.0)
    elif kind == "odd":
        n, d = 12345, 17
        X = rng.standard_normal((n, d)).astype(np.float32)
    elif kind == "negative":
        X = (-np.abs(rng.standard_normal((n, d))) - 2.0).astype(np.float32)
    else:
        raise ValueError(kind)
    return X


#: MEASURED with numpy, 2026-09-16, eps=0.9 over X[:6000, :4]: ordered pairs
#: within eps (the CSR edge count the ball cover arm emits) and the maximum
#: degree. The refusal ORDER in bench/results/identity_break/2026-09-16_amd-mi300x
#: does not track it -- `dupes` (687484) ran clean immediately after `denormal`
#: (12608098) refused, and `odd` (667520) refused right after `dupes` passed --
#: which is why neither "ordered" nor "shaped" describes that column on its own.
NNZ_L2 = {"base": 687484, "ties": 33474, "hashed": 333690, "wide": 36000000,
          "denormal": 12608098, "denormal_ftz": 12608098, "dupes": 687484,
          "odd": 667520, "negative": 5034848}
NNZ_L1 = {"base": 106110, "ties": 33474, "hashed": 56996, "wide": 36000000,
          "denormal": 12129154, "denormal_ftz": 12129154, "dupes": 106110,
          "odd": 103032, "negative": 1027624}

ROWS, COLS = 6000, 4
EPS, MIN_SAMPLES = 0.9, 5

#: the three DBSCAN lanes of tools/identity_break.py, verbatim
LANE_KW = {
    "dbscan": dict(eps=EPS, min_samples=MIN_SAMPLES, prediction_data=True),
    "dbscan-brute-l1": dict(eps=EPS, min_samples=MIN_SAMPLES,
                            metric="manhattan", algorithm="brute",
                            prediction_data=True),
    "dbscan-weighted": dict(eps=EPS, min_samples=MIN_SAMPLES,
                            prediction_data=True),
}
LANE_ORDER = ["dbscan", "dbscan-brute-l1", "dbscan-weighted"]


def _hw(shape, seed, lo, hi):
    """identity_break's `_hw`, copied: a float32 tensor of `shape` from the
    hashed stream, uniform on [lo, hi), the seed a string naming the lane and
    the tensor. `dbscan-weighted` fits with `_hw((6000,), "dbscan:sample_weight",
    0.5, 1.5)` and a different draw is a different fit."""
    n = int(np.prod(shape))
    u = _hashed_uniform(n, 1, seed).reshape(-1)
    return np.ascontiguousarray(
        (u * np.float32(hi - lo) + np.float32(lo)).astype(np.float32)
        .reshape(shape))


# ------------------------------------------------------------- the memory read
class MemoryReader:
    """Used and total device memory in BYTES, from outside our library.

    Not from our own allocator: a leak reported by the thing that leaks can be
    flat for the wrong reason, and the vendor's figure also counts the context,
    the loaded code objects and pinned host mappings that a buffer-level tally
    would miss.

    Sources, in the order tried:
      amdgpu-sysfs  /sys/class/drm/card*/device/mem_info_vram_{used,total}, the
                    kernel driver's own counters, no tool version to match
      nvidia-smi    --query-gpu=memory.used,memory.total
      rocm-smi      --showmeminfo vram --csv, header-matched by column name

    `proc-rss` is a FOURTH source that is NEVER chosen automatically and must be
    named with `--source proc-rss`. It reads the PROCESS's resident set size,
    not the device, and exists for ONE job: rehearsing this file's own plumbing
    on a machine with no device tool (an M4, a CPU pod) so that the first time
    the arms run is not on a paid box. A reading from it is evidence about host
    memory and about nothing else; the JSON records `source` so a run that used
    it can never be read as a device measurement.

    Construction REFUSES when nothing answers. A reader that returns zeros is
    the exact shape of probe that cannot fail, and this lane was told not to
    build one.
    """

    def __init__(self, index: int = 0, prefer: str = ""):
        self.index = index
        self.kind = None
        self.sysfs = None
        tried = []
        auto = ["amdgpu-sysfs", "nvidia-smi", "rocm-smi"]
        if prefer and prefer not in auto + ["proc-rss"]:
            raise SystemExit("dbscan_memory_probe: unknown --source " + prefer)
        for kind in ([prefer] if prefer else auto):
            tried.append(kind)
            try:
                if kind == "proc-rss":
                pass
            if kind == "amdgpu-sysfs":
                    cards = sorted(glob.glob(
                        "/sys/class/drm/card*/device/mem_info_vram_used"))
                    if len(cards) <= index:
                        continue
                    self.sysfs = os.path.dirname(cards[index])
                probe = self._read(kind)
            except Exception:
                continue
            if probe is None:
                continue
            self.kind = kind
            break
        if self.kind is None:
            raise SystemExit(
                "dbscan_memory_probe: no device memory source answered (tried "
                + ", ".join(tried) + "). REFUSING to run, because a probe that "
                "cannot read the device reports a flat line that means nothing.")

    def _read(self, kind):
        if kind == "proc-rss":
            # host RSS, for rehearsal only; see the class docstring
            try:
                with open("/proc/self/status") as fh:
                    for line in fh:
                        if line.startswith("VmRSS:"):
                            return int(line.split()[1]) * 1024, 0
            except OSError:
                pass
            out = subprocess.run(["ps", "-o", "rss=", "-p", str(os.getpid())],
                                 capture_output=True, text=True, timeout=60)
            if out.returncode != 0 or not out.stdout.strip():
                return None
            return int(out.stdout.strip()) * 1024, 0
        if kind == "amdgpu-sysfs":
            with open(os.path.join(self.sysfs, "mem_info_vram_used")) as fh:
                used = int(fh.read().strip())
            with open(os.path.join(self.sysfs, "mem_info_vram_total")) as fh:
                total = int(fh.read().strip())
            return used, total
        if kind == "nvidia-smi":
            out = subprocess.run(
                ["nvidia-smi", "--query-gpu=memory.used,memory.total",
                 "--format=csv,noheader,nounits", "-i", str(self.index)],
                capture_output=True, text=True, timeout=60)
            if out.returncode != 0 or not out.stdout.strip():
                return None
            used_mib, total_mib = [float(v) for v in
                                   out.stdout.strip().splitlines()[0].split(",")]
            return int(used_mib) << 20, int(total_mib) << 20
        if kind == "rocm-smi":
            out = subprocess.run(["rocm-smi", "--showmeminfo", "vram", "--csv"],
                                 capture_output=True, text=True, timeout=60)
            if out.returncode != 0:
                return None
            rows = [r for r in out.stdout.strip().splitlines() if "," in r]
            if len(rows) < 2:
                return None
            head = [c.strip().lower() for c in rows[0].split(",")]
            ti = next((i for i, c in enumerate(head)
                       if "total" in c and "used" not in c), None)
            ui = next((i for i, c in enumerate(head) if "used" in c), None)
            if ti is None or ui is None:
                return None
            cells = [c.strip() for c in rows[1 + self.index].split(",")]
            return int(cells[ui]), int(cells[ti])
        return None

    def read(self):
        v = self._read(self.kind)
        if v is None:
            raise RuntimeError(
                "dbscan_memory_probe: the %s reader stopped answering mid-run. "
                "The readings after this point would be stale, not flat."
                % self.kind)
        return v


# ------------------------------------------------------------------- the fits
class Runner:
    def __init__(self, reader, out_rows, quiet=False):
        self.reader = reader
        self.rows = out_rows
        self.quiet = quiet
        self.index = 0
        import mojolearn as ml
        self.ml = ml
        self.X = {}
        self.W = _hw((ROWS,), "dbscan:sample_weight", 0.5, 1.5)

    def data(self, kind):
        if kind not in self.X:
            self.X[kind] = np.ascontiguousarray(fixture(kind)[:ROWS, :COLS])
        return self.X[kind]

    def fit(self, arm, lane, kind, **override):
        kw = dict(LANE_KW[lane])
        kw.update(override)
        X = self.data(kind)
        w = self.W if lane == "dbscan-weighted" else None
        used0, total = self.reader.read()
        t0 = time.time()
        err = None
        n_clusters = None
        try:
            m = self.ml.DBSCAN(**kw).fit(X, sample_weight=w)
            labels = m.labels_
            n_clusters = int(len(set(np.asarray(labels).tolist())))
            del m, labels
        except BaseException as exc:          # noqa: BLE001 -- a raise is data
            err = "".join(traceback.format_exception_only(type(exc), exc)).strip()
        wall = time.time() - t0
        used1, _ = self.reader.read()
        self.index += 1
        row = dict(i=self.index, arm=arm, lane=lane, fixture=kind,
                   kw={k: v for k, v in kw.items()},
                   weighted=w is not None,
                   nnz=(NNZ_L1 if kw.get("metric") == "manhattan" else NNZ_L2)
                       .get(kind),
                   used_before=used0, used_after=used1,
                   delta=used1 - used0, total=total,
                   wall_s=round(wall, 3), n_clusters=n_clusters,
                   ok=err is None, error=err)
        self.rows.append(row)
        if not self.quiet:
            print("FIT %3d %-10s %-16s %-13s %-3s used %7.1f -> %7.1f MiB "
                  "(delta %+8.1f) free %8.1f  %6.2fs %s"
                  % (row["i"], arm, lane, kind, "ok" if err is None else "RAISE",
                     used0 / 2**20, used1 / 2**20, (used1 - used0) / 2**20,
                     (total - used1) / 2**20 if total else float("nan"), wall,
                     "" if err is None else err.splitlines()[-1][:80]),
                  flush=True)
        return row


def trend(rows):
    """Least squares slope of used-after against fit index, MiB per fit, over
    the fits that SUCCEEDED. A slope computed over raises would be measuring the
    failure path, which is a different question."""
    pts = [(r["i"], r["used_after"]) for r in rows if r["ok"]]
    if len(pts) < 3:
        return None
    xs = np.array([p[0] for p in pts], dtype=np.float64)
    ys = np.array([p[1] for p in pts], dtype=np.float64) / 2**20
    slope, intercept = np.polyfit(xs, ys, 1)
    return dict(n=len(pts), slope_mib_per_fit=float(slope),
                first_used_mib=float(ys[0]), last_used_mib=float(ys[-1]),
                span_mib=float(ys[-1] - ys[0]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="JSON evidence file")
    ap.add_argument("--arm", action="append", default=[],
                    choices=["seq", "repeat", "shapes", "arms"],
                    help="repeatable; default is every arm")
    ap.add_argument("--repeats", type=int, default=2,
                    help="seq: repeats per cell, as identity_break's --repeats")
    ap.add_argument("--n", type=int, default=12,
                    help="repeat/arms: fits per configuration")
    ap.add_argument("--repeat-fixture", default="base")
    ap.add_argument("--device", type=int, default=0)
    ap.add_argument("--source", default="",
                    help="force a memory source: amdgpu-sysfs, nvidia-smi, "
                         "rocm-smi, or proc-rss (HOST RSS, rehearsal only, "
                         "never a device reading)")
    ap.add_argument("--move-floor", type=float, default=8.0,
                    help="MiB the self check must see move")
    ap.add_argument("--no-self-check", action="store_true")
    ap.add_argument("--fixture-digest", action="store_true",
                    help="print each fixture's sha256 and exit")
    args = ap.parse_args()

    if args.fixture_digest:
        for k in FIXTURES:
            X = np.ascontiguousarray(fixture(k)[:ROWS, :COLS])
            print("%-14s %s %s" % (k, X.shape,
                                   hashlib.sha256(X.tobytes()).hexdigest()))
        return 0

    arms = args.arm or ["seq", "repeat", "shapes", "arms"]
    reader = MemoryReader(args.device, args.source)
    used_boot, total = reader.read()
    print("source=%s device=%d total=%.1f MiB used_before_any_gpu_work=%.1f MiB"
          % (reader.kind, args.device, total / 2**20, used_boot / 2**20),
          flush=True)

    rows = []
    run = Runner(reader, rows)
    result = dict(source=reader.kind, device=args.device, total=total,
                  used_boot=used_boot, arms=arms, repeats=args.repeats,
                  n=args.n, rows=rows,
                  started=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))

    def save():
        os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".",
                    exist_ok=True)
        result["trend"] = {a: trend([r for r in rows if r["arm"] == a])
                           for a in sorted({r["arm"] for r in rows})}
        result["trend"]["ALL"] = trend(rows)
        result["refused"] = [f"{r['arm']}/{r['lane']}/{r['fixture']}#{r['i']}"
                             for r in rows if not r["ok"]]
        with open(args.out, "w") as fh:
            json.dump(result, fh, indent=1)

    # ------------------------------------------------ the probe must move first
    if not args.no_self_check:
        run.fit("selfcheck", "dbscan", "ties")
        moved = (rows[-1]["used_after"] - used_boot) / 2**20
        result["self_check_moved_mib"] = moved
        save()
        if moved < args.move_floor:
            print("PROBE CANNOT MOVE: used memory went %.2f MiB across the "
                  "first fit, under the %.1f MiB floor. Every reading after "
                  "this would be unfalsifiable; refusing to run an arm."
                  % (moved, args.move_floor), flush=True)
            return 3
        print("probe moves: %.1f MiB across the first fit (floor %.1f)"
              % (moved, args.move_floor), flush=True)

    if "seq" in arms:
        # tools/identity_break.py's own order: lane-major, fixtures inside,
        # repeats inside that.
        for lane in LANE_ORDER:
            for kind in FIXTURES:
                for _ in range(args.repeats):
                    run.fit("seq", lane, kind)
            save()

    if "repeat" in arms:
        for _ in range(args.n):
            run.fit("repeat", "dbscan", args.repeat_fixture)
        save()

    if "shapes" in arms:
        for kind in sorted(FIXTURES, key=lambda k: NNZ_L2[k]):
            run.fit("shapes", "dbscan", kind)
        save()

    if "arms" in arms:
        f = args.repeat_fixture
        for tag, lane, extra in (
                ("pred-off", "dbscan", dict(prediction_data=False)),
                ("pred-on", "dbscan", dict(prediction_data=True)),
                ("brute-l2", "dbscan", dict(algorithm="brute")),
                ("weighted", "dbscan-weighted", {}),
        ):
            for _ in range(args.n):
                run.fit("arms:" + tag, lane, f, **extra)
            save()

    save()
    print("\n--- trend (MiB per fit, over the fits that succeeded) ---",
          flush=True)
    for a, t in sorted(result["trend"].items()):
        if t is None:
            print("%-14s (fewer than three successful fits)" % a)
        else:
            print("%-14s slope %+9.3f  first %9.1f  last %9.1f  span %+9.1f "
                  "over %d fits" % (a, t["slope_mib_per_fit"],
                                    t["first_used_mib"], t["last_used_mib"],
                                    t["span_mib"], t["n"]))
    print("refused %d of %d: %s" % (len(result["refused"]), len(rows),
                                    ", ".join(result["refused"][:12])),
          flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
