# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""DEVIATION 3010: release a resident forest, then prepare another.

The user-visible shape of the hang `tools/forest_groves_identity.py large`
met on main (2026-09-17, RTX 4090): a program that fits or loads a second
`inference_engine="parallel_groves"` forest AFTER the first one was
collected never returns. It needs no dataset and no saved model: two fits
of the public estimator on synthetic rows are enough, which is what makes
it a repro rather than a leg.

    python3 tools/forest_release_prepare_repro.py --mode release
    python3 tools/forest_release_prepare_repro.py --mode keep

`--mode release` drops the first estimator and runs `gc.collect()` (that
finalizer is `native.forest_release_gpu`, `_ResidentForest.__init__`'s
`weakref.finalize`) before the second estimator's first `predict`, which is
where the second snapshot's `forest_prepare_gpu` runs. `--mode keep` holds
the first estimator alive across the second, the control that passed on
main. Both print one flushed line per phase, so a hang names its phase.

On a deadline the watchdog prints every Python thread's stack, asks the
main thread for a native backtrace when `tools/native_stack_dump.c` is
preloaded (`MOJOLEARN_NATIVE_STACK_FILE` set, DEVIATION 2518), and exits
124. EXIT 124 IS THE HANG; exit 0 with a `DONE` line is the pass. The
predictions' SHA-256 goes to stdout in both modes, so the two arms of a
teardown change can be compared bit for bit on the mode that completes on
both of them.
"""
import argparse
import faulthandler
import gc
import hashlib
import os
import signal
import sys
import threading
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "python"))


def _say(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _watchdog(seconds, main_tid):
    def run():
        time.sleep(seconds)
        print(f"\nWATCHDOG: no progress for {seconds}s; this is the hang.", flush=True)
        stack_file = os.environ.get("MOJOLEARN_NATIVE_STACK_FILE")
        if stack_file:
            for _ in range(3):
                try:
                    signal.pthread_kill(main_tid, signal.SIGUSR2)
                except Exception as exc:  # the handler may not be installed
                    print(f"WATCHDOG: SIGUSR2 failed: {exc}", flush=True)
                    break
                time.sleep(2)
            print(f"WATCHDOG: native stack samples in {stack_file}", flush=True)
        faulthandler.dump_traceback()
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(124)
    t = threading.Thread(target=run, daemon=True)
    t.start()
    return t


def _digest(a):
    """The prediction's raw bytes, whatever container carries them."""
    try:
        raw = a.tobytes()
    except AttributeError:
        raw = memoryview(a).cast("B").tobytes()
    return hashlib.sha256(raw).hexdigest()[:16]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=("release", "keep"), default="release")
    ap.add_argument("--rows", type=int, default=60000)
    ap.add_argument("--features", type=int, default=16)
    ap.add_argument("--trees", type=int, default=64)
    ap.add_argument("--depth", type=int, default=12)
    ap.add_argument("--predict-rows", type=int, default=2000)
    ap.add_argument("--deadline", type=float, default=180.0)
    args = ap.parse_args()

    faulthandler.enable()
    _watchdog(args.deadline, threading.get_ident())

    import numpy as np
    import mojolearn as ml
    _say(f"mojolearn from {ml.__file__}")

    rng = np.random.default_rng(20260918)
    X = np.ascontiguousarray(rng.standard_normal((args.rows, args.features)), dtype=np.float32)
    y0 = (X[:, 0] + 0.5 * X[:, 1] > 0).astype(np.int32)
    y1 = (X[:, 2] - 0.25 * X[:, 3] > 0).astype(np.int32)
    Xq = np.ascontiguousarray(X[:args.predict_rows])

    def fit(y, seed):
        return ml.RandomForestClassifier(
            n_estimators=args.trees, max_depth=args.depth, random_state=seed,
            inference_engine="parallel_groves").fit(X, y)

    _say("phase 1: fit model A")
    a = fit(y0, 7)
    _say("phase 2: fit model B")
    b = fit(y1, 11)
    _say("phase 3: predict with A (prepares A's device snapshot)")
    pa = a.predict(Xq)
    _say(f"phase 3 done: A predict sha256={_digest(pa)}")

    if args.mode == "release":
        _say("phase 4: del A + gc.collect() (runs forest_release_gpu)")
        del a
        gc.collect()
        _say("phase 4 done: A released")
    else:
        _say("phase 4: A KEPT alive (control)")

    _say("phase 5: predict with B (prepares B's device snapshot) <-- the hang is here")
    pb = b.predict(Xq)
    _say(f"phase 5 done: B predict sha256={_digest(pb)}")
    print(f"HASHES mode={args.mode} A={_digest(pa)} B={_digest(pb)}", flush=True)
    print("DONE", args.mode, flush=True)


if __name__ == "__main__":
    main()
