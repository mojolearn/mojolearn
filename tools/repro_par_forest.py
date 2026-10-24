#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Reproduce the verify cell par-forest/base OUTSIDE the verify worker.

Test tooling only (no runtime code). The cell is
`tools/identity_break.py` lane "par-forest": fit_forest of
RandomForestClassifier(n_estimators=16, max_depth=8, random_state=7) on the
harness's "base" fixture (20000 x 16 float32, binary labels, seed 0) with
devices=(0,) and trees_per_shard=4, i.e. tree_start 0, 4, 8, 12, identical
mode.

Modes (one line per step, with wall time, flushed as it happens):

  (default)   the same four shard requests through DevicePool, one at a
              time, then the whole fit_forest call and a plain fit.
  --direct    in this process, no worker: `_fit_with_tree_start` (which runs
              the native `rf_classifier_fit_shard`) for tree_start 0 and 4,
              and a plain fit, on the same arrays. With --inputs pool, on
              exactly what a pool worker receives (as_f32_c X, the
              flatten_labels Python list y, through a pickle round trip).

MOJOLEARN_POOL_TIMING=1 (set unless --no-timing) makes the pool print
POOL-TIMING lines (parent: pickle_request, write, wait_first_byte,
read+unpickle_reply; worker: read+unpickle, execute, native_fit with the
input types, pickle_reply, write) and POOL-SNAPSHOT lines (environment and
loaded native modules of the worker and of a --direct process) on stderr.
  --verify-layout
              re-run this script (with the other flags) in a fresh process
              that leads its own session with stdout/stderr redirected to a
              file, as `_verify_worker.run` launches a cell; print that file.

Every mode arms faulthandler.dump_traceback_later(--timeout) so a hang
prints the Python frames of this process and exits.

Flags for the environment the lane considered:
  --cpu-threads N   set MOJOLEARN_CPU_THREADS=N (the verify CLI budget sets 1;
                    the L4 bisect already showed it is not required to hang)
  --export SEL      set MOJOLEARN_FOREST_EXPORT=SEL (the fit export route the
                    forest owner reads, default 'into')

Usage on a GPU box, from the repository root:
  PYTHONPATH=python python tools/repro_par_forest.py
  PYTHONPATH=python python tools/repro_par_forest.py --direct
  PYTHONPATH=python python tools/repro_par_forest.py --verify-layout
"""
import argparse
import faulthandler
import os
import subprocess
import sys
import tempfile
import time

TREES, DEPTH, SEED, PER_SHARD = 16, 8, 7, 4


def _say(*parts):
    print(*parts, flush=True)


def _fixture():
    from mojolearn import _verify_all
    path, how = _verify_all.harness_path()
    harness = _verify_all.load_harness(path)
    X, yc, _yr = harness.fixture("base")
    _say(f"fixture base: X {X.shape} {X.dtype}, labels {sorted(set(int(v) for v in yc))} "
         f"(harness {how}: {path})")
    return X, yc


def _timed(label, call):
    started = time.monotonic()
    result = call()
    _say(f"{label}: {time.monotonic() - started:.2f}s")
    return result


def run_pool(X, yc):
    import mojolearn as ml
    from mojolearn._buffer import as_f32_c
    from mojolearn._labels import flatten_labels
    from mojolearn._parallel_pool import DevicePool
    from mojolearn.parallel_ensemble import fit_forest
    params = ml.RandomForestClassifier(n_estimators=TREES, max_depth=DEPTH, random_state=SEED).get_params()
    params["numeric_mode"] = "identical"
    xs, _ = as_f32_c(X, ndim=2, name="X")
    ys = flatten_labels(yc)
    pool = DevicePool((0,))
    try:
        for start in range(0, TREES, PER_SHARD):
            request = ("forest_fit", ("RandomForestClassifier", dict(params, n_estimators=PER_SHARD)),
                       (xs, ys, start))
            _timed(f"pool shard tree_start={start}", lambda: pool.map([request]))
    finally:
        pool.close()
    _timed("fit_forest (4 shards, devices=(0,))",
           lambda: fit_forest(ml.RandomForestClassifier(n_estimators=TREES, max_depth=DEPTH, random_state=SEED),
                              X, yc, devices=(0,), trees_per_shard=PER_SHARD))
    _timed("plain fit (16 trees)",
           lambda: ml.RandomForestClassifier(n_estimators=TREES, max_depth=DEPTH, random_state=SEED).fit(X, yc))


def run_direct(X, yc, inputs="numpy"):
    """`inputs`: 'numpy' (the fixture arrays), or 'pool' (exactly what a pool
    worker receives: as_f32_c X and flatten_labels y, a Python list,
    round-tripped through pickle protocol 5)."""
    import pickle
    import mojolearn as ml
    from mojolearn import _parallel_worker
    if os.environ.get("MOJOLEARN_POOL_TIMING") == "1":
        _parallel_worker._timing_snapshot("direct")
    if inputs != "numpy":
        from mojolearn._buffer import as_f32_c
        from mojolearn._labels import flatten_labels
        started = time.monotonic()
        xs = as_f32_c(X, ndim=2, name="X")[0] if inputs in ("pool", "pool-x") else X
        ys = flatten_labels(yc) if inputs in ("pool", "pool-y") else yc
        X, yc = pickle.loads(pickle.dumps((xs, ys), protocol=5))
        _say(f"direct inputs as the pool sends them: X {type(X).__name__}, y {type(yc).__name__} "
             f"(prepare+pickle round trip {time.monotonic() - started:.2f}s)")
    for start in (0, 4):
        model = ml.RandomForestClassifier(n_estimators=PER_SHARD, max_depth=DEPTH, random_state=SEED,
                                          numeric_mode="identical")
        _timed(f"direct ({inputs} inputs) rf_classifier_fit_shard tree_start={start}",
               lambda: model._fit_with_tree_start(X, yc, start))
    _timed("direct plain fit (4 trees)",
           lambda: ml.RandomForestClassifier(n_estimators=PER_SHARD, max_depth=DEPTH, random_state=SEED,
                                             numeric_mode="identical").fit(X, yc))


def run_verify_layout(argv, timeout):
    forwarded = [a for a in argv if a != "--verify-layout"]
    env = dict(os.environ, MOJOLEARN_NUMERIC_MODE="identical")
    with tempfile.TemporaryDirectory(prefix="repro-par-forest-") as directory:
        log = os.path.join(directory, "worker.log")
        with open(log, "w+b") as output:
            process = subprocess.Popen([sys.executable, os.path.abspath(__file__), *forwarded], env=env,
                                       stdout=output, stderr=output, start_new_session=True)
            try:
                code = process.wait(timeout=timeout + 30)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, 9)
                code = "killed after timeout"
        with open(log, encoding="utf-8", errors="replace") as stream:
            sys.stdout.write(stream.read())
        _say(f"verify-layout child exit: {code}")


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--direct", action="store_true")
    parser.add_argument("--inputs", choices=("numpy", "pool", "pool-x", "pool-y"), default="numpy",
                        help="--direct: the fixture arrays; exactly what a pool worker receives; or only its X "
                             "(a mojolearn Array) or only its y (a Python list)")
    parser.add_argument("--no-timing", action="store_true",
                        help="do not set MOJOLEARN_POOL_TIMING=1 (parent/worker phase timings on stderr)")
    parser.add_argument("--verify-layout", action="store_true")
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument("--cpu-threads", type=int)
    parser.add_argument("--export")
    args = parser.parse_args(argv)
    if args.cpu_threads is not None:
        os.environ["MOJOLEARN_CPU_THREADS"] = str(args.cpu_threads)
    if args.export:
        os.environ["MOJOLEARN_FOREST_EXPORT"] = args.export
    os.environ.setdefault("MOJOLEARN_NUMERIC_MODE", "identical")
    if not args.no_timing:
        os.environ["MOJOLEARN_POOL_TIMING"] = "1"
    if args.verify_layout:
        run_verify_layout(argv, args.timeout)
        return
    faulthandler.dump_traceback_later(args.timeout, exit=True)
    _say(f"pid {os.getpid()} pgid {os.getpgid(0)} sid {os.getsid(0)} "
         f"stdout_tty={os.isatty(1)} MOJOLEARN_CPU_THREADS={os.environ.get('MOJOLEARN_CPU_THREADS')} "
         f"MOJOLEARN_FOREST_EXPORT={os.environ.get('MOJOLEARN_FOREST_EXPORT')}")
    X, yc = _fixture()
    if args.direct:
        run_direct(X, yc, args.inputs)
    else:
        run_pool(X, yc)
    faulthandler.cancel_dump_traceback_later()
    _say("done")


if __name__ == "__main__":
    main()
