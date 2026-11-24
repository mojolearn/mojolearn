#!/usr/bin/env python3
"""Diagnostic (r6-proving, 2026-11-23): which test of tests/proving/test_proving_categories.py leaves process
state that changes a later forest regressor's `digest_` against its saved file's CPU twin.

L40S finding (orchestrator, wheel 4519097d380a): `pytest tests/proving -p no:xdist` fails the forest regressor twin
digests (RandomForestRegressor, ExtraTreesRegressor; classifiers pass) only when test_proving_categories.py ran
earlier in the same process; tools/ops/forest_digest_diff.py alone is SAME on 28 of 28 cases.

What it does: collects the test ids of the categories file, then for EACH id starts one fresh process that runs that
one test (pytest, in-process), then the forest cases of forest_digest_diff.py (RandomForestRegressor and
ExtraTreesRegressor, 5 trees, 400 x 6, device cpu and gpu) and a floating-point environment probe before and after
the test. The ids are independent processes, run across every core of the allocation (worker count from the
process's CPU affinity, each worker one small fit process of a few hundred MB). Testing tool only; it times nothing.

    python tools/ops/digest_order_bisect.py <tests tree, e.g. /root/mojolearn-ab/tests> <out.txt>

What decides: lines `ORDER <test id> SAME|DIFFERENT fp_before=<..> fp_after=<..>`. The ids marked DIFFERENT are the
tests that leave the state. If fp_after differs from fp_before on exactly those ids, the state is the calling
thread's floating-point environment (flush-to-zero / denormals-are-zero / rounding mode) left set by that test's
GBDT fit or refusal. Otherwise the FIELD lines under each DIFFERENT case name the side (model-before, loaded, twin,
model-after) and the fitted field whose bytes moved. `SUMMARY` gives ids, DIFFERENT count and errors.
"""
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
FOREST_CASES = [("RandomForestRegressor", dict(n_estimators=5, random_state=0), "r"),
                ("ExtraTreesRegressor", dict(n_estimators=5, random_state=0), "r"),
                ("RandomForestClassifier", dict(n_estimators=5, random_state=0), "c")]


def fp_probe():
    """Three float32 results that depend on the thread's floating-point environment: a subnormal product
    (0 under flush-to-zero), a subnormal operand (0 under denormals-are-zero) and a half-way sum (rounding)."""
    import numpy as np
    tiny = np.float32(1e-39)                        # a float32 subnormal
    product = np.float32(1e-20) * np.float32(1e-19)  # subnormal result
    half = np.float32(1.0) + np.float32(2.0 ** -24) * np.float32(1.5)
    return f"{float(product):.3e}/{float(tiny * np.float32(1.0)):.3e}/{float(half).hex()}"


def one(test_id, tests):
    """Child: run `test_id`, then the forest cases; print the ORDER line and the case lines."""
    import tempfile
    import pytest
    sys.path.insert(0, str(HERE))
    import forest_digest_diff as diff
    import mojolearn
    before = fp_probe()
    os.chdir(Path(tests).parent)                # the repo root: the ids read tests/proving/...
    code = pytest.main([test_id, "-q", "-p", "no:xdist", "-p", "no:cacheprovider", "-p", "no:randomly"])
    after = fp_probe()
    lines = []
    devices = ["cpu", "gpu"] if mojolearn.vendor() != "cpu" else ["cpu"]
    with tempfile.TemporaryDirectory() as folder:
        for name, params, target in FOREST_CASES:   # glue: three tiny fits per device in one child
            for device in devices:
                diff.case(lines.append, name, params, target, device, 400, 6, folder)
    verdict = "DIFFERENT" if any(line.startswith("CASE") and line.endswith("DIFFERENT") for line in lines) else "SAME"
    print(f"ORDER {test_id} {verdict} pytest_rc={int(code)} fp_before={before} fp_after={after}", flush=True)
    for line in lines:
        if verdict == "DIFFERENT" and not line.lstrip().startswith("MEMBER"):
            print("  " + line, flush=True)


def main():
    if len(sys.argv) == 4 and sys.argv[1] == "--one":
        one(sys.argv[2], sys.argv[3])
        return 0
    tests, out = Path(sys.argv[1]).resolve(), Path(sys.argv[2])
    collected = subprocess.run([sys.executable, "-m", "pytest", "--collect-only", "-q", "-p", "no:cacheprovider",
                                "tests/proving/test_proving_categories.py"], cwd=tests.parent, capture_output=True, text=True)
    ids = [line.strip() for line in collected.stdout.splitlines() if "::" in line]
    ids = ["control::none"] + ids                   # a control child that runs no test first
    workers = max(1, len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else os.cpu_count() or 1)

    def run(test_id):
        if test_id == "control::none":
            test_id = "tests/proving/test_proving_categories.py::__no_such_test__"
        child = subprocess.run([sys.executable, str(Path(__file__).resolve()), "--one", test_id, str(tests)],
                               capture_output=True, text=True, timeout=900)
        # Only the ORDER line and the case lines under it: pytest's own output also has indented lines.
        lines = child.stdout.splitlines()
        start = next((i for i, line in enumerate(lines) if line.startswith("ORDER")), None)
        kept = [] if start is None else [lines[start]] + [line for line in lines[start + 1:] if line.startswith("  ")]
        if not any(line.startswith("ORDER") for line in kept):
            kept = [f"ORDER {test_id} ERROR rc={child.returncode} " + (child.stderr.strip().splitlines() or [""])[-1]]
        return kept

    with ThreadPoolExecutor(max_workers=min(workers, len(ids))) as pool, open(out, "w") as sink:
        results = list(pool.map(run, ids))
        for kept in results:
            sink.write("\n".join(kept) + "\n")
        different = sum(kept[0].split()[2] == "DIFFERENT" for kept in results)
        errors = sum(kept[0].split()[2] == "ERROR" for kept in results)
        sink.write(f"SUMMARY ids={len(ids) - 1} workers={min(workers, len(ids))} different={different} errors={errors}\n")
    print(out.read_text().splitlines()[-1])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
