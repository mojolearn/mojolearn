#!/usr/bin/env python3
"""Diagnostic (r6-proving, 2026-11-23): what state test_object_array_matches_list_rows leaves for HostForest.digest_.

The bisect (tools/ops/digest_order_bisect.py, L40S wheel 4519097d380a) found one test that changes the digest of
every later HostForest twin (regressors and classifiers; the model and its own loaded copy keep their digest), with
the twin's fields and arrays byte-equal and no floating-point mode change. This runs each PRELUDE below in its own
fresh process, then fits RandomForestRegressor (5 trees, 400 x 6, device cpu), saves it, loads its twin, and prints
the twin's digest verdict beside the registration state the twin digest reads (python/mojolearn/_digest_fields.py
CANONICAL / SCHEMAS / CONNECTION, python/mojolearn/_native_family_layouts.py LAYOUTS, and which mojolearn modules
were imported before the twin). The preludes are independent processes, run across the allocation's cores.

    python tools/ops/digest_twin_state_probe.py <tests tree, e.g. /root/mojolearn-ab/tests> <out.txt>

What decides: lines `PRELUDE <name> SAME|DIFFERENT ...`. If `import-classical-host` (or `import-forest-host`) is
DIFFERENT, the cause is the import order of mojolearn._classical_host / _forest_host (a twin registration that
depends on which module loads first), not the GBDT fit; the `state` fields then show which registration differs
from `control`. If only `gbdt-object-fit*` are DIFFERENT, the state is left by the object-array fit (fit-only vs
fit+predict says which call).
"""
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

PRELUDES = {
    "control": "",
    "import-classical-host": "import mojolearn._classical_host",
    "import-forest-host": "import mojolearn._forest_host",
    "import-gbdt-host": "import mojolearn._gbdt_host",
    "gbdt-list-fit-predict": "rows=[[c[i], float(n[i])] for i in range(N)]; m=G().fit(rows, y); m.predict(rows)",
    "gbdt-object-fit": "X=np.empty((N,2),dtype=object); X[:,0]=c; X[:,1]=n; G().fit(X, y)",
    "gbdt-object-fit-predict": "X=np.empty((N,2),dtype=object); X[:,0]=c; X[:,1]=n; m=G().fit(X, y); m.predict(X)",
    "pytest-polluter": "import pytest; pytest.main([TESTS + '/proving/test_proving_categories.py::"
                       "test_object_array_matches_list_rows', '-q', '-p', 'no:xdist', '-p', 'no:cacheprovider'])",
}

CHILD = r'''
import os, sys, tempfile
import numpy as np
import mojolearn
TESTS = sys.argv[2]
N = 200
rng = np.random.default_rng(3)
c = [("red", "green", "blue", "violet")[i % 4] for i in range(N)]
n = rng.normal(size=N).astype(np.float64)
y = np.array([float(i % 4) + 0.1 * n[i] for i in range(N)], dtype=np.float32)
G = lambda: mojolearn.GradientBoosting(n_estimators=8, max_depth=3, cat_features=[0], device="cpu")
os.chdir(os.path.dirname(TESTS))
exec(sys.argv[1])
before = sorted(m for m in sys.modules if m.startswith("mojolearn._") and ("host" in m or "digest" in m))
r = np.random.default_rng(0); X = r.random((400, 6), dtype=np.float32); Y = X @ np.arange(1, 7, dtype=np.float32)
model = mojolearn.RandomForestRegressor(n_estimators=5, random_state=0, device="cpu").fit(X, Y)
path = os.path.join(tempfile.mkdtemp(), "rf.npz"); model.save(path)
twin = mojolearn.host_model(path)
from mojolearn import _digest_fields as d, _native_family_layouts as lay
H = type(twin)
ctx = d.CONNECTION["context"] or {}
state = dict(twin_class=H.__module__ + "." + H.__qualname__, canonical=d.CANONICAL.get(H), schema=H in d.SCHEMAS,
             layout=H in lay.LAYOUTS, ctx_canonical_is=ctx.get("digest_canonical") is d.CANONICAL,
             ctx_schemas_is=ctx.get("digest_schemas") is d.SCHEMAS, same_class_objects=sum(
                 1 for m in list(sys.modules.values()) if getattr(m, "HostForest", None) is not None
                 and getattr(m, "HostForest") is not H))
verdict = "SAME" if twin.digest_ == model.digest_ else "DIFFERENT"
print("RESULT", verdict, "state=" + repr(state), "loaded_before_twin=" + ",".join(before), flush=True)
'''


def main():
    tests, out = str(Path(sys.argv[1]).resolve()), Path(sys.argv[2])
    workers = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else (os.cpu_count() or 1)

    def run(item):
        name, code = item
        child = subprocess.run([sys.executable, "-c", CHILD, code, tests], capture_output=True, text=True,
                               timeout=900)
        found = [line for line in child.stdout.splitlines() if line.startswith("RESULT")]
        if not found:
            return f"PRELUDE {name} ERROR rc={child.returncode} " + (child.stderr.strip().splitlines() or [""])[-1]
        return f"PRELUDE {name} " + found[-1][len("RESULT "):]

    with ThreadPoolExecutor(max_workers=max(1, min(workers, len(PRELUDES)))) as pool:
        lines = list(pool.map(run, PRELUDES.items()))
    out.write_text("\n".join(lines) + "\n")
    print("\n".join(line[:200] for line in lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
