"""One cell of the ExtraTrees cross-block mutex A/B.

WHAT THIS MEASURES. `extratrees/impl/decisiontree/batched_levelalgo/split.mojo` claims
the per-node device mutex from `bpn = ceildiv(k, TPB)` blocks, `k` the sampled column
count and `TPB` 512 on a 64-lane wavefront. At `k <= TPB` there is exactly ONE claimant
per mutex and the ordering hole of DEVIATION 106 cannot fire whatever the memory model
does. So the column count is not a nuisance parameter here, it is the SWITCH, and this
driver takes it as an arm.

CELL = (binary arm, column count). The binary arm is the build define. The column arm is
`n_cols`, and a cell below the threshold is the control that shares the binary with the
cell above it: if the stock binary moves at `cols=640` and is stable at `cols=256`, the
difference is the multi-block merge and not the build.

Env: ET_JSON ET_ARM ET_COLS ET_REPEATS ET_SECS ET_SO_SHA ET_ROWS ET_TREES ET_DEPTH
"""
import json, os, time
import numpy as np

OUT = os.environ["ET_JSON"]
ARM = os.environ["ET_ARM"]
COLS = int(os.environ["ET_COLS"])
ROWS = int(os.environ.get("ET_ROWS", "2000"))
TREES = int(os.environ.get("ET_TREES", "16"))
DEPTH = int(os.environ.get("ET_DEPTH", "8"))
SO_SHA = os.environ.get("ET_SO_SHA", "")
REPEATS = int(os.environ.get("ET_REPEATS", "300"))
_secs = float(os.environ.get("ET_SECS", "1800"))
# The fetch reserve is PROPORTIONAL. A flat 90 s reserve turned a 200 s control
# cell into 110 s, and a 60 s rehearsal into zero fits before the first one ran.
DEADLINE = time.time() + _secs - max(15.0, min(90.0, 0.15 * _secs))

import mojolearn
from mojolearn import ExtraTreesRegressor

NAMES = ("_offsets", "_colid", "_quesval", "_left_child", "_leaves")


def np_of(est, name):
    v = getattr(est, name)
    if isinstance(v, np.ndarray):
        return v
    for a in (lambda: np.asarray(memoryview(v)),
              lambda: np.frombuffer(v.tobytes(), dtype=np.dtype(v.dtype)),
              lambda: np.asarray(list(v))):
        try:
            return a()
        except Exception:
            continue
    raise RuntimeError("cannot view " + name)


def digest(est):
    import hashlib
    h = hashlib.sha256()
    for n in NAMES:
        a = np.ascontiguousarray(np_of(est, n))
        h.update(n.encode())
        h.update(str(a.dtype).encode())
        h.update(str(a.shape).encode())
        h.update(a.tobytes())
    return h.hexdigest()


try:
    with open(OUT) as fh:
        doc = json.load(fh)
except Exception:
    doc = {"cells": [], "notes": []}


def flush():
    tmp = OUT + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(doc, fh, indent=1)
    os.replace(tmp, OUT)


# The fixture is generated here, not staged, so the cell is reproducible from the
# driver alone. Continuous columns, no ties by construction at float32 resolution.
rng = np.random.default_rng(20260916)
X = rng.standard_normal((ROWS, COLS), dtype=np.float32)
y = (X[:, : min(8, COLS)].sum(axis=1) + 0.25 * rng.standard_normal(ROWS)).astype(np.float32)

# bpn as the binary WILL compute it. TPB is 512 on a 64-lane wavefront, 128 on 32.
tpb = int(os.environ.get("ET_TPB", "512"))
bpn = max(1, -(-COLS // tpb))

rec = {"arm": ARM, "cols": COLS, "rows": ROWS, "trees": TREES, "depth": DEPTH,
       "tpb_assumed": tpb, "bpn_expected": bpn, "contends": bpn > 1,
       "so_sha256": SO_SHA[:16], "runs": 0, "moved": 0, "distinct": 0,
       "error": None, "version": getattr(mojolearn, "__version__", "?"),
       "first": None}
doc["cells"].append(rec)
flush()

ref = None
seen = set()
for i in range(REPEATS):
    if i > 0 and time.time() > DEADLINE:
        doc["notes"].append("deadline %s cols=%d at %d" % (ARM, COLS, i))
        flush()
        break
    try:
        est = ExtraTreesRegressor(n_estimators=TREES, max_depth=DEPTH,
                                  random_state=7).fit(X, y)
        key = digest(est)
    except Exception as exc:
        rec["error"] = repr(exc)[:400]
        flush()
        break
    seen.add(key)
    rec["runs"] += 1
    if ref is None:
        ref = key
        rec["first"] = key[:16]
    elif key != ref:
        rec["moved"] += 1
    rec["distinct"] = len(seen)
    flush()

doc["done_%s_cols%d" % (ARM, COLS)] = True
flush()
print("CELL %s cols=%d bpn=%d %d/%d moved distinct=%d"
      % (ARM, COLS, bpn, rec["moved"], rec["runs"], rec["distinct"]))
