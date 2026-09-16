"""ONE ROUND OF ONE ARM of the gfx942 RF mutex-claim A/B.

Each round is its own process, so the reference fit it compares against is its
own first fit (the defect is a within-process, same-seed, same-data mover).
Rounds alternate arms with the ORDER ROTATED, so a box-level drift cannot be
read as the effect.  Every record carries RF_ROUND_TOKEN so a stale R2 object
from an earlier leg cannot be mistaken for this one's numbers.
"""
import importlib.util, json, os, time
import numpy as np

OUTDIR = os.environ["RF_ROUND_OUTDIR"]
ARM = os.environ["RF_ROUND_ARM"]
CFG = os.environ["RF_ROUND_CFG"]           # cols16 | cols10
ROUND = int(os.environ["RF_ROUND_INDEX"])
REPEATS = int(os.environ["RF_ROUND_REPEATS"])
MAXF = float(os.environ["RF_ROUND_MAXF"])
SO_SHA = os.environ.get("RF_ROUND_SO_SHA", "")
SECTIONS = os.environ.get("RF_ROUND_SECTIONS", "")
TOKEN = os.environ["RF_ROUND_TOKEN"]
DEADLINE = float(os.environ["RF_ROUND_DEADLINE"])   # absolute unix epoch

import mojolearn
from mojolearn import RandomForestRegressor

_spec = importlib.util.spec_from_file_location(
    "ib", "/root/mojolearn/tools/identity_break.py")
ib = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ib)
fixture, _h = ib.fixture, ib._h
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


X, yc, yr = fixture("wide")
Xt, yt = X[:2000], yr[:2000]

rec = {"token": TOKEN, "arm": ARM, "cfg": CFG, "round": ROUND,
       "max_features": MAXF, "columns": int(Xt.shape[1]),
       "so_sha256": SO_SHA[:16], "sections_sha256": SECTIONS[:16],
       "fits": 0, "comparisons": 0, "moved": 0, "distinct": 0,
       "error": None, "truncated": False,
       "version": getattr(mojolearn, "__version__", "?"),
       "started": time.time(), "seconds_per_fit": None}
path = os.path.join(OUTDIR, "r%02d_%s_%s.json" % (ROUND, CFG, ARM))


def flush():
    tmp = path + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(rec, fh, indent=1)
    os.replace(tmp, path)


flush()
ref = None
seen = set()
t0 = time.time()
for i in range(REPEATS):
    if time.time() > DEADLINE:
        rec["truncated"] = True
        flush()
        break
    try:
        kw = {} if MAXF >= 1.0 else {"max_features": MAXF}
        reg = RandomForestRegressor(n_estimators=16, max_depth=8,
                                    random_state=7, **kw).fit(Xt, yt)
        key = tuple(_h(np.array(np_of(reg, n), copy=True)) for n in NAMES)
    except Exception as exc:
        rec["error"] = repr(exc)[:400]
        flush()
        break
    seen.add(key)
    rec["fits"] += 1
    if ref is None:
        ref = key
    else:
        rec["comparisons"] += 1
        if key != ref:
            rec["moved"] += 1
    rec["distinct"] = len(seen)
    rec["seconds_per_fit"] = round((time.time() - t0) / rec["fits"], 4)
    flush()

rec["finished"] = time.time()
rec["hashes"] = sorted("|".join(k)[:24] for k in seen)
flush()
print("ROUND %s r%02d %s %s moved=%d/%d distinct=%d spf=%s err=%s"
      % (TOKEN, ROUND, CFG, ARM, rec["moved"], rec["comparisons"],
         rec["distinct"], rec["seconds_per_fit"], rec["error"]))
