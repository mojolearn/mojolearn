# rf-score-weighted nondeterminism, leg 4: IS IT THE CROSS-BLOCK MERGE?
# Leg 3 named the divergence: ONE node at depth 6 picks a different FEATURE, with its
# depth-7 children differing only as a downstream cascade. `Split::update` gives an
# equal-gain tie to the HIGHER colid, yet flips run both ways, so either a gain differs
# or a block's candidate is LOST in the mutex-guarded cross-block merge.
#
# find_best_splits launches ONE BLOCK PER (node, sampled column). With max_features
# set so that n_sampled_cols == 1 there is exactly ONE block per node, so there is NO
# cross-block merge to lose a candidate in. Arms sweep the column count.
#   1 column stable while 16 moves -> the cross-block merge (publish/mutex).
#   still moves at 1 column        -> within-block, upstream of the merge.
# CONFOUND, stated: fewer columns also means fewer near-tied candidates, so a null at
# 1 column is suggestive, not conclusive. Repeats are sized to make the 16-column arm
# the internal positive control in the same session.
# Placeholders: @FULL@ @WHEELNAME@ @WHEELSHA@ @WHEELURL@ @PUTURL@ @REPEATS@ @PROBESECS@
set -u
OUT=/root/gemm_leg_out/identity
mkdir -p "$OUT"
cd /root/mojolearn || exit 9
echo @FULL@ > /root/mojolearn/commit.txt
log() { echo "$(date -u +%H:%M:%SZ) $*" >> "$OUT/record.txt"; }
log "started rf cross-block merge discriminator"

PY=/root/mojolearn/.pixi/envs/default/bin/python
[ -x "$PY" ] || PY=$(command -v python3)
W=/root/wheel/@WHEELNAME@
mkdir -p /root/wheel
curl -fsS --retry 3 --max-time 600 -o "$W" '@WHEELURL@'; log "wheel_fetch_exit=$?"
got=$(sha256sum "$W" | cut -d' ' -f1); log "wheel_sha256=$got"
[ "$got" = @WHEELSHA@ ] || { log "WHEEL SHA MISMATCH"; exit 11; }
"$PY" -m venv /root/q > "$OUT/venv.log" 2>&1 || { log "venv failed"; exit 12; }
/root/q/bin/pip install --disable-pip-version-check -q "$W" numpy >> "$OUT/venv.log" 2>&1; log "pip_exit=$?"

mkdir -p /root/run && cd /root/run || exit 13
export MOJOLEARN_NUMERIC_MODE=identical MOJOLEARN_COMMIT=@FULL@ PYTHONNOUSERSITE=1
unset PYTHONPATH
/root/q/bin/python -c "import mojolearn; print(mojolearn.__version__, mojolearn.vendor())" > "$OUT/import.txt" 2>&1
log "import $(tr '\n' ' ' < "$OUT/import.txt")"

JSON="$OUT/rf_probe4.json"
upload() { [ -s "$JSON" ] || return 1; curl -fsS --max-time 180 -X PUT --upload-file "$JSON" '@PUTURL@' > /dev/null 2>&1; }
( while :; do sleep 60; if upload; then log "partial_uploaded bytes=$(wc -c < "$JSON" 2>/dev/null | tr -d ' ')"; fi; done ) &
UPLOADER=$!
log "uploader_pid=$UPLOADER"

cat > /root/run/rf_probe4.py <<'PYEOF'
"""Does the divergence survive when there is only ONE block per node?

n_sampled_cols = max(1, int(n_cols * max_features)) and find_best_splits launches one
block per (node, sampled column), so max_features = 1/16 on a 16-feature fixture leaves
a single publisher per node and no cross-block merge.
"""
import json, os, time
import numpy as np

OUT = os.environ.get("RF_PROBE_JSON", "/root/gemm_leg_out/identity/rf_probe4.json")
REPEATS = int(os.environ.get("RF_PROBE_REPEATS", "100"))
DEADLINE = time.time() + float(os.environ.get("RF_PROBE_SECS", "1500")) - 120

import mojolearn
from mojolearn import RandomForestRegressor
import mojolearn._identity_break as ib

fixture = ib.fixture
_h = ib._h
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


def sig(est):
    return {n: _h(np_of(est, n)) for n in NAMES}


results = {"version": getattr(mojolearn, "__version__", "?"),
           "commit": os.environ.get("MOJOLEARN_COMMIT", ""),
           "repeats_requested": REPEATS, "arms": [], "notes": []}


def flush():
    tmp = OUT + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(results, fh, indent=1)
    os.replace(tmp, OUT)


X, yc, yr = fixture("wide")
NCOLS = int(X.shape[1])

# (label, max_features) -> expected n_sampled_cols = max(1, int(16 * mf))
ARMS = [("cols16", 1.0), ("cols10", 10.0 / 16.0), ("cols2", 2.0 / 16.0), ("cols1", 1.0 / 16.0)]

for label, mf in ARMS:
    arm = {"label": label, "max_features": mf, "n_cols": NCOLS,
           "expected_sampled_cols": max(1, int(NCOLS * mf)),
           "runs": 0, "moved": 0, "distinct": 0, "error": None}
    results["arms"].append(arm)
    flush()
    ref = None
    seen = set()
    for i in range(REPEATS):
        if time.time() > DEADLINE:
            results["notes"].append("deadline in arm %s at repeat %d" % (label, i))
            flush()
            break
        try:
            reg = RandomForestRegressor(n_estimators=16, max_depth=8, random_state=7,
                                        max_features=mf).fit(X[:2000], yr[:2000])
            s = sig(reg)
        except Exception as exc:
            arm["error"] = repr(exc)[:300]
            flush()
            break
        key = tuple(s[n] for n in NAMES)
        seen.add(key)
        arm["runs"] += 1
        if ref is None:
            ref = key
        elif key != ref:
            arm["moved"] += 1
        arm["distinct"] = len(seen)
        flush()

results["done"] = True
flush()
print("PROBE4 DONE " + " ".join("%s:%d/%d" % (a["label"], a["moved"], a["runs"])
                                for a in results["arms"]))
PYEOF

RF_PROBE_JSON="$JSON" RF_PROBE_REPEATS=@REPEATS@ RF_PROBE_SECS=@PROBESECS@ \
  timeout -k 30 @PROBESECS@ /root/q/bin/python /root/run/rf_probe4.py > "$OUT/rf_probe4.log" 2>&1
log "rf_probe4_exit=$?"
tail -5 "$OUT/rf_probe4.log" >> "$OUT/record.txt"

kill "$UPLOADER" 2>/dev/null
if upload; then log "final_uploaded bytes=$(wc -c < "$JSON" 2>/dev/null | tr -d ' ')"; else log "final upload FAILED (the fetch is the fallback)"; fi
log "finished"
