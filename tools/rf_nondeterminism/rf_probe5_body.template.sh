# rf-score-weighted nondeterminism, leg 5: IS IT THE SECOND LAUNCH?
# Leg 4: 16 columns moved 3/100 while 10, 2 and 1 columns were each 0/100. Tie density
# cannot explain that (10 of 16 features has nearly the same candidate density and still
# merges 10 blocks per node through the mutex). What changes is the LAUNCH COUNT:
# N_BLKS_FOR_COLS = 10 caps n_blocks_dimy, and enqueue_best_splits strides c += 10, so
# 16 columns runs TWO find_best_splits launches into a split[node] slot that initSplit
# (fused into the once-per-round setup launch, DEVIATION 1916) initialized only ONCE.
#
# This leg separates "two launches" from "sixteen columns" by straddling the boundary:
#   10 columns -> 1 launch     11 columns -> 2 launches (10 + 1)
# 11 and 10 have nearly identical tie density, so a move at 11 and none at 10 isolates
# the second launch and kills the tie-density confound outright.
# Placeholders: @FULL@ @WHEELNAME@ @WHEELSHA@ @WHEELURL@ @PUTURL@ @REPEATS@ @PROBESECS@
set -u
OUT=/root/gemm_leg_out/identity
mkdir -p "$OUT"
cd /root/mojolearn || exit 9
echo @FULL@ > /root/mojolearn/commit.txt
log() { echo "$(date -u +%H:%M:%SZ) $*" >> "$OUT/record.txt"; }
log "started rf launch-count discriminator"

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

JSON="$OUT/rf_probe5.json"
upload() { [ -s "$JSON" ] || return 1; curl -fsS --max-time 180 -X PUT --upload-file "$JSON" '@PUTURL@' > /dev/null 2>&1; }
( while :; do sleep 60; if upload; then log "partial_uploaded bytes=$(wc -c < "$JSON" 2>/dev/null | tr -d ' ')"; fi; done ) &
UPLOADER=$!
log "uploader_pid=$UPLOADER"

cat > /root/run/rf_probe5.py <<'PYEOF'
"""Straddle the N_BLKS_FOR_COLS = 10 boundary.

n_sampled_cols = max(1, int(n_cols * max_features)); the split search loops
c += 10, so n_sampled_cols <= 10 is ONE find_best_splits launch and 11..20 is TWO.
Arms 10 and 11 differ by a single feature but differ in launch count.
"""
import json, os, time
import numpy as np

OUT = os.environ.get("RF_PROBE_JSON", "/root/gemm_leg_out/identity/rf_probe5.json")
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
    return tuple(_h(np_of(est, n)) for n in NAMES)


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

# n_sampled_cols -> launches = ceil(cols / 10)
for want in (10, 11, 12, 16):
    mf = want / float(NCOLS)
    arm = {"label": "cols%d" % want, "max_features": mf,
           "expected_sampled_cols": max(1, int(NCOLS * mf)),
           "expected_launches": (max(1, int(NCOLS * mf)) + 9) // 10,
           "runs": 0, "moved": 0, "distinct": 0, "error": None}
    results["arms"].append(arm)
    flush()
    ref = None
    seen = set()
    for i in range(REPEATS):
        if time.time() > DEADLINE:
            results["notes"].append("deadline in %s at repeat %d" % (arm["label"], i))
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
        seen.add(s)
        arm["runs"] += 1
        if ref is None:
            ref = s
        elif s != ref:
            arm["moved"] += 1
        arm["distinct"] = len(seen)
        flush()

results["done"] = True
flush()
print("PROBE5 DONE " + " ".join("%s(L%d):%d/%d" % (a["label"], a["expected_launches"],
                                                   a["moved"], a["runs"])
                                for a in results["arms"]))
PYEOF

RF_PROBE_JSON="$JSON" RF_PROBE_REPEATS=@REPEATS@ RF_PROBE_SECS=@PROBESECS@ \
  timeout -k 30 @PROBESECS@ /root/q/bin/python /root/run/rf_probe5.py > "$OUT/rf_probe5.log" 2>&1
log "rf_probe5_exit=$?"
tail -5 "$OUT/rf_probe5.log" >> "$OUT/record.txt"

kill "$UPLOADER" 2>/dev/null
if upload; then log "final_uploaded bytes=$(wc -c < "$JSON" 2>/dev/null | tr -d ' ')"; else log "final upload FAILED (the fetch is the fallback)"; fi
log "finished"
