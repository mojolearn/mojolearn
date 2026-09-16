# DOES THE RACE EXIST IN A SHIPPED RELEASE? Install mojolearn 0.8.5 FROM PyPI and ask.
#
# Code says yes already: the fit path is semantically unchanged since v0.8.5 (the only
# diff in split.mojo and the builder's blocking loop is comment text from the "retire the
# port framing" sweep), every tag v0.8.0..v0.8.5 carries the same N_BLKS_FOR_COLS,
# _publish_to_global and update, and the published 0.8.5 wheel ships
# mojolearn/hip/gfx942/identical/_mojolearn_rf.so with RandomForestRegressor defaulting to
# max_features=1.0. This leg MEASURES it instead of inferring it.
#
# Installs from PyPI, not from a staged copy, so it tests what users actually receive.
#
# ARMS (wide fixture, 2000 rows, 16 trees, depth 8 -- the lane's own configuration):
#   cols16  max_features=1.0   THE SHIPPED DEFAULT. A user sets nothing to get this.
#   cols10  max_features=0.625 INTERNAL CONTROL: <=10 columns was 0/600 on 0.8.6, so this
#           must stay stable. If BOTH arms move, the probe is not measuring what I think.
#           If NEITHER moves, the probe is insensitive on this wheel and proves nothing.
#
# 0.8.5 does NOT ship mojolearn/_identity_break.py (that packaging is newer), so the
# harness is loaded from the shipped source bundle by path.
# Placeholders: @FULL@ @PUTURL@ @REPEATS@ @PROBESECS@
set -u
OUT=/root/gemm_leg_out/identity
mkdir -p "$OUT"
cd /root/mojolearn || exit 9
echo @FULL@ > /root/mojolearn/commit.txt
log() { echo "$(date -u +%H:%M:%SZ) $*" >> "$OUT/record.txt"; }
log "started SHIPPED-RELEASE check: mojolearn==0.8.5 from PyPI"

PY=/root/mojolearn/.pixi/envs/default/bin/python
[ -x "$PY" ] || PY=$(command -v python3)
"$PY" -m venv /root/q > "$OUT/venv.log" 2>&1 || { log "venv failed"; exit 12; }

# THE POINT OF THE LEG: the published artifact, from the public index.
/root/q/bin/pip install --disable-pip-version-check -q "mojolearn==0.8.5" numpy \
    > "$OUT/pip.log" 2>&1
log "pip_install_0.8.5_exit=$?"
tail -3 "$OUT/pip.log" >> "$OUT/record.txt" 2>/dev/null

mkdir -p /root/run && cd /root/run || exit 13
export MOJOLEARN_NUMERIC_MODE=identical PYTHONNOUSERSITE=1
unset PYTHONPATH
/root/q/bin/python -c "
import mojolearn, os, hashlib
print('version', mojolearn.__version__)
try: print('vendor', mojolearn.vendor())
except Exception as e: print('vendor ERR', e)
p = os.path.dirname(mojolearn.__file__)
so = os.path.join(p, 'hip', 'gfx942', 'identical', '_mojolearn_rf.so')
print('rf_so_present', os.path.exists(so))
if os.path.exists(so):
    print('rf_so_sha256', hashlib.sha256(open(so,'rb').read()).hexdigest())
" > "$OUT/installed.txt" 2>&1
log "installed: $(tr '\n' ' ' < "$OUT/installed.txt" | cut -c1-200)"

JSON="$OUT/rf_probe085.json"
upload() { [ -s "$JSON" ] || return 1; curl -fsS --max-time 180 -X PUT --upload-file "$JSON" '@PUTURL@' > /dev/null 2>&1; }
( while :; do sleep 60; if upload; then log "partial_uploaded bytes=$(wc -c < "$JSON" 2>/dev/null | tr -d ' ')"; fi; done ) &
UPLOADER=$!
log "uploader_pid=$UPLOADER"

cat > /root/run/rf_probe085.py <<'PYEOF'
"""Is the rf fit race present in the PUBLISHED 0.8.5 wheel?"""
import importlib.util, json, os, time
import numpy as np

OUT = os.environ.get("RF_PROBE_JSON", "/root/gemm_leg_out/identity/rf_probe085.json")
REPEATS = int(os.environ.get("RF_PROBE_REPEATS", "300"))
DEADLINE = time.time() + float(os.environ.get("RF_PROBE_SECS", "1200")) - 90

import mojolearn
from mojolearn import RandomForestRegressor

# 0.8.5 predates the packaged harness; load it from the shipped source bundle.
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


results = {"version": getattr(mojolearn, "__version__", "?"),
           "question": "does the rf fit race exist in the PUBLISHED 0.8.5 wheel",
           "arms": [], "notes": []}


def flush():
    tmp = OUT + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(results, fh, indent=1)
    os.replace(tmp, OUT)


X, yc, yr = fixture("wide")
NCOLS = int(X.shape[1])

# cols16 is max_features=1.0, the SHIPPED DEFAULT. cols10 is the control.
for want, label in ((16, "cols16_SHIPPED_DEFAULT"), (10, "cols10_CONTROL")):
    mf = want / float(NCOLS)
    rec = {"label": label, "cols": want, "max_features": mf,
           "is_shipped_default": want == NCOLS,
           "runs": 0, "moved": 0, "distinct": 0, "shape_changes": 0, "error": None}
    results["arms"].append(rec)
    flush()
    ref = None
    seen = set()
    for i in range(REPEATS):
        if time.time() > DEADLINE:
            results["notes"].append("deadline %s at %d" % (label, i))
            flush()
            break
        try:
            kw = {} if want == NCOLS else {"max_features": mf}
            reg = RandomForestRegressor(n_estimators=16, max_depth=8,
                                        random_state=7, **kw).fit(X[:2000], yr[:2000])
            cur = {n: np.array(np_of(reg, n), copy=True) for n in NAMES}
        except Exception as exc:
            rec["error"] = repr(exc)[:300]
            flush()
            break
        key = tuple(_h(cur[n]) for n in NAMES)
        seen.add(key)
        rec["runs"] += 1
        if ref is None:
            ref, ref_n = key, len(cur["_colid"])
        elif key != ref:
            rec["moved"] += 1
            if len(cur["_colid"]) != ref_n:
                rec["shape_changes"] += 1
        rec["distinct"] = len(seen)
        flush()

results["done"] = True
flush()
print("0.8.5 PROBE DONE " + " ".join(
    "%s %d/%d" % (a["label"], a["moved"], a["runs"]) for a in results["arms"]))
PYEOF

RF_PROBE_JSON="$JSON" RF_PROBE_REPEATS=@REPEATS@ RF_PROBE_SECS=@PROBESECS@ \
  timeout -k 30 @PROBESECS@ /root/q/bin/python /root/run/rf_probe085.py > "$OUT/rf_probe085.log" 2>&1
log "rf_probe085_exit=$?"
tail -6 "$OUT/rf_probe085.log" >> "$OUT/record.txt"

kill "$UPLOADER" 2>/dev/null
if upload; then log "final_uploaded bytes=$(wc -c < "$JSON" 2>/dev/null | tr -d ' ')"; else log "final upload FAILED (the fetch is the fallback)"; fi
log "finished"
