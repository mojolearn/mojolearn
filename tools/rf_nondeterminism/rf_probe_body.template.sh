# rf-score-weighted nondeterminism probe, one AMD MI300X leg.
# Installs the 0.8.6 RELEASE WHEEL from R2 (the exact bytes the blocked record used), then
# runs a pure-Python probe that repeats the lane's two fits many times and hashes the
# EXPORTED MODEL ARRAYS, not just the final score, so a move is localized to a stage.
# The probe recomputes the lane's OWN cell hash, so a reproduction is proved against the
# recorded values 49be8ea935a47640 / 50ce4a9f62cddf8e rather than merely being self-consistent.
# Placeholders: @FULL@ @WHEELNAME@ @WHEELSHA@ @WHEELURL@ @PUTURL@ @REPEATS@ @PROBESECS@
set -u
OUT=/root/gemm_leg_out/identity
mkdir -p "$OUT"
cd /root/mojolearn || exit 9
echo @FULL@ > /root/mojolearn/commit.txt
log() { echo "$(date -u +%H:%M:%SZ) $*" >> "$OUT/record.txt"; }
log "started rf-score-weighted nondeterminism probe"

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
/root/q/bin/python -c "import mojolearn,sys; print(mojolearn.__version__, mojolearn.vendor(), mojolearn.__file__)" > "$OUT/import.txt" 2>&1
log "import $(tr '\n' ' ' < "$OUT/import.txt")"
{ rocm-smi --showproductname 2>&1; rocminfo 2>/dev/null | grep -m6 -E "Name:|gfx|Compute Unit"; } > "$OUT/gpu.txt" 2>&1 || true
log "gpu $(tr '\n' ' ' < "$OUT/gpu.txt" | cut -c1-160)"

JSON="$OUT/rf_probe.json"
# Best-effort periodic upload so a dead box never costs the whole leg.
upload() { [ -s "$JSON" ] || return 1; curl -fsS --max-time 180 -X PUT --upload-file "$JSON" '@PUTURL@' > /dev/null 2>&1; }
( while :; do sleep 60; if upload; then log "partial_uploaded bytes=$(wc -c < "$JSON" 2>/dev/null | tr -d ' ')"; fi; done ) &
UPLOADER=$!
log "uploader_pid=$UPLOADER"

cat > /root/run/rf_probe.py <<'PYEOF'
"""Localize the rf-score-weighted move to a stage of the fit.

The lane hashes score() only, which is the LAST stage. This probe repeats the same
two fits many times and hashes, per repeat:
  - the lane's own cell hash, recomputed with the lane's own helpers, so a move is
    proved against the recorded 49be8ea935a47640 / 50ce4a9f62cddf8e
  - each of the five exported forest model arrays, separately and per tree
  - the raw predict() vector on the lane's scoring rows
so a move says WHICH array moved (split choice vs float value) and WHICH tree.

Everything is written after every repeat, so a killed box still yields evidence.
"""
import hashlib, json, os, sys, time
import numpy as np

OUT = os.environ.get("RF_PROBE_JSON", "/root/gemm_leg_out/identity/rf_probe.json")
REPEATS = int(os.environ.get("RF_PROBE_REPEATS", "40"))
DEADLINE = time.time() + float(os.environ.get("RF_PROBE_SECS", "1500")) - 120

import mojolearn
from mojolearn import RandomForestClassifier, RandomForestRegressor
import mojolearn._identity_break as ib

fixture = ib.fixture
_h = ib._h
_hw = ib._hw
_train_hash = getattr(ib, "_train_hash", None)

FOREST_ARRAYS = ("_offsets", "_colid", "_quesval", "_left_child", "_leaves")

# The values the three agreeing columns recorded, and the odd ones AMD produced.
KNOWN = {"49be8ea935a47640": "STABLE(wide) nvidia+cpu+amd-leg4",
         "50ce4a9f62cddf8e": "ODD(wide) amd-leg3-fit2",
         "d744878e7c0e31ee": "STABLE(base) nvidia+cpu+amd-leg3",
         "eb475cefa0f32408": "ODD(base) amd-leg4-fit2"}


def np_of(est, name):
    """The exported model array as a numpy view, whatever container it uses."""
    v = getattr(est, name)
    if isinstance(v, np.ndarray):
        return v
    for attempt in (lambda: np.asarray(memoryview(v)),
                    lambda: np.frombuffer(v.tobytes(), dtype=np.dtype(v.dtype)),
                    lambda: np.asarray(list(v))):
        try:
            return attempt()
        except Exception:
            continue
    raise RuntimeError("cannot view " + name)


def model_hashes(est):
    """Per-array hashes, plus per-tree hashes sliced by the offsets table."""
    out, arrs = {}, {}
    for name in FOREST_ARRAYS:
        try:
            a = np_of(est, name)
            arrs[name] = a
            out[name] = _h(a)
        except Exception as exc:
            out[name] = "ERR:" + type(exc).__name__
    try:
        off = np.asarray(arrs["_offsets"]).astype(np.int64)
        nnode = len(arrs["_colid"])
        leaves = np.asarray(arrs["_leaves"])
        nout = max(1, len(leaves) // max(1, nnode))
        out["_per_tree_struct"] = [
            _h(arrs["_colid"][int(off[t]):int(off[t + 1])],
               arrs["_left_child"][int(off[t]):int(off[t + 1])])
            for t in range(len(off) - 1)]
        out["_per_tree_quesval"] = [
            _h(arrs["_quesval"][int(off[t]):int(off[t + 1])]) for t in range(len(off) - 1)]
        out["_per_tree_leaves"] = [
            _h(leaves[int(off[t]) * nout:int(off[t + 1]) * nout]) for t in range(len(off) - 1)]
        out["_n_nodes"] = int(nnode)
        out["_n_trees"] = int(len(off) - 1)
    except Exception as exc:
        out["_per_tree_error"] = repr(exc)[:200]
    return out


def lane_parts(clf, reg, X, yc, yr):
    """The lane's eight parts, built exactly as tools/identity_break.py builds them."""
    lo, hi = 2000, 3024
    Xs = np.ascontiguousarray(X[lo:hi])
    ycs = np.ascontiguousarray(yc[lo:hi])
    yrs = np.ascontiguousarray(yr[lo:hi])
    w = _hw((hi - lo,), "rf-score-weighted:w", 0.25, 4.0)
    wz = w.copy()
    wz[::7] = np.float32(0.0)
    ones = np.ones(hi - lo, dtype=np.float32)
    parts, extra = {}, {}
    for kind, m, y in (("clf", clf, ycs), ("reg", reg, yrs)):
        parts[kind + "_weighted"] = _h(np.float64(m.score(Xs, y, sample_weight=w)))
        parts[kind + "_zeroed"] = _h(np.float64(m.score(Xs, y, sample_weight=wz)))
        parts[kind + "_unit"] = _h(np.float64(m.score(Xs, y, sample_weight=ones)))
        parts[kind + "_unweighted"] = _h(np.float64(m.score(Xs, y)))
        extra[kind + "_predict"] = _h(np.asarray(m.predict(Xs)))
    return parts, extra


results = {"vendor": os.environ.get("MOJOLEARN_TARGET_COLUMN", "amd"),
           "commit": os.environ.get("MOJOLEARN_COMMIT", ""),
           "version": getattr(mojolearn, "__version__", "?"),
           "repeats_requested": REPEATS, "runs": [], "nsweep": [], "notes": []}


def flush():
    tmp = OUT + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(results, fh, indent=1)
    os.replace(tmp, OUT)


def one_repeat(X, yc, yr, n):
    clf = RandomForestClassifier(n_estimators=16, max_depth=8, random_state=7).fit(X[:n], yc[:n])
    reg = RandomForestRegressor(n_estimators=16, max_depth=8, random_state=7).fit(X[:n], yr[:n])
    parts, extra = lane_parts(clf, reg, X, yc, yr)
    rec = {"parts": parts, "predict": extra,
           "clf_model": model_hashes(clf), "reg_model": model_hashes(reg)}
    if _train_hash is not None:
        cell = _train_hash(parts)
        rec["cell_hash"] = cell
        rec["cell_known_as"] = KNOWN.get(cell, "UNSEEN")
    return rec


# -- Stage 1: the lane's own configuration, many repeats, both fixtures that moved.
for kind in ("wide", "base"):
    X, yc, yr = fixture(kind)
    results["notes"].append("fixture %s X=%s" % (kind, tuple(X.shape)))
    flush()
    for i in range(REPEATS):
        if time.time() > DEADLINE:
            results["notes"].append("deadline during %s repeat %d" % (kind, i))
            flush()
            break
        t0 = time.time()
        try:
            rec = one_repeat(X, yc, yr, 2000)
        except Exception as exc:
            rec = {"error": repr(exc)[:400]}
        rec["fixture"] = kind
        rec["repeat"] = i
        rec["secs"] = round(time.time() - t0, 2)
        results["runs"].append(rec)
        flush()

# -- Stage 2: does the row count matter? The stable rf-reg lane fits the FULL X.
for kind in ("wide", "base"):
    X, yc, yr = fixture(kind)
    for n in (500, 1000, 2000, 4000, 8000):
        for i in range(3):
            if time.time() > DEADLINE:
                break
            try:
                reg = RandomForestRegressor(n_estimators=16, max_depth=8,
                                            random_state=7).fit(X[:n], yr[:n])
                rec = {"fixture": kind, "n": n, "repeat": i, "reg_model": model_hashes(reg)}
            except Exception as exc:
                rec = {"fixture": kind, "n": n, "repeat": i, "error": repr(exc)[:400]}
            results["nsweep"].append(rec)
            flush()

results["done"] = True
flush()
print("PROBE DONE runs=%d nsweep=%d" % (len(results["runs"]), len(results["nsweep"])))
PYEOF

RF_PROBE_JSON="$JSON" RF_PROBE_REPEATS=@REPEATS@ RF_PROBE_SECS=@PROBESECS@ \
  timeout -k 30 @PROBESECS@ /root/q/bin/python /root/run/rf_probe.py > "$OUT/rf_probe.log" 2>&1
log "rf_probe_exit=$?"
tail -5 "$OUT/rf_probe.log" >> "$OUT/record.txt"

kill "$UPLOADER" 2>/dev/null
if upload; then log "final_partial_uploaded bytes=$(wc -c < "$JSON" 2>/dev/null | tr -d ' ')"; else log "final upload FAILED (the fetch is the fallback)"; fi
log "finished"
