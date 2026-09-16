# rf-score-weighted nondeterminism, leg 6: NAME THE MECHANISM FROM THE CODE SIDE.
#
# Established: the defect needs a SECOND find_best_splits launch merging into a
# split[node] slot initSplit populated once. ONE launch 0/400, TWO launches 10/400,
# Fisher p = 9.2e-4, decisive contrast 10 cols (1 launch) vs 11 cols (2 launches).
#
# This leg builds the rf binding FROM SOURCE twice on one box:
#   arm A  stock, N_BLKS_FOR_COLS = 10
#   arm B  -D MOJOLEARN_RF_BLKS_COLS16=1, N_BLKS_FOR_COLS = 16
#
# launches = ceil(n_sampled_cols / cap), so the prediction table is:
#
#   case        cap 10 (arm A)        cap 16 (arm B)
#   wide / 11   2 launches -> MOVE    1 launch  -> stable
#   wide / 16   2 launches -> MOVE    1 launch  -> stable
#   odd  / 17   2 launches -> MOVE    2 launches-> MOVE   <-- arm B's own control
#
# The `odd` fixture has 17 features, which is why it is here: with the cap at 16 it
# is the ONLY configuration that still needs two launches, so it separates "one
# launch fixed it" from "the rebuild perturbed timing and masked a 3% race" (the cap
# also enlarges the histogram arena 1.6x, which could do that on its own).
# IF arm B's odd/17 goes stable TOO, the rebuild masked the race and this leg is VOID.
#
# Arm A is the POSITIVE CONTROL for the whole leg: a source build must still
# reproduce, or a null anywhere in arm B means nothing.
#
# CAUTION BAKED IN: both builds land at the SAME path
# (python/mojolearn/identical/_mojolearn_rf.so; MOJOLEARN_EXTRA_DEFINES does not move
# OUTDIR), so each arm records the .so sha256 and arm B REFUSES to run if the digest
# did not change -- otherwise it would silently re-test arm A's binary and any
# "stable" reading would be an artifact.
# Placeholders: @FULL@ @PUTURL@ @REPEATS@ @PROBESECS@
set -u
OUT=/root/gemm_leg_out/identity
mkdir -p "$OUT"
cd /root/mojolearn || exit 9
echo @FULL@ > /root/mojolearn/commit.txt
log() { echo "$(date -u +%H:%M:%SZ) $*" >> "$OUT/record.txt"; }
log "started rf launch-count CODE-SIDE test (build x2)"

ROOT=/root/mojolearn
JSON="$OUT/rf_probe6.json"
upload() { [ -s "$JSON" ] || return 1; curl -fsS --max-time 180 -X PUT --upload-file "$JSON" '@PUTURL@' > /dev/null 2>&1; }
( while :; do sleep 60; if upload; then log "partial_uploaded bytes=$(wc -c < "$JSON" 2>/dev/null | tr -d ' ')"; fi; done ) &
UPLOADER=$!
log "uploader_pid=$UPLOADER"

pixi run mojo --version > "$OUT/mojo_version.txt" 2>&1; log "mojo $(tr -d '\n' < "$OUT/mojo_version.txt")"
SO="$ROOT/python/mojolearn/identical/_mojolearn_rf.so"

build_arm() {   # <label> <extra defines>
    _lab=$1; _def=$2
    log "build $_lab defines='$_def' start"
    rm -f "$SO"
    ( cd "$ROOT" && MOJOLEARN_NUMERIC_MODE=identical MOJOLEARN_TARGET_COLUMN=amd \
        MOJOLEARN_COMPILE_JOBS=4 MOJOLEARN_EXTRA_DEFINES="$_def" \
        sh bindings/build_rf.sh ) > "$OUT/build_$_lab.log" 2>&1
    _rc=$?
    log "build $_lab exit=$_rc"
    [ "$_rc" = 0 ] || return 1
    [ -f "$SO" ] || { log "build $_lab produced no .so at $SO"; return 1; }
    sha256sum "$SO" | cut -d' ' -f1 > "$OUT/so_$_lab.sha256"
    log "build $_lab sha256=$(cat "$OUT/so_$_lab.sha256")"
    return 0
}

run_arm() {     # <label>
    _lab=$1
    ( cd "$ROOT" && PYTHONPATH="$ROOT/python:$ROOT" MOJOLEARN_NUMERIC_MODE=identical \
        RF_PROBE_JSON="$JSON" RF_PROBE_ARM="$_lab" RF_PROBE_REPEATS=@REPEATS@ \
        RF_PROBE_SECS=@PROBESECS@ \
        RF_PROBE_SO_SHA="$(cat "$OUT/so_$_lab.sha256")" \
        timeout -k 30 @PROBESECS@ pixi run python /root/run/rf_probe6.py ) \
        >> "$OUT/rf_probe6.log" 2>&1
    log "run $_lab exit=$?"
}

mkdir -p /root/run
cat > /root/run/rf_probe6.py <<'PYEOF'
"""One arm of the code-side launch-count test, appended to a shared JSON.

Arms are separate processes because the binding is rebuilt between them at the same
path; each arm records the .so digest it actually loaded, so a stale binary cannot
masquerade as a result.
"""
import json, os, time
import numpy as np

OUT = os.environ["RF_PROBE_JSON"]
ARM = os.environ["RF_PROBE_ARM"]
SO_SHA = os.environ.get("RF_PROBE_SO_SHA", "")
REPEATS = int(os.environ.get("RF_PROBE_REPEATS", "100"))
DEADLINE = time.time() + float(os.environ.get("RF_PROBE_SECS", "1200")) - 90

import mojolearn
from mojolearn import RandomForestRegressor

# LEG 6 FAILED HERE. `mojolearn._identity_break` is a WHEEL PACKAGING ARTIFACT (a copy of
# tools/identity_break.py inserted at package time). Legs 1-5 installed the wheel so it was
# present; this body runs from a SOURCE CHECKOUT, where it does not exist. The shipped
# bundle does carry tools/identity_break.py, so load it by path when the module is absent.
try:
    import mojolearn._identity_break as ib
except ModuleNotFoundError:
    import importlib.util
    _spec = importlib.util.spec_from_file_location(
        "ib", "/root/mojolearn/tools/identity_break.py")
    ib = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(ib)

fixture = ib.fixture
_h = ib._h
NAMES = ("_offsets", "_colid", "_quesval", "_left_child", "_leaves")

# (fixture, sampled columns wanted). odd has 17 features and is arm B's control.
CASES = (("wide", 11), ("wide", 16), ("odd", 17))


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


def arrays(est):
    return {n: np.array(np_of(est, n), copy=True) for n in NAMES}


try:
    with open(OUT) as fh:
        doc = json.load(fh)
except Exception:
    doc = {"arms": [], "notes": []}


def flush():
    tmp = OUT + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(doc, fh, indent=1)
    os.replace(tmp, OUT)


for kind, want in CASES:
    X, yc, yr = fixture(kind)
    ncols = int(X.shape[1])
    mf = want / float(ncols)
    got = max(1, int(ncols * mf))
    rec = {"arm": ARM, "so_sha256": SO_SHA[:16], "fixture": kind, "cols_wanted": want,
           "n_features": ncols, "sampled_cols": got, "max_features": mf,
           "runs": 0, "moved": 0, "distinct": 0, "divergences": [], "error": None,
           "version": getattr(mojolearn, "__version__", "?")}
    doc["arms"].append(rec)
    flush()
    ref = None
    ref_arrays = None
    seen = set()
    for i in range(REPEATS):
        if time.time() > DEADLINE:
            doc["notes"].append("deadline %s %s/%d repeat %d" % (ARM, kind, want, i))
            flush()
            break
        try:
            reg = RandomForestRegressor(n_estimators=16, max_depth=8, random_state=7,
                                        max_features=mf).fit(X[:2000], yr[:2000])
            cur = arrays(reg)
        except Exception as exc:
            rec["error"] = repr(exc)[:300]
            flush()
            break
        key = tuple(_h(cur[n]) for n in NAMES)
        seen.add(key)
        rec["runs"] += 1
        if ref is None:
            ref_arrays, ref = cur, key
        elif key != ref:
            rec["moved"] += 1
            off = np.asarray(ref_arrays["_offsets"]).astype(np.int64)
            bad = np.nonzero(
                (np.asarray(cur["_colid"]) != np.asarray(ref_arrays["_colid"]))
                | (np.asarray(cur["_quesval"]) != np.asarray(ref_arrays["_quesval"])))[0]
            det = []
            for gi in bad[:8]:
                gi = int(gi)
                t = int(np.searchsorted(off, gi, side="right") - 1)
                det.append({"tree": t, "node_local": gi - int(off[t]),
                            "ref_colid": int(ref_arrays["_colid"][gi]),
                            "cur_colid": int(cur["_colid"][gi])})
            rec["divergences"].append({"repeat": i, "n_nodes_differing": int(len(bad)),
                                       "nodes": det})
        rec["distinct"] = len(seen)
        flush()

doc["done_%s" % ARM] = True
flush()
print("ARM %s done: " % ARM + " ".join(
    "%s/%d %d/%d" % (a["fixture"], a["cols_wanted"], a["moved"], a["runs"])
    for a in doc["arms"] if a["arm"] == ARM))
PYEOF

# ---- arm A: stock cap of 10 (positive control for the whole leg) --------
if build_arm stock ""; then
    run_arm stock
else
    log "STOCK BUILD FAILED - no control, nothing below is interpretable"
fi

# ---- arm B: cap raised to 16 -------------------------------------------
if build_arm cols16cap "-D MOJOLEARN_RF_BLKS_COLS16=1"; then
    if [ "$(cat "$OUT/so_stock.sha256" 2>/dev/null)" = "$(cat "$OUT/so_cols16cap.sha256" 2>/dev/null)" ]; then
        log "REFUSED arm B: the .so digest did NOT change between builds, so the define never reached the compiler"
    else
        run_arm cols16cap
    fi
else
    log "CAP BUILD FAILED"
fi

grep -c "MOJOLEARN_RF_BLKS_COLS16" "$OUT/build_cols16cap.log" > "$OUT/define_seen.txt" 2>&1 || true
log "define_mentions_in_build_log=$(cat "$OUT/define_seen.txt")"
tail -6 "$OUT/rf_probe6.log" >> "$OUT/record.txt" 2>/dev/null

kill "$UPLOADER" 2>/dev/null
if upload; then log "final_uploaded bytes=$(wc -c < "$JSON" 2>/dev/null | tr -d ' ')"; else log "final upload FAILED (the fetch is the fallback)"; fi
log "finished"
