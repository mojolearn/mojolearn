# DOES THE POST-CLAIM ACQUIRE LOAD CLOSE THE MI300X RANDOM FOREST RACE? A/B on one box.
#
# lane/rf-mutex-claim-acquire, 2026-09-16. The repair adds ONE acquire load of the mutex
# after the weak relaxed claim succeeds (DEVIATION 106 in
# ensemble/decisiontree/batched_levelalgo/split.mojo says why). The shipped spelling is now
# the repaired one; -D MOJOLEARN_RF_MUTEX_CLAIM_STOCK=1 compiles the PRE-repair claim and is
# the CONTROL arm of this leg only.
#
# THE CONTROL MUST MOVE OR THE LEG PROVES NOTHING: the pre-repair arm reproduced 13/300 to
# 16/300 on the wide fixture at 16 columns (max_features=1.0, the default). The two .so
# digests must differ or the define never reached the compiler and the arms are the same
# binary. Every line this leg writes carries the token CLAIMFIX-cols16 so a stale R2 object
# from an earlier leg cannot be read as this leg's numbers.
# Placeholders: @FULL@ @PUTURL@ @REPEATS@ @PROBESECS@
set -u
OUT=/root/gemm_leg_out/identity
mkdir -p "$OUT"
cd /root/mojolearn || exit 9
echo @FULL@ > /root/mojolearn/commit.txt
log() { echo "$(date -u +%H:%M:%SZ) CLAIMFIX-cols16 $*" >> "$OUT/record.txt"; }
log "started mutex post-claim acquire A/B, commit @FULL@"

ROOT=/root/mojolearn
JSON="$OUT/rf_claimfix_ab.json"
upload() { [ -s "$JSON" ] || return 1; curl -fsS --max-time 180 -X PUT --upload-file "$JSON" '@PUTURL@' > /dev/null 2>&1; }
( while :; do sleep 60; if upload; then log "partial_uploaded bytes=$(wc -c < "$JSON" 2>/dev/null | tr -d ' ')"; fi; done ) &
UPLOADER=$!
log "uploader_pid=$UPLOADER"

pixi run mojo --version > "$OUT/mojo_version.txt" 2>&1
log "mojo $(tr -d '\n' < "$OUT/mojo_version.txt")"
rocminfo 2>/dev/null | grep -m1 -o 'gfx[0-9a-f]*' > "$OUT/gfx.txt"; log "gfx $(cat "$OUT/gfx.txt")"

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
    [ "$_rc" = 0 ] && [ -f "$SO" ] || { tail -15 "$OUT/build_$_lab.log" >> "$OUT/record.txt"; return 1; }
    sha256sum "$SO" | cut -d' ' -f1 > "$OUT/so_$_lab.sha256"
    log "build $_lab sha256=$(cat "$OUT/so_$_lab.sha256")"
    return 0
}

run_arm() {     # <label>
    _lab=$1
    log "run $_lab start repeats=@REPEATS@"
    ( cd "$ROOT" && PYTHONPATH="$ROOT/python:$ROOT" MOJOLEARN_NUMERIC_MODE=identical \
        RF_PROBE_JSON="$JSON" RF_PROBE_ARM="$_lab" RF_PROBE_REPEATS=@REPEATS@ \
        RF_PROBE_SECS=@PROBESECS@ RF_PROBE_SO_SHA="$(cat "$OUT/so_$_lab.sha256")" \
        timeout -k 30 @PROBESECS@ pixi run python /root/run/rf_ab.py ) \
        >> "$OUT/rf_claimfix_ab.log" 2>&1
    log "run $_lab exit=$? $(tail -1 "$OUT/rf_claimfix_ab.log" 2>/dev/null)"
}

mkdir -p /root/run
cat > /root/run/rf_ab.py <<'PYEOF'
"""One arm of the mutex A/B, appended to a shared JSON."""
import importlib.util, json, os, time
import numpy as np

OUT = os.environ["RF_PROBE_JSON"]
ARM = os.environ["RF_PROBE_ARM"]
SO_SHA = os.environ.get("RF_PROBE_SO_SHA", "")
REPEATS = int(os.environ.get("RF_PROBE_REPEATS", "300"))
DEADLINE = time.time() + float(os.environ.get("RF_PROBE_SECS", "1200")) - 90

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


X, yc, yr = fixture("wide")
rec = {"arm": ARM, "so_sha256": SO_SHA[:16], "runs": 0, "moved": 0,
       "distinct": 0, "error": None, "version": getattr(mojolearn, "__version__", "?"),
       "fixture": "wide", "columns": int(X.shape[1]), "token": "CLAIMFIX-cols16"}
doc["arms"].append(rec)
flush()

ref = None
seen = set()
for i in range(REPEATS):
    if time.time() > DEADLINE:
        doc["notes"].append("deadline %s at %d" % (ARM, i))
        flush()
        break
    try:
        reg = RandomForestRegressor(n_estimators=16, max_depth=8,
                                    random_state=7).fit(X[:2000], yr[:2000])
        key = tuple(_h(np.array(np_of(reg, n), copy=True)) for n in NAMES)
    except Exception as exc:
        rec["error"] = repr(exc)[:300]
        flush()
        break
    seen.add(key)
    rec["runs"] += 1
    if ref is None:
        ref = key
    elif key != ref:
        rec["moved"] += 1
    rec["distinct"] = len(seen)
    flush()

doc["done_%s" % ARM] = True
flush()
print("ARM %s %d/%d moved distinct=%d CLAIMFIX-cols16" % (ARM, rec["moved"], rec["runs"], rec["distinct"]))
PYEOF

# Control first: the pre-repair claim has to be SEEN to move on this box and this build.
if build_arm stock_prerepair "-D MOJOLEARN_RF_MUTEX_CLAIM_STOCK=1"; then
    run_arm stock_prerepair
else
    log "CONTROL BUILD FAILED - no control, the A/B is uninterpretable"
fi

if build_arm claimfix ""; then
    if [ "$(cat "$OUT/so_stock_prerepair.sha256" 2>/dev/null)" = "$(cat "$OUT/so_claimfix.sha256" 2>/dev/null)" ]; then
        log "REFUSED claimfix: digest UNCHANGED, the control define never reached the compiler"
    else
        run_arm claimfix
    fi
else
    log "CLAIMFIX BUILD FAILED"
fi

_s=$(cat "$OUT/so_stock_prerepair.sha256" 2>/dev/null); _a=$(cat "$OUT/so_claimfix.sha256" 2>/dev/null)
if [ -n "$_s" ] && [ "$_s" = "$_a" ]; then
    log "DIGEST COLLISION -- arms are NOT independent, results VOID"
else
    log "digests: stock_prerepair=$(echo "$_s" | cut -c1-8) claimfix=$(echo "$_a" | cut -c1-8)"
fi
grep -h "^ARM " "$OUT/rf_claimfix_ab.log" >> "$OUT/record.txt" 2>/dev/null

kill "$UPLOADER" 2>/dev/null
if upload; then log "final_uploaded bytes=$(wc -c < "$JSON" 2>/dev/null | tr -d ' ')"; else log "final upload FAILED (the fetch is the fallback)"; fi
log "finished"
