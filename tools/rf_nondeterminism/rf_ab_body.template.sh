# IS THE MUTEX CLAIM THE CAUSE? Capability probe, then A/B behind a define.
#
# Leg 12 named the mechanism: a LOST CANDIDATE in the cross-block merge. Three of four
# divergences had best_metric_val BIT-IDENTICAL with a LOWER colid winning, which
# Split::update's higher-colid tie-break forbids, so the higher-colid candidate was ABSENT,
# not outvoted. The fourth lost a strictly higher gain. Same direction every time.
#
# THE SUSPECT LINE: _publish_to_global's lock claim. The ACQUIRE is on a LOAD; the lock is
# taken by a weak RELAXED compare_exchange. An RMW must read the latest value in coherence
# order, a plain acquire-load need not, so the two can observe DIFFERENT releases -- and then
# the previous holder's PLAIN store to split[node] is unordered against this thread's PLAIN
# read of it. Merge into a stale split, write back, and the prior candidate is erased.
#
# STEP 1, THE CAPABILITY PROBE (ensemble/checks/acquire_rmw_probe.mojo). One mutex, many
# blocks, a PLAIN read-modify-write under the lock -- the shape of _publish_to_global. Run
# WITHOUT and WITH -D MOJOLEARN_PROBE_ACQUIRE_CAS=1. A COMPILE FAILURE of the acquire arm is
# the ANSWER to "can this column legalize an acquire RMW", not a leg failure, so the two
# builds are separate and both outcomes are recorded. The probe carries its own SABOTAGE arm
# (unlocked) whose shortfall must be large, or the cell never contended and nothing counts.
#
# STEP 2, THE A/B. Build the rf binding stock and with -D MOJOLEARN_RF_ACQUIRE_CAS=1, and run
# the lane's fit configuration on each. THE CONTROL MUST MOVE OR THE LEG PROVES NOTHING, and
# the arms are sized against the rate measured in THIS leg, not a pooled rate from another
# binary. Digests must differ or the arms are not independent.
# Placeholders: @FULL@ @PUTURL@ @REPEATS@ @PROBESECS@
set -u
OUT=/root/gemm_leg_out/identity
mkdir -p "$OUT"
cd /root/mojolearn || exit 9
echo @FULL@ > /root/mojolearn/commit.txt
log() { echo "$(date -u +%H:%M:%SZ) $*" >> "$OUT/record.txt"; }
log "started ACQUIRE-RMW capability probe + mutex A/B"

ROOT=/root/mojolearn
JSON="$OUT/rf_ab.json"
upload() { [ -s "$JSON" ] || return 1; curl -fsS --max-time 180 -X PUT --upload-file "$JSON" '@PUTURL@' > /dev/null 2>&1; }
( while :; do sleep 60; if upload; then log "partial_uploaded bytes=$(wc -c < "$JSON" 2>/dev/null | tr -d ' ')"; fi; done ) &
UPLOADER=$!
log "uploader_pid=$UPLOADER"

pixi run mojo --version > "$OUT/mojo_version.txt" 2>&1
log "mojo $(tr -d '\n' < "$OUT/mojo_version.txt")"

# ---------------- STEP 1: the capability probe -------------------------
PROBE=ensemble/checks/acquire_rmw_probe.mojo
( cd "$ROOT" && MOJOLEARN_TARGET_COLUMN=amd timeout -k 20 420 \
    pixi run mojo run -I . "$PROBE" ) > "$OUT/probe_relaxed.log" 2>&1
log "probe_relaxed_exit=$?"
( cd "$ROOT" && MOJOLEARN_TARGET_COLUMN=amd timeout -k 20 420 \
    pixi run mojo run -I . -D MOJOLEARN_PROBE_ACQUIRE_CAS=1 "$PROBE" ) \
    > "$OUT/probe_acquire.log" 2>&1
log "probe_acquire_exit=$?   (a COMPILE failure here is the capability ANSWER)"
grep -hE "column |blocks |acquire_arm_compiled|relaxed:|acquire:|unlocked|CONTROL FAILED|control OK|RELAXED CLAIM|ACQUIRE CLAIM|error:" \
    "$OUT/probe_relaxed.log" "$OUT/probe_acquire.log" 2>/dev/null | head -30 >> "$OUT/record.txt"

# ---------------- STEP 2: build both bindings --------------------------
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
    ( cd "$ROOT" && PYTHONPATH="$ROOT/python:$ROOT" MOJOLEARN_NUMERIC_MODE=identical \
        RF_PROBE_JSON="$JSON" RF_PROBE_ARM="$_lab" RF_PROBE_REPEATS=@REPEATS@ \
        RF_PROBE_SECS=@PROBESECS@ RF_PROBE_SO_SHA="$(cat "$OUT/so_$_lab.sha256")" \
        timeout -k 30 @PROBESECS@ pixi run python /root/run/rf_ab.py ) \
        >> "$OUT/rf_ab.log" 2>&1
    log "run $_lab exit=$?"
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
       "distinct": 0, "error": None, "version": getattr(mojolearn, "__version__", "?")}
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
print("ARM %s %d/%d moved" % (ARM, rec["moved"], rec["runs"]))
PYEOF

if build_arm stock ""; then
    run_arm stock
else
    log "STOCK BUILD FAILED - no control, the A/B is uninterpretable"
fi

if build_arm acqcas "-D MOJOLEARN_RF_ACQUIRE_CAS=1"; then
    if [ "$(cat "$OUT/so_stock.sha256" 2>/dev/null)" = "$(cat "$OUT/so_acqcas.sha256" 2>/dev/null)" ]; then
        log "REFUSED acqcas: digest UNCHANGED, so the define never reached the compiler"
    else
        run_arm acqcas
    fi
else
    log "ACQCAS BUILD FAILED - if this is a legalization error, that is the capability answer"
fi

_s=$(cat "$OUT/so_stock.sha256" 2>/dev/null); _a=$(cat "$OUT/so_acqcas.sha256" 2>/dev/null)
if [ -n "$_s" ] && [ "$_s" = "$_a" ]; then
    log "DIGEST COLLISION -- arms are NOT independent, results VOID"
else
    log "digests: stock=$(echo "$_s" | cut -c1-8) acqcas=$(echo "$_a" | cut -c1-8)"
fi
tail -6 "$OUT/rf_ab.log" >> "$OUT/record.txt" 2>/dev/null

kill "$UPLOADER" 2>/dev/null
if upload; then log "final_uploaded bytes=$(wc -c < "$JSON" 2>/dev/null | tr -d ' ')"; else log "final upload FAILED (the fetch is the fallback)"; fi
log "finished"
