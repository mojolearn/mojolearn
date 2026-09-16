# DOES THE POST-CLAIM ACQUIRE FENCE CLOSE THE MI300X RANDOM FOREST RACE? A/B on one box.
#
# lane/rf-mutex-claim-acquire, 2026-09-16. The repair adds ONE post-claim ACQUIRE LOAD of the
# mutex, in a one-iteration loop so its value is consumed and the load is not deleted
# (DEVIATION 106 in ensemble/decisiontree/batched_levelalgo/split.mojo says why, and why it
# is not the acquire FENCE that would be one instruction instead of three: the fence cannot
# launch on Apple). The shipped spelling is the repaired one;
# -D MOJOLEARN_RF_MUTEX_CLAIM_STOCK=1 compiles the PRE-repair claim and is the CONTROL arm of
# this leg only. On gfx942 the repair is a `global_load_dword ... sc0 sc1` followed by
# `buffer_inv sc0 sc1`, and that invalidate lands between the `global_atomic_cmpswap` that
# takes the lock and the plain `global_load_dword` of `split[node]`, which is the read the
# traces showed going stale.
#
# THE CONTROL MUST MOVE OR THE LEG PROVES NOTHING: the pre-repair arm reproduced 13/300 to
# 16/300 on the wide fixture at 16 columns (max_features=1.0, the default). Every line this
# leg writes carries the token CLAIMFIX-cols16 so a stale R2 object from an earlier leg
# cannot be read as this leg's numbers.
#
# THE ARMS MUST BE TWO PROGRAMS, AND THAT IS CHECKED BEFORE EITHER IS TIMED (2026-09-16).
# This leg used to compare the whole-file sha256 of the two `.so` files, and to do it AFTER
# the arms had run. Both were wrong. A file digest differs for reasons that are not code
# (the `mktemp` install name, a build id), so its `results VOID` branch was unreachable; and
# a check that runs last cannot stop a leg from spending the box on one program timed twice.
# The repair as FIRST WRITTEN was a `_ = Atomic.load[ACQUIRE](mutex)` whose result is
# discarded, which the compiler deletes: every section of both arms was byte-identical while
# their file digests differed. NOTE WHAT THIS GATE STILL CANNOT SEE. It compares artifacts,
# so it catches a repair that did not compile in; it cannot catch a repair that compiled in
# and cannot LAUNCH, which is what the acquire fence turned out to be on Apple. If the
# claimfix arm reports an error on every fit, suspect the spelling before the box. Both arms are now built first, their CODE AND CONSTANT
# sections are compared by tools/rf_nondeterminism/section_digest.py, and nothing is timed
# unless that comparison says they DIFFER.
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
    # KEEP THE ARM. The next build writes the same path, and an arm that has
    # been overwritten cannot be re-run or re-digested without paying for it
    # again.
    cp "$SO" "$OUT/so_$_lab.so"
    sha256sum "$SO" | cut -d' ' -f1 > "$OUT/so_$_lab.sha256"
    # The digest that decides anything is of .text and .rodata, not of the
    # file. See the header.
    if ! python3 "$ROOT/tools/rf_nondeterminism/section_digest.py" "$SO" \
            > "$OUT/sec_$_lab.txt" 2>&1; then
        log "build $_lab SECTION DIGEST REFUSED: $(cat "$OUT/sec_$_lab.txt")"
        return 1
    fi
    cut -d' ' -f1 < "$OUT/sec_$_lab.txt" > "$OUT/so_$_lab.sections"
    log "build $_lab file_sha256=$(cat "$OUT/so_$_lab.sha256") sections=$(cat "$OUT/so_$_lab.sections")"
    return 0
}

run_arm() {     # <label>
    _lab=$1
    # Reinstall THIS arm: both arms are built before either runs, so the path
    # holds whichever was built last.
    cp "$OUT/so_$_lab.so" "$SO" || { log "run $_lab: arm artifact missing"; return 1; }
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

# BUILD BOTH FIRST, TIME NEITHER YET. The gate below decides whether the box
# is worth spending, and a gate that runs after the arms cannot do that.
BUILT_OK=1
build_arm stock_prerepair "-D MOJOLEARN_RF_MUTEX_CLAIM_STOCK=1" \
    || { BUILT_OK=0; log "CONTROL BUILD FAILED - no control, the A/B is uninterpretable"; }
build_arm claimfix "" \
    || { BUILT_OK=0; log "CLAIMFIX BUILD FAILED"; }

# THE GATE. Section digests, before anything is timed. Exit 0 DIFFER, exit 1
# IDENTICAL, exit 2 the instrument refused; 1 and 2 are different numbers on
# purpose, so a broken reader is never read as a passing comparison.
ARMS_INDEPENDENT=0
if [ "$BUILT_OK" = 1 ]; then
    python3 "$ROOT/tools/rf_nondeterminism/section_digest.py" \
        "$OUT/so_stock_prerepair.so" "$OUT/so_claimfix.so" \
        > "$OUT/sections_ab.txt" 2>&1
    case $? in
        0) ARMS_INDEPENDENT=1
           log "sections DIFFER: stock_prerepair=$(cut -c1-8 < "$OUT/so_stock_prerepair.sections") claimfix=$(cut -c1-8 < "$OUT/so_claimfix.sections")" ;;
        1) log "SECTION DIGEST COLLISION -- .text and .rodata are IDENTICAL, the two arms are ONE PROGRAM, results VOID, nothing timed"
           log "   both arms hash $(cut -c1-16 < "$OUT/so_claimfix.sections"); file digests stock=$(cut -c1-8 < "$OUT/so_stock_prerepair.sha256") claimfix=$(cut -c1-8 < "$OUT/so_claimfix.sha256") differ and mean NOTHING"
           log "   this is what a repair deleted by the optimizer looks like; read the emitted kernel IR before renting again" ;;
        *) log "SECTION DIGEST REFUSED, results VOID, nothing timed: $(cat "$OUT/sections_ab.txt")" ;;
    esac
fi

if [ "$ARMS_INDEPENDENT" = 1 ]; then
    # Control first: the pre-repair claim has to be SEEN to move on this box
    # and this build.
    run_arm stock_prerepair
    run_arm claimfix
else
    log "NOT TIMING EITHER ARM"
fi
grep -h "^ARM " "$OUT/rf_claimfix_ab.log" >> "$OUT/record.txt" 2>/dev/null

kill "$UPLOADER" 2>/dev/null
if upload; then log "final_uploaded bytes=$(wc -c < "$JSON" 2>/dev/null | tr -d ' ')"; else log "final upload FAILED (the fetch is the fallback)"; fi
log "finished"
