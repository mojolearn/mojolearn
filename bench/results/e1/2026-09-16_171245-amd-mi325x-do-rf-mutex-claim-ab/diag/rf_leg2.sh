#!/bin/bash
# THE MISSING MEASUREMENT: on gfx942, does the repaired mutex claim stop moving
# while the pre-repair claim still moves?  A/B on one box, order rotated.
#
# Arms: 36ec47446 DEFAULT (repaired: post-claim ACQUIRE LOAD whose value is
# CONSUMED) against the same commit with -D MOJOLEARN_RF_MUTEX_CLAIM_STOCK=1
# (the pre-repair claim).  Fixture `wide`, 16 columns, max_features=1.0 -- the
# shipped default, the configuration measured at 13/300 on MI300X.  Control
# configuration cols10 (max_features=0.625) runs only if the lease allows; its
# 0/600 is what makes a cols16 difference mean something.
#
# usage: bash rf_leg.sh <absolute unix deadline>
set -u
DEADLINE_ABS=${1:?absolute unix deadline}
ROOT=/root/mojolearn
RUNDIR=/root/mojolearn/bench/results/e1/2026-09-16_171245-amd-mi325x-do-rf-mutex-claim-ab
OUT=$RUNDIR/diag
ROUNDS=$OUT/rounds
mkdir -p "$ROUNDS"
TOKEN=RFCLAIM-36ec474-gfx942-2026-09-16
JSON=$OUT/rf_claim_ab.json
PUTURL='https://e5ba0319f9962a1aceb6a9f03245d1f5.r2.cloudflarestorage.com/mojolearn-data/legs/rf-mutex-claim-acquire/2026-09-16-mi325x-gfx942-cols16-36ec474/rf_claim_ab.json?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=d98e3aab60d37b3490f7fcf065695e94%2F20260916%2Fauto%2Fs3%2Faws4_request&X-Amz-Date=20260916T170605Z&X-Amz-Expires=14400&X-Amz-SignedHeaders=host&X-Amz-Signature=0ebb5034244deeca4d8bbbfeea304cefdba466440a7e48d8644277eca51db8fd'
REPEATS=51          # 51 fits -> 50 comparisons per round; 6 rounds = 300 per arm
NROUNDS16=120
NROUNDS10=40
SO=$ROOT/python/mojolearn/identical/_mojolearn_rf.so

log() { echo "$(date -u +%H:%M:%SZ) $TOKEN $*" >> "$OUT/record.txt"; echo "$(date -u +%H:%M:%SZ) $TOKEN $*"; }

merge() {
  python3 - "$ROUNDS" "$JSON" "$TOKEN" "$OUT/record.txt" <<'PY'
import glob, json, os, sys
rd, out, token, rec = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
doc = {"token": token, "leg": "rf mutex claim A/B on DigitalOcean MI325X gfx942",
       "commit": "36ec474465e1944c1201f1c52f45e321d5de2431", "rounds": [], "totals": {}}
for p in sorted(glob.glob(os.path.join(rd, "*.json"))):
    try:
        doc["rounds"].append(json.load(open(p)))
    except Exception as e:
        doc["rounds"].append({"file": os.path.basename(p), "unreadable": repr(e)})
t = {}
for r in doc["rounds"]:
    k = "%s/%s" % (r.get("cfg"), r.get("arm"))
    a = t.setdefault(k, {"moved": 0, "comparisons": 0, "fits": 0, "rounds": 0,
                         "errors": [], "distinct_keys": 0})
    a["moved"] += r.get("moved", 0); a["comparisons"] += r.get("comparisons", 0)
    a["fits"] += r.get("fits", 0); a["rounds"] += 1
    a["distinct_keys"] += r.get("distinct", 0)
    if r.get("error"): a["errors"].append(r["error"][:120])
doc["totals"] = t
try:
    doc["record_tail"] = open(rec).read()[-6000:]
except Exception:
    pass
tmp = out + ".tmp"
json.dump(doc, open(tmp, "w"), indent=1)
os.replace(tmp, out)
print("  totals " + json.dumps({k: {"moved": v["moved"], "comparisons": v["comparisons"]} for k, v in t.items()}))
PY
}

upload() {
  merge >> "$OUT/record.txt" 2>&1
  [ -s "$JSON" ] || return 1
  curl -fsS --max-time 150 -X PUT --upload-file "$JSON" "$PUTURL" >/dev/null 2>&1
}

log "start; commit 36ec474465e1944c1201f1c52f45e321d5de2431; lease deadline $DEADLINE_ABS ($(( DEADLINE_ABS - $(date +%s) ))s from now)"
cd "$ROOT" || exit 9
rocminfo 2>/dev/null | grep -m1 -o 'gfx[0-9a-f]*' > "$OUT/gfx.txt"
log "gfx $(cat "$OUT/gfx.txt" 2>/dev/null)"
git rev-parse HEAD > "$OUT/commit_on_box.txt" 2>/dev/null
log "commit on box $(cat "$OUT/commit_on_box.txt" 2>/dev/null)"

# The python environment the harness runs in (the default pixi env has no
# python at all; gbmbench is the one with numpy).  Start it now, in parallel
# with the builds, because it is pure download.
( pixi install -e gbmbench > "$OUT/pixi_gbmbench.log" 2>&1; echo $? > "$OUT/pixi_gbmbench.rc" ) &
PIXIPID=$!

# ---------------------------------------------------------------------------
# GUARD 2: THE DEFINE MUST BE SEEN TO REACH ARGV.
# bindings/build_rf.sh never echoes its command line, so a misspelled variable
# is silently empty and both arms are the same program.  Build under `bash -x`,
# then PRINT THE MATCHING `mojo build` LINE (never a count), and require the
# token to be ABSENT from the claimfix arm's line -- the same probe run on the
# side where it must fail.
# ---------------------------------------------------------------------------
build_arm() {   # <label> <extra defines> <expect_define 0|1>
  _lab=$1; _def=$2; _exp=$3
  log "build $_lab MOJOLEARN_EXTRA_DEFINES='$_def' start"
  _t0=$(date +%s)
  rm -f "$SO"
  ( cd "$ROOT" && MOJOLEARN_NUMERIC_MODE=identical MOJOLEARN_TARGET_COLUMN=amd \
      MOJOLEARN_COMPILE_JOBS=8 MOJOLEARN_EXTRA_DEFINES="$_def" \
      bash -x bindings/build_rf.sh ) > "$OUT/build_$_lab.log" 2>&1
  _rc=$?
  log "build $_lab exit=$_rc seconds=$(( $(date +%s) - _t0 ))"
  grep -m1 'mojo build' "$OUT/build_$_lab.log" > "$OUT/cmdline_$_lab.txt" 2>/dev/null
  log "build $_lab ARGV: $(cat "$OUT/cmdline_$_lab.txt" 2>/dev/null)"
  if grep -q 'MOJOLEARN_RF_MUTEX_CLAIM_STOCK' "$OUT/cmdline_$_lab.txt" 2>/dev/null; then
    _saw=1
  else
    _saw=0
  fi
  if [ "$_saw" != "$_exp" ]; then
    log "build $_lab DEFINE CHECK FAILED: expected present=$_exp, saw present=$_saw on the mojo build line"
    return 1
  fi
  log "build $_lab define check OK (MOJOLEARN_RF_MUTEX_CLAIM_STOCK present=$_saw as required)"
  [ "$_rc" = 0 ] && [ -f "$SO" ] || { tail -20 "$OUT/build_$_lab.log" >> "$OUT/record.txt"; return 1; }
  cp "$SO" "$OUT/so_$_lab.so"
  sha256sum "$SO" | cut -d' ' -f1 > "$OUT/so_$_lab.sha256"
  if ! python3 /root/section_digest.py "$SO" > "$OUT/sec_$_lab.txt" 2>&1; then
    log "build $_lab SECTION DIGEST REFUSED: $(cat "$OUT/sec_$_lab.txt")"
    return 1
  fi
  cut -d' ' -f1 < "$OUT/sec_$_lab.txt" > "$OUT/so_$_lab.sections"
  log "build $_lab file_sha256=$(cat "$OUT/so_$_lab.sha256") sections=$(cat "$OUT/so_$_lab.sections")"
  # READ THE ARCHITECTURE BACK OUT OF THE ARTIFACT. If no gfx942 device code is
  # embedded, the section digest is comparing host code only and says nothing
  # about the kernel that takes the mutex.
  _arch=$(strings -a "$OUT/so_$_lab.so" 2>/dev/null | grep -om1 'gfx9[0-9a-f]*')
  log "build $_lab embedded device arch: ${_arch:-NONE FOUND (the .so may carry no AMDGPU code)}"
  return 0
}

BUILT_OK=1
build_arm stock_prerepair "-D MOJOLEARN_RF_MUTEX_CLAIM_STOCK=1" 1 \
  || { BUILT_OK=0; log "CONTROL BUILD FAILED - no control, the A/B is uninterpretable"; }
build_arm claimfix "" 0 \
  || { BUILT_OK=0; log "CLAIMFIX BUILD FAILED"; }
upload

# ---------------------------------------------------------------------------
# GUARD 1: PROVE THE ARMS ARE TWO PROGRAMS BEFORE SPENDING THE BOX.
# .text and .rodata, not the whole file: the file digest differs for reasons
# that are not code (the mktemp install name), so it can never report a
# collision.  Exit 0 DIFFER, exit 1 IDENTICAL, exit 2 the instrument refused;
# 1 and 2 are different numbers on purpose.
# ---------------------------------------------------------------------------
ARMS_INDEPENDENT=0
if [ "$BUILT_OK" = 1 ]; then
  python3 /root/section_digest.py "$OUT/so_stock_prerepair.so" "$OUT/so_claimfix.so" \
    > "$OUT/sections_ab.txt" 2>&1
  case $? in
    0) ARMS_INDEPENDENT=1
       log "SECTIONS DIFFER: stock_prerepair=$(cat "$OUT/so_stock_prerepair.sections") claimfix=$(cat "$OUT/so_claimfix.sections")" ;;
    1) log "SECTION DIGEST COLLISION -- .text and .rodata are IDENTICAL, the two arms are ONE PROGRAM, results VOID, nothing timed."
       log "  both arms hash $(cat "$OUT/so_claimfix.sections"); the file digests (stock=$(cat "$OUT/so_stock_prerepair.sha256") claimfix=$(cat "$OUT/so_claimfix.sha256")) differ and mean NOTHING."
       log "  On gfx942 that would mean the consumed acquire load emits no instruction at all and the repair is not in the binary." ;;
    *) log "SECTION DIGEST REFUSED, results VOID, nothing timed: $(cat "$OUT/sections_ab.txt")" ;;
  esac
fi
upload

wait $PIXIPID
log "pixi -e gbmbench rc=$(cat "$OUT/pixi_gbmbench.rc" 2>/dev/null) $(tail -2 "$OUT/pixi_gbmbench.log" 2>/dev/null | tr '\n' ' ')"

run_round() {   # <arm> <cfg> <maxf> <round> <repeats>
  _arm=$1; _cfg=$2; _maxf=$3; _r=$4; _rep=$5
  cp "$OUT/so_$_arm.so" "$SO" || { log "round $_r $_cfg $_arm: arm artifact missing"; return 1; }
  ( cd "$ROOT" && PYTHONPATH="$ROOT/python:$ROOT" MOJOLEARN_NUMERIC_MODE=identical \
      RF_ROUND_OUTDIR="$ROUNDS" RF_ROUND_ARM="$_arm" RF_ROUND_CFG="$_cfg" \
      RF_ROUND_MAXF="$_maxf" RF_ROUND_INDEX="$_r" RF_ROUND_REPEATS="$_rep" \
      RF_ROUND_SO_SHA="$(cat "$OUT/so_$_arm.sha256")" \
      RF_ROUND_SECTIONS="$(cat "$OUT/so_$_arm.sections")" \
      RF_ROUND_TOKEN="$TOKEN" RF_ROUND_DEADLINE="$DEADLINE_ABS" \
      timeout -k 30 $(( DEADLINE_ABS - $(date +%s) + 60 < 90 ? 90 : DEADLINE_ABS - $(date +%s) + 60 )) \
      pixi run -e gbmbench python3 -u /root/rf_round.py ) >> "$OUT/rounds.log" 2>&1
  _rrc=$?
  log "round $_r $_cfg $_arm exit=$_rrc :: $(grep -h " r$(printf %02d $_r) $_cfg $_arm " "$OUT/rounds.log" | tail -1)"
  upload
}

SPF=1.0
STARTUP=40
if [ "$ARMS_INDEPENDENT" = 1 ]; then
  # A SMOKE ROUND FIRST, on the repaired arm, so a harness that errors on every
  # fit is found in sixty seconds rather than after the lease is gone -- and so
  # the round budget below is computed from a MEASURED per-fit cost.
  run_round claimfix smoke 1.0 0 4
  _s=$(python3 -c "
import json,sys
try:
    d=json.load(open('$ROUNDS/r00_smoke_claimfix.json'))
    print(d.get('seconds_per_fit') or 1.0, 1 if d.get('error') else 0)
except Exception:
    print(1.0, 1)")
  SPF=$(echo "$_s" | cut -d' ' -f1); SMOKE_ERR=$(echo "$_s" | cut -d' ' -f2)
  log "smoke seconds_per_fit=$SPF error=$SMOKE_ERR"
  if [ "$SMOKE_ERR" != "0" ]; then
    log "SMOKE FAILED -- the harness errors on this box; not timing the arms"
    ARMS_INDEPENDENT=0
  fi
fi

pair_seconds() {  # <repeats>
  python3 -c "print(int(2*($STARTUP + $1*$SPF) + 40))"
}

if [ "$ARMS_INDEPENDENT" = 1 ]; then
  for _r in $(seq 20 $(( 19 + NROUNDS16 ))); do
    _need=$(pair_seconds $REPEATS)
    _left=$(( DEADLINE_ABS - $(date +%s) ))
    if [ "$_left" -lt "$_need" ]; then
      log "cols16 stopping before round $_r: ${_left}s left, a balanced pair needs ${_need}s"
      break
    fi
    if [ $(( _r % 2 )) -eq 1 ]; then _A=stock_prerepair; _B=claimfix; else _A=claimfix; _B=stock_prerepair; fi
    log "cols16 round $_r order: $_A then $_B (${_left}s of lease left)"
    run_round "$_A" cols16 1.0 "$_r" "$REPEATS"
    run_round "$_B" cols16 1.0 "$_r" "$REPEATS"
  done

  # THE CONTROL CONFIGURATION, only with lease to spare.  cols10 was 0/300 on
  # the pre-repair build; if it moves here the fixture, not the mutex, is the
  # story.
  for _r in $(seq 200 $(( 199 + NROUNDS10 ))); do
    _need=$(pair_seconds $REPEATS)
    _left=$(( DEADLINE_ABS - $(date +%s) ))
    if [ "$_left" -lt "$_need" ]; then
      log "cols10 stopping before round $_r: ${_left}s left, a balanced pair needs ${_need}s"
      break
    fi
    if [ $(( _r % 2 )) -eq 1 ]; then _A=stock_prerepair; _B=claimfix; else _A=claimfix; _B=stock_prerepair; fi
    log "cols10 round $_r order: $_A then $_B (${_left}s of lease left)"
    run_round "$_A" cols10 0.625 "$_r" "$REPEATS"
    run_round "$_B" cols10 0.625 "$_r" "$REPEATS"
  done
else
  log "NOT TIMING EITHER ARM"
fi

rm -f "$OUT"/so_*.so
upload && log "final upload OK bytes=$(wc -c < "$JSON" | tr -d ' ')" || log "final upload FAILED"
log "finished"
