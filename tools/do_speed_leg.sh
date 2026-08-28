#!/usr/bin/env bash
# ONE GUARDED DIGITALOCEAN LEG THAT MEASURES SPEED: create -> build ->
# run tools/local_speed_run.sh -> fetch -> DESTROY.
#
#   bash tools/do_speed_leg.sh amd forest    <token_file>
#   bash tools/do_speed_leg.sh amd classical <token_file>
#   bash tools/do_speed_leg.sh amd gemmseq   <token_file>
#
# WHY THIS FILE EXISTS AT ALL. The FAST board has three columns and only two
# entry points. Apple is this desk (tools/local_speed_run.sh) and NVIDIA is
# RunPod (tools/gemm_remote_leg.sh --speed). AMD had NEITHER on 2026-08-28:
# RunPod's MI300X was out of stock (`lowestPrice.stockStatus: null`) and the
# only AMD silicon reachable from this account was a DigitalOcean MI325X,
# whose leg script (tools/e2_remote_leg.sh) runs the IDENTITY bootstrap and
# has no speed payload. So an AMD row was structurally impossible rather
# than merely unmeasured, and a board with a structurally impossible column
# is a board that will quietly keep two columns forever.
#
# WHAT AN AMD ROW CAN AND CANNOT SAY, AND THE FILE REFUSES TO BLUR IT.
# Every GPU opponent this repository benchmarks against -- cuBLAS, cuML,
# cuVS, CatBoost's GPU learner, XGBoost's, LightGBM's CUDA learner -- is
# CUDA-only. There is NO legal opponent on AMD. The standing rule is that a
# lane on a GPU vendor's box is compared against THAT VENDOR'S GPU arm or
# against nothing; it does not fall back to sklearn on the host CPU, because
# a GPU-vs-CPU ratio dressed as a vendor comparison is worse than a missing
# row. So this leg runs OURS ONLY, deliberately, and the table it feeds says
# `no legal opponent on this vendor` rather than inventing one. What the
# column is FOR is the absolute: our own arm's milliseconds on a third
# vendor, beside the same arm's milliseconds on the other two.
#
# THE GUARDS ARE tools/e2_remote_leg.sh's, DELIBERATELY COPIED RATHER THAN
# FACTORED OUT. DigitalOcean bills until a droplet is DESTROYED -- powering
# it off does not stop it, which is how a 90-second job once billed 10h48m
# (~$41) -- so the destroy is (a) an EXIT trap, so every failure path runs
# it, and (b) a DETACHED dead-man keyed by TAG AND NAME rather than by id,
# so a create whose response cannot be parsed still cannot leave an orphan.
# Sharing that code with the identity leg would make one file able to break
# both lanes' teardown at once; the duplication is the isolation.
#
# ONE FAMILY PER LEASE, for the same arithmetic as the RunPod payload: a
# cold box pays pixi solve, a first-ever HIP build of every driver, and (for
# forest) a first-ever HIP build of five CPython extensions, inside one hour.
set -uo pipefail

VENDOR="${1:?amd|nv}"
FAMILY="${2:?forest|classical|gemmseq}"
TOKFILE="${3:?token file}"
TOK="$(cat "$TOKFILE")"
DEADMAN_SECONDS="${DEADMAN_SECONDS:-3600}"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
API=https://api.digitalocean.com/v2
SSH_KEY_FP="df:f7:6b:0c:56:da:48:a5:6f:6d:ae:44:af:de:f3:0b"
STAMP="$(date +%Y-%m-%d_%H%M%S)"
OUT="$REPO/bench/results/fast_speed/do-$STAMP-$VENDOR-$FAMILY"
STATE="$OUT/leg.state"

case "$VENDOR" in
  amd) NAME=mojolearn-speed-amd; REGION=tor1; SIZE=gpu-mi325x1-256gb; IMAGE=188571990 ;;
  nv)  NAME=mojolearn-speed-nv;  REGION=nyc2; SIZE=gpu-h100x1-80gb;   IMAGE=236925144 ;;
  *) echo "vendor must be amd or nv"; exit 2 ;;
esac
case "$FAMILY" in forest|classical|gemmseq) : ;; *) echo "family must be forest|classical|gemmseq"; exit 2 ;; esac

mkdir -p "$OUT"
api() { curl -s -H "Authorization: Bearer $TOK" "$@"; }
log() { echo "[$(date +%T) $VENDOR/$FAMILY] $*"; }

DROPLET_ID=""; DEADMAN_PID=""

destroy() {
  if [ -z "$DROPLET_ID" ]; then
    for id in $(api "$API/droplets?tag_name=speed&per_page=50" | python3 -c "import json,sys
d=json.load(sys.stdin); print(' '.join(str(x['id']) for x in d.get('droplets',[]) if x['name']=='$NAME'))" 2>/dev/null); do
      code=$(curl -s -o /dev/null -w '%{http_code}' -X DELETE -H "Authorization: Bearer $TOK" "$API/droplets/$id")
      log "DELETE by-name droplet $id -> HTTP $code"
    done
    return 0
  fi
  for i in 1 2 3 4 5 6; do
    code=$(curl -s -o /dev/null -w '%{http_code}' -X DELETE -H "Authorization: Bearer $TOK" "$API/droplets/$DROPLET_ID")
    log "DELETE droplet $DROPLET_ID -> HTTP $code"
    case "$code" in 204|404) break ;; esac
    sleep 10
  done
  sleep 5
  left=$(api "$API/droplets/$DROPLET_ID" | python3 -c \
    "import json,sys; d=json.load(sys.stdin); print(d.get('droplet',{}).get('status','gone'))" 2>/dev/null)
  log "post-destroy status: ${left:-gone}"
  echo "destroyed $(date -u +%FT%TZ) status=${left:-gone}" >> "$STATE"
}
teardown() {
  rc=$?
  log "teardown (rc=$rc)"
  destroy
  if [ -n "$DEADMAN_PID" ]; then
    pkill -P "$DEADMAN_PID" 2>/dev/null
    kill "$DEADMAN_PID" 2>/dev/null && log "dead-man timer cancelled"
  fi
  exit $rc
}
trap teardown EXIT

LEG_START=$(date +%s)
COMMIT="$(git -C "$REPO" rev-parse "${SPEED_COMMIT:-HEAD}")"
{
  echo "vendor=$VENDOR"; echo "family=$FAMILY"; echo "size=$SIZE"; echo "region=$REGION"
  echo "commit=$COMMIT"; echo "deadman_seconds=$DEADMAN_SECONDS"; echo "started=$(date -u +%FT%TZ)"
} > "$STATE"

nohup bash -c "sleep $DEADMAN_SECONDS; for id in \$(curl -s -H 'Authorization: Bearer $TOK' '$API/droplets?tag_name=speed&per_page=50' | python3 -c \"import json,sys; d=json.load(sys.stdin); print(' '.join(str(x['id']) for x in d.get('droplets',[]) if x['name']=='$NAME'))\"); do curl -s -o /dev/null -w \"deadman DELETE \$id -> %{http_code}\\n\" -X DELETE -H 'Authorization: Bearer $TOK' \"$API/droplets/\$id\" >> '$STATE'; done" \
  >/dev/null 2>&1 < /dev/null &
DEADMAN_PID=$!
disown "$DEADMAN_PID" 2>/dev/null || true
log "dead-man pid $DEADMAN_PID ($DEADMAN_SECONDS s, tag speed + name $NAME)"

log "creating $NAME ($SIZE, $REGION)"
CREATE_BODY="$(api -X POST -H "Content-Type: application/json" \
  -d "{\"name\":\"$NAME\",\"region\":\"$REGION\",\"size\":\"$SIZE\",\"image\":$IMAGE,\"ssh_keys\":[\"$SSH_KEY_FP\"],\"tags\":[\"speed\"]}" \
  "$API/droplets")"
DROPLET_ID=$(printf '%s' "$CREATE_BODY" | python3 -c "import json,sys
try:
    d=json.load(sys.stdin); print(d['droplet']['id'] if 'droplet' in d else '')
except Exception: print('')")
if [ -z "$DROPLET_ID" ]; then
  log "create returned no id; body: $(printf '%s' "$CREATE_BODY" | head -c 300)"
  sleep 5
  DROPLET_ID=$(api "$API/droplets?tag_name=speed&per_page=50" | python3 -c "import json,sys
d=json.load(sys.stdin); ids=[x['id'] for x in d.get('droplets',[]) if x['name']=='$NAME']
print(ids[0] if ids else '')")
  [ -n "$DROPLET_ID" ] && log "ADOPTED droplet $DROPLET_ID found by name" || { log "create FAILED"; exit 3; }
fi
echo "droplet=$DROPLET_ID" >> "$STATE"
log "droplet id $DROPLET_ID"

IP=""
for i in $(seq 1 90); do
  read -r status IP < <(api "$API/droplets/$DROPLET_ID" | python3 -c "
import json,sys
d=json.load(sys.stdin)['droplet']
ips=[n['ip_address'] for n in d['networks'].get('v4',[]) if n['type']=='public']
print(d['status'], ips[0] if ips else '')")
  [ "$status" = "active" ] && [ -n "$IP" ] && break
  sleep 10
done
[ -n "$IP" ] || { log "never became active"; exit 4; }
log "active at $IP"; echo "ip=$IP" >> "$STATE"

SSH="ssh -o StrictHostKeyChecking=accept-new -o ConnectTimeout=10 -o ServerAliveInterval=30 root@$IP"
ok=0
for i in $(seq 1 30); do
  $SSH 'echo SSH-OK' 2>/dev/null | grep -q SSH-OK && { ok=1; break; }
  sleep 15
done
[ $ok = 1 ] || { log "ssh never came up"; exit 5; }
$SSH 'rocm-smi --showproductname 2>/dev/null | head -3; nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1' \
  | tee "$OUT/device.txt" | sed "s/^/[$VENDOR gpu] /"

# SHIP THE COMMIT, NOT THE WORKING TREE -- the identity leg's rule and for
# its reason: a bundle of a named ref is exact by construction and the
# commit recorded in the artifacts is the commit that ran.
BUNDLE="/tmp/mojolearn-speed-$COMMIT.bundle"
git -C "$REPO" branch -f "speed-run" "$COMMIT" >/dev/null 2>&1
git -C "$REPO" bundle create "$BUNDLE" speed-run >/dev/null 2>&1 || { log "bundle failed"; exit 6; }
scp -q -o StrictHostKeyChecking=accept-new "$BUNDLE" "root@$IP:/root/speed.bundle" || { log "scp failed"; exit 6; }
$SSH "rm -rf /root/mojolearn && git clone -q -b speed-run /root/speed.bundle /root/mojolearn && cd /root/mojolearn && git rev-parse HEAD" \
  || { log "remote clone failed"; exit 6; }
log "shipped $COMMIT"

FETCH_RESERVE="${FETCH_RESERVE:-480}"
NOW=$(date +%s)
WORK_SECONDS=$(( LEG_START + DEADMAN_SECONDS - NOW - FETCH_RESERVE ))
[ "$WORK_SECONDS" -lt 300 ] && { log "only ${WORK_SECONDS}s left, skipping the work"; exit 7; }
log "work bound ${WORK_SECONDS}s"

# THE BINDINGS ARE A forest-ONLY COST AND THEY HAVE TO BE BUILT HERE.
# python/mojolearn/*.so are not committed and the Mac's are Apple objects,
# so each of these is a first-ever HIP build of that extension. None is
# fatal: a family that comes home with four lanes and one refusal says
# something true, and a family that dies on a build says nothing at all.
#
# THREE THINGS THIS GETS RIGHT THAT THE FIRST VERSION DID NOT, ALL OF THEM
# ALREADY SOLVED IN tools/gemm_remote_leg.sh AND ALL OF THEM RE-BROKEN HERE
# ON THE 2026-08-28_125022 LEG, WHICH CAME HOME WITH 84 REFUSALS AND ZERO
# TIMED ROUNDS. Ported across rather than rediscovered a third time.
#
#  1. EVERY BINDING, DERIVED FROM THE FILESYSTEM. `mojolearn/__init__.py`
#     imports the whole package surface, so `import mojolearn` needs every
#     extension present regardless of which lane is timed. A hand-written
#     list omitted `bindings/build.sh` -- the one script not named
#     `build_<name>.sh` -- and every arm then refused with "cannot import
#     name '_mojolearn' from partially initialized module".
#  2. TWO PASSES. Pass 1 sets MOJOLEARN_SKIP_BUILD_GATE because each script's
#     smoke gate imports siblings that do not exist yet; pass 2 re-runs them
#     with the gate LIVE against a complete package, and pass 2's exit codes
#     are the ones that count. Ordering cannot fix this: base's smoke needs
#     estimators and estimators' needs base.
#  3. PIXI'S INTERPRETER ON PATH FOR THE BUILDS ONLY, ON AMD. The gates run
#     `python3 - <<PY` with numpy; the ROCm image's python3 has none, so the
#     rf/gbdt/trees MOJO BUILDS SUCCEEDED and were then reported as failures
#     because their gate died on `import numpy`. Scoped to the build loop.
BUILDS=""
[ "$FAMILY" = "forest" ] && BUILDS="all"

$SSH "cd /root/mojolearn && \
  export PATH=/root/.pixi/bin:\$PATH && \
  (command -v pixi >/dev/null || curl -fsSL https://pixi.sh/install.sh | bash) && \
  export PATH=/root/.pixi/bin:\$PATH && \
  timeout -k 30 1500 pixi install > /root/pixi_install.log 2>&1; echo PIXI_INSTALL_EXIT=\$?; \
  if [ '$BUILDS' = all ]; then \
    BSCRIPTS=\"bindings/build.sh \$(ls bindings/build_*.sh 2>/dev/null | tr '\n' ' ')\"; \
    BPATH=\"\$PATH\"; \
    [ '$VENDOR' = amd ] && BPATH=\"\$PWD/.pixi/envs/default/bin:\$PATH\"; \
    for pass in 1 2; do \
      for bs in \$BSCRIPTS; do \
        b=\$(basename \$bs .sh | sed 's/^build_//; s/^build\$/base/'); \
        skip=1; [ \$pass = 2 ] && skip=; \
        PATH=\"\$BPATH\" MOJOLEARN_SKIP_BUILD_GATE=\"\$skip\" timeout -k 30 900 \
          bash \$bs > /root/build_\$b.pass\$pass.log 2>&1; \
        echo BUILD_\${b}_PASS\${pass}_EXIT=\$?; \
      done; \
    done; \
  fi; \
  ( cd python && python3 -c 'import mojolearn; print(\"mojolearn imports OK\")' ) 2>&1 | tail -2" \
  2>&1 | tee -a "$OUT/console.log"

# THE DATASETS ARE FETCHED AS THEIR OWN NAMED STEP, ONCE, BEFORE ANY ARM.
# Ported from tools/gemm_remote_leg.sh, which has always done this, after the
# 2026-08-28_130709 leg came home with every higgs rung refused --
#
#   speed_gbdt_arm: dataset 'higgs' unavailable (higgs is not downloaded:
#   /root/datasets/gbm-bench/higgs/HIGGS.csv.gz is missing)
#
# -- and every lane silently falling back to its shipped synthetic fixture.
# That is the worst shape of failure this harness has: the leg came home with
# sixty timed rounds and twelve `ours` headers, all of them REAL, and none of
# them on the dataset the other two columns were measured on. A board built
# from it would have compared AMD on synthclf-720000x100 against NVIDIA on
# higgs-1000000x28 and shown a ratio.
#
# Bounded generously because this is a network fetch and not a measurement:
# HIGGS is 2.6 GB plus a gzip csv parse of 11M x 29 that runs several
# minutes, and paying it once here is the difference between one download and
# six inside a per-arm budget. The exit code is recorded so a table missing
# the higgs rows says WHY.
if [ -n "${SPEED_DATASET:-}" ]; then
    $SSH "cd /root/mojolearn && export PATH=/root/.pixi/bin:\$PATH && \
      timeout -k 30 2400 .pixi/envs/default/bin/python tools/speed_gbdt_arm.py \
        --download '${SPEED_DATASET}' > /root/download.log 2>&1; \
      echo DOWNLOAD_${SPEED_DATASET}_EXIT=\$?; tail -2 /root/download.log" \
      2>&1 | tee -a "$OUT/console.log"
fi

# OURS ONLY, AND THE INTERPRETERS ARE POINTED AT NOTHING ON PURPOSE. Every
# opponent arm then prints FSPEED-REFUSED with its own reason, which is the
# record this column needs: the absence is IN THE LOGS rather than in a
# sentence somebody has to remember to write.
$SSH "cd /root/mojolearn && export PATH=/root/.pixi/bin:\$PATH && \
  MOJOLEARN_SPEED_ROUNDS='${SPEED_ROUNDS:-5}' \
  MOJOLEARN_SPEED_DATASET='${SPEED_DATASET:-}' \
  MOJOLEARN_SPEED_ROWS='${SPEED_ROWS:-}' \
  MOJOLEARN_VENDOR_PY='.pixi/envs/default/bin/python' \
  MOJOLEARN_TORCH_PY='.pixi/envs/default/bin/python' \
  MOJOLEARN_ARM_BUDGET='${ARM_BUDGET:-900}' \
  timeout -k 60 $WORK_SECONDS sh tools/local_speed_run.sh $FAMILY /root/speedrun \
  > /root/speed_run.log 2>&1; echo SPEED_RUN_EXIT=\$?; tail -25 /root/speed_run.log" \
  2>&1 | tee -a "$OUT/console.log"

log "fetch artifacts"
rsync -az "root@$IP:/root/speedrun/" "$OUT/run/" && log "fetched run/" || log "FETCH FAILED (run/)"
for f in speed_run.log pixi_install.log; do
  rsync -az "root@$IP:/root/$f" "$OUT/$f" 2>/dev/null || true
done
rsync -az --include 'build_*.log' --exclude '*' "root@$IP:/root/" "$OUT/builds/" 2>/dev/null || true

{
  echo "finished=$(date -u +%FT%TZ)"
  echo "fspeed_lines=$(cat "$OUT"/run/logs/*.log 2>/dev/null | grep -c '^FSPEED')"
  echo "fspeed_rounds=$(cat "$OUT"/run/logs/*.log 2>/dev/null | grep -c '^FSPEED ')"
  echo "fspeed_refused=$(cat "$OUT"/run/logs/*.log 2>/dev/null | grep -c '^FSPEED-REFUSED')"
  echo "ours_headers_fast=$(grep -h '^FSPEED-HEADER' "$OUT"/run/logs/*.log 2>/dev/null | grep -c 'arm=ours mode=FAST')"
  echo "ours_headers_identical=$(grep -h '^FSPEED-HEADER' "$OUT"/run/logs/*.log 2>/dev/null | grep -c 'arm=ours mode=IDENTICAL')"
} >> "$STATE"
log "done -- $OUT"
