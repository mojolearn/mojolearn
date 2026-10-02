#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
#
# af_board_queue.sh: an M3 FAST board refresh for the lanes a branch changed.
#
#   bash tools/af_board_queue.sh <branch> [--lanes a,b,...] [--batch N] [--prefix TAG]
#       Laptop side. Run inside a checkout of <branch> (it reads the diff vs
#       origin/main). Prints the `lq add m3 CMD` lines to queue (it never queues
#       them itself): one line per family batch (at most N lane x dataset pairs,
#       default 18, so `lq log`'s 40-line tail holds a whole batch), plus one
#       `opp` line that dumps the M3 0.8.34 board's FAST and opponent quality.
#       Selection: tools/lane_select.py --changed-since <merge base with
#       origin/main> (so main's own later moves are not counted; the identity
#       registry lanes whose code reaches a changed file), intersected by name
#       with the board lanes of the classical drivers (bench_board_algos,
#       bench_board_more, classical_two_datasets) and the trees driver
#       (bench/speed/forest_speed_arm.py). --lanes overrides the selection.
#
#   bash tools/af_board_queue.sh run <tag> <family> <lane:ds[,lane:ds...]>
#       Box side (a CMD job on the M3). Builds the FAST bindings the batch needs
#       (base + each lane's binding, MOJOLEARN_NUMERIC_MODE=fast, skipped when a
#       stamp says this head already built it in FAST), then races OUR FAST arm
#       only, 1 warm-up + 3 timed rounds, at board size (rows-full / shipped).
#       family: algos | classical2 | classical | trees. Prints one line per pair:
#         AFB tag=.. family=.. lane=.. ds=.. status=.. median_ms=.. runs=[..] digest=.. q=k=v,..
#       into ~/mq/out/race-<tag>/race.log (so `lq log m3 <tag> '^AFB '` reads it).
#
#   bash tools/af_board_queue.sh opp <tag>
#       Box side. Prints AFB-OPPZ lines (gzip+base64 chunks of the 0.8.34 M3
#       board's per-race FAST and best-opponent ms and quality) for
#       tools/af_board_merge.py.
#
# Never races or times MOJOLEARN_VENDOR=cpu; opponents are not re-raced.
set -u
here=$(cd "$(dirname "$0")/.." && pwd)
LQ=${LQ:-$HOME/mojolearn-evidence/lq/lq}

board_root() {
  for b in board-0834 board-0833; do [ -d "$HOME/$b/cache/algos-data/rows-full" ] && { echo "$HOME/$b"; return; }; done
  echo "$HOME/board-0834"
}

# ----------------------------------------------------------------- box: run
if [ "${1:-}" = run ]; then
  TAG=$2 FAM=$3 PAIRS=$4
  cd "$here"
  B=$(board_root); VP=$B/cache/venv/bin/python
  [ -x "$VP" ] || VP=python3
  OUT=$HOME/mq/out/race-$TAG; mkdir -p "$OUT"; LOG=$OUT/race.log
  HEAD=$(git rev-parse --short HEAD)
  echo "AFB-JOB tag=$TAG family=$FAM head=$HEAD board=$B" >> "$LOG"
  SAFE=$(basename "$here")
  ST=$HOME/afb-stamps/$SAFE; mkdir -p "$ST"
  fastbuild() {  # $1 build script basename without .sh
    local s=$1 stamp=$ST/$1
    [ -f "bindings/$s.sh" ] || { echo "AFB-BUILD $s rc=missing" >> "$LOG"; return; }
    if [ -f "$stamp" ] && [ "$(cat "$stamp")" = "$(git rev-parse HEAD)" ]; then
      echo "AFB-BUILD $s rc=cached" >> "$LOG"; return
    fi
    # The M3 queue's own FAST build at this head (~/mq mq.sh fbuild: core + its bindset) counts.
    local core=$HOME/mq/out/${SAFE}-fbuildb.log one=$HOME/mq/out/${SAFE}-fbuild-${s#build_}.log
    if [ "$(cat .mq_fbuilt 2>/dev/null)" = "$HEAD" ] && [ -f "$core" ] && \
       { [ $s = build ] || { [ -f "$one" ] && [ ! "$one" -ot "$core" ]; }; }; then
      echo "AFB-BUILD $s rc=queue-fbuild" >> "$LOG"; return
    fi
    rm -f "$stamp"
    MOJOLEARN_NUMERIC_MODE=fast MOJOLEARN_SKIP_BUILD_GATE=1 pixi run bash "bindings/$s.sh" > "$OUT/build_$s.log" 2>&1
    local rc=$?
    echo "AFB-BUILD $s rc=$rc" >> "$LOG"
    if [ $rc = 0 ]; then git rev-parse HEAD > "$stamp"; else grep -m 3 -B 2 -A 6 error "$OUT/build_$s.log" | cut -c1-300 >> "$LOG"; fi
  }
  # The bindings this batch needs, from the drivers' own lane tables.
  BUILDS=$(PYTHONPATH=$here/tools "$VP" "$here/tools/af_board_merge.py" builds "$FAM" "$PAIRS")
  fastbuild build
  for s in $BUILDS; do [ "$s" = build ] || fastbuild "$s"; done
  qcompact() { python3 -c '
import json, sys
s = sys.stdin.read().strip()
try:
    q = json.loads(s)
except Exception:
    print(s.replace(" ", "")[:160]); sys.exit()
out = []
for k, v in sorted(q.items()) if isinstance(q, dict) else []:
    if isinstance(v, bool) or v is None:
        continue
    if isinstance(v, float):
        out.append("%s=%.6g" % (k, v))
    elif isinstance(v, int):
        out.append("%s=%d" % (k, v))
print(",".join(out)[:160] or "-")'; }
  for p in ${PAIRS//,/ }; do
    L=${p%%:*} D=${p##*:}
    d=$OUT/$L-$D; rm -rf "$d"; mkdir -p "$d"
    if [ "$FAM" = trees ]; then
      MOJOLEARN_NUMERIC_MODE=fast MOJOLEARN_SPEED_ROUNDS=3 MOJOLEARN_SPEED_SIZE=shipped \
      MOJOLEARN_SPEED_EXPECTED_VENDOR=metal GBM_BENCH_DATA=${GBM_BENCH_DATA:-$HOME/datasets/gbm-bench} \
      PYTHONPATH="$here/python" "$VP" -u bench/speed/forest_speed_arm.py --lane "$L" --dataset "$D" --ours-only \
        > "$d/race.txt" 2>&1
      line=$(python3 - "$d/race.txt" <<'EOF'
import re, statistics, sys
t = open(sys.argv[1], errors="replace").read()
ms = [float(m) for m in re.findall(r"^FSPEED lane=\S+ arm=ours\S* shape=\S+ round=\d+ ms=([\d.]+)", t, re.M)]
hs = sorted(set(re.findall(r"^FSPEED lane=\S+ arm=ours\S* shape=\S+ round=\d+ ms=\S+ hash=(\S+)", t, re.M)))
acc = re.findall(r"^FSPEED-ACC lane=\S+ arm=ours\S* metric=(\S+) value=(\S+)", t, re.M)
ref = re.findall(r"^FSPEED-REFUSED lane=\S+ arm=\S+ reason=(.*)$", t, re.M)
st = "ok" if ms else ("refused" if ref else "error")
q = ",".join("%s=%.6g" % (k, float(v)) for k, v in acc) or "-"
print("status=%s median_ms=%s runs=[%s] digest=%s q=%s%s" % (
    st, round(statistics.median(ms), 3) if ms else "none", " ".join("%.1f" % x for x in ms),
    ("+".join(h[:16] for h in hs)) or "none", q, (" why=" + ref[0][:120].replace(" ", "_")) if ref and not ms else ""))
EOF
)
    else
      case $FAM in
        algos) DRV=tools/bench_board_algos.py; PFX=ALGOS; DATA=$B/cache/algos-data/rows-full; XARGS= ;;
        classical2) DRV=tools/bench_board_more.py; PFX=MORE
           DATA=$(find "$B" -maxdepth 4 -type d -name more-data 2>/dev/null | head -1)/rows-full; XARGS= ;;
        classical) DRV=tools/classical_two_datasets.py; PFX=CTD
           DATA=$(find "$B" -maxdepth 4 -type d -name ctd-data 2>/dev/null | head -1)/rows-full; XARGS="--root $here" ;;
        *) echo "AFB tag=$TAG family=$FAM lane=$L ds=$D status=bad-family" >> "$LOG"; continue ;;
      esac
      MOJOLEARN_BENCH_INSTALLED=0 PYTHONPATH=$here/python \
        "$VP" $DRV race --lane "$L" --dataset "$D" --data "$DATA" --arms ours-fast \
        --rounds 3 --out "$d/res" --work "$d/work" $XARGS > "$d/race.txt" 2>&1
      rm -rf "$d/work"
      r=$(grep -o "$PFX lane=$L dataset=$D arm=ours-fast .*" "$d/race.txt" | tail -1)
      m=$(echo "$r" | grep -o 'median_ms=[^ ]*' | cut -d= -f2)
      st=$(echo "$r" | grep -o 'status=[A-Za-z_-]*' | cut -d= -f2)
      q=$(echo "$r" | sed -n 's/.* quality=//p' | qcompact)
      g=$(grep -o 'digest=[0-9a-f]*' "$d/race.txt" | tail -1 | cut -d= -f2 | cut -c1-16)
      runs=$(python3 - "$d/res" <<'EOF'
import glob, json, sys
for f in glob.glob(sys.argv[1] + "/*.json"):
    try:
        a = json.load(open(f))["arms"]["ours-fast"]
        print(" ".join("%.1f" % x for x in a.get("ms") or []))
        break
    except Exception:
        pass
EOF
)
      why=
      [ -z "$m" ] || [ "$m" = None ] && why=" why=$(grep -E -m 1 'REFUSED|SKIPPED|Error|error' "$d/race.txt" | tr ' ' _ | cut -c1-120)"
      line="status=${st:-none} median_ms=${m:-none} runs=[$runs] digest=${g:-none} q=${q:--}$why"
    fi
    echo "AFB tag=$TAG family=$FAM lane=$L ds=$D head=$HEAD $line" | cut -c1-390 >> "$LOG"
  done
  echo "AFB-DONE tag=$TAG family=$FAM" >> "$LOG"
  # stdout (~/mq/out/<tag>.log, which lq log also greps) gets no AFB lines, so nothing doubles
  echo "afb done tag=$TAG pairs=$(grep -c '^AFB tag=' "$LOG") ok=$(grep -c '^AFB tag=.* status=ok ' "$LOG")"
  exit 0
fi

# ----------------------------------------------------------------- box: opp
if [ "${1:-}" = opp ]; then
  TAG=$2; cd "$here"
  B=$(board_root)
  OUT=$HOME/mq/out/race-$TAG; mkdir -p "$OUT"
  python3 tools/af_board_merge.py extract "$B/board/board.json" > "$OUT/race.log"
  echo "afb opp chunks=$(grep -c '^AFB-OPPZ' "$OUT/race.log") board=$B"
  exit 0
fi

# ----------------------------------------------------------------- laptop: emit
BR=${1:?usage: af_board_queue.sh <branch> [--lanes a,b] [--batch N] [--prefix TAG] | run ... | opp ...}; shift
LANES= BATCH=18 PREFIX=afb
while [ $# -gt 0 ]; do case $1 in
  --lanes) LANES=$2; shift 2;; --batch) BATCH=$2; shift 2;; --prefix) PREFIX=$2; shift 2;;
  *) echo "unknown arg $1" >&2; exit 2;; esac; done
cd "$here"
if [ "$(git rev-parse HEAD)" != "$(git rev-parse "origin/$BR" 2>/dev/null)" ]; then
  echo "# note: HEAD is not origin/$BR; selection reads this checkout's diff vs origin/main" >&2
fi
SEL=
if [ -z "$LANES" ]; then
  mkdir -p "$HOME/mojolearn-evidence/apple-fast-board"
  SEL=$HOME/mojolearn-evidence/apple-fast-board/lanesel-$(git rev-parse --short HEAD).json
  [ -s "$SEL" ] || python3 tools/lane_select.py --changed-since "$(git merge-base origin/main HEAD)" --json "$SEL" > /dev/null 2>&1 \
    || { echo "lane_select failed" >&2; exit 1; }
fi
PYTHONPATH=$here/tools python3 tools/af_board_merge.py select --branch "$BR" --batch "$BATCH" \
  --prefix "$PREFIX" --lq "$LQ" ${SEL:+--lanesel "$SEL"} ${LANES:+--lanes "$LANES"}
