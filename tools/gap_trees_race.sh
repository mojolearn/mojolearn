#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
#
# gap_trees_race.sh <arms> <rounds> <rows> <lane:dataset> ...
#
# Lane gap-trees-nv's box measurement, run through `lq add nv|amd|apple CMD`. Per arm it builds the GBDT
# binding (and the svm one, which holds IsolationForest, when an iforest spec is asked) with that arm's
# defines, then runs bench/speed/forest_speed_arm.py for every spec with ours alone, at the board's
# parameters (tools/speed_gbdt_arm.py) and IDENTICAL.
#
#   arms     comma list; `base` builds with no define; `hr2old` (lane hr2-gbdt-host) is `base` run with
#            MOJOLEARN_HR2_OLD_LINK=1 MOJOLEARN_HR2_OLD_BORDERS=1 (the host link and host borders, the A arm); any other word is define names joined by `+`,
#            each built as `-D NAME=1` (e.g. base,MOJOLEARN_GBDT_X+MOJOLEARN_GBDT_Y)
#   rounds   timed rounds after the warm-up
#   rows     `full`: the board shape, timing only.  N: the first N training rows, plus the host column
#            in a separate process (MOJOLEARN_VENDOR=cpu --host-digest: hashes only, never timed; our
#            CPU is never raced), and the line says whether the device prediction digest equals the
#            host one (the same-bits check)
#   spec     lane:driver-dataset, e.g. gbdt-lossguide:taxi gbdt-categorical:taxicat
#            gbdt-rank-pairlogit:istellarank iforest:taxi
#   GAPTREES_STAGE=1 adds MOJOLEARN_STAGE_TIMES=1 (every stage drains: triage, never a timing) and
#   prints the stage table summed over the fit's trees.
#
# One line per arm x spec:
#   GAPTREES arm=<arm> lane=<l> ds=<d> rows=<rows> median_ms=<m> rounds_ms=<a/b/c> quality=<metric=v ...>
#            digest=<device> [host=<host> MATCH|DIFFER]
set -u
cd "$(dirname "$0")/.."
arms=$1 rounds=$2 rows=$3; shift 3
L=${GAPTREES_LOG:-$PWD/gaptrees-logs}; mkdir -p "$L"
# the vendor the driver checks the binding against (cuda, hip or metal), from the box
if [ "$(uname)" = Darwin ]; then vend=metal
else case "${MOJOLEARN_TARGET_COLUMN:-}" in amd*) vend=hip;; nvidia) vend=cuda;; *) command -v nvidia-smi > /dev/null && vend=cuda || vend=hip;; esac; fi
export MOJOLEARN_SPEED_EXPECTED_VENDOR=${MOJOLEARN_SPEED_EXPECTED_VENDOR:-$vend}
# the CMD tree is raw source: the base binding (host helpers every estimator imports) and, for the
# host column, the core host binding, once per job
bash bindings/build.sh > "$L/build-base.log" 2>&1 \
  || echo "GAPTREES build base FAILED: $(grep -m 1 -i error "$L/build-base.log" | cut -c1-200)"
if [ "$rows" != full ]; then
  env -u MOJOLEARN_GPU_ARCHS MOJOLEARN_TARGET_COLUMN=cpu bash bindings/build_core_host.sh > "$L/build-core_host.log" 2>&1 \
    || echo "GAPTREES build core_host FAILED: $(grep -m 1 -i error "$L/build-core_host.log" | cut -c1-200)"
fi
need="gbdt"; case "$*" in *iforest*) need="gbdt svm";; esac
[ "$need" = "gbdt svm" ] && case " $* " in *" gbdt-"*) ;; *) need="svm";; esac
for arm in ${arms//,/ }; do
  defs=""; armenv=""
  if [ "$arm" = hr2old ]; then armenv="MOJOLEARN_HR2_OLD_LINK=1 MOJOLEARN_HR2_OLD_BORDERS=1"
  elif [ "$arm" != base ]; then for d in ${arm//+/ }; do defs="$defs -D $d=1"; done; fi
  export MOJOLEARN_EXTRA_DEFINES="$defs"
  ok=1
  for b in $need; do
    bash bindings/build_$b.sh > "$L/build-$b-$arm.log" 2>&1 || { ok=0; echo "GAPTREES arm=$arm build_$b FAILED: $(grep -m 1 -i error "$L/build-$b-$arm.log" | cut -c1-200)"; }
    if [ "$rows" != full ]; then
      env -u MOJOLEARN_GPU_ARCHS MOJOLEARN_TARGET_COLUMN=cpu bash bindings/build_${b}_host.sh > "$L/build-${b}_host-$arm.log" 2>&1 \
        || { ok=0; echo "GAPTREES arm=$arm build_${b}_host FAILED: $(grep -m 1 -i error "$L/build-${b}_host-$arm.log" | cut -c1-200)"; }
    fi
  done
  [ $ok = 1 ] || continue
  for s in "$@"; do
    lane=${s%%:*} ds=${s#*:}
    extra=""; [ "$rows" != full ] && extra="--rows $rows"
    log="$L/$arm-$lane-$ds-$rows.log"
    if [ "${GAPTREES_STAGE:-0}" = 1 ]; then st="MOJOLEARN_STAGE_TIMES=1"; else st="MOJOLEARN_STAGE_TIMES=0"; fi
    env $armenv $st MOJOLEARN_SPEED_ROUNDS=$rounds MOJOLEARN_SPEED_SIZE=shipped MOJOLEARN_NUMERIC_MODE=identical \
      python3 -u bench/speed/forest_speed_arm.py --lane "$lane" --dataset "$ds" --ours-only $extra > "$log" 2>&1
    rc=$?
    : > "$log.host"
    if [ "$rows" != full ]; then
      env $armenv MOJOLEARN_VENDOR=cpu MOJOLEARN_SPEED_ROUNDS=1 MOJOLEARN_SPEED_SIZE=shipped MOJOLEARN_NUMERIC_MODE=identical \
        python3 -u bench/speed/forest_speed_arm.py --lane "$lane" --dataset "$ds" --ours-only --host-digest \
        --rows "$rows" > "$log.host" 2>&1
    fi
    python3 - "$log" "$arm" "$lane" "$ds" "$rows" "$rc" "$log.host" <<'EOS'
import re, sys, statistics
log, arm, lane, ds, rows, rc, hostlog = sys.argv[1:]
kv = lambda line: dict(re.findall(r"(\S+?)=(\S+)", line))
ms, hashes, acc, refused, stage = {}, {}, {}, [], {}
for line in open(log, errors="replace"):
    head = line.split(" ", 1)[0]
    if head == "FSPEED":
        f = kv(line); ms.setdefault(f["arm"], []).append(float(f["ms"])); hashes[f["arm"]] = f.get("hash", "-")
    elif head == "FSPEED-ACC":
        f = kv(line)
        if f["arm"] == "ours": acc[f["metric"]] = f["value"]
    elif head == "FSPEED-REFUSED":
        refused.append(line.strip()[:200])
    else:
        m = re.match(r"\[stage-times\]\s+(\S+)\s+([0-9.]+) ms", line)
        if m:
            stage[m.group(1)] = stage.get(m.group(1), 0.0) + float(m.group(2))
        m = re.match(r"  (\w+)\t([0-9.e+-]+) s$", line)
        if m:
            stage["L:" + m.group(1)] = stage.get("L:" + m.group(1), 0.0) + 1e3 * float(m.group(2))
o = ms.get("ours", [])
out = "GAPTREES arm=%s lane=%s ds=%s rows=%s rc=%s median_ms=%s rounds_ms=%s quality=%s digest=%s" % (
    arm, lane, ds, rows, rc, "%.1f" % statistics.median(o) if o else "-", "/".join("%.0f" % x for x in o) or "-",
    ",".join("%s=%s" % kv_ for kv_ in sorted(acc.items())) or "-", hashes.get("ours", "-")[:16])
if rows != "full":
    h = "-"
    for line in open(hostlog, errors="replace"):
        if line.startswith("FSPEED-DIGEST ") and kv(line).get("arm") == "ours":
            h = kv(line).get("hash", "-")[:16]
    h = h or "-"
    out += " host=%s %s" % (h, "MATCH" if h != "-" and h == hashes.get("ours", "")[:16] else "DIFFER")
if refused:
    out += " REFUSED: " + " | ".join(refused)[:300]
print(out, flush=True)
if stage:
    top = sorted(stage.items(), key=lambda kv_: -kv_[1])[:14]
    print("GAPTREES-STAGE arm=%s lane=%s ds=%s %s" % (arm, lane, ds, " ".join("%s=%.0f" % t for t in top)), flush=True)
EOS
  done
done
