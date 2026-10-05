#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
# afc_ab_def.sh <tag> <binding> <lane> <dataset> <reps> <rounds> "<defines A>" "<defines B>"
#
# afc_ab.sh for a BUILD-TIME switch: two FAST builds of ONE binding (arm A
# with MOJOLEARN_MOJO_BUILD_FLAGS="<defines A>", arm B with "<defines B>",
# "" = none), then afc_ab.sh's race alternated A B A B ... reps times with
# that arm's .so installed, one afc_ab.sh call (1 rep, <rounds> rounds) per
# arm and rep. binding: base (bindings/build.sh, _mojolearn.so) or the
# suffix of bindings/build_<binding>.sh (x_sequence, kernel_methods,
# estimators, x_linear, ...; python/mojolearn/_mojolearn_<binding>.so), or
# auto for a trees lane (gbdt-* -> gbdt, rf -> rf, et -> trees, iforest -> svm).
# Env as afc_ab.sh (AFC_FAMILY, AFC_ARM); <dataset> may be a synthetic shape
# s-r<rows>-f<features> as in afc_ab.sh. Builds under ~/afc-def/<tag>/
# (AFC_SKIP_BUILD=1 reuses them); the race lines land in afc_ab.sh's
# ~/mq/out/race-<tag>/race.log tagged env='AFC_DEF_ARM=A|B', and the
# AFC-DEF-SUMMARY lines at the end give each arm's median of medians.
# Arm B's .so is left installed.
set -u
TAG=$1 BIND=$2 LANE=$3 DS=$4 REPS=$5 ROUNDS=$6 DA=$7 DB=$8
# One run per arm (Andrew, Oct 3): reps and rounds are 1 unless AB_MULTI_RUN=1.
[ "${AB_MULTI_RUN:-0}" = 1 ] || { REPS=1; ROUNDS=1; }
here=$(cd "$(dirname "$0")/.." && pwd); cd "$here"
# binding auto (AFC_FAMILY=trees): the lane's own binding, as
# bench/speed/forest_speed_arm.py's OUR_ENTRY_POINTS reach it: gbdt-* -> gbdt,
# rf -> rf, et -> trees, iforest -> svm.
if [ "$BIND" = auto ]; then
  case $LANE in gbdt*) BIND=gbdt;; rf) BIND=rf;; et) BIND=trees;; iforest) BIND=svm;;
    *) echo "AFC-DEF $TAG bind=auto has no binding for lane=$LANE"; exit 1;; esac
fi
so=python/mojolearn/_mojolearn_$BIND.so; script=bindings/build_$BIND.sh
[ "$BIND" = base ] && { so=python/mojolearn/_mojolearn.so; script=bindings/build.sh; }
# AFC_DEF_BUILD_TAG: share one pair of builds across several tags (a shape
# sweep, tools/afc_shape_sweep.py): an arm's .so under ~/afc-def/<build tag>/
# is reused when its stamp (head, binding, defines) matches, else rebuilt.
out=$HOME/afc-def/${AFC_DEF_BUILD_TAG:-$TAG}; mkdir -p "$out"
echo "AFC-DEF $TAG head=$(git rev-parse --short HEAD) bind=$BIND lane=$LANE ds=$DS A='$DA' B='$DB'"
build() {  # $1 arm, $2 defines
  if [ "${AFC_SKIP_BUILD:-0}" = 1 ] && [ -f "$out/$1.so" ]; then return 0; fi
  stamp="$(git rev-parse HEAD) $BIND $2"
  if [ -n "${AFC_DEF_BUILD_TAG:-}" ] && [ -f "$out/$1.so" ] && [ "$(cat "$out/$1.stamp" 2>/dev/null)" = "$stamp" ]; then
    echo "AFC-DEF-BUILD $TAG arm=$1 reused=$AFC_DEF_BUILD_TAG defines='$2'"; return 0
  fi
  rm -f "$out/$1.stamp"
  MOJOLEARN_NUMERIC_MODE=fast MOJOLEARN_MOJO_BUILD_FLAGS="$2" MOJOLEARN_SKIP_BUILD_GATE=1 \
    bash "$script" > "$out/build_$1.log" 2>&1
  rc=$?; echo "AFC-DEF-BUILD $TAG arm=$1 rc=$rc defines='$2'"
  [ $rc = 0 ] || { grep -m 5 -B 2 -A 8 -i error "$out/build_$1.log" | cut -c1-300; exit 1; }
  cp "$so" "$out/$1.so"
  echo "$stamp" > "$out/$1.stamp"
}
build A "$DA"; build B "$DB"
for r in $(seq 1 "$REPS"); do
  for a in A B; do
    cp "$out/$a.so" "$so.tmp" && mv -f "$so.tmp" "$so"
    bash tools/afc_ab.sh "$TAG" "$LANE" "$DS" 1 "$ROUNDS" "AFC_DEF_ARM=$a" | sed "s/^AFC-AB /AFC-AB def=$a rep=$r /"
  done
done
python3 - "$HOME/mq/out/race-$TAG/race.log" "$DA" "$DB" <<'PY'
import re, sys, statistics
log, da, db = sys.argv[1:4]
ms = {"A": [], "B": []}; q = {"A": set(), "B": set()}
for line in open(log):
    m = re.search(r"env='AFC_DEF_ARM=([AB])'", line)
    v = re.search(r"median_ms=([0-9.]+)", line)
    if m and v:
        ms[m.group(1)].append(float(v.group(1)))
        qq = re.search(r"quality=(\S+)", line)
        if qq: q[m.group(1)].add(qq.group(1)[:120])
for a, d in (("A", da), ("B", db)):
    print("AFC-DEF-SUMMARY arm=%s n=%d median_ms=%s all=%s quality=%s defines='%s'" % (
        a, len(ms[a]), round(statistics.median(ms[a]), 1) if ms[a] else None,
        [round(x) for x in ms[a]], sorted(q[a]), d))
PY
