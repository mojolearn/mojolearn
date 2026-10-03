#!/bin/bash
# af_batch_build.sh <plan.tsv> <outroot>
# Run ON THE M2 in a lane/apple-fast-batch worktree (lane apple-fast-batch).
# Builds each distinct (binding, defines) FAST .so of the plan ONCE (arm A ""
# is shared by every line of that binding) and copies it to
# <outroot>/<tag>/A.so and B.so, the names tools/aft_ab.sh and
# tools/afc_ab_def.sh reuse under AFT_SKIP_BUILD=1 / AFC_SKIP_BUILD=1.
# Defines go through MOJOLEARN_MOJO_BUILD_FLAGS (every build_*.sh passes it to
# mojo build). afcenv rows build nothing. One line per build:
#   AF-BATCH-BUILD <binding> defines='<d>' rc=<n> [first error lines]
# Exit 0 only if every build passed. A failed build's rows get no .so (the M3
# wrapper then falls back to building).
set -u
plan=$1; outroot=$2
[ -f "$plan" ] || { sed -n 2,12p "$0"; exit 2; }
here=$(cd "$(dirname "$0")/.." && pwd); cd "$here"
mkdir -p "$outroot/.cache"
fail=0
key() { printf '%s|%s' "$1" "$2" | shasum | cut -c1-16; }
build_one() {  # $1 binding $2 defines -> $outroot/.cache/<key>.so
  local b=$1 d=$2 k; k=$(key "$1" "$2")
  local dst="$outroot/.cache/$k.so"
  [ -f "$dst" ] && return 0
  [ -f "$outroot/.cache/$k.fail" ] && return 1
  local so=python/mojolearn/_mojolearn_$b.so script=bindings/build_$b.sh
  [ "$b" = base ] && { so=python/mojolearn/_mojolearn.so; script=bindings/build.sh; }
  MOJOLEARN_NUMERIC_MODE=fast MOJOLEARN_MOJO_BUILD_FLAGS="$d" MOJOLEARN_SKIP_BUILD_GATE=1 \
    bash "$script" < /dev/null > "$outroot/.cache/$k.log" 2>&1
  local rc=$?
  echo "AF-BATCH-BUILD $b defines='$d' rc=$rc"
  if [ $rc != 0 ]; then
    grep -m 5 -B 2 -A 8 -i error "$outroot/.cache/$k.log" | cut -c1-300
    touch "$outroot/.cache/$k.fail"; return 1
  fi
  cp "$so" "$dst"
}
# builds sorted by binding so one binding's arms compile back to back
while IFS=$'\037' read -r tag tool bind dA dB outdir; do
  [ "$tag" = tag ] && continue
  [ "$tool" = afcenv ] && continue
  printf '%s\037%s\037%s\037%s\n' "$bind" "$dA" "$dB" "$tag"
done < <(tr '\t' '\037' < "$plan") | sort > "$outroot/.cache/rows.tsv"
while IFS=$'\037' read -r bind dA dB tag; do  # \037 is not IFS whitespace: empty define fields survive
  okA=1; okB=1
  build_one "$bind" "$dA" || okA=0
  build_one "$bind" "$dB" || okB=0
  if [ $okA = 1 ] && [ $okB = 1 ]; then
    mkdir -p "$outroot/$tag"
    cp "$outroot/.cache/$(key "$bind" "$dA").so" "$outroot/$tag/A.so"
    cp "$outroot/.cache/$(key "$bind" "$dB").so" "$outroot/$tag/B.so"
  else
    fail=1; echo "AF-BATCH-ROW $tag NO-SO (a build failed)"
  fi
done < "$outroot/.cache/rows.tsv"
echo "AF-BATCH-DONE builds=$(ls "$outroot/.cache/"*.so 2>/dev/null | wc -l | tr -d ' ') failed=$(ls "$outroot/.cache/"*.fail 2>/dev/null | wc -l | tr -d ' ') rows=$(ls -d "$outroot"/*/ 2>/dev/null | wc -l | tr -d ' ')"
exit $fail
