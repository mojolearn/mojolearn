#!/bin/sh
# make_lanes_body.sh <out body> <slug> <quick 0|1> <budget seconds> <overlay 0|1> <commit> <name=lane,lane>...
# Substitutes the per-lease placeholders of body.lanes.template.sh. RunPod
# passes no environment to the body, so they are baked in. <commit> must come
# from `git rev-parse`, never from a keyboard.
set -eu
here=$(cd "$(dirname "$0")" && pwd)
out=$1; slug=$2; quick=$3; budget=$4; overlay=$5; commit=$6; shift 6
sets="$*"
case "$commit" in *[!0-9a-f]*|"") echo "make_lanes_body: commit is not hex" >&2; exit 1 ;; esac
[ ${#commit} -eq 40 ] || { echo "make_lanes_body: commit is not 40 hex" >&2; exit 1; }
sed -e "s|@VERSION@|0.8.10|g" -e "s|@SLUG@|$slug|g" -e "s|@QUICK@|$quick|g" \
    -e "s|@BUDGET@|$budget|g" -e "s|@LANES@||g" -e "s|@FIXTURES@|base|g" \
    -e "s|@LANESETS@|$sets|g" -e "s|@OVERLAY@|$overlay|g" -e "s|@COMMIT@|$commit|g" \
    "$here/body.lanes.template.sh" > "$out"
sh -n "$out"
# the header comment names the placeholders; none may survive in the code below it
if sed '1,/^set -u$/d' "$out" | grep -n '@[A-Z]*@'; then
    echo "make_lanes_body: a placeholder survived" >&2; exit 1
fi
# THE LIGHT-RUNS RULE, CHECKED: no full-sweep command may be in a body
if sed '1,/^set -u$/d' "$out" | grep -n -e '--par all' -e '--par default'; then
    echo "make_lanes_body: a full-sweep command is in the body" >&2; exit 1
fi
echo "wrote $out"
