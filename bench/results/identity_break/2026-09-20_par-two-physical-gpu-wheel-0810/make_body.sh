#!/bin/sh
# make_body.sh <out body> <slug> <quick 0|1> <budget seconds> <lanes or -> <fixture>...
# Substitutes the per-lease placeholders of body.template.sh. RunPod passes no
# environment to the body, so they are baked in.
set -eu
here=$(cd "$(dirname "$0")" && pwd)
out=$1; slug=$2; quick=$3; budget=$4; lanes=$5; shift 5
[ "$lanes" = "-" ] && lanes=""
fixtures="$*"
sed -e "s|@VERSION@|0.8.10|g" -e "s|@SLUG@|$slug|g" -e "s|@QUICK@|$quick|g" \
    -e "s|@BUDGET@|$budget|g" -e "s|@LANES@|$lanes|g" -e "s|@FIXTURES@|$fixtures|g" \
    "$here/body.template.sh" > "$out"
sh -n "$out"
# the header comment names the placeholders; none may survive in the code below it
if sed '1,/^set -u$/d' "$out" | grep -n '@[A-Z]*@'; then
    echo "make_body: a placeholder survived" >&2; exit 1
fi
echo "wrote $out"
