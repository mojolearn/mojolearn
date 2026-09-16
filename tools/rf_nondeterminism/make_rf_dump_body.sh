#!/bin/bash
# make_rf_dump_body.sh <out body> [repeats] [trace_repeats] [probesecs]
# Fills rf_dump_body.template.sh (the lane-by-lane .cand field diff). Installs mojolearn==0.8.5 from
# PyPI, so no wheel GET is needed -- only a presigned PUT for partials. The url lands only
# in the body file, which is never committed.
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
REPO=/Users/andrewhendel/CascadeProjects/mojolearn
OUTB=${1:?usage: make_rf_dump_body.sh <out body> [repeats] [trace_repeats] [probesecs]}
REPEATS=${2:-100}
TRACEREPEATS=${3:-150}
PROBESECS=${4:-1500}
PKEY=legs/rf-score-weighted-probe/2026-09-16/rf_dump.json
FULL=$(git -C "$REPO" rev-parse db9047b9f)
PUTURL=$(cd "$REPO" && sh tools/dataset_store.sh presign-put "$PKEY" 10800)
[ -n "$PUTURL" ] || { echo "presign-put returned nothing for $PKEY" >&2; exit 1; }
FULL="$FULL" PUTURL="$PUTURL" REPEATS="$REPEATS" TRACEREPEATS="$TRACEREPEATS" \
PROBESECS="$PROBESECS" \
python3 - "$HERE/rf_dump_body.template.sh" "$OUTB" <<'PY'
import os, sys
t = open(sys.argv[1]).read()
t = t.replace("@FULL@", os.environ["FULL"]).replace("@PUTURL@", os.environ["PUTURL"])
t = t.replace("@REPEATS@", os.environ["REPEATS"])
t = t.replace("@TRACEREPEATS@", os.environ["TRACEREPEATS"])
t = t.replace("@PROBESECS@", os.environ["PROBESECS"])
open(sys.argv[2], "w").write(t)
PY
grep -q "$FULL" "$OUTB" && ! grep -qE '@[A-Z]+@' "$OUTB" && sh -n "$OUTB" \
  && echo "BODY_OK $OUTB commit ${FULL:0:9} repeats=$REPEATS trace_repeats=$TRACEREPEATS put_key=$PKEY"
