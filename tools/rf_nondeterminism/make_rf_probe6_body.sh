#!/bin/bash
# make_rf_probe6_body.sh <out body> [repeats] [probesecs]
# Fills rf_probe6_body.template.sh (the code-side launch-count test). Unlike legs 1-5
# this body BUILDS the rf binding from the shipped source, so it needs no wheel GET --
# only a presigned PUT for its partials. The url lands only in the body, never committed.
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
REPO=/Users/andrewhendel/CascadeProjects/mojolearn
OUTB=${1:?usage: make_rf_probe6_body.sh <out body> [repeats] [probesecs]}
REPEATS=${2:-100}
PROBESECS=${3:-900}
PKEY=legs/rf-score-weighted-probe/2026-09-16/rf_probe6.json
FULL=$(git -C "$REPO" rev-parse db9047b9f)
PUTURL=$(cd "$REPO" && sh tools/dataset_store.sh presign-put "$PKEY" 10800)
[ -n "$PUTURL" ] || { echo "presign-put returned nothing for $PKEY" >&2; exit 1; }
FULL="$FULL" PUTURL="$PUTURL" REPEATS="$REPEATS" PROBESECS="$PROBESECS" \
python3 - "$HERE/rf_probe6_body.template.sh" "$OUTB" <<'PY'
import os, sys
t = open(sys.argv[1]).read()
t = t.replace("@FULL@", os.environ["FULL"]).replace("@PUTURL@", os.environ["PUTURL"])
t = t.replace("@REPEATS@", os.environ["REPEATS"]).replace("@PROBESECS@", os.environ["PROBESECS"])
open(sys.argv[2], "w").write(t)
PY
grep -q "$FULL" "$OUTB" && ! grep -qE '@[A-Z]+@' "$OUTB" && sh -n "$OUTB" \
  && echo "BODY_OK $OUTB commit ${FULL:0:9} repeats=$REPEATS probesecs=$PROBESECS put_key=$PKEY"
