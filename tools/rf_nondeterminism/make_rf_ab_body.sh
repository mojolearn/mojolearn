#!/bin/bash
# make_rf_ab_body.sh <out body> [repeats] [probesecs]
# Fills rf_ab_body.template.sh: the acquire-RMW capability probe plus the A/B of the
# shipped mutex claim against the acquire claim. Needs only a presigned PUT.
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
REPO=/Users/andrewhendel/CascadeProjects/mojolearn
OUTB=${1:?usage: make_rf_ab_body.sh <out body> [repeats] [probesecs]}
REPEATS=${2:-300}
PROBESECS=${3:-1200}
PKEY=legs/rf-score-weighted-probe/2026-09-16/rf_ab.json
FULL=$(git -C "$REPO" rev-parse db9047b9f)
PUTURL=$(cd "$REPO" && sh tools/dataset_store.sh presign-put "$PKEY" 10800)
[ -n "$PUTURL" ] || { echo "presign-put returned nothing for $PKEY" >&2; exit 1; }
FULL="$FULL" PUTURL="$PUTURL" REPEATS="$REPEATS" PROBESECS="$PROBESECS" \
python3 - "$HERE/rf_ab_body.template.sh" "$OUTB" <<'PY'
import os, sys
t = open(sys.argv[1]).read()
t = t.replace("@FULL@", os.environ["FULL"]).replace("@PUTURL@", os.environ["PUTURL"])
t = t.replace("@REPEATS@", os.environ["REPEATS"]).replace("@PROBESECS@", os.environ["PROBESECS"])
open(sys.argv[2], "w").write(t)
PY
grep -q "$FULL" "$OUTB" && ! grep -qE '@[A-Z]+@' "$OUTB" && sh -n "$OUTB" \
  && echo "BODY_OK $OUTB commit ${FULL:0:9} repeats=$REPEATS put_key=$PKEY"
