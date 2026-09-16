#!/bin/bash
# make_rf_probe5_body.sh <out body> [repeats] [probesecs]
# Fills rf_probe5_body.template.sh (the launch-count discriminator): mints a short-lived
# presigned GET for the 0.8.6 release wheel and a presigned PUT for this leg's partials.
# Both urls land only in the body file, which is never committed.
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
REPO=/Users/andrewhendel/CascadeProjects/mojolearn
E=$HOME/mojolearn-evidence/release-0.8.6
OUTB=${1:?usage: make_rf_probe5_body.sh <out body> [repeats] [probesecs]}
REPEATS=${2:-40}
PROBESECS=${3:-1500}

WHEEL=$E/linux-wheel/dist/final/mojolearn-0.8.6-py3-none-manylinux_2_35_x86_64.whl
KEY=$(cat "$E/linux-wheel/r2-key.txt")
PKEY=legs/rf-score-weighted-probe/2026-09-16/rf_probe5.json

FULL=$(git -C "$REPO" rev-parse db9047b9f)
WSHA=$(shasum -a 256 "$WHEEL" | cut -d' ' -f1)
WNAME=$(basename "$WHEEL")
URL=$(cd "$REPO" && sh tools/dataset_store.sh presign "$KEY" 10800)
PUTURL=$(cd "$REPO" && sh tools/dataset_store.sh presign-put "$PKEY" 10800)
[ -n "$PUTURL" ] || { echo "presign-put returned nothing for $PKEY" >&2; exit 1; }

FULL="$FULL" WNAME="$WNAME" WSHA="$WSHA" URL="$URL" PUTURL="$PUTURL" \
REPEATS="$REPEATS" PROBESECS="$PROBESECS" \
python3 - "$HERE/rf_probe5_body.template.sh" "$OUTB" <<'PY'
import os, sys
t = open(sys.argv[1]).read()
t = t.replace("@FULL@", os.environ["FULL"])
t = t.replace("@WHEELNAME@", os.environ["WNAME"]).replace("@WHEELSHA@", os.environ["WSHA"])
t = t.replace("@WHEELURL@", os.environ["URL"]).replace("@PUTURL@", os.environ["PUTURL"])
t = t.replace("@REPEATS@", os.environ["REPEATS"]).replace("@PROBESECS@", os.environ["PROBESECS"])
open(sys.argv[2], "w").write(t)
PY

grep -q "$FULL" "$OUTB" && grep -q "$WSHA" "$OUTB" && ! grep -qE '@[A-Z]+@' "$OUTB" && sh -n "$OUTB" \
  && echo "BODY_OK $OUTB commit ${FULL:0:9} wheel ${WSHA:0:12} repeats=$REPEATS probesecs=$PROBESECS put_key=$PKEY"
