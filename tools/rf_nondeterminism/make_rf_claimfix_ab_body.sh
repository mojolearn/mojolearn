#!/bin/bash
# make_rf_claimfix_ab_body.sh <out body> [repeats] [probesecs]
# Fills rf_claimfix_ab_body.template.sh: the A/B of the pre-repair mutex claim (control,
# -D MOJOLEARN_RF_MUTEX_CLAIM_STOCK=1) against the repaired claim (default). Run it from the
# checkout whose HEAD carries the repair; the leg ships that HEAD. Needs only a presigned PUT.
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
REPO=$(cd "$HERE/../.." && pwd)
OUTB=${1:?usage: make_rf_claimfix_ab_body.sh <out body> [repeats] [probesecs]}
REPEATS=${2:-300}
PROBESECS=${3:-1500}
PKEY=legs/rf-mutex-claim-acquire/$(date -u +%Y-%m-%d_%H%M%S)/rf_claimfix_ab.json
FULL=$(git -C "$REPO" rev-parse HEAD)
PUTURL=$(cd "$REPO" && sh tools/dataset_store.sh presign-put "$PKEY" 10800)
[ -n "$PUTURL" ] || { echo "presign-put returned nothing for $PKEY" >&2; exit 1; }
FULL="$FULL" PUTURL="$PUTURL" REPEATS="$REPEATS" PROBESECS="$PROBESECS" \
python3 - "$HERE/rf_claimfix_ab_body.template.sh" "$OUTB" <<'FILL'
import os, sys
t = open(sys.argv[1]).read()
t = t.replace("@FULL@", os.environ["FULL"]).replace("@PUTURL@", os.environ["PUTURL"])
t = t.replace("@REPEATS@", os.environ["REPEATS"]).replace("@PROBESECS@", os.environ["PROBESECS"])
open(sys.argv[2], "w").write(t)
FILL
grep -q "$FULL" "$OUTB" && ! grep -qE '@[A-Z]+@' "$OUTB" && sh -n "$OUTB" \
  && echo "BODY_OK $OUTB commit ${FULL:0:9} repeats=$REPEATS put_key=$PKEY"
echo "$PKEY" > "$OUTB.putkey"
