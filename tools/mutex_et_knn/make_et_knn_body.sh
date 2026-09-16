#!/bin/bash
# make_et_knn_body.sh <out body>
# Fills the @PUTURL@ placeholder of tools/mutex_et_knn/et_knn_mutex_leg.sh with a fresh
# presigned R2 PUT, at a FRESH KEY. A reused key has already served one leg's numbers to
# another leg's monitor in this repository.
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
REPO=$(cd "$HERE/../.." && pwd)
OUTB=${1:?usage: make_et_knn_body.sh <out body>}
STAMP=$(date -u +%Y-%m-%d_%H%M%S)
PKEY="legs/mutex-et-knn/$STAMP/et_knn.json"
PUTURL=$(cd "$REPO" && sh tools/dataset_store.sh presign-put "$PKEY" 10800)
[ -n "$PUTURL" ] || { echo "presign-put returned nothing for $PKEY" >&2; exit 1; }
PUTURL="$PUTURL" python3 - "$HERE/et_knn_mutex_leg.sh" "$OUTB" <<'PY'
import os, sys
t = open(sys.argv[1]).read()
if "@PUTURL@" not in t:
    raise SystemExit("the template has no @PUTURL@ left to fill")
open(sys.argv[2], "w").write(t.replace("@PUTURL@", os.environ["PUTURL"]))
PY
# Verify the GENERATED artifact, not the template: two legs in this family died from a
# change that never reached what ran.
! grep -qE '@[A-Z]+@' "$OUTB" || { echo "unfilled placeholder in $OUTB" >&2; exit 1; }
sh -n "$OUTB"
grep -c 'MOJOLEARN_RF_MUTEX_CLAIM_STOCK=1' "$OUTB" > /dev/null
echo "BODY_OK $OUTB put_key=$PKEY stock_define_occurrences=$(grep -c 'MOJOLEARN_RF_MUTEX_CLAIM_STOCK=1' "$OUTB")"
