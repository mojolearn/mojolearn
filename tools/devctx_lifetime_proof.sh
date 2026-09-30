#!/bin/sh
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
# THE devctx-lifetime PROOF.
#
#   sh tools/devctx_lifetime_proof.sh [OUT_DIR]
#
# On a Mac (Metal, with the API validation layer on) or a Linux GPU box:
#   1. the lane check (GPU == CPU, bit for bit) of every touched family, one
#      lane per invocation so one failure does not hide the rest; the par-*
#      lanes read KNOWN REFUSAL on their CPU arm;
#   2. THE STRESS: every one of those lanes, every fixture, three repeats, in
#      ONE identity_break process (many bindings called many times in one
#      process, the shape that ran the M2 Pro out of Metal command queues);
#      MOVED across repeats or any refusal fails it;
#   3. a scan of every log for the queue failure and for Metal's
#      "(kernel threadgroup size limit)" validation report.
# The last line is `DEVCTX PROOF: PASS` or `DEVCTX PROOF: FAIL (...)`.
set -u
cd "$(dirname "$0")/.."
PIXI=${PIXI:-pixi}
command -v "$PIXI" >/dev/null 2>&1 || PIXI="$HOME/.pixi/bin/pixi"
OUT=${1:-$HOME/mojolearn-evidence/devctx-lifetime/$(date -u +%Y%m%dT%H%M%SZ)}
mkdir -p "$OUT"
LANES=${DEVCTX_LANES:-"knn knn-clf knn-reg knn-rbc kmeans kmeans-weighted arima arima-seasonal-c pca-full-whiten tsvd-inverse ols ridge logistic dbscan kde kde-weighted gbdt-rmse gbdt-symmetric gbdt-ordered-rmse gbdt-feature-freq gbdt-multiclass hdbscan ivf ivf-extend linalg-eigh linalg-qr linalg-svdvals cholesky metrics metrics-classification metrics-homogeneity-completeness spectral spectral-embedding umap rf-clf rf-reg et-clf et-reg lasso elasticnet agglomerative holtwinters kpss select-d iforest svc mamba2 standard-scaler minmax-scaler bootstrap permutation-test monte-carlo"}
PAR=${DEVCTX_PAR:-"par-gp par-logistic"}
case "$(uname)" in
  Darwin) BACKEND=metal
          export MTL_DEBUG_LAYER=1 MTL_DEBUG_LAYER_ERROR_MODE=nslog ;;
  *) if command -v rocminfo >/dev/null 2>&1 && rocminfo 2>/dev/null | grep -q gfx; then BACKEND=hip; else BACKEND=cuda; fi ;;
esac
echo "devctx proof: $(hostname) backend=$BACKEND commit=$(git rev-parse --short HEAD) out=$OUT"
fails=""
for lane in $LANES $PAR; do
  sh tools/algos_lane_check.sh "$lane" --out "$OUT/check-$lane" > "$OUT/check-$lane.out" 2>&1
  last=$(tail -1 "$OUT/check-$lane.out")
  echo "CHECK $lane: $last"
  case "$last" in "RESULT: PASS"*) ;; *) fails="$fails check:$lane" ;; esac
done

# THE STRESS: one process, every lane, three repeats (the lane checks above built every binding).
# On a Mac this IS an intentional Metal investigation (the per-process queue
# class), so it names MOJOLEARN_APPLE_FULL_DIAGNOSTIC=1 for this one process.
csv=$(echo $LANES | tr ' ' ',')
MOJOLEARN_APPLE_FULL_DIAGNOSTIC=1 PYTHONPATH="$PWD/python" MOJOLEARN_NUMERIC_MODE=identical OMP_NUM_THREADS=1 \
  "$PIXI" run -e default python -u tools/identity_break.py --lanes "$csv" --repeats 3 --fail-on-refused \
  --require-backend "$BACKEND" --json "$OUT/stress.json" > "$OUT/stress.out" 2>&1
src=$?
echo "STRESS: $(echo $LANES | wc -w | tr -d ' ') lanes x every fixture x 3 repeats in one process: exit $src"
[ "$src" -eq 0 ] || fails="$fails stress(exit $src)"

q=$(cat "$OUT"/*.out "$OUT"/check-*/lane_check.log 2>/dev/null | grep -c "Failed to create Metal command queue")
v=$(cat "$OUT"/*.out "$OUT"/check-*/lane_check.log 2>/dev/null | grep -c "kernel threadgroup size limit")
echo "SCAN: $q 'Failed to create Metal command queue', $v 'kernel threadgroup size limit'"
[ "$q" -eq 0 ] || fails="$fails queue-refusals:$q"
[ "$v" -eq 0 ] || fails="$fails size-limit:$v"
if [ -z "$fails" ]; then echo "DEVCTX PROOF: PASS"; exit 0; fi
echo "DEVCTX PROOF: FAIL ($fails)"; exit 1
