#!/bin/sh
# NOT RUN TO ANY CELL: pod 5qpixku3t7syju (driver 580.159.04) started setup at
# 14:28:59Z and was reaped at 14:34Z on the wind-down order (HTTP 404
# verified). No timing, fingerprint or buffer-check result exists. RUN OWED.
# H100 leg 2026-09-11 taxi (lane/trees-taxi-h100), IDENTICAL default build
# (DEVIATION 2502 pure-node leaf ON), never below 1M rows. NYC taxi opponent
# rows are measured ONCE here (cuML 26.08.00 RF, CatBoost GPU, XGBoost GPU,
# LightGBM CUDA if the setup built it, scikit-learn ExtraTrees on the pod
# CPU), each interleaved round by round with ours, 1 warm-up plus 5 rounds.
# Istella-S opponents are cached (pod 5gvizdykv4gqwm) and not re-run, except
# the new scikit-learn ET CPU row. FAST beside IDENTICAL, ours only, both
# datasets, all five lanes. XGBoost lossguide is NOT re-run: at depth 6 the
# 64-leaf budget cannot bind on any arm (ours, CatBoost, XGBoost), so the
# config already matches our lossguide arm and the tuple is unchanged.
cd /root/mojolearn || exit 9
AB="sh tools/trees_identical_ab.sh"
OUT=/root/trees_out
DATA="${GBM_BENCH_DATA:-/root/datasets/gbm-bench}"
PATH="$HOME/.pixi/bin:$PATH"; export PATH
export MOJOLEARN_NUMERIC_MODE=identical
mark() { echo "$1 $(date -u +%T)" | tee -a $OUT/ab.txt; : > "$OUT/phase.$1"; }
# No timed cell overlaps the Istella-S extraction and decode or the setup's
# closing real-cuML buffer check: once train.txt exists (or the step has
# exited) and setup.done is absent, wait. The tar download itself (network)
# may overlap; each cell's ledger line says whether it did.
quiet() {
    while [ ! -f $OUT/setup.done ] && { grep -q '^download_istella=' $OUT/setup.txt \
          || [ -n "$(find "$DATA/istella" -name train.txt 2>/dev/null | head -1)" ]; }; do
        sleep 10
    done
    echo "cell $1 $(date -u +%T) load $(cut -d' ' -f1-3 /proc/loadavg) setup_done=$([ -f $OUT/setup.done ] && echo 1 || echo 0) istella_tar_mb=$(du -m "$DATA/istella/istella-s-letor.tar.gz" 2>/dev/null | cut -f1)" >> $OUT/ab.txt
}
while [ ! -f $OUT/track_mojo.done ]; do sleep 15; done
echo "batchT start $(date -u +%T)" | tee -a $OUT/ab.txt
{ echo "nproc $(nproc)"; cat /sys/fs/cgroup/cpu.max 2>/dev/null; lscpu | grep -E '^CPU\(s\)|Model name'; \
  python3 -c "import joblib; print('joblib.cpu_count', joblib.cpu_count())" 2>&1; } > $OUT/cpu.txt

# ---------- PHASE_A: the IDENTICAL set the setup built (default build), then
# FAST builds of gbdt, rf, trees (base ships IDENTICAL only, DEVIATION 2490).
mkdir -p /root/bins/baseline && cp python/mojolearn/identical/*.so /root/bins/baseline/
sha256sum /root/bins/baseline/*.so | tee $OUT/baseline_so_sha256.txt
for b in gbdt rf trees; do
    MOJOLEARN_NUMERIC_MODE=fast MOJOLEARN_SKIP_BUILD_GATE=1 MOJOLEARN_COMPILE_JOBS=4 \
        timeout -k 30 1500 bash bindings/build_$b.sh > $OUT/logs/build.fast.$b.log 2>&1
    echo "build_fast_$b=$? $(date -u +%T)" | tee -a $OUT/ab.txt
done
sha256sum python/mojolearn/*.so > $OUT/fast_so_sha256.txt 2>&1
( cd python && python3 -c "
import mojolearn
for cls in (mojolearn.GradientBoosting, mojolearn.RandomForestClassifier, mojolearn.ExtraTreesClassifier):
    m = cls(numeric_mode='fast')
    print(cls.__name__, 'fast ->', m.numeric_mode_used(), m.vendor_used())
" ) > $OUT/logs/import_fast.log 2>&1
echo "import_fast=$? $(date -u +%T)" | tee -a $OUT/ab.txt
mark PHASE_A_DONE

# ---------- PHASE_B: fingerprints of the default set against the Sep 11
# H100 confirmation leg's default set (source 352d9781, 2502 ON).
cp bench/results/trees_identical/h100_2026-09-11/ib/baseline.json $OUT/ib/conf0911_baseline.json
$AB ib baseline
$AB diff conf0911_baseline baseline
mark PHASE_B_DONE

# ---------- PHASE_C: taxi opponents, once. Needs cuML/xgboost (track_pip),
# the taxi cache, and the LightGBM CUDA build finished (CPU heavy).
while ! grep -q '^download_taxi=' $OUT/setup.txt; do sleep 10; done
if ! grep -q '^download_taxi=0' $OUT/setup.txt; then
    mkdir -p "$DATA/taxi"
    for m in 2024-01 2024-02; do
        curl -fL --retry 3 -A "Mozilla/5.0 (X11; Linux x86_64) mojolearn-bench" \
            -o "$DATA/taxi/yellow_tripdata_$m.parquet.part" \
            "https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_$m.parquet" \
            && mv "$DATA/taxi/yellow_tripdata_$m.parquet.part" "$DATA/taxi/yellow_tripdata_$m.parquet"
    done >> $OUT/logs/download_taxi.curl.log 2>&1
    timeout -k 30 1200 python3 tools/speed_gbdt_arm.py --download taxi >> $OUT/logs/download_taxi.curl.log 2>&1
    echo "download_taxi_curl=$? $(date -u +%T)" | tee -a $OUT/ab.txt
fi
[ -f "$DATA/taxi/taxi_speed.npz" ] || { echo "TAXI CACHE MISSING $(date -u +%T)" | tee -a $OUT/ab.txt; exit 3; }
while [ ! -f $OUT/track_lgbm.done ]; do sleep 10; done
grep lightgbm_cuda $OUT/setup.txt
quiet rf.taxi.1m
MOJOLEARN_SPEED_ARMS=cuml-rf-gpu $AB speed baseline rf taxi 1000000 5 full
quiet sym.taxi.1m
MOJOLEARN_SPEED_ARMS=catboost-gpu $AB speed baseline gbdt-symmetric taxi 1000000 5 full
quiet dw.taxi.1m
MOJOLEARN_SPEED_ARMS=catboost-gpu,xgboost-gpu $AB speed baseline gbdt-depthwise taxi 1000000 5 full
quiet lg.taxi.1m
MOJOLEARN_SPEED_ARMS=catboost-gpu,xgboost-gpu,lightgbm-cuda $AB speed baseline gbdt-lossguide taxi 1000000 5 full
quiet et.taxi.1m
MOJOLEARN_SPEED_DEVICES=cpu,gpu MOJOLEARN_SPEED_ARMS=sklearn-et-cpu $AB speed baseline et taxi 1000000 5 full
quiet rf.taxi.2m
MOJOLEARN_SPEED_ARMS=cuml-rf-gpu $AB speed baseline rf taxi 2000000 5 full
mark PHASE_C_DONE

# ---------- PHASE_D: FAST beside IDENTICAL on taxi, ours only.
for L in gbdt-symmetric gbdt-depthwise gbdt-lossguide rf et; do
    quiet fast.$L.taxi
    MOJOLEARN_SPEED_OURS_AB="numeric_mode='fast'" MOJOLEARN_SPEED_TAG=fastab \
        $AB speed baseline $L taxi 1000000 5 ours
done
mark PHASE_D_DONE

# ---------- PHASE_E: Istella-S, after the setup finished (decode and buffer
# check done): the scikit-learn ET CPU row, then FAST beside IDENTICAL.
while [ ! -f $OUT/setup.done ]; do sleep 15; done
cat $OUT/setup.txt
[ -f "$DATA/istella/istella_speed.npz" ] || { echo "ISTELLA CACHE MISSING $(date -u +%T)" | tee -a $OUT/ab.txt; exit 4; }
quiet et.istella.1m
MOJOLEARN_SPEED_DEVICES=cpu,gpu MOJOLEARN_SPEED_ARMS=sklearn-et-cpu $AB speed baseline et istella 1000000 5 full
mark PHASE_E_DONE
for L in gbdt-symmetric gbdt-depthwise gbdt-lossguide rf et; do
    quiet fast.$L.istella
    MOJOLEARN_SPEED_OURS_AB="numeric_mode='fast'" MOJOLEARN_SPEED_TAG=fastab \
        $AB speed baseline $L istella 1000000 5 ours
done
mark PHASE_F_DONE
echo "batchT end $(date -u +%T)" | tee -a $OUT/ab.txt
