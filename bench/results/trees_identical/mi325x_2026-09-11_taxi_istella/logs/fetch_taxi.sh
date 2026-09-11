#!/bin/sh
# Leg 1: the setup's taxi fetch died on HTTP 403 (the TLC CloudFront refuses
# Python-urllib's default User-Agent). tools/speed_gbdt_arm.py now sends an
# explicit agent (pushed to the droplet); this re-runs the fetch and decode
# with it, and falls back to curl with the same agent if it still fails.
cd /root/mojolearn || exit 9
OUT=/root/trees_out
export GBM_BENCH_DATA=/mnt/mojolearn-data/gbm-bench
D=$GBM_BENCH_DATA/taxi
mkdir -p "$D"
timeout -k 30 1200 python3 tools/speed_gbdt_arm.py --download taxi > $OUT/logs/download_taxi.fixed.log 2>&1
rc=$?
echo "download_taxi_fixed=$rc $(date -u +%H:%M:%S)" >> $OUT/setup.txt
if [ "$rc" != 0 ]; then
    for m in 2024-01 2024-02; do
        curl -fL --retry 3 -A "Mozilla/5.0 (X11; Linux x86_64) mojolearn-bench" \
            -o "$D/yellow_tripdata_$m.parquet.part" \
            "https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_$m.parquet" \
            && mv "$D/yellow_tripdata_$m.parquet.part" "$D/yellow_tripdata_$m.parquet"
    done >> $OUT/logs/download_taxi.curl.log 2>&1
    timeout -k 30 1200 python3 tools/speed_gbdt_arm.py --download taxi >> $OUT/logs/download_taxi.curl.log 2>&1
    rc=$?
    echo "download_taxi_curl=$rc $(date -u +%H:%M:%S)" >> $OUT/setup.txt
fi
[ "$rc" = 0 ] && : > $OUT/taxi.ok
