set -u
# The x86-64 CPU column for the Holt-Winters estimated-init lanes and the
# gbdt-binary-columns lane, 2026-09-22, under tools/runpod_cpu_leg.sh
# --build core,tsa,gbdt --sabotage-build core,tsa. One fit per cell. The
# sabotage column covers the four Holt-Winters lanes only (the gbdt lane's
# control is MOJOLEARN_GBDT_BINARY_SABOTAGE, recorded on the M4).
HW=holtwinters,holtwinters-multiplicative,par-holtwinters,par-forecast-holtwinters
st() { echo "$1	$2" >> "$LEG_OUT/status.tsv"; }
C=$(cat commit.txt 2>/dev/null || cat COMMIT 2>/dev/null)
export MOJOLEARN_COMMIT=${MOJOLEARN_COMMIT:-$C}
export MOJOLEARN_NUMERIC_MODE=identical
echo "commit=$MOJOLEARN_COMMIT" > "$LEG_OUT/commit.txt"
python tools/cpu_identity_gate_check.py readback --binding _mojolearn_tsa_host --binding _mojolearn_core_host \
    --binding _mojolearn_gbdt_host > "$LEG_OUT/readback.log" 2>&1; st readback $?
python tools/cpu_identity_gate_check.py run-column --lanes "$HW" --shards 4 \
    --json "$LEG_OUT/cpu-hw.json" -- --repeats 1 > "$LEG_OUT/cpu_hw_column.log" 2>&1; st run-column-hw $?
python tools/cpu_identity_gate_check.py column "$LEG_OUT/cpu-hw.json" --covered "$HW" \
    --commit "$MOJOLEARN_COMMIT" --binding _mojolearn_tsa_host > "$LEG_OUT/cpu_hw_column_check.log" 2>&1; st column-check-hw $?
python tools/cpu_identity_gate_check.py run-column --lanes gbdt-binary-columns --shards 1 \
    --json "$LEG_OUT/cpu-binary-columns.json" -- --repeats 1 --batch-grad --batch-scale --ragged --step-full \
    > "$LEG_OUT/cpu_binary_column.log" 2>&1; st run-column-binary $?
python tools/cpu_identity_gate_check.py column "$LEG_OUT/cpu-binary-columns.json" --covered gbdt-binary-columns \
    --commit "$MOJOLEARN_COMMIT" --binding _mojolearn_gbdt_host > "$LEG_OUT/cpu_binary_column_check.log" 2>&1; st column-check-binary $?
export MOJOLEARN_HOST_DIR=python/mojolearn/host-sabotage MOJOLEARN_HOST_ALLOW_SABOTAGE=1
python tools/cpu_identity_gate_check.py run-column --lanes "$HW" --shards 4 --sabotage --production "$LEG_OUT/cpu-hw.json" \
    --json "$LEG_OUT/cpu-hw-sab.json" -- --repeats 1 > "$LEG_OUT/cpu_hw_sabotage.log" 2>&1; st run-column-hw-sab $?
