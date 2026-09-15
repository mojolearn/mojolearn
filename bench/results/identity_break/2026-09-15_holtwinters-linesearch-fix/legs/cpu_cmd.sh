set -u
# DEVIATION 2717, the CPU column of the tsa host family's covered lanes, the
# way cpu-identity-gate.yml runs it: readback, production column, column
# check, the four-column diff against the 166-lane record, then the sabotage
# column, which must diverge. kpss also needs the core host binding.
LANES=holtwinters,holtwinters-multiplicative,par-holtwinters,kpss
st() { echo "$1	$2" >> "$LEG_OUT/status.tsv"; }
C=$(cat commit.txt 2>/dev/null || cat COMMIT 2>/dev/null)
export MOJOLEARN_COMMIT=${MOJOLEARN_COMMIT:-$C}
echo "commit=$MOJOLEARN_COMMIT" > "$LEG_OUT/commit.txt"
COLS=$(python python/mojolearn/host_surface.py --training-gpu-columns)
for c in $COLS; do [ -f "$c" ] || { echo "missing column $c" >> "$LEG_OUT/status.tsv"; }; done
python tools/cpu_identity_gate_check.py readback --binding _mojolearn_tsa_host --binding _mojolearn_core_host \
    > "$LEG_OUT/readback.log" 2>&1; st readback $?
python tools/cpu_identity_gate_check.py run-column --lanes "$LANES" --shards 4 \
    --json "$LEG_OUT/cpu.json" > "$LEG_OUT/cpu_column.log" 2>&1; st run-column $?
python tools/cpu_identity_gate_check.py column "$LEG_OUT/cpu.json" --covered "$LANES" \
    --commit "$MOJOLEARN_COMMIT" --binding _mojolearn_tsa_host,_mojolearn_core_host > "$LEG_OUT/cpu_column_check.log" 2>&1; st column-check $?
python tools/identity_break.py --diff $COLS "$LEG_OUT/cpu.json" --require-columns 4 --lanes "$LANES" \
    --owed-json "$LEG_OUT/owed_cells.json" > "$LEG_OUT/diff_four_columns.txt" 2>&1; st diff-four $?
export MOJOLEARN_HOST_DIR=python/mojolearn/host-sabotage MOJOLEARN_HOST_ALLOW_SABOTAGE=1
python tools/cpu_identity_gate_check.py run-column --lanes "$LANES" --shards 4 \
    --json "$LEG_OUT/cpu-sab.json" -- --repeats 1 > "$LEG_OUT/cpu_sabotage.log" 2>&1; st run-column-sab $?
python tools/cpu_identity_gate_check.py column "$LEG_OUT/cpu-sab.json" --covered "$LANES" \
    --commit "$MOJOLEARN_COMMIT" > "$LEG_OUT/cpu_sabotage_check.log" 2>&1; st column-check-sab $?
python tools/identity_break.py --diff $COLS "$LEG_OUT/cpu-sab.json" --require-columns 4 --lanes "$LANES" \
    > "$LEG_OUT/diff_four_columns_sab.txt" 2>&1; st diff-four-sab-expect-nonzero $?
if [ -s "$LEG_OUT/owed_cells.json" ]; then
    python tools/cpu_identity_gate_check.py owed "$LEG_OUT/owed_cells.json" --production "$LEG_OUT/cpu.json" \
        --sabotage "$LEG_OUT/cpu-sab.json" > "$LEG_OUT/owed_sabotage_check.log" 2>&1; st owed-sab $?
fi
grep -E '^summary' "$LEG_OUT"/diff_four_columns*.txt > "$LEG_OUT/summary.txt"
