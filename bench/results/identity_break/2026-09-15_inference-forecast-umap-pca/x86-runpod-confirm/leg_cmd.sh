set -u
L=pca-full-whiten,umap,arima,arima-011,arima-seasonal-c
R=bench/results/identity_break/2026-09-14_166-lanes
C="$R/apple-m4.json $R/nvidia-h100-sm_90a.json $R/amd-mi325x-gfx942.json"
G="--gpu-column $R/apple-m4.json --gpu-column $R/nvidia-h100-sm_90a.json --gpu-column $R/amd-mi325x-gfx942.json"
python3 tools/identity_break.py --lanes $L --repeats 2 --json "$LEG_OUT/cpu-x86.json" > "$LEG_OUT/ib-cpu.log" 2>&1; echo "ib clean rc=$?"
MOJOLEARN_HOST_DIR=python/mojolearn/host-sabotage MOJOLEARN_HOST_ALLOW_SABOTAGE=1 python3 tools/identity_break.py --lanes $L --repeats 1 --json "$LEG_OUT/cpu-x86.sabotage.json" > "$LEG_OUT/ib-sab.log" 2>&1; echo "ib sab rc=$?"
python3 tools/identity_break.py --diff $C "$LEG_OUT/cpu-x86.json" --require-columns 4 --lanes $L --owed-json "$LEG_OUT/owed_cells.json" > "$LEG_OUT/diff.record-vs-cpu.txt" 2>&1; echo "diff clean rc=$?"
python3 tools/identity_break.py --diff $C "$LEG_OUT/cpu-x86.sabotage.json" --require-columns 4 --lanes $L > "$LEG_OUT/diff.record-vs-cpu-sabotage.txt" 2>&1; echo "diff sab rc=$? (1 expected)"
python3 tools/cpu_identity_gate_check.py owed "$LEG_OUT/owed_cells.json" --production "$LEG_OUT/cpu-x86.json" --sabotage "$LEG_OUT/cpu-x86.sabotage.json" > "$LEG_OUT/owed_sabotage_check.log" 2>&1; echo "owed rc=$?"
python3 tools/classical_host_gate.py check bench/results/classical_host/2026-09-15-apple-m4-umap-pca bench/results/classical_host/2026-09-15-apple-m4-arima $G --report "$LEG_OUT/classical_gate_cpu.json" > "$LEG_OUT/classical_gate_cpu.log" 2>&1; echo "gate clean rc=$?"
MOJOLEARN_HOST_DIR=python/mojolearn/host-sabotage MOJOLEARN_HOST_ALLOW_SABOTAGE=1 python3 tools/classical_host_gate.py check bench/results/classical_host/2026-09-15-apple-m4-umap-pca bench/results/classical_host/2026-09-15-apple-m4-arima --expect-mismatch --report "$LEG_OUT/classical_gate_sab.json" > "$LEG_OUT/classical_gate_sab.log" 2>&1; echo "gate sab rc=$?"
grep -hE "^summary|require-columns 4|gate verdict|owed verdict" "$LEG_OUT"/diff.*.txt "$LEG_OUT"/*.log
