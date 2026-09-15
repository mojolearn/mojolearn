set -u
# lane/cpu-verifier-gaps-7: the CPU identity gate's covered-lanes path for the
# seven lanes, as .github/workflows/cpu-identity-gate.yml runs it (run-column,
# column, four-column diff with --owed-json, the sabotage run with the forest
# binary override, the sabotage diff, and the owed check). --build placed the
# production set in python/mojolearn/host, --sabotage-build the sabotage set in
# python/mojolearn/host-sabotage.
M=python/mojolearn/host_surface.py
LANES=gmm-sample,gmm-random-init-sample,gp-sample-y,gp-sample-y-normalize,tokenizer,gbdt-categorical-ctr-tables,gbdt-tensor-ctr-tables
BIND=_mojolearn_core_host,_mojolearn_preprocessing_host,_mojolearn_gp_host,_mojolearn_mixture_host,_mojolearn_tokenizer_host,_mojolearn_forest_host
export OMP_NUM_THREADS=1 MOJOLEARN_CPU_THREADS=1 MOJOLEARN_NUMERIC_MODE=identical
SHARDS=$(getconf _NPROCESSORS_ONLN); [ "$SHARDS" -le 4 ] || SHARDS=4
echo "commit $MOJOLEARN_COMMIT shards $SHARDS"
for l in ${LANES//,/ }; do
  case ",$(python3 $M --record-covered-lanes)," in *",$l,"*) echo "manifest: $l record-covered" ;; *) echo "manifest: $l NOT covered"; exit 1 ;; esac
done
export MOJOLEARN_IDENTITY_GBDT_CTR_MODELS=$PWD/$(python3 $M --gbdt-ctr-models)
ls "$MOJOLEARN_IDENTITY_GBDT_CTR_MODELS" | wc -l
GPU_COLUMNS=$(python3 $M --training-gpu-columns)
sha256sum python/mojolearn/host/*.so python/mojolearn/host-sabotage/*.so | tee "$LEG_OUT/so_sha256.txt"
bindings=""; for b in ${BIND//,/ }; do bindings="$bindings --binding $b"; done
python3 tools/cpu_identity_gate_check.py readback $bindings > "$LEG_OUT/readback.log" 2>&1; echo "readback exit $?"

echo "== production $(date +%T)"
python3 tools/cpu_identity_gate_check.py run-column --lanes "$LANES" --shards "$SHARDS" --json "$LEG_OUT/cpu.json" > "$LEG_OUT/cpu_column.log" 2>&1
echo "run-column exit $?"
python3 tools/cpu_identity_gate_check.py column "$LEG_OUT/cpu.json" --covered "$LANES" --commit "$MOJOLEARN_COMMIT" --binding "$BIND" > "$LEG_OUT/cpu_column_check.log" 2>&1
echo "column exit $?"; tail -n 3 "$LEG_OUT/cpu_column_check.log"
python3 tools/identity_break.py --diff $GPU_COLUMNS "$LEG_OUT/cpu.json" --require-columns 4 --lanes "$LANES" \
  --owed-json "$LEG_OUT/owed_cells.json" > "$LEG_OUT/diff_four_columns.txt" 2>&1
echo "diff production exit $? (must be 0)"; grep -E "^summary|require-columns" "$LEG_OUT/diff_four_columns.txt"

echo "== sabotage $(date +%T)"
(
  export MOJOLEARN_HOST_DIR=$PWD/python/mojolearn/host-sabotage MOJOLEARN_HOST_ALLOW_SABOTAGE=1
  export MOJOLEARN_FOREST_HOST_BINARY=$MOJOLEARN_HOST_DIR/_mojolearn_forest_host.so MOJOLEARN_FOREST_HOST_ALLOW_SABOTAGE=1
  python3 tools/cpu_identity_gate_check.py readback $bindings > "$LEG_OUT/readback_sab.log" 2>&1; echo "readback sabotage exit $?"
  python3 tools/cpu_identity_gate_check.py run-column --lanes "$LANES" --shards "$SHARDS" --json "$LEG_OUT/cpu-sab.json" -- --repeats 1 > "$LEG_OUT/cpu_sabotage.log" 2>&1
  echo "run-column sabotage exit $?"
)
python3 tools/cpu_identity_gate_check.py column "$LEG_OUT/cpu-sab.json" --covered "$LANES" --commit "$MOJOLEARN_COMMIT" > "$LEG_OUT/cpu_sabotage_check.log" 2>&1
echo "column sabotage exit $?"; tail -n 3 "$LEG_OUT/cpu_sabotage_check.log"
python3 tools/identity_break.py --diff $GPU_COLUMNS "$LEG_OUT/cpu-sab.json" --require-columns 4 --lanes "$LANES" > "$LEG_OUT/diff_four_columns_sab.txt" 2>&1
echo "diff sabotage exit $? (must be non-zero)"; grep -E "^summary" "$LEG_OUT/diff_four_columns_sab.txt"
python3 tools/cpu_identity_gate_check.py owed "$LEG_OUT/owed_cells.json" --production "$LEG_OUT/cpu.json" --sabotage "$LEG_OUT/cpu-sab.json" > "$LEG_OUT/owed_sabotage_check.log" 2>&1
echo "owed exit $?"; tail -n 1 "$LEG_OUT/owed_sabotage_check.log"
# every hashed part of every cell, production against sabotage
python3 - "$LEG_OUT/cpu.json" "$LEG_OUT/cpu-sab.json" > "$LEG_OUT/moved_every_part.txt" 2>&1 <<'EOF'
import json, sys
p, s = (json.load(open(f))["cells"] for f in sys.argv[1:3])
moved = same = na = 0
for k in sorted(p):
    for part in ("train", "infer", "model", "batch"):
        pv = p[k]["hashes"] if part == "train" else p[k].get(part)
        sv = s.get(k, {}).get("hashes") if part == "train" else s.get(k, {}).get(part)
        if not pv or str(pv[0]).startswith("n/a"):
            na += 1; continue
        if sv and sv[0] is not None and sv[0] != pv[0] and not str(sv[0]).startswith("n/a"):
            moved += 1
        else:
            same += 1; print("NOT MOVED", k, part, pv[0], sv)
print(f"hashed parts moved={moved} not_moved={same} n/a={na}")
EOF
tail -n 1 "$LEG_OUT/moved_every_part.txt"

echo "== against the Metal columns $(date +%T)"
E=bench/results/identity_break
R=$E/2026-09-14_166-lanes
python3 tools/identity_break.py --diff $E/2026-09-15_gmm-sample/apple-m4.json $E/2026-09-15_gp-sample-y/apple-m4.json \
  $E/2026-09-15_gbdt-ctr-tables/apple-m4.json $E/2026-09-15_cpu-verifier-gaps-7/metal-ctr/tensor-6fx.json \
  $E/2026-09-15_cpu-verifier-gaps-7/metal-ctr/categorical-*.json "$LEG_OUT/cpu.json" \
  --lanes "$LANES" --allow-separate-builds > "$LEG_OUT/diff_metal.txt" 2>&1
echo "diff metal exit $?"; grep -E "^summary" "$LEG_OUT/diff_metal.txt"

echo "== spot check existing lanes, base fixture $(date +%T)"
python3 tools/identity_break.py --lanes gmm,gmm-random-init,gp,gp-normalize-y --fixtures base --repeats 2 --json "$LEG_OUT/spot.json" > "$LEG_OUT/spot.log" 2>&1
echo "spot exit $?"
python3 tools/identity_break.py --diff $GPU_COLUMNS "$LEG_OUT/spot.json" --lanes gmm,gmm-random-init,gp,gp-normalize-y \
  --require-columns 4 --owed-json "$LEG_OUT/spot_owed.json" > "$LEG_OUT/diff_spot.txt" 2>&1
echo "diff spot exit $?"; grep -E "^summary" "$LEG_OUT/diff_spot.txt"
echo "== done $(date +%T)"
true
