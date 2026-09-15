# proof (b)-(d): readback, the CPU column on x86, and the sabotage column from the cached negative-control bindings
export OMP_NUM_THREADS=1 MOJOLEARN_CPU_THREADS=1
python3 tools/cpu_identity_gate_check.py readback --binding _mojolearn_core_host --binding _mojolearn_estimators_host > "$LEG_OUT/readback.txt" 2>&1
echo "readback_exit=$?" >> "$LEG_OUT/proof.txt"
python3 tools/identity_break.py --lanes ols,ridge,kmeans --fixtures base --repeats 2 --json "$LEG_OUT/cpu-x86.json" > "$LEG_OUT/identity.log" 2>&1
rc=$?; echo "identity_exit=$rc" >> "$LEG_OUT/proof.txt"
MOJOLEARN_HOST_DIR=python/mojolearn/host-sabotage MOJOLEARN_HOST_ALLOW_SABOTAGE=1 \
  python3 tools/identity_break.py --lanes ols,ridge,kmeans --fixtures base --repeats 2 --json "$LEG_OUT/cpu-x86.host-sabotage.json" > "$LEG_OUT/identity_sabotage.log" 2>&1
echo "sabotage_identity_exit=$?" >> "$LEG_OUT/proof.txt"
exit $rc
