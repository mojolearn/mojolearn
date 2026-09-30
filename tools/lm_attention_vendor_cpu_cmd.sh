# lane/attention-replay-vendors: CPU column identity_break AFTER and BEFORE
# (tools/runpod_cpu_leg.sh --cmd-file). The leg built the after host families
# into python/mojolearn/host; BEFORE rebuilds them at the SAME path from main's
# source (tools/lm_attention_replay_vendors_before.py apply). Runs with bash in
# /root/mojolearn, MOJOLEARN_COMMIT = shipped HEAD.
set -u
LANES=byte-lm,byte-lm-resident,byte-lm-host-infer,byte-lm-host-infer-threaded,byte-lm-host-train,transformer,transformer-window,samba,samba-untied-dropout-accum,par-byte-lm,par-byte-lm-model-pool,par-byte-lm-offload,par-samba,par-samba-clip
FAMS="core byte_lm transformer training neural mamba"
AFTER_SHA=$MOJOLEARN_COMMIT
BEFORE_SHA=$(python3 -c "import json;print(json.load(open('tools/lm_attention_replay_vendors_before.json'))['_base'])")
sha256sum python/mojolearn/host/*.so > "$LEG_OUT/after_bindings.sha256" 2>&1
python3 tools/identity_break.py --lanes "$LANES" --require-backend cpu --vendor cpu-x86 \
    --json "$LEG_OUT/identity_break.after.cpu-x86.json" > "$LEG_OUT/id_after.log" 2>&1
echo "id_after exit=$?" >> "$LEG_OUT/status.txt"
python3 tools/lm_attention_replay_vendors_before.py apply tools/lm_attention_replay_vendors_before.json > "$LEG_OUT/before_apply.log" 2>&1 || { echo "before apply FAILED" >> "$LEG_OUT/status.txt"; exit 1; }
grep -n "return column == COLUMN_NVIDIA$" checks/kernel_matrix.mojo >> "$LEG_OUT/before_apply.log"
rm -f python/mojolearn/host/*.so
for f in $FAMS; do
    env -u MOJOLEARN_GPU_ARCHS MOJOLEARN_TARGET_COLUMN=cpu MOJOLEARN_NUMERIC_MODE=identical MOJOLEARN_SKIP_BUILD_GATE=1 \
        MOJOLEARN_COMPILE_JOBS=8 sh "bindings/build_${f}_host.sh" > "$LEG_OUT/before_build_$f.log" 2>&1
    echo "before_build_$f exit=$?" >> "$LEG_OUT/status.txt"
done
sha256sum python/mojolearn/host/*.so > "$LEG_OUT/before_bindings.sha256" 2>&1
MOJOLEARN_COMMIT=$BEFORE_SHA python3 tools/identity_break.py --lanes "$LANES" --require-backend cpu --vendor cpu-x86 \
    --json "$LEG_OUT/identity_break.before.cpu-x86.json" > "$LEG_OUT/id_before.log" 2>&1
echo "id_before exit=$?" >> "$LEG_OUT/status.txt"
python3 tools/identity_break.py --diff "$LEG_OUT/identity_break.before.cpu-x86.json" \
    "$LEG_OUT/identity_break.after.cpu-x86.json" > "$LEG_OUT/id_diff.log" 2>&1
echo "id_diff exit=$?" >> "$LEG_OUT/status.txt"
python3 tools/lm_attention_replay_vendors_before.py restore tools/lm_attention_replay_vendors_before.json >> "$LEG_OUT/before_apply.log" 2>&1
