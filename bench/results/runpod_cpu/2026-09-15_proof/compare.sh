#!/bin/bash
# compare.sh <leg dir>: the proof diffs for one pod leg, written beside the leg
set -u
SP=/private/tmp/claude-501/-Users-andrewhendel-CascadeProjects-mojolearn/e52730fc-0ee7-4bf3-8741-5fa0e0f1876f/scratchpad
L=$1; D=$L/remote/leg_out
cd $SP/wt-runpod-cpu || exit 9
set -- $(python3 python/mojolearn/host_surface.py --training-gpu-columns)
echo "== proof.txt"; cat $D/proof.txt
echo "== readback"; grep -E 'verdict|column=' $D/readback.txt
echo "== column check (pod)"
python3 tools/cpu_identity_gate_check.py column $D/cpu-x86.json --covered ols,ridge,kmeans \
  --commit "$(git rev-parse HEAD)" --binding _mojolearn_core_host,_mojolearn_estimators_host | tail -2
python3 -c "import json,sys;j=json.load(open(sys.argv[1]));print('pod vendor',j['vendor'],'cpu_model',j['host']['cpu_model'],'target',j['host'].get('target_cpu'))" $D/cpu-x86.json
echo "== (b) pod x86 vs M4"
python3 tools/identity_break.py --diff $D/cpu-x86.json $SP/m4ref/cpu-apple-m4.json > $L/diff_pod_m4.txt 2>&1; echo "rc=$?"
grep -E '^summary' $L/diff_pod_m4.txt
echo "== (b) pod + M4 vs the three committed GPU columns (base fixture rows)"
python3 tools/identity_break.py --diff $D/cpu-x86.json $SP/m4ref/cpu-apple-m4.json "$@" --lanes ols,ridge,kmeans > $L/diff_pod_m4_gpu3.txt 2>&1; echo "rc=$?"
grep -E '^\| (ols|ridge|kmeans)/base ' $L/diff_pod_m4_gpu3.txt | awk -F'|' '{print $2 "|" $3 "|" $4}'
grep -c -E 'DIVERGENT|MOVED' $L/diff_pod_m4_gpu3.txt
echo "== (d) pod production vs pod sabotage"
python3 tools/identity_break.py --diff $D/cpu-x86.json $D/cpu-x86.host-sabotage.json > $L/diff_pod_sabotage.txt 2>&1; echo "rc=$?"
grep -E '^summary' $L/diff_pod_sabotage.txt
echo "== bindings"; cat $D/so_sha256.txt
echo "== cache"; cat $D/cache.tsv; cut -f2,3,5 $D/bincache/provenance.tsv
echo "== status"; cat $D/status.tsv
echo "== timings"; cat $L/timings.tsv; grep -E 'VERIFIED|NOT CONFIRMED' $L/../../legs_*.log 2>/dev/null | tail -3
