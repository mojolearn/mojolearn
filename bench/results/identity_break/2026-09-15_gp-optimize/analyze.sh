set -u
# Step 4 of lane/gp-optimizer: the Metal column against the CPU column, the
# owed accounting against the 166-lane record, and the gradient-sabotage
# negative control. Run after the pod results land in gpo/pod1.
WT=/private/tmp/claude-501/-Users-andrewhendel-CascadeProjects-mojolearn/e52730fc-0ee7-4bf3-8741-5fa0e0f1876f/scratchpad/wt-gp-optimizer
OUT=/private/tmp/claude-501/-Users-andrewhendel-CascadeProjects-mojolearn/e52730fc-0ee7-4bf3-8741-5fa0e0f1876f/scratchpad/gpo
POD=$OUT/pod1/remote/leg_out
REC=$WT/bench/results/identity_break/2026-09-14_166-lanes
export OUT POD REC
LANES=gp,gp-matern12,gp-matern32,gp-matern52-ard,gp-normalize-y,gp-optimize,gp-optimize-restarts
cd $WT
export PYTHONPATH=$WT/python
IB="python3 tools/identity_break.py --lanes $LANES"

echo "== Metal against the CPU column"
$IB --diff $OUT/apple-m4.json $POD/cpu.json --allow-separate-builds > $OUT/diff_metal_cpu.txt 2>&1
grep -E "^summary" $OUT/diff_metal_cpu.txt

echo "== the two columns against the record (owed)"
$IB --diff $OUT/apple-m4.json $POD/cpu.json $REC/apple-m4.json $REC/nvidia-h100-sm_90a.json \
    $REC/amd-mi325x-gfx942.json --require-columns 4 --allow-separate-builds \
    --owed-json $OUT/owed.json > $OUT/diff_record_owed.txt 2>&1
grep -E "^summary|require-columns" $OUT/diff_record_owed.txt | tail -5

echo "== the gradient sabotage against the production CPU column"
$IB --diff $POD/cpu.json $POD/cpu_gsab.json --allow-separate-builds > $OUT/diff_cpu_gsab.txt 2>&1
grep -E "^summary" $OUT/diff_cpu_gsab.txt

echo "== the gradient sabotage against Metal"
$IB --diff $OUT/apple-m4.json $POD/cpu_gsab.json --allow-separate-builds > $OUT/diff_metal_gsab.txt 2>&1
grep -E "^summary" $OUT/diff_metal_gsab.txt

echo "== which new cells the sabotage moved (every gp-optimize cell must move)"
python3 - <<'PY'
import json, os
out = os.environ["OUT"]
pod = os.path.join(out, "pod1", "remote", "leg_out")
prod = json.load(open(os.path.join(pod, "cpu.json")))["cells"]
sab = json.load(open(os.path.join(pod, "cpu_gsab.json")))["cells"]
# The columns may differ in --repeats, so compare the FIRST hash and the
# first parts dict, never the whole cell value (a length difference in
# "hashes" would make every cell read moved, a check that cannot fail).
for lane in ("gp-optimize", "gp-optimize-restarts"):
    keys = sorted(k for k in prod if k.split("/")[0] == lane)
    moved, same, missing = [], [], []
    for k in keys:
        if k not in sab:
            missing.append(k)
        elif sab[k]["hashes"][0] != prod[k]["hashes"][0] or sab[k].get("parts", [None])[0] != prod[k].get("parts", [None])[0]:
            moved.append(k)
        else:
            same.append(k)
    print(f"{lane}: {len(moved)} moved, {len(same)} unmoved, {len(missing)} missing, of {len(keys)}")
    for k in same:
        print("   UNMOVED", k, prod[k].get("verdict"))
    for k in missing:
        print("   MISSING", k)
PY
echo "== done"
