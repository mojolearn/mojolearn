#!/bin/bash
# lane/ties-sabotage: old arms (origin/main sources rebuilt here) then new arms, one RunPod CPU pod.
set -u
O="$LEG_OUT"
NB=knn-cosine,knn-rbc,radius,radius-manhattan
IV=ivf,ivf-euclidean
R=bench/results/identity_break/2026-09-14_166-lanes
C166="$R/apple-m4.json $R/nvidia-h100-sm_90a.json $R/amd-mi325x-gfx942.json"
V=bench/results/identity_break/2026-09-14_ivf-euclidean
CIVF="$V/identity_break.apple-m4.json $V/identity_break.nvidia-nvidia-h100-80gb-hbm3-sm_90a.json $V/identity_break.amd-mi300x-gfx942.json"
G=""; for c in $(python3 python/mojolearn/host_surface.py --classical-gpu-columns); do G="$G --gpu-column $c"; done
SAVED=$(python3 python/mojolearn/host_surface.py --saved-model-recorded)
ND=bench/results/classical_host/2026-09-15-apple-m4-neighbors-density
# The uplink limit (40 MB) ships only this classical recording of CLASSICAL_RECORDED: the 18 neighbor and density lanes.
CLASSICAL=$ND
IE=bench/results/classical_host/2026-09-15-apple-m4-ivf-embedding
SIX_REC="$(ls -d $ND/knn-cosine/* $ND/knn-rbc/* $ND/radius/* $ND/radius-manhattan/* $IE/ivf/* $IE/ivf-euclidean/* | tr '\n' ' ')"
NEW="MOJOLEARN_HOST_DIR=python/mojolearn/host-sabotage MOJOLEARN_HOST_ALLOW_SABOTAGE=1"
OLD="MOJOLEARN_HOST_DIR=/tmp/oldsab MOJOLEARN_HOST_ALLOW_SABOTAGE=1"
st() { local name=$1; shift; local t=$(date +%s); "$@"; local rc=$?; printf '%s\t%s\t%ss\n' "$name" "$rc" "$(( $(date +%s) - t ))" >> "$O/step_exit_codes.txt"; return 0; }
echo "nproc=$(nproc) $(uname -m) $(python3 --version 2>&1)" > "$O/box.txt"
sha256sum python/mojolearn/host/*.so python/mojolearn/host-sabotage/*.so > "$O/so_sha256.txt" 2>&1

# ---- the OLD arms: origin/main's two oracle files, rebuilt into /tmp/oldsab-build with the sabotage define
mkdir -p /tmp/old /tmp/branch /tmp/oldsab-build
echo "@@OLD_B64@@" | base64 -d | tar xzf - -C /tmp/old
sha256sum /tmp/old/* > "$O/old_sources_sha256.txt"
cp core/knn_host_predict.mojo ivf/host/ivf_host.mojo /tmp/branch/
cp /tmp/old/knn_host_predict.mojo core/knn_host_predict.mojo
cp /tmp/old/ivf_host.mojo ivf/host/ivf_host.mojo
grep -c "sabotage_value_flip" core/knn_host_predict.mojo ivf/host/ivf_host.mojo > "$O/old_sources_flip_count.txt" 2>&1
for f in core ivf ivf_search; do
  st "build_old_sab_$f" env MOJOLEARN_BUILD_EXTRA_DEFINES="-D MOJOLEARN_HOST_SABOTAGE=1" MOJOLEARN_HOST_OUTDIR=/tmp/oldsab-build \
    MOJOLEARN_BUILD_JOBS=$(nproc) sh bindings/build_${f}_host.sh > "$O/build_old_sab_$f.log" 2>&1
done
cp /tmp/branch/knn_host_predict.mojo core/knn_host_predict.mojo
cp /tmp/branch/ivf_host.mojo ivf/host/ivf_host.mojo
{ cmp /tmp/branch/knn_host_predict.mojo core/knn_host_predict.mojo && cmp /tmp/branch/ivf_host.mojo ivf/host/ivf_host.mojo && echo "branch sources restored"
  grep -c "sabotage_value_flip" core/knn_host_predict.mojo ivf/host/ivf_host.mojo; } >> "$O/old_sources_flip_count.txt" 2>&1
rm -rf /tmp/oldsab && mkdir -p /tmp/oldsab && cp python/mojolearn/host-sabotage/*.so /tmp/oldsab/ && cp /tmp/oldsab-build/*.so /tmp/oldsab/
sha256sum /tmp/oldsab-build/*.so > "$O/old_sab_sha256.txt" 2>&1

# ---- identity columns, the six lanes, all nine fixtures
st ib_prod python3 tools/identity_break.py --lanes $NB,$IV --repeats 2 --json "$O/cpu-x86.json" > "$O/ib-prod.log" 2>&1
st ib_sab_old env $OLD python3 tools/identity_break.py --lanes $NB,$IV --repeats 1 --json "$O/cpu-x86.sabotage-old.json" > "$O/ib-sab-old.log" 2>&1
st ib_sab_new env $NEW python3 tools/identity_break.py --lanes $NB,$IV --repeats 1 --json "$O/cpu-x86.sabotage-new.json" > "$O/ib-sab-new.log" 2>&1

# production against the committed records (unchanged hashes), and the owed parts
st diff_nb python3 tools/identity_break.py --diff $C166 "$O/cpu-x86.json" --require-columns 4 --lanes $NB --owed-json "$O/owed_nb.json" > "$O/diff.nb.record-vs-cpu.txt" 2>&1
st diff_iv python3 tools/identity_break.py --diff $CIVF "$O/cpu-x86.json" --require-columns 4 --lanes $IV --owed-json "$O/owed_iv.json" > "$O/diff.iv.record-vs-cpu.txt" 2>&1
for arm in old new; do
  st diff_nb_sab_$arm python3 tools/identity_break.py --diff $C166 "$O/cpu-x86.sabotage-$arm.json" --lanes $NB > "$O/diff.nb.record-vs-sabotage-$arm.txt" 2>&1
  st diff_iv_sab_$arm python3 tools/identity_break.py --diff $CIVF "$O/cpu-x86.sabotage-$arm.json" --lanes $IV > "$O/diff.iv.record-vs-sabotage-$arm.txt" 2>&1
  for w in nb iv; do
    st owed_${w}_$arm python3 tools/cpu_identity_gate_check.py owed "$O/owed_$w.json" --production "$O/cpu-x86.json" --sabotage "$O/cpu-x86.sabotage-$arm.json" > "$O/owed_${w}_sabotage_$arm.txt" 2>&1
  done
done

# per lane, fixture and part: did the sabotage value move from production?
python3 - "$O" > "$O/moved_counts.txt" 2>&1 <<'PY'
import json, sys
o = sys.argv[1]
prod = json.load(open(f"{o}/cpu-x86.json"))
fixtures = list(prod["fixtures"])
lanes = ["knn-cosine", "knn-rbc", "radius", "radius-manhattan", "ivf", "ivf-euclidean"]
def vals(cell, part):
    if not cell or cell.get("verdict") == "REFUSED":
        return None
    v = cell.get("hashes") if part == "train" else cell.get(part)
    return v[0] if v else None
for arm in ("old", "new"):
    sab = json.load(open(f"{o}/cpu-x86.sabotage-{arm}.json"))
    print(f"== sabotage {arm}")
    tot_moved = tot = 0
    for lane in lanes:
        row = []
        for fx in fixtures:
            key = f"{lane}/{fx}"
            parts = []
            for part in ("train", "infer", "model", "batch"):
                p, s = vals(prod["cells"].get(key), part), vals(sab["cells"].get(key), part)
                if p is None or str(p).startswith("n/a"):
                    continue
                tot += 1
                if s is not None and not str(s).startswith("n/a") and s != p:
                    tot_moved += 1
                    parts.append(part)
                else:
                    parts.append(part.upper() + "-UNMOVED")
            row.append(f"{fx}:{'+'.join(parts)}")
        print(lane, " ".join(row))
    print(f"sabotage {arm}: {tot_moved} of {tot} production cell parts moved")
PY

# ---- the classical host gate: production, then the old arms on the six lanes, then the new arms everywhere
st gate_prod_classical python3 tools/classical_host_gate.py check $CLASSICAL $G --report "$O/gate_prod_classical.json" > "$O/gate_prod_classical.txt" 2>&1
st gate_prod_saved python3 tools/classical_host_gate.py check $SAVED $G --report "$O/gate_prod_saved.json" > "$O/gate_prod_saved.txt" 2>&1
st gate_old_six_every_fixture env $OLD python3 tools/classical_host_gate.py check $SIX_REC --expect-mismatch --every-fixture --report "$O/gate_old_six_every_fixture.json" > "$O/gate_old_six_every_fixture.txt" 2>&1
st gate_old_saved_every_fixture env $OLD python3 tools/classical_host_gate.py check $SAVED --expect-mismatch --every-fixture --report "$O/gate_old_saved_every_fixture.json" > "$O/gate_old_saved_every_fixture.txt" 2>&1
st gate_old_saved_every_lane env $OLD python3 tools/classical_host_gate.py check $SAVED --expect-mismatch --every-lane --report "$O/gate_old_saved_every_lane.json" > "$O/gate_old_saved_every_lane.txt" 2>&1
st gate_new_six_every_fixture env $NEW python3 tools/classical_host_gate.py check $SIX_REC --expect-mismatch --every-fixture --report "$O/gate_new_six_every_fixture.json" > "$O/gate_new_six_every_fixture.txt" 2>&1
st gate_new_saved_every_fixture env $NEW python3 tools/classical_host_gate.py check $SAVED --expect-mismatch --every-fixture --report "$O/gate_new_saved_every_fixture.json" > "$O/gate_new_saved_every_fixture.txt" 2>&1
st gate_new_classical_every_fixture env $NEW python3 tools/classical_host_gate.py check $CLASSICAL --expect-mismatch --every-fixture --report "$O/gate_new_classical_every_fixture.json" > "$O/gate_new_classical_every_fixture.txt" 2>&1
st unit python3 tools/test_cpu_identity_gate.py > "$O/test_cpu_identity_gate.txt" 2>&1

{ cat "$O/step_exit_codes.txt"; cat "$O/old_sources_flip_count.txt"
  grep -H "sabotage .*production cell parts moved" "$O/moved_counts.txt"
  grep -HE "^summary|require-columns 4" "$O"/diff.*.txt
  grep -H "owed verdict" "$O"/owed_*.txt
  grep -H "gate verdict" "$O"/gate_*.txt
  grep -H "did not move" "$O"/gate_*.txt | sed 's/^.*gate_/gate_/' | sort | uniq -c | sort -rn | head -60
  tail -3 "$O/test_cpu_identity_gate.txt"; } > "$O/SUMMARY.out" 2>&1
exit 0
