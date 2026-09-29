#!/bin/bash
# lane/lowbit-default: THE RESIDENT SESSIONS UNDER fixed15_v1, on ONE box,
# ALONE on it (submit with every GPU slot).
#   1. build the bindings the model path needs (base, linalg, training,
#      transformer); the transformer binding carries the new entries
#   2. tools/lowbit_default/resident_gate.py: resident == per-layer, bit for
#      bit, both profiles (MUST exit 0)
#   3. generate timing, the default (now resident) against fp32_v1's resident
#      session, alternated (tools/lowbit_default/default_gate.py)
#   4. two SABOTAGE arms, each a copy of this tree with one edit of the new
#      resident code and the transformer binding rebuilt there; the gate MUST
#      exit 1 on each: (a) the resident block session swaps k_proj's and
#      v_proj's planes; (b) the resident head swaps its hi and lo planes
set -u
cd "$(dirname "$0")/../.." || exit 9
T=$PWD
export PATH="$HOME/.pixi/bin:$PATH"
export MOJOLEARN_NUMERIC_MODE=identical
unset MOJOLEARN_NUMERIC_PROFILE
MODEL=""
for d in "${LB_MODEL:-}" "$HOME/models/SmolLM2-360M" /root/models/SmolLM2-360M; do
    [ -n "$d" ] && [ -f "$d/model.safetensors" ] && { MODEL=$d; break; }
done
[ -n "$MODEL" ] || { echo "no staged SmolLM2-360M"; exit 2; }
BOX=${LB_BOX:-$(hostname -s)}
OUT=$T/bench/results/lowbit_default/$BOX/resident
mkdir -p "$OUT"
PX="pixi run --manifest-path $T/pixi.toml -e default"
nvidia-smi --query-gpu=index,name,driver_version --format=csv,noheader 2>/dev/null
red=0
if [ "${LB_BUILD:-1}" = 1 ]; then
    for b in build build_linalg build_training build_transformer; do
        $PX sh bindings/$b.sh > "$OUT/$b.log" 2>&1; rc=$?; echo "build $b exit $rc"
        [ $rc -eq 0 ] || { tail -30 "$OUT/$b.log"; red=1; }
    done
fi
[ $red -eq 0 ] || { echo "RESIDENT JOB RED (build)"; exit 1; }
# the gates of the kernels the resident step calls (lever 1's heads product
# and the block under the profile), each with its arms
gate() {  # task, expected (pass|fail)
    $PX $1 > "$OUT/$1.log" 2>&1; rc=$?
    if [ "$2" = pass ]; then [ $rc -eq 0 ] && v=held || { v=BROKEN; red=1; }
    else [ $rc -ne 0 ] && v=held || { v=BROKEN; red=1; }; fi
    echo "GATE $1 exit=$rc expected=$2 $v"; grep -E "^   ok|FAIL|gates," "$OUT/$1.log" | tail -8
}
gate check-gemm-int15-heads pass
gate check-gemm-int15-heads-sabotage fail
gate check-gemm-int15-heads-epilogue-sabotage fail
gate check-gemm-int15-heads-exponent-sabotage fail
gate check-gemm-int15-heads-host-sabotage fail
gate check-transformer-int15 pass
gate check-transformer-int15-sabotage fail
gate check-gemm-int15-tuned pass
echo "== resident gate (clean tree), MUST exit 0"
$PX python tools/lowbit_default/resident_gate.py --model "$MODEL" --new ${LB_NEW:-32} --box "$BOX" 2>&1 | grep -v tcmalloc | tee "$OUT/gate_clean.log"
rc=${PIPESTATUS[0]}; echo "resident gate clean exit $rc"; [ $rc -eq 0 ] || red=1
echo "== generate timing"
$PX python tools/lowbit_default/default_gate.py --model "$MODEL" --phases generate --new ${LB_NEW:-32} \
    --rounds ${LB_ROUNDS:-3} --box "$BOX" --out "$OUT" 2>&1 | grep -v tcmalloc | tee "$OUT/generate.log"
[ "${PIPESTATUS[0]}" -eq 0 ] || red=1
for arm in blocks head; do
    S=/root/lbd/sab_$arm
    rm -rf "$S"; mkdir -p "$S"
    tar --exclude=./.pixi --exclude=./bench/results --exclude=./.git -cf - . | tar -xf - -C "$S"
    F=$S/bindings/_mojolearn_transformer.mojo
    if [ $arm = blocks ]; then
        python3 - "$F" <<'PY'
import sys
p = sys.argv[1]; s = open(p).read()
a = "    var planes = _upload_planes(ctx, dims, opts, a, 13 + BLOCK_OPTION_ADDRS)\n"
assert s.count(a) == 1
s = s.replace(a, a + "    var sab_k = planes.k.copy()  # SABOTAGE\n    planes.k = planes.v.copy()\n    planes.v = sab_k^\n")
open(p, "w").write(s)
PY
    else
        python3 - "$F" <<'PY'
import sys
p = sys.argv[1]; s = open(p).read()
a = "    var a: List[Int] = [planes[0], planes[1], planes[2]]\n"
assert s.count(a) == 1
s = s.replace(a, "    var a: List[Int] = [planes[1], planes[0], planes[2]]  # SABOTAGE\n")
open(p, "w").write(s)
PY
    fi
    echo "sabotage sites in $arm copy: $(grep -c SABOTAGE\$ "$F")"
    (cd "$S" && $PX sh bindings/build_transformer.sh > "$OUT/sab_${arm}_build.log" 2>&1); echo "sab $arm build exit $?"
    echo "== resident gate, sabotage arm $arm, MUST exit 1"
    (cd "$S" && $PX python tools/lowbit_default/resident_gate.py --model "$MODEL" --new ${LB_NEW:-32} --box "$BOX-sab-$arm") 2>&1 | grep -v tcmalloc | tee "$OUT/gate_sab_$arm.log"
    rc=${PIPESTATUS[0]}; echo "resident gate sabotage $arm exit $rc"
    [ $rc -eq 1 ] || { echo "SABOTAGE ARM $arm DID NOT FAIL"; red=1; }
done
echo "out $OUT"
if [ $red -eq 0 ]; then echo "RESIDENT JOB GREEN"; else echo "RESIDENT JOB RED"; fi
exit $red
