#!/bin/bash
# tools/ann_apple3_cuda.sh (lane ann-apple3, 2026-09-28): the lane's ONE small
# NVIDIA queue job (brief: a lane that changed code that is not Apple-only may
# show that the IDENTICAL digests did not move on CUDA).
#
#   tools/nvidia_central.sh sync ann-apple3 ~/mojolearn-wt/ann-apple3
#   tools/nvidia_central.sh submit ann-apple3 --gpus 1 --cap 90 \
#       /root/mojolearn-ann-apple3/tools/ann_apple3_cuda.sh
#
# ONE tree, IDENTICAL only, two builds of the ann bindings one after the other:
#   old  every ann-apple3 switch that is on by default turned OFF by its define
#        (the statements of lane/apple3-merged 6856b5f8f);
#   new  the default build of this tree.
# Each build runs bench/speed/ann_cpu_speed.py once (HIGGS 1M x 28 from R2, the
# Apple A/B's shapes) and the digests are compared three ways: old against new
# (this lane moved no bit on CUDA), the second search against the first, and
# new against the Apple IDENTICAL digests of m3ultra-b job 1790627848135 (the
# two vendors agree). Timing is printed and is not the point.
set -u
T=/root/mojolearn-ann-apple3
EV=${EV:-/root/ev-ann-apple3/$(date -u +%m%d-%H%M)}
DATA=${DATA:-/root/datasets/gbm-bench/higgs/higgs_speed.npz}
# every switch that is on by default names its revert define in
# x_ann/switches.mojo (`not is_defined["MOJOLEARN_ANN3_..._OFF"]`)
OFF=$(grep -o 'not is_defined\["MOJOLEARN_ANN3_[A-Z0-9_]*_OFF"\]' /root/mojolearn-ann-apple3/x_ann/switches.mojo \
      | grep -o 'MOJOLEARN_ANN3_[A-Z0-9_]*_OFF' | sort -u | sed 's/^/-D /' | paste -sd' ' -)
mkdir -p "$EV"
cd "$T" || exit 1
[ -f "$DATA" ] || { echo "ann_apple3_cuda: missing $DATA (stage it from R2: tools/dataset_store.sh stage)"; exit 2; }
echo "$(date -u +%FT%TZ) $(hostname) out $EV tree $(git log -1 --format=%h 2>/dev/null) old arm: $OFF"
[ -n "$OFF" ] || { echo "ann_apple3_cuda: no default-on switch found in x_ann/switches.mojo"; exit 2; }
nvidia-smi --query-gpu=name --format=csv,noheader | head -1
PIXI=$(command -v pixi || echo ~/.pixi/bin/pixi)
$PIXI install -e default > "$EV/pixi.log" 2>&1 || { echo "PIXI INSTALL FAIL"; tail -5 "$EV/pixi.log"; exit 1; }
export MOJOLEARN_SKIP_BUILD_GATE=1 MOJOLEARN_NUMERIC_MODE=identical MOJOLEARN_COMPILE_JOBS=${MOJOLEARN_COMPILE_JOBS:-6}

arm() {  # arm <name> <defines>
    for b in build.sh build_x_ann.sh build_estimators.sh build_ivf.sh; do
        t0=$(date +%s)
        if MOJOLEARN_MOJO_BUILD_FLAGS="$2" $PIXI run -e default sh "bindings/$b" > "$EV/build_$1_$b.log" 2>&1; then
            echo "BUILT $1 $b $2 $(( $(date +%s) - t0 ))s"
        else
            echo "BUILD FAIL $1 $b $2"; grep -v '^\s*$' "$EV/build_$1_$b.log" | tail -60; return 1
        fi
    done
    PYTHONPATH="$T/python" $PIXI run -e default python -u bench/speed/ann_cpu_speed.py --data "$DATA" \
        --algos all --out "$EV/$1.json" 2>&1 | grep -v '^\s*$' | sed "s/^/[$1 identical] /"
}

rc=0
arm old "$OFF" || rc=1
arm new "" || rc=1
# leave the tree's bindings as the default build (they are: `new` ran last)
$PIXI run -e default python - "$EV" <<'PY' || rc=1
import json, sys
ev = sys.argv[1]
apple = {  # m3ultra-b 1790627848135, IDENTICAL, equal in all three arms
    "ivf": (None, "d730b8082a1cdbfb"), "ivf_pq": ("6bb7a6c5fc753846", "3b1e0c1ae73444eb"),
    "ivf_sq": ("c55eedfcb6459d7b", "1d9c53fd8c13f452"), "ivf_rabitq": ("a4f2343eb268b0a7", "056a570709713477"),
    "refine": (None, "3c73bf8ae59e47e4"), "cagra": ("54d696296c9c7c8a", "45e435db03654b4e"),
    "tsne": (None, "2bb1d3d75ffa1885"),
}
try:
    old = json.load(open(f"{ev}/old.json"))["cells"]
    new = json.load(open(f"{ev}/new.json"))["cells"]
except OSError as exc:
    print("ANN-CUDA NO RESULT", exc)
    sys.exit(1)
bad = 0
for cell, want in apple.items():
    o, n = old.get(cell), new.get(cell)
    if o is None or n is None:
        print(f"ANN-CUDA {cell}: MISSING (old {o is not None}, new {n is not None})")
        bad += 1
        continue
    od, nd = (o.get("model"), o.get("out")), (n.get("model"), n.get("out"))
    warm = "n/a" if "out2" not in n else ("EQUAL" if n["out2"] == n["out"] else "DIFFERS")
    verdict = [
        "old==new " + ("EQUAL" if od == nd else "DIFFERS"),
        "second search " + warm,
        "new==apple " + ("EQUAL" if nd == want else "DIFFERS"),
        "old==apple " + ("EQUAL" if od == want else "DIFFERS"),
    ]
    if od != nd or warm == "DIFFERS":
        bad += 1
    print(f"ANN-CUDA {cell}: " + "; ".join(verdict) + f"; new {nd} fit {n.get('fit_s')} s search {n.get('search_s')} / "
          f"{n.get('search2_s')} s; old fit {o.get('fit_s')} s search {o.get('search_s')} / {o.get('search2_s')} s")
print("ANN-CUDA VERDICT", "PASS (this lane moved no IDENTICAL digest on CUDA)" if bad == 0 else f"FAIL ({bad} cells)")
sys.exit(0 if bad == 0 else 1)
PY
echo "$(date -u +%FT%TZ) done rc=$rc ev=$EV"
exit $rc
