#!/bin/sh
# Reproducible target-shape baseline/candidate matrix. The caller builds both
# binaries for the SAME target and supplies fresh evidence storage.
set -eu

[ "$#" -eq 3 ] || {
    echo "usage: $0 BASELINE_SO CANDIDATE_SO OUT_DIR" >&2
    exit 2
}
base=$1
candidate=$2
out=$3
root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
binding="$root/python/mojolearn/identical/_mojolearn_byte_lm.so"
[ -f "$base" ] && [ -f "$candidate" ] || {
    echo "both binding inputs must be regular files" >&2
    exit 2
}
[ ! -e "$out" ] || {
    echo "output already exists: $out" >&2
    exit 2
}
mkdir -p "$out"
export MOJOLEARN_NUMERIC_MODE=identical
export PYTHONPATH="$root/python${PYTHONPATH:+:$PYTHONPATH}"

run_arm() {
    round=$1
    name=$2
    binary=$3
    shift 3
    mkdir -p "$out/$round/$name"
    cp "$binary" "$binding"
    python3 "$root/tools/lm_shards_probe.py" \
        --out "$out/$round/$name" --target --shards 1 2 4 --steps 8 \
        --seed 93261 --devices "$@" --gpu-index 0 \
        >"$out/$round/$name.log" 2>&1
}

# Reverse the second round to expose thermal/cache/order drift.
run_arm round1 baseline_one "$base" 0
run_arm round1 candidate_one "$candidate" 0
run_arm round1 candidate_two "$candidate" 0 1
run_arm round1 baseline_two "$base" 0 1
run_arm round2 baseline_two "$base" 0 1
run_arm round2 candidate_two "$candidate" 0 1
run_arm round2 candidate_one "$candidate" 0
run_arm round2 baseline_one "$base" 0

for round in round1 round2; do
    python3 "$root/tools/lm_shards_ab_compare.py" \
        "$out/$round/baseline_one/result.json" \
        "$out/$round/baseline_two/result.json" \
        "$out/$round/candidate_one/result.json" \
        "$out/$round/candidate_two/result.json" \
        >"$out/$round/comparison.json"
done
sha256sum "$base" "$candidate" >"$out/binding_sha256.txt"
echo "PASS: strict full-state A/B matrix is in $out"
