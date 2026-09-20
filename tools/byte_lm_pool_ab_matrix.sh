#!/bin/sh
# Run fault/rollback/replay gates for baseline and candidate fault builds.
set -eu
[ "$#" -eq 4 ] || {
    echo "usage: $0 BASELINE_FAULT_SO CANDIDATE_FAULT_SO CORPUS OUT_DIR" >&2
    exit 2
}
base=$1 candidate=$2 corpus=$3 out=$4
root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
binding="$root/python/mojolearn/identical/_mojolearn_byte_lm.so"
[ -f "$base" ] && [ -f "$candidate" ] && [ -f "$corpus" ] || exit 2
[ ! -e "$out" ] || { echo "output already exists: $out" >&2; exit 2; }
mkdir -p "$out"
export MOJOLEARN_NUMERIC_MODE=identical
export PYTHONPATH="$root/python${PYTHONPATH:+:$PYTHONPATH}"
for shards in 2 3 5; do
    for arm in baseline candidate; do
        binary=$base
        [ "$arm" = candidate ] && binary=$candidate
        cp "$binary" "$binding"
        python3 "$root/tools/byte_lm_optimizer_pool_check.py" --cloud --faults \
            --logical-shards "$shards" --corpus "$corpus" \
            --report "$out/$arm-$shards.json" >"$out/$arm-$shards.log" 2>&1
    done
    python3 "$root/tools/byte_lm_pool_ab_compare.py" \
        "$out/baseline-$shards.json" "$out/candidate-$shards.json" \
        >"$out/comparison-$shards.json"
done
sha256sum "$base" "$candidate" >"$out/binding_sha256.txt"
echo "PASS: exact rollback/replay A/B matrix is in $out"
