#!/usr/bin/env bash
# Small GPU correctness matrix; never a performance benchmark.
set -euo pipefail
cd "$(dirname "$0")/.."
output_dir="${1:-$(mktemp -d /tmp/mojolearn-forest-layouts.XXXXXX)}"
mkdir -p "$output_dir"
for mode in fast identical; do
    for layout in separate_arrays packed_siblings; do
        defines=()
        if [[ "$mode" == identical ]]; then
            defines+=(-D MOJOLEARN_NUMERIC_IDENTICAL=1)
        fi
        if [[ "$layout" == packed_siblings ]]; then
            defines+=(-D MOJOLEARN_FOREST_PACKED_NODES=1)
        fi
        stem="$output_dir/$mode-$layout"
        echo "CHECK resident layout=$layout mode=$mode (small correctness fixture)"
        tools/with_build_lock.sh pixi run mojo build -I . "${defines[@]}" \
            checks/forest_inference_model.mojo -o "$stem" > "$stem.build.log" 2>&1
        tools/with_build_lock.sh "$stem" > "$stem.run.log" 2>&1
        tail -1 "$stem.run.log"
    done
done
echo "PASS forest resident layout matrix; logs: $output_dir"
