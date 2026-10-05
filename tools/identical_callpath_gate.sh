#!/usr/bin/env bash
# Compile and run the isolated callpath gate on a provisioned NVIDIA/AMD box.
# The controller must verify harvesting of OUT before invoking this script.
set -euo pipefail
vendor=${1:?nvidia or amd}
out=${2:?harvested output directory}
mkdir -p "$out"
out=$(cd "$out" && pwd)
root=$(cd "$(dirname "$0")/.." && pwd)
cd "$root"
case "$vendor" in
    nvidia) arch=sm_89; column=MOJOLEARN_COLUMN_NVIDIA ;;
    amd) arch=gfx942; column=MOJOLEARN_COLUMN_AMD ;;
    *) echo 'Only NVIDIA and AMD are supported by this runner' >&2; exit 2 ;;
esac
trap 'rc=$?; printf "%s\n" "$rc" > "$out/exit-code"' EXIT
git rev-parse HEAD > "$out/source-commit"
export MOJOLEARN_COMPILE_JOBS=1
slot=${MOJOLEARN_COMPILE_SLOT:-$HOME/mojolearn-evidence/compile_slot.sh}
test -f "$slot"
flags=(-j 1 -I . --target-accelerator "$arch" -D "$column"
       -D MOJOLEARN_NUMERIC_IDENTICAL -D MOJOLEARN_EXPERIMENT_IDENTICAL_CALLPATH)
for probe in compile_probe identity_gate; do
    bash "$slot" pixi run mojo build "${flags[@]}" \
        "experiments/identical_callpath/$probe.mojo" -o "$out/$probe" \
        > "$out/$probe-build.log" 2>&1
    printf '%s build=PASS\n' "$probe" >> "$out/summary.txt"
done
# compile_probe intentionally is never run: it instantiates typed interfaces.
timeout 180 "$out/identity_gate" > "$out/identity.log" 2>&1
grep -E '^CALLPATH_GATE .*status=PASS|^CALLPATH_GATE status=PASS' "$out/identity.log" >> "$out/summary.txt"
