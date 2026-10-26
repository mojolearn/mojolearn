#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# One BATCH of coverage sabotage arms on one box (lane precise-coverage-0859, 2026-10-26).
#
# `python3 tools/coverage_sabotage.py batch-plan --audit A.json --json PLAN.json` groups the
# unproven arms of tools/identity_lanes/coverage_sabotage.tsv into conflict-free batches: no
# arm's lanes can be moved by another arm's patch in the same batch, and the patches apply
# together. One sabotaged build per batch then proves every arm in it:
#
#   1. clean      the batch's lanes on this tree's built bindings (a record per column)
#   2. sabotaged  every patch of the batch applied, the batch's bindings rebuilt, the lanes again
#   3. restored   the patches reverted, the saved clean native files put back, the lanes again
#   4. check      tools/coverage_sabotage.py check per arm and column (--batch-bindings): the arm
#                 is CAUGHT when its lanes read DIVERGENT on the sabotaged build and IDENTICAL
#                 to clean on the restored build
#
# usage: bash tools/coverage_sabotage_batch.sh <plan.json> <batch id>
# env:   PYTHON        the interpreter (default: `pixi run -e default python` when pixi and
#                      pixi.toml are present, else python3)
#        BUILD_PREFIX  a command prefix for the binding builds (default: `pixi run -e default`
#                      when pixi is present, else none)
#        OUT           the output directory (default: $HOME/mojolearn-sabotage/<batch id>)
#        FIXTURES      identity_break fixtures (default: base)
# Output: one line per arm and column, `SABOTAGE batch=<id> arm=<arm> column=<c> verdict=CAUGHT|MISSED`,
# and $OUT/summary.txt; the records and per-arm check logs stay in $OUT.
# The tree is left as it was found (patches reverted, native files restored) on every exit path.
set -u
PLAN=${1:?usage: coverage_sabotage_batch.sh <plan.json> <batch id>}
BATCH=${2:?usage: coverage_sabotage_batch.sh <plan.json> <batch id>}
ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT"
OUT=${OUT:-$HOME/mojolearn-sabotage/$BATCH}
FIXTURES=${FIXTURES:-base}
mkdir -p "$OUT"
if [ -n "${PYTHON:-}" ]; then
    read -r -a PY <<< "$PYTHON"
elif command -v pixi > /dev/null 2>&1 && [ -f pixi.toml ]; then
    PY=(pixi run -e default python)
else
    PY=(python3)
fi
if [ -n "${BUILD_PREFIX+x}" ]; then
    read -r -a BP <<< "$BUILD_PREFIX"
elif command -v pixi > /dev/null 2>&1 && [ -f pixi.toml ]; then
    BP=(pixi run -e default)
else
    BP=()
fi
log() { echo "$(date -u +%FT%TZ) $*" | tee -a "$OUT/summary.txt"; }
: > "$OUT/summary.txt"

field() { python3 - "$PLAN" "$BATCH" "$1" <<'EOF'
import json, sys
plan, bid, key = sys.argv[1:4]
b = [x for x in json.load(open(plan))["batches"] if x["id"] == bid]
if not b:
    sys.exit(f"no batch {bid} in {plan}")
v = b[0][key]
print(",".join(v) if isinstance(v, list) else v)
EOF
}
ARMS=$(field arms) || exit 2
PATCHES=$(field patches) || exit 2
BUILDS=$(field builds) || exit 2
LANES=$(field lanes) || exit 2
COLUMNS=$(field columns) || exit 2
log "BATCH $BATCH arms=$ARMS builds=$BUILDS columns=$COLUMNS lanes=$LANES head=$(git rev-parse --short HEAD)"

if [ -n "$(git status --porcelain --untracked-files=no)" ]; then
    log "TREE NOT CLEAN"; exit 4
fi
# NATIVE_DIRS: where the builds write native files (default python/mojolearn; add a host set dir if
# the box keeps one outside the tree)
natives() { find ${NATIVE_DIRS:-python/mojolearn} \( -name '*.so' -o -name '*.dylib' \) -type f 2>/dev/null | sort; }
if command -v sha256sum > /dev/null 2>&1; then HASH=(sha256sum); else HASH=(shasum -a 256); fi
snap() { natives | while read -r f; do "${HASH[@]}" "$f"; done; }
natives > "$OUT/clean_natives.txt"
snap > "$OUT/clean_natives.sha"
tar -cf "$OUT/clean_natives.tar" -T "$OUT/clean_natives.txt"

run_cols() { # $1 = clean | sab | restored
    local tag=$1 rc
    case ",$COLUMNS," in *,device,*)
        MOJOLEARN_NUMERIC_MODE=identical PYTHONPATH="$ROOT/python" "${PY[@]}" tools/identity_break.py \
            --lanes "$LANES" --fixtures "$FIXTURES" --repeats 1 --json "$OUT/${tag}_device.json" \
            > "$OUT/${tag}_device.log" 2>&1; rc=$?; log "RUN $tag device rc=$rc";;
    esac
    case ",$COLUMNS," in *,cpu,*)
        MOJOLEARN_VENDOR=cpu MOJOLEARN_NUMERIC_MODE=identical PYTHONPATH="$ROOT/python" "${PY[@]}" tools/identity_break.py \
            --lanes "$LANES" --fixtures "$FIXTURES" --repeats 1 --json "$OUT/${tag}_cpu.json" \
            > "$OUT/${tag}_cpu.log" 2>&1; rc=$?; log "RUN $tag cpu rc=$rc";;
    esac
}

APPLIED=0
restore() {
    if [ "$APPLIED" = 1 ]; then
        # shellcheck disable=SC2046
        git apply -R $(python3 -c 'import sys; print(" ".join(reversed(sys.argv[1].split(","))))' "$PATCHES") \
            && log "REVERTED" || log "REVERT FAILED"
        APPLIED=0
    fi
    natives | while read -r f; do grep -qxF "$f" "$OUT/clean_natives.txt" || rm -f "$f"; done
    tar -xf "$OUT/clean_natives.tar"
    if [ "$(snap)" = "$(cat "$OUT/clean_natives.sha")" ]; then log "NATIVE RESTORED"; else log "NATIVE RESTORE FAILED"; fi
}
trap 'restore' EXIT INT TERM

run_cols clean
# shellcheck disable=SC2046
if ! git apply $(echo "$PATCHES" | tr ',' ' '); then
    log "APPLY FAILED"; exit 3
fi
APPLIED=1
brc=0
for b in ${BUILDS//,/ }; do
    t=$(date +%s)
    if [[ $b == *_host ]]; then
        env MOJOLEARN_COMPILE_JOBS=1 MOJOLEARN_TARGET_COLUMN=cpu "${BP[@]}" bash "bindings/$b.sh" > "$OUT/build_$b.log" 2>&1; rc=$?
    else
        env MOJOLEARN_COMPILE_JOBS=1 MOJOLEARN_NUMERIC_MODE=identical "${BP[@]}" bash "bindings/$b.sh" > "$OUT/build_$b.log" 2>&1; rc=$?
    fi
    log "BUILD $b rc=$rc secs=$(( $(date +%s) - t ))"
    if [ $rc -ne 0 ]; then
        brc=$rc
        grep -m 3 -B 2 -A 8 -i 'error' "$OUT/build_$b.log" | cut -c1-240 | head -30 >> "$OUT/summary.txt"
        break
    fi
done
[ $brc = 0 ] && run_cols sab
restore
trap - EXIT INT TERM
if [ $brc != 0 ]; then
    for arm in ${ARMS//,/ }; do echo "SABOTAGE batch=$BATCH arm=$arm column=all verdict=BUILD-FAIL" | tee -a "$OUT/summary.txt"; done
    exit 5
fi
run_cols restored
miss=0
for arm in ${ARMS//,/ }; do
    binds=$("${PY[@]}" tools/coverage_sabotage.py list | awk -F'\t' -v a="$arm" '$1==a {print $3}')
    for col in device cpu; do
        [ -f "$OUT/clean_$col.json" ] || continue
        # an arm is checked on the columns its own bindings build
        if [ $col = cpu ]; then [[ ",$binds," == *_host,* ]] || continue
        else echo ",$binds," | tr ',' '\n' | grep -qv -e '_host$' -e '^$' || continue; fi
        ev=""; [ -f "$OUT/evidence_${arm}_$col.json" ] && ev="--evidence $OUT/evidence_${arm}_$col.json"
        # shellcheck disable=SC2086
        "${PY[@]}" tools/coverage_sabotage.py check --arm "$arm" --clean "$OUT/clean_$col.json" \
            --sabotaged "$OUT/sab_$col.json" --restored "$OUT/restored_$col.json" \
            --batch-bindings "$BUILDS" $ev > "$OUT/check_${arm}_$col.txt" 2>&1
        v=$(tail -n 1 "$OUT/check_${arm}_$col.txt" | awk '{print $3}')
        [ "$v" = CAUGHT ] || miss=1
        echo "SABOTAGE batch=$BATCH arm=$arm column=$col verdict=${v:-ERROR}" | tee -a "$OUT/summary.txt"
    done
done
log "DONE"
exit $miss
