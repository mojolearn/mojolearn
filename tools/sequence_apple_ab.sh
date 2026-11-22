#!/bin/sh
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
# lane/sequence-apple2 (2026-09-28): the sequence family's Apple A/B, the
# timing command of ONE steward speed request (tools/apple_steward.py submit
# --kind speed, no --builds). Each variant is a `git worktree` at its commit
# with SAB_BUILDS built there in every mode of SAB_MODES; then this tree's
# tools/sequence_speed.py times every variant (SEQ_SPEED_PYTHON = the
# variant's python/), alternating variant by variant SAB_REPS times, so drift
# hits every arm alike. One line per record on stdout:
#   SAB rep=<r> mode=<m> variant=<label> {json record}
#   SAB_VARIANTS  space separated label=commit (default base=HEAD)
#   SAB_MODES     default "identical fast"
#   SAB_BUILDS    default bindings/build_x_sequence.sh
#   SAB_ALGOS     sequence_speed.py --algos (default all)
#   SAB_REPS      default 1
#   SAB_QUALITY   when set, also runs tools/sequence_quality.py per variant and mode
set -u
OUT=${SAB_OUT:-$HOME/mojolearn-evidence/sequence-apple2/ab-$(git rev-parse --short HEAD)-$(date -u +%H%M%S)}
mkdir -p "$OUT"
echo "SAB OUT $OUT host $(sysctl -n machdep.cpu.brand_string 2>/dev/null) conductor $(git rev-parse --short HEAD)"
HERE=$(pwd)
DATA=""
for d in "$HOME/data/higgs_speed.npz" "$HOME/datasets/gbm-bench/higgs/higgs_speed.npz"; do
    [ -f "$d" ] && { DATA=$d; break; }
done
[ -n "$DATA" ] || { echo "SAB no HIGGS npz on this box"; exit 3; }
labels=""
for v in ${SAB_VARIANTS:-base=HEAD}; do
    label=${v%%=*}; ref=${v#*=}
    wt="$OUT/wt-$label"
    git worktree add --detach "$wt" "$ref" > "$OUT/wt-$label.log" 2>&1 || { echo "SAB $label WORKTREE FAILED"; continue; }
    if cmp -s pixi.lock "$wt/pixi.lock"; then ln -s "$HERE/.pixi" "$wt/.pixi"; else echo "SAB $label pixi.lock differs; it installs its own env"; fi
    echo "SAB variant $label = $(git -C "$wt" rev-parse --short HEAD)"
    ok=1
    for mode in ${SAB_MODES:-identical fast}; do
        for s in ${SAB_BUILDS:-bindings/build_x_sequence.sh}; do
            case "$mode:$s" in fast:*_host.sh) continue ;; esac   # the CPU column builds IDENTICAL only
            t0=$(date +%s)
            (cd "$wt" && MOJOLEARN_SKIP_BUILD_GATE=1 MOJOLEARN_NUMERIC_MODE=$mode pixi run -e default sh "$s") >> "$OUT/build-$label-$mode.log" 2>&1 \
                || { echo "SAB $label BUILD $mode $s FAILED"; tail -n 30 "$OUT/build-$label-$mode.log"; ok=0; }
            echo "SAB build $label $mode $s wall_s=$(( $(date +%s) - t0 ))"
        done
    done
    if [ -n "${SAB_MATH:-}" ]; then
        # libMojolearnMath (packaging/portable_math), which ARIMA's python layer dlopens
        (cd "$wt" && PYTHONPATH=packaging/portable_math pixi run -e default python -c \
            "import pathlib, stage; stage.build(pathlib.Path('python/mojolearn/.dylibs/libMojolearnMath.dylib'))") \
            >> "$OUT/build-$label-math.log" 2>&1 || echo "SAB $label MATH BUILD FAILED"
    fi
    [ $ok = 1 ] && labels="$labels $label"
done
for rep in $(seq 1 "${SAB_REPS:-1}"); do
    for mode in ${SAB_MODES:-identical fast}; do
        for label in $labels; do
            SEQ_SPEED_PYTHON="$OUT/wt-$label/python" MOJOLEARN_NUMERIC_MODE=$mode \
                pixi run -e default python tools/sequence_speed.py --data "$DATA" ${SAB_ALGOS:+--algos "$SAB_ALGOS"} \
                --out "$OUT/speed-$label-$mode-$rep.json" > "$OUT/speed-$label-$mode-$rep.log" 2>&1
            grep -E '^\{' "$OUT/speed-$label-$mode-$rep.log" | sed "s/^/SAB rep=$rep mode=$mode variant=$label /"
            grep -E 'Traceback|Error' "$OUT/speed-$label-$mode-$rep.log" | head -n 5 | sed "s/^/SAB ERR $label $mode: /"
        done
    done
done
if [ -n "${SAB_CPU_ALGOS:-}" ]; then
    # the CPU column (the *_host bindings in SAB_BUILDS), once per variant and mode:
    # its digests before and after, never a timing that matters
    for mode in identical; do
        for label in $labels; do
            MOJOLEARN_VENDOR=cpu SEQ_SPEED_PYTHON="$OUT/wt-$label/python" MOJOLEARN_NUMERIC_MODE=$mode \
                pixi run -e default python tools/sequence_speed.py --data "$DATA" --algos "$SAB_CPU_ALGOS" \
                --out "$OUT/cpu-$label-$mode.json" > "$OUT/cpu-$label-$mode.log" 2>&1
            grep -E '^\{' "$OUT/cpu-$label-$mode.log" | sed "s/^/SAB CPU mode=$mode variant=$label /"
            grep -E 'Traceback|Error' "$OUT/cpu-$label-$mode.log" | head -n 5 | sed "s/^/SAB CPUERR $label $mode: /"
        done
    done
fi
if [ -n "${SAB_QUALITY:-}" ]; then
    for mode in ${SAB_MODES:-identical fast}; do
        for label in $labels; do
            SAB_LABEL=$label SEQ_SPEED_PYTHON="$OUT/wt-$label/python" MOJOLEARN_NUMERIC_MODE=$mode \
                pixi run -e default python tools/sequence_quality.py --data "$DATA" --what "$SAB_QUALITY" \
                > "$OUT/quality-$label-$mode.log" 2>&1
            grep -E '^QUAL' "$OUT/quality-$label-$mode.log" | sed "s/^/SAB mode=$mode variant=$label /"
            grep -E 'Traceback|Error' "$OUT/quality-$label-$mode.log" | head -n 5 | sed "s/^/SAB QERR $label $mode: /"
        done
    done
fi
for v in ${SAB_VARIANTS:-base=HEAD}; do git worktree remove --force "$OUT/wt-${v%%=*}" > /dev/null 2>&1; done
git worktree prune > /dev/null 2>&1
exit 0
