#!/bin/bash
# On-pod payload for the Samba training lane, after pod_bootstrap.sh.
# $1 selects the phase: cards | bytelm | surface | samba | torch
set -u
ROOT=/root/mojolearn
OUT=/root/samba_out
mkdir -p "$OUT"
cd "$ROOT" || exit 9
export PATH="$HOME/.pixi/bin:$PATH"
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
export PYTHONPATH="$ROOT/python:$ROOT"
export MOJOLEARN_NUMERIC_MODE=identical
PY=$(pixi run which python)
phase="${1:-all}"
case "$phase" in
cards)
    mkdir -p "$OUT/lanes"
    for lane in training-loss:loss_check training-optimizer:optimizer_check; do
        name=${lane%%:*}; file=${lane#*:}
        MOJOLEARN_IDENTITY_TRACE="$OUT/lanes/$name.identical.card" \
            pixi run mojo run -D MOJOLEARN_NUMERIC_IDENTICAL=1 -I . "training/checks/$file.mojo" \
            > "$OUT/lanes/$name.identical.log" 2>&1
        echo "$name exit=$? md5=$(md5sum "$OUT/lanes/$name.identical.card" 2>/dev/null | cut -c1-8) records=$(grep -vc '^#\|^$' "$OUT/lanes/$name.identical.card" 2>/dev/null)"
    done
    ;;
bytelm)
    "$PY" tools/byte_lm_real_text_capture.py --expected-vendor cuda --steps 128 \
        --action continuous --output "$OUT/bytelm_full128" > "$OUT/bytelm_full128.log" 2>&1
    echo "bytelm exit=$?"
    tail -5 "$OUT/bytelm_full128.log"
    find "$OUT/bytelm_full128" -name '*.json' | head -20
    ;;
surface)
    (cd python && "$PY" -m mojolearn.tests.test_training_surface) > "$OUT/test_training_surface.log" 2>&1
    echo "training_surface exit=$?"; tail -3 "$OUT/test_training_surface.log"
    (cd python && "$PY" -m mojolearn.tests.test_samba_surface) > "$OUT/test_samba_surface.log" 2>&1
    echo "samba_surface exit=$?"; tail -3 "$OUT/test_samba_surface.log"
    ;;
samba)
    shift
    for run in 1 2; do
        rm -rf "$OUT/samba_run$run"
        "$PY" tools/samba_train_run.py --output "$OUT/samba_run$run" "$@" > "$OUT/samba_run$run.log" 2>&1
        echo "samba run$run exit=$?"; tail -2 "$OUT/samba_run$run.log"
    done
    ;;
torch)
    shift
    python3 tools/samba_torch_reference.py --output "$OUT/torch_ref" "$@" > "$OUT/torch_ref.log" 2>&1
    echo "torch exit=$?"; tail -2 "$OUT/torch_ref.log"
    ;;
esac
