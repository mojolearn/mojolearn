#!/bin/sh
# Run by the existing expiring rental supervisor. Each build/check is bounded.
set -eu
cd /root/mojolearn
OUT=/root/gemm_leg_out/wp67
mkdir -p "$OUT"
export PATH=/root/.pixi/bin:$PATH
export MOJOLEARN_GPU_ARCHS=sm_89 MOJOLEARN_TARGET_COLUMN=nvidia
export MOJOLEARN_SKIP_BUILD_GATE=1 MOJOLEARN_COMPILE_JOBS=2
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
PY=/opt/venv/bin/python
[ -x "$PY" ] || PY=python3
"$PY" -m pip install -q numpy scipy scikit-learn pytest einops > "$OUT/dependencies.log" 2>&1
# Keep the harness outside the source tree before reversing the product patch.
for helper in surface_capture classical_capture gp_predict_bench knn_classify_bench lm_surface verify_cache; do
    cp "tools/wp67_$helper.py" "/root/wp67_$helper.py"
done
# The invoking extra body provides a reviewed patch between baseline/candidate.
cp /root/wp67.patch "$OUT/source.patch"
# Candidate source arrives through the immutable source archive. Save changed
# source files, then reverse the patch; no checkout or shared environment edits.
python3 - <<'PY'
from pathlib import Path
import subprocess, json
patch=Path('/root/wp67.patch').read_text()
paths=[line[6:] for line in patch.splitlines() if line.startswith('+++ b/')]
backup=Path('/root/wp67-candidate')
for path in paths:
    p=Path(path)
    if p.exists():
        target=backup/path;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(p.read_bytes())
Path('/root/wp67-paths.json').write_text(json.dumps(paths))
subprocess.run(['git','apply','--reverse','/root/wp67.patch'],check=True)
PY
for arm in before after; do
    if [ "$arm" = after ]; then
        python3 - <<'PY'
from pathlib import Path
import json
for path in json.loads(Path('/root/wp67-paths.json').read_text()):
    p=Path(path); saved=Path('/root/wp67-candidate')/path
    if saved.exists():
        p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(saved.read_bytes())
PY
    fi
    for mode in identical deterministic fast; do
        export MOJOLEARN_NUMERIC_MODE=$mode
        DIR="$OUT/$arm/$mode"
        mkdir -p "$DIR"
        for binding in base gp byte_lm mamba svm metrics preprocessing estimators linalg arima tsa solver training transformer; do
            case "$binding" in byte_lm|mamba|training|transformer) [ "$mode" = identical ] || continue ;; esac
            script=bindings/build_$binding.sh
            [ "$binding" != base ] || script=bindings/build.sh
            if [ "$arm" = after ] && [ -f "$OUT/preflight-$mode.exit" ] && [ "$(cat "$OUT/preflight-$mode.exit")" = 0 ]; then
                python3 /root/wp67_verify_cache.py
                name=_mojolearn_$binding
                [ "$binding" != base ] || name=_mojolearn
                tierdir=$mode
                [ "$mode" != fast ] || tierdir=.
                cp "/root/wp67-compile/python/mojolearn/$tierdir/$name.so" "python/mojolearn/$tierdir/$name.so"
                printf 'reused source-verified preflight build %s\n' "$binding" > "$DIR/build-$binding.log"
                continue
            fi
            if [ "$binding" = byte_lm ]; then
                rm -f python/mojolearn/identical/_mojolearn_byte_lm.so
            fi
            printf '%s %s build %s\n' "$arm" "$mode" "$binding"
            timeout -k 10 300 sh "$script" > "$DIR/build-$binding.log" 2>&1
        done
        export PYTHONPATH=/root/mojolearn/python
        skip_surfaces=0
        if [ "$arm" = after ] && [ "$mode" = identical ] && [ -f "$OUT/after-identical-surfaces.exit" ] && [ "$(cat "$OUT/after-identical-surfaces.exit")" = 0 ]; then skip_surfaces=1; fi
        for surface in gp svr mamba mamba1_backward mamba23_backward transformer arima; do
            [ "$skip_surfaces" = 0 ] || continue
            case "$surface" in mamba|mamba1_backward|mamba23_backward|transformer) [ "$mode" = identical ] || continue ;; esac
            printf '%s %s surface %s\n' "$arm" "$mode" "$surface"
            MOJOLEARN_IDENTITY_TRACE="$DIR/$surface.trace" timeout -k 10 180 "$PY" \
                /root/wp67_surface_capture.py "python/mojolearn/tests/test_${surface}_surface.py" \
                "$DIR/$surface.bin" > "$DIR/surface-$surface.log" 2>&1
        done
        if [ "$skip_surfaces" = 0 ]; then
            timeout -k 10 90 "$PY" /root/wp67_classical_capture.py "$DIR/classical.bin" > "$DIR/classical.log" 2>&1
        fi
        if [ "$mode" = identical ] && [ "$skip_surfaces" = 0 ]; then
            for resident in 0 1; do
              for action in train eval; do
                WP67_LM_ACTION=$action WP67_LM_RESIDENT=$resident timeout -k 10 90 "$PY" /root/wp67_surface_capture.py /root/wp67_lm_surface.py \
                    "$DIR/byte_lm-$resident-$action.bin" > "$DIR/surface-byte_lm-$resident-$action.log" 2>&1
              done
            done
        fi
        if [ "$mode" = identical ] && [ "$skip_surfaces" = 0 ]; then
            WP67_LM_ACTION=train_eval WP67_LM_STEPS=2 WP67_LM_RESIDENT=1 timeout -k 5 45 "$PY" /root/wp67_surface_capture.py /root/wp67_lm_surface.py "$DIR/resident-multi.bin" > "$DIR/resident-multi.log" 2>&1
        fi
        mode_define=-D\ MOJOLEARN_NUMERIC_IDENTICAL=1
        [ "$mode" != deterministic ] || mode_define=-D\ MOJOLEARN_NUMERIC_DETERMINISTIC=1
        [ "$mode" != fast ] || mode_define=""
        pending=""
        count=0
        failed=0
        for check in kernel_methods/checks/km_check.mojo mixture/checks/gmm_check.mojo \
            kde/checks/kde_check.mojo resample/checks/resample_check.mojo \
            ivf/checks/ivf_check.mojo holtwinters/checks/hw_check.mojo \
            metrics/checks/trustworthiness_check.mojo spectral/checks/spectral_check.mojo \
            hdbscan/checks/hdbscan_check.mojo tsa/checks/stationarity_check.mojo umap/checks/transform_check.mojo umap/checks/estimator_check.mojo; do
            label=$(basename "$check" .mojo)
            printf '%s %s check %s\n' "$arm" "$mode" "$label"
            (
                if MOJOLEARN_IDENTITY_TRACE="$DIR/$label.trace" timeout -k 10 300 pixi run mojo \
                    -I . --target-accelerator sm_89 -D MOJOLEARN_COLUMN_NVIDIA $mode_define \
                    "$check" > "$DIR/check-$label.log" 2>&1; then
                    echo 0 > "$DIR/check-$label.exit"
                else
                    echo $? > "$DIR/check-$label.exit"
                    exit 1
                fi
            ) &
            pending="$pending $!"
            count=$((count + 1))
            if [ "$count" = 2 ]; then
                for child in $pending; do wait "$child" || failed=1; done
                pending=""; count=0
            fi
        done
        for child in $pending; do wait "$child" || failed=1; done
        [ "$failed" = 0 ] || exit 1
        [ ! -f /tmp/mojolearn.km.card.a ] || cp /tmp/mojolearn.km.card.a "$DIR/km.card"
        [ ! -f /tmp/gmm.card.a ] || cp /tmp/gmm.card.a "$DIR/gmm.card"
        printf '%s %s native surfaces complete\n' "$arm" "$mode"
    done
done
python3 - <<'PY'
from pathlib import Path
import json
root=Path('/root/gemm_leg_out/wp67'); rows=[]
for before in sorted([*(root/'before').rglob('*.bin'), *(root/'before').rglob('*.trace'), *(root/'before').rglob('*.card')]):
    after=root/'after'/before.relative_to(root/'before')
    same=before.read_bytes()==after.read_bytes()
    rows.append({'surface':str(before.relative_to(root/'before')),'bytes':before.stat().st_size,'bits_match':same})
    assert same, rows[-1]
(root/'comparisons.json').write_text(json.dumps(rows,indent=2)+'\n')
print('PASS',len(rows),'complete exported-array captures')
PY

MOJOLEARN_NUMERIC_MODE=identical timeout -k 10 120 "$PY" /root/wp67_gp_predict_bench.py "$OUT/before/identical/gp.so" "$OUT/after/identical/gp.so" "$OUT/gp-large.json" > "$OUT/gp-large.log" 2>&1

MOJOLEARN_NUMERIC_MODE=identical timeout -k 10 180 "$PY" /root/wp67_knn_classify_bench.py "$OUT/binaries/base-before.so" python/mojolearn/identical/_mojolearn.so "$OUT/knn-large.json" > "$OUT/knn-large.log" 2>&1
