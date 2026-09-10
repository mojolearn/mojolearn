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
            [ "$binding" != byte_lm ] || [ "$mode" = identical ] || continue
            script=bindings/build_$binding.sh
            [ "$binding" != base ] || script=bindings/build.sh
            if [ "$binding" = byte_lm ]; then
                rm -f python/mojolearn/identical/_mojolearn_byte_lm.so
            fi
            printf '%s %s build %s\n' "$arm" "$mode" "$binding"
            timeout -k 10 300 sh "$script" > "$DIR/build-$binding.log" 2>&1
        done
        export PYTHONPATH=/root/mojolearn/python
        for surface in gp svr mamba mamba1_backward mamba23_backward transformer arima; do
            printf '%s %s surface %s\n' "$arm" "$mode" "$surface"
            MOJOLEARN_IDENTITY_TRACE="$DIR/$surface.trace" timeout -k 10 180 "$PY" \
                /root/wp67_surface_capture.py "python/mojolearn/tests/test_${surface}_surface.py" \
                "$DIR/$surface.bin" > "$DIR/surface-$surface.log" 2>&1
        done
        if [ "$mode" = identical ]; then
            timeout -k 10 180 "$PY" /root/wp67_surface_capture.py packaging/language_model_smoke.py \
                "$DIR/byte_lm.bin" > "$DIR/surface-byte_lm.log" 2>&1
        fi
        mode_define=-D\ MOJOLEARN_NUMERIC_IDENTICAL=1
        [ "$mode" != deterministic ] || mode_define=-D\ MOJOLEARN_NUMERIC_DETERMINISTIC=1
        [ "$mode" != fast ] || mode_define=""
        for check in kernel_methods/checks/km_check.mojo mixture/checks/gmm_check.mojo \
            kde/checks/kde_check.mojo resample/checks/resample_check.mojo \
            ivf/checks/ivf_check.mojo holtwinters/checks/hw_check.mojo \
            metrics/checks/trustworthiness_check.mojo spectral/checks/spectral_check.mojo \
            hdbscan/checks/hdbscan_check.mojo tsa/checks/stationarity_check.mojo; do
            label=$(basename "$check" .mojo)
            printf '%s %s check %s\n' "$arm" "$mode" "$label"
            MOJOLEARN_IDENTITY_TRACE="$DIR/$label.trace" timeout -k 10 300 pixi run mojo \
                -I . --target-accelerator sm_89 -D MOJOLEARN_COLUMN_NVIDIA $mode_define \
                "$check" > "$DIR/check-$label.log" 2>&1
        done
        printf '%s %s native surfaces complete\n' "$arm" "$mode"
    done
done
python3 - <<'PY'
from pathlib import Path
import json
root=Path('/root/gemm_leg_out/wp67'); rows=[]
for before in sorted([*(root/'before').rglob('*.bin'), *(root/'before').rglob('*.trace')]):
    after=root/'after'/before.relative_to(root/'before')
    same=before.read_bytes()==after.read_bytes()
    rows.append({'surface':str(before.relative_to(root/'before')),'bytes':before.stat().st_size,'bits_match':same})
    assert same, rows[-1]
(root/'comparisons.json').write_text(json.dumps(rows,indent=2)+'\n')
print('PASS',len(rows),'complete exported-array captures')
PY
