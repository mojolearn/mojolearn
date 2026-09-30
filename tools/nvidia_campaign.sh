#!/usr/bin/env bash
# Main-lane, serial NVIDIA source/API campaign. No rental or publishing.
set -uo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT"
OUT=${MOJOLEARN_CAMPAIGN_OUT:?set artifact directory}
mkdir -p "$OUT"
OUT=$(cd "$OUT" && pwd)
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 NUMEXPR_NUM_THREADS=2
export MOJOLEARN_CPU_THREADS=2 MAX_JOBS=2 CMAKE_BUILD_PARALLEL_LEVEL=2
export PYTHONPATH="$ROOT/python"
export MOJOLEARN_MAMBA_CERT_VENDOR=nvidia
export MOJOLEARN_NUMERIC_MODE=identical
PY=${MOJOLEARN_CAMPAIGN_PYTHON:-python3}
cores=$("$PY" -c 'import os; print(",".join(map(str, sorted(os.sched_getaffinity(0))[:2])))')
taskset -pc "$cores" $$ > "$OUT/cpu-affinity.log" 2>&1 || exit 9
started=$(date +%s)
deadline=$((started + ${MOJOLEARN_CAMPAIGN_SECONDS:-3600} - 40))
rc=0
: > "$OUT/results.tsv"
run() {
    local name=$1 cap=$2 remaining status
    shift 2
    remaining=$((deadline - $(date +%s)))
    if ((remaining < 20)); then
        printf '%s\t124\tSKIPPED_DEADLINE\n' "$name" | tee -a "$OUT/results.tsv"
        rc=1
        return 124
    fi
    ((cap > remaining)) && cap=$remaining
    timeout -k 10 "$cap" "$@" > "$OUT/$name.log" 2>&1
    status=$?
    printf '%s\t%s\t%s\n' "$name" "$status" "$(date -u +%FT%TZ)" | tee -a "$OUT/results.tsv"
    ((status == 0)) || rc=1
    return "$status"
}
trap 'printf "%s\n" "$rc" > "$OUT/exit_code"' EXIT
{
    echo "commit=${MOJOLEARN_COMMIT:-unknown}"
    echo 'artifact_kind=source-built CUDA bindings; not an installed release wheel'
    echo 'serial_main_lane=true'
    echo "work_seconds=${MOJOLEARN_CAMPAIGN_SECONDS:-3600}"
    "$PY" --version
    nvidia-smi --query-gpu=name,driver_version --format=csv,noheader
} > "$OUT/environment.txt" 2>&1

# Preserve current source qualification, including the corrected Mamba2 API.
# Its internal jobs remain serial; outer timeout protects this campaign.
run umap-mamba-followup 900 env MOJOLEARN_FOLLOWUP_OUT="$OUT/followup" \
    bash tools/umap_mamba_followup.sh
run transformer-corpus 180 pixi run -e skgpu python transformer/corpus/gen_corpus.py
run transformer-forward 300 env MOJOLEARN_IDENTITY_TRACE="$OUT/transformer-forward.card" pixi run mojo run -D MOJOLEARN_NUMERIC_IDENTICAL=1 \
    -I . transformer/checks/transformer_check.mojo
run transformer-backward 300 env MOJOLEARN_IDENTITY_TRACE="$OUT/transformer-backward.card" pixi run mojo run -D MOJOLEARN_NUMERIC_IDENTICAL=1 \
    -I . transformer/checks/transformer_backward_check.mojo
if run build-transformer-identical 240 bash bindings/build_transformer.sh; then
    run transformer-api 180 pixi run -e skgpu python python/mojolearn/tests/test_transformer_surface.py
fi
run gemv-candidate 240 env MOJOLEARN_GEMV_LAYOUT_LARGE=1 pixi run mojo run -D MOJOLEARN_NUMERIC_IDENTICAL=1 \
    -I . bench/gemv_serial_layout_main.mojo
run knn-layout-candidate 240 env MOJOLEARN_KNN_LAYOUT_LARGE=1 pixi run mojo run -D MOJOLEARN_NUMERIC_IDENTICAL=1 \
    -I . bench/knn_index_layout_main.mojo

# One active compiler at a time. Partial builds produce explicit lane
# refusals; failed builds never cause source/native validation to disappear.
for mode in fast identical; do
    for binding in core linalg metrics; do
        script="bindings/build_${binding}.sh"
        [[ "$binding" = core ]] && script=bindings/build.sh
        run "build-$binding-$mode" 240 env MOJOLEARN_NUMERIC_MODE="$mode" bash "$script"
    done
done

# Use the image's CUDA PyTorch interpreter for comparator processes. Native
# extensions use the release's multi-Python ABI, and expose source paths in
# metadata. Never permit pip to compile a dependency from source.
run vendor-venv 30 "$PY" -m venv --system-site-packages "$ROOT/.nvidia-comparator-venv"
PY="$ROOT/.nvidia-comparator-venv/bin/python"
run vendor-torch-probe 30 "$PY" -c 'import torch,numpy; assert torch.cuda.is_available() and not torch.version.hip; print(torch.__version__, numpy.__version__)'
"$PY" -c 'import torch; print("torch==" + torch.__version__)' > "$OUT/vendor-constraints.txt"
run vendor-rapids-wheels 240 "$PY" -m pip install --disable-pip-version-check \
    --no-input --only-binary=:all: 'cupy-cuda12x' 'cuml-cu12' \
    --constraint "$OUT/vendor-constraints.txt" --extra-index-url https://pypi.nvidia.com
run vendor-freeze 30 "$PY" -m pip freeze
for lane in gemv nt gram knn umap; do
    run "compare-$lane" 300 "$PY" tools/nvidia_public_compare.py --lane "$lane" \
        --rounds 7 --out "$OUT/public-$lane"
done

# Fused Mamba availability is optional to the job, mandatory to admission
# of its own row. Wheel-only installs either succeed or refuse promptly.
run vendor-mamba-python-deps 60 "$PY" -m pip install --disable-pip-version-check \
    --no-input --only-binary=:all: einops packaging 'transformers==4.44.2'
cat > "$OUT/select_mamba_wheel.py" <<'PYWHEEL'
import json, sys, urllib.request
import torch
major, minor = torch.__version__.split('.')[:2]
cuda = torch.version.cuda.split('.')[0]
abi = str(torch._C._GLIBCXX_USE_CXX11_ABI).upper()
py = 'cp%d%d' % sys.version_info[:2]
needle = '+cu%storch%s.%scxx11abi%s-%s-%s-linux_x86_64.whl' % (cuda, major, minor, abi, py, py)
# This official release supplies the torch2.4/Python3.11 image's prebuilt
# selective_scan_cuda. Refuse another ABI; never attempt an sdist build.
req = urllib.request.Request('https://api.github.com/repos/state-spaces/mamba/releases/tags/v2.2.4',
                             headers={'Accept': 'application/vnd.github+json'})
with urllib.request.urlopen(req, timeout=20) as response:
    release = json.load(response)
matches = [a['browser_download_url'] for a in release['assets'] if needle in a['name']]
if len(matches) != 1:
    raise SystemExit('No unique prebuilt mamba wheel for ' + needle)
print(matches[0])
PYWHEEL
if run vendor-mamba-wheel-selection 40 "$PY" "$OUT/select_mamba_wheel.py"; then
    wheel_url=$(tail -1 "$OUT/vendor-mamba-wheel-selection.log")
    run vendor-mamba-wheel-install 120 "$PY" -m pip install --disable-pip-version-check \
        --no-input --no-deps --only-binary=:all: "$wheel_url"
fi

mkdir -p "$OUT/bin" "$OUT/seq-fast" "$OUT/seq-identical"
seq_ready=1
for mode in fast identical; do
    define=()
    [[ "$mode" = identical ]] && define=(-D MOJOLEARN_NUMERIC_IDENTICAL=1)
    run "build-sequence-$mode" 300 pixi run mojo build "${define[@]}" -I . \
        bench/speed/seq_speed_main.mojo -o "$OUT/bin/seq-$mode" || seq_ready=0
done
if [[ "$seq_ready" = 1 ]]; then
    for lane in transformer mamba; do
        row=2
        [[ "$lane" = mamba ]] && row=8
        fused=()
        [[ "$lane" = mamba ]] && fused=(--require-fused)
        # Setup leg supplies both mode logs/output dumps before external
        # admission; the following seven rounds rotate all three arms.
        for mode in fast identical; do
            run "seq-$lane-$mode-setup" 180 env MOJOLEARN_SPEED_LANE="$lane" \
                MOJOLEARN_SPEED_ROW="$row" MOJOLEARN_SPEED_ROUNDS=1 \
                MOJOLEARN_SPEED_DUMP_DIR="$OUT/seq-$mode" "$OUT/bin/seq-$mode"
        done
        for round in 1 2 3 4 5 6 7; do
            case $((round % 3)) in
                0) arms='fast identical external' ;;
                1) arms='identical external fast' ;;
                2) arms='external fast identical' ;;
            esac
            for arm in $arms; do
                if [[ "$arm" = external ]]; then
                    run "seq-$lane-external-$round" 180 "$PY" tools/speed_torch_seq.py \
                        --lane "$lane" --row "$row" --rounds 1 --warmups 1 --no-bf16 \
                        "${fused[@]}" --dump-dir "$OUT/seq-identical" \
                        --mojo-log "$OUT/seq-$lane-fast-setup.log" \
                        --mojo-log "$OUT/seq-$lane-identical-setup.log"
                else
                    run "seq-$lane-$arm-$round" 180 env MOJOLEARN_SPEED_LANE="$lane" \
                        MOJOLEARN_SPEED_ROW="$row" MOJOLEARN_SPEED_ROUNDS=1 \
                        MOJOLEARN_SPEED_DUMP_DIR="$OUT/seq-$arm" "$OUT/bin/seq-$arm"
                fi
            done
        done
        run "seq-$lane-fast-agreement" 180 "$PY" tools/speed_torch_seq.py \
            --lane "$lane" --row "$row" --rounds 1 --warmups 1 --no-bf16 \
            "${fused[@]}" --dump-dir "$OUT/seq-fast" \
            --mojo-log "$OUT/seq-$lane-fast-setup.log"
    done
fi
printf 'Mamba2/3 production-speed comparator: NOT IMPLEMENTED\n' > "$OUT/comparator-gaps.txt"
# Runtime binaries are disposable, not artifacts to fetch over a rental's
# reserve. Logs, output arrays, manifests, and qualification files remain.
rm -f "$OUT/bin/seq-fast" "$OUT/bin/seq-identical"
exit "$rc"
