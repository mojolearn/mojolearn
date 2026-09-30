#!/bin/sh
# Run ON a RunPod host after shipping source and staging the R2 corpus.
# Never invokes a local compiler or test runner from the laptop.
set -u
: "${RUNPOD_POD_ID:?RunPod marker required; invoke this script on the pod}"
ROOT=${MOJOLEARN_PARALLEL_ROOT:-/root/mojolearn}
OUT=${MOJOLEARN_PARALLEL_OUT:-/root/parallel-out}
cd "$ROOT" || exit 2
export PATH="$HOME/.pixi/bin:$PATH"
export MOJOLEARN_NUMERIC_MODE=identical MOJOLEARN_TARGET_COLUMN=nvidia
export MOJOLEARN_SKIP_BUILD_GATE=1 PYTHONPATH="$ROOT/python"
mkdir -p "$OUT"
nvidia-smi --query-gpu=name,compute_cap,driver_version,memory.total --format=csv > "$OUT/gpu.txt" || exit 2
cap=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -n 1 | tr -d ' ')
case "$cap" in 9.0) MOJOLEARN_GPU_ARCHS=sm_90a ;; *) MOJOLEARN_GPU_ARCHS=sm_$(printf '%s' "$cap" | tr -d '.') ;; esac
export MOJOLEARN_GPU_ARCHS
[ -f training/corpus/enwik8/input.txt ] || { echo 'Stage corpus/enwik8/input.txt from R2 first'; exit 2; }
sha256sum training/corpus/enwik8/input.txt > "$OUT/corpus.sha256"
phase() {
    name=$1; shift
    "$@" > "$OUT/$name.log" 2>&1
    code=$?
    printf '%s\t%s\n' "$name" "$code" >> "$OUT/status.tsv"
    return "$code"
}
: > "$OUT/status.tsv"
phase install pixi install || exit $?
for name in byte_lm base training linalg rf trees mamba transformer; do
    script=bindings/build_$name.sh
    [ "$name" = base ] && script=bindings/build.sh
    phase "build-$name" sh "$script" || exit $?
done
phase ordered-gradient pixi run mojo run --target-accelerator "$MOJOLEARN_GPU_ARCHS" -D MOJOLEARN_NUMERIC_IDENTICAL=1 -I . training/checks/ordered_gradient_check.mojo
phase byte-lm pixi run python tools/byte_lm_parallel_check.py --cloud --corpus training/corpus/enwik8/input.txt --report "$OUT/byte-lm.json"
phase existing-session pixi run python tools/byte_lm_session_check.py --run
for lane in mlp forest samba; do
    phase "$lane" pixi run python tools/parallel_training_check.py --cloud --lane "$lane" --corpus training/corpus/enwik8/input.txt --report "$OUT/$lane.json"
done
phase samba-attention pixi run python tools/parallel_training_check.py --cloud --lane samba --attention-dropout --corpus training/corpus/enwik8/input.txt --report "$OUT/samba-attention.json"
phase kmeans pixi run python tools/parallel_kmeans_check.py --cloud --corpus training/corpus/enwik8/input.txt --report "$OUT/kmeans.json"
sha256sum python/mojolearn/identical/*.so > "$OUT/binaries.sha256"
# Record the source independently of the checkout commit: overlays are legal.
python3 - "$OUT/source.sha256" <<'PY'
import hashlib, pathlib, sys
paths = sorted(p for directory in ('training', 'core', 'cluster', 'ensemble', 'extratrees', 'bindings', 'python/mojolearn', 'tools')
               for p in pathlib.Path(directory).rglob('*') if p.is_file() and p.suffix in ('.mojo', '.py', '.sh'))
pathlib.Path(sys.argv[1]).write_text(''.join(hashlib.sha256(p.read_bytes()).hexdigest() + '  ' + str(p) + '\n' for p in paths))
PY
awk '$2 != 0 {failed=1} END {exit failed}' "$OUT/status.tsv"
