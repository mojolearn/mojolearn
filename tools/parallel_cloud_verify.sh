#!/bin/sh
# Run on a previously built RunPod host. No build or test is run locally.
set -u
: "${RUNPOD_POD_ID:?RunPod marker required}"
ROOT=${MOJOLEARN_PARALLEL_ROOT:-/root/mojolearn}
OUT=${MOJOLEARN_PARALLEL_OUT:-/root/parallel-out}
cd "$ROOT" || exit 2
export PATH="$HOME/.pixi/bin:$PATH"
export MOJOLEARN_NUMERIC_MODE=identical PYTHONPATH="$ROOT/python"
mkdir -p "$OUT"
: > "$OUT/verify.tsv"
phase() {
    name=$1; shift
    "$@" > "$OUT/verify-$name.log" 2>&1
    code=$?
    printf '%s\t%s\n' "$name" "$code" >> "$OUT/verify.tsv"
}
phase byte-lm pixi run python tools/byte_lm_parallel_check.py --cloud --corpus training/corpus/enwik8/input.txt --report "$OUT/byte-lm.json"
for lane in mlp forest samba; do
    phase "$lane" pixi run python tools/parallel_training_check.py --cloud --lane "$lane" --corpus training/corpus/enwik8/input.txt --report "$OUT/$lane.json"
done
phase samba-attention pixi run python tools/parallel_training_check.py --cloud --lane samba --attention-dropout --corpus training/corpus/enwik8/input.txt --report "$OUT/samba-attention.json"
phase kmeans pixi run python tools/parallel_kmeans_check.py --cloud --corpus training/corpus/enwik8/input.txt --report "$OUT/kmeans.json"
sha256sum python/mojolearn/identical/*.so > "$OUT/binaries.sha256"
python3 - "$OUT/source.sha256" <<'PY'
import hashlib, pathlib, sys
paths = sorted(p for directory in ('training', 'core', 'cluster', 'ensemble', 'extratrees', 'bindings', 'python/mojolearn', 'tools')
               for p in pathlib.Path(directory).rglob('*') if p.is_file() and p.suffix in ('.mojo', '.py', '.sh'))
pathlib.Path(sys.argv[1]).write_text(''.join(hashlib.sha256(p.read_bytes()).hexdigest() + '  ' + str(p) + '\n' for p in paths))
PY
awk '$2 != 0 {failed=1} END {exit failed}' "$OUT/verify.tsv"
