#!/bin/sh
# runs ON the droplet: the 201 GEMM-reaching lanes in chunks of 25 (the body's
# list was under bench/results, which the bundle leaves out), then the GEMM A/B
set -u
ROOT=/root/mojolearn; OUT=/root/gemm_leg_out/amd-step-time; BIN=/root/amd_bin
cd "$ROOT" || exit 9
PATH="$HOME/.pixi/bin:$PATH"; export PATH
export MOJOLEARN_NUMERIC_MODE=identical MOJOLEARN_TARGET_COLUMN=amd MOJOLEARN_GPU_ARCHS=gfx942
export PYTHONPATH="$ROOT/python:$ROOT"
ST="$OUT/session.txt"
say() { echo "$(date -u +%H:%M:%S) $*" >> "$ST"; }
mkdir -p "$OUT/verify"
mv "$OUT/verify/chunk.log" "$OUT/verify/chunk-emptylist.log" 2>/dev/null
mv "$OUT/verify/chunk.json.gz" "$OUT/verify/chunk-emptylist.json.gz" 2>/dev/null
mv "$OUT/verify/chunk.json" "$OUT/verify/chunk-emptylist.json" 2>/dev/null
rm -f "$BIN"/lanechunk*
split -l 25 -d /root/amd_in/lanes_gemm_nonpar.txt "$BIN/lanechunk"
say "verify (lane list pushed; the body's copy was not in the bundle): $(ls "$BIN"/lanechunk* | wc -l) chunks"
for f in "$BIN"/lanechunk*; do
    k=$(basename "$f" | sed 's/lanechunk//')
    t0=$(date +%s)
    pixi run python -m mojolearn verify --lanes "$(tr '\n' ',' < "$f" | sed 's/,$//')" --json-out "$OUT/verify/chunk$k.json" > "$OUT/verify/chunk$k.log" 2>&1
    say "verify chunk$k exit=$? secs=$(( $(date +%s) - t0 )): $(grep RESULT "$OUT/verify/chunk$k.log" | tail -1 | cut -c1-300)"
done
gzip -9 -f "$OUT"/verify/*.json
say "verify done"
sh tools/amd_step_time_session.sh ab mi325x > /dev/null 2>&1
grep '^EXCP_AB call' "$OUT/ab/mi325x.log" | sed 's/ ms=.*//' > "$OUT/ab/mi325x.hashes"
say "ab mi325x: $(grep '^EXCP_AB call' "$OUT/ab/mi325x.log" | grep ordinary | awk '{split($2,a,"=");split($9,b,"="); printf "%s=%s ", a[2], b[2]}')"
rocm-smi --showclocks --showpower --showmaxpower --showperflevel > "$OUT/gpu_after.txt" 2>&1
say "remote extra done"
touch /root/amd_verify_done
