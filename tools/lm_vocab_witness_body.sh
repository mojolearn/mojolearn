#!/bin/sh
# tools/lm_vocab_witness_body.sh -- lane/tokenized-corpus, 2026-09-18.
#
# Runs on the pod as tools/gemm_remote_leg.sh's MOJOLEARN_GEMM_LEG_EXTRA, from
# /root/mojolearn, with enwik8 and pile_github staged from R2:
#
#   MOJOLEARN_RUNPOD_KEY_FILE=$HOME/.mojolearn_runpod_key \
#   MOJOLEARN_GEMM_LEG_EXTRA=tools/lm_vocab_witness_body.sh \
#   MOJOLEARN_GEMM_LEG_LOCAL_CARD=bench/results/e1g/2026-09-13_221244-nvidia-h100-feature-freq-2710/local/apple.card \
#   MOJOLEARN_STAGE_KEYS="corpus/enwik8/input.txt corpus/pile_github/input.txt" MOJOLEARN_STAGE_STRICT=1 \
#   MOJOLEARN_GEMM_LEG_OUT=$HOME/mojolearn-evidence/tokenized-corpus-sep18/pod/<stamp> \
#   sh tools/gemm_remote_leg.sh nvidia --payload gemm --rent --allow-concurrent --minutes 90
#
# In order, each step's exit code in status.txt, never `set -e`:
#   1. build the IDENTICAL base and byte-LM bindings, the tokenizer host
#      binding and tokenizer/train/train_main;
#   2. train a VOCAB-rank vocabulary (default 8192) with train_main on the
#      first 10 MB of enwik8 (inside its train range) + the first 10 MB of
#      pile_github, timed;
#   3. prepare: tokenize ALL of enwik8 once with that vocabulary
#      (tools/lm_train.py --prepare-only): the encode throughput;
#   4. GRADIENT WITNESS, two arms at the SAME shape, seed and vocab_size
#      (1 256 64 4 2 16 128 2 n_vocab), two steps each, per-row gradient
#      reduction of `embed` and `lm_head`:
#        tokens  lm_train.py --vocab ranks: embedding rows >= 256 of ids that
#                occurred must be NONZERO;
#        bytes   lm_train.py --bytes (CorpusBatches): every embedding row
#                >= 256 must be EXACTLY zero;
#   5. BYTE PATH vs MAIN: tools/lm_step_memory_probe.py --corpus enwik8,
#      2 steps, --witness-every-step, under this branch's python/ and under a
#      copy whose _byte_lm_impl.py is rebuilt to main's exact bytes
#      (tools/lm_byte_path_main_copy.py). The hashes must be equal; a third
#      arm with another seed must DIFFER, or the comparison cannot fail.
set -u
ROOT=/root/mojolearn
OUT=/root/gemm_leg_out/lm-vocab-witness
mkdir -p "$OUT"
cd "$ROOT" || exit 9
ST="$OUT/status.txt"
PATH="$HOME/.pixi/bin:$PATH"
export PATH
export MOJOLEARN_NUMERIC_MODE=identical
export MOJOLEARN_TARGET_COLUMN="${MOJOLEARN_TARGET_COLUMN:-nvidia}"
export PYTHONPATH="$ROOT/python:$ROOT"
VOCAB="${MOJOLEARN_WITNESS_VOCAB:-8192}"
ENW=training/corpus/enwik8/input.txt
PILE=training/corpus/pile_github/input.txt

say() { echo "$*" >> "$ST"; }
say "started=$(date -u +%Y-%m-%dT%H:%M:%SZ) vocab=$VOCAB nproc=$(nproc) $(uname -m)"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader > "$OUT/gpu.txt" 2>&1
lscpu > "$OUT/lscpu.txt" 2>&1
free -g > "$OUT/free.txt" 2>&1

if [ -z "${MOJOLEARN_GPU_ARCHS:-}" ]; then
    cap=$(nvidia-smi -i 0 --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 | tr -d ' ')
    case "$cap" in
        9.0) MOJOLEARN_GPU_ARCHS=sm_90a ;;
        [0-9].[0-9]|1[0-9].[0-9]) MOJOLEARN_GPU_ARCHS="sm_$(echo "$cap" | tr -d .)" ;;
        *) say "arch unknown (compute_cap='$cap')"; exit 2 ;;
    esac
fi
export MOJOLEARN_GPU_ARCHS
say "gpu_archs=$MOJOLEARN_GPU_ARCHS"
for f in "$ENW" "$PILE"; do
    if [ -f "$f" ]; then say "corpus $f bytes=$(wc -c < "$f") sha256=$(sha256sum "$f" | cut -c1-64)"; else say "corpus MISSING $f"; fi
done
[ -f "$ENW" ] || { say "enwik8 not staged; nothing run"; exit 3; }

# 1. BUILD
t0=$(date +%s)
rm -f python/mojolearn/identical/_mojolearn.so python/mojolearn/identical/_mojolearn_byte_lm.so
MOJOLEARN_NUMERIC_MODE=identical MOJOLEARN_SKIP_BUILD_GATE=1 sh bindings/build.sh > "$OUT/build_base.log" 2>&1
say "build base exit=$? secs=$(( $(date +%s) - t0 ))"
t0=$(date +%s)
sh bindings/build_byte_lm.sh > "$OUT/build_byte_lm.log" 2>&1
say "build byte_lm exit=$? secs=$(( $(date +%s) - t0 ))"
t0=$(date +%s)
rm -f python/mojolearn/host/_mojolearn_tokenizer_host.so
sh bindings/build_tokenizer_host.sh > "$OUT/build_tokenizer.log" 2>&1
say "build tokenizer_host exit=$? secs=$(( $(date +%s) - t0 ))"
mkdir -p build
t0=$(date +%s)
pixi run mojo build -I . -o build/train_main tokenizer/train/train_main.mojo > "$OUT/build_train_main.log" 2>&1
say "build train_main exit=$? secs=$(( $(date +%s) - t0 ))"
if ! pixi run python -c 'import numpy' > "$OUT/numpy.log" 2>&1; then
    pixi run python -m pip install numpy >> "$OUT/numpy.log" 2>&1
    say "numpy pip exit=$?"
fi
sha256sum python/mojolearn/host/*.so python/mojolearn/identical/*.so build/train_main > "$OUT/binaries_sha256.txt" 2>&1

# 2. VOCABULARY
mkdir -p "$OUT/vocab"
head -c 10000000 "$ENW" > "$OUT/vocab/enwik8-head10M.txt"
head -c 10000000 "$PILE" > "$OUT/vocab/pile_github-head10M.txt" 2>/dev/null
t0=$(date +%s)
build/train_main "$OUT/vocab/mojolearn-bpe-$((VOCAB + 1))" "$VOCAB" 2 \
    "$OUT/vocab/enwik8-head10M.txt" "$OUT/vocab/pile_github-head10M.txt" > "$OUT/vocab/train.log" 2>&1
say "train_main vocab=$VOCAB exit=$? secs=$(( $(date +%s) - t0 ))"
rm -f "$OUT/vocab/enwik8-head10M.txt" "$OUT/vocab/pile_github-head10M.txt"
RANKS="$OUT/vocab/mojolearn-bpe-$((VOCAB + 1)).ranks.tsv"
[ -f "$RANKS" ] || { say "no ranks file; nothing more run"; exit 4; }
sha256sum "$RANKS" >> "$OUT/binaries_sha256.txt"

# 3. PREPARE: tokenize all of enwik8 once
t0=$(date +%s)
pixi run python tools/lm_train.py --corpus "$ENW" --vocab "$RANKS" --cache "$OUT/cache" \
    --out "$OUT/prepare" --prepare-only > "$OUT/prepare.log" 2>&1
say "prepare exit=$? secs=$(( $(date +%s) - t0 ))"
NV=$(pixi run python -c "import json;print(json.load(open('$OUT/prepare/run.json'))['tokens_manifest']['vocabulary']['n_vocab'])" 2>/dev/null)
say "n_vocab=$NV"
[ -n "$NV" ] || { say "prepare gave no n_vocab; nothing more run"; exit 5; }

# 4. GRADIENT WITNESS, both arms at the same shape and seed
SHAPE="1 256 64 4 2 16 128 2 $NV"
t0=$(date +%s)
# shellcheck disable=SC2086
pixi run python tools/lm_train.py --corpus "$ENW" --vocab "$RANKS" --cache "$OUT/cache" \
    --shape $SHAPE --steps 2 --witness-rows --out "$OUT/witness-tokens" > "$OUT/witness-tokens.log" 2>&1
say "witness tokens exit=$? secs=$(( $(date +%s) - t0 ))"
t0=$(date +%s)
# shellcheck disable=SC2086
pixi run python tools/lm_train.py --corpus "$ENW" --bytes \
    --shape $SHAPE --steps 2 --witness-rows --out "$OUT/witness-bytes" > "$OUT/witness-bytes.log" 2>&1
say "witness bytes exit=$? secs=$(( $(date +%s) - t0 ))"

# 5. BYTE PATH vs MAIN
pixi run python tools/lm_byte_path_main_copy.py python "$ROOT/python_main" > "$OUT/main_copy.log" 2>&1
say "main copy exit=$?"
for arm in branch main seed2; do
    case "$arm" in
        branch) PP="$ROOT/python:$ROOT"; SEED=93261 ;;
        main)   PP="$ROOT/python_main:$ROOT"; SEED=93261 ;;
        seed2)  PP="$ROOT/python:$ROOT"; SEED=93262 ;;
    esac
    t0=$(date +%s)
    PYTHONPATH="$PP" pixi run python tools/lm_step_memory_probe.py --out "$OUT/bytepath-$arm" \
        --corpus "$ENW" --steps 2 --seed "$SEED" --witness-every-step --budget-seconds 900 \
        > "$OUT/bytepath-$arm.log" 2>&1
    say "bytepath $arm exit=$? secs=$(( $(date +%s) - t0 ))"
done
say "finished=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
