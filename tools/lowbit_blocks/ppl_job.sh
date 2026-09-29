#!/bin/bash
# lane/lowbit-blocks (e): perplexity of SmolLM2-360M under fixed15_v1 and fp32_v1 on
# lane/lowbit-quality's two texts, by its protocol (tools/lowbit_blocks/ppl.py).
set -u
cd "$(dirname "$0")/../.." || exit 9
export PATH="$HOME/.pixi/bin:$PATH"
OUT=$PWD/bench/results/lowbit_blocks/${LB_BOX:-rtx4090-nvc2}/ppl
mkdir -p "$OUT"
M=/root/models/SmolLM2-360M
C=/root/mojolearn-wt/lowbit-quality/training/corpus
VP=${LB_TOKENIZER_PY:-/root/lowbit-quality-venv/bin/python}
$VP tools/lowbit_blocks/ppl.py ids --corpus $C/enwik8/input.txt --tail-from 99000000 --tokenizer $M/tokenizer.json --out "$OUT/ids_enwik8.bin"
$VP tools/lowbit_blocks/ppl.py ids --corpus $C/pile_github/input.txt --tail-from 96000000 --tokenizer $M/tokenizer.json --out "$OUT/ids_pile_github.bin"
for t in enwik8 pile_github; do
    for p in fixed15_v1 fp32_v1; do
        pixi run -e default python tools/lowbit_blocks/ppl.py score --ids "$OUT/ids_$t.bin" --model $M --profile $p \
            --batch ${LB_BATCH:-4} --out "$OUT/ppl_${t}_$p.json" 2>&1 | grep -v tcmalloc | grep -E "RESULT|rror|Traceback|windows 200"
    done
done
rm -f "$OUT"/ids_*.bin
echo "out $OUT"
