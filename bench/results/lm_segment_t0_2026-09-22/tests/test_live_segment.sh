#!/bin/bash
set -u
PY=/Users/andrewhendel/CascadeProjects/mojolearn/.pixi/envs/test/bin/python
export PYTHONPATH=/Users/andrewhendel/mojolearn-wt/gpt3-tooling/python
cd /Users/andrewhendel/mojolearn-wt/gpt3-tooling
T=<scratchpad>/seg
S=tools/lm_segment.py; TOK=$(ls -d /Users/andrewhendel/mojolearn-evidence/tokenized-corpus-sep18/lmcache/tokens/2b49720ec4d78c3c-*/ | head -1)
rm -rf $T/LIVE $T/LIVE2
echo "=== live coordinator + in-process extra worker, blocks 0:2 and 2:4, held to the one-box chain A1"
$PY $S run --recipe $T/recipe.json --tokens $TOK --from $T/ckpt_00000000.blm --steps 4 --boundary 4 --out $T/LIVE --route L --segment live --label m4-live \
   --live-role coordinator --live-shards 0:2 --live-workers 2 --live-local-extra 2:4 --live-port 7811 --live-timeout 120 --expect-chain $T/A1/chain.jsonl 2>&1 | grep -E "coordinator|agrees|DISAGREE|checkpoint|PASS|FAIL|ERROR" | tail -8
echo "=== unequal blocks 0:1 and 1:4, from A1's checkpoint 4 into segment 2, held to A2"
$PY $S run --recipe $T/recipe.json --tokens $TOK --from $T/A1/ckpt_00000004.blm --steps 4 --boundary 8 --out $T/LIVE2 --route L --segment live2 --label m4-live \
   --live-role coordinator --live-shards 0:1 --live-workers 2 --live-local-extra 1:4 --live-port 7812 --live-timeout 120 --expect-chain $T/A2/chain.jsonl 2>&1 | grep -E "PASS|FAIL|ERROR" | tail -2
echo "=== chains and manifests against the one-box route"
$PY $S compare $T/A1/chain.jsonl $T/LIVE/chain.jsonl; $PY $S manifests $T/A1/manifest.tsv $T/LIVE/manifest.tsv; $PY $S compare $T/A2/chain.jsonl $T/LIVE2/chain.jsonl
head -c 400 $T/LIVE/coordinator.jsonl; echo
