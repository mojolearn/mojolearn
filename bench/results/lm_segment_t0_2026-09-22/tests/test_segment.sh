#!/bin/bash
set -u
PY=/Users/andrewhendel/CascadeProjects/mojolearn/.pixi/envs/test/bin/python
export PYTHONPATH=/Users/andrewhendel/mojolearn-wt/gpt3-tooling/python
cd /Users/andrewhendel/mojolearn-wt/gpt3-tooling
T=<scratchpad>/seg; rm -rf $T; mkdir -p $T
S="tools/lm_segment.py"
TOK=/Users/andrewhendel/mojolearn-evidence/tokenized-corpus-sep18/lmcache/tokens/2b49720ec4d78c3c-3d547b17821cf465-d1048576/
fail=0
step() { echo "=== $1"; }
step recipe; $PY $S recipe --out $T/recipe.json --shape 2 32 32 4 2 8 64 2 0 --tokens $TOK --shards 4 --steps 12 --peak-lr 1e-3 --warmup 2 --checkpoint-every 4 --boundaries 4,8,12 || fail=1
step keys; $PY $S keys --recipe $T/recipe.json --from-step 4 --steps 4 --boundary 8 | tr '\n' ' '; echo
step init; $PY $S init --recipe $T/recipe.json --tokens $TOK --out $T/ckpt_00000000.blm || fail=1
step A1; $PY $S run --recipe $T/recipe.json --tokens $TOK --from $T/ckpt_00000000.blm --steps 4 --boundary 4 --out $T/A1 --route A --segment 1 --label m4-a --record-window 0:1 | grep -E "checkpoint|PASS|FAIL|REFUSED" || fail=1
step A2; $PY $S run --recipe $T/recipe.json --tokens $TOK --from $T/A1/ckpt_00000004.blm --steps 4 --boundary 8 --out $T/A2 --route A --segment 2 --label m4-a | grep -E "PASS|FAIL|REFUSED" || fail=1
step A3; $PY $S run --recipe $T/recipe.json --tokens $TOK --from $T/A2/ckpt_00000008.blm --steps 4 --boundary 12 --out $T/A3 --route A --segment 3 --label m4-a | grep -E "PASS|FAIL|REFUSED" || fail=1
step "B1 (expect A1)"; $PY $S run --recipe $T/recipe.json --tokens $TOK --from $T/ckpt_00000000.blm --steps 4 --boundary 4 --out $T/B1 --route B --segment 1 --label m4-b --expect-chain $T/A1/chain.jsonl | grep -E "agrees|DISAGREE|PASS|FAIL" | tail -3 || fail=1
step "B2 from A1's checkpoint (expect A2)"; $PY $S run --recipe $T/recipe.json --tokens $TOK --from $T/A1/ckpt_00000004.blm --steps 4 --boundary 8 --out $T/B2 --route B --segment 2 --label m4-b --expect-chain $T/A2/chain.jsonl | grep -E "PASS|FAIL" || fail=1
step "arrival replay at boundary 4 (must PASS)"; $PY $S run --recipe $T/recipe.json --tokens $TOK --from $T/A1/ckpt_00000002.blm --steps 2 --no-checkpoints --out $T/ARR --route B --segment arrival --label m4-b --expect-chain $T/A1/chain.jsonl | grep -E "PASS|FAIL" || fail=1
step "zero-moments control (must FAIL)"; $PY $S run --recipe $T/recipe.json --tokens $TOK --from $T/A1/ckpt_00000002.blm --steps 2 --no-checkpoints --zero-moments --out $T/CTRL --route B --segment control --label m4-b --expect-chain $T/A1/chain.jsonl | grep -E "DISAGREE|PASS|FAIL"; rc=${PIPESTATUS[0]}; echo "control exit=$rc"; [ "$rc" = 1 ] || fail=1
step "compare chains"; $PY $S compare $T/A1/chain.jsonl $T/B1/chain.jsonl $T/ARR/chain.jsonl || fail=1
step "compare manifests"; $PY $S manifests $T/A1/manifest.tsv $T/B1/manifest.tsv || fail=1
step "wrong recipe (K=3) must be refused"; $PY $S recipe --out $T/recipe3.json --shape 2 32 32 4 2 8 64 2 0 --tokens $TOK --shards 3 --steps 12 --peak-lr 1e-3 --warmup 2 --checkpoint-every 4 > /dev/null; $PY $S run --recipe $T/recipe3.json --tokens $TOK --from $T/A1/ckpt_00000004.blm --steps 1 --out $T/K3 2>&1 | tail -1; rc=${PIPESTATUS[0]}; echo "refusal exit=$rc"; [ "$rc" != 0 ] || fail=1
step "chain line sample"; head -c 700 $T/A1/chain.jsonl; echo; cat $T/A1/manifest.tsv
echo "OVERALL fail=$fail"; exit $fail
