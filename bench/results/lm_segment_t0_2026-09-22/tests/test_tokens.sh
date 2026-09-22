#!/bin/bash
set -u
PY=/Users/andrewhendel/CascadeProjects/mojolearn/.pixi/envs/test/bin/python
export PYTHONPATH=/Users/andrewhendel/mojolearn-wt/gpt3-tooling/python
cd /Users/andrewhendel/mojolearn-wt/gpt3-tooling
O=<scratchpad>/tok; rm -rf $O; mkdir -p $O
head -c 20000000 /Users/andrewhendel/mojolearn-evidence/gpt3-small-vocab-sep21/fineweb-edu-10BT-000-rg0-19.txt > $O/sample.txt
V=/Users/andrewhendel/mojolearn-evidence/gpt3-small-vocab-sep21/m4-shard000/vocab.ranks.tsv
echo "=== tokenize 20 MB of FineWeb-Edu text, last 50 documents held out"
$PY tools/fineweb_tokens.py --text $O/sample.txt --vocab $V --out $O/tokens --held-out-last 50 2>&1 | tail -2
$PY - <<PY
import json; m=json.load(open("$O/tokens/manifest.json"))
print("tokens", m["tokens"], "docs", m["n_documents"], "train_range", m["train_range"], "validation_range", m.get("validation_range"), "max_id", m["max_id"], "vocab n_vocab", m["vocabulary"]["n_vocab"], "vocab sha", m["vocabulary"]["sha256"][:16])
print("encode MB/s %.3f" % (m["encode_bytes_per_second"]/1e6))
PY
echo "=== TokenBatches reads it; recipe accepts it"
$PY tools/lm_segment.py recipe --out $O/recipe.json --shape 2 32 32 4 2 8 64 2 0 --tokens $O/tokens --shards 4 --steps 8 --peak-lr 1e-3 --warmup 2 --checkpoint-every 4 --boundaries 4,8 | tail -1
echo "=== decode round trip of the first document"
$PY - <<PY
import json, sys
sys.path.insert(0, "python")
from mojolearn import lm_corpus, tokenizer as tk
b = lm_corpus.TokenBatches("$O/tokens", 1, 16)
tok = lm_corpus.tokenizer_for(b.manifest, "$V")
first = open("$O/sample.txt","rb").readline()[:-1]
ids = tok.encode_batch([first])[0]
back = tok.decode_bytes(list(ids))
print("first document:", len(first), "bytes ->", len(ids), "ids; round trip equal:", back == first)
print("stream head equals first document ids:", list(b.ids_all._mv[:len(ids)]) == list(ids))
PY
