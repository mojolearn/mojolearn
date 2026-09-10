#!/bin/bash
set -euo pipefail
export PATH=/root/.pixi/bin:$PATH
cd /root/identitygates
tar -xzf /root/mamba-fixtures.tar.gz
mkdir -p transformer/corpus
cp /root/transformer-gen-corpus.py transformer/corpus/gen_corpus.py
export MOJOLEARN_TRANSFORMER_REPO=/root/identitygates MOJOLEARN_TRANSFORMER_RESULTS=/root/evidence/transformer-fresh MOJOLEARN_TRANSFORMER_PYTHON=/usr/bin/python3
bash tools/transformer_fresh_prefill_leg.sh
