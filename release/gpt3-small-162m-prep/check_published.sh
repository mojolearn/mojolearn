#!/bin/bash
# After publishing: download the Hugging Face repo into a fresh directory and rerun
# every check a user would, from the downloaded files only.
#   bash release/gpt3-small-162m-prep/check_published.sh <namespace>/<repo> [workdir]
# Needs network access to huggingface.co and PyPI. Writes check.log in the workdir.
set -euo pipefail
repo=${1:?usage: check_published.sh <namespace>/<repo> [workdir]}
work=${2:-$HOME/mojolearn-evidence/hf-published-check-$(date +%Y%m%d-%H%M%S)}
mkdir -p "$work"
cd "$work"
python3 -m venv venv
venv/bin/pip -q install mojolearn numpy tokenizers safetensors torch huggingface_hub
venv/bin/hf download "$repo" --local-dir model > download.log 2>&1
cd model
{
  ../venv/bin/python -I tools/verify_release.py
  ../venv/bin/python -I tools/check_gpt3_evidence.py
  ../venv/bin/python -I tools/generate_mojolearn.py 16
  ../venv/bin/python -I reference_torch.py "The water cycle describes how water" --tokens 16
} > ../check.log 2>&1
echo "rc=$? log=$work/check.log"
grep -E 'PASS|Final step|Parameters SHA-256' ../check.log
