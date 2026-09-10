#!/bin/bash
set -euo pipefail
cd /root/mojolearn
export MOJOLEARN_NUMERIC_MODE=identical PYTHONPATH=python
trap 'echo "$?" > /root/forest_out/check-all.exit' EXIT
/root/forest_out/grove-scalar > /root/forest_out/grove-scalar.run.log 2>&1
/root/forest_out/grove-vector > /root/forest_out/grove-vector.run.log 2>&1
/root/forest_out/resident-check > /root/forest_out/resident-check.run.log 2>&1
python3 checks/forest_inference_public.py > /root/forest_out/public-identical.run.log 2>&1
sha256sum python/mojolearn/identical/_mojolearn_rf.so python/mojolearn/identical/_mojolearn_trees.so > /root/forest_out/baseline-bindings.sha256
