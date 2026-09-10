#!/bin/bash
set -euo pipefail
export PATH=/root/.pixi/bin:$PATH
cd /root/knnfinal
while [ ! -f /root/jobs/knn-final.rc ]; do sleep 3; done
if [ "${KNN_PUBLIC_ACTIVE:-0}" != 1 ]; then
 export KNN_PUBLIC_ACTIVE=1
 exec pixi run bash /root/jobs/knn-public.sh
fi
out=/root/evidence/knn-final
mkdir -p python/mojolearn/identical
mojo build --emit shared-lib -D MOJOLEARN_NUMERIC_IDENTICAL=1 -I . -I bindings bindings/_mojolearn.mojo -o python/mojolearn/identical/_mojolearn.so > "$out/build-public-binding.log" 2>&1
MOJOLEARN_NUMERIC_MODE=identical PYTHONPATH=python /usr/bin/python3 tools/knn_public_query_batch_check.py --expected-vendor cuda --expected-tile 512 > "$out/python-public.log" 2>&1
