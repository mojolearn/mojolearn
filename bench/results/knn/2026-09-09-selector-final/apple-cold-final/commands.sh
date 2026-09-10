#!/bin/bash
set -euo pipefail
cd /Users/andrewhendel/CascadeProjects/mojolearn
out=/tmp/mojolearn-knn-cold-final-apple
mkdir -p "$out"
run_mojo() { local tag=$1; shift; pixi run mojo run -D MOJOLEARN_NUMERIC_IDENTICAL=1 -I . "$@" > "$out/$tag.log" 2>&1; }
run_mojo distance bench/knn_index_layout_main.mojo
MOJOLEARN_KNN_CHECKS_ONLY=1 MOJOLEARN_LAYOUT_PRICE_OUT="$out/layout" bash tools/knn_layout_dispatch_price.sh
python3 - "$out/layout" <<'CHECK'
import sys, pathlib, hashlib
for p in sorted(pathlib.Path(sys.argv[1]).glob('check-*.log')):
 lines=sorted(line for line in p.read_bytes().splitlines(keepends=True) if b'_CELL' in line)
 assert len(lines)==143628,(p,len(lines))
 h=hashlib.sha256(b''.join(lines)).hexdigest()
 assert h=='49c0f02513c2722db1855acd01d6d02941ccd29f4fb1b27d59a533db6b7350ce',(p,h)
 print(p.name,'PASS',len(lines),h)
CHECK
run_mojo identity neighbors/checks/knn_identity_check.mojo
run_mojo main neighbors/knn_main.mojo
MOJOLEARN_IDENTITY_TRACE="$out/knn.card" MOJOLEARN_UNSUP_ARM=knn run_mojo card bench/unsupervised_trace_main.mojo
python3 tools/identity_trace_diff.py bench/results/e1/2026-08-28_122543-runpod-nvidia/e1u/knn.card "$out/knn.card" > "$out/card-diff.log" 2>&1
run_mojo umap-graph umap/checks/graph_check.mojo
run_mojo umap-identity umap/checks/identity_check.mojo
run_mojo umap-broader umap/checks/identity_broader_check.mojo
pixi run mojo build -D MOJOLEARN_NUMERIC_IDENTICAL=1 -I . bench/umap_phase_price_main.mojo -o "$out/umap-phase" > "$out/build-umap.log" 2>&1
MOJOLEARN_UMAP_ROWS=20000 "$out/umap-phase" > "$out/umap-20k.log" 2>&1
python3 - "$out" <<'CHECK'
from pathlib import Path
import sys
assert '12938647291752780014' in (Path(sys.argv[1])/'umap-20k.log').read_text()
print('Final cold-helper kNN Apple gates PASS')
CHECK
