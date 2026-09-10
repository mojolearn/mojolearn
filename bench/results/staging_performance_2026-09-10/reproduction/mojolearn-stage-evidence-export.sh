#!/bin/bash
set -euo pipefail
cd /root/evidence
python3 - <<'PY'
from pathlib import Path
import tarfile
root=Path('/root/evidence')
with tarfile.open('/root/staging-evidence.tar.gz','w:gz') as tar:
 for lane in ['gemm-stage','knn-stage','mamba-stage','final-stage','knn-batch','knn-final','mamba-repeat']:
  p=root/lane
  if not p.exists():continue
  for f in sorted(p.rglob('*')):
   if not f.is_file():continue
   # Keep compact logs, hashes, arrays and exact kNN outputs; omit binaries
   # and large Mamba/transformer tensors whose full SHA256 is in summaries.
   if f.suffix not in ['.log','.json','.txt','.trace','.cells','.bin']:continue
   if f.suffix=='.bin' and not lane.startswith('knn'):continue
   tar.add(f,arcname=str(f.relative_to(root)))
 for f in sorted(Path('/root/jobs').glob('*')):
  if f.suffix in ['.log','.rc','.done','.sh','.txt']:tar.add(f,arcname='jobs/'+f.name)
PY
