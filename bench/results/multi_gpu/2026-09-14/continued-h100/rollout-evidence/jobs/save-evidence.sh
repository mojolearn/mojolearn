#!/bin/sh
set -eu
cd /root/mojolearn
mkdir -p /root/rollout-evidence/jobs /root/rollout-evidence/source
cp -a /root/gram-out /root/holt-out /root/logistic-out /root/solver-out /root/rollout-evidence/
cp /root/jobs/*.rc /root/jobs/*.sh /root/rollout-evidence/jobs/
cp commit.txt /root/rollout-evidence/base-commit.txt
cp /root/*overlay.tgz /root/continued-source.tgz /root/rollout-evidence/source/
sha256sum /root/rollout-evidence/source/*.tgz > /root/rollout-evidence/source/archives.sha256
python3 - <<'PY'
from pathlib import Path
import hashlib
root=Path('.')
paths=sorted(p for p in root.rglob('*') if p.is_file() and p.suffix in ('.mojo','.py','.sh','.toml','.lock') and '.pixi' not in p.parts and '__pycache__' not in p.parts)
Path('/root/rollout-evidence/source/final-source.sha256').write_text(''.join(hashlib.sha256(p.read_bytes()).hexdigest()+'  '+str(p)+'\n' for p in paths))
PY
cat /root/jobs/solver-next.rc /root/jobs/dot-gate.rc
