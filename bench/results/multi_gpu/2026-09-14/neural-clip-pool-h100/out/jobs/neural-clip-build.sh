#!/bin/sh
set -eu
export PATH="$HOME/.pixi/bin:$PATH"
export RUNPOD_POD_ID=smqlvlvt7exixd MOJOLEARN_NUMERIC_MODE=identical
export MOJOLEARN_GPU_ARCHS=sm_90 MOJOLEARN_TARGET_COLUMN=nvidia MOJOLEARN_SKIP_BUILD_GATE=1
mkdir -p /root/neural-clip-out /root/neural-clip-production
cd /root/mojolearn
tar xzf /root/neural-clip-source.tgz
cp /root/neural-clip-source.tgz /root/neural-clip-out/source.tgz
sha256sum /root/neural-clip-source.tgz > /root/neural-clip-out/source.sha256
python3 - <<'PY'
from pathlib import Path
s=Path('bindings/build_training.sh').read_text().replace('OUTDIR="python/mojolearn/identical"','OUTDIR="/root/neural-clip-production"')
Path('bindings/build_training_clip.sh').write_text(s)
Path('/root/neural-clip-out/build-script.sh').write_text(s)
PY
sh bindings/build_training_clip.sh > /root/neural-clip-out/build.log 2>&1
sha256sum /root/neural-clip-production/_mojolearn_training.so > /root/neural-clip-out/production.sha256
