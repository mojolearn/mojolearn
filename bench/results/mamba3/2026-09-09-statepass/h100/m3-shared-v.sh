#!/bin/bash
set -euo pipefail
export PATH="$HOME/.pixi/bin:$PATH"
cd /root/mamba3
export MOJOLEARN_NUMERIC_MODE=identical PYTHONPATH=/root/mamba3/python
out=/root/jobs/m3-shared-v
mkdir -p "$out"
flags=(-D MOJOLEARN_NUMERIC_IDENTICAL=1 -D MOJOLEARN_MAMBA3_DIRECT_TRANSFER=1 -D MOJOLEARN_MAMBA3_TILED_YSTATE=1 -D MOJOLEARN_MAMBA3_TILED_QKS=1 -D MOJOLEARN_MAMBA3_SHARED_INCREMENT_V=1)
pixi run mojo run "${flags[@]}" -I . mamba/checks/mamba3_fold_check.mojo > "$out/fold-gate.log" 2>&1
pixi run mojo build "${flags[@]}" -I . mamba/checks/mamba3_check.mojo -o "$out/check" > "$out/build-check.log" 2>&1
for gate in default decode-cross continuation refusal; do
 if [ "$gate" = default ]; then "$out/check" > "$out/$gate.log" 2>&1; else "$out/check" "$gate" > "$out/$gate.log" 2>&1; fi
done
MOJOLEARN_MAMBA3_CHECK_B=2 MOJOLEARN_MAMBA3_CHECK_L=65 MOJOLEARN_MAMBA3_CHECK_DM=64 "$out/check" > "$out/l65.log" 2>&1
pixi run mojo build -j 2 --emit shared-lib --target-cpu x86-64-v3 "${flags[@]}" -D MOJOLEARN_MAMBA3_PHASE_TIMERS=1 -I . -I bindings bindings/_mojolearn_mamba.mojo -o "$out/profile.so" > "$out/build-profile.log" 2>&1
cp "$out/profile.so" python/mojolearn/identical/_mojolearn_mamba.so
python3 bench/speed/seq_py_speed_arm.py --lane mamba3 --rounds 1 > "$out/profile.log" 2>&1
pixi run mojo build -j 2 --emit shared-lib --target-cpu x86-64-v3 "${flags[@]}" -I . -I bindings bindings/_mojolearn_mamba.mojo -o "$out/optimized.so" > "$out/build-optimized.log" 2>&1
cp "$out/optimized.so" python/mojolearn/identical/_mojolearn_mamba.so
MOJOLEARN_SPEED_DUMP_DIR="$out/dump-optimized" python3 bench/speed/seq_py_speed_arm.py --lane mamba3 --rounds 5 > "$out/price-optimized.log" 2>&1
python3 python/mojolearn/tests/test_mamba_surface.py > "$out/surface.log" 2>&1
python3 - <<'PY'
import hashlib,pathlib,json
out=pathlib.Path('/root/jobs/m3-shared-v');rows=[]
for f in sorted(pathlib.Path('/root/jobs/m3-h100/dump-baseline').glob('*.bin')):
 a=hashlib.sha256(f.read_bytes()).hexdigest();other=out/'dump-optimized'/f.name;b=hashlib.sha256(other.read_bytes()).hexdigest()
 rows.append(dict(file=f.name,bytes=f.stat().st_size,baseline_sha256=a,optimized_sha256=b,equal=a==b))
assert len(rows)==3 and all(r['equal'] for r in rows),rows
(out/'full-output-identity.json').write_text(json.dumps(rows,indent=2)+'\n')
PY
