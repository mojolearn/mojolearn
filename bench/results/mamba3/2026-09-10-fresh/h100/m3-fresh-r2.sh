#!/bin/bash
set -euo pipefail
if [ "${M3_ENV_ACTIVE:-0}" != 1 ]; then
 export M3_ENV_ACTIVE=1
 exec /root/.pixi/bin/pixi run --manifest-path /root/mojolearn/pixi.toml bash "$0"
fi
cd /root/mamba3-residual
export PYTHONPATH=/root/mamba3-residual/python MOJOLEARN_NUMERIC_MODE=identical
mojo_bin=/root/mojolearn/.pixi/envs/default/bin/mojo
out=/root/jobs/m3-fresh
mkdir -p "$out" python/mojolearn/identical
flags=(-D MOJOLEARN_NUMERIC_IDENTICAL=1 -D MOJOLEARN_GEMM_FULL_FOLD_STACK=1)
"$mojo_bin" --version > "$out/compiler.txt" 2>&1
nvidia-smi --query-gpu=name,driver_version --format=csv > "$out/gpu.txt"
sha256sum bindings/_mojolearn_mamba.mojo python/mojolearn/_mamba_impl.py gemm/checks/gemm_identical.mojo checks/kernel_matrix.mojo bench/speed/seq_py_speed_arm.py tools/speed_torch_seq.py > "$out/source-sha256.txt"
"$mojo_bin" build "${flags[@]}" -I . mamba/checks/mamba3_check.mojo -o "$out/check" > "$out/build-native.log" 2>&1
for gate in default decode-cross continuation refusal; do
 if [ "$gate" = default ]; then "$out/check" > "$out/native-$gate.log" 2>&1; else "$out/check" "$gate" > "$out/native-$gate.log" 2>&1; fi
done
MOJOLEARN_MAMBA3_CHECK_B=2 MOJOLEARN_MAMBA3_CHECK_L=65 MOJOLEARN_MAMBA3_CHECK_DM=64 "$out/check" > "$out/native-long.log" 2>&1
for arm in baseline fresh caller; do
 extra=()
 if [ "$arm" != baseline ]; then extra+=(-D MOJOLEARN_MAMBA3_FRESH_PREFILL=1); fi
 if [ "$arm" = caller ]; then extra+=(-D MOJOLEARN_MAMBA3_CALLER_TRANSFER=1); fi
 "$mojo_bin" build -j 2 --emit shared-lib --target-cpu x86-64-v3 "${flags[@]}" "${extra[@]}" -I . -I bindings bindings/_mojolearn_mamba.mojo -o "$out/$arm.so" > "$out/build-$arm.log" 2>&1
 cp "$out/$arm.so" python/mojolearn/identical/_mojolearn_mamba.so
 if [ "$arm" != baseline ]; then MOJOLEARN_MAMBA3_FRESH_LARGE=1 python3 tools/mamba3_fresh_prefill_check.py > "$out/fresh-$arm.log" 2>&1; fi
 python3 python/mojolearn/tests/test_mamba_surface.py > "$out/surface-$arm.log" 2>&1
 MOJOLEARN_SPEED_DUMP_DIR="$out/dump-$arm" python3 bench/speed/seq_py_speed_arm.py --lane mamba3 --rounds 5 > "$out/price-$arm.log" 2>&1
done
python3 - <<'PY'
from pathlib import Path
import hashlib,json,re,statistics
out=Path('/root/jobs/m3-fresh');identities=[];timings={}
for arm in ['baseline','fresh','caller']:
 rows={}
 for shape,ms in re.findall(r'^FSPEED lane=mamba3 arm=ours shape=(\S+) round=\d+ ms=(\S+)',(out/f'price-{arm}.log').read_text(),re.M):rows.setdefault(shape,[]).append(float(ms))
 timings[arm]={k:dict(samples_ms=v,median_ms=statistics.median(v)) for k,v in rows.items()}
 for f in sorted((out/'dump-baseline').glob('*.bin')):
  base=hashlib.sha256(f.read_bytes()).hexdigest();candidate=out/f'dump-{arm}'/f.name;actual=hashlib.sha256(candidate.read_bytes()).hexdigest()
  identities.append(dict(arm=arm,file=f.name,bytes=f.stat().st_size,baseline_sha256=base,actual_sha256=actual,equal=base==actual))
assert len(identities)==9 and all(r['equal'] for r in identities),identities
(out/'full-output-identity.json').write_text(json.dumps(identities,indent=2)+'\n')
(out/'timings.json').write_text(json.dumps(timings,indent=2)+'\n')
PY
