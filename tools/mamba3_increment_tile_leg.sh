#!/bin/bash
# Run only in an isolated checkout and an exclusively assigned GPU slot.
# No opponent timing: reuse bench/OPPONENT_REFERENCE.md.
set -euo pipefail
repo=${MOJOLEARN_MAMBA3_REPO:-$(cd "$(dirname "$0")/.." && pwd)}
out=${MOJOLEARN_MAMBA3_RESULTS:?set an absolute evidence directory}
if [ "${M3_INCREMENT_ENV_ACTIVE:-0}" != 1 ]; then
  export M3_INCREMENT_ENV_ACTIVE=1
  exec "${MOJOLEARN_PIXI:-pixi}" run --manifest-path "$repo/pixi.toml" bash "$0"
fi
cd "$repo"
python=${MOJOLEARN_MAMBA3_PYTHON:-python3}
mkdir -p "$out"
# The rental archive intentionally omits the corpus; stage its generator and
# the three surface smoke cases before starting this leg.
test -f mamba/corpus/gen_corpus.py
test -d mamba/corpus/base_b2_l4_d8
test -d mamba/corpus/mamba2/m2_base_b2_l4_d32
test -d mamba/corpus/mamba3/m3_base_b2_l4_d32
mojo --version > "$out/compiler.txt" 2>&1
"$python" -c 'import sys,numpy; print(sys.version,numpy.__version__,sys.executable)' > "$out/python.txt"
sha256sum mamba/impl/mamba_ssm/ops/mamba3_siso.mojo gemm/checks/gemm_identical.mojo bindings/_mojolearn_mamba.mojo python/mojolearn/_mamba_impl.py > "$out/source-sha256.txt"
mojo build -D MOJOLEARN_NUMERIC_IDENTICAL=1 -I . mamba/checks/mamba3_increment_tile_check.mojo -o "$out/increment-check" > "$out/build-increment.log" 2>&1
"$out/increment-check" > "$out/increment.log" 2>&1
# Restore the exact archived public harness in the isolated checkout if absent.
if [ ! -f bench/speed/seq_py_speed_arm.py ]; then
  mkdir -p bench/speed
  cp bench/results/mamba3/2026-09-09-statepass/reproduction/seq_py_speed_arm.py bench/speed/
fi
if [ ! -f tools/speed_torch_seq.py ]; then
  cp bench/results/mamba3/2026-09-09-statepass/reproduction/speed_torch_seq.py tools/
fi
mkdir -p python/mojolearn/identical
export PYTHONPATH="$repo/python" MOJOLEARN_NUMERIC_MODE=identical
for arm in baseline tiled; do
  extra=()
  if [ "$arm" = tiled ]; then extra=(-D MOJOLEARN_MAMBA3_TILED_INCREMENT=1); fi
  mojo build -D MOJOLEARN_NUMERIC_IDENTICAL=1 "${extra[@]}" -I . mamba/checks/mamba3_check.mojo -o "$out/native-$arm" > "$out/build-native-$arm.log" 2>&1
  MOJOLEARN_IDENTITY_TRACE="$out/default-$arm.trace" "$out/native-$arm" > "$out/native-$arm.log" 2>&1
  MOJOLEARN_IDENTITY_TRACE="$out/long-$arm.trace" MOJOLEARN_MAMBA3_CHECK_B=2 MOJOLEARN_MAMBA3_CHECK_L=65 MOJOLEARN_MAMBA3_CHECK_DM=64 "$out/native-$arm" > "$out/native-long-$arm.log" 2>&1
  for gate in decode-cross continuation refusal; do "$out/native-$arm" "$gate" > "$out/native-$gate-$arm.log" 2>&1; done
  mojo build --emit shared-lib -D MOJOLEARN_NUMERIC_IDENTICAL=1 "${extra[@]}" -I . -I bindings bindings/_mojolearn_mamba.mojo -o "$out/$arm.so" > "$out/build-binding-$arm.log" 2>&1
  cp "$out/$arm.so" python/mojolearn/identical/_mojolearn_mamba.so
  MOJOLEARN_MAMBA3_FRESH_LARGE=1 "$python" tools/mamba3_fresh_prefill_check.py > "$out/fresh-$arm.log" 2>&1
  "$python" python/mojolearn/tests/test_mamba_surface.py > "$out/surface-$arm.log" 2>&1
  MOJOLEARN_SPEED_DUMP_DIR="$out/dump-$arm" "$python" bench/speed/seq_py_speed_arm.py --lane mamba3 --rounds 5 > "$out/price-$arm.log" 2>&1
done
cmp "$out/default-baseline.trace" "$out/default-tiled.trace"
cmp "$out/long-baseline.trace" "$out/long-tiled.trace"
"$python" - "$out" <<'PY'
from pathlib import Path
import hashlib,json,re,statistics,sys
out=Path(sys.argv[1]); rows=[]; timings={}
for f in sorted((out/'dump-baseline').glob('*.bin')):
    a=hashlib.sha256(f.read_bytes()).hexdigest(); b=hashlib.sha256((out/'dump-tiled'/f.name).read_bytes()).hexdigest()
    rows.append(dict(file=f.name,bytes=f.stat().st_size,baseline_sha256=a,tiled_sha256=b,equal=a==b))
assert len(rows)==3 and all(r['equal'] for r in rows)
for arm in ['baseline','tiled']:
    samples={}
    for shape,ms in re.findall(r'^FSPEED lane=mamba3 arm=ours shape=(\S+) round=\d+ ms=(\S+)',(out/f'price-{arm}.log').read_text(),re.M): samples.setdefault(shape,[]).append(float(ms))
    timings[arm]={k:dict(samples_ms=v,median_ms=statistics.median(v)) for k,v in samples.items()}
(out/'full-output-identity.json').write_text(json.dumps(rows,indent=2)+'\n')
(out/'timings.json').write_text(json.dumps(timings,indent=2)+'\n')
PY
