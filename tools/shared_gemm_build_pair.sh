#!/bin/bash
# M2 only, compile-only static downstream pair. Never SSH/import/execute models.
set -euo pipefail
unset PYTHONOPTIMIZE
cd "$(dirname "$0")/.."
source_sha=${1:?SOURCE_SHA required}
family=${2:?core or estimators required}
[[ $source_sha =~ ^[0-9a-f]{40}$ ]] || exit 2
[[ $(uname -s) == Darwin && $(uname -m) == arm64 ]] || exit 2
[[ $(sysctl -n machdep.cpu.brand_string) == *"Apple M2"* ]] || { echo 'M2 compile-only machine required' >&2; exit 2; }
if [[ ${MOJOLEARN_BUILD_LOCK_HELD:-} != 1 ]]; then
    exec bash tools/with_build_lock.sh bash tools/shared_gemm_build_pair.sh "$source_sha" "$family"
fi
[[ $(git rev-parse HEAD) == "$source_sha" ]] || { echo 'source HEAD mismatch' >&2; exit 2; }
git diff --quiet HEAD
[[ $(df -k . | tail -1 | awk '{print $4}') -gt 8388608 ]] || { echo 'M2 below 8 GiB free' >&2; exit 2; }
case "$family" in
    core) build=bindings/build.sh; artifact=_mojolearn.so ;;
    estimators) build=bindings/build_estimators.sh; artifact=_mojolearn_estimators.so ;;
    *) echo 'Unsupported family' >&2; exit 2 ;;
esac
variant=$(/usr/bin/python3 -c 'import json; c=json.load(open("tools/shared_gemm_variant.json")); assert c["contract"]=="shared-gemm-static-downstream-v1" and c["variant"] in (1,5); print(c["variant"])')
defines_a='-D MOJOLEARN_APPLE_FAST_SHARED_GEMM_COUNTERS'
defines_b="$defines_a -D MOJOLEARN_APPLE_FAST_SHARED_GEMM_G$variant"
/usr/bin/python3 - "$defines_a" "$defines_b" <<'PY'
import json,sys
c=json.load(open('tools/shared_gemm_variant.json'))
assert [c['defines_A'],c['defines_B']]==sys.argv[1:]
PY
output_root=$HOME/m2-arms/$source_sha
mkdir -p "$output_root"
out=$output_root/$family
# A partial failure stays visible; retry requires a fresh source/tag or an
# explicitly reviewed cleanup. Never overwrite an existing manifest/arm.
mkdir "$out"
export MOJOLEARN_NUMERIC_MODE=fast MOJOLEARN_SKIP_BUILD_GATE=1 MOJOLEARN_COMPILE_JOBS=1
export MOJOLEARN_GPU_ARCHS=metal:1 MOJOLEARN_TARGET_COLUMN=apple
export MOJOLEARN_MOJO_BUILD_FLAGS=''
unset MACOSX_DEPLOYMENT_TARGET PYTHONOPTIMIZE
for arm in A B; do
    export MOJOLEARN_BUILD_EXTRA_DEFINES=$defines_a
    [[ $arm != B ]] || export MOJOLEARN_BUILD_EXTRA_DEFINES=$defines_b
    bash "$build" > "$out/$arm.build.log" 2>&1 || { tail -64 "$out/$arm.build.log"; exit 1; }
    cp "python/mojolearn/$artifact" "$out/$arm.so"
    nm -gU "$out/$arm.so" > "$out/$arm.exports.txt"
    symbol=_PyInit_${artifact%.so}
    awk -v symbol="$symbol" '$NF == symbol {found=1} END {exit !found}' "$out/$arm.exports.txt"
done
/usr/bin/python3 - "$out" "$source_sha" "$family" "$artifact" <<'PY'
import hashlib,json,sys
from pathlib import Path
out=Path(sys.argv[1]); config=Path('tools/shared_gemm_variant.json')
c=json.loads(config.read_text())
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
m=dict(c,source_sha=sys.argv[2],binding=sys.argv[3],artifact=sys.argv[4],numeric_mode='fast',builder='m2',
       target_cpu='apple-m1',target_accelerator='metal:1',target_column='apple',compile_jobs=1,
       compile_only=True,contract_sha=sha(config),hashes={a:sha(out/(a+'.so')) for a in ('A','B')})
with (out/'manifest.json').open('x') as f: json.dump(m,f,indent=2); f.write('\n')
print('ARMS_READY',out)
PY
