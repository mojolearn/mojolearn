#!/bin/bash
# Compile-only prerequisite from an EXACT existing source checkout on M2.
# Usage: bash /path/to/repaired/tools/shared_gemm_build_ibase.sh CHECKOUT SOURCE
set -euo pipefail
unset PYTHONOPTIMIZE
builder_path=$(cd "$(dirname "$0")" && pwd)/$(basename "$0")
checkout=${1:?exact source checkout required}
source_sha=${2:?40-character compiled source required}
cd "$checkout"
[[ $source_sha =~ ^[0-9a-f]{40}$ && $(git rev-parse HEAD) == "$source_sha" ]] || exit 2
git diff --quiet HEAD
[[ $(uname -s) == Darwin && $(uname -m) == arm64 ]] || exit 2
[[ $(sysctl -n machdep.cpu.brand_string) == *"Apple M2"* ]] || { echo 'M2 compile-only machine required' >&2; exit 2; }
if [[ ${MOJOLEARN_BUILD_LOCK_HELD:-} != 1 ]]; then
    exec bash tools/with_build_lock.sh bash "$builder_path" "$PWD" "$source_sha"
fi
[[ $(df -k . | tail -1 | awk '{print $4}') -gt 8388608 ]] || { echo 'M2 below 8 GiB free' >&2; exit 2; }
mkdir -p "$HOME/m2-arms/$source_sha"
out=$HOME/m2-arms/$source_sha/ibase
mkdir "$out"
export MOJOLEARN_NUMERIC_MODE=identical MOJOLEARN_SKIP_BUILD_GATE=1 MOJOLEARN_COMPILE_JOBS=1
export MOJOLEARN_GPU_ARCHS=metal:1 MOJOLEARN_TARGET_COLUMN=apple
export MOJOLEARN_MOJO_BUILD_FLAGS='' MOJOLEARN_BUILD_EXTRA_DEFINES=''
unset MACOSX_DEPLOYMENT_TARGET
bash bindings/build.sh > "$out/build.log" 2>&1 || { tail -64 "$out/build.log"; exit 1; }
cp python/mojolearn/identical/_mojolearn.so "$out/_mojolearn.so"
nm -gU "$out/_mojolearn.so" > "$out/exports.txt"
awk '$NF == "_PyInit__mojolearn" {found=1} END {exit !found}' "$out/exports.txt"
/usr/bin/python3 - "$out" "$source_sha" "$builder_path" <<'PY'
import hashlib,json,sys
from pathlib import Path
out=Path(sys.argv[1])
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
m=dict(contract='shared-gemm-prerequisite-v1',source_sha=sys.argv[2],binding='ibase',module='_mojolearn',
       artifact='_mojolearn.so',install_path='identical/_mojolearn.so',numeric_mode='identical',defines='',
       builder='m2',compile_only=True,target_cpu='apple-m1',target_accelerator='metal:1',target_column='apple',
       sha256=sha(out/'_mojolearn.so'),required_exports=['all_finite_f32'],tooling_sha=sha(sys.argv[3]))
with (out/'manifest.json').open('x') as f: json.dump(m,f,indent=2); f.write('\n')
print('PREREQUISITE_READY',out)
PY
