#!/usr/bin/env bash
# Experimental CPU build body; invoked only through nvidia_baseline_build.py.
# Outputs retain read-backs and final hashes, never a native release proof.
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT"
[[ $(uname -s) == Linux && $# == 4 ]] || { echo 'Linux CPU build: SHA ORIGIN SECONDS JOBS required' >&2; exit 2; }
commit=$1 origin=$2 seconds=$3 jobs=$4
[[ "$commit" =~ ^[0-9a-f]{40}$ && "$seconds" =~ ^[0-9]+$ ]] || exit 2
((seconds >= 120 && seconds <= 6000)) || exit 2
[[ "$jobs" = auto || "$jobs" =~ ^([1-9]|1[0-6])$ ]] || exit 2
OUT=${LEG_OUT:?existing leased runner output directory required}/ptx-baseline
[[ ! -e "$OUT" ]] || { echo 'Experimental output already exists' >&2; exit 2; }
mkdir -p "$OUT"
PY="$ROOT/.pixi/envs/default/bin/python"
[[ -x "$PY" ]] || { echo 'Locked default Pixi environment is missing' >&2; exit 2; }
export PATH="$ROOT/.pixi/envs/default/bin:/root/.pixi/bin:$PATH"
"$PY" - "$origin" <<'PY'
import sys
sys.path.insert(0, 'tools')
from nvidia_baseline_build import transport_origin
from cpu_build_guard import visible_devices
assert transport_origin(sys.argv[1]) == sys.argv[1]
assert not visible_devices(), 'Refusing GPU-visible CPU build'
PY

# The transport tar is not a Git repository. Fetch original objects, never
# synthesize a commit label. Sparse checkout omits historical run data only;
# all compiler/runtime source remains the advertised frozen source.
if [[ ! -e .git ]]; then
  git init -q
  git remote add origin "$origin"
  git sparse-checkout init --no-cone
  git sparse-checkout set --no-cone '/*' '!/bench/results/' '!/mamba/corpus/' '!/bench/oracle_*' '!/bench/minentropy_oracle.txt'
  timeout -k 10 300 git fetch --filter=blob:none --depth=1 origin "$commit" > "$OUT/source-fetch.log" 2>&1
  timeout -k 10 180 git checkout --force --detach FETCH_HEAD >> "$OUT/source-fetch.log" 2>&1
fi
[[ $(git rev-parse HEAD) == "$commit" ]] || { echo 'Frozen checkout differs' >&2; exit 2; }
git diff --quiet HEAD --
. /etc/os-release
[[ "$VERSION_ID" == 22.04 && $(gcc -dumpfullversion) == 11.4.0 ]] || { echo 'Pinned Ubuntu/GCC image differs' >&2; exit 2; }
[[ $(ld --version | head -1) == 'GNU ld (GNU Binutils for Ubuntu) 2.38' ]] || exit 2
"$PY" -m venv "$OUT/tools"
timeout -k 10 180 "$OUT/tools/bin/python" -m pip install -q --disable-pip-version-check --only-binary=:all: \
  --retries 2 --timeout 30 'patchelf==0.17.2.4' > "$OUT/tools-setup.log" 2>&1
export PATH="$OUT/tools/bin:$PATH"
[[ $(patchelf --version) == 'patchelf 0.17.2' ]] || exit 2
"$PY" tools/build_sizing.py --jobs "$jobs" --shell > "$OUT/sizing.env"
. "$OUT/sizing.env"
export MOJOLEARN_BUILD_JOBS="$BUILD_JOBS" MOJOLEARN_COMMIT="$commit"
export MOJOLEARN_BUILD_PIXI_ENV=default MOJOLEARN_PACKAGE_BYTE_LM=1
export MOJOLEARN_GPU_ARCHS=sm_80 MOJOLEARN_TARGET_COLUMN=nvidia MOJOLEARN_LINUX_CPU=x86-64-v3
export MOJOLEARN_CUDA_CODE_FORMAT=ptx-baseline MOJOLEARN_BUILD_TIERS='fast deterministic identical'
export MOJOLEARN_BINCACHE=0
unset MOJOLEARN_NUMERIC_MODE MOJOLEARN_GPU_ARCH MOJOLEARN_VENDOR PYTHONPATH PYTHONHOME
"$PY" tools/cpu_build_guard.py --seconds "$seconds" --rss-gib "$BUILD_RSS_GIB" --cores "$((BUILD_JOBS * 2))" -- \
  bash packaging/linux/build_sets.sh "$OUT/build" > "$OUT/build.log" 2>&1

# The existing CPU runner built this helper from the frozen tar before its
# command. Rebuild it after checkout so the retained bytes have this source.
PYTHONPATH="$ROOT/packaging/portable_math" "$PY" - "$OUT/libMojolearnMath.so" <<'PY'
import pathlib, stage, sys
stage.build(pathlib.Path(sys.argv[1]))
PY
"$PY" - "$OUT" "$commit" "$seconds" <<'PY'
import hashlib, json, pathlib, subprocess, sys
out=pathlib.Path(sys.argv[1]); commit=sys.argv[2]
root=out/'build/sets/cuda/sm_80'
manifest=root/'PTX_BASELINE.json'
doc=json.loads(manifest.read_text())
assert doc['source_commit']==commit and doc['source_dirty'] is False and not doc['errors']
assert doc['code_format']=='ptx-baseline' and doc['identical_qualified'] is False
sha=lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
for row in doc['files']:
    assert sha(root/row['file'])==row['sha256']
assert (root/'readback.txt').is_file() and (root/'manifest.json').is_file()
record=dict(schema='mojolearn.experimental-ptx-build.v1', source_commit=commit,
    source_tree=subprocess.check_output(['git','rev-parse','HEAD^{tree}'],text=True).strip(),
    source_dirty=False, code_format='ptx-baseline', architecture='sm_80',
    build_seconds_limit=int(sys.argv[3]), manifest_sha256=sha(manifest),
    readback_sha256=sha(root/'readback.txt'), runtime_manifest_sha256=sha(root/'manifest.json'),
    portable_math_sha256=sha(out/'libMojolearnMath.so'), pixi_lock_sha256=sha(pathlib.Path('pixi.lock')),
    experimental=True, identical_qualified=False, release_qualified=False)
(out/'experimental-build.json').write_text(json.dumps(record,indent=2,sort_keys=True)+'\n')
print(json.dumps(record,sort_keys=True))
PY
