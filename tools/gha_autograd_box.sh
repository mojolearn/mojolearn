#!/usr/bin/env bash
# One manual GitHub CPU build job, inside the pinned release ROCm image.
# Source is an exact clean git checkout at /root/mojolearn; tooling is mounted
# read-only at /tooling. Never overlays source or executes the built checks.
# Keep logs out of context: complete output is retained under /root/results-autograd.
set -euo pipefail
VENDOR=${1:?nvidia or amd}
TARGET=${2:?sm_89 or gfx942}
JOBS=${3:?compiler workers}
COMPONENT=${4:-autograd}
case "$COMPONENT" in autograd|adapter-native) ;; *) echo 'unsupported component'; exit 2 ;; esac
R=/root/mojolearn
OUT=/root/results-autograd
G=$OUT/GHA
PIXIVER=0.77.0
case "$VENDOR:$TARGET" in nvidia:sm_89|amd:gfx942) ;; *) echo 'unsupported vendor/target' >&2; exit 2 ;; esac
[[ "$JOBS" =~ ^[1-9][0-9]?$ && "$JOBS" -le 16 ]] || { echo 'compiler workers must be 1..16' >&2; exit 2; }
[[ "${MOJOLEARN_COMMIT:-}" =~ ^[0-9a-f]{40}$ ]] || { echo 'MOJOLEARN_COMMIT must be a full SHA' >&2; exit 2; }
[[ $(uname -s) = Linux && $(uname -m) = x86_64 ]] || { echo 'requires Linux x86-64 CPU runner' >&2; exit 2; }
mkdir -p "$G"
[[ ! -e "$G/controller.log" && ! -e "$OUT/artifacts" ]] || { echo 'use a fresh output directory' >&2; exit 2; }
exec > "$G/controller.log" 2>&1
PHASE=setup
finish() {
    local code=$?
    trap - EXIT
    # Kernel counters cover the whole container job (setup plus serial builds).
    # Read once at exit; never start a background monitoring service. Missing
    # counters stay unavailable rather than being presented as zero memory use.
    {
        printf 'scope=container-job\n'
        for field in memory.peak memory.swap.peak memory.max memory.swap.max; do
            value=unavailable
            [[ ! -r /sys/fs/cgroup/$field ]] || value=$(cat "/sys/fs/cgroup/$field") || value=unavailable
            printf '%s=%s\n' "$field" "$value"
        done
        if [[ -r /sys/fs/cgroup/memory/memory.max_usage_in_bytes ]]; then
            printf 'cgroup_v1_memory_peak_bytes=%s\n' "$(cat /sys/fs/cgroup/memory/memory.max_usage_in_bytes)"
        fi
        cat /proc/self/cgroup || true
    } > "$G/memory.txt"
    printf 'exit_code=%s\nphase=%s\nvendor=%s\ntarget=%s\nruntime_qualified=false\n' "$code" "$PHASE" "$VENDOR" "$TARGET" > "$G/status.txt"
    exit "$code"
}
trap finish EXIT
trap 'exit 124' TERM INT
{ uname -a; nproc; head -4 /etc/os-release; free -g; df -h /root
  printf 'runner=%s\nrun=%s\njob=%s\nimage=%s\n' "${GHA_RUNNER:-}" "${GHA_RUN:-}" "${GHA_JOB:-}" "${GHA_IMAGE:-}"
} > "$G/box.txt"
need=()
command -v curl >/dev/null || need+=(curl ca-certificates)
command -v git >/dev/null || need+=(git)
if (( ${#need[@]} )); then
    export DEBIAN_FRONTEND=noninteractive
    apt-get -o Acquire::Retries=3 update -qq > "$G/apt.log" 2>&1
    apt-get -o Acquire::Retries=3 install -y -qq --no-install-recommends "${need[@]}" >> "$G/apt.log" 2>&1
fi
git config --global --add safe.directory "$R"
cd "$R"
[[ $(git rev-parse HEAD) = "$MOJOLEARN_COMMIT" && -z $(git status --porcelain --untracked-files=no) ]] || { echo 'requires exact clean source checkout'; exit 3; }
{
    printf 'source_commit=%s\nsource_tree=%s\n' "$MOJOLEARN_COMMIT" "$(git rev-parse 'HEAD^{tree}')"
    sha256sum tools/gate_autograd_artifacts.py tools/autograd_smoke.py /tooling/tools/gha_autograd_box.sh pixi.lock pixi.toml
} > "$G/source.txt"
cp pixi.lock pixi.toml "$G/"
export PIXI_HOME=/root/.pixi PIXI_NO_PATH_UPDATE=1
PIXI=$PIXI_HOME/bin/pixi
for attempt in 1 2 3; do
    printf 'installer attempt=%s\n' "$attempt" >> "$G/pixi-install.log"
    [[ -x "$PIXI" ]] && break
    curl -fsSL --max-time 300 https://pixi.sh/install.sh | PIXI_VERSION="$PIXIVER" bash >> "$G/pixi-install.log" 2>&1 || true
    [[ -x "$PIXI" ]] || sleep 10
done
[[ -x "$PIXI" ]] || { echo 'pixi installation failed'; exit 3; }
export PATH="$R/.pixi/envs/default/bin:$PIXI_HOME/bin:$PATH"
ready=0
for attempt in 1 2 3; do
    printf 'environment attempt=%s\n' "$attempt" >> "$G/pixi-default.log"
    if "$PIXI" install --locked -e default >> "$G/pixi-default.log" 2>&1; then ready=1; break; fi
    sleep 15
done
[[ "$ready" = 1 ]] || { echo 'locked default environment installation failed'; exit 3; }
MOJO=$R/.pixi/envs/default/bin/mojo
PYTHON=$R/.pixi/envs/default/bin/python3
{ "$PIXI" --version; "$MOJO" --version; sha256sum "$MOJO"; } > "$G/compiler.txt" 2>&1
if [[ -d "$R/.pixi/envs/default/conda-meta" ]]; then
    tar -czf "$G/conda-meta.tar.gz" -C "$R/.pixi/envs/default" conda-meta
fi
find "$R/.pixi/envs/default/lib/mojo" -maxdepth 1 -type f \( -name '*.mojoc' -o -name '*.mojopkg' \) -exec sha256sum {} + > "$G/compiler-packages.sha256"
# Every artifact is invalidated for this source freeze. The producer's existing
# supported commands build all eight sequentially, with the vendor's compiler
# worker policy (-j is compiler workers inside ONE compile, not artifact
# concurrency). AMD -j8 is the existing supported policy after the reproduced
# one-worker compiler crash; capped runner swap supports peaks over RAM.
# No retained-artifact substitution and no GPU execution.
PHASE=compile
if [[ "$COMPONENT" = adapter-native ]]; then
    # Retain the actual transitive source bytes before compilation. Only this
    # native library changed; the eight autograd executables remain reusable.
    "$PYTHON" - "$OUT" <<'PY'
import hashlib,json,pathlib,sys,tarfile
sys.path.insert(0,'tools')
import retained_release_composition as retained
root=pathlib.Path.cwd(); out=pathlib.Path(sys.argv[1])
inputs=retained.adapter_inputs(root)
(out/'adapter-inputs.json').write_text(json.dumps(inputs,sort_keys=True,indent=2)+'\n')
with tarfile.open(out/'adapter-source-closure.tar.gz','w:gz') as archive:
    for name in inputs:
        archive.add(root/name,arcname=name)
    for name in ('pixi.toml','pixi.lock','tools/build_native_adapters.sh'):
        archive.add(root/name,arcname=name)
PY
    MOJOLEARN_COMPILE_JOBS="$JOBS" "$PIXI" run -e default bash tools/build_native_adapters.sh "$OUT/artifacts" "$TARGET"
    PHASE=validate
    "$PYTHON" - "$OUT" "$VENDOR" "$TARGET" "$MOJOLEARN_COMMIT" <<'PY'
import hashlib,json,pathlib,sys
sys.path.insert(0,'tools')
import retained_release_composition as retained
out=pathlib.Path(sys.argv[1]); artifact=out/'artifacts/libmojolearn_adapter.so'
assert json.loads((out/'adapter-inputs.json').read_text()) == retained.adapter_inputs(pathlib.Path.cwd())
result=dict(component='adapter-native',source_commit=sys.argv[4],vendor=sys.argv[2],target=sys.argv[3],
            state='built',runtime_qualified=False,sha256=hashlib.sha256(artifact.read_bytes()).hexdigest())
(out/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
PY
    exit 0
fi
# Pixi activation supplies MODULAR_HOME from the pinned package's activation
# script. Calling its Python directly omitted that supported environment and
# both vendors refused before compiling because Mojo could not locate std.
"$PIXI" run -e default "$PYTHON" -c 'import os; print("MODULAR_HOME=" + os.environ.get("MODULAR_HOME", ""))' > "$G/activation.txt"
"$PIXI" run -e default "$PYTHON" tools/gate_autograd_artifacts.py build --vendor "$VENDOR" --source-commit "$MOJOLEARN_COMMIT" \
    --target "$TARGET" --mojo "$MOJO" --compile-jobs "$JOBS" --out "$OUT/artifacts" > "$G/build.log" 2>&1
PHASE=validate
"$PYTHON" - "$VENDOR" "$MOJOLEARN_COMMIT" "$OUT" <<'PY'
import json, pathlib, sys
sys.path.insert(0, 'tools')
import gate_autograd_artifacts as gate
vendor, commit, output = sys.argv[1:]
out = pathlib.Path(output)
record = gate.validate(out / 'artifacts/build-manifest.json', vendor, commit)
assert len(record['files']) == len(gate.expected(vendor)) == 8, 'expected eight compiled gate artifacts'
summary = {'source_commit': commit, 'vendor': vendor, 'target': record['target'],
           'state': record['state'], 'artifacts': len(record['files']), 'runtime_qualified': False,
           'files': [{'file': row['file'], 'sha256': row['sha256'], 'exit_code': row['exit_code']} for row in record['files']]}
(out / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
print(json.dumps({key: value for key, value in summary.items() if key != 'files'}))
PY
[[ $(git rev-parse HEAD) = "$MOJOLEARN_COMMIT" && -z $(git status --porcelain --untracked-files=no) ]] || { echo 'source changed during artifact build'; exit 3; }
PHASE=complete
