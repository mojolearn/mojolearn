#!/bin/bash
# Experimental GitHub CPU bootstrap. Source and tooling are separate mounts.
# Invoke the frozen source's existing build body without overlaying tracked files.
set -euo pipefail
SOURCE=/root/mojolearn
OUT=/root/leg_out
[[ ${SOURCE_COMMIT:-} =~ ^[0-9a-f]{40}$ && ${TOOLING_COMMIT:-} =~ ^[0-9a-f]{40}$ ]] || exit 2
[[ ${BUILD_JOBS:-} =~ ^[12]$ ]] || exit 2
mkdir -p "$OUT/bootstrap"
exec > >(tee "$OUT/bootstrap/bootstrap.log") 2>&1
trap 'rc=$?; echo "$rc" > "$OUT/bootstrap/exit-code.txt"' EXIT
# Bootstrap utilities are not part of the pinned GCC/binutils numerical toolchain.
need=''
for pair in curl:curl git:git xz:xz-utils python3:python3; do
    command -v "${pair%%:*}" >/dev/null || need="$need ${pair#*:}"
done
if [[ -n "$need" ]]; then
    export DEBIAN_FRONTEND=noninteractive
    timeout -k 10 180 apt-get -o Acquire::Retries=2 update -qq
    timeout -k 10 180 apt-get -o Acquire::Retries=2 install -y --no-install-recommends $need ca-certificates
fi
git config --global --add safe.directory "$SOURCE"
git config --global --add safe.directory /tooling
cd "$SOURCE"
[[ $(git rev-parse HEAD) == "$SOURCE_COMMIT" ]] || { echo 'Source checkout mismatch'; exit 2; }
[[ $(git -C /tooling rev-parse HEAD) == "$TOOLING_COMMIT" ]] || { echo 'Tooling checkout mismatch'; exit 2; }
git diff --quiet HEAD --
git -C /tooling diff --quiet HEAD --
[[ -f tools/nvidia_baseline_cpu_box.sh ]] || { echo 'Frozen source lacks experimental build body'; exit 2; }
{
    echo "source_commit=$SOURCE_COMMIT"
    echo "source_tree=$(git rev-parse HEAD^{tree})"
    echo "tooling_commit=$TOOLING_COMMIT"
    echo "tooling_tree=$(git -C /tooling rev-parse HEAD^{tree})"
    echo "runner=${GHA_RUNNER:-} run=${GHA_RUN:-}"
    sha256sum tools/nvidia_baseline_cpu_box.sh pixi.lock /tooling/tools/gha_nvidia_baseline_box.sh
    uname -a
    free -m
} > "$OUT/bootstrap/provenance.txt"
export PIXI_HOME=/root/.pixi PIXI_NO_PATH_UPDATE=1
PIXI=/root/.pixi/bin/pixi
# Same pinned installer version as the existing GitHub/native CPU route.
timeout -k 10 300 curl -fsSL --retry 2 --max-time 90 https://pixi.sh/install.sh -o "$OUT/bootstrap/install-pixi.sh"
timeout -k 10 300 env PIXI_VERSION=0.77.0 bash "$OUT/bootstrap/install-pixi.sh"
[[ $("$PIXI" --version) == 'pixi 0.77.0' ]] || { echo 'Pixi version differs'; exit 2; }
export PATH="$SOURCE/.pixi/envs/default/bin:/root/.pixi/bin:$PATH"
timeout -k 20 600 "$PIXI" install --locked -e default > "$OUT/bootstrap/pixi.log" 2>&1
PY="$SOURCE/.pixi/envs/default/bin/python"
"$PY" - <<'PY'
import sys
sys.path.insert(0, 'tools')
from cpu_build_guard import visible_devices
assert not visible_devices(), 'GPU-visible environment refused'
PY
# The leased runner normally prepares this helper before invoking the body.
PYTHONPATH="$SOURCE/packaging/portable_math" "$PY" - <<'PY'
import pathlib,stage
stage.build(pathlib.Path('python/mojolearn/.libs/libMojolearnMath.so'))
PY
export LEG_OUT="$OUT"
bash tools/nvidia_baseline_cpu_box.sh "$SOURCE_COMMIT" https://github.com/mojolearn/mojolearn.git 6000 "$BUILD_JOBS"
# Source remains frozen; successful manifest validation is performed by the body.
git diff --quiet HEAD --
