#!/bin/bash
# tools/ops/published_install_job.sh -- ONE published-smoke-<vendor> job on a GPU box (lane release-no-hands,
# 2026-11-04; since lane published-no-reverify ONLY for an explicit `release.py --published-smoke <vendor>`
# with a recorded reason: published-<vendor> itself is a bytes check on the Mac and needs no box):
# tools/release_published_install.py run against the PUBLISHED release, installed from PyPI,
# compared with the qualified candidate wheels. tools/ops/published_install.py uploads this file verbatim
# beside its published.env and the candidate wheels (sha256 checked on the box) and launches it through the
# supported wrappers, which run it under tools/box_persist_run.sh (results to R2 from the box):
#   tools/nvidia_central.sh submit release-gate --cap 120 --results <res> <in>/published_install_job.sh
#   tools/do_amd_steward.sh run <tag> <res> bash <in>/published_install_job.sh
# It takes no arguments (`--detect-vendor` prints the box's vendor and exits). The vendor is the box's own
# (nvidia-smi -> nvidia, rocm-smi -> amd), as tools/identity_box_job.sh decides it; published.env's
# PUB_VENDOR, when set, must agree.
#
# In order: tree (PUB_TREE at PUB_SOURCE_COMMIT, clean; AMD: fetched here from PUB_FETCH_URL, NVIDIA: the
# lane tree tools/nvidia_central.sh sync placed there), runtime (AMD: the ROCm 7.2.3 runtime the adapter
# smoke needs; python venv support), then the producer with its report in $MOJOLEARN_RESULTS_DIR/out
# (published-report.json; it comes home to <commit12>/published-smoke-<vendor>/). The job's own record
# goes to $MOJOLEARN_RESULTS_DIR/job.
set -euo pipefail

detect_vendor() {
    if command -v nvidia-smi > /dev/null 2>&1; then echo nvidia
    elif command -v rocm-smi > /dev/null 2>&1 || [ -x /opt/rocm/bin/rocm-smi ]; then echo amd
    else return 2; fi
}
if [ "${1:-}" = --detect-vendor ]; then
    detect_vendor || { echo "published-install-job: neither nvidia-smi nor rocm-smi found" >&2; exit 2; }
    exit 0
fi
[ $# -eq 0 ] || { echo "usage: $0 [--detect-vendor]" >&2; exit 2; }
IN="$(cd "$(dirname "$0")" && pwd)"
RES="${MOJOLEARN_RESULTS_DIR:?published_install_job.sh runs under tools/box_persist_run.sh (nvidia_central.sh submit or do_amd_steward.sh run)}"
[ -f "$IN/published.env" ] || { echo "published-install-job: no $IN/published.env" >&2; exit 2; }
# shellcheck disable=SC1091
. "$IN/published.env"
BOX_VENDOR=$(detect_vendor) || { echo "published-install-job: neither nvidia-smi nor rocm-smi found" >&2; exit 2; }
V=${PUB_VENDOR:-$BOX_VENDOR}
[ "$V" = "$BOX_VENDOR" ] || { echo "published-install-job: published.env says $V; this box is $BOX_VENDOR" >&2; exit 3; }
[[ "$PUB_VERSION" =~ ^[0-9A-Za-z.+-]+$ ]] || { echo "published-install-job: bad PUB_VERSION" >&2; exit 2; }
[[ "$PUB_SOURCE_COMMIT" =~ ^[0-9a-f]{40}$ ]] || { echo "published-install-job: bad PUB_SOURCE_COMMIT" >&2; exit 2; }
[[ "$PUB_TREE" =~ ^/root/[A-Za-z0-9/_.-]+$ ]] || { echo "published-install-job: PUB_TREE must be a plain /root path" >&2; exit 2; }
[[ "${PUB_TIMEOUT:-1800}" =~ ^[1-9][0-9]*$ ]] || { echo "published-install-job: bad PUB_TIMEOUT" >&2; exit 2; }
[ ! -e "$RES/out" ] || { echo "published-install-job: $RES/out exists; a new job uses a new tag" >&2; exit 2; }
J="$RES/job"
mkdir -p "$J"
exec > "$J/job.log" 2>&1
PHASE=inputs
export PATH="/root/.pixi/bin:/usr/local/cuda/bin:/opt/rocm/bin:$PATH"
say() { echo "[$(date -u +%H:%M:%S) published-install-job] $*"; }
finish() {
    local code=$?
    trap - EXIT
    printf '{"vendor":"%s","version":"%s","source_commit":"%s","tag":"%s","phase":"%s","exit":%d,"job_script_sha256":"%s","env_sha256":"%s"}\n' \
        "$V" "$PUB_VERSION" "$PUB_SOURCE_COMMIT" "${PUB_TAG:-}" "$PHASE" "$code" \
        "$(sha256sum "$IN/$(basename "$0")" | cut -d' ' -f1)" "$(sha256sum "$IN/published.env" | cut -d' ' -f1)" > "$J/status.json"
    say "exit $code in phase $PHASE"
    exit "$code"
}
trap finish EXIT
trap 'exit 124' TERM INT
say "published-$V $PUB_VERSION at $PUB_SOURCE_COMMIT (${PUB_TAG:-}); inputs $IN"
wheel_args=()
for w in "${PUB_WHEELS[@]}"; do
    [ -f "$w" ] || { say "missing input $w"; exit 2; }
    wheel_args+=(--candidate-wheel "$w")
done
[ ${#wheel_args[@]} -gt 0 ] || { say "no candidate wheel"; exit 2; }
{ uname -a; nproc; free -g 2>/dev/null || true; df -h /root; } > "$J/box.txt" 2>&1 || true

PHASE=tree
if [ -n "${PUB_FETCH_URL:-}" ]; then
    mkdir -p "$PUB_TREE" && cd "$PUB_TREE"
    [ -d .git ] || { git init -q . && git remote add origin "$PUB_FETCH_URL"; }
    git config --global --add safe.directory "$PUB_TREE" 2>/dev/null || true
    if [ "$(git rev-parse -q --verify HEAD 2>/dev/null || true)" != "$PUB_SOURCE_COMMIT" ]; then
        git fetch -q --depth=1 origin "$PUB_SOURCE_COMMIT"
        git checkout -q --detach -f "$PUB_SOURCE_COMMIT"
    fi
fi
cd "$PUB_TREE"
git config --global --add safe.directory "$PUB_TREE" 2>/dev/null || true
[ "$(git rev-parse HEAD)" = "$PUB_SOURCE_COMMIT" ] && [ -z "$(git status --porcelain --untracked-files=no)" ] \
    || { say "$PUB_TREE is not a clean tree at $PUB_SOURCE_COMMIT (HEAD $(git rev-parse HEAD 2>/dev/null))"; exit 3; }

PHASE=runtime
if [ "$V" = amd ] && [ ! -d /opt/rocm-7.2.3/lib ]; then
    say "installing the ROCm 7.2.3 runtime (tools/amd_box_runtime_setup.sh)"
    bash tools/amd_box_runtime_setup.sh > "$J/amd-runtime.log" 2>&1
fi
# the controller needs Python >= 3.11 (hashlib.file_digest); the venv base is the same interpreter
PY=""
for c in python3.12 python3.13 python3.11 python3; do
    if command -v "$c" > /dev/null 2>&1 && "$c" -c 'import sys; sys.exit(sys.version_info < (3, 11))' 2>/dev/null; then
        PY=$(command -v "$c"); break
    fi
done
[ -n "$PY" ] || { say "no python3 >= 3.11 on this box"; exit 3; }
if ! "$PY" -c 'import ensurepip, venv' 2>/dev/null; then
    say "$PY lacks venv support; installing it"
    export DEBIAN_FRONTEND=noninteractive
    pyv=$("$PY" -c 'import sys; print("%d.%d" % sys.version_info[:2])')
    { apt-get -o DPkg::Lock::Timeout=300 update -qq && apt-get -o DPkg::Lock::Timeout=300 install -y -qq "python$pyv-venv"; } > "$J/apt.log" 2>&1
fi
say "python $PY ($("$PY" --version 2>&1))"

PHASE=published
prc=0
"$PY" tools/release_published_install.py run --vendor "$V" --version "$PUB_VERSION" "${wheel_args[@]}" \
    --python "$PY" --timeout "${PUB_TIMEOUT:-1800}" --out "$RES/out" > "$J/published.log" 2>&1 || prc=$?
grep -m 2 '^PUBLISHED' "$J/published.log" || tail -n 5 "$J/published.log"
exit "$prc"
