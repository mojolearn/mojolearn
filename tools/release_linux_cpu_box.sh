#!/usr/bin/env bash
# ON THE CPU BUILD BOX ONLY (a RunPod CPU pod running the pinned Ubuntu 22.04
# image; tools/release_linux_build.sh rents it and runs this as its command).
# Builds the three Linux release sets, cuda/sm_90a, cuda/sm_89 and hip/gfx942,
# one after another in /root/mojolearn (the binaries carry the source path, so
# every set is built at the same path the GPU legs used), each through the
# unchanged tools/release061_remote_build.sh with MOJOLEARN_RELEASE_NO_DEVICE=1.
#
#   bash tools/release_linux_cpu_box.sh OUT_DIR [arch ...]    (default: all three)
#
# Output: OUT_DIR/<vendor>-<arch>/release-build/, the same tree a GPU leg
# writes (build/sets/<vendor>/<arch>/..., build/build-provenance.json), plus
# OUT_DIR/route.txt (toolchain, image and timing witness).
#
# Environment: MOJOLEARN_COMMIT (full SHA of the shipped source), and
# optionally MOJOLEARN_BUILD_JOBS (default: cores / 2, at most 16 GiB of RAM
# per job, at most 16 jobs).
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT"
OUT=${1:?output directory}
shift
ARCHS=("$@")
[[ ${#ARCHS[@]} -gt 0 ]] || ARCHS=(sm_90a sm_89 gfx942)
commit=${MOJOLEARN_COMMIT:?full source commit required}
[[ "$commit" =~ ^[0-9a-f]{40}$ ]] || { echo 'MOJOLEARN_COMMIT must be a full SHA' >&2; exit 2; }
[[ "$OUT" = /* ]] || { echo 'absolute OUT required' >&2; exit 2; }
mkdir -p "$OUT"
say() { echo "[$(date -u +%T) cpu-build-box] $*"; }

# ---------------------------------------------------------------- the image
# The same pins tools/release_ubuntu22_build.sh asserts inside the AMD leg's
# container: this box IS that image, so the same four facts must hold.
. /etc/os-release
[[ "$VERSION_ID" = 22.04 ]] || { echo "Ubuntu 22.04 required, found $VERSION_ID" >&2; exit 3; }
[[ $(gcc -dumpfullversion) = 11.4.0 ]] || { echo "GCC 11.4.0 required, found $(gcc -dumpfullversion)" >&2; exit 3; }
[[ $(ld --version | head -1) = "GNU ld (GNU Binutils for Ubuntu) 2.38" ]] || { echo "ld 2.38 required: $(ld --version | head -1)" >&2; exit 3; }
for dev in /dev/nvidia0 /dev/nvidiactl /dev/kfd; do
    [[ ! -e "$dev" ]] || { echo "a GPU device ($dev) is visible; this route builds without one" >&2; exit 3; }
done
command -v taskset >/dev/null && command -v objdump >/dev/null
PY=$ROOT/.pixi/envs/default/bin/python
[[ -x "$PY" && -x "$ROOT/.pixi/envs/default/bin/mojo" ]] || { echo 'locked default pixi env required' >&2; exit 3; }

# patchelf 0.17.2.4, the stager every GPU leg pins (gemm_remote_leg.sh's
# pinned-private-venv, do_release061_leg.sh's /root/release-tools).
if [[ ! -x /root/release-tools/bin/patchelf ]]; then
    "$PY" -m venv /root/release-tools
    /root/release-tools/bin/python -m pip install -q --disable-pip-version-check --only-binary=:all: \
        --retries 2 --timeout 30 'patchelf==0.17.2.4'
fi
export PATH=/root/release-tools/bin:$ROOT/.pixi/envs/default/bin:/root/.pixi/bin:$PATH
[[ $(patchelf --version) = "patchelf 0.17.2" ]] || { echo "patchelf 0.17.2 required: $(patchelf --version)" >&2; exit 3; }

ncpu=$(nproc)
mem_gib=$(awk '/^MemTotal:/ {print int($2 / 1048576)}' /proc/meminfo)
# A pod's /proc/meminfo is the HOST's (755 GiB on the first proof pod, which
# had 128); the container's own limit is the cgroup's.
for cg in /sys/fs/cgroup/memory.max /sys/fs/cgroup/memory/memory.limit_in_bytes; do
    if [[ -r "$cg" ]] && [[ $(cat "$cg") =~ ^[0-9]+$ ]]; then
        cg_gib=$(( $(cat "$cg") / 1073741824 ))
        ((cg_gib >= mem_gib)) || mem_gib=$cg_gib
        break
    fi
done
if [[ -z "${MOJOLEARN_BUILD_JOBS:-}" ]]; then
    jobs=$((ncpu / 2)); by_mem=$((mem_gib / 16))
    ((jobs <= by_mem)) || jobs=$by_mem
    ((jobs <= 16)) || jobs=16
    ((jobs >= 1)) || jobs=1
    MOJOLEARN_BUILD_JOBS=$jobs
fi
printf '%s\n' "$commit" > "$ROOT/commit.txt"
{
    echo "route=cpu-build-box"
    echo "source_commit=$commit"
    echo "os=$PRETTY_NAME"
    echo "gcc=$(gcc -dumpfullversion)"
    echo "ld=$(ld --version | head -1)"
    echo "patchelf=$(patchelf --version)"
    echo "glibc=$(ldd --version | head -1)"
    echo "mojo=$("$ROOT/.pixi/envs/default/bin/mojo" --version 2>&1 | head -1)"
    echo "dpkg=$(dpkg-query -W -f '${Package}=${Version} ' gcc-11 cpp-11 libgcc-11-dev libstdc++-11-dev binutils binutils-x86-64-linux-gnu libc6 libc6-dev libc-dev-bin 2>/dev/null)"
    echo "cpu=$(grep -m1 'model name' /proc/cpuinfo | cut -d: -f2- | sed 's/^ //')"
    echo "nproc=$ncpu mem_gib=$mem_gib build_jobs=$MOJOLEARN_BUILD_JOBS"
    echo "archs=${ARCHS[*]}"
    for f in tools/release061_remote_build.sh tools/cpu_build_guard.py tools/release_linux_cpu_box.sh \
             tools/linux_surface_qualification.sh packaging/linux/build_sets.sh tools/bincache.py; do
        [[ -f "$f" ]] && echo "route_file $f $(sha256sum "$f" | cut -c1-64)"
    done
} > "$OUT/route.txt"
cat "$OUT/route.txt"

# ---------------------------------------------------------------- the sets
# Sequential by design: all three must build at /root/mojolearn, and
# build_sets.sh builds in python/mojolearn/ before moving each set out.
# EVERY SET STARTS FROM THE SAME TREE. A build writes untracked files into the
# checkout (tokenizer/impl/unicode_table_generated.mojo, generated by the host
# tokenizer build), and linux_surface_qualification.sh's source inventory,
# taken before each set, walks untracked files too. On the first proof run
# (2026-09-22) the second and third sets therefore recorded one more inventory
# file than the first, and pack_wheel.py refuses sets whose inventories
# differ. The GPU legs never met this because each built on a fresh box. So
# every file the previous set added is removed before the next set starts.
tree_files() { find . -path ./.pixi -prune -o -path ./.git -prune -o -type f -print | LC_ALL=C sort; }
tree_files > /root/.release-tree-before.txt
rc=0
for arch in "${ARCHS[@]}"; do
    tree_files | LC_ALL=C comm -13 /root/.release-tree-before.txt - > /root/.release-tree-added.txt
    if [[ -s /root/.release-tree-added.txt ]]; then
        say "removing $(wc -l < /root/.release-tree-added.txt) file(s) the previous set added: $(head -5 /root/.release-tree-added.txt | tr '\n' ' ')"
        xargs -d '\n' rm -f < /root/.release-tree-added.txt
    fi
    case "$arch" in sm_*) vendor=cuda ;; gfx*) vendor=hip ;; *) echo "unknown arch $arch" >&2; exit 2 ;; esac
    dest="$OUT/$vendor-$arch"
    mkdir -p "$dest"
    t0=$(date +%s)
    say "building $vendor/$arch with $MOJOLEARN_BUILD_JOBS jobs"
    status=0
    env -u MOJOLEARN_NUMERIC_MODE -u PYTHONPATH \
        MOJOLEARN_RELEASE_NO_DEVICE=1 MOJOLEARN_COMMIT="$commit" MOJOLEARN_PYTHON="$PY" \
        MOJOLEARN_RELEASE_BUILD_SECONDS=2400 MOJOLEARN_BUILD_JOBS="$MOJOLEARN_BUILD_JOBS" \
        bash tools/release061_remote_build.sh "$vendor" "$arch" "$dest/release-build" \
        > "$dest/release-build-console.log" 2>&1 || status=$?
    t1=$(date +%s)
    printf '%s\t%s\t%s\t%s\n' "$vendor" "$arch" "$status" "$((t1 - t0))" >> "$OUT/sets.tsv"
    say "$vendor/$arch exit $status in $((t1 - t0)) s"
    cat "$dest/release-build/results.tsv" 2>/dev/null | sed 's/^/    /'
    grep -h '"guard"' "$dest/release-build/full46-build.log" 2>/dev/null | sed 's/^/    /' || true
    [[ "$status" = 0 ]] || rc=1
done
exit "$rc"
