#!/usr/bin/env bash
# Match the NVIDIA builders' linker/CRT environment on a disposable AMD host.
# The pinned AMD userspace image uses the host's already installed GPU driver.
# No source or artifact is substituted: the complete build and its existing
# physical-device, source-inventory and byte-hash gates run inside the image.
set -euo pipefail
IMAGE=rocm/dev-ubuntu-22.04@sha256:a3850e6638c6c390436ef1aacd72fd1359af36083ac823d5136818206998c484
[[ $(uname -s) = Linux && $(id -u) = 0 && -e /dev/kfd ]] || exit 2
case ${1:-} in
    prepare)
        [[ $# = 1 ]] || exit 2
        timeout -k 10 480 docker pull "$IMAGE"
        exit
        ;;
    run) shift ;;
    *) echo 'Expected prepare or run hip gfx942 NEW_OUT' >&2; exit 2 ;;
esac
[[ $# = 3 && $1 = hip && $2 = gfx942 && $3 = /root/* ]] || exit 2
# The same parallel build as the NVIDIA legs (packaging/linux/build_sets.sh):
# MOJOLEARN_BUILD_JOBS extensions at a time, default 4, each capped at two
# compiler workers, so the container gets 2 x jobs cores and 16 GiB per job
# (at most three quarters of the droplet's memory). The byte compare in
# pack_wheel.py refuses the wheel if any host binding differs across legs.
JOBS=${MOJOLEARN_BUILD_JOBS:-4}
[[ "$JOBS" =~ ^[1-9][0-9]?$ && "$JOBS" -le 16 ]] || { echo 'MOJOLEARN_BUILD_JOBS must be 1..16' >&2; exit 2; }
# The core host probe is advisory (pack_wheel.py compares every host binding
# across legs): `skip` lets this leg start before the NVIDIA legs finish.
[[ ${MOJOLEARN_EXPECT_CORE_HOST_SHA256:-} =~ ^([0-9a-f]{64}|skip)$ ]] || { echo 'MOJOLEARN_EXPECT_CORE_HOST_SHA256 must be the STAGED NVIDIA set copy digest or skip' >&2; exit 2; }
cores=$(python3 -c 'import os, sys; print(",".join(map(str, sorted(os.sched_getaffinity(0))[:2 * int(sys.argv[1])])))' "$JOBS")
[[ -n "$cores" ]] || { echo 'Empty CPU affinity' >&2; exit 2; }
ncpus=$(awk -F, '{print NF}' <<< "$cores")
mem_gib=$(awk -v jobs="$JOBS" '/^MemTotal:/ {cap = int($2 / 1048576 * 3 / 4); want = 16 * jobs; print (want < cap ? want : cap)}' /proc/meminfo)
[[ "$mem_gib" =~ ^[0-9]+$ && "$mem_gib" -ge 16 ]] || { echo 'Container build needs at least 16 GiB' >&2; exit 2; }
echo "container_build_jobs=$JOBS cpuset=$cores cpus=$ncpus memory=${mem_gib}g"
# /root is the disposable rental's source, locked Pixi environment, tools and
# output tree. Keeping those paths identical preserves the compiler witnesses.
exec docker run --rm --pull=never --cpuset-cpus "$cores" --cpus "$ncpus" --memory "${mem_gib}g" \
    --device /dev/kfd --device /dev/dri --security-opt seccomp=unconfined \
    --network host --mount type=bind,src=/root,dst=/root \
    --workdir /root/mojolearn \
    --env MOJOLEARN_COMMIT --env MOJOLEARN_RELEASE_BUILD_SECONDS \
    --env MOJOLEARN_EXPECT_CORE_HOST_SHA256 \
    --env MOJOLEARN_BUILD_JOBS="$JOBS" \
    --env MOJOLEARN_PYTHON=/root/mojolearn/.pixi/envs/default/bin/python \
    --env PATH=/root/mojolearn/.pixi/envs/default/bin:/root/release-tools/bin:/root/.pixi/bin:/opt/rocm/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin \
    --entrypoint bash "$IMAGE" -c '
        set -euo pipefail
        . /etc/os-release
        [[ "$VERSION_ID" = 22.04 ]]
        gcc --version | head -1
        ld --version | head -1
        [[ $(gcc -dumpfullversion) = 11.4.0 ]]
        [[ $(ld --version | head -1) = "GNU ld (GNU Binutils for Ubuntu) 2.38" ]]
        patchelf --version
        [[ $(patchelf --version) = "patchelf 0.17.2" ]]
        # A small real binding proves the complete compiler/linker/stager
        # combination agrees with NVIDIA before spending a full build lease.
        probe=/root/release-toolchain-probe
        start=$(date +%s)
        if [[ "$MOJOLEARN_EXPECT_CORE_HOST_SHA256" = skip ]]; then
          echo "core_host_probe=skipped (compared at pack time)"
          exec bash tools/release061_remote_build.sh "$@"
        fi
        [[ ! -e "$probe" ]]
        mkdir "$probe"
        start=$(date +%s)
        OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
          MOJOLEARN_NUMERIC_MODE=identical MOJOLEARN_TARGET_COLUMN=cpu MOJOLEARN_SKIP_BUILD_GATE=1 \
          MOJOLEARN_CORE_HOST_OUTDIR="$probe" MOJOLEARN_BUILD_JOBS=1 \
          timeout -k 10 120 pixi run -e default bash bindings/build_core_host.sh
        patchelf --set-rpath '\''$ORIGIN/../.libs:$ORIGIN/../cuda/.libs:$ORIGIN/../hip/.libs:$ORIGIN/.libs'\'' \
          "$probe/_mojolearn_core_host.so"
        actual=$(sha256sum "$probe/_mojolearn_core_host.so" | cut -d" " -f1)
        echo "core_host_probe_sha256=$actual expected=$MOJOLEARN_EXPECT_CORE_HOST_SHA256"
        [[ "$actual" = "$MOJOLEARN_EXPECT_CORE_HOST_SHA256" ]]
        export MOJOLEARN_RELEASE_BUILD_SECONDS=$((MOJOLEARN_RELEASE_BUILD_SECONDS - $(date +%s) + start))
        exec bash tools/release061_remote_build.sh "$@"
    ' release-ubuntu22 "$@"
