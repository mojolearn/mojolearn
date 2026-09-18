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
[[ ${MOJOLEARN_BUILD_JOBS:-1} = 1 ]] || { echo 'Container release build requires one compiler' >&2; exit 2; }
[[ ${MOJOLEARN_EXPECT_CORE_HOST_SHA256:-} =~ ^[0-9a-f]{64}$ ]] || { echo 'Expected NVIDIA core-host SHA256 required' >&2; exit 2; }
cores=$(python3 -c 'import os; print(",".join(map(str, sorted(os.sched_getaffinity(0))[:2])))')
# /root is the disposable rental's source, locked Pixi environment, tools and
# output tree. Keeping those paths identical preserves the compiler witnesses.
exec docker run --rm --pull=never --cpuset-cpus "$cores" --cpus 2 --memory 16g \
    --device /dev/kfd --device /dev/dri --security-opt seccomp=unconfined \
    --network host --mount type=bind,src=/root,dst=/root \
    --workdir /root/mojolearn \
    --env MOJOLEARN_COMMIT --env MOJOLEARN_RELEASE_BUILD_SECONDS \
    --env MOJOLEARN_EXPECT_CORE_HOST_SHA256 \
    --env MOJOLEARN_BUILD_JOBS=1 \
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
        # A small real binding proves the complete compiler/linker/stager
        # combination agrees with NVIDIA before spending a full build lease.
        probe=/root/release-toolchain-probe
        [[ ! -e "$probe" ]]
        mkdir "$probe"
        start=$(date +%s)
        OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
          MOJOLEARN_CORE_HOST_OUTDIR="$probe" MOJOLEARN_BUILD_JOBS=2 \
          timeout -k 10 120 bash bindings/build_core_host.sh
        patchelf --set-rpath '\''$ORIGIN/../.libs:$ORIGIN/../cuda/.libs:$ORIGIN/../hip/.libs:$ORIGIN/.libs'\'' \
          "$probe/_mojolearn_core_host.so"
        actual=$(sha256sum "$probe/_mojolearn_core_host.so" | cut -d" " -f1)
        echo "core_host_probe_sha256=$actual expected=$MOJOLEARN_EXPECT_CORE_HOST_SHA256"
        [[ "$actual" = "$MOJOLEARN_EXPECT_CORE_HOST_SHA256" ]]
        export MOJOLEARN_RELEASE_BUILD_SECONDS=$((MOJOLEARN_RELEASE_BUILD_SECONDS - $(date +%s) + start))
        exec bash tools/release061_remote_build.sh "$@"
    ' release-ubuntu22 "$@"
