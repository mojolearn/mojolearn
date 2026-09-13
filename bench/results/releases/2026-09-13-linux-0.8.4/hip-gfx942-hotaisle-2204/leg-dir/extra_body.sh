# 0.8.4 HIP gfx942 release build in the rocm/dev-ubuntu-22.04 container
# (GCC 11, the same crt objects and linker as the RunPod CUDA legs), because
# the DigitalOcean 24.04 image linked a vendor-neutral CPU binding whose
# bytes differ from the two CUDA legs' identical copies and pack_wheel
# refuses a disagreeing copy. Same frozen commit, same script the DO leg runs.
set -u
cd /root/mojolearn || exit 9
export PATH=/root/.pixi/bin:$PATH
echo bfde04428b4f548bac0d9ebeeb471ffbe81b084e > /root/mojolearn/commit.txt
command -v patchelf >/dev/null 2>&1 || python3 -m pip install -q --disable-pip-version-check patchelf==0.17.2.4 2>&1 | tail -1
command -v patchelf >/dev/null 2>&1 || { apt-get -qq update >/dev/null 2>&1; apt-get -qq install -y python3-pip >/dev/null 2>&1; python3 -m pip install -q --disable-pip-version-check patchelf==0.17.2.4; }
for t in taskset objdump patchelf pixi python3; do command -v $t >/dev/null || echo MISSING_$t; done > /root/gemm_leg_out/tools.txt
patchelf --version >> /root/gemm_leg_out/tools.txt 2>&1; gcc --version 2>/dev/null | head -1 >> /root/gemm_leg_out/tools.txt; ld --version 2>/dev/null | head -1 >> /root/gemm_leg_out/tools.txt
MOJOLEARN_COMMIT=bfde04428b4f548bac0d9ebeeb471ffbe81b084e MOJOLEARN_PYTHON=/usr/bin/python3 MOJOLEARN_RELEASE_BUILD_SECONDS=2400 MOJOLEARN_BUILD_JOBS=4 \
  timeout -k 20 2500 bash tools/release061_remote_build.sh hip gfx942 /root/gemm_leg_out/release-build > /root/gemm_leg_out/release-build-console.log 2>&1
echo "release_build_exit=$?" >> /root/gemm_leg_out/tools.txt
sha256sum /root/gemm_leg_out/release-build/build/sets/hip/gfx942/host/*.so >> /root/gemm_leg_out/tools.txt 2>&1
