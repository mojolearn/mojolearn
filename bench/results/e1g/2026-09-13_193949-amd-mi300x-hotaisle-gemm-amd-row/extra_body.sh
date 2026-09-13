# The AMD row of lib_gemm_kernel_body_for (DEVIATION 2707, brief section 18.2):
# price the gather staging on the MI300X. `shipped` is AMD's row-0 dispatch
# (ksplit at 110), `kpack_gs` is the gather staging alone, `kpack_hg` the
# shipped NVIDIA body, whose hardware fold flush is comptime-inert off NVIDIA
# (gemm_identical.mojo: `comptime if HW and TUNED_HW_FTZ_FMA`), so on this
# column kpack_hg must price like kpack_gs and every witness must equal shipped.
set -u
cd /root/mojolearn || exit 9
MOJOLEARN_GPU_ARCHS=${MOJOLEARN_GPU_ARCHS:-gfx942} \
MOJOLEARN_TARGET_COLUMN=amd \
MOJOLEARN_COMPILE_JOBS=8 \
MOJOLEARN_GEMM_STEP_LEG_ARMS=shipped,kpack_gs,kpack_hg \
MOJOLEARN_GEMM_STEP_LEG_LM_ARMS=kpack_gs,kpack_hg \
MOJOLEARN_GEMM_STEP_LEG_CHECK_ARMS=shipped,kpack_gs,kpack_hg \
MOJOLEARN_GEMM_STEP_LEG_OUT=/root/gemm_leg_out/gemm-amd-row \
    sh tools/gemm_kernel_leg.sh
