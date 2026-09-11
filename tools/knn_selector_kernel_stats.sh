#!/bin/sh
# tools/knn_selector_kernel_stats.sh -- DEVIATION 2519: read the compiled
# small-k selector's register / local-memory / spill footprint off the
# NVIDIA box, per instantiation, and say nothing about speed.
#
# Runs ON THE POD as tools/gemm_remote_leg.sh's MOJOLEARN_GEMM_LEG_EXTRA
# hook (after the leg's own IDENTICAL device check and card), from
# /root/mojolearn with pixi on PATH; everything it writes under
# /root/gemm_leg_out/knn-kernel-stats/ comes home with the leg's fetch.
#
#   MOJOLEARN_RUNPOD_KEY_FILE=$HOME/.mojolearn_runpod_key \
#   MOJOLEARN_GEMM_LEG_EXTRA=tools/knn_selector_kernel_stats.sh \
#   MOJOLEARN_GEMM_LEG_OUT=bench/results/e1g/$(date +%Y-%m-%d_%H%M%S)-nvidia-h100-knn-kernel-stats \
#   sh tools/gemm_remote_leg.sh nvidia --rent --minutes 60 \
#       --gpu "NVIDIA H100 80GB HBM3" \
#       --local-card bench/results/e1/2026-08-28_131651-runpod-nvidia/lanes/gemm.identical.card
#
# WHY. The brief (docs/lanes/BRIEF_knn_selection_2026-09-10.md, "Step 5
# result") closed the bound family and the deferred insertion: three
# mechanisms that run the K-chain less often left the 0.80 ms per unit of k
# untouched, so the cost is a property of the compiled kernel, not of how
# often the chain executes. The three candidate properties are (a) the
# CAP-wide SIMD[uint64] list lives in local memory (ld.local / st.local in
# the body, a nonzero local depot), (b) its register footprint caps
# occupancy (registers above 128 per thread at 256 threads per block means
# at most one block per SM), (c) neither, and the cost is instruction count.
# This body measures (a) and (b); the brief says which fix each implies.
#
# METHOD (documented API first, repo-proven fallback second):
#
#   driver    A generated Mojo driver (written below, built here, never
#             launches a kernel) calls, for each shipped instantiation of
#             `smallk_bucket_kernel`,
#               DeviceContext.compile_function[kernel, dump_asm=<path or
#               True>, _dump_sass=<path or True>, _ptxas_info_verbose=True]()
#             and then DeviceFunction.get_attribute(Attribute.NUM_REGS /
#             LOCAL_SIZE_BYTES / SHARED_SIZE_BYTES / CONST_SIZE_BYTES /
#             MAX_THREADS_PER_BLOCK) and
#             DeviceFunction.occupancy_max_active_blocks_per_multiprocessor(256, 0).
#             Those are the runtime's own answers about the cubin it will
#             launch: the authoritative registers / local bytes / occupancy.
#             API reference: max.modular.com/api/mojo/max/gpu/host/device_context/DeviceContext
#             (compile_function), .../DeviceFunction (get_attribute,
#             occupancy_max_active_blocks_per_multiprocessor, dump_rep) and
#             .../func_attribute/Attribute (NUM_REGS = CU_FUNC_ATTRIBUTE_NUM_REGS,
#             LOCAL_SIZE_BYTES, SHARED_SIZE_BYTES). `_dump_sass` and
#             `_ptxas_info_verbose` are documented as NVIDIA-only and needing
#             the CUDA toolkit on the box, so they are requested only when
#             a toolkit is found. The file-path dump form is a function
#             returning a Path (one of the documented Variant members); if
#             that variant of the driver does not build, the Bool form
#             (dump to stdout, documented) is built instead and the parser
#             splits stdout.
#   assemble  `ptxas --verbose --gpu-name <the PTX's .target>` on every
#             dumped PTX gives the SPILL stores / loads (nothing else
#             reports them), then `cuobjdump --dump-resource-usage` and
#             `--dump-sass` on the cubin give REG / STACK / LOCAL / SHARED
#             and the LDL / STL that execute. Same tools and flags as
#             tools/gemm_cuda_resources.py (the GEMM lane's H100 pass,
#             docs/lanes/HANDOFF_speed_gemm_2026-09-10.md, "H100 resource
#             inspection"); NVIDIA's binary-utilities documentation names
#             them. An offline ptxas is that toolkit's answer, not
#             necessarily the runtime JIT's: where the two disagree the
#             driver's runtime attributes win (stats.tsv carries both).
#   emit-asm  `mojo build --emit asm` of bench/knn_reference_price_main.mojo
#             retains one `<out>_<module>_<hash>.ptx` sidecar per GPU kernel
#             (the form the GEMM lane used for h100-current-128.ptx.gz,
#             bench/results/gemm_swizzle_2026-09-10/README.md). Entry names
#             are module + hash, not parameters, so the instantiation is
#             identified by elimination: the default build has [16,10],
#             [16,15], [16,0], [32,0], [64,0]; the -D
#             MOJOLEARN_KNN_IDENTICAL_GENERIC_K=1 build has only the three
#             K=0 buckets, so the two sidecars present only in the default
#             build are the k10 / k15 specializations. This is the
#             cross-check for the driver, and the whole answer if the
#             driver fails to build.
#   binding   the trial binding is built the way tools/knn_selection_gate.sh
#             builds it (bindings/build.sh with the trial define through
#             MOJOLEARN_BUILD_EXTRA_DEFINES) and witnessed by hash; any
#             NVPTX text blobs inside the .so are extracted as a third
#             source (attempt only: whether Mojo embeds PTX as text in a
#             shared library is not documented and this records the
#             answer either way).
#
# OUTPUT (all under $OUT): stats.tsv (one row per instantiation: label,
# cap, k, entry, registers, local_bytes, spill_stores, spill_loads,
# ld_local, st_local, sass_ldl, sass_stl, shared_bytes, blocks_per_sm_*,
# occupancy_pct, runtime_blocks_per_sm, runtime_occupancy_pct, sources),
# driver.log, dumps/<label>.{ptx,sass,ptxas.log,resources.log}, emit/,
# status.tsv, stats.txt. Dumps larger than 256 KB are gzipped, binaries
# and cubins are deleted, and the whole directory is fenced at 2 MB.
#
# IDENTICAL only. No kernel changes. Nothing here is a timing number.
# POSIX sh only: RunPod's Ubuntu images link /bin/sh to dash.
set -u
ROOT=${MOJOLEARN_KNN_ROOT:-/root/mojolearn}
OUT=${MOJOLEARN_KNN_STATS_OUT:-/root/gemm_leg_out/knn-kernel-stats}
JOBS=${MOJOLEARN_COMPILE_JOBS:-2}
BUILD_TIMEOUT=${MOJOLEARN_KNN_STATS_BUILD_TIMEOUT:-900}
SIZE_FENCE_KB=2048
mkdir -p "$OUT/bin" "$OUT/dumps" "$OUT/emit" "$OUT/blobs"
cd "$ROOT" || exit 9
export PATH="$HOME/.pixi/bin:$PATH"
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 NUMEXPR_NUM_THREADS=2

# MAX's bundled CUDA 13 assembler needs driver 580. Older-driver pods use
# their installed assembler at BOTH build and runtime (the gate does the
# same, tools/knn_selection_gate.sh).
driver_major=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1 | cut -d. -f1)
case "$driver_major" in
    ''|*[!0-9]*) ;;
    *) if [ "$driver_major" -lt 580 ] && [ -x /usr/local/cuda/bin/ptxas ]; then
           export MODULAR_NVPTX_COMPILER_PATH=${MODULAR_NVPTX_COMPILER_PATH:-/usr/local/cuda/bin/ptxas}
       fi ;;
esac

rc=0
run() {
    # run <name> <command...>: log to $OUT/<name>.log, record the exit code.
    _name=$1
    shift
    _t0=$(date +%s)
    "$@" > "$OUT/$_name.log" 2>&1
    _code=$?
    printf '%s\t%s\t%ss\n' "$_name" "$_code" "$(( $(date +%s) - _t0 ))" >> "$OUT/status.tsv"
    [ "$_code" -eq 0 ] || rc=1
    return "$_code"
}
note() { echo "$*" >> "$OUT/stats.txt"; }
# fence <seconds> <command...>: coreutils `timeout` when the box has it (the
# pod's Ubuntu does); a plain run where it does not, so a missing utility
# never reads as a failed build.
if command -v timeout > /dev/null 2>&1; then
    fence() { _s=$1; shift; timeout "$_s" "$@"; }
else
    fence() { shift; "$@"; }
fi

{
    echo "deviation=2519"
    echo "started=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "root=$ROOT"
    echo "jobs=$JOBS"
    [ -f /root/gemm_leg_out/leg.txt ] && grep -E '^(commit|vendor)=' /root/gemm_leg_out/leg.txt
} > "$OUT/stats.txt"
nvidia-smi --query-gpu=name,driver_version,uuid,compute_cap --format=csv > "$OUT/gpu.csv" 2>&1
pixi run mojo --version > "$OUT/mojo_version.txt" 2>&1

# ---- CUDA binary utilities: PATH, the usual toolkit roots, then the pixi
# environment (a pip nvidia-cuda-nvcc wheel puts ptxas under
# site-packages/nvidia/cuda_nvcc/bin). Logged either way.
find_tool() {
    _t=$1
    _p=$(command -v "$_t" 2>/dev/null)
    if [ -z "$_p" ]; then
        for _d in /usr/local/cuda/bin /usr/local/cuda-13*/bin /usr/local/cuda-12*/bin /opt/cuda/bin; do
            [ -x "$_d/$_t" ] && { _p="$_d/$_t"; break; }
        done
    fi
    if [ -z "$_p" ] && [ -d "$ROOT/.pixi/envs/default" ]; then
        _p=$(find "$ROOT/.pixi/envs/default" -maxdepth 8 -type f -name "$_t" -perm -u+x 2>/dev/null | head -1)
    fi
    echo "$_p"
}
PTXAS=$(find_tool ptxas)
CUOBJDUMP=$(find_tool cuobjdump)
NVDISASM=$(find_tool nvdisasm)
{
    echo "ptxas=${PTXAS:-<none>}"
    echo "cuobjdump=${CUOBJDUMP:-<none>}"
    echo "nvdisasm=${NVDISASM:-<none>}"
    echo "MODULAR_NVPTX_COMPILER_PATH=${MODULAR_NVPTX_COMPILER_PATH:-<unset>}"
    [ -n "$PTXAS" ] && "$PTXAS" --version 2>&1 | tail -2
    [ -n "$CUOBJDUMP" ] && "$CUOBJDUMP" --version 2>&1 | tail -2
} > "$OUT/tools.txt" 2>&1
HAVE_TOOLKIT=0
[ -n "$PTXAS" ] && [ -n "$CUOBJDUMP" ] && HAVE_TOOLKIT=1
note "cuda_toolkit=$HAVE_TOOLKIT"

IDENT="-D MOJOLEARN_NUMERIC_IDENTICAL=1"
TRIAL="-D MOJOLEARN_KNN_SELECT_TRIAL=1"

# ---- binding: the trial binding, built the way the gate builds it --------
# `env`, not a prefix assignment: dash does not reliably pass prefix
# assignments through a shell function (the gate says the same).
run build-binding-trial fence "$BUILD_TIMEOUT" env MOJOLEARN_NUMERIC_MODE=identical MOJOLEARN_COMPILE_JOBS="$JOBS" \
    MOJOLEARN_BUILD_EXTRA_DEFINES="$TRIAL ${MOJOLEARN_KNN_SELECTION_EXTRA_DEFINES:-}" \
    sh bindings/build.sh
SO=python/mojolearn/identical/_mojolearn.so
if [ -f "$SO" ]; then
    sha256sum "$SO" > "$OUT/binding.sha256"
    # Attempt: NVPTX text blobs inside the shared library. Recorded either way.
    run extract-ptx-blobs pixi run python3 tools/knn_selector_kernel_stats.py --extract-ptx "$SO" "$OUT/blobs"
    _nblob=$(ls "$OUT/blobs" 2>/dev/null | wc -l | tr -d ' ')
    note "binding_ptx_blobs=$_nblob"
    if [ -n "$CUOBJDUMP" ]; then
        "$CUOBJDUMP" --list-ptx "$SO" > "$OUT/binding-cuobjdump-list.log" 2>&1
        note "binding_cuobjdump_list_exit=$?"
    fi
fi

# ---- driver: the documented compile_function / get_attribute route --------
# Generated here, not checked in: it is a measurement fixture that must
# track the kernel's parameter list, and the lane may not touch the kernel.
# The shipped instantiations are exactly what smallk_select_launch reaches
# with the build defaults (imported, not restated); the scanonly1 variant
# is the timing-only CAP=1 / K=1 form from the trial hook.
DUMPDIR="$OUT/dumps"
gen_path_fn() {
    # gen_path_fn <label>: two functions returning the dump paths.
    printf 'def ptx_path_%s() -> Path:\n    return Path("%s/%s.ptx")\n\n' "$1" "$DUMPDIR" "$1"
    printf 'def sass_path_%s() -> Path:\n    return Path("%s/%s.sass")\n\n' "$1" "$DUMPDIR" "$1"
}
gen_stat() {
    # gen_stat <mode files|stdout> <label> <cap> <k> "<remaining kernel parameters>"
    _mode=$1; _l=$2; _c=$3; _k=$4; _rest=$5
    if [ "$_mode" = files ]; then
        _dump="dump_asm=ptx_path_${_l}"
        [ "$HAVE_TOOLKIT" = 1 ] && _dump="$_dump, _dump_sass=sass_path_${_l}, _ptxas_info_verbose=True"
    else
        _dump="dump_asm=True"
        [ "$HAVE_TOOLKIT" = 1 ] && _dump="$_dump, _dump_sass=True, _ptxas_info_verbose=True"
    fi
    cat <<EOF
def stat_${_l}(ctx: DeviceContext) raises:
    comptime kern = smallk_bucket_kernel[${_c}, ${_k}, ${_rest}]
    print("KNN_KERNEL_STATS_BEGIN label=${_l} cap=${_c} k=${_k}")
    var f = ctx.compile_function[kern, ${_dump}]()
    print(
        "KNN_KERNEL_STATS label=${_l} cap=${_c} k=${_k}",
        " regs=", f.get_attribute(Attribute.NUM_REGS),
        " local=", f.get_attribute(Attribute.LOCAL_SIZE_BYTES),
        " shared=", f.get_attribute(Attribute.SHARED_SIZE_BYTES),
        " const=", f.get_attribute(Attribute.CONST_SIZE_BYTES),
        " max_threads=", f.get_attribute(Attribute.MAX_THREADS_PER_BLOCK),
        " blocks_per_sm_256=", f.occupancy_max_active_blocks_per_multiprocessor(SMALLK_BLOCK, 0),
        sep="",
    )


EOF
}
gen_call() {
    printf '    try:\n        stat_%s(ctx)\n    except e:\n        print("KNN_KERNEL_STATS_ERROR label=%s error=", e, sep="")\n' "$1" "$1"
}
DEFAULTS="SMALLK_UNIFORM_TRIP_DEFAULT, SMALLK_HEAD_BOUND_DEFAULT, False, SMALLK_WARPBOUND_DEFAULT, SMALLK_PHASE_FULL, SMALLK_DEFERRED_DEFAULT"
SCANONLY="True, False, False, False, SMALLK_PHASE_SKIPRANK, False"
LABELS="cap16_k0_generic cap16_k10 cap16_k15 cap1_k1_scanonly1 cap32_k0_generic cap64_k0_generic"
gen_driver() {
    # gen_driver <mode> > file
    _mode=$1
    cat <<'EOF'
# Generated by tools/knn_selector_kernel_stats.sh (DEVIATION 2519). RUNS ON
# THE POD. Compiles the shipped small-k selector instantiations for this
# device, dumps PTX (and SASS when the CUDA toolkit is present) and prints
# the runtime's own resource attributes. Launches nothing.
from std.pathlib import Path
from max.gpu.host import Attribute, DeviceContext
from neighbors.checks.select_smallk_identical_candidate import (
    smallk_bucket_kernel,
    SMALLK_BLOCK,
    SMALLK_PHASE_FULL,
    SMALLK_PHASE_SKIPRANK,
    SMALLK_UNIFORM_TRIP_DEFAULT,
    SMALLK_HEAD_BOUND_DEFAULT,
    SMALLK_WARPBOUND_DEFAULT,
    SMALLK_DEFERRED_DEFAULT,
)


EOF
    if [ "$_mode" = files ]; then
        for _l in $LABELS; do gen_path_fn "$_l"; done
    fi
    gen_stat "$_mode" cap16_k0_generic 16 0 "$DEFAULTS"
    gen_stat "$_mode" cap16_k10 16 10 "$DEFAULTS"
    gen_stat "$_mode" cap16_k15 16 15 "$DEFAULTS"
    gen_stat "$_mode" cap1_k1_scanonly1 1 1 "$SCANONLY"
    gen_stat "$_mode" cap32_k0_generic 32 0 "$DEFAULTS"
    gen_stat "$_mode" cap64_k0_generic 64 0 "$DEFAULTS"
    cat <<'EOF'
def main() raises:
    var ctx = DeviceContext()
    print("KNN_KERNEL_STATS_DEVICE name=", ctx.name(), sep="")
EOF
    for _l in $LABELS; do gen_call "$_l"; done
    printf '    print("KNN_KERNEL_STATS_DONE")\n'
}

gen_driver files > "$OUT/driver_files.mojo"
gen_driver stdout > "$OUT/driver_stdout.mojo"
DRIVER_BIN=""
# shellcheck disable=SC2086
if run build-driver-files fence "$BUILD_TIMEOUT" pixi run mojo build -j "$JOBS" -I . $IDENT $TRIAL \
        "$OUT/driver_files.mojo" -o "$OUT/bin/driver_files"; then
    DRIVER_BIN="$OUT/bin/driver_files"
    note "driver_variant=files"
else
    # The path-function form did not build (its Variant member is documented
    # but untried here); the Bool form dumps to stdout and the parser splits.
    rc=0
    # shellcheck disable=SC2086
    if run build-driver-stdout fence "$BUILD_TIMEOUT" pixi run mojo build -j "$JOBS" -I . $IDENT $TRIAL \
            "$OUT/driver_stdout.mojo" -o "$OUT/bin/driver_stdout"; then
        DRIVER_BIN="$OUT/bin/driver_stdout"
        note "driver_variant=stdout"
    else
        note "driver_variant=none (both variants failed to build; see build-driver-*.log)"
    fi
fi
if [ -n "$DRIVER_BIN" ]; then
    run driver fence 600 "$DRIVER_BIN"
    # When the dump went to stdout, cut each instantiation's PTX / SASS out
    # of the log into dumps/<label>.* so the assemble step sees files.
    run split-driver-log pixi run python3 tools/knn_selector_kernel_stats.py --split-driver-log "$OUT"
    grep -E '^KNN_KERNEL_STATS(_ERROR|_DEVICE|_DONE)? ' "$OUT/driver.log" > "$OUT/driver_lines.txt" 2>/dev/null
fi

# ---- emit-asm: the repo-proven sidecar route (cross-check / fallback) ----
# shellcheck disable=SC2086
run emit-asm-default fence "$BUILD_TIMEOUT" pixi run mojo build -j "$JOBS" --emit asm -I . $IDENT \
    bench/knn_reference_price_main.mojo -o "$OUT/emit/default"
# shellcheck disable=SC2086
run emit-asm-generic fence "$BUILD_TIMEOUT" pixi run mojo build -j "$JOBS" --emit asm -I . $IDENT \
    -D MOJOLEARN_KNN_IDENTICAL_GENERIC_K=1 \
    bench/knn_reference_price_main.mojo -o "$OUT/emit/generic"
# Sidecars are `<out>_<module>_<hash>.ptx`; keep the selector's, list all.
ls -la "$OUT/emit" > "$OUT/emit/listing.txt" 2>&1
for f in "$OUT"/emit/*.ptx; do
    [ -f "$f" ] || continue
    case "$(basename "$f")" in
        *select_smallk_identical_candidate*) ;;
        *) rm -f "$f" ;;
    esac
done
rm -f "$OUT/emit/default" "$OUT/emit/generic"
# Elimination: a sidecar whose hash appears in both builds is a K=0 bucket;
# one only in the default build is a k10 / k15 specialization.
{
    echo "file	in_default	in_generic	entry	inferred"
    for f in "$OUT"/emit/default_*.ptx; do
        [ -f "$f" ] || continue
        h=$(basename "$f" .ptx | sed 's/^default_//')
        g="$OUT/emit/generic_$h.ptx"
        entry=$(grep -m1 -oE '\.entry[[:space:]]+[^[:space:](]+' "$f" | awk '{print $2}')
        if [ -f "$g" ]; then inf="generic-bucket-K0"; else inf="k10-or-k15-specialization"; fi
        printf '%s\t1\t%s\t%s\t%s\n' "$(basename "$f")" "$([ -f "$g" ] && echo 1 || echo 0)" "$entry" "$inf"
    done
    for f in "$OUT"/emit/generic_*.ptx; do
        [ -f "$f" ] || continue
        h=$(basename "$f" .ptx | sed 's/^generic_//')
        [ -f "$OUT/emit/default_$h.ptx" ] && continue
        entry=$(grep -m1 -oE '\.entry[[:space:]]+[^[:space:](]+' "$f" | awk '{print $2}')
        printf '%s\t0\t1\t%s\tgeneric-only\n' "$(basename "$f")" "$entry"
    done
} > "$OUT/emit/manifest.tsv"
# The emit sidecars become stats rows of their own (label = file stem), so
# the parser reports them beside the driver's labeled rows.
for f in "$OUT"/emit/default_*.ptx "$OUT"/emit/generic_*.ptx; do
    [ -f "$f" ] || continue
    cp "$f" "$OUT/dumps/emit_$(basename "$f")"
done

# ---- assemble: ptxas --verbose, cuobjdump resource usage and SASS ---------
if [ "$HAVE_TOOLKIT" = 1 ]; then
    for p in "$OUT"/dumps/*.ptx; do
        [ -f "$p" ] || continue
        stem=${p%.ptx}
        tgt=$(grep -m1 -oE '^[[:space:]]*\.target[[:space:]]+[^[:space:]]+' "$p" | awk '{print $2}')
        [ -n "$tgt" ] || tgt=sm_90a
        "$PTXAS" --verbose --gpu-name "$tgt" -o "$stem.cubin" "$p" > "$stem.ptxas.log" 2>&1
        echo "ptxas_exit=$? target=$tgt" >> "$stem.ptxas.log"
        if [ -s "$stem.cubin" ]; then
            "$CUOBJDUMP" --dump-resource-usage "$stem.cubin" > "$stem.resources.log" 2>&1
            if [ ! -s "$stem.sass" ]; then
                "$CUOBJDUMP" --dump-sass "$stem.cubin" > "$stem.sass" 2>&1
                # nvdisasm is the second reader of the same cubin; used only
                # when cuobjdump produced nothing.
                [ -s "$stem.sass" ] || [ -z "$NVDISASM" ] || "$NVDISASM" "$stem.cubin" > "$stem.sass" 2>&1
            fi
        fi
        rm -f "$stem.cubin"
    done
    printf 'assemble\t0\t0s\n' >> "$OUT/status.tsv"
else
    note "assemble=skipped (no ptxas + cuobjdump on the box; spill counts and SASS LDL/STL are OWED; the driver's runtime attributes and the PTX ld/st.local counts stand)"
    printf 'assemble\t2\t0s\n' >> "$OUT/status.tsv"
fi

# ---- stats.tsv --------------------------------------------------------------
run stats pixi run python3 tools/knn_selector_kernel_stats.py --out "$OUT" --json "$OUT/stats.json"

# ---- fence: nothing binary comes home, dumps stay small ------------------
rm -rf "$OUT/bin"
for f in "$OUT"/dumps/*.ptx "$OUT"/dumps/*.sass "$OUT"/emit/*.ptx "$OUT"/blobs/*.ptx "$OUT"/driver.log.raw; do
    [ -f "$f" ] || continue
    if [ "$(wc -c < "$f")" -gt 262144 ]; then gzip -9 "$f"; fi
done
# The stdout-form driver log carries every dump; keep the lines the parser
# needs uncompressed and gzip the rest.
if [ -f "$OUT/driver.log" ] && [ "$(wc -c < "$OUT/driver.log")" -gt 262144 ]; then
    gzip -9 "$OUT/driver.log"
fi
total_kb=$(du -sk "$OUT" | cut -f1)
note "output_kb=$total_kb"
if [ "$total_kb" -gt "$SIZE_FENCE_KB" ]; then
    # Largest first until under the fence; the tsv, txt and logs are small.
    for f in $(ls -S "$OUT"/dumps/* "$OUT"/emit/* "$OUT"/blobs/* 2>/dev/null); do
        [ "$(du -sk "$OUT" | cut -f1)" -le "$SIZE_FENCE_KB" ] && break
        case "$f" in *.tsv|*.txt|*.log) continue ;; esac
        rm -f "$f"
        note "size_fence_removed=$(basename "$f")"
    done
fi
note "finished=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
cat "$OUT/stats.txt"
[ -f "$OUT/stats.tsv" ] && cat "$OUT/stats.tsv"
cat "$OUT/status.tsv"
exit "$rc"
