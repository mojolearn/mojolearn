# tools/lowbit_mma_speed/box_env.sh -- sourced by the lane's job scripts.
# What a box needs before a build, and one line that says what the box is.
if command -v nvidia-smi > /dev/null 2>&1 && nvidia-smi -L > /dev/null 2>&1; then
    # tools/lowbit_mma_leg.sh's lines: MAX's bundled assembler needs driver
    # 580; an older driver uses the box's own, at build and at run time.
    driver_major=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1 | cut -d. -f1)
    case "$driver_major" in
        ''|*[!0-9]*) ;;
        *) if [ "$driver_major" -lt 580 ] && [ -x /usr/local/cuda/bin/ptxas ]; then
               export MODULAR_NVPTX_COMPILER_PATH=${MODULAR_NVPTX_COMPILER_PATH:-/usr/local/cuda/bin/ptxas}
           fi ;;
    esac
fi
box_describe() {
    if [ "$(uname -s)" = Darwin ]; then
        echo "machine=$(sysctl -n machdep.cpu.brand_string 2>/dev/null) macOS $(sw_vers -productVersion 2>/dev/null)"
    elif command -v nvidia-smi > /dev/null 2>&1 && nvidia-smi -L > /dev/null 2>&1; then
        echo "machine=$(nvidia-smi --query-gpu=name,driver_version --format=csv,noheader 2>/dev/null | head -1)"
    else
        echo "machine=$( (rocm-smi --showproductname 2>/dev/null || amd-smi static --asic 2>/dev/null) | grep -i -m1 -E 'card series|product name|market' | tr -s ' ')"
    fi
}
