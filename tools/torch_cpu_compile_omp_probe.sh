#!/bin/sh
# Why torch.compile on the CPU aborts in the bench board's venv on macOS
# ("OMP: Error #15: Initializing libomp.dylib, but found libomp.dylib already
# initialized"): the torch-cpu-compile-* arms of neural/*-infer on the
# M3 Ultra board, 2026-09-29. Light: one tiny compiled function, then the
# board's own worker at the small shape. Prints every OpenMP runtime each
# process loads (DYLD_PRINT_LIBRARIES) and inductor's link line for the
# compiled kernel. Run by the apple steward (speed kind) from the repo root.
#
#   sh tools/torch_cpu_compile_omp_probe.sh [python]
set -u
PY="${1:-$HOME/bench-board-cache/venv/bin/python}"
D="$(mktemp -d)"
echo "== python: $PY"
"$PY" -c "import torch, sklearn; print('torch', torch.__version__, 'sklearn', sklearn.__version__)" 2>&1 | tail -1
echo "== OpenMP runtimes in the venv"
"$PY" -c "import site; print(site.getsitepackages()[0])" > "$D/sp" 2>/dev/null
find "$(cat "$D/sp")" -name "libomp*.dylib" -o -name "libiomp*.dylib" -o -name "libgomp*.dylib" 2>/dev/null
echo "== bare torch.compile on the CPU"
cat > "$D/bare.py" <<'EOF'
import torch
f = torch.compile(lambda x: (x.sin() * 2.0).sum(1))
print("bare compile result", float(f(torch.randn(256, 256)).sum()))
EOF
TORCH_LOGS=output_code DYLD_PRINT_LIBRARIES=1 "$PY" "$D/bare.py" > "$D/bare.log" 2>&1
echo "exit $?"
grep -i -E "omp|Error|bare compile" "$D/bare.log" | sort | uniq -c | head -40
grep -o -E "(clang|c\+\+|g\+\+)[^']*-o [^ ]*\.so[^']*" "$D/bare.log" | head -2 | cut -c1-2000
echo "== the board's worker: neural/mamba2-infer, shape small, torch-cpu-compile-fp32"
DYLD_PRINT_LIBRARIES=1 "$PY" tools/bench_board_neural.py race --lane mamba2-infer --shape small \
    --arms torch-cpu-compile-fp32 --rounds 1 --out "$D/out" --work "$D/work" > "$D/race.log" 2>&1
echo "exit $?"
grep -E "^NEURAL" "$D/race.log" | cut -c1-300
for f in "$D"/out/*.log; do
    echo "-- $f"
    grep -i -E "omp|Error" "$f" | sort | uniq -c | head -40
done
echo "== the inductor build of the worker (cpp_builder, the link flags)"
"$PY" - <<'EOF' 2>&1 | tail -20
import torch._inductor.cpp_builder as cb
try:
    opts = cb.CppTorchOptions(vec_isa=cb.pick_vec_isa() if hasattr(cb, "pick_vec_isa") else None)
except Exception as exc:
    try:
        opts = cb.CppTorchOptions()
    except Exception as exc2:
        print("CppTorchOptions failed:", exc, exc2)
        raise SystemExit(0)
for k in ("get_libraries_dirs", "get_libraries", "get_include_dirs", "get_ldflags", "get_cflags"):
    fn = getattr(opts, k, None)
    if fn:
        print(k, fn())
EOF
rm -rf "$D"
exit 0
