# Second leg: the tools arm (never ran on the first pod: collection aborted on
# both versions) and the files fixed on lane/python-310-311, on 3.10 and 3.11
# against the 0.8.15 wheel's binaries under main's source (the 0.8.16 shape).
set -u
R=/root/mojolearn
W=/root/pyver
mkdir -p "$W"
unset PYTHONPATH
export MOJOLEARN_NUMERIC_MODE=identical
T0=$(date +%s)
echo "start $(date -u +%FT%TZ)" > "$LEG_OUT/timeline.txt"
curl -LsSf https://astral.sh/uv/install.sh | sh > "$LEG_OUT/uv_install.log" 2>&1
export PATH="$HOME/.local/bin:$PATH"
uv --version >> "$LEG_OUT/uv_install.log" 2>&1 || { echo "uv missing" >> "$LEG_OUT/timeline.txt"; exit 2; }

make_repo() { mkdir -p "$1"; tar -C "$R" --exclude=./.pixi --exclude=./.git --exclude='__pycache__' -cf - . | tar -C "$1" -xf -; }

one_version() {
    V=$1; tag=py${V/./}
    O="$LEG_OUT/$tag"; mkdir -p "$O"
    t=$(date +%s)
    uv venv --python "$V" "$W/venv-$tag" > "$O/venv.log" 2>&1 || { echo "venv failed" >> "$O/venv.log"; return; }
    PY="$W/venv-$tag/bin/python"
    uv pip install --python "$PY" "mojolearn==0.8.15" numpy pytest pytest-timeout > "$O/pip_install.log" 2>&1
    "$PY" -c 'import sys, platform, numpy, importlib.metadata as m; print("python", sys.version.split()[0], platform.python_implementation(), sys.executable); print("numpy", numpy.__version__); print("mojolearn", m.version("mojolearn"))' > "$O/env.txt" 2>&1
    SP=$("$PY" -c 'import mojolearn, os; print(os.path.dirname(mojolearn.__file__))' 2>/dev/null)
    [ -n "$SP" ] || { echo "mojolearn did not import" >> "$O/env.txt"; return; }
    RM="$W/repo-main-$tag"; make_repo "$RM"
    (cd "$SP" && find . -type f ! -path './tests/*' ! -path '*__pycache__*' -print0 | while IFS= read -r -d '' f; do
        case "$f" in
            *.py) [ -e "$RM/python/mojolearn/$f" ] || { mkdir -p "$RM/python/mojolearn/$(dirname "$f")"; cp "$f" "$RM/python/mojolearn/$f"; } ;;
            *) mkdir -p "$RM/python/mojolearn/$(dirname "$f")"; cp "$f" "$RM/python/mojolearn/$f" ;;
        esac
    done)
    echo "setup_seconds=$(( $(date +%s) - t ))" >> "$O/timeline.txt"

    # the crossvendor subprocess question: what does a child of this interpreter see?
    ( cd "$RM/python" && "$PY" - <<'PYEOF' > "$O/child_interpreter.txt" 2>&1
import subprocess, sys, tempfile, os
print("parent", sys.executable, sys.prefix, sys.version.split()[0])
d = tempfile.mkdtemp()
r = subprocess.run([sys.executable, "-c", "import sys; print('child', sys.executable, sys.prefix); import numpy; print('numpy', numpy.__version__, numpy.__file__)"], cwd=d, capture_output=True, text=True)
print("rc", r.returncode); print(r.stdout); print(r.stderr)
print("env PYTHON*:", {k: v for k, v in os.environ.items() if k.startswith("PYTHON") or k in ("VIRTUAL_ENV", "PATH")})
PYEOF
    )

    tb=$(date +%s)
    ( cd "$RM/python" && timeout 900 "$PY" -m pytest -q -rfEs --tb=short -p no:cacheprovider --timeout=300 --basetemp="$W/tmp-fixed-$tag" \
        mojolearn/tests/test_native_nonzero.py mojolearn/tests/test_native_convert.py mojolearn/tests/test_native_helpers.py mojolearn/tests/test_crossvendor_coverage.py mojolearn/tests/test_safetensors_f16_py310.py mojolearn/tests/test_models_loader.py \
        > "$O/pytest_fixed.log" 2>&1; echo $? > "$O/pytest_fixed.exit" )
    echo "fixed_seconds=$(( $(date +%s) - tb ))" >> "$O/timeline.txt"

    tc=$(date +%s)
    ( cd "$RM" && timeout 1500 env PYTHONPATH="$RM/python" "$PY" -m pytest -q -rfEs --tb=short -p no:cacheprovider --timeout=120 --basetemp="$W/tmp-tools-$tag" tools bench/model/tests > "$O/pytest_tools.log" 2>&1; echo $? > "$O/pytest_tools.exit" )
    echo "tools_seconds=$(( $(date +%s) - tc ))" >> "$O/timeline.txt"
    echo "version_total_seconds=$(( $(date +%s) - t ))" >> "$O/timeline.txt"
}
one_version 3.10 &
p1=$!
one_version 3.11 &
p2=$!
wait $p1; wait $p2
echo "end $(date -u +%FT%TZ) total_seconds=$(( $(date +%s) - T0 ))" >> "$LEG_OUT/timeline.txt"
