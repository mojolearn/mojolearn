# Third leg: 3.14 control arm (full python suite + tools, main source over the
# wheel binaries) and, on 3.10 and 3.11, the nonzero fix plus a probe of the
# crossvendor child interpreter that could not import numpy on 3.11.
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

setup() {  # V tag -> PY, RM
    V=$1; tag=$2; O="$LEG_OUT/$tag"; mkdir -p "$O"
    uv venv --python "$V" "$W/venv-$tag" > "$O/venv.log" 2>&1 || { echo "venv failed" >> "$O/venv.log"; return 1; }
    PY="$W/venv-$tag/bin/python"
    uv pip install --python "$PY" "mojolearn==0.8.15" numpy pytest pytest-timeout > "$O/pip_install.log" 2>&1
    "$PY" -c 'import sys, platform, numpy, importlib.metadata as m; print("python", sys.version.split()[0], platform.python_implementation(), sys.executable); print("numpy", numpy.__version__); print("mojolearn", m.version("mojolearn"))' > "$O/env.txt" 2>&1
    SP=$("$PY" -c 'import mojolearn, os; print(os.path.dirname(mojolearn.__file__))' 2>/dev/null)
    [ -n "$SP" ] || { echo "mojolearn did not import" >> "$O/env.txt"; return 1; }
    RM="$W/repo-main-$tag"; make_repo "$RM"
    (cd "$SP" && find . -type f ! -path './tests/*' ! -path '*__pycache__*' -print0 | while IFS= read -r -d '' f; do
        case "$f" in
            *.py) [ -e "$RM/python/mojolearn/$f" ] || { mkdir -p "$RM/python/mojolearn/$(dirname "$f")"; cp "$f" "$RM/python/mojolearn/$f"; } ;;
            *) mkdir -p "$RM/python/mojolearn/$(dirname "$f")"; cp "$f" "$RM/python/mojolearn/$f" ;;
        esac
    done)
    cat > "$RM/python/mojolearn/tests/test_zz_probe_child.py" <<'PYEOF'
import json, os, subprocess, sys
from pathlib import Path
import mojolearn  # noqa: F401  the real test imports the package first


def test_probe_child(tmp_path):
    root = Path(__file__).resolve().parents[3]
    env_now = {k: v for k, v in os.environ.items() if k.startswith(("PYTHON", "MOJOLEARN", "LD_", "VIRTUAL"))}
    (tmp_path / "tools").mkdir()
    probe = tmp_path / "tools" / "probe.py"
    probe.write_text("import sys, os\nprint(json.dumps if False else '')\nprint('executable', sys.executable)\nprint('prefix', sys.prefix, sys.base_prefix)\nprint('flags', sys.flags)\nprint('path', sys.path)\nprint('env', {k: v for k, v in os.environ.items() if k.startswith(('PYTHON', 'VIRTUAL'))})\nimport numpy\nprint('numpy', numpy.__version__)\n")
    r = subprocess.run([sys.executable, str(probe)], cwd=tmp_path, capture_output=True, text=True)
    r2 = subprocess.run([sys.executable, "-c", "import numpy; print('c-numpy', numpy.__file__)"], cwd=tmp_path, capture_output=True, text=True)
    r3 = subprocess.run([sys.executable, str(probe)], cwd=tmp_path, capture_output=True, text=True, env={"PATH": os.environ["PATH"]})
    out = dict(parent_executable=sys.executable, parent_prefix=sys.prefix, parent_env=env_now, parent_path=sys.path,
               script=dict(rc=r.returncode, out=r.stdout, err=r.stderr[-2000:]),
               dash_c=dict(rc=r2.returncode, out=r2.stdout, err=r2.stderr[-2000:]),
               script_clean_env=dict(rc=r3.returncode, out=r3.stdout, err=r3.stderr[-2000:]))
    Path(os.environ["PROBE_OUT"]).write_text(json.dumps(out, indent=1, default=str))
    assert r.returncode == 0, r.stderr
PYEOF
    return 0
}

small() {  # 3.10 / 3.11: the fixed file, the crossvendor test and the probe
    V=$1; tag=$2; O="$LEG_OUT/$tag"
    setup "$V" "$tag" || return
    ( cd "$RM/python" && PROBE_OUT="$O/probe_child.json" timeout 600 "$PY" -m pytest -q -rfEs --tb=short -p no:cacheprovider --timeout=120 --basetemp="$W/tmp-$tag" \
        mojolearn/tests/test_native_nonzero.py mojolearn/tests/test_crossvendor_coverage.py mojolearn/tests/test_zz_probe_child.py \
        > "$O/pytest_fixed.log" 2>&1; echo $? > "$O/pytest_fixed.exit" )
}

control() {  # 3.14: everything, as a reference for what is CPU-only-install shaped
    V=$1; tag=$2; O="$LEG_OUT/$tag"
    setup "$V" "$tag" || return
    rm -f "$RM/python/mojolearn/tests/test_zz_probe_child.py"
    ta=$(date +%s)
    ( cd "$RM/python" && timeout 1500 "$PY" -m pytest -q -rfEs --tb=short -p no:cacheprovider --timeout=300 --basetemp="$W/tmp-main-$tag" mojolearn/tests > "$O/pytest_main.log" 2>&1; echo $? > "$O/pytest_main.exit" )
    echo "main_seconds=$(( $(date +%s) - ta ))" >> "$O/timeline.txt"
    tc=$(date +%s)
    ( cd "$RM" && timeout 1200 env PYTHONPATH="$RM/python" "$PY" -m pytest -q -rfEs --tb=short -p no:cacheprovider --timeout=120 --basetemp="$W/tmp-tools-$tag" tools bench/model/tests > "$O/pytest_tools.log" 2>&1; echo $? > "$O/pytest_tools.exit" )
    echo "tools_seconds=$(( $(date +%s) - tc ))" >> "$O/timeline.txt"
}

small 3.10 py310 &
p1=$!
small 3.11 py311 &
p2=$!
control 3.14 py314 &
p3=$!
wait $p1; wait $p2; wait $p3
echo "end $(date -u +%FT%TZ) total_seconds=$(( $(date +%s) - T0 ))" >> "$LEG_OUT/timeline.txt"
