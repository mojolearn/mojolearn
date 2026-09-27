# Fourth leg: every fix on lane/python-310-311, on 3.10 and 3.11 against the
# 0.8.15 wheel's binaries under main's source; the full tools arm; the slow
# verifier test alone with a long timeout; and a probe of the 3.11 child
# interpreter that came up without numpy under pytest on this image.
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

{ echo "== which python3.10 / python3.11 on the image"; for p in /usr/bin/python3.10 /usr/bin/python3.11 /usr/bin/python3; do ls -l "$p"; readlink -f "$p"; file "$(readlink -f "$p")" 2>/dev/null; done
  echo "== dpkg"; dpkg -l 2>/dev/null | grep -E 'python3\.1[01]' | awk '{print $2, $3}'
  echo "== env of the leg command"; env | sort | cut -c1-200; } > "$LEG_OUT/image_python.txt" 2>&1

one_version() {
    V=$1; tag=py${V/./}; O="$LEG_OUT/$tag"; mkdir -p "$O"
    t=$(date +%s)
    uv venv --python "$V" "$W/venv-$tag" > "$O/venv.log" 2>&1 || { echo "venv failed" >> "$O/venv.log"; return; }
    PY="$W/venv-$tag/bin/python"
    ls -l "$W/venv-$tag/bin/" >> "$O/venv.log"; readlink -f "$PY" >> "$O/venv.log"; cat "$W/venv-$tag/pyvenv.cfg" >> "$O/venv.log"
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
    cat > "$RM/python/mojolearn/tests/test_zz_probe_child.py" <<'PYEOF'
import json, os, subprocess, sys
from pathlib import Path
import mojolearn  # noqa: F401


def test_probe_child(tmp_path):
    probe = tmp_path / "probe.py"
    probe.write_text("import os, sys\nprint('argv0', sys.argv[0])\nprint('executable', sys.executable)\nprint('exe', os.readlink('/proc/self/exe'))\nprint('cmdline', open('/proc/self/cmdline','rb').read())\nprint('prefix', sys.prefix)\nprint('path', sys.path)\nenv={k: v for k, v in os.environ.items()}\nprint('envkeys', sorted(env))\nprint('PYTHONPATH', repr(env.get('PYTHONPATH')), 'PYTHONHOME', repr(env.get('PYTHONHOME')))\ntry:\n    import numpy\n    print('numpy', numpy.__file__)\nexcept Exception as e:\n    print('numpy-missing', e)\n")
    r = subprocess.run([sys.executable, str(probe)], cwd=tmp_path, capture_output=True, text=True)
    out = {"parent_executable": sys.executable, "parent_exe": os.readlink("/proc/self/exe"), "parent_prefix": sys.prefix,
           "parent_env_keys": sorted(os.environ), "parent_PYTHONPATH": os.environ.get("PYTHONPATH"), "parent_sys_path": sys.path,
           "child": {"rc": r.returncode, "out": r.stdout, "err": r.stderr[-1500:]}}
    Path(os.environ["PROBE_OUT"]).write_text(json.dumps(out, indent=1, default=str))
PYEOF
    echo "setup_seconds=$(( $(date +%s) - t ))" >> "$O/timeline.txt"

    tb=$(date +%s)
    ( cd "$RM/python" && PROBE_OUT="$O/probe_child.json" timeout 900 "$PY" -m pytest -q -rfEs --tb=short -p no:cacheprovider --timeout=300 --basetemp="$W/tmp-fixed-$tag" \
        mojolearn/tests/test_native_nonzero.py mojolearn/tests/test_native_convert.py mojolearn/tests/test_native_helpers.py mojolearn/tests/test_device_array_refusal.py \
        mojolearn/tests/test_crossvendor_coverage.py mojolearn/tests/test_safetensors_f16_py310.py mojolearn/tests/test_models_loader.py mojolearn/tests/test_zz_probe_child.py \
        > "$O/pytest_fixed.log" 2>&1; echo $? > "$O/pytest_fixed.exit" )
    echo "fixed_seconds=$(( $(date +%s) - tb ))" >> "$O/timeline.txt"
    ts=$(date +%s)
    ( cd "$RM/python" && timeout 1000 "$PY" -m pytest -q -rfEs --tb=short -p no:cacheprovider --timeout=900 --basetemp="$W/tmp-slow-$tag" \
        "mojolearn/tests/test_verify_all.py::test_shipped_verifier_hashes_like_the_harness" --durations=3 > "$O/pytest_slow.log" 2>&1; echo $? > "$O/pytest_slow.exit" )
    echo "slow_seconds=$(( $(date +%s) - ts ))" >> "$O/timeline.txt"
    tc=$(date +%s)
    ( cd "$RM" && timeout 1200 env PYTHONPATH="$RM/python" "$PY" -m pytest -q -rfEs --tb=short -p no:cacheprovider --timeout=120 --basetemp="$W/tmp-tools-$tag" tools bench/model/tests > "$O/pytest_tools.log" 2>&1; echo $? > "$O/pytest_tools.exit" )
    echo "tools_seconds=$(( $(date +%s) - tc ))" >> "$O/timeline.txt"
    echo "version_total_seconds=$(( $(date +%s) - t ))" >> "$O/timeline.txt"
}
one_version 3.10 &
p1=$!
one_version 3.11 &
p2=$!
wait $p1; wait $p2
echo "end $(date -u +%FT%TZ) total_seconds=$(( $(date +%s) - T0 ))" >> "$LEG_OUT/timeline.txt"
