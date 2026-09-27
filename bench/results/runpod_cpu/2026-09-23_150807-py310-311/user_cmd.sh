# Python 3.10 and 3.11 test run against the published 0.8.15 Linux wheel.
# Runs on the RunPod CPU pod under tools/runpod_cpu_leg.sh; $LEG_OUT comes home.
# Four arms per version, each in a copy of the shipped checkout so that the
# repo-relative paths the tests use (parents[3]) resolve:
#   A  repo-wheel: python/mojolearn is the wheel's package, tests/ from main
#   B  repo-main:  python/mojolearn is main's source, the wheel's binaries under it
#   C  tools tests in repo-main (bindings present)
#   D  gate-style module files in repo-wheel
set -u
R=/root/mojolearn
W=/root/pyver
mkdir -p "$W"
unset PYTHONPATH
export MOJOLEARN_NUMERIC_MODE=identical
T0=$(date +%s)
echo "start $(date -u +%FT%TZ)" > "$LEG_OUT/timeline.txt"
nproc > "$LEG_OUT/nproc.txt"

curl -LsSf https://astral.sh/uv/install.sh | sh > "$LEG_OUT/uv_install.log" 2>&1
export PATH="$HOME/.local/bin:$PATH"
uv --version >> "$LEG_OUT/uv_install.log" 2>&1 || { echo "uv missing" >> "$LEG_OUT/timeline.txt"; exit 2; }

MODULES="test_arima_surface test_cholesky_surface test_embedding_surface test_gmm_sample test_gp_optimizer test_gp_sample_y test_gp_surface test_hdbscan_surface test_ivf_surface test_kernel_methods_surface test_kmeans_metric_surface test_kmeans_predict test_kmeans_transform test_linalg_identity test_mamba_surface test_mixture_surface test_resample_surface test_rf_leaf_budget test_samba_surface test_spectral_predict test_svr_surface test_training_primitives_surface test_training_surface test_transductive_predict test_transformer_hd128 test_transformer_surface"

make_repo() {  # dest: the shipped checkout without the pixi envs
    mkdir -p "$1"
    tar -C "$R" --exclude=./.pixi --exclude=./.git --exclude='__pycache__' -cf - . | tar -C "$1" -xf -
}

one_version() {
    V=$1; tag=py${V/./}
    O="$LEG_OUT/$tag"; mkdir -p "$O/modules"
    t=$(date +%s)
    uv venv --python "$V" "$W/venv-$tag" > "$O/venv.log" 2>&1 || { echo "venv failed" >> "$O/venv.log"; return; }
    PY="$W/venv-$tag/bin/python"
    uv pip install --python "$PY" "mojolearn==0.8.15" numpy pytest pytest-timeout > "$O/pip_install.log" 2>&1
    echo "install_rc=$? seconds=$(( $(date +%s) - t ))" >> "$O/pip_install.log"
    "$PY" -c 'import sys, platform, numpy, mojolearn, os, importlib.metadata as m; print("python", sys.version.split()[0], platform.python_implementation()); print("numpy", numpy.__version__); print("mojolearn", m.version("mojolearn"), os.path.dirname(mojolearn.__file__))' > "$O/env.txt" 2>&1
    uv pip freeze --python "$PY" >> "$O/env.txt" 2>&1
    SP=$("$PY" -c 'import mojolearn, os; print(os.path.dirname(mojolearn.__file__))' 2>/dev/null)
    [ -n "$SP" ] || { echo "mojolearn did not import" >> "$O/env.txt"; return; }
    (cd "$SP" && find . -type f ! -path '*__pycache__*' | sort) > "$O/wheel_files.txt"

    RW="$W/repo-wheel-$tag"; RM="$W/repo-main-$tag"
    make_repo "$RW"; make_repo "$RM"
    # A: the wheel's package, main's tests
    rm -rf "$RW/python/mojolearn"; cp -r "$SP" "$RW/python/mojolearn"
    rm -rf "$RW/python/mojolearn/tests"; cp -r "$R/python/mojolearn/tests" "$RW/python/mojolearn/tests"
    # B: main's source, plus every wheel file the source tree does not carry (binaries, generated modules)
    (cd "$SP" && find . -type f ! -path './tests/*' ! -path '*__pycache__*' -print0 | while IFS= read -r -d '' f; do
        case "$f" in
            *.py) [ -e "$RM/python/mojolearn/$f" ] || { mkdir -p "$RM/python/mojolearn/$(dirname "$f")"; cp "$f" "$RM/python/mojolearn/$f"; echo "$f" >> "$O/overlay_added_py.txt"; } ;;
            *) mkdir -p "$RM/python/mojolearn/$(dirname "$f")"; cp "$f" "$RM/python/mojolearn/$f" ;;
        esac
    done)
    diff -rq "$RW/python/mojolearn" "$RM/python/mojolearn" | grep -v '/tests' > "$O/wheel_vs_main.txt" 2>&1
    echo "setup_seconds=$(( $(date +%s) - t ))" >> "$O/timeline.txt"

    ta=$(date +%s)
    ( cd "$RW/python" && timeout 1500 "$PY" -m pytest -q -rfEs --tb=short -p no:cacheprovider --timeout=300 --basetemp="$W/tmp-wheel-$tag" mojolearn/tests > "$O/pytest_wheel.log" 2>&1; echo $? > "$O/pytest_wheel.exit" ) &
    pa=$!
    ( cd "$RM/python" && timeout 1500 "$PY" -m pytest -q -rfEs --tb=short -p no:cacheprovider --timeout=300 --basetemp="$W/tmp-main-$tag" mojolearn/tests > "$O/pytest_main.log" 2>&1; echo $? > "$O/pytest_main.exit" ) &
    pb=$!
    wait $pa; wait $pb
    echo "pytest_arms_seconds=$(( $(date +%s) - ta ))" >> "$O/timeline.txt"

    tc=$(date +%s)
    ( cd "$RM" && timeout 900 env PYTHONPATH="$RM/python" "$PY" -m pytest -q -rfEs --tb=short -p no:cacheprovider --timeout=120 --basetemp="$W/tmp-tools-$tag" tools bench/model/tests > "$O/pytest_tools.log" 2>&1; echo $? > "$O/pytest_tools.exit" )
    echo "tools_seconds=$(( $(date +%s) - tc ))" >> "$O/timeline.txt"

    td=$(date +%s)
    ( cd "$RW/python" && for m in $MODULES; do
        s=$(date +%s)
        timeout 90 "$PY" -m "mojolearn.tests.$m" > "$O/modules/$m.log" 2>&1; rc=$?
        printf '%s\t%s\t%s\n' "$m" "$rc" "$(( $(date +%s) - s ))" >> "$O/modules.tsv"
        [ $(( $(date +%s) - td )) -lt 600 ] || { echo "modules budget exhausted after $m" >> "$O/modules.tsv"; break; }
      done )
    echo "modules_seconds=$(( $(date +%s) - td ))" >> "$O/timeline.txt"
    echo "version_total_seconds=$(( $(date +%s) - t ))" >> "$O/timeline.txt"
}

one_version 3.10 &
p1=$!
one_version 3.11 &
p2=$!
wait $p1; wait $p2
echo "end $(date -u +%FT%TZ) total_seconds=$(( $(date +%s) - T0 ))" >> "$LEG_OUT/timeline.txt"
