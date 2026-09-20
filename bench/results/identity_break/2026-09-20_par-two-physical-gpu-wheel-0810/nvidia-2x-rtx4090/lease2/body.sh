#!/bin/sh
# MOJOLEARN_GEMM_LEG_EXTRA body: `python -m mojolearn verify --par` on TWO
# PHYSICAL GPUs in one pod, against the PUBLISHED pip wheel mojolearn==0.8.10.
#
# The runner (tools/gemm_remote_leg.sh, gemm payload) ships a source archive
# and runs its own gates first. NOTHING BELOW USES THAT SOURCE TREE. The
# measurement runs from /root/parrun (outside /root/mojolearn) inside a fresh
# venv that holds only numpy and the wheel pip downloaded from PyPI, and the
# body refuses to measure unless `mojolearn.__file__` resolves inside the venv.
#
# Placeholders substituted per lease by make_body.sh (RunPod passes no env):
#   0.8.10   the wheel version
#   nvidia-lease2      this lease's distinct output slug
#   base ties hashed wide denormal denormal_ftz dupes odd negative  space separated fixtures this lease runs, in order
#        empty for every par-* lane, or a comma separated --lanes value
#   1     1 to run `verify --par quick` before the fixtures, else 0
#   2500    seconds this body may spend, from its own start
#
# Every cell is fitted ONCE per column: no --repeats anywhere. Every command
# is bounded with timeout(1). set -u and NOT set -e: a red verdict is a
# result and its log has to come home.
set -u
VERSION="0.8.10"
SLUG="nvidia-lease2"
FIXTURES="base ties hashed wide denormal denormal_ftz dupes odd negative"
LANES=""
QUICK="1"
BUDGET="2500"

OUT="/root/gemm_leg_out/$SLUG"
mkdir -p "$OUT"
G="$OUT/gate.txt"
ST="$OUT/status.tsv"
T0=$(date +%s)
say() { echo "$@" >> "$G"; }
left() { echo $(( T0 + BUDGET - $(date +%s) )); }
# status.tsv: name <TAB> exit <TAB> seconds
run() {   # <name> <timeout seconds> <command...>; stdout -> name.out, stderr -> name.log
    _n=$1; _t=$2; shift 2
    _s=$(date +%s)
    timeout -k 15 "$_t" "$@" > "$OUT/$_n.out" 2> "$OUT/$_n.log"; _e=$?
    printf '%s\t%s\t%s\n' "$_n" "$_e" "$(( $(date +%s) - _s ))" >> "$ST"
    return "$_e"
}
say "slug=$SLUG"
say "started=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
say "budget_seconds=$BUDGET"
say "fixtures=$FIXTURES"
say "lanes=${LANES:-<all par-* lanes>}"

# ---- the hardware, before anything else
uname -a > "$OUT/uname.txt" 2>&1
if command -v nvidia-smi > /dev/null 2>&1 && nvidia-smi -L > /dev/null 2>&1; then
    VENDOR=nvidia
    nvidia-smi -L > "$OUT/gpu_inventory.txt" 2>&1
    nvidia-smi --query-gpu=index,name,uuid,pci.bus_id,driver_version,compute_cap,memory.total \
        --format=csv >> "$OUT/gpu_inventory.txt" 2>&1
    nvidia-smi > "$OUT/nvidia-smi.txt" 2>&1
    nvidia-smi topo -m > "$OUT/nvidia-smi-topo.txt" 2>&1
    VISIBLE=$(nvidia-smi -L 2>/dev/null | grep -c '^GPU ')
elif command -v rocm-smi > /dev/null 2>&1 || command -v rocminfo > /dev/null 2>&1; then
    VENDOR=amd
    { rocm-smi --showid --showuniqueid --showbus --showproductname 2>&1; } > "$OUT/gpu_inventory.txt"
    rocm-smi > "$OUT/rocm-smi.txt" 2>&1
    rocminfo > "$OUT/rocminfo.txt" 2>&1
    VISIBLE=$(rocminfo 2>/dev/null | grep -c 'Device Type:[[:space:]]*GPU')
else
    say "NO GPU TOOL ANSWERED: neither nvidia-smi nor rocm-smi/rocminfo. Nothing was run."
    exit 8
fi
say "vendor=$VENDOR"
say "visible_gpus=$VISIBLE"
cat "$OUT/gpu_inventory.txt" >> "$G"
if [ "$VISIBLE" -lt 2 ]; then
    say "REFUSED: $VISIBLE visible GPU(s). The claim is about two physical devices. Nothing was run."
    exit 8
fi

# ---- a fresh venv holding ONLY numpy and the published wheel
PY=""
for c in python3.12 python3.11 python3.10 python3; do
    if command -v "$c" > /dev/null 2>&1 && \
       "$c" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)' 2>/dev/null; then
        PY=$(command -v "$c"); break
    fi
done
if [ -z "$PY" ]; then
    # the AMD image may carry no python >= 3.10 of its own
    say "no system python >= 3.10; installing one with apt"
    ( apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y python3 python3-venv python3-pip ) \
        > "$OUT/apt_python.log" 2>&1
    PY=$(command -v python3 || true)
fi
say "system_python=$PY ($("$PY" --version 2>&1))"
VENV=/root/parwheel-venv
rm -rf "$VENV"
"$PY" -m venv "$VENV" > "$OUT/venv.log" 2>&1
if [ ! -x "$VENV/bin/pip" ]; then
    ( apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y python3-venv ) >> "$OUT/venv.log" 2>&1
    rm -rf "$VENV"; "$PY" -m venv "$VENV" >> "$OUT/venv.log" 2>&1
fi
if [ ! -x "$VENV/bin/pip" ]; then
    "$PY" -m pip install --quiet virtualenv >> "$OUT/venv.log" 2>&1
    rm -rf "$VENV"; "$PY" -m virtualenv "$VENV" >> "$OUT/venv.log" 2>&1
fi
[ -x "$VENV/bin/pip" ] || { say "VENV FAILED; see venv.log. Nothing was run."; exit 12; }

WH=/root/parwheel-download
rm -rf "$WH"; mkdir -p "$WH"
run pip_download 600 "$VENV/bin/pip" download --disable-pip-version-check --no-deps \
    -d "$WH" "mojolearn==$VERSION"
say "pip_download_exit=$?"
( cd "$WH" && sha256sum ./*.whl ) > "$OUT/wheel_sha256.txt" 2>&1
( cd "$WH" && ls -l ) >> "$OUT/wheel_sha256.txt" 2>&1
cat "$OUT/wheel_sha256.txt" >> "$G"
# install EXACTLY the file that was hashed, plus numpy, and nothing else
run pip_install 600 "$VENV/bin/pip" install --disable-pip-version-check numpy "$WH"/mojolearn-"$VERSION"-*.whl
say "pip_install_exit=$?"
"$VENV/bin/pip" show mojolearn > "$OUT/pip_show.txt" 2>&1
"$VENV/bin/pip" freeze > "$OUT/pip_freeze.txt" 2>&1

# ---- run from OUTSIDE the checkout, with nothing of the checkout in the environment
RUN=/root/parrun
rm -rf "$RUN"; mkdir -p "$RUN"; cd "$RUN" || exit 9
unset PYTHONPATH MOJOLEARN_PAR_DEVICES MOJOLEARN_NUMERIC_MODE MOJOLEARN_GPU_ARCHS
env | grep -E '^(MOJOLEARN|PYTHON|CUDA_VISIBLE|HIP_VISIBLE|ROCR_VISIBLE)' > "$OUT/env_relevant.txt" 2>&1
P="$VENV/bin/python"
"$P" -c "import mojolearn,sys; print(mojolearn.__version__, mojolearn.__file__)" > "$OUT/import_where.txt" 2>&1
say "import_where=$(cat "$OUT/import_where.txt")"
case "$(cat "$OUT/import_where.txt")" in
    "$VERSION $VENV/"*) : ;;
    *) say "REFUSED: import mojolearn did not resolve to $VERSION inside $VENV. Nothing was measured."; exit 13 ;;
esac
run env_report 120 "$P" -m mojolearn env --json

# ---- LEASE 1 FINDING (pod n55vv7n35fizci): with the venv NOT activated, every
# `verify --par` command ended CANNOT RUN (exit 4) because the child that
# `_gpu_witness.require_device_count` spawns with `sys.executable` was
# /usr/bin/python3, which has no mojolearn. This diagnostic prints
# sys.executable at each step, first exactly as lease 1 ran (venv python by
# full path, PATH untouched), then with the venv ACTIVATED the way its own
# bin/activate does it. The measurement below runs ACTIVATED.
cat > "$RUN/where_is_python.py" <<'PYDIAG'
import os, shutil, sys
print("argv0_executable      ", sys.executable)
print("base_executable       ", getattr(sys, "_base_executable", None))
print("prefix / base_prefix  ", sys.prefix, "/", sys.base_prefix)
print("which python3 on PATH ", shutil.which("python3"))
print("VIRTUAL_ENV           ", os.environ.get("VIRTUAL_ENV"))
import mojolearn
print("after import mojolearn", sys.executable)
from mojolearn import _backend
print("vendor                ", _backend.vendor())
print("after vendor()        ", sys.executable)
from mojolearn._verify_all import load_harness
load_harness(par_axis=True)
print("after load_harness    ", sys.executable)
from mojolearn._gpu_witness import require_device_count
try:
    require_device_count(_backend.vendor(), 2)
    print("require_device_count(2): ok")
except Exception as exc:
    print("require_device_count(2): REFUSED:", exc)
print("at exit               ", sys.executable)
PYDIAG
run python_where_unactivated 300 "$P" "$RUN/where_is_python.py"
. "$VENV/bin/activate"
P=python
say "activated: VIRTUAL_ENV=$VIRTUAL_ENV python=$(command -v python)"
run python_where_activated 300 "$P" "$RUN/where_is_python.py"
"$P" -c "import mojolearn,sys; print(mojolearn.__version__, mojolearn.__file__)" > "$OUT/import_where_activated.txt" 2>&1
say "import_where_activated=$(cat "$OUT/import_where_activated.txt")"
case "$(cat "$OUT/import_where_activated.txt")" in
    "$VERSION $VENV/"*) : ;;
    *) say "REFUSED: activated import did not resolve to $VERSION inside $VENV. Nothing was measured."; exit 13 ;;
esac

# ---- FALLBACK, used only if the ACTIVATED venv still cannot count devices:
# install the SAME hashed wheel file into the system interpreter (no venv) and
# measure there. Which interpreter measured is recorded in gate.txt.
INSTALL_MODE=venv-activated
if ! grep -q 'require_device_count(2): ok' "$OUT/python_where_activated.out"; then
    say "ACTIVATED VENV STILL CANNOT COUNT DEVICES; falling back to the system interpreter $PY"
    deactivate 2>/dev/null || true
    run pip_install_system 600 "$PY" -m pip install --disable-pip-version-check numpy "$WH"/mojolearn-"$VERSION"-*.whl
    say "pip_install_system_exit=$?"
    P="$PY"
    INSTALL_MODE=system-interpreter
    "$P" -m pip show mojolearn > "$OUT/pip_show_system.txt" 2>&1
    run python_where_system 300 "$P" "$RUN/where_is_python.py"
    "$P" -c "import mojolearn,sys; print(mojolearn.__version__, mojolearn.__file__)" > "$OUT/import_where_system.txt" 2>&1
    say "import_where_system=$(cat "$OUT/import_where_system.txt")"
    case "$(cat "$OUT/import_where_system.txt")" in
        "$VERSION /root/mojolearn"*|"$VERSION $RUN"*) say "REFUSED: system import resolved into the checkout."; exit 13 ;;
        "$VERSION "*) : ;;
        *) say "REFUSED: system import is not $VERSION."; exit 13 ;;
    esac
fi
say "install_mode=$INSTALL_MODE"

# ---- (1) THE SELF TEST, FIRST. It must exit 0: its arm is supposed to fail.
run par_self_test 900 "$P" -m mojolearn verify --par --par-self-test
say "par_self_test_exit=$(awk -F'\t' '$1=="par_self_test"{print $2}' "$ST")  (MUST BE 0)"
run par_self_test_json 900 "$P" -m mojolearn verify --par --par-self-test --json
say "par_self_test_json_exit=$(awk -F'\t' '$1=="par_self_test_json"{print $2}' "$ST")  (MUST BE 0)"

# The shipped formatter renders a --json document into the same human report
# the command prints without --json, so one fit per column yields both.
render() {   # <name>
    "$P" -c "
import json, sys
from mojolearn._verify_par import format_par_check
print(format_par_check(json.load(open(sys.argv[1]))))" "$OUT/$1.out" > "$OUT/$1.report.txt" 2>> "$OUT/$1.log" \
        || echo "render failed for $1 (no JSON document; see $1.log)" >> "$G"
}

# ---- (2) quick: one lane per family, base fixture
if [ "$QUICK" = 1 ]; then
    _l=$(left); [ "$_l" -lt 60 ] && _l=60
    run par_quick "$_l" "$P" -m mojolearn verify --par quick --json
    say "par_quick_exit=$(awk -F'\t' '$1=="par_quick"{print $2}' "$ST") seconds=$(awk -F'\t' '$1=="par_quick"{print $3}' "$ST")"
    render par_quick
fi

# ---- (3) `--par all`, ONE FIXTURE PER COMMAND so a lease that ends early
# keeps every finished fixture's JSON. The union over the nine fixtures is
# exactly `verify --par all`: that scope differs from the default only in the
# fixtures it runs. A fixture is started only if the budget left exceeds the
# longest fixture measured so far by a margin; otherwise it is recorded as
# NOT STARTED and a later lease runs it.
LONGEST=0
for fx in $FIXTURES; do
    _l=$(left)
    _need=$(( LONGEST + LONGEST / 3 + 60 ))
    if [ "$_l" -lt "$_need" ]; then
        say "NOT STARTED: fixture $fx (budget left ${_l}s, longest fixture so far ${LONGEST}s)"
        printf 'par_all_%s\tNOT-STARTED\t0\n' "$fx" >> "$ST"
        continue
    fi
    if [ -n "$LANES" ]; then
        run "par_all_$fx" "$_l" "$P" -m mojolearn verify --par all --fixtures "$fx" --lanes "$LANES" --json
    else
        run "par_all_$fx" "$_l" "$P" -m mojolearn verify --par all --fixtures "$fx" --json
    fi
    _secs=$(awk -F'\t' -v n="par_all_$fx" '$1==n{print $3}' "$ST" | tail -1)
    say "par_all_${fx}_exit=$(awk -F'\t' -v n="par_all_$fx" '$1==n{print $2}' "$ST" | tail -1) seconds=$_secs"
    [ "$_secs" -gt "$LONGEST" ] && LONGEST=$_secs
    render "par_all_$fx"
done

say "elapsed_total=$(( $(date +%s) - T0 ))"
say "finished=$(date -u +%Y-%m-%dT%H:%M:%SZ)"

# ---- HOLD THE BOX UNTIL THE PULL IS VERIFIED (bounded). The runner fetches
# and terminates as soon as this body returns. The operator pulls $OUT over
# ssh first, checks the files are non-empty and name the expected lanes, and
# only then creates /root/par_pull_ok. The hold is bounded so it can never
# outlive the lease reserve: it gives up by itself and the runner's own fetch
# still happens.
: > "$OUT/BODY_DONE"
_hold=0
while [ ! -f /root/par_pull_ok ] && [ "$_hold" -lt 420 ]; do
    sleep 5; _hold=$(( _hold + 5 ))
done
say "pull_hold_seconds=$_hold pull_ok=$([ -f /root/par_pull_ok ] && echo yes || echo no)"
exit 0
