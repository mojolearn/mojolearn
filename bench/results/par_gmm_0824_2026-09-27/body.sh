#!/bin/sh
# MOJOLEARN_GEMM_LEG_EXTRA body: par-gmm on TWO PHYSICAL GPUs from the
# PUBLISHED wheel mojolearn==0.8.24, in a fresh venv, outside the checkout.
# Pattern: bench/results/identity_break/2026-09-20_par-two-physical-gpu-wheel-0810.
#   (1) verify --par --par-self-test
#   (2) verify --par --lanes par-gmm --json      (two-device column vs shipped table)
#   (3) direct.py: par-gmm through the verifier's own par_check/run_cell at
#       devices (0,), (1,) and (0,1), with the lane's sub-part hashes, plus the
#       plain gmm lane's sub-part hashes on device 0
#   (4) verify --lanes gmm,gmm-sample --fixtures base --json (one device vs shipped table)
# set -u, NOT set -e: a red verdict is a result and must come home.
set -u
VERSION="0.8.24"
OUT="/root/gemm_leg_out/pargmm"
mkdir -p "$OUT"
G="$OUT/gate.txt"; ST="$OUT/status.tsv"
T0=$(date +%s); BUDGET=1800
say() { echo "$@" >> "$G"; }
run() { _n=$1; _t=$2; shift 2; _s=$(date +%s)
    timeout -k 15 "$_t" "$@" > "$OUT/$_n.out" 2> "$OUT/$_n.log"; _e=$?
    printf '%s\t%s\t%s\n' "$_n" "$_e" "$(( $(date +%s) - _s ))" >> "$ST"; return "$_e"; }
say "started=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
uname -a > "$OUT/uname.txt" 2>&1
nvidia-smi -L > "$OUT/gpu_inventory.txt" 2>&1
nvidia-smi --query-gpu=index,name,uuid,pci.bus_id,driver_version,compute_cap,memory.total --format=csv >> "$OUT/gpu_inventory.txt" 2>&1
nvidia-smi topo -m > "$OUT/nvidia-smi-topo.txt" 2>&1
VISIBLE=$(nvidia-smi -L 2>/dev/null | grep -c '^GPU ')
cat "$OUT/gpu_inventory.txt" >> "$G"; say "visible_gpus=$VISIBLE"
[ "$VISIBLE" -ge 2 ] || { say "REFUSED: fewer than two GPUs"; exit 8; }

PY=""
for c in python3.12 python3.11 python3.10 python3; do
    if command -v "$c" >/dev/null 2>&1 && "$c" -c 'import sys; sys.exit(0 if sys.version_info >= (3,10) else 1)' 2>/dev/null; then PY=$(command -v "$c"); break; fi
done
say "system_python=$PY ($("$PY" --version 2>&1))"
VENV=/root/pargmm-venv; rm -rf "$VENV"
"$PY" -m venv "$VENV" > "$OUT/venv.log" 2>&1
if [ ! -x "$VENV/bin/pip" ]; then "$PY" -m pip install --quiet virtualenv >> "$OUT/venv.log" 2>&1; rm -rf "$VENV"; "$PY" -m virtualenv "$VENV" >> "$OUT/venv.log" 2>&1; fi
[ -x "$VENV/bin/pip" ] || { say "VENV FAILED"; exit 12; }
RUN=/root/pargmm-run; rm -rf "$RUN"; mkdir -p "$RUN"; cd "$RUN" || exit 9
unset PYTHONPATH MOJOLEARN_PAR_DEVICES MOJOLEARN_NUMERIC_MODE MOJOLEARN_GPU_ARCHS MOJOLEARN_IDENTITY_BREAK
. "$VENV/bin/activate"
run pip_install 900 pip install --disable-pip-version-check "mojolearn==$VERSION" numpy
say "pip_install_exit=$?"
pip freeze > "$OUT/pip_freeze.txt" 2>&1
python - > "$OUT/wheel_record.txt" 2>&1 <<'PYW'
import hashlib, importlib.metadata as md
for n in ("mojolearn", "mojolearn-nvidia", "mojolearn-amd", "numpy"):
    try:
        d = md.distribution(n)
        print(n, d.version, d.read_text("direct_url.json") or "", (d.read_text("WHEEL") or "").replace("\n", " "))
    except Exception as e:
        print(n, "NOT INSTALLED", e)
PYW
python -c "import mojolearn; print(mojolearn.__version__, mojolearn.__file__)" > "$OUT/import_where.txt" 2> "$OUT/import_where.err"
say "import_where=$(cat "$OUT/import_where.txt")"
case "$(cat "$OUT/import_where.txt")" in "$VERSION $VENV/"*) : ;; *) say "REFUSED: import not in venv"; exit 13 ;; esac
run env_report 180 python -m mojolearn env --json

run par_self_test 900 python -m mojolearn verify --par --par-self-test
say "par_self_test_exit=$(awk -F'\t' '$1=="par_self_test"{print $2}' "$ST") (MUST BE 0)"

run par_gmm 900 python -m mojolearn verify --par --lanes par-gmm --json
say "par_gmm_exit=$(awk -F'\t' '$1=="par_gmm"{print $2}' "$ST")"
python -c "
import json,sys
from mojolearn._verify_par import format_par_check
print(format_par_check(json.load(open(sys.argv[1]))))" "$OUT/par_gmm.out" > "$OUT/par_gmm.report.txt" 2>> "$OUT/par_gmm.log"

cat > "$RUN/direct.py" <<'PYD'
import os, json, sys
os.environ["MOJOLEARN_NUMERIC_MODE"] = "identical"
os.environ.pop("MOJOLEARN_PAR_DEVICES", None)
import mojolearn as ml
from mojolearn import _backend
from mojolearn._verify_all import load_harness, run_cell
from mojolearn._verify_par import par_check, par_devices, PoolWitness
from mojolearn._cpu_reference import reference_training
from mojolearn._identity import harness_path
h = load_harness(par_axis=True)
print("harness", harness_path(), file=sys.stderr)
out = dict(version=ml.__version__, vendor=_backend.vendor(), numeric_mode=ml.numeric_mode(), runs={})
X, yc, yr = h.fixture("base"); held = h.heldout("base")
def subparts(lane, devs):
    w = PoolWitness(_backend.vendor(), devs)
    with reference_training(), par_devices(devs), w.watching():
        try:
            fit = h.LANES[lane](ml, X, yc, yr, held.copy())
            sub = {k: v for k, v in dict(fit).items()}
            err = None
        except Exception as e:
            sub, err = getattr(e, "parts", None), f"{type(e).__name__}: {e}"[:2000]
    return dict(subparts=sub, error=err, train=(h._train_hash(sub) if sub and not err else None),
                witness=w.summary(), witness_refusal=w.refusal())
for devs in ((0,), (1,), (0, 1)):
    key = ",".join(map(str, devs))
    r = par_check(h, ml, ["par-gmm"], ["base"], devices=devs)
    cells = {c["part"]: dict(verdict_vs_table=c["verdict"], value=c["two"], table=c["one"], error=c.get("two_error")) for c in r["cells"]}
    out["runs"]["par-gmm@" + key] = dict(cells=cells, witness=r["witness"], witness_refusal=r["witness_refusal"],
                                          direct=subparts("par-gmm", devs))
    print("par-gmm", key, {p: c["value"] for p, c in cells.items()}, file=sys.stderr)
out["runs"]["gmm@0"] = dict(direct=subparts("gmm", (0,)))
print(json.dumps(out, indent=1, sort_keys=True, default=str))
PYD
run direct 900 python "$RUN/direct.py"
say "direct_exit=$(awk -F'\t' '$1=="direct"{print $2}' "$ST")"

run gmm_vs_table 900 python -m mojolearn verify --lanes gmm,gmm-sample --fixtures base --json
say "gmm_vs_table_exit=$(awk -F'\t' '$1=="gmm_vs_table"{print $2}' "$ST")"
say "elapsed_total=$(( $(date +%s) - T0 ))"
say "finished=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
exit 0
