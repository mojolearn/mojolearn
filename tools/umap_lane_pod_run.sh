#!/bin/bash
# On-pod payload for lane/umap-optimizer (2026-09-09). Phases, each writing
# $OUT/<phase>.done with its exit code so the desk can poll one file:
#
#   bash tools/umap_lane_pod_run.sh bootstrap   pixi + mojo version
#   bash tools/umap_lane_pod_run.sh cuml        cuML venv (background-safe)
#   bash tools/umap_lane_pod_run.sh build       the IDENTICAL phase bench in
#                                               three launch widths + the host
#                                               arm, the two stage-identity
#                                               fixtures
#   bash tools/umap_lane_pod_run.sh gate        launch-width bit gate at 20k,
#                                               stage fixtures, host-arm bits
#   bash tools/umap_lane_pod_run.sh price       100k x 3 rounds (+ dump), 5k dump
#   bash tools/umap_lane_pod_run.sh million     1M rows, one round
#   bash tools/umap_lane_pod_run.sh quality     ours vs cuML neighborhood_quality
#   bash tools/umap_lane_pod_run.sh cuml1m      cuML UMAP at 1M rows, 5 rounds
#
# Nothing here is run on the Mac; `RUN OWED` in docs/lanes/HANDOFF_umap.md
# lists the Apple commands.
set -u
ROOT=/root/mojolearn
OUT=${MOJOLEARN_UMAP_OUT:-/root/umap_out}
mkdir -p "$OUT"
cd "$ROOT" || exit 9
export PATH="$HOME/.pixi/bin:$PATH"
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 NUMEXPR_NUM_THREADS=2
export MOJOLEARN_COMPILE_JOBS=${MOJOLEARN_COMPILE_JOBS:-4}
phase=${1:?phase}
rc=0
run() {
    local name=$1 code=0
    shift
    "$@" > "$OUT/$name.log" 2>&1 || code=$?
    printf '%s\t%s\n' "$name" "$code" >> "$OUT/status.tsv"
    if (( code != 0 )); then rc=1; fi
    return "$code"
}
finish() { echo "$rc" > "$OUT/$phase.done"; exit "$rc"; }
rm -f "$OUT/$phase.done"
IDENT="-D MOJOLEARN_NUMERIC_IDENTICAL=1"
build() {
    # build <name> <source> [extra -D ...]
    local name=$1 src=$2
    shift 2
    # shellcheck disable=SC2086
    run "build-$name" pixi run mojo build -j "$MOJOLEARN_COMPILE_JOBS" -I . $IDENT "$@" "$src" -o "$OUT/bin/$name"
}
case "$phase" in
bootstrap)
    nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv > "$OUT/nvidia.csv" 2>&1
    uname -a > "$OUT/uname.txt"
    cat commit.txt > "$OUT/commit.txt" 2>/dev/null
    if [ ! -x "$HOME/.pixi/bin/pixi" ] && ! command -v pixi >/dev/null 2>&1; then
        run pixi-install sh -c 'curl -fsSL --max-time 300 https://pixi.sh/install.sh | sh'
    fi
    run pixi-env pixi install --locked
    run mojo-version pixi run mojo --version
    cat "$OUT/mojo-version.log"
    finish ;;
cuml)
    run cuml-venv python3 -m venv --system-site-packages /root/cuml-venv
    run cuml-wheels /root/cuml-venv/bin/python -m pip install --disable-pip-version-check --no-input \
        --only-binary=:all: numpy==2.4.6 cupy-cuda12x==14.2.0 cuml-cu12==26.8.0 --extra-index-url https://pypi.nvidia.com
    run cuml-freeze /root/cuml-venv/bin/python -m pip freeze
    finish ;;
build)
    mkdir -p "$OUT/bin"
    build umap-phase bench/umap_phase_price_main.mojo
    build umap-phase-tpb64 bench/umap_phase_price_main.mojo -D MOJOLEARN_UMAP_IDENTICAL_OPT_TPB_64=1
    build umap-phase-tpb256 bench/umap_phase_price_main.mojo -D MOJOLEARN_UMAP_IDENTICAL_OPT_TPB_256=1
    build umap-phase-host bench/umap_phase_price_main.mojo -D MOJOLEARN_UMAP_IDENTICAL_HOST_OPTIMIZER=1
    build umap-identity umap/checks/identity_check.mojo
    build umap-identity-broader umap/checks/identity_broader_check.mojo
    build umap-identity-host umap/checks/identity_check.mojo -D MOJOLEARN_UMAP_IDENTICAL_HOST_OPTIMIZER=1
    build umap-identity-broader-host umap/checks/identity_broader_check.mojo -D MOJOLEARN_UMAP_IDENTICAL_HOST_OPTIMIZER=1
    ls -l "$OUT/bin" > "$OUT/bin.txt"
    finish ;;
gate)
    mkdir -p "$OUT/dump"
    for w in umap-phase umap-phase-tpb64 umap-phase-tpb256; do
        run "gate-20k-$w" env MOJOLEARN_UMAP_ROWS=20000 MOJOLEARN_UMAP_DUMP="$OUT/dump/20k-$w.f32" "$OUT/bin/$w"
    done
    run gate-20k-host env MOJOLEARN_UMAP_ROWS=20000 MOJOLEARN_UMAP_DUMP="$OUT/dump/20k-host.f32" "$OUT/bin/umap-phase-host"
    (cd "$OUT/dump" && sha256sum 20k-*.f32) > "$OUT/gate-20k-sha256.txt"
    grep -h "embedding_fnv1a64" "$OUT"/gate-20k-*.log | sed 's/.*embedding_fnv1a64/embedding_fnv1a64/' > "$OUT/gate-20k-fingerprints.txt"
    run identity "$OUT/bin/umap-identity"
    run identity-broader "$OUT/bin/umap-identity-broader"
    run identity-host "$OUT/bin/umap-identity-host"
    run identity-broader-host "$OUT/bin/umap-identity-broader-host"
    finish ;;
price)
    mkdir -p "$OUT/dump"
    run price-100k env MOJOLEARN_UMAP_ROWS=100000 MOJOLEARN_UMAP_ROUNDS=3 MOJOLEARN_UMAP_DUMP="$OUT/dump/100k.f32" "$OUT/bin/umap-phase"
    run price-20k env MOJOLEARN_UMAP_ROWS=20000 MOJOLEARN_UMAP_ROUNDS=3 "$OUT/bin/umap-phase"
    run price-5k env MOJOLEARN_UMAP_ROWS=5000 MOJOLEARN_UMAP_ROUNDS=1 MOJOLEARN_UMAP_DUMP="$OUT/dump/5k.f32" "$OUT/bin/umap-phase"
    run price-5k-host env MOJOLEARN_UMAP_ROWS=5000 MOJOLEARN_UMAP_ROUNDS=1 MOJOLEARN_UMAP_DUMP="$OUT/dump/5k-host.f32" "$OUT/bin/umap-phase-host"
    finish ;;
million)
    run price-1m env MOJOLEARN_UMAP_ROWS=1000000 MOJOLEARN_UMAP_ROUNDS=1 timeout 1500 "$OUT/bin/umap-phase"
    finish ;;
quality)
    run quality-5k /root/cuml-venv/bin/python tools/umap_quality_vs_cuml.py --rows 5000 --ours-dump "$OUT/dump/5k.f32" --out "$OUT/quality-5k.json"
    run quality-5k-host /root/cuml-venv/bin/python tools/umap_quality_vs_cuml.py --rows 5000 --ours-dump "$OUT/dump/5k-host.f32" --out "$OUT/quality-5k-host.json"
    finish ;;
cuml1m)
    run cuml-1m /root/cuml-venv/bin/python tools/umap_cuml_reference.py --rows 1000000 --rounds 5 --out "$OUT/cuml-umap-1m.json"
    finish ;;
*)
    echo "unknown phase $phase"; rc=2; finish ;;
esac
