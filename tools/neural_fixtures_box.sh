#!/bin/bash
# tools/neural_fixtures_box.sh -- make, ship and stage the neural-layer fixtures
# (tools/neural_fixtures.py) on a Linux lq box. Run from the repo tree (an lq CMD job runs in
# the branch tree). Credentials never reach a box: the Mac mints presigned URLs with
# tools/dataset_store.sh and hands only the URL over, as tools/stage_full_box.sh does for
# rows-full.
#
#   bash tools/neural_fixtures_box.sh generate
#       A clean CPU-torch venv (/root/nf-venv: numpy, torch==$NF_TORCH from the PyTorch CPU
#       index, torch-geometric==$NF_PYG; the board's pins), then
#       tools/neural_fixtures.py generate over every algos-data mirror present (rows-full,
#       rows-small) into $NF_DIR, verify, and tar it to $NF_TAR (+ .sha256).
#   PUT_URL='<presigned PUT>' bash tools/neural_fixtures_box.sh put
#       Upload $NF_TAR to R2 (key $NF_KEY).
#   GET_URL='<presigned GET>' [NF_SHA=<tar sha256>] bash tools/neural_fixtures_box.sh stage
#       Fetch the tar, check its sha256, extract to $NF_DIR, re-verify every file. The race
#       harness finds it there by default (<dirname(--data)>/neural-fixtures).
set -euo pipefail
ROOT=${NF_ROOT:-/root/board-0833/cache/algos-data}
NF_DIR=${NF_DIR:-$ROOT/neural-fixtures}
NF_TAR=${NF_TAR:-$ROOT/neural-fixtures-v1.tar.gz}
NF_KEY=${NF_KEY:-datasets/neural-fixtures-v1.tar.gz}
NF_TORCH=${NF_TORCH:-2.13.0}          # tools/bench_board.py DEFAULT_TORCH_SPEC
NF_PYG=${NF_PYG:-2.8.0.post1}         # tools/bench_board_algos.py PINS_COMMON
VENV=${NF_VENV:-/root/nf-venv}
HERE=$(cd "$(dirname "$0")" && pwd)
LOG=${NF_LOG:-/root/lq/neural-fixtures.log}
mkdir -p "$(dirname "$LOG")"

cmd=${1:-}
case $cmd in
generate)
    exec >>"$LOG" 2>&1
    echo "== $(date -u +%FT%TZ) generate start torch=$NF_TORCH pyg=$NF_PYG dir=$NF_DIR"
    if [ ! -x "$VENV/bin/python" ]; then python3 -m venv "$VENV"; fi
    "$VENV/bin/python" -m pip install -q --upgrade pip
    "$VENV/bin/python" -m pip install -q numpy "torch==$NF_TORCH" --index-url https://download.pytorch.org/whl/cpu \
        --extra-index-url https://pypi.org/simple
    "$VENV/bin/python" -m pip install -q "torch-geometric==$NF_PYG"
    DATA_ARGS=()
    for d in "$ROOT/rows-full" "$ROOT/rows-small"; do [ -d "$d" ] && DATA_ARGS+=(--data "$d"); done
    rm -rf "$NF_DIR.new"
    MOJOLEARN_REPO_COMMIT=$(git -C "$HERE/.." rev-parse HEAD 2>/dev/null || echo unknown) \
        "$VENV/bin/python" "$HERE/neural_fixtures.py" generate --out "$NF_DIR.new" "${DATA_ARGS[@]}"
    python3 "$HERE/neural_fixtures.py" verify --dir "$NF_DIR.new"
    rm -rf "$NF_DIR"; mv "$NF_DIR.new" "$NF_DIR"
    tar -czf "$NF_TAR.partial" -C "$(dirname "$NF_DIR")" "$(basename "$NF_DIR")"; mv "$NF_TAR.partial" "$NF_TAR"
    sha256sum "$NF_TAR" | tee "$NF_TAR.sha256"
    echo "NF-TAR $(cut -d' ' -f1 "$NF_TAR.sha256") bytes=$(stat -c %s "$NF_TAR") key=$NF_KEY"
    echo "== $(date -u +%FT%TZ) GENERATED"
    tail -n 3 "$LOG" >&2 || true
    ;;
put)
    : "${PUT_URL:?PUT_URL (tools/dataset_store.sh presign-put $NF_KEY) is required}"
    [ -s "$NF_TAR" ] || { echo "no $NF_TAR: run generate first" >&2; exit 2; }
    cfg=$(mktemp); trap 'rm -f "$cfg"' EXIT
    printf 'url = "%s"\n' "$PUT_URL" > "$cfg"; unset PUT_URL
    curl -fsS --retry 3 -K "$cfg" -T "$NF_TAR"
    echo "NF-PUT key=$NF_KEY sha256=$(cut -d' ' -f1 "$NF_TAR.sha256")"
    ;;
stage)
    : "${GET_URL:?GET_URL (tools/dataset_store.sh presign $NF_KEY) is required}"
    mkdir -p "$ROOT"
    cfg=$(mktemp); trap 'rm -f "$cfg" "$NF_TAR.partial"' EXIT
    printf 'url = "%s"\n' "$GET_URL" > "$cfg"; unset GET_URL
    curl -fsS --retry 3 -K "$cfg" -o "$NF_TAR.partial"; mv "$NF_TAR.partial" "$NF_TAR"
    got=$(sha256sum "$NF_TAR" | cut -d' ' -f1)
    if [ -n "${NF_SHA:-}" ] && [ "$got" != "$NF_SHA" ]; then
        echo "NF-STAGE FAIL tar sha256 $got != $NF_SHA" >&2; exit 3
    fi
    rm -rf "$NF_DIR.new"; mkdir -p "$NF_DIR.new"
    tar -xzf "$NF_TAR" -C "$NF_DIR.new"
    src="$NF_DIR.new/$(basename "$NF_DIR")"; [ -d "$src" ] || src="$NF_DIR.new"
    python3 "$HERE/neural_fixtures.py" verify --dir "$src"
    rm -rf "$NF_DIR"; mv "$src" "$NF_DIR"; rm -rf "$NF_DIR.new"
    echo "NF-STAGED dir=$NF_DIR tar_sha256=$got"
    ;;
*)
    sed -n '2,20p' "$0"; exit 2 ;;
esac
