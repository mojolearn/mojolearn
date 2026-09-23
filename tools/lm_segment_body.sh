#!/bin/sh
# tools/lm_segment_body.sh -- ONE segment of the six-segment run on ONE rented
# box, as a MOJOLEARN_GEMM_LEG_EXTRA body for tools/gemm_remote_leg.sh (RunPod,
# nvidia or amd) or tools/do_extra_leg.sh (DigitalOcean, amd). Rendered by
# tools/lm_segment_leg.py, which substitutes every @PLACEHOLDER@ (a rented box
# gets no environment from the runner) and mints every URL on the Mac; no
# credential is in this file or on the box, only presigned URLs.
#
#   @ARM@            nvidia | amd
#   @MODE@           one | live-coordinator | live-worker
#   @DEVICES@        devices for a one-box segment ("0" or "0,1,...")
#   @ROUTE@ @SEGMENT@ @LABEL@ @STEPS@ @BOUNDARY@   (BOUNDARY "" for none)
#   @TOKENS_GROUP_PARTS@   the staged token stream's part files, joined on the box
#   @FROM_NAME@ @FROM_URL@ @FROM_SHA@   the checkpoint this segment starts from
#   @EXPECT_URL@     a chain to hold every step to ("" for none): route B's
#                    segments hold to route A's chain
#   @REPLAY_NAME@ @REPLAY_URL@ @REPLAY_SHA@ @REPLAY_CHAIN_URL@
#                    the ARRIVAL REPLAY: the previous segment's boundary-minus-two
#                    checkpoint and the sender's chain; two steps on THIS
#                    hardware must land on the boundary hash before the segment
#                    starts ("" skips it, only for the first segment)
#   @UPLOADS@        JSON {file name: presigned PUT URL} for every expected key
#   @LIVE_SHARDS@ @LIVE_WORKERS@ @LIVE_PORT@   the live segment's block, group size, port
#   @RECIPE_URL@ @RECIPE_SHA@   the recipe, pinned
#   @WHEEL@          "" builds the bindings from this commit's source on the box;
#                    a version (e.g. 0.8.15) pip-installs that PUBLISHED wheel
#                    into a venv and runs the tools with it, the repository's
#                    own package kept off the path, so the run's provenance is
#                    one wheel sha256 per platform and no build happens on a box
#   @TOKENS_URLS@    JSON {part file name: presigned GET} of the token stream, used
#                    only when the runner staged nothing ("" otherwise)
#
# The live worker waits for /root/live_peer.txt ("HOST PORT"), written by
# the orchestrator once the tunnel to the coordinator's box is up, and every
# body writes /root/lm_segment_ready when its build is done so the
# orchestrator can tell a slow build from a dead box.
#
# POSIX sh. Never `set -e`: a failure is a result and its log comes home.
set -u
ARM="@ARM@"; MODE="@MODE@"; DEVICES="@DEVICES@"
ROUTE="@ROUTE@"; SEGMENT="@SEGMENT@"; LABEL="@LABEL@"; STEPS="@STEPS@"; BOUNDARY="@BOUNDARY@"
FROM_NAME="@FROM_NAME@"; FROM_SHA="@FROM_SHA@"
REPLAY_NAME="@REPLAY_NAME@"; REPLAY_SHA="@REPLAY_SHA@"
LIVE_SHARDS="@LIVE_SHARDS@"; LIVE_WORKERS="@LIVE_WORKERS@"; LIVE_PORT="@LIVE_PORT@"
RECIPE_SHA="@RECIPE_SHA@"
WHEEL="@WHEEL@"
ROOT=/root/mojolearn
OUT=/root/gemm_leg_out/lm-segment-$ROUTE-$SEGMENT
mkdir -p "$OUT"
cd "$ROOT" || exit 9
ST="$OUT/status.txt"
PATH="$HOME/.pixi/bin:$PATH"; export PATH
export MOJOLEARN_NUMERIC_MODE=identical
export PYTHONPATH="$ROOT/python:$ROOT"
say() { echo "$(date -u +%H:%M:%S) $*" >> "$ST"; }
say "arm=$ARM mode=$MODE route=$ROUTE segment=$SEGMENT label=$LABEL steps=$STEPS boundary=$BOUNDARY started=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
uname -a > "$OUT/uname.txt" 2>&1
nvidia-smi --query-gpu=name,uuid,driver_version,memory.total --format=csv,noheader > "$OUT/gpu.txt" 2>&1 || rocm-smi --showproductname --showuniqueid > "$OUT/gpu.txt" 2>&1

case "$ARM" in
    nvidia)
        export MOJOLEARN_TARGET_COLUMN=nvidia
        if [ -z "${MOJOLEARN_GPU_ARCHS:-}" ]; then
            cap=$(nvidia-smi -i 0 --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 | tr -d ' ')
            case "$cap" in
                9.0) MOJOLEARN_GPU_ARCHS=sm_90a ;;
                [0-9].[0-9]) MOJOLEARN_GPU_ARCHS="sm_$(echo "$cap" | tr -d .)" ;;
                *) say "arch unknown (compute_cap='$cap')"; exit 2 ;;
            esac
        fi ;;
    amd) export MOJOLEARN_TARGET_COLUMN=amd; : "${MOJOLEARN_GPU_ARCHS:=gfx942}" ;;
    *) say "unknown arm $ARM"; exit 2 ;;
esac
export MOJOLEARN_GPU_ARCHS
say "gpu_archs=$MOJOLEARN_GPU_ARCHS"

# ---- the package: the published wheel, or the bindings built from this commit ----
if [ -n "$WHEEL" ]; then
    # THE PUBLISHED WHEEL. A venv on the box, mojolearn==WHEEL from PyPI, the
    # tools run with the venv's interpreter and the repository's package OFF
    # the path (PYTHONPATH unset for every python call below), so the bits
    # come from the wheel and its sha256 is the provenance. No build.
    _t0=$(date +%s)
    PYSYS=""
    for c in python3.12 python3.11 python3.10 python3; do
        if command -v "$c" > /dev/null 2>&1 && "$c" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)' 2>/dev/null; then PYSYS=$(command -v "$c"); break; fi
    done
    [ -n "$PYSYS" ] || { say "no python >= 3.10 on the box"; exit 1; }
    rm -rf /root/lm-venv
    "$PYSYS" -m venv /root/lm-venv > "$OUT/venv.log" 2>&1 || { ( apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y python3-venv ) >> "$OUT/venv.log" 2>&1; rm -rf /root/lm-venv; "$PYSYS" -m venv /root/lm-venv >> "$OUT/venv.log" 2>&1; }
    [ -x /root/lm-venv/bin/pip ] || { say "venv failed; see venv.log"; exit 1; }
    /root/lm-venv/bin/pip install --disable-pip-version-check --quiet numpy "mojolearn==$WHEEL" > "$OUT/pip_install.log" 2>&1; _rc=$?
    say "pip install mojolearn==$WHEEL exit=$_rc secs=$(( $(date +%s) - _t0 ))"
    [ "$_rc" -eq 0 ] || { say "the wheel did not install; nothing run"; exit 1; }
    /root/lm-venv/bin/pip freeze > "$OUT/pip_freeze.txt" 2>&1
    /root/lm-venv/bin/pip download --no-deps --quiet --dest /root/lm-wheel "mojolearn==$WHEEL" > "$OUT/wheel_download.log" 2>&1 && sha256sum /root/lm-wheel/*.whl > "$OUT/wheel.sha256" 2>&1
    say "wheel: $(cat "$OUT/wheel.sha256" 2>/dev/null | cut -c1-100)"
    unset PYTHONPATH
    PYBIN=/root/lm-venv/bin/python
    run_py() { "$PYBIN" "$@"; }
else
    rm -f python/mojolearn/identical/_mojolearn.so python/mojolearn/identical/_mojolearn_byte_lm.so
    _t0=$(date +%s)
    MOJOLEARN_SKIP_BUILD_GATE=1 sh bindings/build.sh > "$OUT/build_base.log" 2>&1; _rc=$?
    say "build base exit=$_rc secs=$(( $(date +%s) - _t0 ))"
    [ "$_rc" -eq 0 ] || { say "base build failed; nothing run"; exit 1; }
    _t0=$(date +%s)
    MOJOLEARN_BUILD_EXTRA_DEFINES="" sh bindings/build_byte_lm.sh > "$OUT/build_byte_lm.log" 2>&1; _rc=$?
    say "build byte_lm exit=$_rc secs=$(( $(date +%s) - _t0 ))"
    [ "$_rc" -eq 0 ] || { say "byte_lm build failed; nothing run"; exit 1; }
    pixi run python -c 'import numpy' > "$OUT/numpy.log" 2>&1 || { pixi run python -m pip install numpy >> "$OUT/numpy.log" 2>&1; say "numpy pip exit=$?"; }
    run_py() { pixi run python "$@"; }
fi
run_py -c 'import mojolearn, sys; b = mojolearn._backend.binding("_mojolearn_byte_lm", "identical"); print("mojolearn", getattr(mojolearn, "__version__", "?"), mojolearn.__file__, "vendor", b.byte_lm_vendor(), "set_lr", callable(getattr(b, "byte_lm_parallel_set_lr", None)), "device_fold", callable(getattr(b, "byte_lm_parallel_fold_export", None)))' > "$OUT/binding.txt" 2>&1
say "binding: $(tail -1 "$OUT/binding.txt" | cut -c1-200)"

# ---- the token stream: staged parts joined and verified by the manifest ----
# A runner without R2 staging (Hot Aisle) leaves nothing under /root; then the
# parts are fetched here by the presigned GETs the renderer baked in.
cat > "$OUT/tokens_urls.json" <<'TOKENS_URLS'
@TOKENS_URLS@
TOKENS_URLS
if [ -z "$(find /root -name tokens.i32.part00 -o -name tokens.i32 2>/dev/null | head -1)" ] && [ "$(head -c 1 "$OUT/tokens_urls.json")" = "{" ]; then
    _td=/root/tokens_stream; mkdir -p "$_td"
    _t0=$(date +%s)
    # every part at once, each with its own time limit and retries: a single
    # stalled transfer once held a pod for two hours (T2 segment 3, attempt 2)
    run_py - "$OUT/tokens_urls.json" "$_td" > "$OUT/tokens_fetch.log" 2>&1 <<'PY'
import json, os, subprocess, sys, time
from concurrent.futures import ThreadPoolExecutor
urls, out = json.load(open(sys.argv[1])), sys.argv[2]
# Byte ranges of 100 MB, twenty at a time, each with a real speed floor and
# its own retries; only the ranges still missing are fetched again. One
# stream per part sat at 130 KB/s for 25 minutes on an H100 pod (T3
# rehearsal, 2026-09-23) while range requests to the same objects ran at
# 35 to 48 MB/s, and a whole-file floor of 100 KB/s never fired.
CHUNK = 100_000_000
def size_of(url):
    # a presigned GET URL refuses HEAD (403); one byte's Content-Range says the total
    r = subprocess.run(["curl", "-fsS", "--max-time", "60", "-r", "0-0", "-D", "-", "-o", "/dev/null", url],
                       capture_output=True, text=True)
    for line in r.stdout.splitlines():
        if line.lower().startswith("content-range:"):
            return int(line.rsplit("/", 1)[1])
    print("FAILED size of", url[:80], r.stderr.strip()[:200], flush=True)
    raise SystemExit(1)
def fetch_range(item):
    name, url, i, a, b, path = item
    for attempt in range(6):
        r = subprocess.run(["curl", "-fsS", "--max-time", "120", "--speed-limit", "1000000", "--speed-time", "30",
                            "-r", "%d-%d" % (a, b), "-o", path, url], capture_output=True, text=True)
        if r.returncode == 0 and os.path.getsize(path) == b - a + 1:
            return True
        time.sleep(2)
    return False
def get(item):
    name, url = item
    t0 = time.time()
    total = size_of(url)
    ranges = [(i, i * CHUNK, min(total, (i + 1) * CHUNK) - 1) for i in range((total + CHUNK - 1) // CHUNK)]
    paths = {i: "%s/.%s.r%04d" % (out, name, i) for i, _, _ in ranges}
    with ThreadPoolExecutor(max_workers=20) as pool:
        ok = list(pool.map(fetch_range, [(name, url, i, a, b, paths[i]) for i, a, b in ranges]))
    if not all(ok):
        print("FAILED", name, "ranges", [i for (i, _, _), good in zip(ranges, ok) if not good], flush=True)
        return 1
    with open(out + "/" + name, "wb") as f:
        for i, _, _ in ranges:
            with open(paths[i], "rb") as g:
                f.write(g.read())
            os.remove(paths[i])
    print("fetched", name, "%d bytes in %d ranges, %.0f s" % (total, len(ranges), time.time() - t0), flush=True)
    return 0
with ThreadPoolExecutor(max_workers=2) as pool:
    codes = list(pool.map(get, sorted(urls.items())))
sys.exit(0 if all(c == 0 for c in codes) else 1)
PY
    say "tokens fetched by URL exit=$? secs=$(( $(date +%s) - _t0 ))"
fi
TOK=$(dirname "$(find /root -name tokens.i32.part00 2>/dev/null | head -1)")
if [ -n "$TOK" ] && [ -f "$TOK/manifest.json" ]; then
    if [ ! -f "$TOK/tokens.i32" ]; then
        _t0=$(date +%s)
        cat "$TOK"/tokens.i32.part?? > "$TOK/tokens.i32"
        say "joined $(ls "$TOK"/tokens.i32.part?? | wc -l | tr -d ' ') parts in $(( $(date +%s) - _t0 )) s: $(wc -c < "$TOK/tokens.i32" | tr -d ' ') bytes"
        rm -f "$TOK"/tokens.i32.part??
    fi
else
    TOK=$(dirname "$(find /root -name tokens.i32 2>/dev/null | head -1)")
    [ -f "$TOK/manifest.json" ] || { say "no staged token stream with a manifest under /root"; exit 3; }
fi
say "tokens=$TOK"

# ---- the recipe and the checkpoints, fetched and verified ----
fetch() {  # $1 name  $2 url  $3 sha256 ("" skips the check)
    _t0=$(date +%s)
    curl -fsS --retry 3 -o "$OUT/in/$1" "$2" >> "$OUT/fetch.log" 2>&1; _rc=$?
    say "fetch $1 exit=$_rc secs=$(( $(date +%s) - _t0 ))"
    [ "$_rc" -eq 0 ] || return 1
    if [ -n "$3" ]; then
        printf '%s  %s\n' "$3" "$OUT/in/$1" | sha256sum -c - >> "$OUT/fetch.log" 2>&1 || { say "$1 FAILED its sha256"; return 1; }
    fi
    return 0
}
mkdir -p "$OUT/in"
fetch recipe.json '@RECIPE_URL@' "$RECIPE_SHA" || exit 4
if [ "$FROM_NAME" = init ]; then
    # the first segment: the seed checkpoint is drawn HERE from the recipe's
    # seed, pinned by its sha256, and uploaded like any other checkpoint
    FROM_NAME=ckpt_00000000.blm
    _t0=$(date +%s)
    run_py tools/lm_segment.py init --recipe "$OUT/in/recipe.json" --tokens "$TOK" --out "$OUT/in/$FROM_NAME" > "$OUT/init.log" 2>&1; _rc=$?
    say "init exit=$_rc secs=$(( $(date +%s) - _t0 )): $(tail -1 "$OUT/init.log" | cut -c1-200)"
    [ "$_rc" -eq 0 ] || exit 4
    sha256sum "$OUT/in/$FROM_NAME" > "$OUT/seed.sha256"
    _u='@SEED_URL@'
    if [ -n "$_u" ]; then curl -fsS --retry 3 -T "$OUT/in/$FROM_NAME" "$_u" > "$OUT/upload_seed.log" 2>&1; say "upload seed exit=$?"; fi
else
    fetch "$FROM_NAME" '@FROM_URL@' "$FROM_SHA" || exit 4
fi
EXPECT=""
if [ -n '@EXPECT_URL@' ]; then fetch expect_chain.jsonl '@EXPECT_URL@' "" && EXPECT="--expect-chain $OUT/in/expect_chain.jsonl"; fi
R="$OUT/in/recipe.json"
S=tools/lm_segment.py

run() {  # $1 name, rest: lm_segment run args
    _n="$1"; shift
    _t0=$(date +%s)
    run_py $S run --recipe "$R" --tokens "$TOK" --out "$OUT/$_n" --label "$LABEL" "$@" > "$OUT/$_n.log" 2>&1; _rc=$?
    say "$_n exit=$_rc secs=$(( $(date +%s) - _t0 ))"
    grep -E "PASS|FAIL|REFUSED|DISAGREE|ERROR" "$OUT/$_n.log" | tail -2 >> "$ST"
    return $_rc
}

# ---- the arrival replay: the sender's last two steps on THIS hardware ----
if [ -n "$REPLAY_NAME" ]; then
    fetch "$REPLAY_NAME" '@REPLAY_URL@' "$REPLAY_SHA" || exit 4
    fetch replay_chain.jsonl '@REPLAY_CHAIN_URL@' "" || exit 4
    run arrival --from "$OUT/in/$REPLAY_NAME" --steps 2 --devices "${DEVICES%%,*}" --route "$ROUTE" --segment "$SEGMENT-arrival" \
        --no-checkpoints --expect-chain "$OUT/in/replay_chain.jsonl" || { say "ARRIVAL REPLAY FAILED: the segment does not start"; exit 5; }
    rm -f "$OUT/in/$REPLAY_NAME"
fi

# ---- ready: everything fetched, the arrival replay passed; a live coordinator
# starts listening a few seconds after this mark, a live worker waits for it ----
touch /root/lm_segment_ready
say "ready"

# ---- the segment ----
cat > "$OUT/uploads.json" <<'UPLOADS'
@UPLOADS@
UPLOADS
_up=""; [ "$(head -c 1 "$OUT/uploads.json")" = "{" ] && _up="--upload-urls $OUT/uploads.json"
_b=""; [ -n "$BOUNDARY" ] && _b="--boundary $BOUNDARY"
case "$MODE" in
    one)
        # shellcheck disable=SC2086
        run segment --from "$OUT/in/$FROM_NAME" --steps "$STEPS" --devices "$DEVICES" --route "$ROUTE" --segment "$SEGMENT" $_b $EXPECT $_up ;;
    live-coordinator)
        # shellcheck disable=SC2086
        run segment --from "$OUT/in/$FROM_NAME" --steps "$STEPS" --devices "${DEVICES%%,*}" --route "$ROUTE" --segment "$SEGMENT" $_b $EXPECT $_up \
            --live-role coordinator --live-shards "$LIVE_SHARDS" --live-workers "$LIVE_WORKERS" --live-port "$LIVE_PORT" --live-timeout 5400 ;;
    live-worker)
        _w=0
        while [ ! -f /root/live_peer.txt ] && [ "$_w" -lt 3600 ]; do sleep 5; _w=$(( _w + 5 )); done
        [ -f /root/live_peer.txt ] || { say "no peer address after ${_w}s; the worker never ran"; exit 6; }
        read -r _host _port < /root/live_peer.txt
        say "peer $_host:$_port after ${_w}s"
        # shellcheck disable=SC2086
        run segment --from "$OUT/in/$FROM_NAME" --steps "$STEPS" --devices "${DEVICES%%,*}" --route "$ROUTE" --segment "$SEGMENT" $EXPECT \
            --live-role worker --live-shards "$LIVE_SHARDS" --live-address "$_host:$_port" --live-timeout 5400 ;;
    *) say "unknown mode $MODE"; exit 2 ;;
esac
sha256sum "$OUT"/segment/ckpt_*.blm > "$OUT/checkpoints.sha256" 2>/dev/null
rm -f "$OUT"/segment/ckpt_*.blm "$OUT/in/$FROM_NAME"   # 1.95 GB each; pinned in the manifest and in R2, never fetched home
say "finished=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
touch /root/lm_segment_done
exit 0
