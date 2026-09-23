#!/bin/sh
# tools/lm_controls_body.sh -- the negative controls of the GPT-3 Small run
# (docs/GPT3_SMALL_SIX_SEGMENT_PLAN.md, section 6 item 7) on ONE rented NVIDIA
# box, as a MOJOLEARN_GEMM_LEG_EXTRA body for tools/gemm_remote_leg.sh.
# Rendered by `tools/lm_segment_leg.py controls`, which fills every
# @PLACEHOLDER@ and mints every URL on the Mac (GETs only: a control uploads
# nothing, anywhere).
#
# From one checkpoint of the real run, each control runs two optimizer steps
# with `tools/lm_segment.py run --no-checkpoints --expect-chain` against the
# real chain's lines, in its own out directory, so lm_segment itself reads the
# verdict:
#
#   positive     the plain replay; MUST PASS or nothing after it is run
#   none         --control none (the stamped plain step); must PASS
#   zero-moments --zero-moments; must FAIL
#   k63          --control shards=63; must FAIL
#   swap-5-40    --control swap=5,40; must FAIL
#   swap-0-1     --control swap=0,1; the losses move, the sum commutes
#   split        --control split (the ulp harness, no edit); must PASS
#   ulp          --control ulp=63,auto; must FAIL
#
# The package is the PUBLISHED wheel (pip, a venv, the repository's package
# off the path); only tools/lm_segment.py comes from the shipped commit.
# POSIX sh. Never `set -e`: a failure is a result and its log comes home.
set -u
FROM_NAME="@FROM_NAME@"
FROM_STEP="@FROM_STEP@"
RECIPE_SHA="@RECIPE_SHA@"
WHEEL="@WHEEL@"
CONTROLS="@CONTROLS@"
ROOT=/root/mojolearn
OUT=/root/gemm_leg_out/lm-controls
mkdir -p "$OUT/in"
cd "$ROOT" || exit 9
ST="$OUT/status.txt"
export MOJOLEARN_NUMERIC_MODE=identical
say() { echo "$(date -u +%H:%M:%S) $*" >> "$ST"; }
say "controls from $FROM_NAME (global step $FROM_STEP), wheel $WHEEL, started=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
uname -a > "$OUT/uname.txt" 2>&1
nvidia-smi --query-gpu=name,uuid,driver_version,memory.total --format=csv,noheader > "$OUT/gpu.txt" 2>&1
export MOJOLEARN_TARGET_COLUMN=nvidia
if [ -z "${MOJOLEARN_GPU_ARCHS:-}" ]; then
    cap=$(nvidia-smi -i 0 --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 | tr -d ' ')
    case "$cap" in
        9.0) MOJOLEARN_GPU_ARCHS=sm_90a ;;
        [0-9].[0-9]) MOJOLEARN_GPU_ARCHS="sm_$(echo "$cap" | tr -d .)" ;;
        *) say "arch unknown (compute_cap='$cap')"; exit 2 ;;
    esac
fi
export MOJOLEARN_GPU_ARCHS
say "gpu_archs=$MOJOLEARN_GPU_ARCHS"

# ---- the published wheel, in a venv; no build ----
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
say "wheel: $(cut -c1-100 "$OUT/wheel.sha256" 2>/dev/null)"
unset PYTHONPATH
PYBIN=/root/lm-venv/bin/python
"$PYBIN" -c 'import mojolearn, sys; b = mojolearn._backend.binding("_mojolearn_byte_lm", "identical"); print("mojolearn", getattr(mojolearn, "__version__", "?"), mojolearn.__file__, "vendor", b.byte_lm_vendor(), "set_lr", callable(getattr(b, "byte_lm_parallel_set_lr", None)), "device_fold", callable(getattr(b, "byte_lm_parallel_fold_export", None)))' > "$OUT/binding.txt" 2>&1
say "binding: $(tail -1 "$OUT/binding.txt" | cut -c1-200)"

# ---- ranged fetches: 100 MB ranges, twenty at a time, each with a speed floor ----
cat > "$OUT/fetch_ranges.py" <<'PY'
import json, os, subprocess, sys, time
from concurrent.futures import ThreadPoolExecutor
urls, out = json.load(open(sys.argv[1])), sys.argv[2]
CHUNK = 100_000_000
def size_of(url):
    # a presigned GET refuses HEAD (403); one byte's Content-Range says the total
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

# ---- the token stream ----
cat > "$OUT/tokens_urls.json" <<'TOKENS_URLS'
@TOKENS_URLS@
TOKENS_URLS
TOK=/root/tokens_stream; mkdir -p "$TOK"
_t0=$(date +%s)
"$PYBIN" "$OUT/fetch_ranges.py" "$OUT/tokens_urls.json" "$TOK" > "$OUT/tokens_fetch.log" 2>&1; _rc=$?
say "tokens fetched exit=$_rc secs=$(( $(date +%s) - _t0 ))"
[ "$_rc" -eq 0 ] || { say "token fetch failed; nothing run"; exit 3; }
_t0=$(date +%s)
cat "$TOK"/tokens.i32.part?? > "$TOK/tokens.i32" && rm -f "$TOK"/tokens.i32.part??
say "joined tokens in $(( $(date +%s) - _t0 )) s: $(wc -c < "$TOK/tokens.i32" | tr -d ' ') bytes"

# ---- the recipe (pinned), the checkpoint, the expected chain ----
curl -fsS --retry 3 -o "$OUT/in/recipe.json" '@RECIPE_URL@' >> "$OUT/fetch.log" 2>&1
printf '%s  %s\n' "$RECIPE_SHA" "$OUT/in/recipe.json" | sha256sum -c - >> "$OUT/fetch.log" 2>&1 || { say "recipe FAILED its sha256; nothing run"; exit 4; }
cat > "$OUT/ckpt_urls.json" <<'CKPT_URLS'
@CKPT_URLS@
CKPT_URLS
_t0=$(date +%s)
"$PYBIN" "$OUT/fetch_ranges.py" "$OUT/ckpt_urls.json" "$OUT/in" > "$OUT/ckpt_fetch.log" 2>&1; _rc=$?
say "checkpoint fetched exit=$_rc secs=$(( $(date +%s) - _t0 ))"
[ "$_rc" -eq 0 ] || { say "checkpoint fetch failed; nothing run"; exit 4; }
sha256sum "$OUT/in/$FROM_NAME" > "$OUT/ckpt.sha256"
say "checkpoint sha256 $(cut -c1-64 "$OUT/ckpt.sha256")"
cat > "$OUT/in/expect_chain.jsonl" <<'EXPECT_CHAIN'
@EXPECT_LINES@
EXPECT_CHAIN
sha256sum "$OUT/in/expect_chain.jsonl" > "$OUT/expect_chain.sha256"

# ---- the controls, in sequence, each in its own directory ----
R="$OUT/in/recipe.json"
S=tools/lm_segment.py
run() {  # $1 name, rest: extra lm_segment run args
    _n="$1"; shift
    _t0=$(date +%s)
    "$PYBIN" $S run --recipe "$R" --tokens "$TOK" --from "$OUT/in/$FROM_NAME" --steps 2 --devices 0 \
        --route A --segment "controls-$_n" --label "nvidia-controls-$_n" --no-checkpoints \
        --expect-chain "$OUT/in/expect_chain.jsonl" --out "$OUT/$_n" "$@" > "$OUT/$_n.log" 2>&1; _rc=$?
    say "$_n exit=$_rc secs=$(( $(date +%s) - _t0 )): $(grep -E 'PASS:|FAIL:|REFUSED|DISAGREE|Error|error' "$OUT/$_n.log" | tail -2 | tr '\n' ' ' | cut -c1-300)"
    return $_rc
}
run positive || { say "THE POSITIVE CONTROL FAILED: the published wheel did not reproduce the chain; nothing else is run"; exit 5; }
for c in $CONTROLS; do
    name=${c%%:*}; flag=${c#*:}
    case "$flag" in
        zero-moments) run "$name" --zero-moments ;;
        *) run "$name" --control "$flag" ;;
    esac
done
rm -f "$OUT/in/$FROM_NAME"
say "finished=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
exit 0
