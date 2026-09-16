# rf-score-weighted nondeterminism, leg 2: THE n_streams DISCRIMINATOR.
# Installs the 0.8.6 RELEASE WHEEL from R2 and repeats the regressor fit at
# n_streams = 4 (the shipped default, the control) and n_streams = 1 (the serialized
# pipeline, the test), hashing the exported model arrays per fit and per tree.
#   K=1 stable while K=4 moves  -> the race is the pipelined shared SplitStaging.
#   both move                   -> it is inside one tree's kernels.
# Placeholders: @FULL@ @WHEELNAME@ @WHEELSHA@ @WHEELURL@ @PUTURL@ @REPEATS@ @PROBESECS@
set -u
OUT=/root/gemm_leg_out/identity
mkdir -p "$OUT"
cd /root/mojolearn || exit 9
echo @FULL@ > /root/mojolearn/commit.txt
log() { echo "$(date -u +%H:%M:%SZ) $*" >> "$OUT/record.txt"; }
log "started rf n_streams discriminator"

PY=/root/mojolearn/.pixi/envs/default/bin/python
[ -x "$PY" ] || PY=$(command -v python3)
W=/root/wheel/@WHEELNAME@
mkdir -p /root/wheel
curl -fsS --retry 3 --max-time 600 -o "$W" '@WHEELURL@'; log "wheel_fetch_exit=$?"
got=$(sha256sum "$W" | cut -d' ' -f1); log "wheel_sha256=$got"
[ "$got" = @WHEELSHA@ ] || { log "WHEEL SHA MISMATCH"; exit 11; }
"$PY" -m venv /root/q > "$OUT/venv.log" 2>&1 || { log "venv failed"; exit 12; }
/root/q/bin/pip install --disable-pip-version-check -q "$W" numpy >> "$OUT/venv.log" 2>&1; log "pip_exit=$?"

mkdir -p /root/run && cd /root/run || exit 13
export MOJOLEARN_NUMERIC_MODE=identical MOJOLEARN_COMMIT=@FULL@ PYTHONNOUSERSITE=1
unset PYTHONPATH
/root/q/bin/python -c "import mojolearn; print(mojolearn.__version__, mojolearn.vendor())" > "$OUT/import.txt" 2>&1
log "import $(tr '\n' ' ' < "$OUT/import.txt")"
{ rocm-smi --showproductname 2>&1 | head -20; } > "$OUT/gpu.txt" 2>&1 || true

JSON="$OUT/rf_probe2.json"
upload() { [ -s "$JSON" ] || return 1; curl -fsS --max-time 180 -X PUT --upload-file "$JSON" '@PUTURL@' > /dev/null 2>&1; }
( while :; do sleep 60; if upload; then log "partial_uploaded bytes=$(wc -c < "$JSON" 2>/dev/null | tr -d ' ')"; fi; done ) &
UPLOADER=$!
log "uploader_pid=$UPLOADER"

cat > /root/run/rf_probe2.py <<'PYEOF'
"""n_streams discriminator for the rf-score-weighted move.

Leg 1 established: the regressor model moves in ~4% of fits, tree SHAPE is
invariant (_offsets/_left_child never move, node counts constant), _colid and
_quesval move, and exactly ONE tree differs per occurrence. Trees are built in
K = n_streams pipelined slots sharing one Builder and one SplitStaging.

This leg repeats the SAME fit at K = 4 (shipped default) and K = 1 (serialized).
"""
import hashlib, json, os, time
import numpy as np

OUT = os.environ.get("RF_PROBE_JSON", "/root/gemm_leg_out/identity/rf_probe2.json")
REPEATS = int(os.environ.get("RF_PROBE_REPEATS", "40"))
DEADLINE = time.time() + float(os.environ.get("RF_PROBE_SECS", "1500")) - 120

import mojolearn
from mojolearn import RandomForestRegressor
import mojolearn._identity_break as ib

fixture = ib.fixture
_h = ib._h
FOREST_ARRAYS = ("_offsets", "_colid", "_quesval", "_left_child", "_leaves")


def np_of(est, name):
    v = getattr(est, name)
    if isinstance(v, np.ndarray):
        return v
    for attempt in (lambda: np.asarray(memoryview(v)),
                    lambda: np.frombuffer(v.tobytes(), dtype=np.dtype(v.dtype)),
                    lambda: np.asarray(list(v))):
        try:
            return attempt()
        except Exception:
            continue
    raise RuntimeError("cannot view " + name)


def model_hashes(est):
    out, arrs = {}, {}
    for name in FOREST_ARRAYS:
        try:
            a = np_of(est, name)
            arrs[name] = a
            out[name] = _h(a)
        except Exception as exc:
            out[name] = "ERR:" + type(exc).__name__
    try:
        off = np.asarray(arrs["_offsets"]).astype(np.int64)
        nnode = len(arrs["_colid"])
        leaves = np.asarray(arrs["_leaves"])
        nout = max(1, len(leaves) // max(1, nnode))
        out["_per_tree"] = [
            _h(arrs["_colid"][int(off[t]):int(off[t + 1])],
               arrs["_quesval"][int(off[t]):int(off[t + 1])],
               arrs["_left_child"][int(off[t]):int(off[t + 1])],
               leaves[int(off[t]) * nout:int(off[t + 1]) * nout])
            for t in range(len(off) - 1)]
        out["_n_nodes"] = int(nnode)
        out["_n_trees"] = int(len(off) - 1)
    except Exception as exc:
        out["_per_tree_error"] = repr(exc)[:200]
    return out


results = {"version": getattr(mojolearn, "__version__", "?"),
           "commit": os.environ.get("MOJOLEARN_COMMIT", ""),
           "repeats_requested": REPEATS, "runs": [], "notes": [],
           "n_streams_supported": None}


def flush():
    tmp = OUT + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(results, fh, indent=1)
    os.replace(tmp, OUT)


# Prove n_streams is actually accepted and actually reaches the fit, before
# spending the box on an arm that silently ignored it.
X0, _, yr0 = fixture("base")
try:
    RandomForestRegressor(n_estimators=2, max_depth=3, random_state=7,
                          n_streams=1).fit(X0[:200], yr0[:200])
    results["n_streams_supported"] = True
except Exception as exc:
    results["n_streams_supported"] = "REFUSED: " + repr(exc)[:300]
flush()

if results["n_streams_supported"] is True:
    for ns in (4, 1):
        for kind in ("wide", "base"):
            X, yc, yr = fixture(kind)
            for i in range(REPEATS):
                if time.time() > DEADLINE:
                    results["notes"].append("deadline ns=%d %s repeat %d" % (ns, kind, i))
                    flush()
                    break
                t0 = time.time()
                try:
                    reg = RandomForestRegressor(n_estimators=16, max_depth=8,
                                                random_state=7, n_streams=ns).fit(
                                                    X[:2000], yr[:2000])
                    rec = {"reg_model": model_hashes(reg)}
                except Exception as exc:
                    rec = {"error": repr(exc)[:400]}
                rec.update({"n_streams": ns, "fixture": kind, "repeat": i,
                            "secs": round(time.time() - t0, 2)})
                results["runs"].append(rec)
                flush()

results["done"] = True
flush()
print("PROBE2 DONE runs=%d n_streams_supported=%s"
      % (len(results["runs"]), results["n_streams_supported"]))
PYEOF

RF_PROBE_JSON="$JSON" RF_PROBE_REPEATS=@REPEATS@ RF_PROBE_SECS=@PROBESECS@ \
  timeout -k 30 @PROBESECS@ /root/q/bin/python /root/run/rf_probe2.py > "$OUT/rf_probe2.log" 2>&1
log "rf_probe2_exit=$?"
tail -5 "$OUT/rf_probe2.log" >> "$OUT/record.txt"

kill "$UPLOADER" 2>/dev/null
if upload; then log "final_partial_uploaded bytes=$(wc -c < "$JSON" 2>/dev/null | tr -d ' ')"; else log "final upload FAILED (the fetch is the fallback)"; fi
log "finished"
