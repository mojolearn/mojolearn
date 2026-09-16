# rf-score-weighted nondeterminism, leg 3: NODE-LEVEL LOCALIZATION.
# Legs 1 and 2 stored per-tree hashes only, so the differing NODE was never named.
# This leg keeps the first fit's arrays as a reference and, on any divergence, dumps
# the exact node indices with both competing (colid, quesval) values and the node's
# depth, so a near-tied gain at a small deep node is distinguishable from a corrupt cell.
# Placeholders: @FULL@ @WHEELNAME@ @WHEELSHA@ @WHEELURL@ @PUTURL@ @REPEATS@ @PROBESECS@
set -u
OUT=/root/gemm_leg_out/identity
mkdir -p "$OUT"
cd /root/mojolearn || exit 9
echo @FULL@ > /root/mojolearn/commit.txt
log() { echo "$(date -u +%H:%M:%SZ) $*" >> "$OUT/record.txt"; }
log "started rf node-level localization"

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

JSON="$OUT/rf_probe3.json"
upload() { [ -s "$JSON" ] || return 1; curl -fsS --max-time 180 -X PUT --upload-file "$JSON" '@PUTURL@' > /dev/null 2>&1; }
( while :; do sleep 60; if upload; then log "partial_uploaded bytes=$(wc -c < "$JSON" 2>/dev/null | tr -d ' ')"; fi; done ) &
UPLOADER=$!
log "uploader_pid=$UPLOADER"

cat > /root/run/rf_probe3.py <<'PYEOF'
"""Name the diverging NODE.

Leg 1: ~4% of regressor fits move; tree shape invariant; _colid and _quesval move;
exactly one tree per occurrence. Leg 2: K=1 moves as often as K=4, so it is not the
pipelined staging. Reading has exhausted the reachable mechanisms, so this leg asks
the data which node disagrees and what the two candidates were.
"""
import json, os, time
import numpy as np

OUT = os.environ.get("RF_PROBE_JSON", "/root/gemm_leg_out/identity/rf_probe3.json")
REPEATS = int(os.environ.get("RF_PROBE_REPEATS", "60"))
DEADLINE = time.time() + float(os.environ.get("RF_PROBE_SECS", "1500")) - 120

import mojolearn
from mojolearn import RandomForestRegressor
import mojolearn._identity_break as ib

fixture = ib.fixture
_h = ib._h
NAMES = ("_offsets", "_colid", "_quesval", "_left_child", "_leaves")


def np_of(est, name):
    v = getattr(est, name)
    if isinstance(v, np.ndarray):
        return v
    for a in (lambda: np.asarray(memoryview(v)),
              lambda: np.frombuffer(v.tobytes(), dtype=np.dtype(v.dtype)),
              lambda: np.asarray(list(v))):
        try:
            return a()
        except Exception:
            continue
    raise RuntimeError("cannot view " + name)


def arrays(est):
    return {n: np.array(np_of(est, n), copy=True) for n in NAMES}


def depth_of(left_child, offsets, t, node_local):
    """Depth by walking down from the tree root, following child links."""
    lo, hi = int(offsets[t]), int(offsets[t + 1])
    depth = {0: 0}
    for i in range(hi - lo):
        lc = int(left_child[lo + i])
        if lc > 0 and i in depth:
            depth.setdefault(lc, depth[i] + 1)
            depth.setdefault(lc + 1, depth[i] + 1)
    return depth.get(node_local, -1)


results = {"version": getattr(mojolearn, "__version__", "?"),
           "commit": os.environ.get("MOJOLEARN_COMMIT", ""),
           "repeats_requested": REPEATS, "fixtures": {}, "divergences": [],
           "notes": []}


def flush():
    tmp = OUT + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(results, fh, indent=1)
    os.replace(tmp, OUT)


for kind in ("wide", "base"):
    X, yc, yr = fixture(kind)
    ref = None
    nstable = 0
    nmoved = 0
    for i in range(REPEATS):
        if time.time() > DEADLINE:
            results["notes"].append("deadline %s repeat %d" % (kind, i))
            break
        try:
            reg = RandomForestRegressor(n_estimators=16, max_depth=8,
                                        random_state=7).fit(X[:2000], yr[:2000])
            cur = arrays(reg)
        except Exception as exc:
            results["notes"].append("fit error %s %d %r" % (kind, i, exc))
            flush()
            continue
        if ref is None:
            ref = cur
            results["fixtures"][kind] = {
                "n_nodes": int(len(cur["_colid"])),
                "n_trees": int(len(cur["_offsets"]) - 1),
                "ref_hash": _h(cur["_colid"], cur["_quesval"], cur["_leaves"]),
            }
            flush()
            continue
        same = all(_h(cur[n]) == _h(ref[n]) for n in NAMES)
        if same:
            nstable += 1
            continue
        nmoved += 1
        rec = {"fixture": kind, "repeat": i,
               "shape_same": bool(_h(cur["_offsets"]) == _h(ref["_offsets"])
                                  and _h(cur["_left_child"]) == _h(ref["_left_child"])),
               "arrays_differing": [n for n in NAMES if _h(cur[n]) != _h(ref[n])]}
        off = np.asarray(ref["_offsets"]).astype(np.int64)
        bad = np.nonzero((np.asarray(cur["_colid"]) != np.asarray(ref["_colid"]))
                         | (np.asarray(cur["_quesval"]) != np.asarray(ref["_quesval"])))[0]
        rec["n_differing_nodes"] = int(len(bad))
        details = []
        for gi in bad[:24]:
            gi = int(gi)
            t = int(np.searchsorted(off, gi, side="right") - 1)
            local = gi - int(off[t])
            details.append({
                "tree": t, "node_global": gi, "node_local": local,
                "depth": depth_of(ref["_left_child"], off, t, local),
                "left_child": int(ref["_left_child"][gi]),
                "is_leaf": bool(int(ref["_left_child"][gi]) <= 0),
                "ref_colid": int(ref["_colid"][gi]), "cur_colid": int(cur["_colid"][gi]),
                "ref_quesval": float(ref["_quesval"][gi]),
                "cur_quesval": float(cur["_quesval"][gi]),
                "same_colid": bool(int(ref["_colid"][gi]) == int(cur["_colid"][gi])),
            })
        rec["nodes"] = details
        results["divergences"].append(rec)
        flush()
    results["fixtures"].setdefault(kind, {})["stable"] = nstable
    results["fixtures"][kind]["moved"] = nmoved
    flush()

results["done"] = True
flush()
print("PROBE3 DONE divergences=%d" % len(results["divergences"]))
PYEOF

RF_PROBE_JSON="$JSON" RF_PROBE_REPEATS=@REPEATS@ RF_PROBE_SECS=@PROBESECS@ \
  timeout -k 30 @PROBESECS@ /root/q/bin/python /root/run/rf_probe3.py > "$OUT/rf_probe3.log" 2>&1
log "rf_probe3_exit=$?"
tail -5 "$OUT/rf_probe3.log" >> "$OUT/record.txt"

kill "$UPLOADER" 2>/dev/null
if upload; then log "final_uploaded bytes=$(wc -c < "$JSON" 2>/dev/null | tr -d ' ')"; else log "final upload FAILED (the fetch is the fallback)"; fi
log "finished"
