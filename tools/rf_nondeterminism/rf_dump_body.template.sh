# WHICH FIELD MOVES? Lane-by-lane diff of the .cand dumps on the SHIPPED 0.8.5 wheel.
#
# Leg 11 put the first difference at tree<i>.batch6.round0.cand while the SAME round's
# colsamples and BOTH column-block histograms matched bit for bit. So the selection step
# introduces it. _record_splits packs 10 u32 lanes per split:
#   0 is_valid | 1 colid | 2,3 quesval | 4,5 best_metric_val | 6,7 global_n_left | 8,9 local_n_left
#
#   best_metric_val SAME, colid DIFFERS -> gains matched, the merge chose differently
#   best_metric_val DIFFERS             -> a gain from an identical histogram moved:
#                                          a lost candidate or a partial reduction read
#
# The race is confirmed in published 0.8.5: 13/300 at the shipped default (max_features=1.0,
# >=11 features), 0/300 at 10 columns. Exposure is understood; the MECHANISM is not.
#
# NO REBUILD IS NEEDED. `instr.trace.enabled` is a RUNTIME check inside fit_forest, which
# constructs a live FitInstruments() whose IdentityTrace() reads getenv(MOJOLEARN_IDENTITY_TRACE).
# So the shipped binding emits an ordered stage trace from an env var alone, and because the
# trace object is rebuilt per fit_forest call, a DISTINCT path per fit gives one trace per fit.
#
# THE DECISIVE READING. _compute_split records tree<id>.batch<n>.round<r>.cols<col>.hist BEFORE
# each split launch. Nothing checkpoints split[]. So for a fit whose MODEL moved:
#   traces IDENTICAL  -> every histogram matched; the divergence is downstream of accumulation,
#                        i.e. in SPLIT SELECTION (the merge): stale read vs dropped write-back.
#   a .hist DIFFERS   -> accumulation itself moved, upstream of the merge.
#
# ARM 1 is UNTRACED and is the POSITIVE CONTROL. record_device DRAINS the queue, so tracing
# changes concurrency and may mask the race; a traced null means nothing without arm 1 moving.
# Placeholders: @FULL@ @PUTURL@ @REPEATS@ @TRACEREPEATS@ @PROBESECS@
set -u
OUT=/root/gemm_leg_out/identity
mkdir -p "$OUT"
cd /root/mojolearn || exit 9
echo @FULL@ > /root/mojolearn/commit.txt
log() { echo "$(date -u +%H:%M:%SZ) $*" >> "$OUT/record.txt"; }
log "started STAGE-TRACE diff on shipped 0.8.5"

PY=/root/mojolearn/.pixi/envs/default/bin/python
[ -x "$PY" ] || PY=$(command -v python3)
"$PY" -m venv /root/q > "$OUT/venv.log" 2>&1 || { log "venv failed"; exit 12; }
/root/q/bin/pip install --disable-pip-version-check -q "mojolearn==0.8.5" numpy > "$OUT/pip.log" 2>&1
log "pip_install_0.8.5_exit=$?"

mkdir -p /root/run /root/traces && cd /root/run || exit 13
export MOJOLEARN_NUMERIC_MODE=identical PYTHONNOUSERSITE=1
unset PYTHONPATH

JSON="$OUT/rf_dump.json"
upload() { [ -s "$JSON" ] || return 1; curl -fsS --max-time 180 -X PUT --upload-file "$JSON" '@PUTURL@' > /dev/null 2>&1; }
( while :; do sleep 60; if upload; then log "partial_uploaded bytes=$(wc -c < "$JSON" 2>/dev/null | tr -d ' ')"; fi; done ) &
UPLOADER=$!
log "uploader_pid=$UPLOADER"

cat > /root/run/rf_trace.py <<'PYEOF'
"""Name the first diverging STAGE, or prove no recorded stage diverges."""
import importlib.util, json, os, time
import numpy as np

OUT = os.environ.get("RF_PROBE_JSON", "/root/gemm_leg_out/identity/rf_dump.json")
REPEATS = int(os.environ.get("RF_PROBE_REPEATS", "100"))
TRACE_REPEATS = int(os.environ.get("RF_TRACE_REPEATS", "150"))
DEADLINE = time.time() + float(os.environ.get("RF_PROBE_SECS", "1500")) - 120
TRACE_DIR = "/root/traces"

import mojolearn
from mojolearn import RandomForestRegressor

_spec = importlib.util.spec_from_file_location(
    "ib", "/root/mojolearn/tools/identity_break.py")
ib = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ib)
fixture, _h = ib.fixture, ib._h

NAMES = ("_offsets", "_colid", "_quesval", "_left_child", "_leaves")

LANES = ("is_valid", "colid", "quesval_lo", "quesval_hi", "metric_lo", "metric_hi",
         "gnleft_lo", "gnleft_hi", "lnleft_lo", "lnleft_hi")


def _sidecar(trace_path, seq, tag):
    """`<trace>.<seq>.<tag>.bin`, per the trace tool's dump contract."""
    return "%s.%d.%s.bin" % (trace_path, seq, tag)


def compare_cand(ref_trace_path, mov_trace_path, idx, tag):
    """Lane-by-lane diff of two .cand dumps, addressed by the record's own seq."""
    import numpy as _np
    a_p = _sidecar(ref_trace_path, idx, tag)
    b_p = _sidecar(mov_trace_path, idx, tag)
    if not (os.path.exists(a_p) and os.path.exists(b_p)):
        return {"error": "sidecar missing", "a": a_p, "b": b_p}
    a = _np.fromfile(a_p, dtype=_np.uint32)
    b = _np.fromfile(b_p, dtype=_np.uint32)
    if a.shape != b.shape or a.size % 10:
        return {"error": "shape", "a": int(a.size), "b": int(b.size)}
    a = a.reshape(-1, 10); b = b.reshape(-1, 10)
    rows = _np.nonzero((a != b).any(axis=1))[0]
    per_lane = {LANES[j]: int((a[:, j] != b[:, j]).sum()) for j in range(10)}
    out = {"splits": int(a.shape[0]), "differing_splits": [int(r) for r in rows[:8]],
           "per_lane_differing_counts": per_lane}
    if len(rows):
        r = int(rows[0])
        out["first"] = {"split_index": r,
                        "ref": {LANES[j]: int(a[r, j]) for j in range(10)},
                        "cur": {LANES[j]: int(b[r, j]) for j in range(10)}}
        same_metric = (a[r, 4] == b[r, 4]) and (a[r, 5] == b[r, 5])
        out["reading"] = ("GAINS MATCH, selection differs (merge/tie path)" if same_metric
                          else "GAIN ITSELF MOVED from an identical histogram")
    return out



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


def model_key(est):
    return tuple(_h(np.array(np_of(est, n), copy=True)) for n in NAMES)


def fit_once():
    return RandomForestRegressor(n_estimators=16, max_depth=8,
                                 random_state=7).fit(X[:2000], yr[:2000])


def read_trace(path):
    try:
        with open(path) as fh:
            return [ln.rstrip("\n") for ln in fh if ln.strip()]
    except Exception:
        return []


def first_diff(a, b):
    """(index, tag_a, tag_b, kind) of the first differing record, or None."""
    for i in range(min(len(a), len(b))):
        if a[i] != b[i]:
            fa, fb = a[i].split("\t"), b[i].split("\t")
            ta = fa[1] if len(fa) > 1 else "?"
            tb = fb[1] if len(fb) > 1 else "?"
            return {"index": i, "tag_a": ta, "tag_b": tb,
                    "kind": "TAG_SET_DIFFERS" if ta != tb else "HASH_DIFFERS",
                    "rec_a": a[i][:160], "rec_b": b[i][:160]}
    if len(a) != len(b):
        return {"index": min(len(a), len(b)), "tag_a": "<end>", "tag_b": "<end>",
                "kind": "LENGTH_DIFFERS", "rec_a": str(len(a)), "rec_b": str(len(b))}
    return None


results = {"version": getattr(mojolearn, "__version__", "?"),
           "question": "does any RECORDED stage diverge when the model moves",
           "untraced_control": {}, "traced": {}, "divergences": [], "notes": []}


def flush():
    tmp = OUT + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(results, fh, indent=1)
    os.replace(tmp, OUT)


X, yc, yr = fixture("wide")

# ---- ARM 1: UNTRACED POSITIVE CONTROL ---------------------------------
os.environ.pop("MOJOLEARN_IDENTITY_TRACE", None)
ref = None
moved = 0
runs = 0
for i in range(REPEATS):
    if time.time() > DEADLINE:
        break
    k = model_key(fit_once())
    runs += 1
    if ref is None:
        ref = k
    elif k != ref:
        moved += 1
    results["untraced_control"] = {"runs": runs, "moved": moved}
    flush()
results["untraced_control"]["verdict"] = (
    "REPRODUCES" if moved else "DID NOT REPRODUCE - traced arm is uninterpretable")
flush()

# ---- ARM 2: TRACED ----------------------------------------------------
ref_key = None
ref_trace = None
ref_path_dir = None
tmoved = 0
truns = 0
identical_trace_but_moved = 0
for i in range(TRACE_REPEATS):
    if time.time() > DEADLINE:
        results["notes"].append("deadline in traced arm at %d" % i)
        break
    path = os.path.join(TRACE_DIR, "fit_%04d.trace" % i)
    # IdentityTrace() is constructed per fit_forest call and re-reads getenv,
    # so a distinct path per fit yields one clean trace per fit.
    os.environ["MOJOLEARN_IDENTITY_TRACE"] = path
    os.environ["MOJOLEARN_IDENTITY_TRACE_DUMP"] = "cand"
    try:
        k = model_key(fit_once())
    except Exception as exc:
        results["notes"].append("traced fit %d failed: %r" % (i, exc))
        flush()
        break
    tr = read_trace(path)
    truns += 1
    if ref_key is None:
        ref_key, ref_trace, ref_path_dir = k, tr, path
        results["traced"]["ref_records"] = len(tr)
        flush()
        continue
    if k == ref_key:
        try:
            os.remove(path)       # keep only the reference and the movers
        except Exception:
            pass
        continue
    tmoved += 1
    d = first_diff(ref_trace, tr)
    if d is not None and d.get("kind") == "HASH_DIFFERS":
        # THE SIDECAR IS NAMED BY THE RECORD'S OWN seq, NOT by its line index.
        # Leg 11: list index 141 carried seq 139 (the file has a preamble), so
        # passing the index would name a file that does not exist and every
        # comparison would return "sidecar missing" -- a silent null that reads
        # like a result.
        _seq = int(ref_trace[d["index"]].split("\t")[0])
        fields = compare_cand(ref_path_dir, path, _seq, d["tag_a"])
        if fields is not None:
            d["field_analysis"] = fields
    if d is None:
        identical_trace_but_moved += 1
    results["divergences"].append({
        "repeat": i, "trace_records": len(tr), "ref_records": len(ref_trace),
        "first_diff": d,
        "reading": ("SPLIT SELECTION (no recorded stage moved)" if d is None
                    else "ACCUMULATION or earlier (%s at %s)" % (d["kind"], d["tag_a"]))})
    results["traced"] = {"runs": truns, "moved": tmoved,
                         "ref_records": results["traced"].get("ref_records"),
                         "identical_trace_but_model_moved": identical_trace_but_moved}
    flush()

results["traced"].update({"runs": truns, "moved": tmoved,
                          "identical_trace_but_model_moved": identical_trace_but_moved})
results["done"] = True
flush()
print("TRACE PROBE DONE untraced %s/%s traced %s/%s identical-trace-but-moved %s" % (
    results["untraced_control"].get("moved"), results["untraced_control"].get("runs"),
    tmoved, truns, identical_trace_but_moved))
PYEOF

RF_PROBE_JSON="$JSON" RF_PROBE_REPEATS=@REPEATS@ RF_TRACE_REPEATS=@TRACEREPEATS@ \
RF_PROBE_SECS=@PROBESECS@ \
  timeout -k 30 @PROBESECS@ /root/q/bin/python /root/run/rf_trace.py > "$OUT/rf_trace.log" 2>&1
log "rf_trace_exit=$?"
tail -8 "$OUT/rf_trace.log" >> "$OUT/record.txt"

# Keep the reference trace and any mover traces; they are small and are the evidence.
tar czf "$OUT/traces.tar.gz" -C /root traces 2>/dev/null || true
log "traces_archived bytes=$(wc -c < "$OUT/traces.tar.gz" 2>/dev/null | tr -d ' ')"

kill "$UPLOADER" 2>/dev/null
if upload; then log "final_uploaded bytes=$(wc -c < "$JSON" 2>/dev/null | tr -d ' ')"; else log "final upload FAILED (the fetch is the fallback)"; fi
log "finished"
