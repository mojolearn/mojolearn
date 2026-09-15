set -u
st() { echo "$1	$2" >> "$LEG_OUT/status.tsv"; echo "$(date -u +%T) $1 rc=$2" >> "$LEG_OUT/box.txt"; }
echo "commit=$MOJOLEARN_COMMIT nproc=$(nproc)" > "$LEG_OUT/box.txt"
ls -la python/mojolearn/host python/mojolearn/host-sabotage > "$LEG_OUT/bindings.txt" 2>&1
H=python/mojolearn/host
if [ ! -f $H/_mojolearn_neural_host.so ]; then st neural-missing 1; exit 1; fi
export MOJOLEARN_GATE_COMMIT=$MOJOLEARN_COMMIT
PY=python3
TPY=/root/mojolearn/.pixi/envs/test/bin/python
PKG=/root/mojolearn/.pixi/envs/pkg/bin/python

# ---- nm: no backward, optimizer, loss or decode symbol in the shipped binding
for so in $H/_mojolearn_neural_host.so $H/_mojolearn_training_host.so $H/_mojolearn_mamba_host.so $H/_mojolearn_transformer_host.so; do
  n=$(nm --defined-only "$so" 2>/dev/null | grep -ciE 'backward|optimizer|adamw|decode|clip_grad|ce_loss|train_step' || true)
  t=$(nm --defined-only "$so" 2>/dev/null | wc -l)
  echo "$(basename $so) $(stat -c %s $so) bytes, $t defined symbols, nm backward|optimizer|adamw|decode|clip_grad|ce_loss|train_step = $n" >> "$LEG_OUT/nm.txt"
done
nm --defined-only $H/_mojolearn_neural_host.so > "$LEG_OUT/nm-neural.txt" 2>&1

# ---- pytest (source tree, reference bindings present)
( cd python && for t in test_neural_inference test_host_surface test_cpu_inference_boundary; do
    MOJOLEARN_HOST_DIR=mojolearn/host $TPY -m pytest -q -p no:cacheprovider mojolearn/tests/$t.py > "$LEG_OUT/pytest_$t.log" 2>&1
    echo "pytest $t rc=$? $(tail -1 "$LEG_OUT/pytest_$t.log")" >> "$LEG_OUT/box.txt"
  done )

PARTS="--batch-grad --batch-scale --ragged"
LANESET="mamba1 mamba2 mamba3 mamba2-dtlimit transformer transformer-window samba samba-untied-dropout-accum byte-lm byte-lm-resident"
run_groups() {  # tag hostdir fixtures repeats extra-env...
  tag=$1; hd=$2; fx=$3; rp=$4; shift 4
  pids=""; files=""
  for g in $LANESET; do
    gn=$(echo $g | tr , _)
    env "$@" MOJOLEARN_HOST_DIR=$hd $PY tools/identity_break.py --lanes $g ${fx:+--fixtures $fx} --repeats $rp $PARTS \
      --json "$LEG_OUT/parts/$tag.$gn.json" > "$LEG_OUT/run.$tag.$gn.log" 2>&1 &
    pids="$pids $!"; files="$files $LEG_OUT/parts/$tag.$gn.json"
  done
  rc=0; for p in $pids; do wait $p || rc=1; done
  st "run-$tag" $rc
  [ "$tag" = batchsab ] && return 0
  $PY tools/identity_break.py --merge $files --json "$LEG_OUT/cpu-x86.$tag.json" > "$LEG_OUT/merge.$tag.log" 2>&1; st "merge-$tag" $?
}
mkdir -p "$LEG_OUT/parts"
s=$(date +%s)
run_groups prod $H "" 2 X=1
echo "prod $(( $(date +%s) - s ))s" >> "$LEG_OUT/box.txt"

# ---- neural host sabotage: every reference binding clean, only the shipped
# neural binding and the byte LM inference binary built with the sabotage
MIX=/tmp/host-mix; rm -rf $MIX; mkdir -p $MIX; cp -a $H/. $MIX/
cp python/mojolearn/host-sabotage/_mojolearn_neural_host.so $MIX/_mojolearn_neural_host.so
cp python/mojolearn/host-sabotage/_mojolearn_byte_lm_host.so $MIX/_mojolearn_byte_lm_host.so
cmp -s $MIX/_mojolearn_neural_host.so $H/_mojolearn_neural_host.so && st neural-sabotage-identical-to-prod 1
s=$(date +%s)
run_groups hostsab $MIX "" 1 MOJOLEARN_HOST_ALLOW_SABOTAGE=1 MOJOLEARN_BYTE_LM_HOST_ALLOW_SABOTAGE=1
echo "hostsab $(( $(date +%s) - s ))s" >> "$LEG_OUT/box.txt"

# ---- batch sabotage (base fixture, one repeat)
s=$(date +%s)
run_groups batchsab $H base 1 MOJOLEARN_IDENTITY_BATCH_SABOTAGE=1
echo "batchsab $(( $(date +%s) - s ))s" >> "$LEG_OUT/box.txt"

# ---- the test wheel: shipped families only, installed into an isolated target
W=/tmp/wheel; rm -rf "$W"; mkdir -p "$W/dist"
cp -a python "$W/src"
rm -rf "$W/src/mojolearn/host-sabotage" "$W/src/mojolearn/host" "$W/src/build" "$W/src"/*.egg-info
mkdir -p "$W/src/mojolearn/host"
for b in $($PY python/mojolearn/host_surface.py --wheel-bindings); do cp "$H/$b.so" "$W/src/mojolearn/host/"; done
find "$W/src" -name __pycache__ -prune -exec rm -rf {} + 2>/dev/null
( cd "$W/src" && $PKG -m build --wheel --no-isolation -o "$W/dist" ) > "$LEG_OUT/wheel.log" 2>&1; st wheel $?
ls -la "$W/dist" > "$LEG_OUT/wheel-size.txt" 2>&1
$PKG - "$W/dist" > "$LEG_OUT/wheel-host-entries.txt" 2>&1 <<'PY'
import glob, sys, zipfile
for w in glob.glob(sys.argv[1] + "/*.whl"):
    z = zipfile.ZipFile(w)
    print(w, sum(i.compress_size for i in z.infolist()), "compressed")
    for i in z.infolist():
        if "/host/" in i.filename:
            print(i.filename, i.file_size, i.compress_size)
PY
# No env on the pod carries pip. A py3-none wheel with no scripts installs by
# extraction; RECORD's sha256 of every file is checked against the archive.
$PKG - "$W"/dist/*.whl /tmp/target > "$LEG_OUT/install.log" 2>&1 <<'PY'
import base64, csv, hashlib, io, sys, zipfile
whl, target = sys.argv[1], sys.argv[2]
z = zipfile.ZipFile(whl)
record = next(n for n in z.namelist() if n.endswith(".dist-info/RECORD"))
checked = 0
for row in csv.reader(io.TextIOWrapper(z.open(record), "utf-8")):
    name, digest = row[0], row[1]
    if not digest:
        continue
    algo, want = digest.split("=", 1)
    got = base64.urlsafe_b64encode(hashlib.new(algo, z.read(name)).digest()).rstrip(b"=").decode()
    assert got == want, name
    checked += 1
z.extractall(target)
print("extracted", whl, "into", target, "RECORD sha256 checked", checked)
PY
st install $?
ls /tmp/target/mojolearn/host >> "$LEG_OUT/install.log" 2>&1

cat > /tmp/wheel_calls.py <<'PY'
"""Every new public call, hashed: run once from the source tree and once
from the installed wheel; the two JSONs must be equal."""
import hashlib, json, os, sys, tempfile
import numpy as np
import mojolearn as ml
out = {"mojolearn": os.path.dirname(ml.__file__), "vendor": ml.vendor()}
h = lambda a: hashlib.sha256(np.ascontiguousarray(np.asarray(a)).tobytes()).hexdigest()[:16]
rng = np.random.default_rng(2026)
def mw(kind, dm=32):
    di = 2 * dm; nh = di // 64
    if kind == "m1":
        r = -(-dm // 16)
        s = {"norm.weight": (dm,), "in_proj.weight": (2 * di, dm), "conv1d.weight": (di, 1, 4), "conv1d.bias": (di,),
             "x_proj.weight": (r + 32, di), "dt_proj.weight": (di, r), "dt_proj.bias": (di,), "A_log": (di, 16),
             "D": (di,), "out_proj.weight": (dm, di)}
    elif kind == "m2":
        s = {"block_norm.weight": (dm,), "in_proj.weight": (2 * di + 256 + nh, dm), "conv1d.weight": (di + 256, 1, 4),
             "conv1d.bias": (di + 256,), "dt_bias": (nh,), "A_log": (nh,), "D": (nh,), "norm.weight": (di,),
             "out_proj.weight": (dm, di)}
    else:
        s = {"block_norm.weight": (dm,), "in_proj.weight": (2 * di + 256 + 3 * nh + 32, dm), "dt_bias": (nh,),
             "B_norm.weight": (128,), "C_norm.weight": (128,), "B_bias": (nh, 128), "C_bias": (nh, 128), "D": (nh,),
             "out_proj.weight": (dm, di)}
    return {n: (rng.standard_normal(v) * 0.1).astype(np.float32) for n, v in s.items()}
x = rng.standard_normal((3, 70, 32)).astype(np.float32)
lens = [70, 9, 65]
for name, blk in (("Mamba1BlockInference", ml.Mamba1BlockInference(mw("m1"))),
                  ("Mamba2BlockInference", ml.Mamba2BlockInference(mw("m2"))),
                  ("Mamba2BlockInference.dt_limit", ml.Mamba2BlockInference(mw("m2"), dt_limit=(0.01, 0.1))),
                  ("Mamba3BlockInference", ml.Mamba3BlockInference(mw("m3")))):
    out[name + ".forward"] = h(blk.forward(x))
    out[name + ".forward(lengths)"] = h(blk.forward(x, lengths=lens))
for tied in (True, False):
    cfg = ml.SambaConfig(vocab=256, d_model=32, layers=("mamba3", "attention"), n_heads=2, intermediate=64,
                         tie_embeddings=tied)
    w = {n: (rng.standard_normal(s) * 0.1).astype(np.float32) for n, s in cfg.registry()}
    inf = ml.SambaInference(cfg, w)
    ids = rng.integers(0, 256, (3, 20)).astype(np.int32)
    out[f"SambaInference(tied={tied}).forward"] = h(inf.forward(ids))
    out[f"SambaInference(tied={tied}).forward(lengths)"] = h(inf.forward(ids, lengths=[20, 1, 7]))
shape = ml.ByteLanguageModelConfig()
p = (rng.standard_normal(shape.n_total) * 0.05).astype(np.float32)
ids = rng.integers(0, 256, (4, shape.length)).astype(np.int32)
for threaded in (False, True):
    lm = ml.LanguageModelInference(p, shape=shape, threaded=threaded)
    out[f"LanguageModelInference(threaded={threaded}).logits"] = h(lm.logits(ids))
    out[f"LanguageModelInference(threaded={threaded}).logits(lengths)"] = h(lm.logits(ids, lengths=[32, 1, 5, 17]))
json.dump(out, open(sys.argv[1], "w"), indent=1, sort_keys=True)
print(len(out) - 2, "calls hashed from", out["mojolearn"])
PY
( cd /tmp && env -u PYTHONPATH -u MOJOLEARN_HOST_DIR PYTHONPATH=/root/mojolearn/python MOJOLEARN_NUMERIC_MODE=identical \
    $PY /tmp/wheel_calls.py "$LEG_OUT/calls.source.json" ) > "$LEG_OUT/calls.source.log" 2>&1; st calls-source $?
( cd /tmp && env -u PYTHONPATH -u MOJOLEARN_HOST_DIR PYTHONPATH=/tmp/target MOJOLEARN_NUMERIC_MODE=identical \
    $PY /tmp/wheel_calls.py "$LEG_OUT/calls.wheel.json" ) > "$LEG_OUT/calls.wheel.log" 2>&1; st calls-wheel $?
$PY - "$LEG_OUT/calls.source.json" "$LEG_OUT/calls.wheel.json" > "$LEG_OUT/calls.compare.txt" 2>&1 <<'PY'
import json, sys
a, b = (json.load(open(p)) for p in sys.argv[1:])
keys = sorted(k for k in a if k not in ("mojolearn", "vendor"))
same = [k for k in keys if a[k] == b.get(k)]
print("source", a["mojolearn"], "wheel", b["mojolearn"], "vendor", a["vendor"], b["vendor"])
print(f"EQUAL {len(same)} of {len(keys)}")
for k in keys:
    print(("EQUAL " if a[k] == b.get(k) else "DIFFER"), k, a[k], b.get(k))
sys.exit(0 if len(same) == len(keys) and a["mojolearn"] != b["mojolearn"] else 1)
PY
st calls-compare $?
cp /tmp/wheel_calls.py "$LEG_OUT/"
mkdir -p /tmp/wtest && cp python/mojolearn/tests/test_neural_inference.py /tmp/wtest/
( cd /tmp/wtest && env -u PYTHONPATH -u MOJOLEARN_HOST_DIR PYTHONPATH=/tmp/target \
    $TPY -m pytest -q -p no:cacheprovider --import-mode=append test_neural_inference.py \
    > "$LEG_OUT/pytest_wheel_neural.log" 2>&1 ); st pytest-wheel $?
tail -1 "$LEG_OUT/pytest_wheel_neural.log" >> "$LEG_OUT/box.txt"
