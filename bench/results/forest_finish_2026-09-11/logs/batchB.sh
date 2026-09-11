#!/bin/sh
# Lane forest-finish, batch B, queued behind batch A on the same pod.
# 1. Reach: count calls into the *_rowmajor native entries during a C-order
#    float32 fit (must be 1 each) and an F-order fit (must be 0), so a same
#    hash cannot mean the new path was never taken.
# 2. Where the Istella-S forest time goes: the Python-side host split
#    (tools/forest_host_split.py) on both sets, and the ET stage replicate.
cd /root/mojolearn || exit 9
AB="sh tools/trees_identical_ab.sh"
OUT=/root/trees_out
PATH="$HOME/.pixi/bin:$PATH"; export PATH
export MOJOLEARN_NUMERIC_MODE=identical
mark() { echo "$1 $(date -u +%T)" | tee -a $OUT/ab.txt; : > "$OUT/phase.$1"; }
while [ ! -f $OUT/phase.A4_STAGE_DONE ]; do sleep 15; done
echo "batchB start $(date -u +%T)"
mkdir -p $OUT/split

$AB use rowmajor
PYTHONPATH=/root/mojolearn/python timeout -k 30 600 python3 - > $OUT/logs/reach.rowmajor.log 2>&1 <<'PY'
import numpy as np, mojolearn
from mojolearn import _mojolearn_rf, _mojolearn_trees
from mojolearn import _backend
counts = {}
def wrap(mod, name):
    fn = getattr(mod, name)
    def w(*a, **k):
        counts[name] = counts.get(name, 0) + 1
        return fn(*a, **k)
    setattr(mod, name, w)
rf = _backend.binding("_mojolearn_rf") if hasattr(_backend, "binding") else _mojolearn_rf
tr = _backend.binding("_mojolearn_trees") if hasattr(_backend, "binding") else _mojolearn_trees
for mod in (rf, tr):
    for name in dir(mod):
        if "_fit" in name:
            try:
                wrap(mod, name)
            except Exception as exc:
                print("WRAP-FAILED", name, exc)
rng = np.random.default_rng(3)
X = rng.normal(size=(50000, 32)).astype(np.float32)
y = (X[:, 0] > 0).astype(np.int64)
for layout in ("C", "F"):
    Xl = X if layout == "C" else np.asfortranarray(X)
    for label, make in (("rf", lambda: mojolearn.RandomForestClassifier(n_estimators=4, random_state=1, device="gpu")),
                        ("et", lambda: mojolearn.ExtraTreesClassifier(n_estimators=4, random_state=1, device="gpu")),
                        ("rfr", lambda: mojolearn.RandomForestRegressor(n_estimators=4, random_state=1, device="gpu")),
                        ("etr", lambda: mojolearn.ExtraTreesRegressor(n_estimators=4, random_state=1, device="gpu"))):
        counts.clear()
        m = make()
        m.fit(Xl, y if label in ("rf", "et") else X[:, 1].copy())
        p = m.predict(X[:2000])
        import hashlib
        print("REACH", layout, label, sorted(counts.items()), hashlib.sha256(np.asarray(p).tobytes()).hexdigest()[:16])
PY
echo "reach_probe=$? $(date -u +%T)" | tee -a $OUT/ab.txt
grep REACH $OUT/logs/reach.rowmajor.log
mark B1_REACH_DONE

for set in rowmajor baseline; do
  $AB use $set
  for lane in et rf; do
    PYTHONPATH=/root/mojolearn/python timeout -k 30 1200 python3 -u tools/forest_host_split.py \
      --lane $lane --dataset istella --rows 1000000 --reps 3 > $OUT/split/$set.$lane.istella.log 2>&1
    echo "split_exit $set.$lane.istella=$? $(date -u +%T)" | tee -a $OUT/ab.txt
  done
done
mark B2_SPLIT_DONE
echo "batchB end $(date -u +%T)"
