#!/bin/sh
# Lane forest-finish, batch E: the REACH proof, queued behind the verdicts.
#
# Batch B's probe reached for `mojolearn._mojolearn_rf`, and this package has
# no such module: `_bind` is `_backend.binding(name, mode)` and that is the
# object an estimator actually calls into. This probe wraps THAT object's fit
# entries and counts them, so a hash that did not move cannot be mistaken for
# a path that was never taken. It runs after batch D so it cannot perturb a
# timed cell.
cd /root/mojolearn || exit 9
AB="sh tools/trees_identical_ab.sh"
OUT=/root/trees_out
PATH="$HOME/.pixi/bin:$PATH"; export PATH
export MOJOLEARN_NUMERIC_MODE=identical
while [ ! -f $OUT/phase.D1_VERDICTS_DONE ]; do sleep 15; done
echo "batchE start $(date -u +%T)"

for set in rowmajor baseline; do
  $AB use $set
  PYTHONPATH=/root/mojolearn/python timeout -k 30 900 python3 - > $OUT/logs/reach2.$set.log 2>&1 <<'PY'
import hashlib
import numpy as np
import mojolearn
from mojolearn import _backend

counts = {}


def wrap(mod, name):
    fn = getattr(mod, name, None)
    if not callable(fn):
        return
    def w(*a, **k):
        counts[name] = counts.get(name, 0) + 1
        return fn(*a, **k)
    try:
        setattr(mod, name, w)
    except Exception as exc:                      # a module that refuses it
        print("WRAP-FAILED", name, type(exc).__name__, exc)


mode = _backend.default_mode()
print("MODE", mode, "package numeric_mode", mojolearn.numeric_mode())
for b in ("_mojolearn_rf", "_mojolearn_trees"):
    m = _backend.binding(b, mode)
    print("BINDING", b, getattr(m, "__name__", "?"), getattr(m, "__file__", "?"))
    entries = sorted(n for n in dir(m) if "_fit" in n)
    print("ENTRIES", b, entries)
    for name in entries:
        wrap(m, name)

rng = np.random.default_rng(3)
Xc = np.ascontiguousarray(rng.normal(size=(50000, 32)).astype(np.float32))
yc = (Xc[:, 0] > 0).astype(np.int64)
yr = np.ascontiguousarray(Xc[:, 1].copy())
cases = (
    ("rf", lambda: mojolearn.RandomForestClassifier(n_estimators=4, max_depth=6,
                                                    random_state=1, device="gpu"), yc),
    ("et", lambda: mojolearn.ExtraTreesClassifier(n_estimators=4, max_depth=6,
                                                  random_state=1, device="gpu"), yc),
    ("rfreg", lambda: mojolearn.RandomForestRegressor(n_estimators=4, max_depth=6,
                                                      random_state=1, device="gpu"), yr),
    ("etreg", lambda: mojolearn.ExtraTreesRegressor(n_estimators=4, max_depth=6,
                                                    random_state=1, device="gpu"), yr),
)
for layout in ("C", "F"):
    X = Xc if layout == "C" else np.asfortranarray(Xc)
    for label, make, y in cases:
        counts.clear()
        try:
            model = make().fit(X, y)
            pred = np.asarray(model.predict(Xc[:2000]))
            h = hashlib.sha256(pred.tobytes()).hexdigest()[:16]
            print("REACH", layout, label, sorted(counts.items()), "predict_hash", h)
        except Exception as exc:
            print("REACH", layout, label, "FAILED", type(exc).__name__, str(exc)[:300])
PY
  echo "reach2_probe $set=$? $(date -u +%T)" | tee -a $OUT/ab.txt
  grep -E "^(REACH|BINDING|MODE|ENTRIES|WRAP-FAILED)" $OUT/logs/reach2.$set.log
done
echo "E1_REACH_DONE $(date -u +%T)" | tee -a $OUT/ab.txt
: > $OUT/phase.E1_REACH_DONE
echo "batchE end $(date -u +%T)"
