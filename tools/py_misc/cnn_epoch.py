"""lane/py-misc: CNNClassifier.fit, the Python step loop (before) against
x_cnn_fit_epoch_r (after), in ONE process on one column.

PYMISC-DIGEST lines: sha256 of losses_, loss_curve_, every trained array,
the optimizer buffers and predict_proba, per config and arm; PYMISC-SAME
says whether the two arms' digests are equal. PYMISC-TIME lines: median
fit wall per arm on the timing shape.
usage: python cnn_epoch.py <repo root> <timing rows> [reps]"""
import hashlib
import statistics
import sys
import time

import numpy as np

sys.path.insert(0, sys.argv[1] + "/python")
import mojolearn as ml  # noqa: E402
from mojolearn import _expansion_cnn as xc  # noqa: E402

rows = int(sys.argv[2])
reps = int(sys.argv[3]) if len(sys.argv) > 3 else 3
b = ml.CNNClassifier(input_shape=(1, 4, 4))._binding()
print(f"PYMISC-COLUMN vendor={b.x_cnn_vendor()} has_epoch={hasattr(b, 'x_cnn_fit_epoch_r')}", flush=True)


def digest(m, x):
    h = hashlib.sha256()
    h.update(np.asarray(m.losses_, np.float64).tobytes())
    h.update(np.asarray(m.loss_curve_, np.float64).tobytes())
    for a in m.weights() + list(m._bufs):
        h.update(np.ascontiguousarray(a).tobytes())
    h.update(np.ascontiguousarray(m.predict_proba(x)).tobytes())
    return h.hexdigest()[:16]


def fit(arm, x, y, **kw):
    xc._EPOCH_ENTRY = arm == "epoch"
    return ml.CNNClassifier(**kw).fit(x, y)


rng = np.random.default_rng(0)
x = rng.standard_normal((300, 3 * 8 * 8)).astype(np.float32)
y = rng.integers(0, 4, 300)
base = dict(input_shape=(3, 8, 8), random_state=3)
configs = dict(
    sgd=dict(conv_channels=(4,), batch_size=32, max_iter=2, momentum=0.9, weight_decay=1e-4),
    sgd2blocks=dict(conv_channels=(4, 6), batch_size=64, max_iter=2),
    nesterov=dict(conv_channels=(4,), batch_size=50, max_iter=2, nesterov=True),
    dampening=dict(conv_channels=(4,), batch_size=33, max_iter=2, dampening=0.3),
    adam=dict(conv_channels=(4, 5), batch_size=40, max_iter=3, optimizer="adam", weight_decay=1e-3),
    adamw=dict(conv_channels=(4,), batch_size=64, max_iter=2, optimizer="adamw", betas=(0.8, 0.99)),
    bigbatch=dict(conv_channels=(4,), batch_size=512, max_iter=2),
    noshuffle=dict(conv_channels=(4,), batch_size=32, max_iter=1, shuffle=False),
    nopool=dict(conv_channels=(3,), batch_size=32, max_iter=1, pool_size=16),
    noconv=dict(conv_channels=(), batch_size=32, max_iter=2),
)
bad = 0
for name, kw in configs.items():
    d = {arm: digest(fit(arm, x, y, **base, **kw), x) for arm in ("python", "epoch")}
    same = d["python"] == d["epoch"]
    bad += not same
    print(f"PYMISC-DIGEST cnn {name} python {d['python']} epoch {d['epoch']} {'SAME' if same else 'DIFFERENT'}",
          flush=True)
print(f"PYMISC-SAME cnn {'ALL SAME' if not bad else f'{bad} DIFFERENT'}", flush=True)

# timing: MNIST-shaped rows, batch 32 (the audit's shape), one epoch
xt = rng.standard_normal((rows, 28 * 28)).astype(np.float32)
yt = rng.integers(0, 10, rows)
for arm in ("python", "epoch"):
    ts = []
    for _ in range(reps):
        s = time.perf_counter()
        m = fit(arm, xt, yt, input_shape=(1, 28, 28), conv_channels=(8,), batch_size=32, max_iter=1, random_state=0)
        ts.append(time.perf_counter() - s)
    print(f"PYMISC-TIME cnn fit rows={rows} batch=32 epochs=1 arm={arm} median_s={statistics.median(ts):.3f} "
          f"all={[round(t, 3) for t in ts]} digest={digest(m, xt[:256])}", flush=True)

raise SystemExit(1 if bad else 0)
