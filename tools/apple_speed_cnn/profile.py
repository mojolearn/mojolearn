"""lane/cnn-apple (2026-09-28): x_cnn on this device, one process.

1. XCNN-SPEED lines: the phase-4 bench (Conv2d N256 fwd/bwd at three shapes,
   CNNClassifier fit + predict_proba at 2048 and 8192 rows), median.
2. XCNN-ENTRY lines: every binding call of one CNNClassifier fit (2048 rows)
   wrapped in a wall clock; each x_cnn entry synchronizes before it returns,
   so an entry's wall is its GPU time plus its one wait. Sorted by total.
3. XCNN-FLOOR: the wall of a 1-element resident entry (one launch, one wait).
4. XCNN-DIGEST lines: sha256 of the fitted weights, the losses and the
   probabilities (fixed seeds), for the before/after bit check on one column.
usage: python profile.py <mode> <repo root>"""
import hashlib
import statistics
import sys
import time
from collections import defaultdict

import numpy as np

sys.path.insert(0, sys.argv[2] + "/python")
import mojolearn as ml  # noqa: E402
from mojolearn import _expansion_cnn as xc  # noqa: E402

mode = sys.argv[1]
only = set(a for a in sys.argv[3:])
if "legacy" in only:  # lane/cnn-apple2: the before arm's Python side
    xc._LEGACY_STEP = True
    only.discard("legacy")
print(f"XCNN-ARM legacy_step={xc._LEGACY_STEP} predict_rows={xc._PREDICT_ROWS}", flush=True)


def t(f, reps):
    f()
    xs = []
    for _ in range(reps):
        s = time.perf_counter()
        f()
        xs.append(time.perf_counter() - s)
    return statistics.median(xs) * 1e3


def digest(*arrs):
    h = hashlib.sha256()
    for a in arrs:
        h.update(np.ascontiguousarray(a).tobytes())
    return h.hexdigest()[:16]


rng = np.random.default_rng(0)
if not only or "speed" in only:
    for (N, C, OC, H) in [(256, 3, 64, 32), (256, 64, 64, 32), (256, 64, 128, 16)]:
        x = rng.standard_normal((N, C, H, H)).astype(np.float32)
        conv = ml.Conv2d(C, OC, 3, padding=1, numeric_mode=mode)
        y = conv.forward(x)
        g = rng.standard_normal(y.shape).astype(np.float32)
        tf = t(lambda: conv.forward(x), 7)
        tb = t(lambda: conv.backward(g), 7)
        print(f"XCNN-SPEED {mode} conv N{N} C{C} OC{OC} H{H}: fwd {tf:.1f} ms  bwd {tb:.1f} ms", flush=True)
        print(f"XCNN-DIGEST conv N{N} C{C} OC{OC} H{H} y {digest(conv.forward(x))} "
              f"bwd {digest(*[a for a in (conv.backward(g),) if a is not None], conv.grad_weight_, conv.grad_bias_)}",
              flush=True)
    for rows, reps in [(2048, 5), (8192, 3)]:
        X = rng.standard_normal((rows, 3 * 32 * 32)).astype(np.float32)
        yy = rng.integers(0, 10, rows)
        clf = ml.CNNClassifier((3, 32, 32), conv_channels=(32, 64), batch_size=256, max_iter=1, numeric_mode=mode)
        tf = t(lambda: clf.fit(X, yy), reps)
        print(f"XCNN-SPEED {mode} CNNClassifier fit {rows}x3x32x32 (32,64) batch 256, 1 epoch: {tf:.1f} ms",
              flush=True)
        tp = t(lambda: clf.predict_proba(X), reps)
        print(f"XCNN-SPEED {mode} CNNClassifier predict_proba {rows}: {tp:.1f} ms", flush=True)
        print(f"XCNN-DIGEST fit {rows} weights {digest(*clf.weights())} losses "
              f"{digest(np.asarray(clf.losses_, np.float64))} proba {digest(clf.predict_proba(X))}", flush=True)

if "family" in only:
    # lane/cnn-apple2: the rest of the family at sizes past the launch floor
    x = rng.standard_normal((64, 64, 32, 32)).astype(np.float32)
    blk = ml.BasicBlock(64, 64, numeric_mode=mode)
    y = blk.forward(x)
    g = rng.standard_normal(y.shape).astype(np.float32)
    tf = t(lambda: blk.forward(x), 5)
    tb = t(lambda: (blk.forward(x), blk.backward(g)), 5)
    print(f"XCNN-SPEED {mode} BasicBlock 64 N64 H32: fwd {tf:.1f} ms  fwd+bwd {tb:.1f} ms", flush=True)
    blk.forward(x)
    print(f"XCNN-DIGEST BasicBlock y {digest(blk.forward(x))} dx {digest(blk.backward(g))}", flush=True)
    bn = ml.BatchNorm2d(64, numeric_mode=mode)
    tf = t(lambda: bn.forward(x), 5)
    tb = t(lambda: bn.backward(x), 5)
    print(f"XCNN-SPEED {mode} BatchNorm2d 64 N64 H32: fwd {tf:.1f} ms  bwd {tb:.1f} ms", flush=True)
    bn2 = ml.BatchNorm2d(64, numeric_mode=mode)
    yb = bn2.forward(x)
    print(f"XCNN-DIGEST BatchNorm2d y {digest(yb)} dx {digest(bn2.backward(x))} "
          f"dgamma {digest(bn2.grad_weight_)} dbeta {digest(bn2.grad_bias_)}", flush=True)
    xp = rng.standard_normal((256, 64, 32, 32)).astype(np.float32)
    mp = ml.MaxPool2d(2, numeric_mode=mode)
    yp = mp.forward(xp)
    gp = rng.standard_normal(yp.shape).astype(np.float32)
    tf = t(lambda: mp.forward(xp), 5)
    tb = t(lambda: mp.backward(gp), 5)
    print(f"XCNN-SPEED {mode} MaxPool2d 2 N256 C64 H32: fwd {tf:.1f} ms  bwd {tb:.1f} ms", flush=True)
    nn_, f_ = 100000, 64
    xg = rng.standard_normal((nn_, f_)).astype(np.float32)
    ei = rng.integers(0, nn_, (2, 10 * nn_))
    for name, layer in (("GCNConv", ml.GCNConv(f_, f_, numeric_mode=mode)),
                        ("SAGEConv", ml.SAGEConv(f_, f_, numeric_mode=mode))):
        yg = layer.forward(xg, ei)
        gg = rng.standard_normal(yg.shape).astype(np.float32)
        tf = t(lambda: layer.forward(xg, ei), 3)
        tb = t(lambda: layer.backward(gg), 3)
        print(f"XCNN-SPEED {mode} {name} n{nn_} e{10 * nn_} f{f_}: fwd {tf:.1f} ms  bwd {tb:.1f} ms", flush=True)
        print(f"XCNN-DIGEST {name} y {digest(layer.forward(xg, ei))} dx {digest(layer.backward(gg))}", flush=True)

if not only or "entries" in only:
    # per-entry walls in one fit
    stats = defaultdict(lambda: [0, 0.0])

    class Timed:
        def __init__(self, b):
            self._b = b

        def __getattr__(self, name):
            f = getattr(self._b, name)
            if not callable(f):
                return f

            def w(*a, **k):
                s = time.perf_counter()
                r = f(*a, **k)
                e = stats[name]
                e[0] += 1
                e[1] += time.perf_counter() - s
                return r
            return w

    orig = xc._Layer._binding
    xc._Layer._binding = lambda self: Timed(orig(self))
    X = rng.standard_normal((2048, 3 * 32 * 32)).astype(np.float32)
    yy = rng.integers(0, 10, 2048)
    clf = ml.CNNClassifier((3, 32, 32), conv_channels=(32, 64), batch_size=256, max_iter=1, numeric_mode=mode)
    clf.fit(X, yy)
    stats.clear()
    s = time.perf_counter()
    clf.fit(X, yy)
    wall = (time.perf_counter() - s) * 1e3
    tot = sum(v[1] for v in stats.values()) * 1e3
    print(f"XCNN-ENTRY fit 2048 wall {wall:.1f} ms, in entries {tot:.1f} ms, calls {sum(v[0] for v in stats.values())}")
    for name, (c, sec) in sorted(stats.items(), key=lambda kv: -kv[1][1]):
        print(f"XCNN-ENTRY {name:40s} calls {c:4d} total {sec * 1e3:8.1f} ms  per {sec * 1e3 / c:7.3f} ms")
    stats.clear()
    s = time.perf_counter()
    clf.predict_proba(X)
    wall = (time.perf_counter() - s) * 1e3
    print(f"XCNN-ENTRY predict 2048 wall {wall:.1f} ms")
    for name, (c, sec) in sorted(stats.items(), key=lambda kv: -kv[1][1]):
        print(f"XCNN-ENTRY-P {name:40s} calls {c:4d} total {sec * 1e3:8.1f} ms  per {sec * 1e3 / c:7.3f} ms")
    xc._Layer._binding = orig
    # the floor: one launch + one wait on a 1-element resident array
    b = orig(clf)
    h = [b.x_cnn_res_alloc(1) for _ in range(3)]
    hyper = [0.01, 0.9, 0.0, 0.0, 0.0, 0.0]
    fl = t(lambda: b.x_cnn_sgd_r(h[0], h[1], h[2], [1], hyper), 50)
    one = np.empty(1, np.float32)
    dl = t(lambda: b.x_cnn_res_download(h[0], one.ctypes.data, 1), 50)
    for x_ in h:
        b.x_cnn_res_free(x_)
    print(f"XCNN-FLOOR sgd_r n=1 {fl:.3f} ms  res_download 1 float {dl:.3f} ms")
