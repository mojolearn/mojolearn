"""Prepare a dataset for bench/interleaved: borders (CatBoost's own
quantizer), pre-binned uint8 col-major for the Mojo arm, flat y.

    pixi run -e bench python tools/interleaved_prep.py <dir> covtype
    pixi run -e bench python tools/interleaved_prep.py <dir> synth [rows feats]

Both arms then train on the SAME rows with the SAME grid; the pool is
quantized outside the timed region on both sides.
np.searchsorted(borders, x, side='left') matches
gbdt/grid_creator/binarization.binarize exactly (borders strictly below).
"""
import sys
import numpy as np
import catboost

d = sys.argv[1]
name = sys.argv[2]

if name == "covtype":
    from sklearn.datasets import fetch_covtype
    ds = fetch_covtype()
    x = ds.data.astype(np.float32)
    y = ds.target.astype(np.float32)
elif name == "synth":
    rows = int(sys.argv[3]) if len(sys.argv) > 3 else 800_000
    feats = int(sys.argv[4]) if len(sys.argv) > 4 else 100
    rng = np.random.default_rng(7)
    x = rng.normal(size=(rows, feats)).astype(np.float32)
    y = (3.0 * x[:, 0] - 2.0 * x[:, 3] + 1.5 * x[:, 7] * x[:, 0]
         + 0.1 * rng.normal(size=rows)).astype(np.float32)
else:
    raise SystemExit("dataset must be covtype or synth")

np.save("%s/%s_X.npy" % (d, name), x)
np.save("%s/%s_y.npy" % (d, name), y)
y.tofile("%s/%s_y.f32" % (d, name))
rows = x.shape[0]

for B in (128, 254):
    pool = catboost.Pool(x, y)
    pool.quantize(border_count=B)
    pool.save_quantization_borders("%s/%s_borders_%d.tsv" % (d, name, B))
    grid = {}
    for line in open("%s/%s_borders_%d.tsv" % (d, name, B)):
        line = line.strip()
        if line:
            fi, bv = line.split("\t")
            grid.setdefault(int(fi), []).append(float(bv))
    kept = sorted(f for f in grid if grid[f])
    folds = [len(grid[f]) for f in kept]
    assert max(folds) <= 254
    bins = np.empty((len(kept), rows), dtype=np.uint8)
    for i, f in enumerate(kept):
        borders = np.sort(np.array(grid[f], dtype=np.float32))
        bins[i] = np.searchsorted(borders, x[:, f], side="left").astype(np.uint8)
    bins.tofile("%s/%s_bins_%d.u8" % (d, name, B))
    with open("%s/%s_folds_%d.txt" % (d, name, B), "w") as fh:
        for c in folds:
            fh.write("%d\n" % c)
    print(name, "B", B, "rows", rows, "kept", len(kept), "max_folds", max(folds))
