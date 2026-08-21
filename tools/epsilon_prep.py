"""Epsilon fixtures for bench/interleaved: the pre-registered scale row.

    pixi run -e bench python tools/epsilon_prep.py [dir]
    pixi run -e bench mojo run -I . \
        bench/interleaved/catboost_interleaved.mojo <dir> epsilon 400000

Epsilon: 400,000 rows x 2,000 DENSE numeric features (LIBSVM
`epsilon_normalized`). It is in the lever order (RESUME "then the epsilon
dataset") because it is the shape the scale finding predicts we win -- the
rows x features product is 10x the 800k x 100 parity shape, far above the
~436k-row crossover, and wide histograms are where replication measured
8.56x -- and because it appears in CatBoost's own benchmark suites, so
nobody chose it after seeing a number here. The run decides; this file
only builds the fixtures.

Source: CatBoost's OWN preprocessed mirror (`catboost/datasets.py`
`epsilon()`: `storage.mds.yandex.net/.../epsilon.tar.gz`, md5
`5bbfac403ac673da7d7ee84bd532e973`, train.tsv 400000 x 2001 with the label
in column 0). Measured 2026-08-21: their mirror moves at ~14 MB/s where
the NTU libsvm original moves at ~0.5, so the 3.86GB tarball beats the
1.7GB bz2 by half an hour. The tarball is md5-checked against their
constant, train.tsv extracted, and the tarball deleted; the parsed X/y are
cached as .npy beside it because the parse is minutes and the reload is
seconds.

Parsing is chunked (pandas C engine, 20k rows at a time, float32
straight from text): peak extra memory is one ~160MB chunk over the 3.2GB
X itself. A whole-file read at pandas' float64 default would peak near
10GB, which a shared 16GB box does not have to give.

Labels ride as float32 exactly as the TSV carries them: every pinned
comparison in this repository trains RMSE on both arms, epsilon rides the
same settings, and both arms read the SAME y file, so the encoding cannot
differ between arms.

Output is the standard interleaved fixture set, exactly interleaved_prep's
discipline: CatBoost's OWN borders at 128 and 254 (pool freed between the
two -- two pools at once is another 3GB), pre-binned uint8 col-major for
the Mojo arm, X.npy/y.npy for theirs. The default output dir is the cache
dir itself, so fixtures persist across sessions and the timed run can wait
for a quiet window without re-prepping.
"""
import hashlib
import os
import subprocess
import sys
import urllib.request

import numpy as np

CACHE = os.path.expanduser("~/.cache/mojolearn")
URLS = (
    "https://storage.mds.yandex.net/get-devtools-opensource/250854/"
    "epsilon.tar.gz",
    "https://proxy.sandbox.yandex-team.ru/785711439",
)
MD5 = "5bbfac403ac673da7d7ee84bd532e973"  # catboost/datasets.py epsilon()
ROWS = 400_000
FEATS = 2_000


def _train_tsv():
    os.makedirs(CACHE, exist_ok=True)
    tsv = os.path.join(CACHE, "epsilon_train.tsv")
    if os.path.exists(tsv):
        return tsv
    tar = os.path.join(CACHE, "epsilon.tar.gz")
    if not os.path.exists(tar):
        part = tar + ".part"
        err = None
        for url in URLS:
            try:
                print("downloading", url)
                urllib.request.urlretrieve(url, part)
                os.rename(part, tar)
                break
            except Exception as e:  # try the next mirror
                err = e
        else:
            raise RuntimeError("all epsilon mirrors failed: %r" % err)
    h = hashlib.md5()
    with open(tar, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 24), b""):
            h.update(block)
    if h.hexdigest() != MD5:
        raise RuntimeError(
            "epsilon.tar.gz md5 %s != catboost's %s" % (h.hexdigest(), MD5)
        )
    print("md5 verified; extracting train.tsv")
    subprocess.run(
        ["tar", "-xzf", tar, "-C", CACHE, "train.tsv"], check=True
    )
    os.replace(os.path.join(CACHE, "train.tsv"), tsv)
    os.remove(tar)  # 3.9GB; the md5-verified extraction is what we keep
    return tsv


def _parse():
    xp = os.path.join(CACHE, "epsilon_X.npy")
    yp = os.path.join(CACHE, "epsilon_y.npy")
    if os.path.exists(xp) and os.path.exists(yp):
        return np.load(xp), np.load(yp)
    import pandas as pd
    tsv = _train_tsv()
    x = np.empty((ROWS, FEATS), dtype=np.float32)
    y = np.empty(ROWS, dtype=np.float32)
    row = 0
    for chunk in pd.read_csv(
        tsv, sep="\t", header=None, dtype=np.float32, chunksize=20_000
    ):
        a = chunk.to_numpy()
        n = a.shape[0]
        assert a.shape[1] == FEATS + 1, a.shape
        y[row:row + n] = a[:, 0]
        x[row:row + n] = a[:, 1:]
        row += n
        if row % 100_000 == 0:
            print("  parsed", row, "rows")
    assert row == ROWS, "expected %d rows, parsed %d" % (ROWS, row)
    np.save(xp, x)
    np.save(yp, y)
    return x, y


def main():
    import catboost
    d = sys.argv[1] if len(sys.argv) > 1 else CACHE
    x, y = _parse()
    print("labels: values", sorted(set(np.unique(y[:10000]).tolist())),
          "-- both arms read this same y, so the encoding cannot differ")
    if d != CACHE:
        np.save("%s/epsilon_X.npy" % d, x)
        np.save("%s/epsilon_y.npy" % d, y)
    y.tofile("%s/epsilon_y.f32" % d)

    for B in (128, 254):
        pool = catboost.Pool(x, y)
        pool.quantize(border_count=B)
        pool.save_quantization_borders("%s/epsilon_borders_%d.tsv" % (d, B))
        del pool  # two 3GB pools at once would swap the box
        grid = {}
        for line in open("%s/epsilon_borders_%d.tsv" % (d, B)):
            line = line.strip()
            if line:
                fi, bv = line.split("\t")
                grid.setdefault(int(fi), []).append(float(bv))
        kept = sorted(f for f in grid if grid[f])
        folds = [len(grid[f]) for f in kept]
        assert max(folds) <= 254
        bins = np.empty((len(kept), ROWS), dtype=np.uint8)
        for i, f in enumerate(kept):
            borders = np.sort(np.array(grid[f], dtype=np.float32))
            bins[i] = np.searchsorted(
                borders, x[:, f], side="left"
            ).astype(np.uint8)
        bins.tofile("%s/epsilon_bins_%d.u8" % (d, B))
        del bins
        with open("%s/epsilon_folds_%d.txt" % (d, B), "w") as fh:
            for c in folds:
                fh.write("%d\n" % c)
        print("epsilon B", B, "rows", ROWS, "kept", len(kept),
              "max_folds", max(folds))


if __name__ == "__main__":
    main()
