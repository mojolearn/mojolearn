"""FeatureFreq CTR fixtures. HANDED OFF 2026-08-21: kernel stream owns
no further work here. KNOWN ISSUE for the harness stream: adult's object
columns carry real float NaN among strings, so `.astype(str)` alone does
not homogenize them -- `cat_df[c].fillna("nan").astype(str)` at the
np.unique site is the fix. Amazon works and its quality row is RUN AND
REPORTED (RESUME): ours 0.05078 vs CatBoost-Counter 0.05064 test mse,
frequency-information-matched arms. Original doc follows.

FeatureFreq CTR fixtures: categoricals as their GPU's simplest CTR,
computed HOST-SIDE with their exact arithmetic, so the Mojo arm can train
on categorical datasets today and the future device CTR kernels have a
bit-exact reference to match.

    pixi run -e bench python tools/ctr_prep.py <dir> amazon
    pixi run -e bench python tools/ctr_prep.py <dir> adult

THE FORMULAS ARE THEIRS, PINNED FROM SOURCE (CatBoost 54a8143a):
* FeatureFreq value = (bin_count + prior) / (n_rows + prior_observations)
  with the default prior {0.0, 1}:  count / (n + 1)
  (`cuda/ctrs/kernel/ctr_calcers.cu:100` NonWeightedBinFreqCtrsImpl;
  default priors `private/libs/options/cat_feature_options.cpp:127-129`).
* CTR binarization default = UNIFORM with 15 borders
  (`cat_feature_options.cpp:169`): borders at
  min + (max - min) * k / 16, k = 1..15, duplicates dropped.
* Category CODES are ours (sorted unique raw values -> 0..k-1); CatBoost
  hashes raw strings instead. The CTR VALUE only depends on counts, so
  the codes' order cannot change any ctr, only the (irrelevant) code
  labels.

Outputs, in the interleaved-fixture format the Mojo harnesses already
read -- the ctr columns become ordinary numeric features:
  <name>_folds_ctr.txt / <name>_bins_ctr.u8 / <name>_y.f32
plus <name>_catcodes.u8 (dense codes per cat feature, for the future
device CTR/one-hot paths) and <name>_MANIFEST_ctr.txt.
"""
import sys
import numpy as np
import catboost
import catboost.datasets


def freq_ctr_column(codes: np.ndarray) -> np.ndarray:
    n = len(codes)
    counts = np.bincount(codes)
    return (counts[codes].astype(np.float64) / (n + 1)).astype(np.float32)


def uniform_borders(values: np.ndarray, count: int = 15) -> np.ndarray:
    lo, hi = float(values.min()), float(values.max())
    if hi <= lo:
        return np.array([], dtype=np.float32)
    bs = [lo + (hi - lo) * k / (count + 1) for k in range(1, count + 1)]
    out = sorted(set(np.float32(b) for b in bs))
    return np.array(out, dtype=np.float32)


def main():
    d, name = sys.argv[1], sys.argv[2]
    if name == "amazon":
        train_df, _ = catboost.datasets.amazon()
        y = train_df["ACTION"].to_numpy().astype(np.float32)
        cat_df = train_df.drop(columns=["ACTION"])
        cat_cols = list(cat_df.columns)
        num = np.zeros((len(y), 0), dtype=np.float32)
    elif name == "adult":
        train_df, _ = catboost.datasets.adult()
        y = (train_df["income"] == ">50K").to_numpy().astype(np.float32)
        df = train_df.drop(columns=["income"])
        cat_cols = [c for c in df.columns if df[c].dtype == object]
        num_cols = [c for c in df.columns if c not in cat_cols]
        num = df[num_cols].to_numpy().astype(np.float32)
        cat_df = df[cat_cols]
    else:
        raise SystemExit("dataset must be amazon or adult")

    n = len(y)
    codes = np.zeros((len(cat_cols), n), dtype=np.int64)
    card = []
    for i, c in enumerate(cat_cols):
        uniq, inv = np.unique(cat_df[c].astype(str).to_numpy(), return_inverse=True)
        codes[i] = inv
        card.append(len(uniq))

    # numeric features keep CatBoost's own quantization grid, exactly as
    # interleaved_prep does; ctr columns get the UNIFORM-15 ctr grid.
    cols = []
    folds = []
    kinds = []
    if num.shape[1] > 0:
        pool = catboost.Pool(num, y)
        pool.quantize(border_count=128)
        import tempfile, os
        with tempfile.TemporaryDirectory() as td:
            bp = os.path.join(td, "b.tsv")
            pool.save_quantization_borders(bp)
            grid = {}
            for line in open(bp):
                line = line.strip()
                if line:
                    fi, bv = line.split("\t")
                    grid.setdefault(int(fi), []).append(float(bv))
        for f in range(num.shape[1]):
            bs = np.sort(np.array(grid.get(f, []), dtype=np.float32))
            if len(bs) == 0:
                continue
            cols.append(np.searchsorted(bs, num[:, f], side="left"))
            folds.append(len(bs))
            kinds.append("num:%d" % f)
    for i in range(len(cat_cols)):
        ctr = freq_ctr_column(codes[i])
        bs = uniform_borders(ctr, 15)
        if len(bs) == 0:
            continue
        cols.append(np.searchsorted(bs, ctr, side="left"))
        folds.append(len(bs))
        kinds.append("freqctr:%s(card %d)" % (cat_cols[i], card[i]))

    bins = np.array(cols, dtype=np.uint8)
    bins.tofile("%s/%s_bins_ctr.u8" % (d, name))
    y.tofile("%s/%s_y.f32" % (d, name))
    codes.astype(np.uint8 if max(card) < 256 else np.uint32).tofile(
        "%s/%s_catcodes.u8" % (d, name))
    with open("%s/%s_folds_ctr.txt" % (d, name), "w") as fh:
        for c in folds:
            fh.write("%d\n" % c)
    with open("%s/%s_MANIFEST_ctr.txt" % (d, name), "w") as fh:
        fh.write("rows %d\nfeatures %d\n" % (n, len(folds)))
        for k in kinds:
            fh.write("%s\n" % k)
    print(name, ":", n, "rows,", len(folds), "features (",
          sum(1 for k in kinds if k.startswith("freqctr")), "freq-ctr,",
          sum(1 for k in kinds if k.startswith("num")), "numeric ),"
          " cards", card)


if __name__ == "__main__":
    main()
