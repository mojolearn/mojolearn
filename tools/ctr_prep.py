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
* CTR binarization default for FeatureFreq on GPU = **MinEntropy with 15
  borders** for SIMPLE ctrs (`Median` for tree ctrs), from
  `CreateDefaultCounter` (`catboost_options.cpp:392-415`) and re-applied by
  `SetDefaultBinarizationsIfNeeded` (`:418-427`).
  THIS FILE USED TO SAY `Uniform, 15`, citing `cat_feature_options.cpp:169`.
  That line is real but it is the GENERIC `TCtrDescription` constructor
  default, and the GPU path never reaches it for FeatureFreq. A correct
  citation on the wrong code path reads as verified and is how it survived.
  Borders now come from CatBoost's own quantizer rather than from a formula
  reimplemented here, the same discipline `interleaved_prep.py` already uses
  for the numeric grid, so there is no second implementation to drift.
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


def catboost_borders(matrix, y, border_count, border_type=None):
    """CatBoost's OWN borders for each column of `matrix`, as a dict.

    Used for both grids here: the numeric columns take their default
    (GreedyLogSum) at 128, the ctr columns take MinEntropy at 15. Asking
    their quantizer beats reimplementing either formula, and it is what
    keeps our arm and theirs splitting on the same grid.
    """
    import os
    import tempfile
    pool = catboost.Pool(matrix, y)
    if border_type is None:
        pool.quantize(border_count=border_count)
    else:
        pool.quantize(border_count=border_count,
                      feature_border_type=border_type)
    grid = {}
    with tempfile.TemporaryDirectory() as td:
        bp = os.path.join(td, "b.tsv")
        pool.save_quantization_borders(bp)
        for line in open(bp):
            line = line.strip()
            if line:
                fi, bv = line.split("\t")
                grid.setdefault(int(fi), []).append(float(bv))
    return {f: np.sort(np.array(v, dtype=np.float32)) for f, v in grid.items()}


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

    # BOTH grids are CatBoost's own, exactly as interleaved_prep does for
    # numeric: 128 GreedyLogSum for numeric columns, 15 MinEntropy for ctr
    # columns, which is their GPU FeatureFreq simple-ctr binarization.
    cols = []
    folds = []
    kinds = []
    if num.shape[1] > 0:
        grid = catboost_borders(num, y, 128)
        for f in range(num.shape[1]):
            bs = grid.get(f, np.array([], dtype=np.float32))
            if len(bs) == 0:
                continue
            cols.append(np.searchsorted(bs, num[:, f], side="left"))
            folds.append(len(bs))
            kinds.append("num:%d" % f)
    ctrs = np.column_stack([freq_ctr_column(codes[i])
                            for i in range(len(cat_cols))])
    ctr_grid = catboost_borders(ctrs, y, 15, "MinEntropy")
    for i in range(len(cat_cols)):
        bs = ctr_grid.get(i, np.array([], dtype=np.float32))
        if len(bs) == 0:
            continue
        cols.append(np.searchsorted(bs, ctrs[:, i], side="left"))
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
