# Preprocessing

Preprocessing turns raw columns into the inputs a model expects: features on a common scale, categories
as numbers, missing values filled, and only the useful columns kept. Fit a transformer on the training
rows, then apply the same fitted transformer to new rows.

| Use | Estimators |
|---|---|
| Scaling | `StandardScaler`, `MinMaxScaler`, `MaxAbsScaler`, `RobustScaler`, `Normalizer` |
| Reshaping distributions | `PowerTransformer`, `QuantileTransformer` |
| Categorical features | `OneHotEncoder`, `OrdinalEncoder`, `TargetEncoder` |
| Labels | `LabelEncoder`, `LabelBinarizer`, `MultiLabelBinarizer` |
| Missing values | `SimpleImputer`, `IterativeImputer`, `KNNImputer` |
| Binning and new features | `KBinsDiscretizer`, `Binarizer`, `PolynomialFeatures`, `SplineTransformer` |
| Feature selection | `SelectKBest`, `VarianceThreshold`, `RFE` |
| Feature scores | `f_classif()`, `f_regression()`, `chi2()`, `mutual_info_classif()`, `r_regression()` |
| Count data | `TotalCountNormalizer`, `Log1pTransformer`, `ClippedStandardScaler`, `RegressOut` |

## Scaling

```python
import numpy as np
import mojolearn

rng = np.random.default_rng(0)
X = rng.random((10_000, 8), dtype=np.float32)
X_train, X_test = X[:8_000], X[8_000:]

scaler = mojolearn.StandardScaler().fit(X_train)
Z_train = scaler.transform(X_train)
Z_test = scaler.transform(X_test)
```

`StandardScaler` and `MinMaxScaler` also take data in batches with `partial_fit`, and save to a file with
`save` and `load`.

## Categories and missing values

`OneHotEncoder` encodes numeric category codes as one-hot columns and returns a dense float32 array.
`SimpleImputer` fills missing values with a per-column statistic.

```python
codes = np.array([[0.0, 1.0], [1.0, 1.0], [2.0, 0.0]], dtype=np.float32)
onehot = mojolearn.OneHotEncoder(drop="first").fit_transform(codes)

gaps = np.array([[1.0, np.nan], [3.0, 4.0], [np.nan, 8.0]], dtype=np.float32)
filled = mojolearn.SimpleImputer(strategy="median").fit_transform(gaps)
```

## Feature selection

```python
X = rng.random((1_000, 20), dtype=np.float32)
y = (X[:, 3] > 0.5).astype(np.int64)
selector = mojolearn.SelectKBest(mojolearn.f_classif, k=5).fit(X, y)
X_small = selector.transform(X)
print(selector.get_support())
```

## Count data

Counts such as words per document, reads per sample (bulk RNA, microbiome) or clicks per user differ mostly in
their totals and are heavily skewed. Four transformers prepare them, and each takes a dense array or a sparse
matrix in CSR form; the first three return a sparse result for a sparse input.

- `TotalCountNormalizer` scales each row to a common total (`target_sum`, or the median row total when it is
  None). `columns=` takes the total over a subset of the columns, for example leaving out a few very frequent
  words or genes; every column is still divided by that total. `exclude_highly_expressed=True` finds that subset
  for you: it leaves out every column holding more than `max_fraction` of some row's total.
- `Log1pTransformer` applies `log(1 + x)`, in base `base` when given.
- `ClippedStandardScaler` centers and scales each column by its sample standard deviation and clips the result at
  `max_value`; with `zero_center=False` it only scales, which keeps a sparse input sparse.
- `RegressOut` removes the part of every column explained by covariates you pass as `y`, such as each row's total.

```python
import numpy as np
import mojolearn

rng = np.random.default_rng(0)
counts = rng.poisson(rng.gamma(1.0, 2.0, 500), (2_000, 500)).astype(np.float32)

X = mojolearn.TotalCountNormalizer(target_sum=1e4, exclude_highly_expressed=True).fit_transform(counts)
X = mojolearn.Log1pTransformer().fit_transform(X)
X = mojolearn.ClippedStandardScaler(max_value=10).fit_transform(X)
```

To keep only the most variable columns of general data, use `VarianceThreshold` or `SelectKBest`; the
single-cell recipes are `HighlyVariableGenes` ([single-cell preprocessing](single_cell_preprocessing.md)).

## Devices and numeric modes

In `identical` mode, mojolearn returns the same bits on Apple, NVIDIA and AMD GPUs. `identical` is the
default. On Apple GPUs you can choose `fast` instead, which is tuned for speed and does not promise
identical bits: call `mojolearn.set_numeric_mode()` with `"fast"`. See [Numeric modes](numeric_modes.md).

## See also

- [Choosing an estimator](choosing_an_estimator.md)
- [Linear models](linear.md), which work best on scaled features
- API reference: [Preprocessing](../api/preprocessing.rst)
