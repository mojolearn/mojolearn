# Does the line before `fit()` return the same bytes on every machine?

**Measured 2026-08-31. The answer, on this evidence, is yes, and the hole is
theoretical.** 24 of 24 digests agree between an Apple M4 (arm64, macOS) and
x86-64 Linux, including `StandardScaler.transform`, `MinMaxScaler.transform`
and `train_test_split`.

## Why the question was worth asking

This library's identity claim starts at `fit()`. Real pipelines do not. The
first thing that happens to anyone's data is a reduction over it, a mean or a
variance or a min and a max, and reductions are exactly where cross-vendor
identity dies, because the answer depends on the order the partial sums were
combined in. So the shape of every real use of mojolearn today is

    X = StandardScaler().fit_transform(X)                  # somebody else's code
    model = mojolearn.RandomForestClassifier().fit(X, y)   # bit-identical

and if the first line is not portable then the second line's guarantee is
about an input the user cannot reproduce. Nothing in this tree measured that.

## The instrument

`tools/preprocess_identity_probe.py`. The matrix is generated from splitmix64
rather than from numpy's RNG, so the INPUT is bit-identical on every platform
by construction, and its digest is the first row printed. If that row ever
differs, nothing below is about summation order.

    Apple    pixi run -e bench python3 tools/preprocess_identity_probe.py OUT.json
    x86-64   docker run --rm --cpus 2 --platform linux/amd64 \
               -v $PWD/tools/preprocess_identity_probe.py:/p.py:ro python:3.12-slim \
               sh -c "pip install -q numpy scikit-learn && python3 /p.py"

Records: `bench/results/preprocess/apple_m4.json`, `.../x86_64_docker.json`.

## The result

| | |
|---|---|
| digests compared | 24 |
| **differing** | **0** |
| numpy | 2.5.2 both sides |
| scikit-learn | 1.9.0 both sides |
| Apple | macOS 26.5.2, arm64 |
| x86-64 | Linux 6.10.14, glibc 2.41 |

Agreeing rows include `np.mean` and `np.var` in both precisions,
`StandardScaler.mean_ / var_ / scale_ / transform`, `MinMaxScaler.min_ /
scale_ / transform`, and both halves of `train_test_split(random_state=7)`.

## WHY IT MATCHES, WHICH IS THE PART THAT MAKES THE RESULT USABLE

A green result is only worth something if you know what it is a green result
ABOUT. Two mechanisms, and they are different:

**1. A scaler's per-column reduction has ONE order, whatever the SIMD width
is.** For `axis=0` over a C-contiguous `(rows, cols)` array numpy walks row by
row and vectorizes ACROSS COLUMNS, so each column accumulates serially. The
probe proves this rather than asserting it: `np.sum.f32` and `serial.sum.f32`
are the SAME digest. So the shape a scaler actually uses is the safe shape,
and it is safe for a structural reason rather than by luck.

**2. And the pairwise tree, where there is one, ALSO agreed.** That is the
stronger half. The probe deliberately includes the cases that put numpy's
pairwise blocking over the reduced axis: `axis=None`, `axis=1`, a
Fortran-ordered `axis=0`, and a single long contiguous column. On BOTH boxes
`np.sum.column.f32` differs from `serial.sum.column.f32`, which is the tree
being demonstrably active, and the two boxes still produced the same bits.
Without those rows, "everything matched" would only have been a restatement
of mechanism 1.

## WHAT THIS DOES NOT SHOW, stated so nobody reads it as more than it is

- **Same numpy build lineage on both sides.** numpy 2.5.2 and scikit-learn
  1.9.0 in both places. A user on a different numpy, or one built with a
  different SIMD baseline, is not covered by this and could still differ.
- **The x86-64 half ran under Docker on Apple silicon**, so numpy's runtime
  CPU dispatch saw an emulated processor and may have selected a more
  conservative kernel than a native Xeon or EPYC would. **A NATIVE x86 run is
  OWED**, and it is nearly free: the probe needs no GPU, so it can ride along
  on any rented leg.
- Nothing here is about one-hot encoding, quantile binning or imputation.
  Those drag in sorting, tie breaking and category ordering, which is a
  different and much larger animal, and they are not measured.

## The action this argues for

Do not build `mojolearn.preprocessing` on the strength of a worry. On this
evidence a scaler and a split are portable through numpy and scikit-learn
already, and shipping our own would add surface without closing a gap.

What IS owed is the sentence the tree does not have: the README should say
where the identity guarantee starts and what a user has to do to keep the
line before it reproducible, and it should cite this measurement rather than
an assumption. The one thing that would reopen the build question is the
native x86 run above disagreeing.
