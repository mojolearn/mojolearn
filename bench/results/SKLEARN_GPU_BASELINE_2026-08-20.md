# Does scikit-learn's own GPU path reach the Apple GPU? Measured.

`bench_sklearn.py` has said this in its docstring since it was written:

> Our GPU against scikit-learn's CPU, on this Mac. That is the comparison the
> project exists to make: cuML and cuVS cannot run on Apple silicon at all, so
> there is no GPU arm on the other side to compare against.

**The second half of that sentence was never checked.** cuML and cuVS indeed
cannot run here. But scikit-learn 1.9 has Array API dispatch, and several
estimators declare `array_api_support`, which means a user can hand them a
torch tensor on the `mps` device and reach the same Apple GPU we do. If that
worked, our PCA and OLS ratios were measured against a baseline running with
one hand tied, and the headline was wrong in our favor.

So it was measured. `bench/bench_sklearn_gpu.py`, environment `skgpu`,
scikit-learn 1.9.0 + pytorch 2.13.0, `torch.backends.mps.is_available()` True.

## Most of the path does not exist on this hardware

Attempted every run; refusals are printed, never swallowed.

    pca_auto_gpu       REFUSED  aten::_linalg_eigh unimplemented for MPS
    pca_cov_eigh_gpu   REFUSED  same op
    ols_normal_eq_gpu  REFUSED  Array API dispatch supports only solver 'svd',
                                got 'cholesky'
    ols_default_gpu    REFUSED  can't convert mps:0 tensor to numpy

Read those carefully, because three of them protect the headline:

1. **scikit-learn's DEFAULT PCA cannot run on the Apple GPU.** `svd_solver="auto"`
   resolves to `covariance_eigh` at our shape, and that op is missing on MPS.
2. **`Ridge(alpha=0, solver="cholesky")` cannot either** -- that is the
   algorithm-matched OLS denominator, the honest one.
3. **`LinearRegression` cannot**, having no Array API support at all. This is
   what makes the default-arm OLS number a claim about the only thing a user
   on this machine can actually run.

`PCA(svd_solver="randomized")` also refuses: Array API has no LU, so sklearn
falls back to QR and the allocation asks for 149 GiB.

Only `PCA(full)` and `Ridge(svd)` reach the device, and those are what the
table below times.

## THE CONFOUND, AND THE ARM THAT REMOVED IT

The first run of this compared a **numpy** array against a **torch** tensor on
mps, saw 2.6x, and would have recorded it as the device. It is not. Those two
arms differ in the LIBRARY as well as the chip: numpy sends sklearn through
scipy/LAPACK, torch sends it through torch's own kernels, and for `svd` torch
has no MPS implementation and falls back to its CPU one. The gap could have
been torch's CPU LAPACK against scipy's, with the GPU contributing nothing.

That is the identical confound this repository already refuses for OLS, where
`LinearRegression` (gelsd) against our normal equations was an ALGORITHM
difference being reported as a hardware one. Same trap, same fix: hold
everything constant but the variable under test. Three arms, each adjacent
pair isolating one thing.

    arm                  median ms       [min, max]        n
    pca_full_cpu   numpy   3964.6   [3869.8, 4062.4]       5
    pca_full_torchcpu      1215.0   [1195.1, 1254.6]       5
    pca_full_gpu   mps     1484.6   [1450.6, 1511.6]       5

    ols_svd_cpu    numpy   4081.9   [3963.9, 4473.5]       5
    ols_svd_torchcpu       1178.7   [1142.5, 1272.2]       5
    ols_svd_gpu    mps     1430.3   [1404.5, 1500.1]       5

    step                        PCA          OLS      variable
    numpy -> torch CPU     3.26x faster  3.46x faster  LIBRARY
    torch CPU -> MPS       0.82x         0.82x         DEVICE

**scikit-learn's GPU path is 1.22x SLOWER than torch on the CPU**, on both
arms, and the torchcpu and mps ranges do not overlap on either, so this is a
finding and not noise. The whole apparent 2.6x was scipy losing to torch on
the CPU; the GPU then gave part of it back. Which is what the fallback warning
already said: `linalg_svd` runs on the CPU regardless, so the mps arm pays the
transfers and buys nothing.

## What this does to the headline: nothing, and that is the result

Every PCA arm available on this machine, same data, same shape:

    ours, GPU                              63.9 ms
    sklearn default (covariance_eigh) CPU 144.4 ms   <- their best
    sklearn full, torch CPU              1215   ms
    sklearn full, MPS                    1485   ms
    sklearn full, numpy CPU              3965   ms

**Enabling scikit-learn's GPU support makes their PCA about 10x worse**, because
the only solver that reaches the GPU is `full`, and `full` is far more
expensive than the `covariance_eigh` their default already chooses. We are
compared against their best arm at 144.4 ms. **2.26x stands.**

Same shape for OLS. Their matched arm (`Ridge` cholesky, 167 ms) and their
default (`LinearRegression`, 913 ms) both cannot reach the GPU; the only arm
that can is 1430 ms. **2.69x and 14.7x both stand.**

## Standing rule this creates

The refusal probes run on every invocation rather than being deleted now that
the answer is known. If a future torch implements `linalg_eigh` on MPS, this
file starts emitting an arm that did not exist before and the PCA comparison
has to be re-read. A harness that swallowed those exceptions would go on
printing the old headline after it stopped being true.

`skgpu` is a SEPARATE pixi environment on purpose. pytorch drags scipy and
numpy solves with it, and adding it to `bench` would re-solve the environment
every recorded timing in this repository was taken under. It pins
scikit-learn 1.9.0 / numpy 2.5.2 / scipy 1.18.0 to match `bench` exactly, so a
future solve that moves them fails loudly instead of quietly making the two
environments incomparable.

## Environment

```
recorded_at=2026-08-20T13:31:00Z
model=Mac16,12   chip=Apple M4   cores=10   memory_bytes=17179869184
macos=26.5.2     power=Now drawing from 'AC Power'
thermal=no warning level recorded
skgpu: python 3.14.7, torch 2.13.0, sklearn 1.9.0, numpy 2.5.2, scipy 1.18.0
```

Reproduce:

    pixi run -e skgpu python bench/bench_sklearn_gpu.py
