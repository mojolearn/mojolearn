# Continued multi-GPU implementation: two H100s

All builds and model checks ran on RunPod pod `kpqg64gsa9lib0`, with two H100s
and the manifest-verified R2 enwik8 corpus. No local build or test was run.
This is same-hardware serial/distributed equality evidence. It does not extend
the earlier frozen-source H100/4090 qualification to these new implementations.

| Change | Final evidence under `rollout-evidence/` |
| --- | --- |
| Concurrent byte-LM gradient waves | `parallel-out/status.tsv`, byte-LM replay and existing-session gates |
| Existing MLP/Samba/forests/KMeans | Rebuilt cloud leg, all final statuses zero |
| Greedy boosting feature histograms | `boost-out/boosting.json`: 16 fixtures |
| ARIMA independent-series fits | `arima-out/arima.json`: 12 fixtures |
| StandardScaler/MinMaxScaler columns | `scaler-out/preprocessing.json`: 16 fit/transform/inverse fixtures |
| OLS/Ridge/covariance PCA/TruncatedSVD | `gram-out/gram.json`: 20 fitted-state/output fixtures |
| Pinned Gram partials | `gram-out/native-bitwise.log`: 40 fixtures; every partial and final cell |
| Holt-Winters independent-series fits | `holt-out/holtwinters.json`: 6 additive/multiplicative fit/forecast fixtures |
| LogisticRegression gradient columns | `logistic-out/logistic.json`: 16 binary/multiclass fixtures |
| Lasso/ElasticNet dot leaves | `solver-out/solver.json`: 12 fitted-state/output fixtures |
| Dot reference and original plans | `solver-out/native-bitwise.log`: 7 row counts, 1/2 devices, automatic and all explicit plans |

The Python fixtures also check that a rejected fit does not publish partial
state. Scalers distribute transforms; time-series forecasts use the original
predictor after merging fitted state. The neural replay gates cover their
existing transactional/checkpoint behavior. These are small correctness runs,
not throughput, eight-GPU, lost-device or beyond-single-GPU memory tests.

Builds were incremental. Each stage retains its binary hash and build/gate
logs. The ship helper recorded the literal `HEAD` in `base-commit.txt`, not
an immutable commit ID. The local branch was at
`19689c801b733efdac9c595e78693e2df950838d`; this is provenance context, not
a separately verified equality assertion for every compiled input. `source/` retains the
available staged overlays, the final changed-file snapshot, and a final source
hash manifest. The final snapshot is not a claim that every earlier extension
was rebuilt after every later edit. The initial cloud leg's source manifest
is retained separately in `parallel-out/source.sha256`.

Development failures are retained. The initial concurrent-byte closure and
GBDT pointer/alias builds failed and were corrected on the cloud host. The
first boosting fixture selected an unsupported random-strength/scoring
combination; its configuration was corrected. The first ARIMA wrapper imported
`memcopy` from the wrong module; that import was corrected.

`gram-out/native.log` records a failure in the existing broad Gram check:
its `check_gram_dispatch` expects a 768-feature IDENTICAL product to refuse,
but the existing general GEMM dispatch admits that shape. That check was not
changed or reported as passing. The dedicated distributed gate instead checks
every partial and output in this implementation's admitted 1..128-feature
path against the unchanged one-GPU path.

Memory limitations remain explicit in [the coverage inventory](../../../../../docs/multi_gpu/COVERAGE.md).
Neural state is replicated. Boosting, Gram, logistic and coordinate descent
retain complete root state. Series and scaler workers receive only their
partitions, but no run exceeding one GPU's memory has been qualified.
