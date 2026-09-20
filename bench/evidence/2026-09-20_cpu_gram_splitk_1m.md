# CPU split-K Gram preparation, 1M rows

Base: `b26e6b12ac740d7cbd06a4669f14dffbf1745c46`. Hardware: Apple M4,
10 logical CPUs, arm64, macOS 26.5.2 (25F84). Dataset: NumPy PCG64 seed
`20260920`, 1,000,000 rows x 16 float32 columns and one float32 target.
Combined input SHA-256:
`51eeba71a9b20c4b845be2b1cec18dead500da2b121dbfb39459554b6bb543ac`.

Command (run alternately against clean-base and candidate libraries):

```sh
.pixi/envs/default/bin/python bench/bench_gram_splitk.py LIBRARY --repeats 1
```

Five independent process pairs were run baseline then candidate. Median wall
seconds (the timer excludes deterministic input construction):

| fit | baseline | candidate | reduction |
|---|---:|---:|---:|
| PCA | 0.304606 | 0.131975 | 56.7% |
| truncated SVD | 0.238037 | 0.073421 | 69.2% |
| OLS | 0.302430 | 0.138047 | 54.4% |

A final candidate-only three-repeat run after adding the small-work threshold
measured PCA 0.139403/0.128924/0.126434 s, TSVD
0.073328/0.073127/0.075818 s, and OLS 0.141241/0.139912/0.134564 s.

Every repeat and every A/B process produced the same output hashes:

- PCA (components, mean, variances, ratios, singular values, noise):
  `0f7e9f99c560054fc5917b13091fe89494ff7364039b5cfecbcb9ceb6201a4cd`
- truncated SVD: `2701037ce78be46539874d5f666bb093545a190188fde628704ea43b39bb5fb3`
- OLS: `4a2f96df1b85393fff61b7705949a196deac79b723d422a68c9082b57500802c`

Peak RSS was 264,634,368 bytes for the baseline and 255,410,176 bytes for the
final candidate (separate processes; approximately 9.2 MiB lower). The change
only distributes the 128 independent chunk computations. Each chunk retains
its ascending row fold, and the final 128-partial fold remains serial and
ascending. Validation happens before this helper, so error precedence is
unchanged. Fits below 131,072 multiply-add cells remain serial to avoid thread
launch overhead.

A 512x5, two-component threshold regression also matched all three baseline
hashes exactly. Median candidate/baseline times were 0.051/0.053 ms for PCA,
0.041/0.039 ms for TSVD, and 0.153/0.183 ms for OLS, showing no material
small-fit regression.

Build passed with `./bindings/build_estimators_host.sh`. The repository Pixi
Python lacks pytest, and direct import of the test module requires the absent
generated `libMojolearnMath.dylib`; the direct exported-binding benchmark above
therefore supplies the runtime exactness check. `git diff --check` passed.
