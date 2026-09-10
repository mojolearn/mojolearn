# Mamba3 fresh prefill and caller transfers — September 10, 2026

The final IDENTICAL defaults reduce public Mamba3 forward time by 18.8% on the narrow grid shape and 25.1% on the wide shape against a same-pod baseline. Full output bits match. The residual gap remains above the requested 1–1.5× target.

| Shape | Same-pod baseline ms | Fresh-only ms | Fresh + caller-copy ms | Final default ms | Reused torch ms | Final / torch |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| B2 L4 D32 | 1.246909 | 1.002133 | 0.992371 | 0.981355 | 1.477005 | 0.66× |
| B8 L4096 D512 | 87.192103 | 79.487529 | 72.250977 | 70.828952 | 16.396241 | 4.32× |
| B8 L1024 D2048 | 166.113550 | 143.104570 | 124.922959 | 124.409189 | 19.383267 | 6.42× |

Medians of five timed rounds, following the archived public harness warmups. The three development arms run sequentially on one H100 80GB HBM3, driver 580.126.09. The final default rebuild restores pristine 5011239a GEMM and uses only `MOJOLEARN_NUMERIC_IDENTICAL=1`; no experimental GEMM candidate was accepted. Mojo 1.0.0(ed45d567), MAX 26.5.0, activated Pixi Python 3.14.7/NumPy 2.5.2. An earlier successful final run used image Python 3.11.10/NumPy 1.26.3; its logs are retained separately as compatibility evidence and its prices are excluded from this table.

## What changed

Stateless `forward(x)` now creates its original zero state directly on device and omits host uploads/downloads for ten discarded resumption-state arrays. It still computes the original numerical block and returns y plus all four public reports. Explicit state, decode, continuation, ordered finite refusals, and caller weight mutation visibility remain covered. There is no weight cache.

IDENTICAL host/device transfers pass the live contiguous NumPy pointer directly to `enqueue_copy`, then synchronize before its owner can expire. This removes a separate HostBuffer allocation and memcpy; see the [Mojo host/device copy documentation](https://docs.modular.com/mojo/manual/gpu/fundamentals/#copying-data-between-host-and-device-memory). The zero-length upload keeps its prior fallback. Other numeric modes retain their old paths.

The original paths remain available with `MOJOLEARN_MAMBA3_LEGACY_FRESH_PREFILL=1` and `MOJOLEARN_MAMBA3_LEGACY_CALLER_TRANSFER=1`. Old extension builds without the new entry retain the Python fallback.

## Validation and evidence

- `h100/m3-fresh-final/fresh-default.log`: all 39,087,232 y/report cells equal explicit-zero-state forward, including both full grid shapes; mutable-weight refusal messages match. The checker also rejects accidental fallback and host cache allocation.
- `h100/m3-fresh-final/surface-default.log`:102 checks, 0failed across the combined Mamba public surface.
- `h100/m3-fresh-final/full-output-identity.json`: three complete output SHA256 values equal the baseline, including both 64 MiB public-grid outputs.
- `h100/m3-fresh/`: baseline/fresh/caller surfaces, full output hashes, timings, source/compiler/GPU manifests, and native default/long/decode-cross/continuation/refusal gates.
- `h100/m3-fresh-final/`: final default results, matching Python runtime, and source manifest. `h100/m3-fresh-final-system-python/` is the additional image-Python compatibility run.
- `apple/`: parent-run combined-candidate build, 146,560-cell fresh check and 102/0surface logs. `apple-final-default/` also passes the rebuilt final integrated default: 146,560 cells, mutable refusals and 102 surface checks.

No opponent was timed here. Torch values reuse the admitted September 7 H100/driver 580.126.09 Mamba3 reference-scan row in `bench/OPPONENT_REFERENCE.md`. Numerical admission passed at rtol5e-4/atol1e-5; the archived 30-witness replay closes the original omitted `--mojo-log` input check. See `../2026-09-09-statepass/opponent-admission/`. Torch 2.4.1+cu124/CUDA 12.4 eager FP32, TF32 off, deterministic off, state 128/head 64/expand 2/chunk 64/seed 7. This compares our public NumPy forward including transfers and reports to the archived device reference scan on matching GPU model/driver; it does not claim a freshly timed opponent on today's physical card.

## Reproduce

Source base 5011239a;  candidate commits f1704718,2396fa6e,c6806c87,8a14ff97,bb98a6cd. Development scripts `h100/m3-fresh-r2.sh` and `h100/m3-fresh-r3.sh` use then-opt-in flags and must run against their recorded intermediate source. For current final source use `h100/m3-fresh-final-build.sh` and `h100/m3-fresh-final-run-r2.sh`, adapting only checkout/job paths. Activate Pixi before both builds and Python execution. The recorded public harness files are the prior `../2026-09-09-statepass/reproduction/` snapshots; supply the committed three Mamba corpus smoke cases for the combined surface gate. Full output binaries and shared libraries are intentionally omitted; SHA256 witnesses and exact commands are retained.
