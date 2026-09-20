# FAST forest automatic inference qualification (2026-09-20)

- Base: `origin/main` `c91978313`.
- Host: Apple M4, 16 GiB, Metal FAST bindings built with two compile jobs.
- Data: deterministic synthetic classification, 1,000,000 rows x 16 float32
  features, seed `20260920`; target is the sign of `x0 + .5*x1 - .25*x2`.
- Models: 16 trees, depth 10, 8 sampled features, seed 7. Public
  `RandomForestClassifier.fit/predict` and `ExtraTreesClassifier.fit/predict`.
- Candidate: default `inference_engine="auto"` selects the already-qualified
  resident parallel-groves predictor only under FAST. DETERMINISTIC and
  IDENTICAL resolve `auto` to the historical sequential traversal. Explicit
  engine selection always wins.

Two complete fits per family produced identical structural model hashes:

```
RF fit_ms 1361.557 1279.529
RF model_sha256 fe49b447a9cd70ad32709ed78e9827723734e50b19dec208d7818bca55342c6f (both)
ET fit_ms 1025.923 1189.118
ET model_sha256 e5f2b24f6d76e0bdce5748770c2bd05d13b3347a048cda43d3d234990737dbab (both)
```

Seven prediction repetitions were interleaved, reversing order each round:

```
RF sequential_ms 1075.514 763.487 672.318 561.059 543.712 905.340 620.598
RF auto_ms         63.282  39.090  39.613  23.755  33.971  26.727  52.501
ET sequential_ms  723.487 834.908 732.050 605.898 436.182 403.379 438.076
ET auto_ms         60.529  53.581  46.155  24.935  47.877  24.545  68.141
```

Medians: RF 672.318 to 39.090 ms (17.20x, 94.19% lower); ET 605.898 to
47.877 ms (12.65x, 92.10% lower). The first automatic calls, including
resident-model preparation, were respectively 63.282 and 60.529 ms.

Quality parity was exact on this qualification rather than merely within a
tolerance: RF accuracy was 0.991558 in both engines and ET accuracy 0.950769
in both. Complete label arrays were equal. Output hashes:

```
RF 23941ecae17469bb5c609fac198421327130968b0894d57ac18371a12e3f55e5
ET 2c33d9f1bd5d90680e66b2cd2beeade4b2b84ca535a924a9b1a3b9918288e17a
```

Exactness is not promoted into a contract: FAST is authorized for these tree
families and the engines use different reduction orders, so another forest can
legitimately differ in low bits or a near-tie class. IDENTICAL and DETERMINISTIC
keep sequential automatically.

Focused gates:

```
MOJOLEARN_NUMERIC_MODE=fast PYTHONPATH=python pixi run -e test \
  python3 -m pytest -q python/mojolearn/tests/test_forest_inference_engine.py
# 32 passed

MOJOLEARN_NUMERIC_MODE=fast PYTHONPATH=python pixi run -e test \
  python3 -m pytest -q python/mojolearn/tests/test_forest_protocol.py \
  python/mojolearn/tests/test_forest_fit_mode.py \
  python/mojolearn/tests/test_forest_export_protocol.py
# 43 passed, 1 skipped
```
