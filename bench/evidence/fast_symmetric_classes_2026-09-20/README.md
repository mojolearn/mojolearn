# FAST symmetric GBDT class-output fusion (2026-09-20)

Scope: public `GradientBoosting.predict_classes`, FAST resident symmetric-tree
models only. Raw `predict`, `predict_proba`, and the IDENTICAL/DETERMINISTIC
routes are unchanged. Widths above three classes retain the old path after the
generalization matrix showed that its already-parallel probability transform is
faster there.

The accepted path reads the resident raw planes back exactly as before, but for
binary and 3-class models performs the existing sigmoid/softmax statements and
first-max selection together. It therefore does not allocate the public
probability matrix. At 1M rows this removes a transient 16 MB Float64 binary
matrix or 12 MB Float32 3-class matrix; the returned 8 MB int64 labels and the
resident raw workspace are unchanged.

## Exactness discovery

An earlier device raw-logit argmax was rejected. A deliberately tie-heavy
3-class model proved that Float32 probability narrowing can create a tie that
does not exist between raw logits. The accepted implementation compares the
actual narrowed probability values, statement-for-statement with
`predict_proba`; all 24 final cases have identical full label SHA-256 values
across arms and rounds. The tie-heavy records are in
`apple-tie-heavy.json`.

## Timing

Hardware was Apple arm64, macOS 26.5.2. Inputs were 500k and 1M rows, 8 or 32
Float32 features, depths 3/6, seeds 3/17/41, balanced and imbalanced labels;
tie-heavy used seed 97. Each cell alternated old/new order for 7 rounds (the two
noisy cells in `apple-reruns.json` use 15). Raw observations, model hashes,
label hashes, and accuracy are in the JSON files.

Representative 1M-row medians:

| classes / features / depth | old ms | new ms | change |
|---|---:|---:|---:|
| binary / 8 / 3 | 13.165 | 12.565 | -4.6% |
| binary / 8 / 6 | 14.507 | 13.215 | -8.9% |
| binary / 32 / 6, 15-round rerun | 57.412 | 56.920 | -0.9% |
| 3 / 8 / 3 | 57.613 | 54.035 | -6.2% |
| 3 / 32 / 6 | 96.095 | 84.652 | -11.9% |
| 3 / 8 / 6 | 62.544 | 54.965 | -12.1% |
| 3 / 8 / 3, tie-heavy | 55.160 | 43.515 | -21.1% |

The initially negative 500k 3-class shallow cell became 31.911 -> 28.573 ms
(-10.5%) in its 15-round rerun. Eight-class calls are deliberately the same
old implementation; differences in their interleaved numbers are scheduler
noise rather than a selected code-path difference.

Commands:

```text
MOJOLEARN_NUMERIC_MODE=fast MOJOLEARN_SKIP_BUILD_GATE=1 MOJOLEARN_BUILD_JOBS=2 bash bindings/build_gbdt.sh
MOJOLEARN_NUMERIC_MODE=fast PYTHONPATH=python pixi run python bench/speed/gbdt_symmetric_classes_ab.py --rows 500000 --rounds 7 --output bench/evidence/fast_symmetric_classes_2026-09-20/apple-500k.json
MOJOLEARN_NUMERIC_MODE=fast PYTHONPATH=python pixi run python bench/speed/gbdt_symmetric_classes_ab.py --rows 1000000 --rounds 7 --output bench/evidence/fast_symmetric_classes_2026-09-20/apple-1m.json
pixi run python checks/gbdt_binary_prediction_smoke.py python/mojolearn/_mojolearn_gbdt.so 0
```
