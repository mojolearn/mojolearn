# RandomForest IDENTICAL on the Apple M4, 2026-09-11: DEVIATION 2502 on the two new datasets

Source ce79f5a9, where 2502 was opt-in for one day: "default" is the
previous forest, "2502 opt-in" was built with `-D MOJOLEARN_2502_PURE_LEAF=1`.
On this evidence the deviation is ON BY DEFAULT since 2026-09-11 evening
(opt out with `-D MOJOLEARN_2502_RETRY_PURE=1`). `bench/speed/forest_speed_arm.py
--lane rf --ours-only`, 1,000,000 train rows, 100 trees, depth 16, sqrt
features, 128 bins, bootstrap, seed 7, 3 rounds, MAX 26.5 Metal. Runs were
sequential (default then opt-in per dataset), so drift is visible in the
istella default arm (34.7 -> 42.4 s over three rounds).

| dataset | arm | fit ms (3 rounds) | hash | logloss | AUC |
|---|---|---:|---|---:|---:|
| taxi 1M x 16 | default | 7206 / 7220 / 7295 | 0ae984630cfca2a8 | 0.525912 | 0.616793 |
| taxi 1M x 16 | 2502 opt-in | 6487 / 6403 / 6554 | d8f64dae01de00bd | 0.525910 | 0.617154 |
| istella 1M x 220 | default | 34667 / 40916 / 42381 | 15e38312cb4bb870 | 0.145578 | 0.964548 |
| istella 1M x 220 | 2502 opt-in | 17929 / 16346 / 18130 | 574b24d0d7af51d0 | 0.145560 | 0.964538 |

The opt-in forest differs from the default forest on both datasets (a
skipped pure node shifts later node indices, which seed the column
sampler). The decision on the default is Andrew's; the H100 leg of the
same day adds the CUDA cells and the opponent rows on these datasets.

## The flipped default, rebuilt from the flip commit (`logs/flipped_default.taxi.identical.log`)

Same binary path with no define: taxi 1M IDENTICAL, one timed round after
one warm-up, 6210 ms, hash d8f64dae01de00bd, logloss 0.525910, AUC
0.617154. Equal to the opt-in arm above, so the opt-out define
`MOJOLEARN_2502_RETRY_PURE` and the flipped default are the same two
forests as this table's two arms, renamed.
