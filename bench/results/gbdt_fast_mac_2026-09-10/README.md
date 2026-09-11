# GBDT on the Apple M4, 2026-09-10 night: DEVIATION 2512 in the boosting loops

HIGGS first 1,000,000 rows x 28 features, the harness's gbdt configs (100
iterations, depth 6, lr 0.1, l2 1, 254 borders, Logloss;
`bench/speed/forest_speed_arm.py --ours-only`), one warm-up dropped, MAX 26.5
/ Mojo 1.0.0, macOS Metal, wall time of `fit`. Before and after were not
interleaved (the "before" is the single stage-mode replicate that ran a few
minutes earlier in the same session), so these are directions with a
same-bytes guarantee, not a certifiable A/B.

Every `enqueue_memset` under `gbdt/` (32 sites, the checks excluded) now
goes through `core/device_zero.enqueue_fill`: the zero kernel for a zero,
a scalar fill kernel for any other value. The bytes written are the same,
so the forests are: every hash below is unchanged, and
`identity_break --lanes gbdt-symmetric,gbdt-depthwise,gbdt-lossguide,gbdt-rmse`
under the IDENTICAL build equals the retained Apple set 36 of 36
(`logs/identity_break.gbdt.txt`). The mechanism is the one measured in
`bench/results/rf_fast_mac_2026-09-10/`: on Metal a memset placed between
kernel launches costs the host about 110 us more than the launches
around it, and more for large regions; a kernel in its place does not.

| lane, FAST | before ms (1 replicate) | after ms (3 rounds) | hash | logloss / AUC |
|---|---:|---:|---|---|
| gbdt-symmetric | 4,312 | 2,544 / 2,557 / 2,591 | a7bf00cc792d1d06 | 0.542247 / 0.800537 |
| gbdt-depthwise | 8,499 | 6,318 / 6,374 / 6,500 | 5d07cb8b24ff2a5b | 0.525078 / 0.813746 |
| gbdt-lossguide | 17,713 | 10,865 / 10,932 / 11,232 | 144e7b90f8144f3b | 0.525733 / 0.812999 |

IDENTICAL gbdt-symmetric after: 2,837 / 2,855 / 2,871 ms, hash
dac2cf366e219cec (the H100 hash of the same cell).

No M4 CPU opponent was run tonight. The Sep 2 M4 row for CatBoost CPU
symmetric at HIGGS 1M (3,081 ms) came from a swap-loaded box and is not a
row to quote (`bench/results/fast_speed/mac-2026-09-02-higgs-1m2m-local/`).
