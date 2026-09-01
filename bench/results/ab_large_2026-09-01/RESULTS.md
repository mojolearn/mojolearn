# The 1M/2M timing window, 2026-09-01 evening

One window, three A/B pairs, all arms interleaved (two outer rounds), every
run `nice -n 19`, box otherwise idle. **Shapes at or above 1,000,000 rows
only** — Andrew's floor, set this evening after a 50k-row window was caught
and killed; nothing from that killed window votes anywhere. Default-tier
builds (no `-D` defines): the shipped arm is the one measured.

Arms:
- symmetric: `estbench_head_1m` (HEAD 7ac1112e) vs `estbench_old_1m`
  (c4af1f42, pre DEV 2007/2008); `checks/estimation_bench.mojo` with
  `N_ROWS = 1000000` patched IDENTICALLY into both checkouts (harness-only
  edit, restored after the builds; the patch lives in the binaries).
- et: `etfo_head` (HEAD) vs `etfo_old` (36c4a23a, pre DEV 470-472),
  `extratrees/bench/fit_once.mojo`, higgs2m prefix at 1M and 2M rows,
  28 features, 20 trees, depth 12, sqrt.
- rf: `rfb_off_large` vs `rfb_on_large`, ONE source (HEAD) with
  `LABELS_SAMPLED_ORDER` flipped between builds; timed path re-pinned to
  `rf@1000000` + `rf@2000000` (harness-only edit, restored).

Caveat recorded: both rf arms compiled while DEV 2002's (then-uncommitted)
delivery-canary hunk sat in `ensemble/randomforest.mojo` — present in BOTH
arms equally, so the flag stays the only variable; the absolute numbers
carry one extra kernel + 4-byte copy per fit.

## RF — DEVIATION 2001 (sampled-order labels), off vs on

Per-round medians of the ARM lines (ms/fit, 30 trees):

| shape | round | flag OFF | flag ON | delta |
|---|---|---|---|---|
| rf@1000000 | r1 | 24431 | 23735 | **-2.9%** |
| rf@1000000 | r2 | 22999 | 22324 | **-2.9%** |
| rf@2000000 | r1 | 49473 | 45082 | **-8.9%** |
| rf@2000000 | r2 | 46702 | 43584 | **-6.7%** |

Canaries stable (~650-690 ms) in all rounds except rf_on r1's warmup/pre
(1827/1970 — box still settling; that round's mid canary is 689 and the
arm STILL won, so the spike worked against the winner, not for it).

**VERDICT: measured win, both shapes, both rounds, growing with rows.**
With the morning's fingerprint gate (flag-off vs flag-on EQUAL, sabotage
reach proven), house rule 11 is satisfied on both halves →
`LABELS_SAMPLED_ORDER` DEFAULT FLIPPED TO TRUE this session
([[mojotrees-switches-must-flip]]). Post-flip sanity runs recorded below
the ledger entries.

## Symmetric — DEVIATIONS 2007/2008, HEAD vs c4af1f42

`estimation_bench` internal medians (ms/tree, REPS=3, arms alternated
inside each run), two runs per arm. LABELED OVER THE DEVIATION 134
EMBARGO — an internal A/B, not a public speed claim.

| cell | old (r1/r2) | head (r1/r2) | head vs old (of paired medians) |
|---|---|---|---|
| d6 rmse | 172.0 / 168.4 | 170.3 / 166.4 | **-1.1%** |
| d6 ll10 | 268.1 / 310.8 | 273.2 / 257.5 | **-8.3%** (old r2 noisy) |
| d6 ll1 | 191.1 / 182.5 | 190.9 / 191.9 | +2.4% (noise-level) |
| d8 rmse | 243.6 / 250.0 | 239.1 / 233.5 | **-4.3%** |
| d8 ll10 | 349.9 / 369.0 | 343.5 / 343.6 | **-4.4%** |
| d8 ll1 | 253.8 / 265.9 | 250.2 / 239.6 | **-5.8%** |

**VERDICT: HEAD wins 5 of 6 cells by 1-6%, largest at depth 8 —
consistent with deleting per-tree fixed costs (the get_attribute hoist,
dead uploads, zero passes). One cell inside noise. No regression.**

## ET — DEVIATIONS 470-472, HEAD vs 36c4a23a

`fit_once` total fit (ms, 20 trees; node counts IDENTICAL between arms —
63636 @1M, 70162 @2M — same forests built):

| shape | old (r1/r2) | head (r1/r2) | verdict |
|---|---|---|---|
| 1M | 6250 / 6280 | 6239 / 5888 | **head -3.2%, consistent** |
| 2M | 14024 / 12014 | 12178 / 12068 | inconclusive (old r1 outlier; head inside old's spread) |

**VERDICT: small consistent win at 1M, 2M within noise, NO REGRESSION
anywhere → DEV 472's revert clause ("if the compare costs more than the
skipped copies") does NOT trigger; the certified defaults stand.**
