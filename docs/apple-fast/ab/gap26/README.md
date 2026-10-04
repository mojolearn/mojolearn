# Gap26 board accounting (2026-10-04)

Measured wins only: AutoARIMA fused evaluation tail and LabelBinarizer direct
presence/scatter. The committed raw receipt was fetched by the manager from
M3 `~/mq/out/<tag>.log`; extraction includes failed setup lines, successful
A/B lines, and bounded quality summaries. Only successful **def=B** lines
were converted into the four `t-ab-*.txt` AFB files. `arm=A` inside these
lines is the inner harness label, not the define arm. Each `runs` list has
one scored timing. Source heads come from the manager handoff, since the
bounded AFC lines themselves omit SHA. Opponents were not rerun.

Existing pipeline used: append converted AFB records to the local cached
`final-boards/r9/afb-merge-input.log`, run `tools/af_board_merge.py merge`
with existing `opp-fill.txt` and `opp8.txt`, then apply existing
`final-boards/ab-update/ab-fix.py` using `main0.md` and original `ab-notes.txt`
plus this bundle's notes. Before adding records, this process reproduced
main's board byte for byte. Only four result rows change; old quality flags
and HOLD exclusions are preserved. Labels' shape field is complemented by
the independent classes/indicator/inverse-transform checker receipt.

The board now has 221 eligible faster rows in its 287-row refresh table,
up from 220: LabelBinarizer istella flips to faster. The complete page
includes historical cells: private preview has **336/375 eligible faster**,
up from 335/376. One added winner; denominator drops one because AutoARIMA
taxi-hourly is now explicitly held for its preexisting opponent-quality gap
(RMSE 74.6591 vs 68.21). The fused-tail change itself preserves baseline
quality byte for byte. No other HOLD is removed or weakened.

## Reviewable artifacts and integration

Private generated page and intermediate files:
`~/mojolearn-evidence/apple-fast/gap26-accounting/` contains `data.json`,
`m3-fast-vs-opponents.html`, pipeline outputs and its copied page builder.
It leaves shared `fastpage/` and `final-boards/r9/` untouched. HTML is a local
preview; the externally hosted artifact has not been republished.

After the manager merges this accounting branch (both kernel promotions
must already be on main), integrate the four `t-ab-*.txt` files into shared
`final-boards/r9/`, append missing lines from `ab-tags.txt` and `ab-notes.txt`
to their shared counterparts, and apply `fastpage-quality-flag.patch` to
shared `fastpage/build_data.py`. Rebuild via the existing board/page pipeline.
The patch adds the AutoARIMA taxi quality flag without altering any existing
flag or neural exclusion. Do not substitute copied scripts for concurrently
updated shared scripts. `ledger-entry.md` is the ready-to-append ledger text.

Timing deltas: synthetic ARIMA -6.4%; taxi ARIMA -6.3%; taxi label -5.9%;
istella label -16.9%. Full precision timings and digests are in the raw and
converted records. No new timing, opponent race, build, SSH or main push was
performed by this accounting lane.
