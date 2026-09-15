# par-forest-pool, par-gmm, par-resample and par-hdbscan — one and two H100s

The four identity_break lanes for the multigpu lane's merged drivers, run on
one RunPod box with two NVIDIA H100 80GB HBM3 (sm_90a): once with
`MOJOLEARN_PAR_DEVICES=0` (`one.json`) and once with `MOJOLEARN_PAR_DEVICES=0,1`
(`two.json`), beside the plain lanes they are held to, then
`tools/identity_break.py --diff one.json two.json` (`diff.txt`).

Final run: pod `akgvh5vela11fr`, commit `f067bbbc0` (`commit.txt`), lanes
rf-clf, gmm, bootstrap, hdbscan, par-forest, par-forest-pool, par-gmm,
par-resample, par-hdbscan on all nine fixtures:

    summary: IDENTICAL=81
    summary (infer/model): IDENTICAL=72, N/A=90
    summary (batch): IDENTICAL=45, N/A=36

No cell reads REFUSED, so every `_same_bytes` hold inside the par lanes (the
driver against the one-device public path) passed on both device counts.
The N/A cells are the lanes' declared `n/a:no-save`, `n/a:function` and
`n/a:transductive` model, infer and batch parts.

`earlier/` keeps the gates of the two earlier runs of the same lanes as they
were added: pod `tovq1aunx17sen` at `22daec9f2` (par-forest-pool, par-gmm:
IDENTICAL=45, 72, 45) and pod `sftj4ig2arrqe2` at `309be6db9` (plus
par-resample: IDENTICAL=63, 72, 45).

This is a same-box one-device against two-device equality on NVIDIA only.
No AMD or Apple column of these lanes has been recorded.
