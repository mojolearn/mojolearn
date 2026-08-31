# The covtype policy-mix defect is closed

**Measured 2026-08-31**, `checks/searcher_parity_covtype_check.mojo` with
`-D MOJOLEARN_COVTYPE_4K=1`, on the M4, at the commit that fixed
`gbdt/methods/pointwise_scores_calcer.mojo`.

    rep 0  greedy 0.133723 s  pointwise 0.184128 s  ratio 0.726
    rep 1  greedy 0.127817 s  pointwise 0.172196 s  ratio 0.742

    mse greedy     0.8593346476554871
    mse pointwise  0.8593346476554871
    identical      True

## What it was

`NEXT_TWO.md` recorded the two searchers disagreeing on the MIXED-policy
covtype fixture: greedy `0.9486324077643835`, pointwise `1.1843180519507341`,
while each policy ALONE was bit-identical -- `covob_*` (10 one-byte features)
and `covbin_*` (43 binary features) both matched. Only the 53 together failed.

## What it was, actually

`PolicyScoreHelper` allocated `d_cat_w` and `d_bin_w` at the policy's
BIN-FEATURE count and `pointwise_scores.mojo:782-785` indexed them by GLOBAL
FEATURE id. Covtype is the only fixture in the tree whose binary features are
not the low-numbered ones: they are ids 10 through 52 against a binary-policy
total of 43, so ten features read off the end of a live device buffer and
multiplied their score and gain by whatever was there.

Every other fixture puts the binary features first, so the read landed in
bounds by a single element and no gate could see it.

## What is still owed

This is the 4,096-row arm, which carries the SAME fold file and therefore the
same mixed-policy shape and the same global feature numbering as the full one.
The original divergence was measured at 581,012 rows and that arm has NOT been
re-run. The shape argument is strong and it is not a measurement; the full arm
is owed before the defect is called closed in `NEXT_TWO.md`.

The A/B named in the fix's commit message -- refill the enlarged buffer's tail
with 0.0 and watch 1.1843 return -- is also not run. It would establish
CAUSATION rather than correlation. Recorded as owed rather than quietly
dropped.
