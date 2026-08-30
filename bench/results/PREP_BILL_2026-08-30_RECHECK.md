# The prep bill, re-measured at HEAD nine days later. It held.

2026-08-30, M4 base 16 GB, commit `6c607835`, `nice -n 19`.
`bench/interleaved/end2end_interleaved.mojo`, which is the HONEST row: raw
floats to fitted model, quantization INSIDE the timer on both arms, arms
alternating in one process, 3 reps, 20 trees, depth 6, 128 borders. One
dataset per process, which is that harness's own rule.

## Why this run exists

`PAPER_PLAN.md` item 1 and `mlsys/CLAIMS.md` both carried, as of this
morning, a blocker written on 2026-08-21: `train()` spends "24 s of
preparation around 0.75 s of trees" at 400k x 500, every benchmark
quantizes outside the timed region, and "a reviewer asks what `fit` costs
and today's honest answer destroys the headline."
`mlsys/results/prep-bill-2026-08-21.json` still reads
`"prediction_status": "PREDICTION, NOT A MEASUREMENT. Implementation
parked 2026-08-21."`

**All of that was superseded the same day it was written**, by steps 1
through 12 of `PREP_BILL_2026-08-21.md`, and nothing propagated to the two
planning documents. The last number in that board is nine days old, so it
was re-measured rather than cited.

## The rows

    eps500  400,000 x 500      theirs (s)    ours (s)    speedup
      rep 0                        2.778       2.715      1.02x
      rep 1                        3.074       2.112      1.45x
      rep 2                        2.971       1.757      1.69x

    covtype 581,012 x 54        theirs (s)    ours (s)    speedup
      rep 0                        0.350       0.368      0.95x
      rep 1                        0.358       0.362      0.99x
      rep 2                        0.357       0.377      0.95x

`speedup > 1` means WE ARE FASTER. Their arm is CatBoost's multi-threaded
CPU quantizer plus its CPU learner on the same box, which is the only arm
that runs here.

## Reading

**eps500 is a win, 1.02x to 1.69x, every rep, from 0.27x at the campaign's
start.** The band is wide and the first rep is barely over parity, so the
sentence this supports is "at or above parity on the user-facing path,
1.0x to 1.7x across three interleaved reps", not "1.7x". The spread is the
M4's, not the code's: their arm moves 2.78 to 3.07 s across the same three
reps while ours moves 2.72 to 1.76, which is our arm getting FASTER as the
process warms, and the protocol quotes the band rather than the best rep.

**covtype is parity, 0.95x to 0.99x**, inside the 0.88 to 1.07x band the
step-9 board recorded for the same shape. Prep there is ~0.07 s total; the
covtype story was never the prep bill and was closed as `run_tree_layout`
per-tree setup in step 9.

**The 24 s number is dead and should not appear in any planning document
again.** End to end at 400k x 500 is now 1.76 to 2.72 s.

## One thing that MOVED and is accounted for

The models are not the ones the campaign gated on:

    eps500   ours today 0.76616453125   campaign gates 0.765715234375
                                        (sampled) / 0.76594453125 (full)
    covtype  ours today 0.9455725527    step-9 gate   0.9544611384962789

Their arm reproduces its own recorded value exactly (eps500
0.7658459111467295 against the board's "theirs 0.765846"), so the move is
ours and only ours. It is ACCOUNTED FOR, not a regression: `b55e0b83`
(2026-08-21, DEVIATION 135, "the border subsample copied a real function on
the wrong code path") landed AFTER the gates in `PREP_BILL_2026-08-21.md`
were recorded and changes which values the border build sees. The gate
numbers in that board are therefore historical and a reader must not
re-run them as assertions.

**DONE, 2026-08-30.** Those constants were still written as bit-identity
gates in a board that reads as current. They are now marked superseded in
place at every site, with a banner at the top of `PREP_BILL_2026-08-21.md`
naming `b55e0b83` and carrying the HEAD values above. Nothing in the speed
row here depended on it, because both arms train the same configuration on
the same bytes and each library derives its own grid, which is the point of
this row.

One precision, recorded rather than guessed. Only the sampled eps500 path
and covtype were re-measured today. The full-path constant 0.76594453125
comes from `border_build_max_samples = 0`, which skips the sampler outright
(`gbdt/train.mojo:1053`), so DEVIATION 135 has no mechanism to move it, but
it was not run today and is recorded as UNVERIFIED AT HEAD rather than as
holding. The fit-only constant 0.7658459375 is unaffected for the same
reason and stands, steps 24 and 27 of the campaign board reproduce it in
windows that ran after `b55e0b83` landed.

One thing this repository cannot fix. The two paper-repo paths cited above,
`mlsys/results/prep-bill-2026-08-21.json` and `mlsys/CLAIMS.md`, do not
exist in this checkout and still carry the dead 24 s blocker. That
retraction is owed in the paper repo and cannot be applied from here.

## Provenance

    harness   bench/interleaved/end2end_interleaved.mojo (DEPTH 6, TREES 20,
              REPS 3), tools/catboost_end2end_arm.py
    fixtures  ~/.cache/mojolearn/{eps500,covtype}_{Xcol.f32,y.f32}
    command   pixi run -e bench mojo run -I . \
                bench/interleaved/end2end_interleaved.mojo \
                ~/.cache/mojolearn <prefix> <rows> <feats>
    caveat    one M4, one window each, three interleaved reps. The M4 has
              been measured drifting 1.7x in twenty minutes, which is why
              only same-process alternating ratios are quoted here and why
              the eps500 band is reported whole.
