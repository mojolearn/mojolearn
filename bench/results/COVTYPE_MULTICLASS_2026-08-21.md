# Covtype at its native task: 7-class MultiClass reaches near-parity on our worst dataset

2026-08-21, M4 base, `bench/interleaved/multiclass_interleaved.mojo` +
`tools/catboost_multiclass_arm.py`, 581,012 rows x 53 features, 7
classes, 20 trees depth 6, Newton-1 estimation on BOTH arms (the loss's
own default, `GetEstimationMethodDefaults` `catboost_options.cpp:
106-112`), 3 reps per border, interleaved. First benchmark of the
MultiClass chassis (`ac88e5c`). Every incumbent suite binarizes covtype;
this row runs the task the dataset actually poses.

## The table (speedup > 1 means ours is faster)

    254 borders:
      theirs 59.8 / 60.1 / 60.9 ms/tree, ours 65.0 / 63.6 / 64.8
      -> 0.92 / 0.94 / 0.94x
      loss ours 0.5951051248 (bit-identical), theirs 0.5951050882 -- 7
      significant figures

    128 borders (their GPU default):
      theirs 59.2 / 61.8 / 60.4, ours 63.1 / 63.5 / 62.8
      -> 0.94 / 0.97 / 0.96x
      loss ours 0.5953034318 (bit-identical), theirs 0.5953033826 -- 7
      significant figures

## The two findings

1. **Near-parity on the dataset shape we lose worst.** Covtype RMSE at
   128 borders trails 1.8-2.1x (the launch floor dominates a ~20 ms
   tree). MultiClass carries 1 + (K-1) = 7 stat planes, multiplying
   per-level histogram work ~3.5x on BOTH arms -- which amortizes our
   fixed floor (now ~15% of a 63 ms tree instead of ~50% of a 20 ms one)
   while their CPU scales with the work. Result: 0.92-0.97x, a 2x deficit
   closed to 3-8% by the task itself. Classes join rows, features and
   depth as the fourth axis that amortizes our floor and multiplies
   their bill.

2. **The first 254-border multiclass fit ever attempted found a real
   dispatch defect.** The fused two-stat 8-bit arm (8010b2f) had replaced
   the >128-bin shared-Int32 route unconditionally; when MultiClass
   landed later in another lane, the first multi-stat 254 fit hit the
   arm's `stat_count != 2` guard instead of a histogram. Fixed in this
   commit: multi-stat shapes route to the PASS family, whose shared-Int32
   arm walks stat pairs on the z axis as their ladder does. Reach proof:
   this run raised before the fix and completed after it; correctness
   proof: the 254 loss column against CatBoost's own output, 7
   significant figures. A cross-lane integration seam of exactly the
   kind the parallel-work ledger predicts -- the guard was written when
   every caller was two-stat, and the caller that wasn't arrived four
   days of commits later.
