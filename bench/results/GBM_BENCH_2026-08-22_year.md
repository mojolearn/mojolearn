# gbm-bench year: the symmetric-trees arm wins on NVIDIA's harness, 2026-08-22

First run of NVIDIA's gbm-bench with the mojolearn arms (bench/external/,
wired at `e9c6c8e`). Their harness, their dataset, their metrics, their
CatBoost arm untouched; the M4 laptop is the box, where CatBoost's own GPU
arm cannot run at all ([[mojotrees-benchmark-arena]] framing: that is the
win condition, not a caveat).

Dataset: year (YearPredictionMSD), 515,345 rows x 90 features, regression,
the harness's own 80/20 split. Config: gbm-bench's shared parameters --
500 trees, depth 8, learning_rate 0.1, l2 1 -- with `mojolearn-gbdt-gpu`
mirroring `cat-cpu` param for param (border_count 254 explicit; RMSE
boost_from_average emulated by target centering INSIDE the fit timer;
PARITY_NOTES in the adapter). Both arms interleaved in ONE process per
invocation; three whole invocations for spread.

    invocation   cat-cpu train_s   mojolearn-gbdt-gpu train_s   ratio
    1            19.606            14.916                       1.31x ours
    2            30.924            18.379                       1.68x ours
    3            20.920            15.795                       1.32x ours

    accuracy (MSE): cat-cpu 79.975 every invocation;
                    ours 80.174 / 80.063 / 80.181  (+0.11..0.26%)
    MAE: ours 6.258-6.262 vs theirs 6.263 (ours marginally better)

OURS IS FASTER IN 3 OF 3 INVOCATIONS, 1.31-1.68x, at accuracy parity.
Invocation 2 drifted on both arms (the box's known thermal drift) and the
in-process interleave is what keeps the ratio meaningful there.

Read this beside the covtype loss (PREP_BILL 2026-08-21 step 29, ours
1.32-1.43x SLOWER): year is 90 features to covtype's 53, so the per-row
work dominates the 94-launch/tree fixed cost -- the marginal-cost regime
where [[mojolearn-fixed-vs-marginal]] measured our per-row work faster than
CatBoost CPU's. The two results are the same diagnosis from both sides.

Two small-print items, named rather than buried:

1. cat-cpu's MSE is bit-stable across processes; ours moves in the 4th
   significant digit (80.06..80.18). Suspect: `border_build_max_samples`
   200000 < 412k train rows, so border building SUBSAMPLES, and the
   subsample seed appears to vary per process. Needs a look -- a fit that
   is reproducible from its seed should not move across invocations of the
   same config. Flagged for the lane that owns binarization.
2. gbm-bench's binary-classification metric calls sklearn's
   `log_loss(eps=)`, removed in sklearn 1.5, so binary datasets (higgs,
   fraud, airline, epsilon, bosch) crash in THEIR metric code under this
   environment. year (regression) and covtype (multiclass) are unaffected.
   covtype is not runnable by the GBDT arm anyway: the python surface has
   no MultiClass and the arm refuses it by name.

Raw JSON + box records beside this file: gbm_bench_year_2026-08-22_0756*,
0757*. Harness commit is pinned in each .env.txt.
