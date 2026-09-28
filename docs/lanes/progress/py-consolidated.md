# py-consolidated: the twelve Python-work lanes, merged, and ONE check

Branch `lane/py-consolidated` (worktree ~/mojolearn-wt/py-consolidated), cut from
origin/lane/apple2-merged a374c8c08. Andrew's order (2026-09-28 ~20:00Z): combine every
Python-work lane into one branch and run ONE consolidated check; no other lane checks.
Brief ~/mojolearn-evidence/py_work_brief.md, audit ~/mojolearn-evidence/python_work_audit.md.

## Branches merged (each at its FINAL tip, all ancestors of this branch)

| branch | tip | what it moved (its own progress file has the detail) |
|---|---|---|
| py-shared | bad280b0a | `_portable_math` fsum / isfinite fast paths; label callers through the native encoder; arena ranges runner and DeviceStore (x_metrics, x_prep) |
| py-bugs | 4edf69335 | pinned host math (exp_array, nsum, powi, powr, erfc, normal cdf/inv cdf; DEVIATIONs 6900 to 6903, rows 250 to 253); search folds once; learning_curve one permutation per fold; RNN int32 order |
| py-sequence | 6f0267b28 | schedules by fixed-width interval (5540); optimizer scalars carried and used slots only; LayerNorm backward skips y; Theta/ETS repeat cache; Prophet O(n) sort check; Holt-Winters lazy components and index slices; KPSS cast; parallel forecasting strided copies |
| py-lm | 3f2da7fc0 | byte LM native validation and argmax; Samba head + loss + head backward in one call; CausalLM.generate in one resident call |
| py-decomp-nbrs | aadfff640 | ParallelQueries estimator once per worker; MinCovDet fast_mcd, LDA online, non-metric MDS native; LDA E-step on device |
| py-dn-kern | 7203261f6 | fused KernelPCA / OCSVM / SVGP chains; spectral dense COO and precomputed kNN affinity in Mojo |
| py-dn-svm | 6aaa86d03 | SVC epilogues, Platt fit, SplitMix shuffle in svm/host/svc_proba.mojo |
| py-dn-ann | 87d6d5875 | IVF-Flat and x_ann indexes resident (DEVIATION 1804 retired); native distributed IVF merge |
| py-misc | 893120acf | CNNClassifier.fit epoch entry (proven on nvc1-0002) |
| py-misc-msel | e93c05c11 | model_selection splitters, scorer column, permutation test, check_cv through core helpers (proven before == after on nvc1-0001) |
| py-misc-metrics | 5a541c18b | x_metrics epilogues (PR/DET, ndcg, class sums, auc, MI cells, CH/DB); DEVIATION 6106 and row 139 narrowed |
| py-misc-prep | 2af9eef2d | IterativeImputer(estimator=) plumbing and CalibratedClassifierCV in Mojo (row 221, DEVIATION 5411) |

Merge order: py-shared first (its APIs), then the decomp-nbrs family, the misc family,
py-bugs, py-sequence, py-lm, then each lane's FINAL progress commit.

## Conflicts resolved (toward both intents)

| file | lanes | resolution |
|---|---|---|
| IDENTITY_PATHS.md row 198 (DEVIATION 6106) | py-misc-metrics, py-bugs | py-misc-metrics' narrowed text (scalars only in Python; every data-length epilogue in `x_metrics/epilogue.mojo`) with py-bugs' `_sq` / nsum and the model_selection squares recorded as done, not owed |
| IDENTITY_PATHS.md row 253 (DEVIATION 6903) | py-bugs, py-dn-svm | the PIN stays; the row now says the estimator runs the Mojo transcription `svm/host/svc_proba.mojo` and the Python is the reference |
| `_expansion_metrics.py` `_fsum`, `_sq` | py-shared, py-bugs, py-misc-metrics | `_fsum = pmath.fsum` (py-shared's shared fast path; the same bits as py-bugs' local copy: math.fsum when finite, +0.0 for a zero sum, the exact sum otherwise); ONE `_sq` at the top (py-bugs and py-misc-metrics each added one) |
| `_portable_math.py` imports | py-shared, py-bugs | `decimal`, `functools` and `math as _cmath` all kept |
| `_expansion_neighbors.py` SVGP | py-dn-kern, py-bugs | py-dn-kern's `_gamma_value` (the fused chains need it) spells `ls * ls` (py-bugs: no platform pow); `_k` calls it |
| `model_selection.py` | py-misc-msel, py-bugs | py-bugs' `_cross_validate_folds` keeps its name (the searches call it); py-misc-msel's permutation_test_score calls it instead of its own `_cross_validate_on` |
| `_svm_impl.py` | py-dn-svm, py-bugs | py-dn-svm's native wrappers and reference note, plus py-bugs' DEVIATION 6903 note |
| `_x_sequence_rnn.py`, `sequence/pyapi.mojo` | py-bugs, py-sequence | both lifted the RNN order cap; kept py-bugs' int32 order (the binding makes the exact float32 copy) and its `_schedule`; the step table is the same int32 pairs |
| `_x_sequence_autoarima.py` | py-bugs, py-sequence | both pinned the BIC log to `_portable_math.log`; kept py-bugs' `_pm` spelling |

Semantic fixes found in the merge (no textual conflict):
- `_expansion_prep.py` IterativeImputer native route: py-misc-prep's convergence twin chose
  CPython's `sum` order by interpreter version, but py-bugs made the Python reference
  `_pm.nsum` (3.12+ order on every interpreter). The twin now always takes the compensated
  spelling (`comp = 1`); row 221's text says so. The now unused local `import sys` is gone.
- `_expansion_cnn.py`: py-misc's epoch entry still averaged `loss_curve_` with builtin `sum`;
  it now uses `_pm.nsum` like py-bugs' step loop, so both routes give the same bits.
- `_surface_metrics.py`, `_surface_decomp.py`: the new host exports of py-misc-metrics (nine
  `x_metrics_*` epilogue entries) and py-decomp-nbrs (`x_decomp_mcd`, `x_decomp_lda_online`,
  five move entries, and their host modules) were registered in PyInit but missing from the
  manifest that `test_host_surface.py` holds equal to it.
- Prep docstrings: the NormalDist exception is gone (py-bugs pinned it, DEVIATION 6902).

NumPy (orchestrator note 1): the merge adds no NumPy import site (14 in the shipped package
before and after; `_ivf_impl.py` moved its lazy import inside a helper). py-sequence's
`np.take` in Theta and the RNN step table live in modules that already import NumPy at module
scope (docs/NUMPY_RELEASE_BLOCKERS_2026-09-28.md lists them); nothing new blocks the release.

## Lanes selected and why (tools/py_consolidated/lanes.txt, 168 lanes)

Function level, not file level: a lane is in when its fit calls a changed function or binding
entry. Shared files were NOT a reason to take every lane:
- `_portable_math` fsum / isfinite fast paths: callers are held by
  `test_portable_math_fast.py` (the exact paths' bits over 4000+ sums and 20000 bit patterns);
  the three tree lanes that sum with it (trees-adaboost-clf, trees-dart-clf, trees-oob-cv-link)
  are in. The new helpers (nsum, powi, powr, exp_array, erfc, normal cdf) are called only at
  py-bugs' sites, whose lanes are in.
- The base binding `_mojolearn` and `_mojolearn_core_host` gained entries only (dense COO):
  only the spectral lanes that call them.
- `training/estimator.mojo` (py-lm's CE split): every caller of cross_entropy through the
  training binding (cross-entropy-arms, training-primitives, samba lanes). The byte LM
  bindings do not import estimator.mojo, so the GPT-3 route's binding is untouched.

| group | lanes | reached through |
|---|---|---|
| py-shared (72) | every x-prep and x-metrics lane; metrics, metrics-classification; svc, svc-linear, svc-poly, x-neighbors-svc-*, x-neighbors-svm-weights; linear-svc, linear-svc-squared-hinge; knn-clf, knn-clf-distance; cross-val, cross-val-folds, permutation-test; trees-adaboost-clf, trees-dart-clf, trees-oob-cv-link | the arena ranges runner (`_Prog.run`, `_execute`: every x_prep and x_metrics program), the native label encoder, fsum |
| py-bugs (24) | x-cluster-bgmm(-inits, -covtypes), x-cluster-gmm-options, x-logistic-cv(-w), x-huber, x-ridge-clf, x-metrics-*, x-cnn-trainer(-options), x-prep-iterative-options, x-prep-user-objects, x-neighbors-svgp, x-neighbors-svc-probability, x-decomp-grp, x-decomp-srp, sequence-autoarima, sequence-rnn/lstm/gru | each pinned host-math site; gbdt-catboost-defaults added for `_c_round`'s powi |
| py-sequence (30) | sequence-* (optimizers, schedules, layernorm, theta, ets, prophet, autoarima, rnn family, ...), arima*, holtwinters*, kpss; par-forecast-holtwinters added | `sequence/pyapi.mojo`, the `_x_sequence_*` modules, `_tsa_impl`, parallel_forecasting |
| py-lm (8 + 6) | byte-lm, byte-lm-resident, byte-lm-host-train, samba, samba-untied-dropout-accum, transformer, transformer-decode-session, mamba1-decode-session; added byte-lm-host-infer, par-byte-lm, par-samba, hf-causal-lm, cross-entropy-arms, training-primitives | byte LM validation and argmax, samba_head_loss, the CE split, CausalLM.generate |
| py-decomp-nbrs family (43) | x-decomp-robust-cov, -lda, -manifold, -spectral-rbf, -nmf, -dict-learning, -sparse-pca, -als, -factor-analysis; the SVC/SVM lanes and par-svm; ivf*, par-ivf, x-ann-*; x-neighbors-kpca, -ocsvm, -svgp; par-queries-*; spectral*, par-graph-spectral, x-cluster-spectral-affinities | the lanes' own lanes_in_scope.txt |
| py-misc family | x-cnn-trainer(-options); x-metrics-splitters/-search; x-prep-iterative-*, x-prep-user-objects, trees-calibrated, trees-oob-cv-link; par-cross-val added | CNN epoch entry, model_selection routes, xtrees platt/isotonic strided entries |

## THE ONE CHECK: nvc1 job (pending)

`tools/py_consolidated/job.sh`, one queue job on the shared NVIDIA pod nvc1 (2x A40, Xeon Gold
6342 as the x86 CPU column), `MOJOLEARN_XD_RES_DEV_MIN=1`. One tree: base = this branch with
`tools/py_consolidated/base.patch` reversed (the code diff against a374c8c08, regenerated by
`make_base_patch.sh`; tools, bench, docs, prose, tests and sabotage patches excluded), built
once; head = the patch re-applied, only moved bindings rebuilt. Phases: base arms and the
py-lm witness; head arms and witness; cross (base vs head per lane and column, part by part;
witness base vs head, the GPT-3 guard); pytest; two sabotage arms (py-dn-ann resident index on
ivf, x-ann-ivf-pq; py-dn-kern fused chain on x-neighbors-kpca); the lanes' in-build reference
arms; one small interleaved timing pass (GPU base head head base, CPU base head). py-sequence's
/root/ps-base was NOT reused: it is a 0a11b50c7 tree, not this base.

Job: nvc1-0018 (submitted 20:10:18Z behind apple2-merged 0006 to 0009 and 0016).

## Intended behavior changes (base vs head MAY differ; GPU == CPU must hold on head)

- py-bugs 2: GridSearchCV, RandomizedSearchCV and validation_curve draw folds once
  (scikit-learn), so an unseeded shuffling splitter no longer redraws per candidate.
- py-bugs 3: learning_curve takes one permutation per fold, nested prefixes per size.
- py-bugs 4 / py-sequence 4: RNN, LSTM and GRU fits of more than 2^24 order entries no
  longer refuse (the order is int32).
- py-bugs 1: bits move only where the host's libm or pow was not correctly rounded
  (cluster predict_proba, LogisticRegressionCV predict_proba and Cs grid, AutoARIMA ic_,
  CNN loss_curve_ on Python 3.10/3.11, IterativeImputer posterior draws, JL near an integer).
- py-sequence: Prophet refuses NaN in t by name; sequence optimizers' `state_dict()` gains
  `scalars`; Theta and ETS fit keep a private copy of y.
- py-decomp-nbrs, py-dn-ann: the native sorts and the distributed IVF merge refuse a NaN
  by name (the Python tuple sort had no defined order); DEVIATION 1804 closed.
- py-shared: x_prep refuses a read of an INPUT slot (inputs no longer come back).

## Verdicts, timing, owed

(pending the job)
