# TEMPORARY. Delete when every workstream below is merged or moved into a lane brief.

Written 2026-09-14 by the mojolearn-97 session at Andrew's request, after three read-only
censuses (detail with file:line in `docs/lanes/BRIEF_claim_surface_census_2026-09-14.md`).
Andrew's three goals, in his words: expose whatever we have implemented and finish what is
half built; make everything bitwise identical across GPU and CPU; make everything as easily
verifiable by external users as reasonably feasible. This file is the findings, the plan,
the split between the two live sessions, and the prompt for the peer session.

## 1. Findings, one screen

Main at 0569b3639. Public surface: about 45 classes with fit, predict, transform, forward or
step; 47 identity_break lanes (46 plus the new pca-whiten), one pinned configuration each,
nine fixtures, three columns (train, infer, model).

| what | state |
|---|---|
| three GPU vendors, training | 46 of 46 lanes, 414 cells identical (2026-09-14 record) |
| three GPU vendors, inference and model bytes | 37 of 46 lanes, 459 cells identical; 9 lanes have no probe (kmeans dbscan agglomerative spectral holtwinters gemm-pinned metrics arima iforest); iforest and holtwinters probes in flight on the peer's lane/identity-infer-probes |
| CPU inference identical to the three GPU columns | forests and GBDT (8 kinds, 7 CPUs), 13 classical lanes (ols ridge tsvd logistic pca lasso elasticnet kde svc pca-whiten knn knn-clf knn-reg), byte LM |
| CPU training identical to the three GPU columns | 7 lanes (gemm-pinned kde holtwinters lasso elasticnet svc, byte LM); agglomerative, et-clf, et-reg, iforest in flight on lane/cpu-training-phase1b |
| no CPU path at all | kmeans dbscan spectral umap gp arima svr scalers metrics mlp mamba transformer samba, and rf, et, gbdt TRAINING (no host oracle exists for the forests, GBDT or DBSCAN) |
| parameter values that select a numeric path and are never run | about 187 across about 128 parameters; 57 lanes plus a NaN fixture cover all (brief section 3) |
| defects in the claim | byte-lm-host-infer pins the NON-default single-thread kernel; GBDT nan_mode unreachable (no fixture has a NaN, quantizer collapses Min/Max to Forbidden); bagging_temperature and subsample silently ignored without bootstrap_type (contract says refuse by name) |
| built, gated, no public door | HDBSCAN 7.1k lines, GMM 7.9k, kernel methods 7.7k, resample 7.2k, Cholesky 6.3k, IVF 5.5k (check-ivf ALL OK with one card on Apple, NVIDIA and AMD since 2026-09-14; door `IVFIndex` on lane/expose-ivf-embedding), embedding lane 6.4k (clause (a) card identical on Apple, NVIDIA and AMD and the sixteen sabotage arms run on all three since 2026-09-14; door `Embedding` on the same lane), tokenizer 1.9k; about 50k lines |
| implemented inside shipped bindings, never routed | multinomial logistic (softmax loss exists, estimator hardcodes binary, five-arm gate `glm/checks/multinomial_check.mojo` has NO INVOKER); six other QN losses; six training primitives exported by the shipped binding but not in `training.py`; KMeans cosine/sqrt metric and classic k-means++ arm; forest resident layout, vector groves, GBDT per-round paths (bench-only) |
| shipped | 16 GPU bindings per vendor; of the 8 host bindings only the byte LM one is in a wheel; the wheel ships no reference card so `python -m mojolearn verify` from pip exits 5 |
| external verification today | free: fork and run the three CPU gates on hosted runners (no secrets); one GPU: pip install, clone, run identity_break, diff against the committed columns (undocumented); three vendors: our rental plumbing only |
| register of named absences `_NOT_YET` | EMPTY |

## 2. Workstreams

Each row: what, files, needs a GPU, size, and who (section 3).

| id | workstream | main files | GPU | size | owner |
|---|---|---|---|---|---|
| A | 57 lanes + `nan` fixture + flip byte-lm-host-infer to threaded=True with a second single-thread lane | `tools/identity_break.py` | write: no; smoke: Mac Metal under the two-core cap; rerun: three boxes | 1 day to write, 1 leg per vendor | 97 WROTE (71 lanes, `lane/claim-surface-lanes`, every lane STABLE on the Apple M4, base fixture, all nine fixtures owed in the log); ea reruns the three columns |
| B | contract defects: refuse `bagging_temperature`/`subsample` without `bootstrap_type`; refuse `cross_val_score(groups=)`; tests | `python/mojolearn/ensemble.py`, `model_selection.py`, tests | no | hours | 97 DONE (`test_refuse_ignored_knobs`) |
| C | multinomial logistic: pixi tasks for `multinomial_check` and `qn_losses_check`, run on three vendors; if green add a loss id to `glm/estimator.mojo` and the binding, lift the C>2 refusal in `linear_model.py`, lane `logistic-multiclass`, host twin (softmax on CPU) | `glm/`, `bindings/_mojolearn_estimators.mojo`, `linear_model.py`, `pixi.toml` | yes | 1 to 2 days | ea |
| D | expose door-less lanes, one at a time: binding + Python class + tests + lane + three columns + host twin. Order: tokenizer, Cholesky, kernel methods, GMM, HDBSCAN, resample, IVF (needs NVIDIA and AMD first), embedding (needs a sabotage arm and an NVIDIA leg first). Route the six training primitives and the KMeans metric arm | one package each | yes | 1 to 3 days each | ea, after C |
| E | CPU training for the 39 lanes without it; forests, GBDT and DBSCAN need a host oracle written first (brief `BRIEF_cpu_training_2026-09-13.md` section 1 costs each) | `bindings/build_host.sh <family>` (the peer's manifest builder), `core/*_host*.mojo` | yes for the columns | weeks; the largest item | ea, phases 2 and on; `lane/cpu-training-e` WROTE knn x3, pca, pca-whiten, tsvd, ols, ridge, dbscan (compile-checked on the M4, four-column diff owed) |
| F | ship: seven host bindings in both wheels (the peer's manifest lane), harness + three columns + reference card in the wheel, `python -m mojolearn identity` that diffs the local box against the shipped columns | `packaging/`, `python/mojolearn/__main__.py`, `_verify.py` | release legs | 2 days plus a 0.8.6 freeze | ea (manifest), then 97 (identity command) |
| G | external verification: `docs/VERIFY_EXTERNALLY.md` with the three recipes at three costs and exact expected output; a per-vendor container recipe that takes a wheel and emits a column JSON with none of our rental plumbing; record the wheel sha256 in each column JSON; attach columns and diff to the GitHub release and Zenodo | `docs/`, `packaging/linux/`, `tools/identity_break.py` (one field) | validate: yes | 1 day to write, 1 leg to validate | 97 WROTE (`docs/VERIFY_EXTERNALLY.md`, `tools/verify_external.sh`, `package.bindings` sha256 in every column JSON); ea validates the script on a box that did not build the wheel |
| H | truth: fill `_NOT_YET` with every row of the door-less table; SUPPORT_MATRIX row for tsa; KPSS three-vendor record; `_forest_host` ImportError text once F ships | `python/mojolearn/__init__.py`, `SUPPORT_MATRIX.md` | KPSS record: yes | hours | 97 DONE (register filled with ten rows, tsa row in the matrix); ea (KPSS record) |

Dependencies. A's rerun waits for lane/identity-infer-probes and lane/cpu-training-phase1b
to merge so ONE three-column rerun covers the probes, the new CPU lanes and the 57 lanes.
F's identity command waits for the manifest lane. D and E share the host builder. C before
D because it is the cheapest new capability and proves the routing pattern.

## 3. The split, and why

Peer session mojolearn-ea owns everything that needs a rented box or the host builder: C,
D, E, the manifest half of F, the validation legs of A and G, the KPSS record. It has the
boxes, the leases, the subagents, and its manifest lane is rewriting the builder that D and
E depend on.

This session (97) does, in isolation on a branch, the four things that need no GPU and touch
files the peer is not editing: B, G, H, and the WRITING of A. A edits
`tools/identity_break.py`, which lane/identity-infer-probes also edits; so A is written on
`lane/claim-surface-lanes` and rebased after the probes lane merges, then handed to ea for
the rerun. Nothing in B, G or H touches a file on the peer's three branches.

Do NOT split C, D or E across sessions; each is one binding, one builder, one set of legs,
and two sessions on one binding is the Sep 13 "another lane moves HEAD under you" failure.

## 4. Prompt for the peer session (copy from the line below)

---

Read `docs/lanes/TEMP_claim_surface_plan_2026-09-14.md` and
`docs/lanes/BRIEF_claim_surface_census_2026-09-14.md` first. Andrew's goals: expose
everything implemented and finish what is half built; make every lane bitwise identical
across the three GPUs and a CPU, training and inference; make it externally verifiable.
The mojolearn-97 session is writing workstreams B, G, H and the lane definitions of A on
`lane/claim-surface-lanes`; do not edit `tools/identity_break.py`, `ensemble.py`,
`model_selection.py`, `python/mojolearn/__init__.py`'s `_NOT_YET`, or `docs/VERIFY_EXTERNALLY.md`
beyond what your in-flight lanes already change. Everything else below is yours.

Finish first, in this order, merging and pushing each as it turns green:

1. Your three in-flight lanes (cpu-training-phase1b, identity-infer-probes,
   host-surface-manifest) and the two legs (RTX 5090 46-lane, NVIDIA + AMD record legs for
   kde, svc, pca-whiten, knn, knn-clf, knn-reg). Never cancel an owed run.
2. Workstream C, multinomial logistic. Add pixi tasks that invoke
   `glm/checks/multinomial_check.mojo` and `glm/checks/qn_losses_check.mojo` (today nothing
   invokes them). Run both on Apple, NVIDIA and AMD. If green: add a loss id parameter to
   `glm/estimator.mojo` (today `pams.loss = QN_LOSS_LOGISTIC` at :320 and :373) and to
   `bindings/_mojolearn_estimators.mojo:441`; lift the more-than-two-classes refusal at
   `python/mojolearn/linear_model.py:1013-1018`; add lane `logistic-multiclass` to
   identity_break (coordinate with 97, who owns that file this week; send the lane body and
   they will add it); add the softmax decision path to `core/classical_host_predict.mojo`
   and the estimators host binding; record on three vendors and check on the CPU path.
3. After 97's `lane/claim-surface-lanes` merges: ONE three-column rerun of every lane
   (about 100) on Apple M4, H100 sm_90a, MI300X gfx942, from the installed wheel where
   possible, with `MOJOLEARN_COMMIT` set and the host set built as the CPU column. Record
   under `bench/results/identity_break/<date>_<n>-lanes/`, README with boxes, commits and
   the diff summary. Any DIVERGENT cell is a lane brief, not a docs edit.
4. Workstream D, one lane at a time, cheapest first: tokenizer (no float arithmetic, a
   binding and a Python class and exact-id tests), Cholesky (already inside the GP build),
   kernel methods, GMM, HDBSCAN, resample, then IVF (needs NVIDIA and AMD evidence first),
   then embedding (build the sabotage arm and run an NVIDIA leg before exposing). For each:
   binding, Python class in `__all__`, tests as modules, identity_break lane (send to 97 or
   add after their merge), three columns, host twin, SUPPORT_MATRIX row. Also route the six
   training primitives (`_training_impl.py:1905-1983`) through `training.py`, and the
   KMeans cosine metric and classic k-means++ arm through `cluster.py`.
5. Workstream E, CPU training for the remaining lanes, in the order of
   `docs/lanes/BRIEF_cpu_training_2026-09-13.md` section 1 by cost, through your
   `bindings/build_host.sh <family>` builder. Forests, GBDT and DBSCAN need a host oracle
   written first; write the oracle as a second spelling that imports only
   `checks/numerics.mojo` leaves, gate it against the device with a sabotage build, then
   promote it. Every lane passes only when
   `identity_break.py --diff <3 gpu columns> <cpu json> --lanes <lane> --require-columns 4`
   reads IDENTICAL on every cell.
6. Workstream F, packaging: ship all eight host bindings in both wheels through the
   manifest; ship `python/mojolearn/reference_cards/` so `verify` works from pip; ship
   `tools/identity_break.py` and the three current column JSONs inside the wheel and add
   `python -m mojolearn identity` that runs the lanes on the local box and diffs against
   the shipped columns. Freeze as 0.8.6 through `docs/RELEASE_CHECKLIST.md`; a GPU-box
   build per vendor and a byte compare of the vendor-neutral host binding is the check the
   CPU gates cannot make.
7. Validate 97's `docs/VERIFY_EXTERNALLY.md` recipes on a box that did not build the wheel,
   and run the per-vendor container recipe once on each vendor. Record the KPSS three-vendor
   card and add the tsa row to `IDENTITY_PATHS.md`.

Rules that bind every step: stage every dataset from R2 (`tools/dataset_store.sh stage`);
box order Hot Aisle, DigitalOcean, RunPod, `tools/pick_box.sh`; never cancel an owed run;
`git rev-parse --abbrev-ref HEAD` before every commit, never `git add -A`, commit with
`-o`; every identity JSON carries a commit witness; a host build is pinned to column cpu
and never embeds the build box's GPU name; sabotage must FAIL before a gate counts; report
INERT when bits do not move, never "unchanged"; a grep that returns 0 is not a pass until
you have seen it fail on the unfixed side; merge and push when green. Update
`docs/lanes/TEMP_claim_surface_plan_2026-09-14.md` section 2's owner column as items close,
and delete the file when the table is empty.

---
