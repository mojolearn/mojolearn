# SPDX-License-Identifier: Apache-2.0
"""THE META MEMBER KIND (lane mf-meta3, 2026-11-23; core/member_abi.mojo,
MEMBER-ABI-V2.md): VotingClassifier, VotingRegressor, StackingClassifier and
StackingRegressor as MEMBERS of another meta-estimator, a search, a
calibration or an imputer, through the one member ABI. Family
`MEMBER_FAMILY_ENSEMBLE`, published by both x_trees images
(bindings/_mojolearn_x_trees.mojo, bindings/_mojolearn_x_trees_host.mojo)
beside their glue table. No Python: no Python object parameter, field, local
or import.

A META STATE OWNS A NESTED SET. Its nested members are ordinary native
members (any family, this one included since lane mf-meta3c: a nested META
state is one more member, admitted by the boundary's work-list), each
with its own typed `MemberPlan`. The plans cannot travel as Float64 words,
so the boundary's UNPACK (providers/python/member_ensemble_boundary_door.mojo)
MINTS them once per admission into the process-wide META PLAN STORE
(`meta_plan_mint`) and puts the handle in the META plan's words; the plan
entry is released by the capsule the admission keeps in the owner's pins
(bindings/meta_plan_capsule_pack.mojo), so it lives exactly as long as any
copy of the META plan can still make a state. `MEMBER_OP_NEW` COPIES the
nested plans into the new state: a state never reads the plan store again.

MEMBER_OP_NEW parameters (Float64 words, count `META_PARAM_COUNT`):
[0] kind (`META_VOTE_CLASSIFIER`, `META_VOTE_REGRESSOR`,
`META_STACK_CLASSIFIER`, `META_STACK_REGRESSOR`), [1] the plan store handle,
[2] soft voting (vote) / passthrough (stack), 1.0 or 0.0, [3] the stacking
cv fold count (0 for a vote), [4] the worker bound (the estimator's
`n_jobs`, 0: none), [5] META_PARAM_SERVING (a serving search plan: 1.0; else 0).

The plan entry holds, in member order: the nested plans (a stack's final
estimator LAST), the vote weights (one per voting member), the stack kinds
(`STACK_PREDICT` / `_PROBA` / `_DECISION`, one per stacked member) and each
nested KIND's capability mask (asked once by the admission).

FIT (`MEMBER_OP_FIT`, x float32 dense host rows x features, y one column:
int32 class codes 0 .. classes-1 for a classifier kind, float32 targets for
a regressor kind, optional float32 weights handed to every nested fit):
  * vote: the voting members as ONE native member set over the META's X
    (core/member_set_run.mojo `member_set_fit`: the GIL-free kinds on host
    workers), exactly the set VotingClassifier / VotingRegressor.fit makes
    (bindings/trees_public_meta_door.mojo `_voting_fit`);
  * stack (`_stacking_fit`, same steps, same order): the members on every
    row (one set); the width probe over the first rows
    (xtrees/trees_meta_vote_run.mojo `stack_place_run`, no meta block); the
    cv folds of `xt_device_folds` (stratified over the codes for a
    classifier, plain for a regressor) as waves of fold units (one set per
    wave, the train rows as each unit's row draw), each fold's test outputs
    placed by `stack_place_run`; passthrough X scattered after the members'
    columns; the meta features rounded to float32 and the final estimator
    fitted on them (a set of one). The fold states are released after their
    test outputs (scikit-learn keeps no fold estimator).
A fold whose train rows lack a class fails the fold set
(`MEMBER_SET_LACKING`, raised before any fit) and so this fit: the caller
takes its own path. A stratified fold of `xt_device_folds` never lacks one
(each class has at least `cv` rows, or the folds refuse).

PREDICT / PREDICT_PROBA / DECISION: the meta-estimator's own predict RUNs
(xtrees/trees_meta_vote_run.mojo, one body per operation): `vote_codes_run`
(int32 codes), `vote_soft_run` (float64 probabilities, soft only),
`vote_regress_run` (float64 predictions); stacking `stack_place_run` over
the members, passthrough, float32 meta features, then ONE call of the final
member, whose output (its own width and element type) is the META's. A
nested member that refuses raises its own words.

OUTPUTS: predict one column (int32 codes for the voting classifier,
float64 for the voting regressor, the final member's type for a stack);
predict_proba `classes` columns (float64 for a vote; the final's for a
stack); decision (stack only) the final's. Without a state the plan entry's
final plan is asked (`member_outputs` / `member_output_dtype`).

FITTED-STATE BLOCKS: `META_BLOCK_NESTED` (64), int64 1 x members: the nested
state handles in member order (the final last), owned by this state;
`META_BLOCK_WIDTHS` (65), int64 1 x stacked members: each member's meta
feature columns; `MEMBER_BLOCK_SCALARS` (4), float64 [features, classes]
(export only). EXPORT hands out views; IMPORT takes the handles over (the
state frees them) and `MEMBER_IMPORT_DONE` reads `call.classes`.
`MEMBER_EXPORT_OWN` is forwarded to every nested state (`member_own`).
The boundary's PACK reads the nested plans of a state with
`meta_state_plans` and its nested fit notices with `meta_state_notices`
(direct reads of this process-wide store: the ABI has no word for a plan).

KIND-EXACT BITS: `MEMBER_GIL_FREE` when every nested kind has it,
`MEMBER_GIL_DOOR` when every nested kind has one of the two (the nested
turns then run on the calling thread, as `member_set_fit` runs them);
`MEMBER_CAN_SAMPLE_WEIGHT` when every nested kind has it;
`MEMBER_BORROWS_FIT` always (the nested states read the META's X by view).
No `MEMBER_CAN_TEST_ROWS`: the caller proves X finite.

THE LIFETIME RULE: a state owns its nested states, the stack's float32 meta
features (the final member's fit buffer) and the folds' row numbers until
its FREE. Parallel: the nested sets run their GIL-free members on host
workers; the fold waves hold at most one gathered train block per worker.

THE UNIT KINDS (lane mf-meta3b, 2026-11-23): OneVsRestClassifier
(`META_OVR`), CalibratedClassifierCV (`META_CALIBRATED`), BaggingClassifier
and BaggingRegressor (`META_BAG_CLASSIFIER` / `_REGRESSOR`). Their plan
entry holds ONE nested plan, the base estimator; a fit makes as many units
of it as the door's fit makes clones (`_expand`), each unit its own nested
state, fitted as one member set. The kind's settings are the entry's
`words` (`META_OVR_*`, `META_CAL_*`, `META_BAG_*`), read by the boundary's
UNPACK as the door's fit reads them:
  * OneVsRest (`_ovr_fit`): k == 2 one unit on the codes, else k units on
    the class indicators (`xt_indicator_codes`), two classes each. predict:
    k == 2 the unit's codes, else `ovr_assemble_run` over each unit's
    decision column (the base has decision_function) or the probability
    column of code 1; predict_proba: k == 2 `xt_binary_proba` of the unit's
    code-1 probabilities, else `ovr_assemble_run` (finish 1).
  * CalibratedClassifierCV (`_cal_native`, an int cv: the stratified folds
    of `xt_device_folds`): one unit per fold over its train rows (plus one
    over every row for ensemble=False), the whole fit as
    xtrees/trees_calibrated_fit_run.mojo `calibrated_fit_run`; the kept
    pairs (each fold's unit, or only the full unit for ensemble=False) with
    their held classes and calibrators. predict_proba / predict:
    `calibrated_proba_run` over the pairs, by the door's pair words
    (`calibrated_pair_words`: the code-1 column for two classes, THE
    PLACEMENT RULE `cal_fold_place` for a pair that lacks a class), then the
    row argmax for predict.
  * Bagging (`_bagging_fit_bags`, the per-member route): unit k draws its
    rows (`xt_sample_indices`, stream 2k) and features (stream 2k + 1) and
    holds the seed `_sub_seed(seed, k)` in its seed words
    (xtrees/trees_ada_fit_run.mojo `ADA_MEMBER_SEED_WORD`, the RF and ET
    layouts' `random_state` slot). A classifier unit whose draw lacks a
    class is fitted on the classes it drew, re-coded (the door's member fit
    on its drawn codes), its probability columns placed at their ensemble
    codes. oob_score: each unit's out-of-bag rows (`xt_unseen_rows`) and
    `bag_oob_run`, then `bag_oob_classifier_finish` / `bag_oob_r2_run`.
    predict / predict_proba: `bag_predict_run` over each unit's feature
    subset of X, then the row argmax for a classifier's predict; X staged
    ONCE in the base's forest image when two or more units read it whole on
    a device route (`_stage_units`, lane mf-meta3c: the door's
    `_staged_input`, no upload per unit).
    THE BATCHED ROUTE (lane mf-meta3c; `_bagging_batched_ok` /
    `_bagging_fit_batched`, a measured winner kept as it is): a
    configuration whose fit may take it is admitted with the forest the
    door builds from the decision-tree base as the entry's SECOND plan
    (`META_BAG_BATCHED`). Each fit decides as the door decides (no weights,
    every feature, bootstrap or every row) and then fits that ONE forest
    (`_fit_bag_batched`: max_samples n_rows / n when bootstrap, the image's
    bin count for a regressor base left at n_bins=None); predict /
    predict_proba widen its float32 output (`_predict_batched`). PACK
    writes `_batched` = (forest, base); IMPORT takes the one forest state
    and `META_BLOCK_BATCHED`.
A unit kind keeps each kept unit's draws and re-coded targets until FREE,
and every unit fitted on a drawn block owns its rows (`member_own`) before
the set's gathered blocks go.

THE ADABOOST KIND (lane mf-ada4, 2026-11-23): AdaBoostClassifier (SAMME,
`META_ADA_CLASSIFIER`) and AdaBoostRegressor (AdaBoost.R2,
`META_ADA_REGRESSOR`), a unit kind whose units are the kept rounds. The plan
entry holds ONE nested plan, the base's forest member (Tree, Forest or
ExtraTrees family; admitted by the boundary from the base's first round
clone, with the caller's target checks, as the door's `_ada_member_admit`),
and the settings words `META_ADA_*`. FIT (`_fit_ada`) is the door's
`ada_fit` RUN whole (xtrees/trees_ada_fit_run.mojo `ada_fit_run`), its
configuration decoded from the base plan's words (`_ada_config`, the words
the forest member decodes for its own fit), from the uniform starting
weights of sample_weight=None:
  * the SESSION ROUNDS (`META_ADA_NATIVE`: a best-splitter DecisionTree
    base, squared error for the regressor; the door's `_ada_native`): the
    data session opened on the base's forest entry over the classifier's
    column-major X or the regressor's X (THE STATE OWNS IT from the open: it
    is closed when the rounds are imported, by FREE and by the next fit),
    the device rounds (XF_CAP_ADA) or the host rounds; each kept round's
    export imported into a NEW state of the base's tree member
    (`_ada_import_round`), its export handle released;
  * the MEMBER ROUNDS (every other tree base; the door's `_ada_members`):
    each round a new state of the base's member, kept as it is; X staged
    once in its image on a device route.
The rounds' fit notices travel to the PACK (nested member 0). Sample
weights (lane mf-split4b): the caller's float32 weights are the words the
door's `_normalized_weights` reads (`admit_weights` makes float32 words),
normalized here as there (`x_trees_exact_sum` into float64, its refusals);
the kind carries MEMBER_CAN_SAMPLE_WEIGHT. PREDICT / PREDICT_PROBA / DECISION: each round's member
predict (in parallel over the rounds on the host column; X staged once on a
device route), then `ada_classifier_run` (ADA_CODES / ADA_PROBA /
ADA_DECISION) or `ada_regressor_run` (float32, the door's dtype), the
door's predict RUNs. EXPORT: `META_BLOCK_NESTED` (the round states) and
`META_BLOCK_ADA` (the weights and errors); IMPORT the same two.

THE RFE KIND (lane mf-meta5, 2026-11-23): RFE (`META_RFE_CLASSIFIER` /
`META_RFE_REGRESSOR`, one mark (39, 14); the admission picks the kind from its
estimator). The plan entry holds ONE nested plan, the estimator's leaf member
with the importance word the rounds read (providers/python/member_rfe_boundary_door.mojo
`rfe_base_admit`, the admission RFE's own door runs), its settings words
`META_RFE_*`, the x_prep image of the rounds' programs and the words a first
round raises (`meta_plan_rfe`). FIT (`_fit_rfe`) is RFE's RUN whole
(bindings/prep_select_run.mojo `rfe_fit_run`, THE ONE BODY the door runs too):
the rounds' member fits and programs, then the final estimator, the state's one
nested member (it owns its rows); sample weights ride every fit (RFE.fit's
`sample_weight`). PREDICT / PREDICT_PROBA / DECISION / SCORE: the kept columns
of X gathered, then the final estimator's own call; TRANSFORM writes the
gathered block. EXPORT `META_BLOCK_NESTED` (the final estimator) and
`META_BLOCK_RFE` (support, ranking, kept columns); IMPORT the same two. A
DOTTED importance_getter (lane mf-del5b): the one nested plan is a META
estimator's (a search, an RFE, a stack: admitted by the boundary's work-list),
`META_RFE_BLOCK` is `META_BLOCK_FORWARD` and the entry carries the forwarded
path's hops (`meta_plan_rfe`), which `_fit_rfe` hands to the rounds' plan.

STACKING, lane mf-meta3b: a user's cv INDEX LIST is read by the UNPACK into
the entry's typed folds (`fold_train` / `fold_test`, int32 row numbers) and
checked against the fit's rows as the door's `_splits` checks them; a fold
whose train rows lack a class is fitted on its own classes, re-coded, and
placed by `stack_place_run`'s lacking-class rule. A classifier stacked with
stack_method='predict' writes its predicted codes as float32 meta features
(`_label_block`: the label scikit-learn's encoded fit predicts).
"""
from std.math import isnan
from std.memory import bitcast, unsafe_memcpy
from std.sys.info import size_of
from std.sys.compile import is_defined
from bindings.native_unit_os_threads_run import native_unit_cores
from core.host_parallel import host_parallelize
from checks.numerics import GLOBAL_NUMERIC_MODE
from core.device_route import DEVICE_CPU
from core.entry_table import EntryRef, entry_ref_unresolved, entry_ref_gil_free
from core.forest_calls import (
    xf_stage_input, xf_unstage_input, xf_dt_default_bins, xf_caps, xf_session_open, xf_session_close,
    xf_export_sizes, xf_export_into, xf_export_release,
)
from core.forest_fit_request import XF_CAP_SESSION, XF_CAP_ADA
from bindings.arena_prog_run import MetricRoute
from bindings.prep_classify_prog_run import ClsRunPlan
from bindings.prep_select_run import (
    RfeFitPlan, RfeFitResult, rfe_fit_run, rfe_select_count, rfe_step_count, rfe_final_plan,
)
from core.msel_search_run import MselSearchPlan, MselSearchResult, msel_search_run
from core.msel_scorer_run import MselScorerOptions, msel_scorer_group_read
from core.member_abi import (
    MemberPlan, MemberCall, MemberMatrix, member_call, member_call_of, member_free, member_own, member_new,
    member_outputs, member_output_dtype, member_outputs_of,
    MEMBER_F32, MEMBER_F64, MEMBER_I32, MEMBER_I64, MEMBER_DTYPE_UNSAID,
    MEMBER_OP_CAPABILITIES, MEMBER_OP_NEW, MEMBER_OP_FREE, MEMBER_OP_FIT, MEMBER_OP_PREDICT,
    MEMBER_OP_PREDICT_PROBA, MEMBER_OP_DECISION, MEMBER_OP_OUTPUTS, MEMBER_OP_EXPORT, MEMBER_OP_IMPORT,
    MEMBER_CAN_FIT, MEMBER_CAN_PREDICT, MEMBER_CAN_PREDICT_PROBA, MEMBER_CAN_DECISION, MEMBER_CAN_EXPORT,
    MEMBER_CAN_IMPORT, MEMBER_CAN_CLASSIFIER, MEMBER_CAN_SAMPLE_WEIGHT, MEMBER_CAN_TEST_ROWS,
    MEMBER_BORROWS_FIT, MEMBER_GIL_FREE, MEMBER_GIL_DOOR, MEMBER_CAN_SCORE, MEMBER_CAN_MULTI_OUTPUT, MEMBER_OP_SCORE,
    MEMBER_BLOCK_FAMILY, MEMBER_BLOCK_SCALARS, MEMBER_BLOCK_BY_INDEX, MEMBER_EXPORT_OWN, MEMBER_IMPORT_DONE,
    MEMBER_CAN_SCOPE, MEMBER_SCOPE_NONE, MEMBER_CAN_TRANSFORM, MEMBER_OP_TRANSFORM,
    MEMBER_READS_Y_GIVEN, MEMBER_HOME_HOST,
    member_publish, member_call_from_spans, member_unsupported,
)
from core.member_families import (
    MEMBER_FAMILY_ENSEMBLE, MEMBER_FAMILY_GP, MEMBER_FAMILY_TREE, MEMBER_FAMILY_FOREST, MEMBER_FAMILY_EXTRA_TREES,
    MEMBER_FAMILY_XNEIGHBORS,
)
from core.member_notices import member_notices_take
from core.array_helpers_impl import class_ratio_f64
from core.class_weight_plan import ClassWeightInputs, class_weight_rows_plan, class_weight_dict_refused
from core.label_weights_native_api import (
    LabelWeightsDriver, LabelWeightsRequest, LABEL_WEIGHTS_COUNTS, LABEL_WEIGHTS_ROWS,
)
from xtrees.label_weights_native import label_weights_native_v1
from core.member_notices import member_notice_add, MEMBER_NOTICE_SPLIT_LEAST
from bindings.msel_split_spec_run import (
    MselSplitSpec, MselGroupCodes, MselSides, MSEL_SPEC_NONE, MSEL_SPEC_LOO, msel_spec_sides, msel_sides_i32,
    msel_group_codes_run, msel_spec_has_n_splits, msel_spec_sides_targets, msel_spec_stratified,
)
from core.member_set_run import (
    MemberSetRun, member_set_fit, member_set_run_release, member_set_first_failure,
    MEMBER_SET_WORKER, MEMBER_SET_TURN_RELEASED, MEMBER_SET_TURN_HELD,
)
from core.native_descriptor_abi import NativeDescriptorError
from core.native_descriptor_call import native_descriptor_report
from core.shared_slot import SharedSlot, shared_slot_lock, shared_slot_unlock
from core.trees_entry_table import TreesEntryKey
from xtrees.glue_calls import (
    xt_ref, xt_device_folds, xt_gather_f32, xt_scatter, xt_indicator_codes, xt_binary_proba, xt_argmax_rows,
    xt_sample_indices, xt_unseen_rows, xt_transpose_f32, xt_exact_sum,
)
from xtrees.trees_ada_fit_run import (
    ADA_MEMBER_SEED_WORD, AdaFitPlan, AdaFitResult, ada_fit_run, ada_release_handles, ada_release_members,
)
from xtrees.trees_forest_run import forest_export_counts
from core.forest_member_run import FM_HEAD, FM_WORD_SLOTS
from bindings.forest_config_run import (
    FOREST_RF_SLOT_BINS, FOREST_RF_SLOT_MAX_SAMPLES, ForestConfig, forest_target_need,
)
from bindings.trees_public_inputs_run import TREES_NONFINITE
from xtrees.trees_boost_run import _sub_seed, boost_count_check, boost_weight_flags_check
from xtrees.trees_calibrated_fit_run import (
    CalFitPlan, CalFitOut, calibrated_fit_run, cal_fold_place, cal_cv_check, CAL_ALL_ANSWERED,
)
from xtrees.trees_meta_run import meta_binary_column_check
from xtrees.trees_meta_vote_run import (
    VoteMember, VoteRows, vote_codes_run, vote_soft_run, vote_regress_run, stack_place_run, member_f32_block,
    ovr_assemble_run, bag_predict_run, bag_oob_run, bag_oob_classifier_finish, bag_oob_r2_run,
    calibrated_proba_run, CalPair, CalFit, ada_classifier_run, ada_regressor_run, ADA_DECISION, ADA_PROBA, ADA_CODES,
    VOTE_ALL_ANSWERED, STACK_PREDICT, STACK_PROBA, STACK_DECISION,
)

comptime META_VOTE_CLASSIFIER = 0
comptime META_VOTE_REGRESSOR = 1
comptime META_STACK_CLASSIFIER = 2
comptime META_STACK_REGRESSOR = 3
comptime META_OVR = 4
comptime META_CALIBRATED = 5
comptime META_BAG_CLASSIFIER = 6
comptime META_BAG_REGRESSOR = 7
comptime META_ADA_CLASSIFIER = 8
comptime META_ADA_REGRESSOR = 9
#: Lane mf-search4: GridSearchCV / RandomizedSearchCV (THE SEARCH KIND; both
#: classes carry the mark (39, 10), the admission writes the regressor word
#: for a regressor base) and MultiOutputClassifier / MultiOutputRegressor
#: (THE MULTI-OUTPUT KIND).
comptime META_SEARCH_CLASSIFIER = 10
comptime META_SEARCH_REGRESSOR = 11
comptime META_MULTI_CLASSIFIER = 12
comptime META_MULTI_REGRESSOR = 13
# Lane mf-meta5: THE RFE KIND (RFE over a classifier / a regressor base; one mark (39, 14), the admission
# picks the kind from its base).
comptime META_RFE_CLASSIFIER = 14
comptime META_RFE_REGRESSOR = 15
comptime META_KIND_LAST = META_RFE_REGRESSOR
comptime META_PARAM_COUNT = 6
#: NEW parameter [5] (lane mf-door7, 2026-11-23): 1.0 marks a SERVING search plan (the admission of a fitted
#: search for a caller that only predicts: providers/python/member_ensemble_boundary_door.mojo
#: `_admit_search_serving`), whose one candidate is the fitted best (imported as candidate 0); 0.0 otherwise.
#: NEW does not read it.
comptime META_PARAM_SERVING = 5
comptime META_BLOCK_NESTED = MEMBER_BLOCK_FAMILY
comptime META_BLOCK_WIDTHS = MEMBER_BLOCK_FAMILY + 1
#: Calibrated: the pairs' words (`meta_cal_words`), float64 1 x count.
comptime META_BLOCK_CAL = MEMBER_BLOCK_FAMILY + 2
#: Bagging: the units' draws (`meta_bag_words`), int32 1 x count.
comptime META_BLOCK_DRAWS = MEMBER_BLOCK_FAMILY + 3
#: Bagging oob_score: the out-of-bag outputs, float64 1 x (n * width + 1), the score last.
comptime META_BLOCK_OOB = MEMBER_BLOCK_FAMILY + 4

#: The entry's settings words of each unit kind.
comptime META_OVR_DECISION = 0          # the base has decision_function (1.0)
comptime META_OVR_WORDS = 1
#: CalibratedClassifierCV(ensemble=False) over a LabelPropagation / LabelSpreading base whose y holds the
#: unlabeled -1 and one class (two labels): scikit-learn's own fit fails (lane mf-ens6). calibration.py
#: `_fit_calibrator`: `Y = label_binarize(y, classes=classes)` makes ONE column for two classes, the base's
#: classes_ exclude -1, so `pos_class_indices` is [1] and `calibrator.fit(this_pred, Y[:, class_idx], ...)`
#: indexes column 1 of a one-column array: NumPy's IndexError, in its words. Raised natively by the meta door
#: (`_cal_native`) and THE CALIBRATED KIND's admission (`_admit_cal_base`).
comptime META_CAL_BINARY_MARKER_WORDS = "IndexError: index 1 is out of bounds for axis 1 with size 1"
comptime META_CAL_SIGMOID = 0           # method='sigmoid' (1.0), else isotonic
comptime META_CAL_ENSEMBLE = 1          # ensemble=True (1.0)
comptime META_CAL_DECISION = 2          # the base has decision_function (1.0)
comptime META_CAL_INT_CV = 3            # cv was an exact int (1.0; scikit-learn's fold-count check reads it, lane mf-del5)
comptime META_CAL_WORDS = 4
comptime META_BAG_COUNT = 0             # n_estimators
comptime META_BAG_SEED = 1              # `_seed(random_state)`
comptime META_BAG_ROWS_INT = 2          # max_samples is an int (1.0) or a fraction (0.0)
comptime META_BAG_ROWS = 3              # its value
comptime META_BAG_COLS_INT = 4          # max_features is an int (1.0) or a fraction (0.0)
comptime META_BAG_COLS = 5              # its value
comptime META_BAG_BOOTSTRAP = 6
comptime META_BAG_BOOTSTRAP_FEATURES = 7
comptime META_BAG_OOB = 8               # oob_score
# Lane mf-meta3c: 1.0 when the entry's SECOND plan is the batched forest
# (THE BATCHED ROUTE in the module note), and 1.0 when its slot-7 bin count
# is the image's (a DecisionTreeRegressor base left at n_bins=None).
comptime META_BAG_BATCHED = 9
comptime META_BAG_IMAGE_BINS = 10
comptime META_BAG_WORDS = 11
#: the import block that marks a batched Bagging fit (one word, any value),
#: sent after `META_BLOCK_NESTED` (its one state is the forest's)
comptime META_BLOCK_BATCHED = MEMBER_BLOCK_FAMILY + 5
#: AdaBoost (lane mf-ada4): float64 [weights count, estimator_weights_..., errors count,
#: estimator_errors_...], the kept rounds' words in round order.
comptime META_BLOCK_ADA = MEMBER_BLOCK_FAMILY + 6
#: AdaBoost's settings words (THE ADABOOST KIND in the module note), read by the boundary's UNPACK as the
#: door's `ada_fit` reads them.
comptime META_ADA_COUNT = 0             # n_estimators
comptime META_ADA_SEED = 1              # `_seed(random_state)`
comptime META_ADA_RATE = 2              # learning_rate
comptime META_ADA_LOSS = 3              # the regressor's loss code (`trees_ada_losses`), 0 for a classifier
comptime META_ADA_NATIVE = 4            # 1.0: the data session rounds (`_ada_native_base`), else the member rounds
comptime META_ADA_SHARE = 5             # 1.0: the session shares its quantile tables (`_member_session`'s share)
comptime META_ADA_SQUARED = 6           # the squared_error criterion code (`plan.squared`)
comptime META_ADA_TARGET = 7            # the regressor's target check codes (`ForestConfig.target_code`,
comptime META_ADA_POISSON = 8           # `.poisson`, `.gamma`, `.inverse_gaussian`; -2: none read)
comptime META_ADA_GAMMA = 9
comptime META_ADA_INVERSE_GAUSSIAN = 10
comptime META_ADA_DT_RANDOM = 11         # 1.0: a DecisionTree splitter='random' base, whose rounds fit its inner
                                        # extra-trees forest (lane mf-split4; `ada_fit_run`'s tree_base word 1)
comptime META_ADA_WORDS = 12
#: THE SEARCH KIND's fitted words (float64, export / import): [best index,
#: folds, best score, candidates m, train flag, sampler seed, tables S,
#: refitted (1.0)], then PER SCORER TABLE (S >= 1, in the scorers' order;
#: lane mf-search4b) m x folds test scores (candidate-major), m means, m
#: deviations, m ranks, and with the train flag m x folds train scores and
#: m train means. The best index and score are the refit scorer's.
comptime META_BLOCK_SEARCH = MEMBER_BLOCK_FAMILY + 7
comptime META_SEARCH_HEAD = 8
#: THE SEARCH KIND's settings words: the fixed head, then `META_SEARCH_COUNT`
#: scorer codes (a multi-metric `scoring` in its name order; none: the one
#: `META_SEARCH_SCORING`).
comptime META_SEARCH_SCORING = 0        # core/msel_search_run.mojo SEARCH_SCORE_MEMBER / a scorer code
comptime META_SEARCH_TOLERATE = 1       # a numeric error_score (1.0)
comptime META_SEARCH_ERROR = 2          # its value
comptime META_SEARCH_TRAIN = 3          # return_train_score (1.0)
comptime META_SEARCH_SEED = 4           # the sampler's seed the PACK re-draws the candidates with (-1: a grid)
comptime META_SEARCH_REFIT = 5          # 1.0: refit the best (refit=True or a scorer name); 0.0: refit=False
comptime META_SEARCH_REFIT_SCORER = 6   # the index among the codes whose table chooses the best (refit=<name>)
comptime META_SEARCH_COUNT = 7          # the number of scorer codes that follow (0: one scorer)
comptime META_SEARCH_WORDS = 8          # the head; then one scorer word group per scorer
#: THE SEARCH KIND's scorer word groups (lane mf-tail5b, one layout for every
#: scorer option; core/msel_scorer_run.mojo SCORER_GROUP_WORDS): one group
#: per scorer (the one `META_SEARCH_SCORING`, or each of `count`), each its
#: code, its positive word (the class code of its pos_label among the fit's
#: labels, -1: none), its make_scorer options and its `labels` codes, read
#: by `msel_scorer_group_read` only. Was: `count` codes, then one positive
#: word per scorer (lane mf-search4b item 3).
#: THE CV TOOLS' words (lane mf-msel5; core/msel_search_run.mojo module
#: note): an optional tail of `META_SEARCH_TOOL_WORDS` after the scorer
#: groups, [keep the units' states (1.0), the test-side response op (-1.0:
#: score), the folds' rows (1.0), and (lane mf-msel5b) the caller's result
#: buffer the responses are written into: its address, its words per row,
#: its element type (MEMBER_I64 / MEMBER_F64), predict_log_proba (1.0)];
#: absent for a public search.
comptime META_SEARCH_TOOL_WORDS = 7
#: THE CV TOOLS' kept unit states (int64 1 x units, export only; the
#: search's own block numbers end at META_BLOCK_SEARCH, this one is set
#: apart from the other kinds' blocks).
comptime META_BLOCK_FOLDS = MEMBER_BLOCK_FAMILY + 16
#: THE MULTI-OUTPUT KIND's settings words: [0] reserved.
comptime META_MULTI_WORDS = 1
#: THE RFE KIND's settings words (lane mf-meta5), read by the boundary's UNPACK as the door reads RFE's
#: fields (bindings/prep_select_run.mojo `rfe_select_count` / `rfe_step_count` resolve them per fit over its
#: X's columns): n_features_to_select's kind (RFE_SELECT_*), fraction and count; step as a fraction (1.0)
#: or a count, and its value and count; the importance block and whether it is squared
#: (`rfe_member_source`); whether the rounds' plan has the importance word set (`rfe_final_plan` clears
#: it); the x_prep image's `work` switch; whether the image keeps a pinned X resident in a caller's scope
#: (device store and range runner: MEMBER_CAN_SCOPE).
comptime META_RFE_SELECT_KIND = 0
comptime META_RFE_SELECT_FRACTION = 1
comptime META_RFE_SELECT_COUNT = 2
comptime META_RFE_STEP_FRACTION = 3
comptime META_RFE_STEP_VALUE = 4
comptime META_RFE_STEP_COUNT = 5
comptime META_RFE_BLOCK = 6
comptime META_RFE_SQUARED = 7
comptime META_RFE_IMPORTANCE_WORD = 8
comptime META_RFE_WORK_ON = 9
comptime META_RFE_SCOPE = 10
comptime META_RFE_WORDS = 11
#: THE RFE KIND's fitted words (float64, export / import): [d, k], the d support words (1.0 kept), the d
#: ranking words, the k kept column ids.
comptime META_BLOCK_RFE = MEMBER_BLOCK_FAMILY + 8
#: THE FORWARDED EXPORT (lane mf-meta5b; export only): a nested member's block along an attribute path, the
#: path a dotted importance_getter names (RFE's 'best_estimator_.coef_', 'estimator_.coef_',
#: 'final_estimator_.coef_'). `call.params` / `call.param_count`: float64 words [index, ..., block], each
#: index a nested member of the state at that step (-1: the last: a stack's final estimator), then the leaf's
#: block; the view is the leaf state's own (it lives as long as this state).
comptime META_BLOCK_FORWARD = MEMBER_BLOCK_FAMILY + 9
#: THE CLASS-WEIGHT TAIL of a tree base (lane mf-split4): appended after a
#: Bagging's META_BAG_WORDS (its batched forest's or per-member tree's
#: class_weight) and an AdaBoost's META_ADA_WORDS (its base's): [mode, K,
#: all-one, K weights] and, for a dict, K named flags (lane mf-split4b).
#: Mode META_CW_NONE, META_CW_BALANCED (the FIT computes the ratios from its
#: codes) or META_CW_DICT (the dict's weights over the codes' labels, read at
#: UNPACK with the forest door's admission; K = -1: no labels were known, the
#: FIT refuses).
comptime META_CW_MODE = 0
comptime META_CW_K = 1
comptime META_CW_ALL_ONE = 2
comptime META_CW_HEAD = 3
comptime META_CW_NONE = 0
comptime META_CW_BALANCED = 1
comptime META_CW_DICT = 2
#: THE SPLITTER CV of a stack, a calibration or a search (lanes mf-split4,
#: mf-split5): one of OUR splitters, read at UNPACK as typed words
#: (bindings/msel_split_spec_run.mojo `MselSplitSpec`, every native splitter),
#: its folds computed inside each FIT over that fit's own rows by the public
#: split door's own plans and fill (`msel_spec_sides`). Before (mf-split4)
#: only KFold and StratifiedKFold had words (META_SPLIT_*, removed).

comptime _CALIB_FAST_HOST_FOLDS = is_defined["MOJOLEARN_CALIB_FAST_HOST_FOLDS"]()

comptime _CHUNK = 64
comptime _INDEX_SPAN = 16777216
comptime _F32P = MutPointer[Float32, MutUntrackedOrigin]
comptime _F64P = MutPointer[Float64, MutUntrackedOrigin]
comptime _I32P = MutPointer[Int32, MutUntrackedOrigin]
comptime _I64P = MutPointer[Int64, MutUntrackedOrigin]
comptime _CallPtr = MutPointer[MemberCall, MutUntrackedOrigin]


def _stack_kind_code(kind: Int) -> Bool:
    return kind == META_STACK_CLASSIFIER or kind == META_STACK_REGRESSOR


def _classifier_kind(kind: Int) -> Bool:
    return (kind == META_VOTE_CLASSIFIER or kind == META_STACK_CLASSIFIER or kind == META_OVR
            or kind == META_CALIBRATED or kind == META_BAG_CLASSIFIER or kind == META_ADA_CLASSIFIER
            or kind == META_SEARCH_CLASSIFIER or kind == META_MULTI_CLASSIFIER or kind == META_RFE_CLASSIFIER)


def _search_kind(kind: Int) -> Bool:
    return kind == META_SEARCH_CLASSIFIER or kind == META_SEARCH_REGRESSOR


def _multi_kind(kind: Int) -> Bool:
    return kind == META_MULTI_CLASSIFIER or kind == META_MULTI_REGRESSOR


def _ada_kind(kind: Int) -> Bool:
    return kind == META_ADA_CLASSIFIER or kind == META_ADA_REGRESSOR


def _rfe_kind(kind: Int) -> Bool:
    return kind == META_RFE_CLASSIFIER or kind == META_RFE_REGRESSOR


def _unit_kind(kind: Int) -> Bool:
    """One nested base estimator, as many units as the fit makes (module note, THE UNIT KINDS; AdaBoost: one
    unit per kept round)."""
    return (kind == META_OVR or kind == META_CALIBRATED or kind == META_BAG_CLASSIFIER or kind == META_BAG_REGRESSOR
            or _ada_kind(kind) or _multi_kind(kind))


def _bag_kind(kind: Int) -> Bool:
    return kind == META_BAG_CLASSIFIER or kind == META_BAG_REGRESSOR


# ------------------------------------------------------------ the plan store


struct MetaPlanSlot(Copyable, Movable, Defaultable):
    var generation: Int
    var live: Bool
    var plans: List[MemberPlan]
    var weights: List[Float64]
    var stack_kinds: List[Int]
    var masks: List[Int]
    # Lane mf-meta3b: the kind's settings words and a stack's user cv folds.
    var words: List[Float64]
    var fold_train: List[List[Int32]]
    var fold_test: List[List[Int32]]
    # Lane mf-search4: THE SEARCH KIND's metric program route (scoring
    # 'accuracy'), resolved by the admission; its connection is pinned there.
    var metric: MetricRoute
    # Lanes mf-split4, mf-split5: THE SPLITTER CV (kind MSEL_SPEC_NONE: an
    # int cv or a user's index list), its entries pinned by the admission.
    var split: MselSplitSpec
    # Lane mf-cluster5: THE SEARCH KIND's cluster label-pair entry and its
    # labels' reprs (`meta_plan_search_labels`).
    var cluster: EntryRef
    var label_words: List[String]
    # Lane mf-meta5: THE RFE KIND's x_prep image (the rounds' programs, `meta_plan_rfe`; the admission
    # pins its selection) and the words its first round raises (empty: none).
    var rfe_run: ClsRunPlan
    var refusal: String
    # Lane mf-refuse5, THE DOOR'S REFUSAL (`meta_plan_refuse`): a non-RFE kind's `refusal` is the meta door's
    # words for a constructor value it refuses; `refusal_late`: raised after the nested fits (the door's check
    # follows them), else before anything (the plan may then hold no members).
    var refusal_late: Bool
    # Lane mf-del5b: THE FORWARDED EXPORT's hops of a dotted importance_getter (empty: none), copied into
    # every state NEW makes from it (`_fit_rfe` hands them to the rounds' plan).
    var rfe_hops: List[Float64]

    def __init__(out self):
        self.generation = 0
        self.live = False
        self.plans = List[MemberPlan]()
        self.weights = List[Float64]()
        self.stack_kinds = List[Int]()
        self.masks = List[Int]()
        self.words = List[Float64]()
        self.fold_train = List[List[Int32]]()
        self.fold_test = List[List[Int32]]()
        self.metric = MetricRoute(entry_ref_unresolved().call, False, False)
        self.split = MselSplitSpec()
        self.cluster = entry_ref_unresolved()
        self.label_words = List[String]()
        self.rfe_run = ClsRunPlan(entry_ref_unresolved(), False, False, False, False, False)
        self.refusal = String("")
        self.refusal_late = False
        self.rfe_hops = List[Float64]()


struct MetaPlanStore(Defaultable, Movable):
    var synchronization: List[Int64]
    var chunks: List[List[MetaPlanSlot]]
    var free: List[Int]
    var count: Int

    def __init__(out self):
        self.synchronization = List[Int64](length=2, fill=0)
        self.chunks = List[List[MetaPlanSlot]]()
        self.free = List[Int]()
        self.count = 0


comptime _PLANS = SharedSlot[MetaPlanStore, "MojolearnEnsembleMemberPlanStoreV1", 1]
comptime _PlanSlotPtr = MutPointer[MetaPlanSlot, MutUntrackedOrigin]


def _plan_slot_address(index: Int) raises -> Int:
    var store = _PLANS.get_or_create_ptr()
    return Int(store[].chunks[index // _CHUNK].unsafe_ptr()) + (index % _CHUNK) * size_of[MetaPlanSlot]()


def meta_plan_mint(var plans: List[MemberPlan], var weights: List[Float64], var stack_kinds: List[Int],
                   var masks: List[Int]) raises -> Int:
    """UNPACK (the boundary's admission): keep the nested plans of one META
    plan; the handle goes into the META plan's words. Released by
    `meta_plan_release` (the admission's capsule)."""
    return meta_plan_mint_with(plans^, weights^, stack_kinds^, masks^, List[Float64](), List[List[Int32]](),
                               List[List[Int32]]())


def meta_plan_mint_with(var plans: List[MemberPlan], var weights: List[Float64], var stack_kinds: List[Int],
                        var masks: List[Int], var words: List[Float64], var fold_train: List[List[Int32]],
                        var fold_test: List[List[Int32]]) raises -> Int:
    """`meta_plan_mint` with the kind's settings words (THE UNIT KINDS) and
    a stack's user cv folds (int32 train / test row numbers per fold, in
    the order the user's list gives them; empty: the int cv of the
    parameter words)."""
    var store = _PLANS.get_or_create_ptr()
    var lock = rebind[MutPointer[Int64, MutUntrackedOrigin]](store[].synchronization.unsafe_ptr())
    shared_slot_lock(lock)
    try:
        var index = -1
        if len(store[].free) > 0:
            index = store[].free.pop()
        else:
            if store[].count >= _INDEX_SPAN - 1:
                raise Error("RuntimeError: mojolearn: too many live meta member plans")
            if store[].count == len(store[].chunks) * _CHUNK:
                store[].chunks.append(List[MetaPlanSlot](length=_CHUNK, fill=MetaPlanSlot()))
            index = store[].count
            store[].count += 1
        var slot = _PlanSlotPtr(unsafe_from_address=_plan_slot_address(index))
        slot[].generation += 1
        slot[].live = True
        slot[].plans = plans^
        slot[].weights = weights^
        slot[].stack_kinds = stack_kinds^
        slot[].masks = masks^
        slot[].words = words^
        slot[].fold_train = fold_train^
        slot[].fold_test = fold_test^
        return slot[].generation * _INDEX_SPAN + index + 1
    finally:
        shared_slot_unlock(lock)


def meta_plan_split(handle: Int, spec: MselSplitSpec) raises:
    """UNPACK (a stack's, a calibration's or a search's admission, lanes
    mf-split4, mf-split5): THE SPLITTER CV of a minted plan entry, copied
    into every state NEW makes from it."""
    var store = _PLANS.get_or_create_ptr()
    var address = _plan_lookup(handle, False)
    var lock = rebind[MutPointer[Int64, MutUntrackedOrigin]](store[].synchronization.unsafe_ptr())
    shared_slot_lock(lock)
    try:
        var entry = _PlanSlotPtr(unsafe_from_address=address)
        entry[].split = spec.copy()
    finally:
        shared_slot_unlock(lock)


def meta_plan_metric(handle: Int, route: MetricRoute) raises:
    """UNPACK (THE SEARCH KIND's admission): the metric program route of a
    minted plan entry, copied into every state NEW makes from it."""
    var store = _PLANS.get_or_create_ptr()
    var address = _plan_lookup(handle, False)
    var lock = rebind[MutPointer[Int64, MutUntrackedOrigin]](store[].synchronization.unsafe_ptr())
    shared_slot_lock(lock)
    try:
        _PlanSlotPtr(unsafe_from_address=address)[].metric = route
    finally:
        shared_slot_unlock(lock)


def meta_plan_search_labels(handle: Int, cluster: EntryRef, var label_words: List[String]) raises:
    """UNPACK (THE SEARCH KIND's admission, lane mf-cluster5): the metrics
    image's cluster label-pair entry (core/metrics_cluster_table.mojo,
    resolved once; its image pinned by the admission) and the fit's labels'
    reprs by class code, copied into every state NEW makes from it."""
    var store = _PLANS.get_or_create_ptr()
    var address = _plan_lookup(handle, False)
    var lock = rebind[MutPointer[Int64, MutUntrackedOrigin]](store[].synchronization.unsafe_ptr())
    shared_slot_lock(lock)
    try:
        var entry = _PlanSlotPtr(unsafe_from_address=address)
        entry[].cluster = cluster
        entry[].label_words = label_words^
    finally:
        shared_slot_unlock(lock)


def meta_plan_rfe(handle: Int, run: ClsRunPlan, var refusal: String, var hops: List[Float64]) raises:
    """UNPACK (THE RFE KIND's admission, lane mf-meta5): the x_prep image the elimination rounds' programs
    run on (its selection pinned by the admission), the words the first round raises (a base kind that
    exports no importance block; empty: none) and (lane mf-del5b) a dotted importance_getter's hops
    (META_BLOCK_FORWARD's words over the nested META base; empty: a leaf base), copied into every state NEW
    makes from it."""
    var store = _PLANS.get_or_create_ptr()
    var address = _plan_lookup(handle, False)
    var lock = rebind[MutPointer[Int64, MutUntrackedOrigin]](store[].synchronization.unsafe_ptr())
    shared_slot_lock(lock)
    try:
        var entry = _PlanSlotPtr(unsafe_from_address=address)
        entry[].rfe_run = run
        entry[].refusal = refusal^
        entry[].rfe_hops = hops^
    finally:
        shared_slot_unlock(lock)


#: The tree / forest member's importance word (core/forest_member_run.mojo FM_WORD_IMPORTANCES): set, its FIT
#: computes the feature importances its EXPORT MEMBER_BLOCK_FEATURE_IMPORTANCES hands out.
comptime META_LEAF_IMPORTANCE_WORD = 10


def meta_plan_leaf_importances(plan: MemberPlan, hops: List[Float64]) raises -> Bool:
    """UNPACK, once per admission (lane mf-rfe6): a dotted importance_getter that ends in a tree or forest
    leaf's `feature_importances_` ('best_estimator_.feature_importances_', ...). The leaf computes them only
    when its importance word asks (META_LEAF_IMPORTANCE_WORD), and the word lives in the nested plan this
    admission minted, so it is set there, in the plan store, before any state is made: every state NEW makes
    copies it, and THE FORWARDED EXPORT reads the leaf's block after each round's fit. `plan`: the META
    estimator's admitted plan; `hops`: META_BLOCK_FORWARD's words (each step's nested index, -1 the last,
    then the leaf's block). A search step sets it on every candidate (any may be best_estimator_). False: a
    step this store does not hold, or a leaf that is not a tree or forest with the word (not served)."""
    var steps = len(hops) - 1
    if steps < 1 or plan.family != MEMBER_FAMILY_ENSEMBLE or len(plan.params) < 2:
        return False
    var store = _PLANS.get_or_create_ptr()
    var lock = rebind[MutPointer[Int64, MutUntrackedOrigin]](store[].synchronization.unsafe_ptr())
    var kinds: List[Int] = [Int(plan.params[0])]
    var handles: List[Int] = [Int(plan.params[1])]
    for s in range(steps):  # driver-loop: over the attribute path's steps
        var last = s == steps - 1
        var next_kinds = List[Int]()
        var next_handles = List[Int]()
        for h in range(len(handles)):  # driver-loop: over this step's plan entries (a search's candidates)
            var address = _plan_lookup(handles[h], False)
            var served = True
            shared_slot_lock(lock)
            try:
                var entry = _PlanSlotPtr(unsafe_from_address=address)
                var count = len(entry[].plans)
                var lo = Int(hops[s])
                if lo < 0:
                    lo = count - 1
                var hi = lo + 1
                if _search_kind(kinds[h]):
                    lo = 0
                    hi = count
                if lo < 0 or hi > count or lo >= hi:
                    served = False
                else:
                    for u in range(lo, hi):  # driver-loop: over the step's nested plans
                        var family = entry[].plans[u].family
                        if last:
                            if ((family != MEMBER_FAMILY_TREE and family != MEMBER_FAMILY_FOREST)
                                    or len(entry[].plans[u].params) <= META_LEAF_IMPORTANCE_WORD):
                                served = False
                        elif family != MEMBER_FAMILY_ENSEMBLE or len(entry[].plans[u].params) < 2:
                            served = False
                    if served:
                        for u in range(lo, hi):  # driver-loop: over the step's nested plans
                            if last:
                                entry[].plans[u].params[META_LEAF_IMPORTANCE_WORD] = 1.0
                            else:
                                next_kinds.append(Int(entry[].plans[u].params[0]))
                                next_handles.append(Int(entry[].plans[u].params[1]))
            finally:
                shared_slot_unlock(lock)
            if not served:
                return False
        kinds = next_kinds^
        handles = next_handles^
    return True


def meta_plan_refuse(handle: Int, var words: String, late: Bool) raises:
    """UNPACK (lane mf-refuse5, THE DOOR'S REFUSAL; not the RFE kind, whose `refusal` is
    `meta_plan_rfe`'s): the meta door's words for a constructor value it refuses, raised by every
    state's FIT (`_fit`) where the door raises them, so the admission serves the input and no caller
    falls back to the door to raise them. `late`: after the nested fits (the plan is whole); else first,
    before anything is read (the plan may hold no members: NEW then checks nothing and CAPABILITIES
    answers `_refused_mask`). Words come from the door's own check (bindings/
    trees_meta_validate_unpack.mojo), never written here."""
    var store = _PLANS.get_or_create_ptr()
    var address = _plan_lookup(handle, False)
    var lock = rebind[MutPointer[Int64, MutUntrackedOrigin]](store[].synchronization.unsafe_ptr())
    shared_slot_lock(lock)
    try:
        var entry = _PlanSlotPtr(unsafe_from_address=address)
        entry[].refusal = words^
        entry[].refusal_late = late
    finally:
        shared_slot_unlock(lock)


def _refused_first(slot: _SlotPtr) -> Bool:
    """THE DOOR'S REFUSAL raised before anything (lane mf-refuse5): a non-RFE state whose entry
    carries the door's words with `late` False."""
    return slot[].refusal.byte_length() != 0 and not slot[].refusal_late and not _rfe_kind(slot[].kind)


def _refused_mask(kind: Int) -> Int:
    """CAPABILITIES of a state refused before anything (lane mf-refuse5): every call a caller may
    admit it for (its FIT raises the door's words first), its kind's classifier bit; no scope, no
    multi-output y but for a multi-output kind, GIL-free (its FIT reads nothing)."""
    var mask = (MEMBER_CAN_FIT | MEMBER_CAN_PREDICT | MEMBER_CAN_PREDICT_PROBA | MEMBER_CAN_DECISION
                | MEMBER_CAN_EXPORT | MEMBER_CAN_IMPORT | MEMBER_CAN_SAMPLE_WEIGHT | MEMBER_CAN_SCORE
                | MEMBER_CAN_TRANSFORM | MEMBER_GIL_FREE)
    if _classifier_kind(kind):
        mask |= MEMBER_CAN_CLASSIFIER
    if _multi_kind(kind):
        mask |= MEMBER_CAN_MULTI_OUTPUT
    return mask


def _plan_first(handle: Int) raises -> MemberPlan:
    """The FIRST nested plan of a plan entry (a search's first candidate),
    copied under the store's lock."""
    var store = _PLANS.get_or_create_ptr()
    var address = _plan_lookup(handle, False)
    var lock = rebind[MutPointer[Int64, MutUntrackedOrigin]](store[].synchronization.unsafe_ptr())
    shared_slot_lock(lock)
    try:
        var slot = _PlanSlotPtr(unsafe_from_address=address)
        if len(slot[].plans) == 0:
            raise Error("member ABI: a meta member plan holds no members")
        return slot[].plans[0].copy()
    finally:
        shared_slot_unlock(lock)


def _plan_words(handle: Int) raises -> List[Float64]:
    """A plan entry's settings words, copied under the store's lock."""
    var store = _PLANS.get_or_create_ptr()
    var address = _plan_lookup(handle, False)
    var lock = rebind[MutPointer[Int64, MutUntrackedOrigin]](store[].synchronization.unsafe_ptr())
    shared_slot_lock(lock)
    try:
        return _PlanSlotPtr(unsafe_from_address=address)[].words.copy()
    finally:
        shared_slot_unlock(lock)


def _plan_lookup(handle: Int, release: Bool) raises -> Int:
    var store = _PLANS.get_or_create_ptr()
    var lock = rebind[MutPointer[Int64, MutUntrackedOrigin]](store[].synchronization.unsafe_ptr())
    shared_slot_lock(lock)
    try:
        var index = handle % _INDEX_SPAN - 1
        if handle <= 0 or index < 0 or index >= store[].count:
            raise Error("RuntimeError: mojolearn: this meta member plan was released")
        var address = _plan_slot_address(index)
        var slot = _PlanSlotPtr(unsafe_from_address=address)
        if not slot[].live or slot[].generation != handle // _INDEX_SPAN:
            raise Error("RuntimeError: mojolearn: this meta member plan was released")
        if release:
            slot[].live = False
            slot[].plans = List[MemberPlan]()
            slot[].weights = List[Float64]()
            slot[].stack_kinds = List[Int]()
            slot[].masks = List[Int]()
            slot[].words = List[Float64]()
            slot[].fold_train = List[List[Int32]]()
            slot[].fold_test = List[List[Int32]]()
            slot[].refusal = String("")
            slot[].refusal_late = False
            slot[].rfe_hops = List[Float64]()
            store[].free.append(index)
        return address
    finally:
        shared_slot_unlock(lock)


def meta_plan_release(handle: Int):
    """The admission's capsule destructor: the plan entry goes. A state made
    from it copied what it reads. Never raises."""
    try:
        _ = _plan_lookup(handle, True)
    except:
        pass


def _plan_final(handle: Int) raises -> MemberPlan:
    """The LAST nested plan of a plan entry (a stack's final estimator), copied
    under the store's lock."""
    var store = _PLANS.get_or_create_ptr()
    var address = _plan_lookup(handle, False)
    var lock = rebind[MutPointer[Int64, MutUntrackedOrigin]](store[].synchronization.unsafe_ptr())
    shared_slot_lock(lock)
    try:
        var slot = _PlanSlotPtr(unsafe_from_address=address)
        if len(slot[].plans) == 0:
            raise Error("member ABI: a meta member plan holds no members")
        return slot[].plans[len(slot[].plans) - 1].copy()
    finally:
        shared_slot_unlock(lock)


# ------------------------------------------------------------ the state store


struct EnsembleMemberSlot(Copyable, Movable, Defaultable):
    var generation: Int
    var live: Bool
    var kind: Int
    var flag: Bool        # soft (vote) / passthrough (stack)
    var folds: Int
    var cap: Int
    var plans: List[MemberPlan]
    var weights: List[Float64]
    var stack_kinds: List[Int]
    var masks: List[Int]
    var glue: EntryRef
    var fitted: Bool
    var classes: Int
    # The fitted X's feature count (0 after an import) and the scalar block
    # [features, classes] PACK reads (MEMBER_BLOCK_SCALARS).
    var features: Int
    var scalar_words: List[Float64]
    # The nested states (the final last), owned; their int64 export view.
    var states: List[Int]
    var state_words: List[Int64]
    # Per nested member, per operation: columns and element type (after fit).
    var pred_cols: List[Int]
    var pred_dtype: List[Int]
    var proba_cols: List[Int]
    var proba_dtype: List[Int]
    var dec_cols: List[Int]
    var dec_dtype: List[Int]
    # Stack: each stacked member's columns, their sum, the float32 meta
    # features the final member was fitted on (it may borrow them).
    var widths: List[Int]
    var width_words: List[Int64]
    var total: Int
    var meta32: List[Float32]
    # The class columns 0 .. classes-1 every nested classifier answers in.
    var iota: List[Int32]
    # The nested fit notices, in fit order: the nested member index and its words.
    var notice_owner: List[Int]
    var notice_words: List[List[Float64]]
    # Lane mf-meta3b. The entry's settings words and a stack's user folds;
    # THE UNIT KINDS' base plan and mask (the entry's; `plans` / `masks` hold
    # one copy per unit after a fit or an import).
    var words: List[Float64]
    var fold_train: List[List[Int32]]
    var fold_test: List[List[Int32]]
    # Lanes mf-split4, mf-split5: THE SPLITTER CV (the entry's): each fit
    # writes its folds into `fold_train` / `fold_test` (`_split_folds`).
    var split: MselSplitSpec
    var base_plans: List[MemberPlan]
    var base_masks: List[Int]
    # THE GIVEN y (lane mf-tail5; core/member_abi.mojo `MemberCall.y_given`):
    # DURING a fit only (`_fit` writes them, clears them on return), the fit's
    # float32 target address `fit_y` and its given y (`fit_given`, 0: none,
    # of `fit_given_dtype`). `_add_drawn` hands the given y to a nested
    # member whose kind reads it and whose target IS `fit_y` (the same rows:
    # a draw selects both); every other target (class codes, re-coded or
    # per-output columns) carries none of it.
    var fit_y: Int
    var fit_given: Int
    var fit_given_dtype: Int
    # Per kept unit: its classes as ensemble codes when it lacks some (empty:
    # every class), its re-coded targets (n words), its row draw and its
    # feature subset (Bagging; empty: every row / column), its out-of-bag rows.
    var unit_held: List[List[Int32]]
    var unit_codes: List[List[Int32]]
    var unit_rows: List[List[Int32]]
    var unit_cols: List[List[Int32]]
    var unit_oob: List[List[Int32]]
    # OneVsRest: the class indicators its units were fitted on.
    var targets: List[List[Int32]]
    # Calibrated: the calibrators (CalFitOut's lists) and each pair's first and count.
    var cal_iso: List[Bool]
    var cal_a: List[Float64]
    var cal_b: List[Float64]
    var cal_m: List[Int]
    var cal_kx: List[List[Float64]]
    var cal_ky: List[List[Float64]]
    var pair_first: List[Int]
    var pair_count: List[Int]
    # The export views: the calibrated pairs' words, the bags' draws, the oob outputs.
    var cal_words: List[Float64]
    var draw_words: List[Int32]
    var oob_words: List[Float64]
    # Lane mf-meta3c, THE BATCHED ROUTE: this Bagging fit is ONE forest
    # state (nested member 0, plan `base_plans[1]`); `batch_fraction` its
    # max_samples (0.0: None) and `batch_bins` its slot-7 bin count, the words
    # PACK writes into the public forest (MEMBER_BLOCK_SCALARS).
    var batched: Bool
    var batch_fraction: Float64
    var batch_bins: Int
    # Lane mf-ada4, THE ADABOOST KIND: the kept rounds' estimator weights and
    # errors (the units are the round members, in round order), their export
    # view (META_BLOCK_ADA), and the fit's data session: its handle, its forest
    # entry and the column-major X it was opened on (a classifier's), OWNED by
    # the state from the open until it is closed (the fit's end, or FREE).
    var ada_alphas: List[Float64]
    var ada_errors: List[Float64]
    var ada_words: List[Float64]
    var ada_session: Int
    var ada_forest: EntryRef
    var ada_xcm: List[Float32]
    # Lane mf-search4. THE SEARCH KIND: the metric route, the fitted words
    # (META_BLOCK_SEARCH; the state's one nested member is the refitted best
    # candidate). THE MULTI-OUTPUT KIND: each output's class count.
    var metric: MetricRoute
    var search_words: List[Float64]
    var out_classes: List[Int]
    # Lane mf-cluster5: the search's cluster entry and labels' reprs.
    var cluster: EntryRef
    var label_words: List[String]
    # Lane mf-msel5, THE CV TOOLS: the kept unit states (owned; 0: a failed
    # fit), each unit's candidate (its plan in `base_plans`), and their
    # handles as export words (META_BLOCK_FOLDS).
    var fold_states: List[Int]
    var fold_owner: List[Int]
    var fold_words: List[Int64]
    # Lane mf-meta5, THE RFE KIND: the entry's x_prep image and first-round refusal; the fitted support and
    # ranking (float32, d words each), the kept column ids (int32, k) and their export view
    # (META_BLOCK_RFE). The one nested member is the final estimator.
    var rfe_run: ClsRunPlan
    var refusal: String
    var refusal_late: Bool  # lane mf-refuse5: THE DOOR'S REFUSAL after the nested fits (`meta_plan_refuse`)
    var rfe_support: List[Float32]
    var rfe_ranking: List[Float32]
    var rfe_columns: List[Int32]
    var rfe_words: List[Float64]
    # Lane mf-del5b: the plan entry's dotted-getter hops (`meta_plan_rfe`; empty: a leaf base).
    var rfe_hops: List[Float64]

    def __init__(out self):
        self.generation = 0
        self.live = False
        self.kind = 0
        self.flag = False
        self.folds = 0
        self.cap = 0
        self.plans = List[MemberPlan]()
        self.weights = List[Float64]()
        self.stack_kinds = List[Int]()
        self.masks = List[Int]()
        self.glue = entry_ref_unresolved()
        self.fitted = False
        self.classes = 0
        self.features = 0
        self.scalar_words = List[Float64]()
        self.states = List[Int]()
        self.state_words = List[Int64]()
        self.pred_cols = List[Int]()
        self.pred_dtype = List[Int]()
        self.proba_cols = List[Int]()
        self.proba_dtype = List[Int]()
        self.dec_cols = List[Int]()
        self.dec_dtype = List[Int]()
        self.widths = List[Int]()
        self.width_words = List[Int64]()
        self.total = 0
        self.meta32 = List[Float32]()
        self.iota = List[Int32]()
        self.notice_owner = List[Int]()
        self.notice_words = List[List[Float64]]()
        self.words = List[Float64]()
        self.fold_train = List[List[Int32]]()
        self.fold_test = List[List[Int32]]()
        self.split = MselSplitSpec()
        self.base_plans = List[MemberPlan]()
        self.base_masks = List[Int]()
        self.fit_y = 0
        self.fit_given = 0
        self.fit_given_dtype = MEMBER_F32
        self.unit_held = List[List[Int32]]()
        self.unit_codes = List[List[Int32]]()
        self.unit_rows = List[List[Int32]]()
        self.unit_cols = List[List[Int32]]()
        self.unit_oob = List[List[Int32]]()
        self.targets = List[List[Int32]]()
        self.cal_iso = List[Bool]()
        self.cal_a = List[Float64]()
        self.cal_b = List[Float64]()
        self.cal_m = List[Int]()
        self.cal_kx = List[List[Float64]]()
        self.cal_ky = List[List[Float64]]()
        self.pair_first = List[Int]()
        self.pair_count = List[Int]()
        self.cal_words = List[Float64]()
        self.draw_words = List[Int32]()
        self.oob_words = List[Float64]()
        self.batched = False
        self.batch_fraction = 0.0
        self.batch_bins = 0
        self.ada_alphas = List[Float64]()
        self.ada_errors = List[Float64]()
        self.ada_words = List[Float64]()
        self.ada_session = 0
        self.ada_forest = entry_ref_unresolved()
        self.ada_xcm = List[Float32]()
        self.metric = MetricRoute(entry_ref_unresolved().call, False, False)
        self.search_words = List[Float64]()
        self.out_classes = List[Int]()
        self.cluster = entry_ref_unresolved()
        self.label_words = List[String]()
        self.fold_states = List[Int]()
        self.fold_owner = List[Int]()
        self.fold_words = List[Int64]()
        self.rfe_run = ClsRunPlan(entry_ref_unresolved(), False, False, False, False, False)
        self.refusal = String("")
        self.refusal_late = False
        self.rfe_support = List[Float32]()
        self.rfe_ranking = List[Float32]()
        self.rfe_columns = List[Int32]()
        self.rfe_words = List[Float64]()
        self.rfe_hops = List[Float64]()


struct EnsembleMemberStore(Defaultable, Movable):
    var synchronization: List[Int64]
    var chunks: List[List[EnsembleMemberSlot]]
    var free: List[Int]
    var count: Int

    def __init__(out self):
        self.synchronization = List[Int64](length=2, fill=0)
        self.chunks = List[List[EnsembleMemberSlot]]()
        self.free = List[Int]()
        self.count = 0


comptime _STORE = SharedSlot[EnsembleMemberStore, "MojolearnEnsembleMemberStoreV1", 1]
comptime _SlotPtr = MutPointer[EnsembleMemberSlot, MutUntrackedOrigin]


def _slot_address(index: Int) raises -> Int:
    var store = _STORE.get_or_create_ptr()
    return Int(store[].chunks[index // _CHUNK].unsafe_ptr()) + (index % _CHUNK) * size_of[EnsembleMemberSlot]()


def _mint() raises -> Int:
    var store = _STORE.get_or_create_ptr()
    var lock = rebind[MutPointer[Int64, MutUntrackedOrigin]](store[].synchronization.unsafe_ptr())
    shared_slot_lock(lock)
    try:
        var index = -1
        if len(store[].free) > 0:
            index = store[].free.pop()
        else:
            if store[].count >= _INDEX_SPAN - 1:
                raise Error("RuntimeError: mojolearn: too many live meta member states")
            if store[].count == len(store[].chunks) * _CHUNK:
                store[].chunks.append(List[EnsembleMemberSlot](length=_CHUNK, fill=EnsembleMemberSlot()))
            index = store[].count
            store[].count += 1
        var slot = _SlotPtr(unsafe_from_address=_slot_address(index))
        slot[].generation += 1
        slot[].live = True
        slot[].fitted = False
        return slot[].generation * _INDEX_SPAN + index + 1
    finally:
        shared_slot_unlock(lock)


def _lookup(handle: Int, release: Bool) raises -> Int:
    """The slot address of a live handle; a released, foreign or made-up
    handle is refused (never a stale read). `release`: the slot's words are
    dropped (the caller freed its nested states first)."""
    var store = _STORE.get_or_create_ptr()
    var lock = rebind[MutPointer[Int64, MutUntrackedOrigin]](store[].synchronization.unsafe_ptr())
    shared_slot_lock(lock)
    try:
        var index = handle % _INDEX_SPAN - 1
        if handle <= 0 or index < 0 or index >= store[].count:
            raise Error("NotFittedError: this member state was released or never created")
        var address = _slot_address(index)
        var slot = _SlotPtr(unsafe_from_address=address)
        if not slot[].live or slot[].generation != handle // _INDEX_SPAN:
            raise Error("NotFittedError: this member state was released or never created")
        if release:
            slot[] = EnsembleMemberSlot()
            slot[].generation = handle // _INDEX_SPAN
            store[].free.append(index)
        return address
    finally:
        shared_slot_unlock(lock)


def _name(kind: Int) -> String:
    if kind == META_VOTE_CLASSIFIER:
        return String("VotingClassifier")
    if kind == META_VOTE_REGRESSOR:
        return String("VotingRegressor")
    if kind == META_STACK_CLASSIFIER:
        return String("StackingClassifier")
    if kind == META_OVR:
        return String("OneVsRestClassifier")
    if kind == META_CALIBRATED:
        return String("CalibratedClassifierCV")
    if kind == META_BAG_CLASSIFIER:
        return String("BaggingClassifier")
    if kind == META_BAG_REGRESSOR:
        return String("BaggingRegressor")
    if kind == META_ADA_CLASSIFIER:
        return String("AdaBoostClassifier")
    if kind == META_ADA_REGRESSOR:
        return String("AdaBoostRegressor")
    if _search_kind(kind):
        return String("BaseSearchCV")
    if kind == META_MULTI_CLASSIFIER:
        return String("MultiOutputClassifier")
    if kind == META_MULTI_REGRESSOR:
        return String("MultiOutputRegressor")
    if _rfe_kind(kind):
        return String("RFE")
    return String("StackingRegressor")


def _ada_close_session(slot: _SlotPtr):
    """Close the state's AdaBoost data session, if one is open, and drop the
    column-major X it was opened on (THE ADABOOST KIND). Never raises."""
    if slot[].ada_session > 0:
        var handle = slot[].ada_session
        slot[].ada_session = 0
        try:
            xf_session_close(slot[].ada_forest, handle)
        except:
            pass
    slot[].ada_xcm = List[Float32]()


def _drop_states(slot: _SlotPtr):
    """Free every nested state the slot owns. Never raises."""
    for u in range(len(slot[].states)):  # driver-loop: over the nested members (their states)
        if slot[].states[u] != 0 and u < len(slot[].plans):
            var call = MemberCall(slot[].plans[u].route, slot[].states[u])
            member_free(slot[].plans[u], call)
        slot[].states[u] = 0
    slot[].states = List[Int]()
    slot[].state_words = List[Int64]()
    # THE CV TOOLS' kept unit states (lane mf-msel5).
    for u in range(len(slot[].fold_states)):  # driver-loop: over the kept unit states (one release each)
        var c = slot[].fold_owner[u] if u < len(slot[].fold_owner) else -1
        if slot[].fold_states[u] != 0 and c >= 0 and c < len(slot[].base_plans):
            var call = MemberCall(slot[].base_plans[c].route, slot[].fold_states[u])
            member_free(slot[].base_plans[c], call)
        slot[].fold_states[u] = 0
    slot[].fold_states = List[Int]()
    slot[].fold_owner = List[Int]()
    slot[].fold_words = List[Int64]()
    slot[].fitted = False
    slot[].meta32 = List[Float32]()
    slot[].notice_owner = List[Int]()
    slot[].notice_words = List[List[Float64]]()
    slot[].unit_held = List[List[Int32]]()
    slot[].unit_codes = List[List[Int32]]()
    slot[].unit_rows = List[List[Int32]]()
    slot[].unit_cols = List[List[Int32]]()
    slot[].unit_oob = List[List[Int32]]()
    slot[].targets = List[List[Int32]]()
    slot[].cal_iso = List[Bool]()
    slot[].cal_a = List[Float64]()
    slot[].cal_b = List[Float64]()
    slot[].cal_m = List[Int]()
    slot[].cal_kx = List[List[Float64]]()
    slot[].cal_ky = List[List[Float64]]()
    slot[].pair_first = List[Int]()
    slot[].pair_count = List[Int]()
    slot[].cal_words = List[Float64]()
    slot[].draw_words = List[Int32]()
    slot[].oob_words = List[Float64]()
    slot[].batched = False
    slot[].batch_fraction = 0.0
    slot[].batch_bins = 0
    _ada_close_session(slot)
    slot[].ada_alphas = List[Float64]()
    slot[].ada_errors = List[Float64]()
    slot[].ada_words = List[Float64]()
    slot[].search_words = List[Float64]()
    slot[].out_classes = List[Int]()
    slot[].rfe_support = List[Float32]()
    slot[].rfe_ranking = List[Float32]()
    slot[].rfe_columns = List[Int32]()
    slot[].rfe_words = List[Float64]()
    slot[].rfe_hops = List[Float64]()


# --------------------------------------------------------------- the masks


def _all(masks: List[Int], bit: Int) -> Bool:
    for u in range(len(masks)):  # driver-loop: over the nested kinds (one mask each)
        if (masks[u] & bit) == 0:
            return False
    return True


def _any(masks: List[Int], bit: Int) -> Bool:
    for u in range(len(masks)):  # driver-loop: over the nested kinds (one mask each)
        if (masks[u] & bit) != 0:
            return True
    return False


def _reads_given(slot: _SlotPtr, masks: List[Int]) -> Int:
    """`MEMBER_READS_Y_GIVEN` of a META state (core/member_abi.mojo, lane
    mf-tail5; kind-exact): a REGRESSOR kind whose own splitter is stratified
    (Stacking's, the search's: `_split_folds` codes the targets' values), or
    one holding a nested kind that reads it (`_add_drawn` hands it on). A
    classifier kind reads class codes, never the given y."""
    if _classifier_kind(slot[].kind):
        return 0
    if (slot[].split.kind != MSEL_SPEC_NONE and msel_spec_stratified(slot[].split.kind)) or _any(masks, MEMBER_READS_Y_GIVEN):
        return MEMBER_READS_Y_GIVEN
    return 0


def _kind_mask(kind: Int, flag: Bool, masks: List[Int]) -> Int:
    """The kind-exact capability mask of a META state over nested kinds
    `masks` (module note, KIND-EXACT BITS; a unit kind: its base's one
    mask). A unit kind answers what the door's operations of its class
    answer: OneVsRest predict_proba when its base has it, Calibrated and a
    Bagging classifier predict_proba always (their admission requires the
    base's scores), sample weights for Bagging only (the OneVsRest and
    Calibrated fits take none), no decision_function."""
    var mask = MEMBER_CAN_FIT | MEMBER_CAN_PREDICT | MEMBER_CAN_EXPORT | MEMBER_CAN_IMPORT | MEMBER_BORROWS_FIT
    if _classifier_kind(kind):
        mask |= MEMBER_CAN_CLASSIFIER
    if _search_kind(kind):
        # THE SEARCH KIND: what every candidate answers (the refitted best may
        # be any of them); a fit takes no sample weights.
        if len(masks) > 0 and _all(masks, MEMBER_CAN_PREDICT_PROBA):
            mask |= MEMBER_CAN_PREDICT_PROBA
        if len(masks) > 0 and _all(masks, MEMBER_CAN_DECISION):
            mask |= MEMBER_CAN_DECISION
        if len(masks) > 0 and _all(masks, MEMBER_CAN_SCORE):
            mask |= MEMBER_CAN_SCORE
        if len(masks) > 0 and _all(masks, MEMBER_CAN_MULTI_OUTPUT):
            mask |= MEMBER_CAN_MULTI_OUTPUT
        if len(masks) > 0 and _all(masks, MEMBER_GIL_FREE):
            mask |= MEMBER_GIL_FREE
        elif len(masks) > 0 and _all_either(masks):
            mask |= MEMBER_GIL_DOOR
        # THE RESIDENT ROUTE (lane mf-search5): `flag` is the admission's
        # word that the search's fold rows may stay on the device (its own
        # `device` is not 'cpu' beside a GPU set and no candidate names a
        # `device`); then a caller in a resident scope hands the fit its
        # scope and X's pin (`_fit_search`), for the candidates that keep X
        # resident.
        if flag and _any(masks, MEMBER_CAN_SCOPE):
            mask |= MEMBER_CAN_SCOPE
        return mask
    if _unit_kind(kind):
        var base = masks[0] if len(masks) > 0 else 0
        if _multi_kind(kind):
            # THE MULTI-OUTPUT KIND: predict and score over a 2-D y; the base's
            # sample weights (MultiOutput's fit hands them to every output).
            mask |= MEMBER_CAN_MULTI_OUTPUT | MEMBER_CAN_SCORE
            if kind == META_MULTI_CLASSIFIER and (base & MEMBER_CAN_PREDICT_PROBA) != 0:
                # (lane mf-tail5) predict_proba: every output's block in one float64 dst
                mask |= MEMBER_CAN_PREDICT_PROBA
        if kind == META_CALIBRATED or kind == META_BAG_CLASSIFIER:
            mask |= MEMBER_CAN_PREDICT_PROBA
        elif kind == META_ADA_CLASSIFIER:
            # AdaBoostClassifier's predict_proba and decision_function
            mask |= MEMBER_CAN_PREDICT_PROBA | MEMBER_CAN_DECISION
        elif kind == META_OVR and (base & MEMBER_CAN_PREDICT_PROBA) != 0:
            mask |= MEMBER_CAN_PREDICT_PROBA
        if (_bag_kind(kind) or _multi_kind(kind)) and (base & MEMBER_CAN_SAMPLE_WEIGHT) != 0:
            mask |= MEMBER_CAN_SAMPLE_WEIGHT
        if _ada_kind(kind):
            # AdaBoost (lane mf-split4b): the caller's float32 sample weights
            # are the words its door normalizes (`_fit_ada`); every round
            # hands its base its own boosting weights (whatever the caller
            # gave), so the base's bit is not read.
            mask |= MEMBER_CAN_SAMPLE_WEIGHT
        # A Bagging entry's batched forest (its second mask) runs on the same
        # thread rules: the GIL bits both kinds have.
        if len(masks) > 0 and _all(masks, MEMBER_GIL_FREE):
            mask |= MEMBER_GIL_FREE
        elif len(masks) > 0 and _all_either(masks):
            mask |= MEMBER_GIL_DOOR
        return mask
    if len(masks) > 0:
        var final = masks[len(masks) - 1]
        if kind == META_VOTE_CLASSIFIER and flag and _all(masks, MEMBER_CAN_PREDICT_PROBA):
            mask |= MEMBER_CAN_PREDICT_PROBA
        if kind == META_STACK_CLASSIFIER:
            if (final & MEMBER_CAN_PREDICT_PROBA) != 0:
                mask |= MEMBER_CAN_PREDICT_PROBA
            if (final & MEMBER_CAN_DECISION) != 0:
                mask |= MEMBER_CAN_DECISION
    if _all(masks, MEMBER_CAN_SAMPLE_WEIGHT):
        mask |= MEMBER_CAN_SAMPLE_WEIGHT
    if _all(masks, MEMBER_GIL_FREE):
        mask |= MEMBER_GIL_FREE
    elif _all_either(masks):
        mask |= MEMBER_GIL_DOOR
    return mask


def _all_either(masks: List[Int]) -> Bool:
    for u in range(len(masks)):  # driver-loop: over the nested kinds (one mask each)
        if (masks[u] & (MEMBER_GIL_FREE | MEMBER_GIL_DOOR)) == 0:
            return False
    return True


comptime _UNION = (MEMBER_CAN_FIT | MEMBER_CAN_PREDICT | MEMBER_CAN_PREDICT_PROBA | MEMBER_CAN_DECISION
    | MEMBER_CAN_EXPORT | MEMBER_CAN_IMPORT | MEMBER_CAN_CLASSIFIER | MEMBER_CAN_SAMPLE_WEIGHT | MEMBER_BORROWS_FIT
    | MEMBER_GIL_FREE | MEMBER_GIL_DOOR | MEMBER_CAN_SCORE | MEMBER_CAN_MULTI_OUTPUT | MEMBER_CAN_SCOPE
    | MEMBER_CAN_TRANSFORM | MEMBER_READS_Y_GIVEN)


def _set_word(mask: Int) -> UInt8:
    """THE GIL RULE word of a nested member in its set."""
    if (mask & MEMBER_GIL_FREE) != 0:
        return MEMBER_SET_WORKER
    if (mask & MEMBER_GIL_DOOR) != 0:
        return MEMBER_SET_TURN_RELEASED
    return MEMBER_SET_TURN_HELD


# ------------------------------------------------------------------ NEW


def _new(call: _CallPtr) raises:
    if call[].param_count != META_PARAM_COUNT or not call[].params:
        raise Error("member ABI: a meta member takes six parameters (kind, plan, soft or passthrough, folds, "
            + "worker bound, reserved)")
    var words = _F64P(unsafe_from_address=call[].params)
    var kind = Int(words[0])
    if kind < META_VOTE_CLASSIFIER or kind > META_KIND_LAST:
        raise Error("member ABI: unknown meta member kind")
    var plan_address = _plan_lookup(Int(words[1]), False)
    var state = _mint()
    var store = _PLANS.get_or_create_ptr()
    var lock = rebind[MutPointer[Int64, MutUntrackedOrigin]](store[].synchronization.unsafe_ptr())
    var slot = _SlotPtr(unsafe_from_address=_lookup(state, False))
    slot[].kind = kind
    slot[].flag = words[2] != 0.0
    slot[].folds = Int(words[3])
    slot[].cap = Int(words[4])
    shared_slot_lock(lock)
    try:
        var entry = _PlanSlotPtr(unsafe_from_address=plan_address)
        slot[].plans = entry[].plans.copy()
        slot[].weights = entry[].weights.copy()
        slot[].stack_kinds = entry[].stack_kinds.copy()
        slot[].masks = entry[].masks.copy()
        slot[].words = entry[].words.copy()
        slot[].fold_train = entry[].fold_train.copy()
        slot[].fold_test = entry[].fold_test.copy()
        slot[].metric = entry[].metric
        slot[].split = entry[].split.copy()
        slot[].cluster = entry[].cluster
        slot[].label_words = entry[].label_words.copy()
        slot[].rfe_run = entry[].rfe_run
        slot[].refusal = entry[].refusal.copy()
        slot[].refusal_late = entry[].refusal_late
        slot[].rfe_hops = entry[].rfe_hops.copy()
    finally:
        shared_slot_unlock(lock)
    try:
        if _refused_first(slot):
            # THE DOOR'S REFUSAL (lane mf-refuse5): its FIT raises the words first; nothing to check.
            slot[].glue = xt_ref(TreesEntryKey(call[].route.mode, call[].route.vendor))
            call[].state = state
            return
        if len(slot[].plans) == 0:
            raise Error("member ABI: a meta member plan holds no members")
        var user_folds = len(slot[].fold_test)
        # THE SPLITTER CV: its fold count is the fit's (word 3 is 0).
        var splitter = slot[].split.kind != MSEL_SPEC_NONE
        if len(slot[].fold_train) != user_folds:
            raise Error("member ABI: a meta member's user folds need a train and a test side each")
        if _stack_kind_code(kind) and (len(slot[].stack_kinds) + 1 != len(slot[].plans)
                                       or (slot[].folds < 2 and user_folds == 0 and not splitter)):
            raise Error("member ABI: a stacking member needs its members, its final estimator and cv >= 2")
        if _search_kind(kind):
            var count = Int(slot[].words[META_SEARCH_COUNT]) if len(slot[].words) >= META_SEARCH_WORDS else -1
            var plain = _search_groups_end(slot[].words, count) if count >= 0 else -1
            if (count < 0 or plain < 0
                    or (len(slot[].words) != plain and len(slot[].words) != plain + META_SEARCH_TOOL_WORDS)
                    or len(slot[].masks) != len(slot[].plans) or (slot[].folds < 2 and user_folds == 0 and not splitter)
                    or (count > 0 and (slot[].words[META_SEARCH_REFIT_SCORER] < 0.0
                                       or Int(slot[].words[META_SEARCH_REFIT_SCORER]) >= count))):
                raise Error("member ABI: a search member needs its candidates, one mask each, its "
                            + String(META_SEARCH_WORDS) + " settings words, one scorer word group per "
                            + "scorer, a refit scorer among them and cv >= 2")
            slot[].base_plans = slot[].plans.copy()
            slot[].base_masks = slot[].masks.copy()
        elif _rfe_kind(kind):
            if len(slot[].plans) != 1 or len(slot[].masks) != 1 or len(slot[].words) != META_RFE_WORDS:
                raise Error("member ABI: an RFE member needs its one estimator, its mask and its "
                            + String(META_RFE_WORDS) + " settings words")
            slot[].base_plans = slot[].plans.copy()
            slot[].base_masks = slot[].masks.copy()
        elif _unit_kind(kind):
            var need = META_OVR_WORDS
            if kind == META_CALIBRATED:
                need = META_CAL_WORDS
            elif _bag_kind(kind):
                need = META_BAG_WORDS
            elif _ada_kind(kind):
                need = META_ADA_WORDS
            elif _multi_kind(kind):
                need = META_MULTI_WORDS
            # A Bagging or AdaBoost entry may carry its tree's class-weight
            # tail after its settings words (META_CW_*, lane mf-split4; none:
            # no class_weight).
            if (_bag_kind(kind) or _ada_kind(kind)) and len(slot[].words) >= need + META_CW_HEAD:
                need += _cw_tail_words(slot[].words, need)
            # A Bagging entry may hold a second plan: its batched forest.
            var plans = 1
            if _bag_kind(kind) and len(slot[].words) == need and slot[].words[META_BAG_BATCHED] != 0.0:
                plans = 2
            if len(slot[].plans) != plans or len(slot[].masks) != plans or len(slot[].words) != need:
                raise Error("member ABI: a " + _name(kind) + " member needs its one base estimator and its "
                            + String(need) + " settings words")
            if kind == META_CALIBRATED and slot[].folds < 2 and user_folds == 0 and not splitter:
                raise Error("member ABI: a CalibratedClassifierCV member needs cv >= 2 or its listed folds")
            if _ada_kind(kind):
                var family = slot[].plans[0].family
                if (family != MEMBER_FAMILY_TREE and family != MEMBER_FAMILY_FOREST
                        and family != MEMBER_FAMILY_EXTRA_TREES):
                    raise Error("member ABI: an " + _name(kind) + " member's base is a tree, forest or "
                                + "extra-trees member")
                if slot[].words[META_ADA_NATIVE] != 0.0 and family != MEMBER_FAMILY_TREE:
                    raise Error("member ABI: an " + _name(kind) + " member's session rounds fit a decision tree")
                if Int(slot[].words[META_ADA_COUNT]) < 1:
                    raise Error("member ABI: an " + _name(kind) + " member needs n_estimators >= 1")
            slot[].base_plans = slot[].plans.copy()
            slot[].base_masks = slot[].masks.copy()
        elif not _stack_kind_code(kind) and len(slot[].weights) != len(slot[].plans):
            raise Error("member ABI: a voting member needs one weight per member")
        # The glue entry of the x_trees image this META entry runs on,
        # resolved once per state (never per call).
        slot[].glue = xt_ref(TreesEntryKey(call[].route.mode, call[].route.vendor))
    except error:
        _ = _lookup(state, True)
        raise error^
    call[].state = state


# ------------------------------------------------------------------ helpers


def _nested_outputs(slot: _SlotPtr) raises:
    """After a fit or import: each nested member's columns and element type
    per operation it answers (asked of its fitted state once)."""
    var count = len(slot[].plans)
    slot[].pred_cols = List[Int](length=count, fill=0)
    slot[].pred_dtype = List[Int](length=count, fill=MEMBER_F32)
    slot[].proba_cols = List[Int](length=count, fill=0)
    slot[].proba_dtype = List[Int](length=count, fill=MEMBER_F32)
    slot[].dec_cols = List[Int](length=count, fill=0)
    slot[].dec_dtype = List[Int](length=count, fill=MEMBER_F32)
    for u in range(count):  # driver-loop: over the nested members (one OUTPUTS question per operation)
        if slot[].states[u] == 0:
            continue  # not fitted yet (a stack's final, during its fit)
        var plan = slot[].plans[u].copy()
        var call = member_call_of(plan, slot[].states[u])
        slot[].pred_cols[u] = member_outputs_of(plan, call, MEMBER_OP_PREDICT)
        slot[].pred_dtype[u] = call.out_dtype if call.out_dtype != MEMBER_DTYPE_UNSAID else MEMBER_F32
        if plan.member.can(MEMBER_CAN_PREDICT_PROBA) and (slot[].masks[u] & MEMBER_CAN_PREDICT_PROBA) != 0:
            slot[].proba_cols[u] = member_outputs_of(plan, call, MEMBER_OP_PREDICT_PROBA)
            slot[].proba_dtype[u] = call.out_dtype if call.out_dtype != MEMBER_DTYPE_UNSAID else MEMBER_F32
        if plan.member.can(MEMBER_CAN_DECISION) and (slot[].masks[u] & MEMBER_CAN_DECISION) != 0:
            slot[].dec_cols[u] = member_outputs_of(plan, call, MEMBER_OP_DECISION)
            slot[].dec_dtype[u] = call.out_dtype if call.out_dtype != MEMBER_DTYPE_UNSAID else MEMBER_F32
    slot[].iota = List[Int32](length=max(slot[].classes, 1), fill=Int32(0))
    for c in range(slot[].classes):  # small-loop(the class columns 0 .. classes-1, once per fit)
        slot[].iota[c] = Int32(c)
    slot[].state_words = List[Int64](length=count, fill=Int64(0))
    for u in range(count):  # driver-loop: over the nested members (the export view)
        slot[].state_words[u] = Int64(slot[].states[u])
    # [features, classes]; a Bagging state adds [batched, max_samples, bins]
    # (THE BATCHED ROUTE: the words PACK writes into the public forest).
    slot[].scalar_words = [Float64(slot[].features), Float64(slot[].classes)]
    if _bag_kind(slot[].kind):
        slot[].scalar_words.append(1.0 if slot[].batched else 0.0)
        slot[].scalar_words.append(slot[].batch_fraction)
        slot[].scalar_words.append(Float64(slot[].batch_bins))
    slot[].width_words = List[Int64](length=len(slot[].widths), fill=Int64(0))
    for j in range(len(slot[].widths)):  # driver-loop: over the stacked members (the export view)
        slot[].width_words[j] = Int64(slot[].widths[j])


def _member_of(slot: _SlotPtr, u: Int, state: Int, operation: Int, weight: Float64) -> VoteMember:
    """Nested member `u` over `state` as a RUN's `VoteMember` (its plan read
    in place: the slot's list never moves while the state lives)."""
    var cols = slot[].pred_cols[u]
    var dtype = slot[].pred_dtype[u]
    if operation == MEMBER_OP_PREDICT_PROBA:
        cols = slot[].proba_cols[u]
        dtype = slot[].proba_dtype[u]
    elif operation == MEMBER_OP_DECISION:
        cols = slot[].dec_cols[u]
        dtype = slot[].dec_dtype[u]
    return VoteMember(Int(UnsafePointer(to=slot[].plans[u])), state, operation, cols, dtype,
                      Int(slot[].iota.unsafe_ptr()), weight)


def _stack_operation(kind: Int) -> Int:
    if kind == STACK_PROBA:
        return MEMBER_OP_PREDICT_PROBA
    if kind == STACK_DECISION:
        return MEMBER_OP_DECISION
    return MEMBER_OP_PREDICT


def _raise_member(m: VoteMember, rows: VoteRows) raises:
    """The words of a nested member whose call a RUN reported refused: the
    same call again, raising."""
    var plan = MutPointer[MemberPlan, MutUntrackedOrigin](unsafe_from_address=m.plan)
    var call = member_call_of(plan[], m.state)
    call.x = MemberMatrix(rows.x, rows.n, rows.d)
    var scratch = List[Int64](length=max(rows.n * m.cols, 1), fill=Int64(0))
    call.dst = MemberMatrix(Int(scratch.unsafe_ptr()), rows.n, m.cols, m.dtype)
    _ = member_call(plan[].member, m.operation, call)
    _ = len(scratch)  # written through the call above
    raise Error("RuntimeError: mojolearn: a meta member's member refused its call")


def _take_notices(slot: _SlotPtr, u: Int, plan: MemberPlan, state: Int):
    """Keep nested member `u`'s fit notices (raised by the boundary's PACK
    with that member's own context, in fit order). Never raises."""
    try:
        var words = List[Float64]()
        member_notices_take(plan, state, words)
        if len(words) > 0:
            slot[].notice_owner.append(u)
            slot[].notice_words.append(words^)
    except:
        pass


def _set_of(slot: _SlotPtr, x: Int, rows: Int, cols: Int, y_dtype: Int) -> MemberSetRun:
    var run = MemberSetRun(x, rows, cols, y_dtype, MEMBER_I32)
    run.cap = slot[].cap
    return run^


def _add(slot: _SlotPtr, mut run: MemberSetRun, u: Int, classes: Int, target: Int, weights: Int, draw: Int,
         count: Int):
    _add_drawn(slot, run, u, classes, target, weights, draw, count, 0, 0)


def _add_drawn(slot: _SlotPtr, mut run: MemberSetRun, u: Int, classes: Int, target: Int, weights: Int, draw: Int,
               count: Int, column_draw: Int, column_count: Int):
    """Nested member `u` into `run`, reading its row draw and its feature
    subset (counts 0: every row / column)."""
    # The slot's own plan carries the class count too: the RUNs' member calls
    # read it in place (`_member_of`).
    slot[].plans[u].classes = classes
    var plan = slot[].plans[u].copy()
    var mask = slot[].masks[u]
    run.add(plan^, target, weights, draw, count, column_draw, column_count,
            UInt8(1) if (mask & MEMBER_CAN_TEST_ROWS) != 0 else UInt8(0),
            UInt8(1) if (mask & MEMBER_BORROWS_FIT) != 0 else UInt8(0), _set_word(mask))
    # THE GIVEN y (lane mf-tail5, the slot's `fit_given`): only to a kind that
    # reads it, and only beside the fit's own targets (the set selects its
    # rows with the member's draw, as it selects y's).
    if slot[].fit_given != 0 and target != 0 and target == slot[].fit_y and (mask & MEMBER_READS_Y_GIVEN) != 0:
        run.given_target(slot[].fit_given, slot[].fit_given_dtype)


def _expand(slot: _SlotPtr, units: Int):
    """A unit kind: one copy of the base plan and mask per unit (module
    note, THE UNIT KINDS)."""
    slot[].plans = List[MemberPlan](capacity=units)
    slot[].masks = List[Int](capacity=units)
    for _ in range(units):  # driver-loop: over the units (one plan copy each)
        slot[].plans.append(slot[].base_plans[0].copy())
        slot[].masks.append(slot[].base_masks[0])


def _held_and_codes(codes: Int, k: Int, n: Int, draws: List[Int], counts: List[Int],
                    mut held: List[List[Int32]], mut recoded: List[List[Int32]]):
    """Per unit (its rows: the int32 row numbers at `draws[u]`, `counts[u]`
    of them; 0: every row): the ensemble codes its rows hold, ascending,
    when it lacks some (else empty), and its re-coded targets (`n` words:
    the unit's own code at every row of a held class; else empty), the y
    the door's member fit on those rows encodes. Both passes on the host
    cores, one task per unit."""
    var units = len(draws)
    var seen = List[UInt8](length=max(units * k, 1), fill=UInt8(0))
    var seen_at = Int(seen.unsafe_ptr())
    var draws_at = Int(draws.unsafe_ptr())
    var counts_at = Int(counts.unsafe_ptr())

    def _mark(u: Int) {imm seen_at, imm draws_at, imm counts_at, imm codes, imm k, imm n}:
        var draw = MutPointer[Int, MutUntrackedOrigin](unsafe_from_address=draws_at)[u]
        var count = MutPointer[Int, MutUntrackedOrigin](unsafe_from_address=counts_at)[u]
        var rows = count if draw != 0 else n
        var picks = MutPointer[Int32, MutUntrackedOrigin](unsafe_from_address=draw)
        var y = MutPointer[Int32, MutUntrackedOrigin](unsafe_from_address=codes)
        var marks = MutPointer[UInt8, MutUntrackedOrigin](unsafe_from_address=seen_at + u * k)
        for i in range(rows):  # small-loop(one pass over one unit's rows)
            var c = Int(y[Int(picks[i]) if draw != 0 else i])
            if c >= 0 and c < k:
                marks[c] = UInt8(1)

    host_parallelize(_mark, units)
    held = List[List[Int32]](capacity=units)
    recoded = List[List[Int32]](capacity=units)
    var maps = List[Int32](length=max(units * k, 1), fill=Int32(0))
    var lacking = List[Int]()
    for u in range(units):  # driver-loop: over the units (their held classes)
        var present = List[Int32]()
        for c in range(k):  # small-loop(the classes of one unit)
            if seen[u * k + c] != UInt8(0):
                maps[u * k + c] = Int32(len(present))
                present.append(Int32(c))
        if len(present) == k:
            held.append(List[Int32]())
            recoded.append(List[Int32]())
        else:
            held.append(present^)
            recoded.append(List[Int32](length=max(n, 1), fill=Int32(0)))
            lacking.append(u)
    _ = len(seen)  # read through the tasks above
    if len(lacking) == 0:
        return
    var targets = List[Int](capacity=len(lacking))
    for i in range(len(lacking)):  # driver-loop: over the lacking units (their target addresses)
        targets.append(Int(recoded[lacking[i]].unsafe_ptr()))
    var lacking_at = Int(lacking.unsafe_ptr())
    var targets_at = Int(targets.unsafe_ptr())
    var maps_at = Int(maps.unsafe_ptr())

    def _recode(i: Int) {imm lacking_at, imm targets_at, imm maps_at, imm codes, imm k, imm n}:
        var u = MutPointer[Int, MutUntrackedOrigin](unsafe_from_address=lacking_at)[i]
        var dst = MutPointer[Int32, MutUntrackedOrigin](
            unsafe_from_address=MutPointer[Int, MutUntrackedOrigin](unsafe_from_address=targets_at)[i])
        var to = MutPointer[Int32, MutUntrackedOrigin](unsafe_from_address=maps_at + 4 * u * k)
        var y = MutPointer[Int32, MutUntrackedOrigin](unsafe_from_address=codes)
        for r in range(n):  # small-loop(one pass over the rows: the unit's code at each)
            var c = Int(y[r])
            dst[r] = to[c] if c >= 0 and c < k else Int32(0)

    host_parallelize(_recode, len(lacking))
    _ = len(maps)  # read through the tasks above
    _ = len(targets)
    _ = len(lacking)


def _held_classes(held: List[Int32], k: Int) -> Int:
    return len(held) if len(held) > 0 else k


def _index_of_code(held: List[Int32], code: Int) -> Int:
    """`list.index(code)` of a unit's classes (`held`; empty: 0 .. k-1, so
    any code below k is its own index), -1 when absent."""
    if len(held) == 0:
        return code
    for c in range(len(held)):  # small-loop(a unit's classes)
        if Int(held[c]) == code:
            return c
    return -1


def _codes_into(slot: _SlotPtr, u: Int, state: Int, rows: VoteRows, dst: Int) raises:
    """Nested member `u`'s int32 class codes over `rows` at `dst` (int64
    codes narrowed once, as the door's int32 cast narrowed them); its call
    refusing raises its own words."""
    var call = member_call_of(slot[].plans[u], state)
    call.x = MemberMatrix(rows.x, rows.n, rows.d)
    if slot[].pred_dtype[u] == MEMBER_I64:
        var wide = List[Int64](length=max(rows.n, 1), fill=Int64(0))
        call.dst = MemberMatrix(Int(wide.unsafe_ptr()), rows.n, 1, MEMBER_I64)
        _ = member_call(slot[].plans[u].member, MEMBER_OP_PREDICT, call)
        var codes = _I32P(unsafe_from_address=dst)
        for i in range(rows.n):  # small-loop(one narrowing pass over one member's codes, as the door's cast)
            codes[i] = Int32(wide[i])
        _ = len(wide)
    else:
        call.dst = MemberMatrix(dst, rows.n, 1, MEMBER_I32)
        _ = member_call(slot[].plans[u].member, MEMBER_OP_PREDICT, call)


def _label_block(slot: _SlotPtr, u: Int, state: Int, rows: VoteRows, held: List[Int32],
                 mut keep: List[List[Float32]]) raises -> VoteMember:
    """A classifier stacked with stack_method='predict' (lane mf-meta3b):
    its predicted codes over `rows` as one float32 column, the meta feature
    of scikit-learn's encoded fit (`held` non-empty: a re-coded fold's
    member, whose code c is the ensemble code held[c]). The block is
    answered here, so the RUN reads it as a door block."""
    var n = rows.n
    var codes = List[Int32](length=max(n, 1), fill=Int32(0))
    _codes_into(slot, u, state, rows, Int(codes.unsafe_ptr()))
    var words = List[Float32](length=max(n, 1), fill=0.0)
    for i in range(n):  # small-loop(one widening pass over one member's codes)
        var c = Int(codes[i])
        if len(held) > 0 and c >= 0 and c < len(held):
            c = Int(held[c])
        words[i] = Float32(c)
    _ = len(codes)
    keep.append(words^)
    return VoteMember(Int(keep[len(keep) - 1].unsafe_ptr()), 1, 0, 1.0)


def _stack_members(slot: _SlotPtr, states: List[Int], offset: Int, rows: VoteRows, held: List[Int32],
                   mut keep: List[List[Float32]]) raises -> List[VoteMember]:
    """The stacked members (member u over `states[offset + u]`) as the
    RUN's VoteMembers over `rows`: a classifier's predict as its label block
    (`_label_block`), every other operation a native call; `held`
    non-empty: the fold's classes (each member's class columns)."""
    var bases = len(slot[].stack_kinds)
    var classifier = slot[].kind == META_STACK_CLASSIFIER
    var members = List[VoteMember](capacity=bases)
    for u in range(bases):  # driver-loop: over the stacked members
        var state = states[offset + u]
        if classifier and slot[].stack_kinds[u] == STACK_PREDICT:
            members.append(_label_block(slot, u, state, rows, held, keep))
            continue
        var m = _member_of(slot, u, state, _stack_operation(slot[].stack_kinds[u]), 1.0)
        if len(held) > 0:
            m.class_cols = Int(held.unsafe_ptr())
        members.append(m)
    return members^


def _stack_counts(slot: _SlotPtr, k: Int, held: List[Int32]) -> List[Int]:
    """Each stacked member's fitted class count (0 for STACK_PREDICT)."""
    var bases = len(slot[].stack_kinds)
    var counts = List[Int](length=bases, fill=0)
    for u in range(bases):  # driver-loop: over the stacked members (their class counts)
        if slot[].stack_kinds[u] != STACK_PREDICT:
            counts[u] = _held_classes(held, k)
    return counts^


def _user_folds(slot: _SlotPtr, n: Int, mut tr_at: List[Int], mut tr_n: List[Int], mut te_at: List[Int],
                mut te_n: List[Int], partition: Bool) raises:
    """A user's cv index list (the entry's typed folds) against the fit's
    `n` rows, refused as the door's `_splits` refuses it, in its words and
    order: per fold its empty side, its train then test rows outside
    [0, n); no fold; then, when `partition` (stacking's and an
    ensemble=False calibration's cross_val_predict, the door's
    `_splits(..., partition)`), a test side that is not a partition. The
    bounds of every fold on the host cores."""
    var folds = len(slot[].fold_test)
    var bad = List[Int](length=max(folds, 1), fill=0)
    var bad_at = Int(bad.unsafe_ptr())
    var tr_ptrs = List[Int](length=max(folds, 1), fill=0)
    var tr_counts = List[Int](length=max(folds, 1), fill=0)
    var te_ptrs = List[Int](length=max(folds, 1), fill=0)
    var te_counts = List[Int](length=max(folds, 1), fill=0)
    for f in range(folds):  # driver-loop: over the user's folds (their addresses)
        tr_ptrs[f] = Int(slot[].fold_train[f].unsafe_ptr())
        tr_counts[f] = len(slot[].fold_train[f])
        te_ptrs[f] = Int(slot[].fold_test[f].unsafe_ptr())
        te_counts[f] = len(slot[].fold_test[f])
    var trp = Int(tr_ptrs.unsafe_ptr())
    var trc = Int(tr_counts.unsafe_ptr())
    var tep = Int(te_ptrs.unsafe_ptr())
    var tec = Int(te_counts.unsafe_ptr())

    def _check(f: Int) {imm bad_at, imm trp, imm trc, imm tep, imm tec, imm n}:
        var flag = MutPointer[Int, MutUntrackedOrigin](unsafe_from_address=bad_at)
        var ntr = MutPointer[Int, MutUntrackedOrigin](unsafe_from_address=trc)[f]
        var nte = MutPointer[Int, MutUntrackedOrigin](unsafe_from_address=tec)[f]
        if ntr == 0 or nte == 0:
            flag[f] = 1
            return
        var tr = MutPointer[Int32, MutUntrackedOrigin](unsafe_from_address=MutPointer[Int, MutUntrackedOrigin](
            unsafe_from_address=trp)[f])
        var te = MutPointer[Int32, MutUntrackedOrigin](unsafe_from_address=MutPointer[Int, MutUntrackedOrigin](
            unsafe_from_address=tep)[f])
        for i in range(ntr):  # small-loop(one bounds pass over one fold's train rows)
            if Int(tr[i]) < 0 or Int(tr[i]) >= n:
                flag[f] = 2
                return
        for i in range(nte):  # small-loop(one bounds pass over one fold's test rows)
            if Int(te[i]) < 0 or Int(te[i]) >= n:
                flag[f] = 2
                return

    host_parallelize(_check, folds)
    for f in range(folds):  # driver-loop: over the user's folds (the first refusal in fold order)
        if bad[f] == 1:
            raise Error("every cv split needs train and test rows")
        if bad[f] == 2:
            raise Error("a cv index is outside [0, " + String(n) + ")")
    if folds == 0:
        raise Error("cv produced no splits")
    if partition:
        var seen = List[Int32](length=max(n, 1), fill=Int32(0))
        for f in range(folds):  # driver-loop: over the user's folds (the partition count)
            for i in range(te_counts[f]):  # small-loop(one count per test row: n rows in all over a partition)
                seen[Int(slot[].fold_test[f][i])] += 1
        for i in range(n):  # small-loop(the partition check: one word per row, once per fit)
            if seen[i] != 1:
                raise Error("cross_val_predict only works for partitions: every row must be in exactly one test set")
    for f in range(folds):  # driver-loop: over the user's folds (the RUN's addresses)
        tr_at.append(tr_ptrs[f])
        tr_n.append(tr_counts[f])
        te_at.append(te_ptrs[f])
        te_n.append(te_counts[f])
    _ = len(bad)
    _ = len(tr_ptrs)
    _ = len(tr_counts)
    _ = len(te_ptrs)
    _ = len(te_counts)


def _split_folds(slot: _SlotPtr, n: Int, codes: Int, k: Int, groups: MselGroupCodes, targets: Int,
                 given: Int, given_dtype: Int) raises:
    """THE SPLITTER CV (lanes mf-split4, mf-split5): the folds of the entry's
    splitter over THIS fit's `n` rows (the door's `_splits` ->
    `prep_native_split` -> bindings/msel_split_door.mojo `msel_split`, whose
    plans and fill these are: bindings/msel_split_spec_run.mojo
    `msel_spec_sides` over the int32 codes `codes` of `k` classes and the
    fit's `groups`), written as the typed folds (`fold_train` / `fold_test`,
    int32 rows), which `_user_folds` then checks as the door checks a
    split's rows (a stack's or an ensemble=False calibration's partition
    included). Each splitter warning ("The least populated class in y has
    only ...", once per plan: a repeated splitter's repeats each warn)
    becomes a fit notice of the state (core/member_notices.mojo
    MEMBER_NOTICE_SPLIT_LEAST, raised by the PACK in its words; the door
    raised it before the fits). A REGRESSOR's fit (no `codes`) hands its
    float32 `targets` (0: none): a stratified splitter stratifies on their
    values, as the splitter door does with a regressor's y, its refusal and
    warning in its words (`msel_spec_sides_targets`, lane mf-del5; the
    door kept every such fit before). `given` (lane mf-tail5): THE GIVEN y
    of those targets (core/member_abi.mojo `MemberCall.y_given`, 0: none),
    whose values are coded instead of the float32 words. No Python."""
    var sides: MselSides
    if codes == 0 and targets != 0:
        sides = msel_spec_sides_targets(slot[].split, n, targets, given, given_dtype, groups)
    else:
        sides = msel_spec_sides(slot[].split, n, codes, k, groups)
    for w in range(len(sides.least)):  # small-loop(one notice per splitter warning)
        var words = List[Float64]()
        member_notice_add(words, MEMBER_NOTICE_SPLIT_LEAST, Float64(sides.least[w]), Float64(slot[].split.k))
        slot[].notice_owner.append(0)
        slot[].notice_words.append(words^)
    var tr32 = List[List[Int32]]()
    var te32 = List[List[Int32]]()
    msel_sides_i32(sides, tr32, te32)
    _ = sides.splits()  # the narrowing read the int64 sides through their addresses
    slot[].fold_train = tr32^
    slot[].fold_test = te32^


def _no_groups() -> MselGroupCodes:
    """A fit with no groups (Stacking and CalibratedClassifierCV hand their
    splitter none, as scikit-learn's do)."""
    return MselGroupCodes()


def _call_groups(call: _CallPtr) raises -> MselGroupCodes:
    """THE SEARCH KIND's groups: the call's group ids (`MemberCall.groups`,
    int32 or int64, rows x 1) as sorted-order codes; none when not given."""
    var g = call[].groups
    if g.rows == 0:
        return MselGroupCodes()
    if g.cols != 1 or not (g.dense_host(MEMBER_I32) or g.dense_host(MEMBER_I64)):
        raise Error("TypeError: a search member reads groups as int32 or int64 ids, rows x 1")
    return msel_group_codes_run(g.data, g.dtype == MEMBER_I64, g.rows)


def _keep_states(slot: _SlotPtr, mut run: MemberSetRun, first: Int):
    """The fitted states of `run` become the slot's nested members `first`,
    `first + 1`, ... (their fit notices kept)."""
    for v in range(len(run.states)):  # driver-loop: over the set's members (their states change owner)
        slot[].states[first + v] = run.states[v]
        _take_notices(slot, first + v, run.plans[v], run.states[v])
        run.states[v] = 0


# ------------------------------------------------------------------ FIT


def _fit_vote(slot: _SlotPtr, call: _CallPtr) raises:
    var classes = call[].classes if slot[].kind == META_VOTE_CLASSIFIER else 0
    var y_dtype = MEMBER_I32 if slot[].kind == META_VOTE_CLASSIFIER else MEMBER_F32
    var weights = call[].weights.data if call[].weights.given() else 0
    var m = len(slot[].plans)
    var run = _set_of(slot, call[].x.data, call[].x.rows, call[].x.cols, y_dtype)
    for u in range(m):  # driver-loop: over the voting members (admission into the set, once each)
        _add(slot, run, u, classes, call[].y.data, weights, 0, 0)
    member_set_fit(run)
    slot[].states = List[Int](length=m, fill=0)
    _keep_states(slot, run, 0)
    slot[].classes = classes
    slot[].widths = List[Int]()


def _fit_stack(slot: _SlotPtr, call: _CallPtr) raises:
    var classifier = slot[].kind == META_STACK_CLASSIFIER
    var k = call[].classes if classifier else 0
    var y_dtype = MEMBER_I32 if classifier else MEMBER_F32
    var n = call[].x.rows
    var d = call[].x.cols
    var x = call[].x.data
    var y = call[].y.data
    var weights = call[].weights.data if call[].weights.given() else 0
    var bases = len(slot[].stack_kinds)
    var glue = slot[].glue
    var binary = k == 2
    var every = List[Int32]()  # a member that holds every class
    # THE SPLITTER CV (lane mf-split4): this fit's folds first, as the door
    # computes its splits before any member fit.
    if slot[].split.kind != MSEL_SPEC_NONE:
        _split_folds(slot, n, y if classifier else 0, k, _no_groups(), 0 if classifier else y,
                     0 if classifier else slot[].fit_given, slot[].fit_given_dtype)
    slot[].classes = k
    slot[].states = List[Int](length=bases + 1, fill=0)
    # 1. The members on every row (one set).
    var full = _set_of(slot, x, n, d, y_dtype)
    for u in range(bases):  # driver-loop: over the stacked members (admission, once each)
        _add(slot, full, u, k, y, weights, 0, 0)
    member_set_fit(full)
    _keep_states(slot, full, 0)
    _nested_outputs(slot)
    var counts = _stack_counts(slot, k, every)
    # 2. The width probe over the first rows.
    var head_keep = List[List[Float32]]()
    var head = VoteRows(x, min(n, 2), d)
    var members = _stack_members(slot, slot[].states, 0, head, every, head_keep)
    var widths = List[Int]()
    var refused = stack_place_run(glue, head, members, slot[].stack_kinds, counts, k, binary, 0, 0, 0, 0, 0,
                                  widths, 0)
    _ = len(head_keep)  # the label blocks are read through the RUN above
    if refused != VOTE_ALL_ANSWERED:
        _raise_member(members[refused], head)
    var total = 0
    for j in range(len(widths)):  # driver-loop: over the stacked members
        total += widths[j]
    var width = total + (d if slot[].flag else 0)
    slot[].widths = widths.copy()
    slot[].total = total
    var meta = List[Float64](length=max(n * width, 1), fill=0.0)
    var meta_at = Int(meta.unsafe_ptr())
    # 3. The cv folds: a user's index list (the entry's typed folds, checked
    # here), else stratified over the codes (classifier) or plain (regressor).
    var tr_at = List[Int]()
    var tr_n = List[Int]()
    var te_at = List[Int]()
    var te_n = List[Int]()
    var n_splits = slot[].folds
    var fold_counts = List[Int32](length=n_splits + 1, fill=Int32(0))
    var fold_rows = List[Int32](length=max(n_splits * n, 1), fill=Int32(0))
    var rows_at = Int(fold_rows.unsafe_ptr())
    if len(slot[].fold_test) > 0:
        _user_folds(slot, n, tr_at, tr_n, te_at, te_n, True)
    else:
        var codes_at = y if classifier else Int(fold_counts.unsafe_ptr())
        var status = xt_device_folds(glue, codes_at, rows_at, Int(fold_counts.unsafe_ptr()), n, n_splits, k)
        if status == 1:
            raise Error("ValueError: n_splits=" + String(n_splits)
                        + " cannot be greater than the number of members in each class")
        if status != 0:
            raise Error("ValueError: y codes outside [0, n_classes)")
        var last = 0
        for f in range(n_splits):  # driver-loop: over the cv folds
            if Int(fold_counts[f]) > 0:
                last = f
        for f in range(last + 1):  # driver-loop: over the cv folds (their train and test rows)
            var c = Int(fold_counts[f])
            tr_at.append(rows_at + 4 * (f * n))
            tr_n.append(n - c)
            te_at.append(rows_at + 4 * (f * n + n - c))
            te_n.append(c)
    var used = len(te_at)
    # A classifier fold whose train rows lack a class (a user's folds can) is
    # fitted on its own classes, re-coded, as the door's member fit on that
    # fold's codes; its members' columns are then placed at their classes.
    var held = List[List[Int32]]()
    var recoded = List[List[Int32]]()
    if classifier:
        _held_and_codes(y, k, n, tr_at, tr_n, held, recoded)
    var columns = List[Int32](length=max(d, 1), fill=Int32(0))
    for j in range(d):  # small-loop(the column numbers 0 .. d-1 of a fold's gather, once per fit)
        columns[j] = Int32(j)
    # Waves of folds: as many folds' train blocks as the workers use at once
    # (ceil(workers / members) folds on the CPU, one fold when a member runs on
    # a GPU), the wave rule of `_stacking_fit`.
    var host = True
    for u in range(bases):  # driver-loop: over the stacked members (their routes)
        if slot[].plans[u].route.device != DEVICE_CPU:
            host = False
    var workers = 1
    if host:
        workers = native_unit_cores()
        if slot[].cap > 0:
            workers = min(workers, slot[].cap)
    var wave = max(1, (workers + bases - 1) // max(bases, 1))
    var f0 = 0
    while f0 < used:  # driver-loop: over waves of cv folds
        var f1 = min(used, f0 + wave)
        var run = _set_of(slot, x, n, d, y_dtype)
        # The held classes were read above: the set makes no second cover pass.
        run.skip_lacking = classifier
        for f in range(f0, f1):  # driver-loop: over the wave's cv folds (their units)
            var target = y
            var classes = k
            if classifier and len(held[f]) > 0:
                target = Int(recoded[f].unsafe_ptr())
                classes = len(held[f])
            for u in range(bases):  # driver-loop: over the stacked members
                _add(slot, run, u, classes, target, weights, tr_at[f], tr_n[f])
        try:
            member_set_fit(run)
            for v in range(len(run.states)):  # driver-loop: over the wave's units (their fit notices)
                _take_notices(slot, v % bases, run.plans[v], run.states[v])
            for f in range(f0, f1):  # driver-loop: over the wave's cv folds (test outputs)
                var c = te_n[f]
                var test_at = te_at[f]
                var fold_held = held[f].copy() if classifier else List[Int32]()
                # The members' calls read their class count from the slot's plans in place.
                for u in range(bases):  # driver-loop: over the stacked members (the fold's class count)
                    slot[].plans[u].classes = _held_classes(fold_held, k) if classifier else 0
                var xte = List[Float32](length=max(c * d, 1), fill=0.0)
                _ = xt_gather_f32(glue, x, test_at, Int(columns.unsafe_ptr()), Int(xte.unsafe_ptr()), n, d, c, d)
                var rows = VoteRows(Int(xte.unsafe_ptr()), c, d)
                var fold_keep = List[List[Float32]]()
                var fold_members = _stack_members(slot, run.states, (f - f0) * bases, rows, fold_held, fold_keep)
                var fold_counts_of = _stack_counts(slot, k, fold_held)
                var got = List[Int]()
                var missed = stack_place_run(glue, rows, fold_members, slot[].stack_kinds, fold_counts_of, k, binary,
                                             meta_at, n, width, test_at, 0, got, 0)
                _ = len(fold_keep)  # the label blocks are read through the RUN above
                _ = len(fold_held)  # the class columns are read through the RUN above
                if missed != VOTE_ALL_ANSWERED:
                    _raise_member(fold_members[missed], rows)
                for j in range(bases):  # driver-loop: over the stacked members
                    if got[j] != widths[j]:
                        raise Error("ValueError: a member's output width changed between folds")
                _ = len(xte)  # read through the RUN above
        finally:
            # scikit-learn keeps no fold estimator: the wave's states and
            # gathered train blocks go here; the full members' class count
            # comes back to the slot's plans.
            member_set_run_release(run)
            for u in range(bases):  # driver-loop: over the stacked members (their class count)
                slot[].plans[u].classes = k
        f0 = f1
    _ = len(recoded)  # the re-coded targets are read through the fold sets above
    # 4. Passthrough: X after the members' columns.
    if slot[].flag:
        var order = List[Int32](length=max(n, 1), fill=Int32(0))
        for i in range(n):  # small-loop(the row numbers 0 .. n-1 the scatter places, once per fit)
            order[i] = Int32(i)
        _ = xt_scatter(glue, meta_at, x, Int(order.unsafe_ptr()), n, width, n, d, total)
        _ = len(order)
    # 5. The final estimator on the float32 meta features.
    slot[].meta32 = List[Float32](length=max(n * width, 1), fill=0.0)
    for i in range(n * width):  # small-loop(one rounding pass of the meta features, as the door's float32 admission)
        slot[].meta32[i] = meta[i].cast[DType.float32]()
    _ = len(meta)
    _ = len(fold_rows)  # the fold draws are read through the fold sets above
    _ = len(fold_counts)
    _ = len(columns)
    var last_set = _set_of(slot, Int(slot[].meta32.unsafe_ptr()), n, width, y_dtype)
    _add(slot, last_set, bases, k, y, weights, 0, 0)
    member_set_fit(last_set)
    _keep_states(slot, last_set, bases)


def _fit_ovr(slot: _SlotPtr, call: _CallPtr) raises:
    """`_ovr_fit`: one binary unit on the codes (k == 2), else one per class
    on its indicator (`xt_indicator_codes`, the door's `_indicator`), every
    unit two classes, fitted as ONE set."""
    var k = call[].classes
    if k < 2:
        raise Error("y has fewer than 2 classes")
    var n = call[].x.rows
    var d = call[].x.cols
    var x = call[].x.data
    var y = call[].y.data
    var units = 1 if k == 2 else k
    _expand(slot, units)
    slot[].targets = List[List[Int32]]()
    if k > 2:
        for j in range(k):  # driver-loop: over the classes (one native indicator each)
            var yj = List[Int32](length=max(n, 1), fill=Int32(0))
            _ = xt_indicator_codes(slot[].glue, y, Int(yj.unsafe_ptr()), 0, n, j)
            slot[].targets.append(yj^)
    var run = _set_of(slot, x, n, d, MEMBER_I32)
    for u in range(units):  # driver-loop: over the units (admission into the set, once each)
        var target = y if k == 2 else Int(slot[].targets[u].unsafe_ptr())
        _add(slot, run, u, 2, target, 0, 0, 0)
    member_set_fit(run)
    slot[].states = List[Int](length=units, fill=0)
    _keep_states(slot, run, 0)
    slot[].classes = k


def _own_drawn(slot: _SlotPtr, drawn: List[Bool]) raises:
    """THE LIFETIME RULE: every kept unit fitted on a drawn block (a fold's
    train rows, a bag's draw) owns its rows before the set's gathered
    blocks, the draws and the re-coded targets go."""
    for u in range(len(slot[].states)):  # driver-loop: over the kept units (one ownership question each)
        if u < len(drawn) and drawn[u] and slot[].states[u] != 0:
            member_own(slot[].plans[u], slot[].states[u])


def _fit_cal(slot: _SlotPtr, call: _CallPtr) raises:
    """`_cal_native`: the folds (an int cv: the stratified folds of
    `xt_device_folds`; THE SPLITTER CV, lane mf-split4b: our KFold /
    StratifiedKFold over this fit's rows, `_split_folds`; a user's index
    list: the entry's typed folds, checked by `_user_folds` as the door's
    `_splits` checks them, a partition only for ensemble=False), one unit
    per fold over its train rows (and one over every row for
    ensemble=False), `calibrated_fit_run`, then the kept pairs and their
    calibrators. A failing fit raises the first failing fold's words in
    fold order (GaussianProcessClassifier's one-class words with the label
    suffix its fit door adds, `_cal_fit_words`)."""
    var k = call[].classes
    if k < 2:
        raise Error("y has fewer than 2 classes")
    var n = call[].x.rows
    var d = call[].x.cols
    var x = call[].x.data
    var y = call[].y.data
    var glue = slot[].glue
    var sigmoid = slot[].words[META_CAL_SIGMOID] != 0.0
    var ensemble = slot[].words[META_CAL_ENSEMBLE] != 0.0
    var decision = slot[].words[META_CAL_DECISION] != 0.0
    # scikit-learn's two cv refusals, before any split or fit (lane mf-del5;
    # the door's one body, `cal_cv_check`): the fold count of an exact int
    # cv or of a splitter that stores `n_splits`, and a LeaveOneOut.
    var n_folds = 0
    if slot[].words[META_CAL_INT_CV] != 0.0:
        n_folds = slot[].folds
    elif msel_spec_has_n_splits(slot[].split.kind):
        n_folds = slot[].split.k
    cal_cv_check(y, n, k, n_folds, slot[].split.kind == MSEL_SPEC_LOO)
    # THE SPLITTER CV: this fit's folds first, as the door computes its
    # splits before any member fit.
    if slot[].split.kind != MSEL_SPEC_NONE:
        _split_folds(slot, n, y, k, _no_groups(), 0, 0, 0)
    var listed = len(slot[].fold_test) > 0
    var tr_at = List[Int]()
    var tr_n = List[Int]()
    var te_at = List[Int]()
    var te_n = List[Int]()
    var n_splits = 0 if listed else slot[].folds
    var fold_counts = List[Int32](length=n_splits + 1, fill=Int32(0))
    var fold_rows = List[Int32](length=max(n_splits * n, 1), fill=Int32(0))
    var rows_at = Int(fold_rows.unsafe_ptr())
    if listed:
        _user_folds(slot, n, tr_at, tr_n, te_at, te_n, not ensemble)
    else:
        var status = xt_device_folds(glue, y, rows_at, Int(fold_counts.unsafe_ptr()), n, n_splits, k)
        if status == 1:
            raise Error("n_splits=" + String(n_splits)
                        + " cannot be greater than the number of members in each class")
        if status != 0:
            raise Error("y codes outside [0, n_classes)")
        var last = 0
        for f in range(n_splits):  # driver-loop: over the cv folds
            if Int(fold_counts[f]) > 0:
                last = f
        for f in range(last + 1):  # driver-loop: over the cv folds (their train and test rows)
            var c = Int(fold_counts[f])
            tr_at.append(rows_at + 4 * (f * n))
            tr_n.append(n - c)
            te_at.append(rows_at + 4 * (f * n + n - c))
            te_n.append(c)
    var used = len(te_at)
    var units = used if ensemble else used + 1
    _expand(slot, units)
    var run = _set_of(slot, x, n, d, MEMBER_I32)
    for f in range(used):  # driver-loop: over the cv folds (admission, once each)
        _add(slot, run, f, k, y, 0, tr_at[f], tr_n[f])
    if not ensemble:
        _add(slot, run, used, k, y, 0, 0, 0)
    var operation = MEMBER_OP_DECISION if decision else MEMBER_OP_PREDICT_PROBA
    var plan = CalFitPlan(x, n, d, y, k, operation, sigmoid, ensemble, _CALIB_FAST_HOST_FOLDS)
    for f in range(used):  # driver-loop: over the cv folds (their test rows)
        plan.fold(te_at[f], te_n[f])
    # THE UNLABELED MARKER (lane mf-meta5b): a LabelPropagation / LabelSpreading base admitted with the
    # caller's code of -1 in its marker word. ensemble=True: every fold's fit reads y as given (its classes_
    # exclude -1). ensemble=False: scikit-learn's cross_val_predict encodes y first, so its fold fits see -1
    # as one more class (the fold units' marker word cleared), and only the fit over every row reads it.
    var marker = _cal_unit_marker(run.plans[0])
    if marker >= 0:
        plan.markers = List[Int](length=units, fill=marker)
        if not ensemble:
            for f in range(used):  # driver-loop: over the cv folds (their marker words cleared)
                run.plans[f].params[_LP_MARKER] = -1.0
                plan.markers[f] = -1
    var res = CalFitOut()
    # A refused scoring call raises its member's words (the RUN released every state).
    var answered = calibrated_fit_run(glue, run, plan, res)
    if answered != CAL_ALL_ANSWERED:
        var first = member_set_first_failure(run)
        var words = String("RuntimeError: mojolearn: a CalibratedClassifierCV member's fit failed")
        if first >= 0:
            words = run.errors[first].copy()
            if (first < plan.folds() and len(plan.fold_classes[first]) == 1
                    and run.plans[first].family == MEMBER_FAMILY_GP and words.endswith("got 1 class")):
                words += " (only class " + String(Int(plan.fold_classes[first][0])) + " is present)"
        member_set_run_release(run)
        raise Error(words)
    # The kept pairs: every fold's unit (ensemble), else only the unit over every row.
    var kept = List[Int]()
    if ensemble:
        for f in range(used):  # driver-loop: over the cv folds
            kept.append(f)
    else:
        kept.append(used)
    var pairs = len(kept)
    var plans = List[MemberPlan](capacity=pairs)
    var masks = List[Int](capacity=pairs)
    var drawn = List[Bool](capacity=pairs)
    slot[].states = List[Int](length=pairs, fill=0)
    slot[].unit_held = List[List[Int32]](capacity=pairs)
    for i in range(pairs):  # driver-loop: over the kept pairs (their states change owner)
        var u = kept[i]
        plans.append(run.plans[u].copy())
        masks.append(slot[].masks[u])
        drawn.append(u < used)
        slot[].states[i] = run.states[u]
        run.states[u] = 0
        slot[].unit_held.append(plan.fold_classes[u].copy() if u < plan.folds() else plan.full_held.copy())
    # Every unit's fit notices, in unit order (a discarded fold's with the first pair's context: one base class).
    for u in range(len(run.states)):  # driver-loop: over the units (their fit notices)
        var owner = 0
        for i in range(pairs):  # small-loop(the kept pairs)
            if kept[i] == u:
                owner = i
        var state = slot[].states[owner] if run.states[u] == 0 else run.states[u]
        _take_notices(slot, owner, run.plans[u], state)
    slot[].plans = plans^
    slot[].masks = masks^
    try:
        _own_drawn(slot, drawn)
    finally:
        # The discarded fold units (ensemble=False) and the gathered blocks go.
        member_set_run_release(run)
    _ = plan.folds()  # its re-coded targets were read through the fits and the ownership above
    _ = len(fold_rows)  # the fold rows are read through the fits and the ownership above
    _ = len(fold_counts)
    _ = len(slot[].fold_train)  # a listed or split cv's rows, read through the fits and the ownership above
    slot[].cal_iso = res.isotonic.copy()
    slot[].cal_a = res.a.copy()
    slot[].cal_b = res.b.copy()
    slot[].cal_m = res.m.copy()
    slot[].cal_kx = res.knots_x.copy()
    slot[].cal_ky = res.knots_y.copy()
    slot[].pair_first = List[Int](capacity=pairs)
    slot[].pair_count = List[Int](capacity=pairs)
    if ensemble:
        for f in range(used):  # driver-loop: over the cv folds (their calibrators)
            slot[].pair_first.append(res.first[f])
            slot[].pair_count.append(res.count[f])
    else:
        slot[].pair_first.append(res.final_first)
        slot[].pair_count.append(res.final_count)
    slot[].classes = k


def _bag_count(is_int: Bool, value: Float64, total: Int, name: String) raises -> Int:
    """`_BaggingBase._count`: an int as it is, a fraction of `total`
    truncated (its domain (0, 1] proved by the UNPACK); the resolved count's
    refusal in the door's words."""
    var k = Int(value) if is_int else Int(value * Float64(total))
    return boost_count_check(name, k, total)


def _seed_words(mut plan: MemberPlan, seed: Int) raises:
    """A Bagging unit's member seed in its forest seed slot words (an exact
    nonnegative int: both readings, no flag; the door's `_clone_from`)."""
    if len(plan.params) < ADA_MEMBER_SEED_WORD + 3:
        raise Error("member ABI: a Bagging member's base has no seed slot")
    plan.params[ADA_MEMBER_SEED_WORD] = bitcast[DType.float64](Int64(seed))
    plan.params[ADA_MEMBER_SEED_WORD + 1] = Float64(seed)
    plan.params[ADA_MEMBER_SEED_WORD + 2] = 0.0


def _cw_tail_words(words: List[Float64], at: Int) -> Int:
    """The class-weight tail's length at `at`: its head, K weights and, for a
    dict, K named flags (lane mf-split4b: 1.0 where the dict names that
    class; the per-member route's absent-label refusal reads them)."""
    var count = max(Int(words[at + META_CW_K]), 0)
    if Int(words[at + META_CW_MODE]) == META_CW_DICT:
        return META_CW_HEAD + 2 * count
    return META_CW_HEAD + count


def _cw_mode(words: List[Float64], at: Int) -> Int:
    """The class-weight tail's mode at `at` (META_CW_NONE when absent)."""
    if len(words) < at + META_CW_HEAD:
        return META_CW_NONE
    return Int(words[at + META_CW_MODE])


def _tree_class_rows(words: List[Float64], at: Int, codes: Int, n: Int, k: Int, mut rows: List[Float32]) raises -> Int:
    """FIT: the float32 class_weight row weights of a tree, forest or
    extra-trees fit over the int32 codes `codes` (n rows, k classes) from
    the class-weight tail at `at` (META_CW_*), the forest door's own rows
    (bindings/class_weight_rows_boundary_pack.mojo
    `class_weight_rows_admitted[CLASS_WEIGHT_TREES]`, as
    bindings/trees_public_inputs_door.mojo `trees_class_rows_only` calls it):
    for 'balanced' the class counts (LABEL_WEIGHTS_COUNTS), widened to
    int64, and scikit-learn's ratios (`class_ratio_f64`, mode 0); for a dict
    the weights read at UNPACK; then the per-class float32 plan
    (`class_weight_rows_plan`) and the row gather (LABEL_WEIGHTS_ROWS), with
    the x_trees image's own label-weights kernels (the descriptor the door
    reaches through its provider: `label_weights_native_v1`). Answers the
    rows' address (in `rows`), or 0 when every class weight is 1 (the door
    then hands the fit no weights). No Python."""
    var mode = _cw_mode(words, at)
    if mode == META_CW_NONE:
        return 0
    var driver = LabelWeightsDriver(label_weights_native_v1)
    var inputs = ClassWeightInputs()
    var balanced = mode == META_CW_BALANCED
    if balanced:
        var counts = List[Int32](length=max(k, 1), fill=Int32(0))
        _ = driver.run(LABEL_WEIGHTS_COUNTS, LabelWeightsRequest(codes if n > 0 else 0, Int(counts.unsafe_ptr()), 0,
                                                                  n, k))
        var wide = List[Int64](length=max(k, 1), fill=Int64(0))
        for c in range(k):  # small-loop(k class counts widened to int64, as the door's cast)
            wide[c] = Int64(counts[c])
        inputs.values = List[Float64](length=k, fill=0.0)
        if k > 0 and class_ratio_f64(_I64P(unsafe_from_address=Int(wide.unsafe_ptr())),
                                     _F64P(unsafe_from_address=Int(inputs.values.unsafe_ptr())), k, n, 0) != 0:
            raise Error("ZeroDivisionError: division by zero")
        _ = len(counts)
        _ = len(wide)
    else:
        var table = Int(words[at + META_CW_K])
        if table < 0:
            raise Error("TypeError: mojolearn: a member whose tree has a class_weight dict is fitted only by a "
                        + "caller that hands over its labels")
        if table != k:
            raise Error("member ABI: the class_weight table has " + String(table) + " entries for " + String(k)
                        + " classes")
        for c in range(k):  # small-loop(the k dict weights read at UNPACK)
            inputs.values.append(words[at + META_CW_HEAD + c])
        inputs.all_one = words[at + META_CW_ALL_ONE] != 0.0
    var plan = class_weight_rows_plan(inputs, balanced)
    if plan.all_one:
        return 0
    rows = List[Float32](length=max(n, 1), fill=0.0)
    if n > 0:
        _ = driver.run(LABEL_WEIGHTS_ROWS, LabelWeightsRequest(codes, Int(plan.words.unsafe_ptr()),
                                                               Int(rows.unsafe_ptr()), n, k))
    _ = len(plan.words)  # the gather above read the plan's words through their address
    return Int(rows.unsafe_ptr())


def _bag_class_rows(words: List[Float64], at: Int, y: Int, k: Int, n: Int, weights: Int, draw_at: List[Int],
                    draw_n: List[Int], held: List[List[Int32]], recoded: List[List[Int32]],
                    mut unit_w: List[List[Float32]], mut unit_w_at: List[Int]) raises:
    """THE PER-MEMBER ROUTE OF A TREE WITH A CLASS_WEIGHT (lane mf-split4b):
    each unit's float32 row weights as the tree's fit door makes them over
    the rows the door's loop gives it (`_bagging_fit_bags`: the draw's codes
    and, weighted, the draw's sample weights; bindings/
    trees_public_inputs_door.mojo `trees_class_rows_only` /
    `trees_weighted_rows`): the classes its draw holds (a lacking draw's
    own classes, as the door's member encodes them), 'balanced' over the
    draw's class counts (scikit-learn's ratios, `class_ratio_f64` over the
    draw's rows), a dict's weights of those classes, refused in the door's
    words when it names a class the draw lacks and leaves a held class
    unweighted (`class_weight_dict_refused`) ("class_weight contains
    labels absent from y", `class_weight_dictionary_values`); the per-class
    float32 plan (`class_weight_rows_plan`) gathered at each drawn row,
    times the caller's float32 weight (one float32 product per row, the
    door's LABEL_WEIGHTS_MULTIPLY) when weighted. A unit whose class
    weights are all one reads the caller's weights (`unit_w_at` stays).
    Each unit's words are `n` long (the set gathers them with the draw); the
    counts and the rows on the host cores, one task per unit. No Python."""
    var units = len(draw_at)
    var balanced = _cw_mode(words, at) == META_CW_BALANCED
    var table = Int(words[at + META_CW_K])
    if not balanced:
        if table < 0:
            raise Error("TypeError: mojolearn: a member whose tree has a class_weight dict is fitted only by a "
                        + "caller that hands over its labels")
        if table != k:
            raise Error("member ABI: the class_weight table has " + String(table) + " entries for " + String(k)
                        + " classes")
    # 1. The draws' class counts (in the ensemble codes), one task per unit.
    var counts = List[Int64](length=max(units * k, 1), fill=Int64(0))
    var counts_at = Int(counts.unsafe_ptr())
    var draws_at = Int(draw_at.unsafe_ptr())
    var draws_n = Int(draw_n.unsafe_ptr())

    def _count(u: Int) {imm counts_at, imm draws_at, imm draws_n, imm y, imm k, imm n}:
        var draw = MutPointer[Int, MutUntrackedOrigin](unsafe_from_address=draws_at)[u]
        var m = MutPointer[Int, MutUntrackedOrigin](unsafe_from_address=draws_n)[u]
        var rows = m if draw != 0 else n
        var picks = _I32P(unsafe_from_address=draw)
        var codes = _I32P(unsafe_from_address=y)
        var dst = _I64P(unsafe_from_address=counts_at + 8 * u * k)
        for i in range(rows):  # small-loop(one pass over one unit's drawn rows)
            var c = Int(codes[Int(picks[i]) if draw != 0 else i])
            if c >= 0 and c < k:
                dst[c] += 1

    if balanced:
        host_parallelize(_count, units)
    # 2. Each unit's per-class plan (k words each, on the calling thread: the
    # plan steps split their own work).
    var plans = List[List[Float32]](capacity=units)
    var weighted = List[Bool](capacity=units)
    for u in range(units):  # driver-loop: over the member estimators (their class plans)
        var classes = _held_classes(held[u], k)
        var inputs = ClassWeightInputs()
        if balanced:
            var wide = List[Int64](length=max(classes, 1), fill=Int64(0))
            for j in range(classes):  # small-loop(the unit's held classes' counts)
                var c = Int(held[u][j]) if len(held[u]) > 0 else j
                wide[j] = counts[u * k + c]
            var rows = draw_n[u] if draw_at[u] != 0 else n
            inputs.values = List[Float64](length=classes, fill=0.0)
            if classes > 0 and class_ratio_f64(_I64P(unsafe_from_address=Int(wide.unsafe_ptr())),
                                               _F64P(unsafe_from_address=Int(inputs.values.unsafe_ptr())), classes,
                                               rows, 0) != 0:
                raise Error("ZeroDivisionError: division by zero")
            _ = len(wide)
        else:
            if len(held[u]) > 0:
                # the door's member sees only the draw's classes: a dict naming another is refused when
                # it also leaves a held class unweighted (`class_weight_dict_refused`, the door's rule)
                var mark = List[Bool](length=max(k, 1), fill=False)
                var unweighted = 0
                for j in range(len(held[u])):  # small-loop(the unit's held classes)
                    mark[Int(held[u][j])] = True
                    if words[at + META_CW_HEAD + k + Int(held[u][j])] == 0.0:
                        unweighted += 1
                var unknown = 0
                for c in range(k):  # small-loop(the dict's named classes)
                    if words[at + META_CW_HEAD + k + c] != 0.0 and not mark[c]:
                        unknown += 1
                if class_weight_dict_refused(unknown, unweighted):
                    raise Error("class_weight contains labels absent from y")
            for j in range(classes):  # small-loop(the dict weights of the unit's classes)
                var c = Int(held[u][j]) if len(held[u]) > 0 else j
                var value = words[at + META_CW_HEAD + c]
                inputs.append(value, value == 1.0)
        var plan = class_weight_rows_plan(inputs, balanced)
        weighted.append(not plan.all_one)
        plans.append(plan.words.copy() if not plan.all_one else List[Float32]())
    # 3. The rows: unit u's plan word at each drawn row (times the caller's weight).
    unit_w = List[List[Float32]](capacity=units)
    for u in range(units):  # driver-loop: over the member estimators (their row words)
        unit_w.append(List[Float32](length=max(n, 1), fill=0.0) if weighted[u] else List[Float32]())
    var plan_at = List[Int](length=max(units, 1), fill=0)
    var dst_at = List[Int](length=max(units, 1), fill=0)
    var codes_at = List[Int](length=max(units, 1), fill=y)
    for u in range(units):  # driver-loop: over the member estimators (their final addresses)
        if weighted[u]:
            plan_at[u] = Int(plans[u].unsafe_ptr())
            dst_at[u] = Int(unit_w[u].unsafe_ptr())
            unit_w_at[u] = dst_at[u]
            if len(held[u]) > 0:
                codes_at[u] = Int(recoded[u].unsafe_ptr())
    var plan_list = Int(plan_at.unsafe_ptr())
    var dst_list = Int(dst_at.unsafe_ptr())
    var code_list = Int(codes_at.unsafe_ptr())

    def _rows(u: Int) {imm plan_list, imm dst_list, imm code_list, imm draws_at, imm draws_n, imm weights, imm n}:
        var dst = MutPointer[Int, MutUntrackedOrigin](unsafe_from_address=dst_list)[u]
        if dst == 0:
            return
        var plan = _F32P(unsafe_from_address=MutPointer[Int, MutUntrackedOrigin](unsafe_from_address=plan_list)[u])
        var codes = _I32P(unsafe_from_address=MutPointer[Int, MutUntrackedOrigin](unsafe_from_address=code_list)[u])
        var out = _F32P(unsafe_from_address=dst)
        var draw = MutPointer[Int, MutUntrackedOrigin](unsafe_from_address=draws_at)[u]
        var m = MutPointer[Int, MutUntrackedOrigin](unsafe_from_address=draws_n)[u]
        var rows = m if draw != 0 else n
        var picks = _I32P(unsafe_from_address=draw)
        var given = _F32P(unsafe_from_address=weights)
        for i in range(rows):  # small-loop(one pass over one unit's drawn rows)
            var r = Int(picks[i]) if draw != 0 else i
            var value = plan[Int(codes[r])]
            out[r] = value * given[r] if weights != 0 else value

    host_parallelize(_rows, units)
    _ = len(counts)  # read through the tasks above
    _ = len(plans)
    _ = len(plan_at)
    _ = len(dst_at)
    _ = len(codes_at)
    _ = len(held)
    _ = len(recoded)
    _ = len(draw_at)
    _ = len(draw_n)


def _fit_bag(slot: _SlotPtr, call: _CallPtr) raises:
    """`_bagging_fit_bags`' per-member route as ONE set: unit u's draws
    (`xt_sample_indices`, streams 2u and 2u + 1) and seed `_sub_seed(seed,
    u)`, the door's order and words; then the out-of-bag outputs
    (oob_score)."""
    var classifier = slot[].kind == META_BAG_CLASSIFIER
    var k = call[].classes if classifier else 0
    if classifier and k < 2:
        raise Error("y has fewer than 2 classes")
    var n = call[].x.rows
    var d = call[].x.cols
    var x = call[].x.data
    var y = call[].y.data
    var weights = call[].weights.data if call[].weights.given() else 0
    var glue = slot[].glue
    var count = Int(slot[].words[META_BAG_COUNT])
    var seed = Int(slot[].words[META_BAG_SEED])
    var n_rows = _bag_count(slot[].words[META_BAG_ROWS_INT] != 0.0, slot[].words[META_BAG_ROWS], n, "max_samples")
    var n_feat = _bag_count(slot[].words[META_BAG_COLS_INT] != 0.0, slot[].words[META_BAG_COLS], d, "max_features")
    var bootstrap = slot[].words[META_BAG_BOOTSTRAP] != 0.0
    var bootstrap_features = slot[].words[META_BAG_BOOTSTRAP_FEATURES] != 0.0
    var all_rows = n_rows == n and not bootstrap
    var all_cols = n_feat == d and not bootstrap_features
    # THE BATCHED ROUTE (`_bagging_batched_ok`, decided per fit as the door
    # decides it): no weights, every feature, and bootstrap or every row.
    if (slot[].words[META_BAG_BATCHED] != 0.0 and weights == 0 and n_feat == d and not bootstrap_features
            and (bootstrap or n_rows == n)):
        _fit_bag_batched(slot, call, k, n_rows, bootstrap)
        return
    _expand(slot, count)
    slot[].unit_rows = List[List[Int32]](capacity=count)
    slot[].unit_cols = List[List[Int32]](capacity=count)
    var draw_at = List[Int](capacity=count)
    var draw_n = List[Int](capacity=count)
    for u in range(count):  # driver-loop: over the member estimators (their draws and seeds, the door's order)
        var rows = List[Int32]()
        if not all_rows:
            rows = List[Int32](length=max(n_rows, 1), fill=Int32(0))
            _ = xt_sample_indices(glue, Int(rows.unsafe_ptr()), n, n_rows, 1 if bootstrap else 0, seed, 2 * u)
        var cols = List[Int32]()
        if not all_cols:
            cols = List[Int32](length=max(n_feat, 1), fill=Int32(0))
            _ = xt_sample_indices(glue, Int(cols.unsafe_ptr()), d, n_feat, 1 if bootstrap_features else 0, seed,
                                  2 * u + 1)
        _seed_words(slot[].plans[u], _sub_seed(seed, u))
        slot[].unit_rows.append(rows^)
        slot[].unit_cols.append(cols^)
    for u in range(count):  # driver-loop: over the member estimators (their draws' addresses, now final)
        draw_at.append(0 if all_rows else Int(slot[].unit_rows[u].unsafe_ptr()))
        draw_n.append(0 if all_rows else n_rows)
    var held = List[List[Int32]]()
    var recoded = List[List[Int32]]()
    if classifier:
        _held_and_codes(y, k, n, draw_at, draw_n, held, recoded)
    # A CLASS_WEIGHT ON THE TREE (lane mf-split4b): each unit's row weights,
    # as the tree's fit door makes them over the rows it was given
    # (`_bag_class_rows`); else every unit reads the caller's weights.
    var unit_w = List[List[Float32]]()
    var unit_w_at = List[Int](length=max(count, 1), fill=weights)
    if classifier and _cw_mode(slot[].words, META_BAG_WORDS) != META_CW_NONE:
        _bag_class_rows(slot[].words, META_BAG_WORDS, y, k, n, weights, draw_at, draw_n, held, recoded, unit_w,
                        unit_w_at)
    var run = _set_of(slot, x, n, d, MEMBER_I32 if classifier else MEMBER_F32)
    # A classifier's held classes were read above: no second cover pass.
    run.skip_lacking = classifier
    for u in range(count):  # driver-loop: over the member estimators (admission, once each)
        var target = y
        var classes = 0
        if classifier:
            classes = k
            if len(held[u]) > 0:
                target = Int(recoded[u].unsafe_ptr())
                classes = len(held[u])
        _add_drawn(slot, run, u, classes, target, unit_w_at[u], draw_at[u], draw_n[u],
                   0 if all_cols else Int(slot[].unit_cols[u].unsafe_ptr()), 0 if all_cols else n_feat)
    member_set_fit(run)
    slot[].states = List[Int](length=count, fill=0)
    _keep_states(slot, run, 0)
    var drawn = List[Bool](length=count, fill=not (all_rows and all_cols))
    try:
        _own_drawn(slot, drawn)
    finally:
        member_set_run_release(run)
    _ = len(recoded)  # the re-coded targets are read through the fits and the ownership above
    _ = len(unit_w)  # the units' class rows are read through the fits and the ownership above
    _ = len(unit_w_at)
    slot[].unit_held = held^
    slot[].classes = k
    _nested_outputs(slot)
    if slot[].words[META_BAG_OOB] != 0.0:
        _bag_oob(slot, call, draw_at, draw_n, n_rows)
    _ = len(draw_at)
    # Each unit owns its rows now and the out-of-bag rows are taken: the row
    # draws are not kept (the door keeps none).
    slot[].unit_rows = List[List[Int32]]()


def _slot_words_float(mut plan: MemberPlan, index: Int, value: Float64) raises:
    """Forest parameter slot `index` of a forest plan's words as the
    admission writes a Python float (bindings/forest_config_run.mojo
    `_slot_float`: both readings, the integer view truncated, no flag)."""
    var at = FM_HEAD + 3 * index
    if len(plan.params) < at + 3:
        raise Error("member ABI: a Bagging member's batched forest has no slot " + String(index))
    var whole = 0
    if not isnan(value) and value > -9.0e18 and value < 9.0e18:
        whole = Int(value)
    plan.params[at] = bitcast[DType.float64](Int64(whole))
    plan.params[at + 1] = value
    plan.params[at + 2] = 0.0


def _slot_words_int(mut plan: MemberPlan, index: Int, value: Int) raises:
    """Forest parameter slot `index` as an exact int (`_slot_int`)."""
    var at = FM_HEAD + 3 * index
    if len(plan.params) < at + 3:
        raise Error("member ABI: a Bagging member's batched forest has no slot " + String(index))
    plan.params[at] = bitcast[DType.float64](Int64(value))
    plan.params[at + 1] = Float64(value)
    plan.params[at + 2] = 0.0


def _fit_bag_batched(slot: _SlotPtr, call: _CallPtr, k: Int, n_rows: Int, bootstrap: Bool) raises:
    """THE BATCHED ROUTE (`_bagging_fit_batched`, a measured winner kept as
    it is: TOMBSTONE MOJOLEARN_TREES_FAST_HR2_BAGGING_MEMBERS): the members
    as ONE forest fit, the forest the door builds from the decision-tree
    base (the entry's second plan, admitted by the boundary with those
    settings), its max_samples the fraction n_rows / n when bootstrap (the
    door's `_F(n_rows / n)`, written as the door's float) and, for a
    DecisionTreeRegressor base left at n_bins=None, its image's bin count
    (`_dt_tier_bins`, XF_DT_DEFAULT_BINS). One set of one member: X, y and
    the classes as given (no weights: the route takes none)."""
    var n = call[].x.rows
    var plan = slot[].base_plans[1].copy()
    var fraction = 0.0
    if bootstrap:
        fraction = Float64(n_rows) / Float64(n)
        _slot_words_float(plan, FOREST_RF_SLOT_MAX_SAMPLES, fraction)
    var bins = 0
    if slot[].words[META_BAG_IMAGE_BINS] != 0.0:
        bins = xf_dt_default_bins(entry_ref_gil_free(plan.route.entry))
        _slot_words_int(plan, FOREST_RF_SLOT_BINS, bins)
    slot[].plans = [plan^]
    slot[].masks = [slot[].base_masks[1]]
    # The tree's class_weight (lane mf-split4): the forest's row weights over
    # the codes, as the forest door computes them for its own fit (0: none).
    var cw_rows = List[Float32]()
    var cw = 0
    if k > 0:
        cw = _tree_class_rows(slot[].words, META_BAG_WORDS, call[].y.data, n, k, cw_rows)
    var run = _set_of(slot, call[].x.data, n, call[].x.cols, MEMBER_I32 if k > 0 else MEMBER_F32)
    _add(slot, run, 0, k, call[].y.data, cw, 0, 0)
    member_set_fit(run)
    slot[].states = List[Int](length=1, fill=0)
    _keep_states(slot, run, 0)
    member_set_run_release(run)
    _ = len(cw_rows)  # the fit (and a borrowing state, until its release above) read the rows through `cw`
    slot[].batched = True
    slot[].batch_fraction = fraction
    slot[].batch_bins = bins
    slot[].unit_held = [List[Int32]()]
    slot[].unit_cols = [List[Int32]()]
    slot[].unit_oob = [List[Int32]()]
    slot[].classes = k
    _nested_outputs(slot)


def _predict_batched(slot: _SlotPtr, call: _CallPtr, operation: Int, rows: VoteRows, k: Int) raises:
    """THE BATCHED ROUTE's predictions (`_bagging_proba` / `bagging_predict`
    over `_batched`): the forest's float32 probabilities (or values) widened
    to float64, then the row argmax for a classifier's predict."""
    var n = rows.n
    var classifier = slot[].kind == META_BAG_CLASSIFIER
    var width = k if classifier else 1
    var f32 = List[Float32](length=max(n * width, 1), fill=0.0)
    var plan = slot[].plans[0].copy()
    var inner = member_call_of(plan, slot[].states[0])
    inner.x = MemberMatrix(rows.x, n, rows.d)
    inner.dst = MemberMatrix(Int(f32.unsafe_ptr()), n, width)
    _ = member_call(plan.member, MEMBER_OP_PREDICT_PROBA if classifier else MEMBER_OP_PREDICT, inner)
    var wide = List[Float64]()
    var dst = call[].dst.data
    if classifier and operation == MEMBER_OP_PREDICT:
        wide = List[Float64](length=max(n * width, 1), fill=0.0)
        dst = Int(wide.unsafe_ptr())
    var p = _F64P(unsafe_from_address=dst)
    for i in range(n * width):  # small-loop(one widening pass of the forest's float32 output, as trees_cast)
        p[i] = f32[i].cast[DType.float64]()
    _ = len(f32)
    if classifier and operation == MEMBER_OP_PREDICT:
        _ = xt_argmax_rows(slot[].glue, dst, call[].dst.data, n, k)
    _ = len(wide)


def _bag_oob(slot: _SlotPtr, call: _CallPtr, draw_at: List[Int], draw_n: List[Int], n_rows: Int) raises:
    """`_set_oob_score`: each unit's output on the rows it never drew (its
    feature subset), summed with `bag_oob_run`; a classifier's
    `oob_decision_function_` and `oob_score_` (`bag_oob_classifier_finish`),
    a regressor's `oob_prediction_` and R2 (`bag_oob_r2_run`). Into
    `oob_words`, the score last. The out-of-bag rows of every unit are kept
    (PACK writes them, `_oob_rows`)."""
    var classifier = slot[].kind == META_BAG_CLASSIFIER
    var n = call[].x.rows
    var d = call[].x.cols
    var x = call[].x.data
    var y = call[].y.data
    var k = slot[].classes if classifier else 1
    var glue = slot[].glue
    var count = len(slot[].states)
    var every = List[Int32](length=max(d, 1), fill=Int32(0))
    for j in range(d):  # small-loop(the column numbers 0 .. d-1, once per fit)
        every[j] = Int32(j)
    slot[].unit_oob = List[List[Int32]](capacity=count)
    var blocks = List[List[Float32]](capacity=count)
    var xs = List[Int](capacity=count)
    var ns = List[Int](capacity=count)
    var ds = List[Int](capacity=count)
    var oobs = List[Int](capacity=count)
    var members = List[VoteMember](capacity=count)
    for u in range(count):  # driver-loop: over the member estimators (their out-of-bag rows and blocks)
        var unseen = List[Int32](length=max(n, 1), fill=Int32(0))
        # oob_score needs bootstrap=True (the UNPACK refuses it otherwise), so every unit drew its rows.
        var m = xt_unseen_rows(glue, draw_at[u], Int(unseen.unsafe_ptr()), draw_n[u], n) if draw_at[u] != 0 else 0
        var oob = List[Int32](length=m, fill=Int32(0))
        for i in range(m):  # small-loop(one copy of a member's out-of-bag rows)
            oob[i] = unseen[i]
        _ = len(unseen)
        var width = d if len(slot[].unit_cols[u]) == 0 else len(slot[].unit_cols[u])
        var block = List[Float32]()
        if m > 0:
            block = List[Float32](length=m * width, fill=0.0)
            var cols_at = Int(every.unsafe_ptr()) if len(slot[].unit_cols[u]) == 0 else Int(
                slot[].unit_cols[u].unsafe_ptr())
            _ = xt_gather_f32(glue, x, Int(oob.unsafe_ptr()), cols_at, Int(block.unsafe_ptr()), n, d, m, width)
        slot[].unit_oob.append(oob^)
        blocks.append(block^)
        ns.append(m)
        ds.append(width)
        if m == 0:
            members.append(VoteMember(0, 0, 0, 1.0))
            continue
        var operation = MEMBER_OP_PREDICT_PROBA if classifier else MEMBER_OP_PREDICT
        var member = _member_of(slot, u, slot[].states[u], operation, 1.0)
        if classifier and len(slot[].unit_held[u]) > 0:
            member.class_cols = Int(slot[].unit_held[u].unsafe_ptr())
        members.append(member)
    for u in range(count):  # driver-loop: over the member estimators (their blocks' addresses, now final)
        xs.append(Int(blocks[u].unsafe_ptr()))
        oobs.append(Int(slot[].unit_oob[u].unsafe_ptr()))
    var acc = List[Float64](length=max(n * k, 1), fill=0.0)
    var counts = List[Int32](length=max(n, 1), fill=Int32(0))
    var refused = bag_oob_run(glue, members, xs, ns, ds, oobs, n, k, classifier, Int(acc.unsafe_ptr()),
                              Int(counts.unsafe_ptr()), 0)
    if refused != VOTE_ALL_ANSWERED:
        _raise_member(members[refused], VoteRows(xs[refused], ns[refused], ds[refused]))
    _ = len(blocks)  # read through the RUN above
    _ = len(every)
    if classifier:
        var pred = List[Int32](length=max(n, 1), fill=Int32(0))
        var hits = bag_oob_classifier_finish(glue, Int(acc.unsafe_ptr()), y, Int(pred.unsafe_ptr()), n, k)
        _ = len(pred)
        slot[].oob_words = List[Float64](capacity=n * k + 1)
        for i in range(n * k):  # small-loop(one copy of the out-of-bag decision function into the export view)
            slot[].oob_words.append(acc[i])
        slot[].oob_words.append(Float64(hits) / Float64(n))
    else:
        var pred = List[Float64](length=max(n, 1), fill=0.0)
        var r2 = bag_oob_r2_run(glue, Int(acc.unsafe_ptr()), Int(counts.unsafe_ptr()), y, Int(pred.unsafe_ptr()), n)
        if not r2.finite:
            raise Error("BaggingRegressor oob_score: a non-finite target or out-of-bag prediction, "
                        + "or a sum that overflows float64")
        var score = 1.0 - r2.ss_res / r2.ss_tot if r2.ss_tot > 0 else (1.0 if r2.ss_res == 0 else 0.0)
        slot[].oob_words = List[Float64](capacity=n + 1)
        for i in range(n):  # small-loop(one copy of the out-of-bag predictions into the export view)
            slot[].oob_words.append(pred[i])
        slot[].oob_words.append(score)
    _ = len(acc)
    _ = len(counts)


# ------------------------------------------------------------ THE ADABOOST KIND (lane mf-ada4)


def _ada_config(base: MemberPlan, words: List[Float64]) raises -> ForestConfig:
    """The base's forest configuration from its member plan's words, decoded
    as the forest member decodes them for its own fit (core/forest_member_run.mojo
    `_mint` / `_fit`: the slots' int and float readings and flags, the
    criterion word, the max_features form, the image bin flag, the ET rows
    fraction), with the regressor target check's codes the admission read
    (`META_ADA_TARGET` ..). The words are the configuration the door's
    `forest_config` gives the base's first validated member; the rounds
    differ only in the seed slot (`ada_fit_run`)."""
    var params = base.params.copy()
    if len(params) < FM_HEAD:
        raise Error("member ABI: an AdaBoost member's base has no forest parameter words")
    var slots = Int(params[FM_WORD_SLOTS])
    if len(params) != FM_HEAD + 3 * slots:
        raise Error("member ABI: an AdaBoost member's base words disagree with their slot count")
    var ints = List[Int](length=slots, fill=0)
    var floats = List[Float64](length=slots, fill=0.0)
    var bad = List[UInt8](length=slots, fill=UInt8(0))
    for k in range(slots):  # small-loop(slots): about twenty parameter slots, once per fit
        ints[k] = Int(bitcast[DType.int64](params[FM_HEAD + 3 * k]))
        floats[k] = params[FM_HEAD + 3 * k + 1]
        bad[k] = UInt8(1) if params[FM_HEAD + 3 * k + 2] != 0.0 else UInt8(0)
    var config = ForestConfig(params[1] != 0.0, ints^, floats^, bad^)
    config.criterion = Int(params[2])
    config.features.form = Int(params[3])
    config.features.count = Int(params[4])
    config.features.fraction = params[5]
    config.bins_image = params[6] != 0.0
    config.rows_fraction = params[7]
    config.target_code = Int(words[META_ADA_TARGET])
    config.poisson = Int(words[META_ADA_POISSON])
    config.gamma = Int(words[META_ADA_GAMMA])
    config.inverse_gaussian = Int(words[META_ADA_INVERSE_GAUSSIAN])
    return config^


def _ada_import_round(base: MemberPlan, forest: EntryRef, handle: Int, d: Int, k: Int) raises -> Int:
    """One kept session round as a NEW state of the base's tree member: its
    export copied out of the image (the five arrays the door's PACK
    `forest_export_handle` copies, XF_EXPORT_INTO), imported block by block
    (core/forest_member_run.mojo IMPORT 0 .. 4, then the scalars [features,
    trees, nodes, outputs, classes, meta count, meta words] with
    MEMBER_IMPORT_DONE: the words its own fit keeps). The handle is NOT
    released here (the caller releases it on every path)."""
    var sizes = xf_export_sizes(forest, handle)
    var counts = forest_export_counts(sizes)
    var offsets = List[Int32](length=max(counts[0], 1), fill=Int32(0))
    var colid = List[Int32](length=max(counts[1], 1), fill=Int32(0))
    var quesval = List[Float32](length=max(counts[2], 1), fill=0.0)
    var left = List[Int32](length=max(counts[3], 1), fill=Int32(0))
    var leaves = List[Float32](length=max(counts[4], 1), fill=0.0)
    var dst: List[Int] = [Int(offsets.unsafe_ptr()), Int(colid.unsafe_ptr()), Int(quesval.unsafe_ptr()),
                          Int(left.unsafe_ptr()), Int(leaves.unsafe_ptr())]
    xf_export_into(forest, sizes, dst)
    var scalars = List[Float64](capacity=6 + len(sizes.meta))
    scalars.append(Float64(d))
    scalars.append(Float64(sizes.trees))
    scalars.append(Float64(sizes.nodes))
    scalars.append(Float64(sizes.outputs))
    scalars.append(Float64(k))
    scalars.append(Float64(len(sizes.meta)))
    for i in range(len(sizes.meta)):  # small-loop(meta): at most XF_META_MAX export metadata words
        scalars.append(Float64(sizes.meta[i]))
    var call = member_new(base)
    try:
        for i in range(5):  # small-loop(the five model arrays of one round)
            call.block = MEMBER_BLOCK_FAMILY + i
            call.flags = 0
            var dtype = MEMBER_F32 if i == 2 or i == 4 else MEMBER_I32
            call.x = MemberMatrix(dst[i], 1, counts[i], dtype)
            _ = member_call(base.member, MEMBER_OP_IMPORT, call)
        call.block = MEMBER_BLOCK_SCALARS
        call.flags = MEMBER_IMPORT_DONE
        call.x = MemberMatrix(Int(scalars.unsafe_ptr()), 1, len(scalars), MEMBER_F64)
        _ = member_call(base.member, MEMBER_OP_IMPORT, call)
    except error:
        member_free(base, call)
        raise error^
    _ = len(dst)  # the export and the imports above read the five arrays through these addresses
    _ = len(offsets)
    _ = len(colid)
    _ = len(quesval)
    _ = len(left)
    _ = len(leaves)
    _ = len(scalars)
    return call.state


def _stage_whole(base: MemberPlan, x: Int, n: Int, d: Int) raises -> Bool:
    """Stage X (n x d float32 host rows) once in the forest image of a tree,
    forest or extra-trees base on a device route (`xf_stage_input`, the
    door's `trees_stage_native` / `_staged_input`). True: the caller unstages
    (`xf_unstage_input`), also on an error. A host route, another family or
    an image that cannot stage answers False."""
    if n < 1 or d < 1 or base.route.device == DEVICE_CPU:
        return False
    if (base.family != MEMBER_FAMILY_TREE and base.family != MEMBER_FAMILY_FOREST
            and base.family != MEMBER_FAMILY_EXTRA_TREES):
        return False
    return xf_stage_input(entry_ref_gil_free(base.route.entry), x, n, d)


def _fit_ada(slot: _SlotPtr, call: _CallPtr) raises:
    """`ada_fit` as the kind's fit (module note, THE ADABOOST KIND): the
    door's UNPACK minus Python (the configuration and route words the
    admission read; the uniform starting weights 1 / n of sample_weight=None,
    `_normalized_weights`; the classifier's column-major X, `_colmajor`; the
    data session opened on the base's forest entry as `_member_session`
    opens it, OWNED by the state until it is closed below or by FREE), then
    `ada_fit_run`, the door's RUN, whole. The kept rounds become the state's
    units in round order: a session round's export imported into a new state
    of the base's tree member (`_ada_import_round`, its handle released), a
    member round's state kept as it is. The rounds' fit notices go to the
    PACK (nested member 0)."""
    var classifier = slot[].kind == META_ADA_CLASSIFIER
    var n = call[].x.rows
    var d = call[].x.cols
    var x = call[].x.data
    var y = call[].y.data
    var k = call[].classes if classifier else 0
    if classifier and k < 2:
        raise Error("y has fewer than 2 classes")
    if n < 1:
        raise Error("ValueError: Found array with 0 sample(s) (shape=(0, " + String(d) + ")) while a minimum of 1 "
                    + "is required.")
    var words = slot[].words.copy()
    var base = slot[].base_plans[0].copy()
    base.classes = k
    var native = words[META_ADA_NATIVE] != 0.0
    var glue = slot[].glue
    # sample_weight=None: 1 / n per row (`_normalized_weights`, the fill of `_class_major([1 / n])`).
    var weights = List[Float64](length=n, fill=1.0 / Float64(n))
    if call[].weights.given():
        # THE CALLER'S SAMPLE WEIGHTS (lane mf-split4b): the door's
        # `_normalized_weights` over them, whole: its weights are the float32
        # words `admit_weights` made (the member's `weights` are those words),
        # summed exactly and divided into float64 by `x_trees_exact_sum`, the
        # refusals from its flag words (`boost_weight_flags_check`).
        var normalized = List[Float64](length=n + 1, fill=0.0)
        var flags = List[Int32](length=8, fill=Int32(0))
        _ = xt_exact_sum(glue, call[].weights.data, Int(normalized.unsafe_ptr()), Int(flags.unsafe_ptr()), n, 1)
        boost_weight_flags_check(flags[0] != 0 or flags[1] != 0 or flags[2] != 0 or flags[3] != 0, flags[4] != 0)
        for i in range(n):  # small-loop(one copy of the n normalized weights past the sum word, once per fit)
            weights[i] = normalized[i + 1]
    var forest = entry_ref_unresolved()
    var caps = 0
    _ada_close_session(slot)
    if native:
        forest = entry_ref_gil_free(base.route.entry)
        caps = xf_caps(forest)
        if classifier:
            # `_colmajor`: the transpose the session uploads and the fit without a session reads.
            slot[].ada_xcm = List[Float32](length=n * d, fill=0.0)
            _ = xt_transpose_f32(glue, x, Int(slot[].ada_xcm.unsafe_ptr()), n, d)
        if (caps & XF_CAP_SESSION) != 0:
            var source = Int(slot[].ada_xcm.unsafe_ptr()) if classifier else x
            var share = words[META_ADA_SHARE] != 0.0
            slot[].ada_forest = forest
            try:
                slot[].ada_session = xf_session_open(forest, source, n, d, not classifier, share)
            except error:
                if TREES_NONFINITE in String(error):
                    raise Error(TREES_NONFINITE)
                raise error^
    var plan = AdaFitPlan(classifier, n, d, k, Int(words[META_ADA_COUNT]), words[META_ADA_RATE],
                          Int(words[META_ADA_SEED]), _ada_config(base, words), glue, forest, caps, slot[].ada_session)
    plan.x = x
    plan.weights = Int(weights.unsafe_ptr())
    if classifier:
        plan.codes = y
    else:
        plan.loss = Int(words[META_ADA_LOSS])
        plan.y = y
    var staged = False
    if native:
        plan.device_rounds = slot[].ada_session > 0 and (caps & XF_CAP_ADA) != 0
        plan.fast = base.params[8] != 0.0
        plan.groves = base.params[9] != 0.0
        plan.squared = Int(words[META_ADA_SQUARED])
        plan.xcm = Int(slot[].ada_xcm.unsafe_ptr())
    else:
        plan.member_route = True
        plan.member_plan = base.copy()
        # A random-splitter tree's rounds fit its inner extra-trees forest and
        # step as the tree they are (the door's `kind.startswith("dt_")`).
        plan.tree_base = 1 if base.family == MEMBER_FAMILY_TREE or words[META_ADA_DT_RANDOM] != 0.0 else 0
        if not classifier:
            plan.target_need = forest_target_need(plan.config)
        # Every round's member predicts the same training X: staged once in
        # the members' image (the door's `trees_stage_native`, m > 1).
        if plan.rounds > 1:
            staged = _stage_whole(base, x, n, d)
    # The base's class_weight (lane mf-split4): its row weights over the
    # codes, as the door's `trees_class_rows_only` (0: every weight 1, no
    # rows), multiplied into each host round's weights by the x_trees
    # image's label-weights kernel (the door's driver; a device round reads
    # the rows itself).
    var cw_rows = List[Float32]()
    if classifier:
        var cw = 0
        try:
            cw = _tree_class_rows(words, META_ADA_WORDS, y, n, k, cw_rows)
        except error:
            if staged:
                xf_unstage_input(entry_ref_gil_free(base.route.entry))
            _ada_close_session(slot)
            raise error^
        if cw != 0:
            plan.class_rows = cw
            if not plan.device_rounds:
                plan.multiply = LabelWeightsDriver(label_weights_native_v1)
    var result = AdaFitResult()
    try:
        result = ada_fit_run(plan)
    finally:
        if staged:
            xf_unstage_input(entry_ref_gil_free(base.route.entry))
    _ = len(weights)  # RUN rescaled the weights in place through their address
    _ = len(cw_rows)  # RUN read the class rows through `plan.class_rows`
    var states = List[Int]()
    try:
        if result.members:
            states = result.handles.copy()
            result.handles = List[Int]()
        else:
            var j = 0
            try:
                while j < len(result.handles):  # driver-loop: over the kept boosting rounds (one import each)
                    var handle = result.handles[j]
                    result.handles[j] = 0
                    try:
                        states.append(_ada_import_round(base, forest, handle, d, k))
                    except error:
                        try:
                            xf_export_release(forest, handle)
                        except:
                            pass
                        raise error^
                    xf_export_release(forest, handle)
                    j += 1
            except error:
                var rest = List[Int]()
                for i in range(j + 1, len(result.handles)):  # small-loop(the kept rounds not imported, error path only)
                    rest.append(result.handles[i])
                ada_release_handles(forest, rest)
                raise error^
    except error:
        ada_release_members(base, states)
        if result.members:
            ada_release_members(base, result.handles)
        _ada_close_session(slot)
        raise error^
    # The rounds are states now: the session and its source go (the door
    # closes it when its fit returns).
    _ada_close_session(slot)
    _expand(slot, len(states))
    for u in range(len(states)):  # driver-loop: over the kept rounds (their class count)
        slot[].plans[u].classes = k
    slot[].states = states^
    slot[].ada_alphas = result.alphas.copy()
    slot[].ada_errors = result.errors.copy()
    if len(result.notices) > 0:
        slot[].notice_owner = [0]
        slot[].notice_words = [result.notices.copy()]
    slot[].classes = k


def _ada_round_outputs(slot: _SlotPtr, rows: VoteRows, classifier: Bool, dsts: List[Int]) raises:
    """Every kept round's own predict of `rows` (int32 codes or float32
    values at `dsts[u]`), the members the door's predict asks one by one: on
    the host column one task per round across the host cores (independent
    states; each forest member's walk splits its rows again), on a device
    route one after the other (each launch fills the device). A round that
    refuses raises its own words."""
    var m = len(slot[].states)
    var host = len(slot[].base_plans) > 0 and slot[].base_plans[0].route.device == DEVICE_CPU
    var dtype = MEMBER_I32 if classifier else MEMBER_F32
    if host and m > 1:
        var failed = List[UInt8](length=m, fill=UInt8(0))
        var failed_at = Int(failed.unsafe_ptr())
        var dst_at = Int(dsts.unsafe_ptr())
        var x = rows.x
        var n = rows.n
        var d = rows.d

        def task(u: Int) {imm slot, imm failed_at, imm dst_at, imm x, imm n, imm d, imm dtype}:
            var flags = MutPointer[UInt8, MutUntrackedOrigin](unsafe_from_address=failed_at)
            try:
                var inner = member_call_of(slot[].plans[u], slot[].states[u])
                inner.x = MemberMatrix(x, n, d, MEMBER_F32)
                inner.dst = MemberMatrix(MutPointer[Int, MutUntrackedOrigin](unsafe_from_address=dst_at)[u], n, 1,
                                         dtype)
                _ = member_call(slot[].plans[u].member, MEMBER_OP_PREDICT, inner)
            except:
                flags[u] = UInt8(1)

        host_parallelize(task, m)
        for u in range(m):  # driver-loop: over the kept rounds (the first refusal, raised in its words)
            if failed[u] != UInt8(0):
                var again = member_call_of(slot[].plans[u], slot[].states[u])
                again.x = MemberMatrix(rows.x, rows.n, rows.d, MEMBER_F32)
                again.dst = MemberMatrix(dsts[u], rows.n, 1, dtype)
                _ = member_call(slot[].plans[u].member, MEMBER_OP_PREDICT, again)
                raise Error("RuntimeError: mojolearn: an AdaBoost member's round refused its predict")
        _ = len(failed)
        return
    for u in range(m):  # driver-loop: over the kept rounds (one member predict each, the door's order)
        var inner = member_call_of(slot[].plans[u], slot[].states[u])
        inner.x = MemberMatrix(rows.x, rows.n, rows.d, MEMBER_F32)
        inner.dst = MemberMatrix(dsts[u], rows.n, 1, dtype)
        _ = member_call(slot[].plans[u].member, MEMBER_OP_PREDICT, inner)


def _predict_ada(slot: _SlotPtr, call: _CallPtr, operation: Int) raises:
    """AdaBoost's public predicts over the kept rounds (`_ada_classify` /
    `ada_predict`): each round's member predict of X (X staged ONCE in the
    base's image on a device route when two or more rounds read it, the
    door's `trees_stage_members`), then `ada_classifier_run` (ADA_CODES:
    predict, ADA_PROBA: predict_proba, ADA_DECISION: decision_function) or
    `ada_regressor_run` (the weighted median, float32), the door's RUNs."""
    var name = _name(slot[].kind)
    var classifier = slot[].kind == META_ADA_CLASSIFIER
    var rows = _rows(call, name)
    var n = rows.n
    var d = rows.d
    var k = slot[].classes
    var glue = slot[].glue
    if classifier:
        if operation == MEMBER_OP_PREDICT:
            _codes_dst(call, n, name)
        elif operation == MEMBER_OP_PREDICT_PROBA:
            _proba_dst(call, n, k, name)
        elif operation == MEMBER_OP_DECISION:
            if not call[].dst.dense_host(MEMBER_F64) or call[].dst.rows != n or call[].dst.cols != (
                    1 if k == 2 else k):
                raise Error("TypeError: an AdaBoostClassifier member writes decision_function as dense host "
                            + "float64 rows (one column for two classes)")
        else:
            member_unsupported(name, operation)
    elif operation == MEMBER_OP_PREDICT:
        if not call[].dst.dense_host(MEMBER_F32) or call[].dst.rows != n:
            raise Error("TypeError: an AdaBoostRegressor member writes predict as one dense host float32 column")
    else:
        member_unsupported(name, operation)
    var m = len(slot[].states)
    var codes = List[List[Int32]](capacity=m)
    var values = List[List[Float32]](capacity=m)
    var dsts = List[Int](capacity=m)
    for _ in range(m):  # driver-loop: over the kept rounds (their output blocks)
        if classifier:
            codes.append(List[Int32](length=max(n, 1), fill=Int32(0)))
        else:
            values.append(List[Float32](length=max(n, 1), fill=0.0))
    for u in range(m):  # driver-loop: over the kept rounds (their blocks' addresses, now final)
        dsts.append(Int(codes[u].unsafe_ptr()) if classifier else Int(values[u].unsafe_ptr()))
    var staged = False
    if m > 1 and len(slot[].base_plans) > 0:
        staged = _stage_whole(slot[].base_plans[0], rows.x, n, d)
    try:
        _ada_round_outputs(slot, rows, classifier, dsts)
    finally:
        if staged:
            xf_unstage_input(entry_ref_gil_free(slot[].base_plans[0].route.entry))
    if classifier:
        var scratch = List[Float64]()
        var acc = call[].dst.data
        var finish = ADA_CODES
        if operation == MEMBER_OP_PREDICT_PROBA:
            finish = ADA_PROBA
        elif operation == MEMBER_OP_DECISION:
            finish = ADA_DECISION
        if finish == ADA_CODES or k == 2:
            # the n x k decision block the finish reads (zero, as the door's)
            scratch = List[Float64](length=max(n * k, 1), fill=0.0)
            acc = Int(scratch.unsafe_ptr())
        else:
            _zero_dst(call, n * k)
        ada_classifier_run(glue, dsts, slot[].ada_alphas, n, k, acc, finish, call[].dst.data)
        _ = len(scratch)
    else:
        ada_regressor_run(glue, dsts, slot[].ada_alphas, n, call[].dst.data)
    _ = len(codes)  # the RUNs above read the round blocks through `dsts`
    _ = len(values)


def _outputs_ada(call: _CallPtr, kind: Int, classes: Int) raises:
    """AdaBoost's outputs: predict one column (int32 codes for the
    classifier, float32 for the regressor, the door's dtypes); predict_proba
    `classes` float64 columns; decision_function float64, one column for two
    classes, else `classes`."""
    var classifier = kind == META_ADA_CLASSIFIER
    if call[].flags == MEMBER_OP_PREDICT:
        call[].out_cols = 1
        call[].out_dtype = MEMBER_I32 if classifier else MEMBER_F32
    elif call[].flags == MEMBER_OP_PREDICT_PROBA and classifier:
        call[].out_cols = classes
        call[].out_dtype = MEMBER_F64
    elif call[].flags == MEMBER_OP_DECISION and classifier:
        call[].out_cols = 1 if classes == 2 else classes
        call[].out_dtype = MEMBER_F64
    else:
        member_unsupported(_name(kind), call[].flags)
    call[].classes = classes if classifier else 0


def _ada_view(slot: _SlotPtr):
    """META_BLOCK_ADA's words: [weights count, estimator_weights_..., errors
    count, estimator_errors_...]."""
    var w = List[Float64](capacity=2 + len(slot[].ada_alphas) + len(slot[].ada_errors))
    w.append(Float64(len(slot[].ada_alphas)))
    for i in range(len(slot[].ada_alphas)):  # small-loop(the kept rounds' weights)
        w.append(slot[].ada_alphas[i])
    w.append(Float64(len(slot[].ada_errors)))
    for i in range(len(slot[].ada_errors)):  # small-loop(the kept rounds' errors)
        w.append(slot[].ada_errors[i])
    slot[].ada_words = w^


def _import_ada(slot: _SlotPtr, words: _F64P, count: Int) raises:
    """IMPORT `META_BLOCK_ADA` (`_ada_view`'s layout)."""
    var bad = String("member ABI: an AdaBoost member's weight words are malformed")
    if count < 2:
        raise Error(bad)
    var na = Int(words[0])
    if na < 0 or 1 + na >= count:
        raise Error(bad)
    var ne = Int(words[1 + na])
    if ne < 0 or 2 + na + ne != count:
        raise Error(bad)
    slot[].ada_alphas = List[Float64](capacity=na)
    for i in range(na):  # small-loop(the rounds' weights)
        slot[].ada_alphas.append(words[1 + i])
    slot[].ada_errors = List[Float64](capacity=ne)
    for i in range(ne):  # small-loop(the rounds' errors)
        slot[].ada_errors.append(words[2 + na + i])


def _unit_views(slot: _SlotPtr):
    """A unit kind's export views after a fit or an import: the calibrated
    pairs' words (`META_BLOCK_CAL`: pairs, then per pair its held count and
    codes, its calibrator count and per calibrator isotonic, a, b, m, the m
    knots x and the m knots y) and the bags' words (`META_BLOCK_DRAWS`:
    units, then per unit its feature count and columns, its held count and
    codes, its out-of-bag count and rows)."""
    if slot[].kind == META_CALIBRATED:
        var w = List[Float64]()
        w.append(Float64(len(slot[].states)))
        for i in range(len(slot[].states)):  # driver-loop: over the pairs
            var held = slot[].unit_held[i].copy() if i < len(slot[].unit_held) else List[Int32]()
            w.append(Float64(len(held)))
            for c in range(len(held)):  # small-loop(a pair's held classes)
                w.append(Float64(Int(held[c])))
            w.append(Float64(slot[].pair_count[i]))
            for j in range(slot[].pair_first[i], slot[].pair_first[i] + slot[].pair_count[i]):  # driver-loop: over the pair's calibrators
                w.append(1.0 if slot[].cal_iso[j] else 0.0)
                w.append(slot[].cal_a[j])
                w.append(slot[].cal_b[j])
                w.append(Float64(slot[].cal_m[j]))
                for q in range(slot[].cal_m[j]):  # small-loop(one calibrator's knots)
                    w.append(slot[].cal_kx[j][q])
                for q in range(slot[].cal_m[j]):  # small-loop(one calibrator's knots)
                    w.append(slot[].cal_ky[j][q])
        slot[].cal_words = w^
    elif _bag_kind(slot[].kind):
        var w = List[Int32]()
        var units = len(slot[].states)
        w.append(Int32(units))
        for u in range(units):  # driver-loop: over the member estimators
            var cols = slot[].unit_cols[u].copy() if u < len(slot[].unit_cols) else List[Int32]()
            w.append(Int32(len(cols)))
            for j in range(len(cols)):  # small-loop(a member's feature subset)
                w.append(cols[j])
            var held = slot[].unit_held[u].copy() if u < len(slot[].unit_held) else List[Int32]()
            w.append(Int32(len(held)))
            for c in range(len(held)):  # small-loop(a member's held classes)
                w.append(held[c])
            var oob = slot[].unit_oob[u].copy() if u < len(slot[].unit_oob) else List[Int32]()
            w.append(Int32(len(oob)))
            for i in range(len(oob)):  # small-loop(a member's out-of-bag rows, once per fit)
                w.append(oob[i])
        slot[].draw_words = w^
    elif _ada_kind(slot[].kind):
        _ada_view(slot)
    elif _rfe_kind(slot[].kind):
        _rfe_view(slot)


# ------------------------------------------------------------ THE SEARCH KIND (lane mf-search4)


def _search_groups_end(words: List[Float64], count: Int) raises -> Int:
    """The index after the search kind's scorer word groups (one per scorer,
    `max(count, 1)`; lane mf-tail5b), -1 when they are cut short."""
    var at = META_SEARCH_WORDS
    var sels = List[Int32]()
    for _ in range(max(count, 1)):  # small-loop(one word group per scorer)
        var code = 0
        var positive = -1
        var opts = MselScorerOptions()
        try:
            at = msel_scorer_group_read(words, at, code, positive, opts, sels)
        except:
            return -1
    return at


def _fit_search(slot: _SlotPtr, call: _CallPtr) raises:
    """THE SEARCH KIND's fit (`_BaseSearch.fit` over native member units):
    core/msel_search_run.mojo `msel_search_run` over the entry's candidates;
    the refitted best is the state's one nested member, the tables are the
    state's search words (`META_BLOCK_SEARCH`), the units' and the refit's
    fit notices go to the PACK with the best estimator's context."""
    var classifier = slot[].kind == META_SEARCH_CLASSIFIER
    var outputs = max(call[].y.cols, 1)
    var k = call[].classes if classifier and outputs == 1 else 0
    var plan = MselSearchPlan(slot[].base_plans.copy(), slot[].base_masks.copy(), call[].x.data, call[].x.rows,
                              call[].x.cols, call[].y.data, k, slot[].folds, slot[].glue, slot[].metric)
    plan.y_dtype = MEMBER_I32 if classifier else MEMBER_F32
    plan.y_cols = outputs
    if not classifier:
        # THE GIVEN y (lane mf-tail5): the RUN hands it to the candidates that read it.
        plan.given = slot[].fit_given
        plan.given_dtype = slot[].fit_given_dtype
    if outputs > 1 and call[].class_counts != 0:
        var counts = MutPointer[Int, MutUntrackedOrigin](unsafe_from_address=call[].class_counts)
        for j in range(outputs):  # small-loop(the outputs' class counts)
            plan.class_counts.append(counts[j])
    plan.scoring = Int(slot[].words[META_SEARCH_SCORING])
    plan.tolerate = slot[].words[META_SEARCH_TOLERATE] != 0.0
    plan.error_value = slot[].words[META_SEARCH_ERROR]
    plan.return_train = slot[].words[META_SEARCH_TRAIN] != 0.0
    # MULTI-METRIC, refit=<name> and refit=False (lane mf-search4b item 1):
    # the scorer codes in the caller's name order, the refit scorer's index,
    # the refit word; then one positive class word per scorer (item 3).
    var count = Int(slot[].words[META_SEARCH_COUNT])
    plan.refit_scorer = Int(slot[].words[META_SEARCH_REFIT_SCORER]) if count > 0 else 0
    plan.refit = slot[].words[META_SEARCH_REFIT] != 0.0
    # One scorer word group per scorer (lane mf-tail5b): its code (each of a
    # multi-metric list's; the one scorer's is META_SEARCH_SCORING's too), its
    # positive word and its make_scorer options with their labels codes.
    var at = META_SEARCH_WORDS
    for j in range(max(count, 1)):  # small-loop(one word group per scorer)
        var code = 0
        var positive = -1
        var opts = MselScorerOptions()
        at = msel_scorer_group_read(slot[].words, at, code, positive, opts, plan.selections)
        if count > 0:
            plan.scorers.append(code)
        plan.positives.append(positive)
        plan.options.append(opts)
    # THE CV TOOLS (lane mf-msel5): the optional tail after the scorer groups.
    var tool_at = at
    var tool = len(slot[].words) == tool_at + META_SEARCH_TOOL_WORDS
    if tool:
        plan.keep = slot[].words[tool_at] != 0.0
        plan.respond = Int(slot[].words[tool_at + 1])
        plan.indices = slot[].words[tool_at + 2] != 0.0
        plan.respond_dst = Int(slot[].words[tool_at + 3])
        plan.respond_width = Int(slot[].words[tool_at + 4])
        plan.respond_dtype = Int(slot[].words[tool_at + 5])
        plan.respond_log = slot[].words[tool_at + 6] != 0.0
    plan.cap = slot[].cap
    plan.cluster = slot[].cluster
    plan.label_words = slot[].label_words.copy()
    # THE SPLITTER CV and a user's index list (lane mf-search4b item 6): the
    # folds of this fit's rows, as Stacking's and Calibrated's (`_split_folds`;
    # the user's are the entry's typed folds), handed to the RUN.
    if slot[].split.kind != MSEL_SPEC_NONE:
        # A one-output regressor hands its targets (a stratified splitter
        # stratifies on their values, as the door's split does; lane mf-del5).
        _split_folds(slot, call[].x.rows, call[].y.data if k > 0 else 0, k, _call_groups(call),
                     call[].y.data if not classifier and outputs == 1 else 0,
                     slot[].fit_given if not classifier and outputs == 1 else 0, slot[].fit_given_dtype)
    if len(slot[].fold_test) > 0:
        # Views of the slot's int32 sides (kept alive by the slot through the
        # RUN), checked as the door checks a split's rows.
        _user_folds(slot, call[].x.rows, plan.fold_train_at, plan.fold_train_n, plan.fold_test_at,
                    plan.fold_test_n, False)
    # THE RESIDENT ROUTE (lane mf-search5; core/msel_search_run.mojo
    # `msel_search_rows`): the caller's resident scope and X's pin, read by
    # its UNPACK once per fit (bindings/msel_member_units_door.mojo, the
    # search's own fit door opening the scope the old DeviceRows units
    # opened), reach every unit's fit and score when the admission allowed
    # resident fold rows (`flag`, the search's own `device`). The RUN writes
    # this call's `pin_kept` the moment a candidate's cache takes X's
    # reference (THE PIN RULE: whether it returns or raises).
    if slot[].flag and call[].scope != MEMBER_SCOPE_NONE and call[].pin != 0:
        plan.scope = call[].scope
        plan.pin = call[].pin
        plan.kept_at = Int(UnsafePointer(to=call[].pin_kept))
    var res = msel_search_run(plan)
    # THE CV TOOLS: the kept unit states are the slot's from here (freed by
    # `_drop_states`), each with its candidate.
    slot[].fold_states = res.unit_states.copy()
    res.unit_states = List[Int]()
    slot[].fold_owner = List[Int](capacity=len(slot[].fold_states))
    slot[].fold_words = List[Int64](capacity=len(slot[].fold_states))
    for u in range(len(slot[].fold_states)):  # driver-loop: over the kept unit states (their candidates and handles)
        slot[].fold_owner.append(u // max(res.n_splits, 1))
        slot[].fold_words.append(Int64(slot[].fold_states[u]))
    var best = slot[].base_plans[res.best].copy()
    best.classes = k
    best.class_counts = plan.class_counts.copy()
    slot[].plans = [best.copy()]
    slot[].masks = [slot[].base_masks[res.best]]
    # refit=False: no refitted best (state 0: `_nested_outputs` skips it,
    # `_forward_best` refuses, the PACK writes no best_estimator_).
    slot[].states = [res.best_state]
    for i in range(len(res.notices)):  # driver-loop: over the units' fit notices, in unit order
        slot[].notice_owner.append(0)
        slot[].notice_words.append(res.notices[i].copy())
    slot[].classes = k
    slot[].out_classes = plan.class_counts.copy()
    var m = len(res.means)
    var folds = res.n_splits
    var tables = len(res.scorer_means)
    var w = List[Float64](capacity=META_SEARCH_HEAD + tables * m * (2 * folds + 4))
    w.append(Float64(res.best))
    w.append(Float64(folds))
    w.append(res.best_score)
    w.append(Float64(m))
    w.append(1.0 if plan.return_train else 0.0)
    w.append(slot[].words[META_SEARCH_SEED])
    w.append(Float64(tables))
    w.append(1.0 if plan.refit else 0.0)
    for j in range(tables):  # driver-loop: over the scorers (one table each, in their order)
        for c in range(m):  # driver-loop: over the candidates (their test scores)
            for f in range(folds):  # small-loop(one candidate's folds)
                w.append(res.scorer_test[j][c][f])
        for c in range(m):  # driver-loop: over the candidates (their means)
            w.append(res.scorer_means[j][c])
        for c in range(m):  # driver-loop: over the candidates (their deviations)
            w.append(res.scorer_stds[j][c])
        for c in range(m):  # driver-loop: over the candidates (their ranks)
            w.append(Float64(res.scorer_ranks[j][c]))
        if plan.return_train:
            for c in range(m):  # driver-loop: over the candidates (their train scores)
                for f in range(folds):  # small-loop(one candidate's folds)
                    w.append(res.scorer_train[j][c][f])
            for c in range(m):  # driver-loop: over the candidates (their train means)
                w.append(res.scorer_train_means[j][c])
    if tool:
        _search_tool_words(res, plan.indices, w)
    slot[].search_words = w^


def _search_tool_words(res: MselSearchResult, rows: Bool, mut w: List[Float64]):
    """THE CV TOOLS' tail of the search words (lane mf-msel5), after the
    tables: [units U], U fit seconds, U test-score seconds; with `rows` per
    fold [train n, test n, its train rows, its test rows]. Exact: rows are
    below 2^53. The units' responses are not here: the RUN wrote them into
    the caller's result buffer (lane mf-msel5b)."""
    var units = len(res.fit_seconds)
    w.append(Float64(units))
    for u in range(units):  # driver-loop: over the units (their fits' seconds)
        w.append(res.fit_seconds[u])
    for u in range(units):  # driver-loop: over the units (their test scores' seconds)
        w.append(res.score_seconds[u])
    if rows:
        for f in range(len(res.fold_test)):  # driver-loop: over the folds (their rows)
            w.append(Float64(len(res.fold_train[f])))
            w.append(Float64(len(res.fold_test[f])))
            for i in range(len(res.fold_train[f])):  # driver-loop: over the fold's train rows
                w.append(Float64(res.fold_train[f][i]))
            for i in range(len(res.fold_test[f])):  # driver-loop: over the fold's test rows
                w.append(Float64(res.fold_test[f][i]))


def _forward_best(slot: _SlotPtr, call: _CallPtr, operation: Int) raises:
    """THE SEARCH KIND's predict / predict_proba / decision_function / score:
    the refitted best's own call (the door's `best_estimator_` delegation),
    on the caller's views."""
    if len(slot[].states) != 1 or slot[].states[0] == 0:
        # refit=False (lane mf-search4b): scikit-learn's `_check_refit` words.
        raise Error("AttributeError: This " + _name(slot[].kind) + " instance was initialized with `refit=False`. "
                    + "The operation is available only after refitting on the best parameters. You can refit an "
                    + "estimator manually using the `best_params_` attribute")
    var plan = slot[].plans[0].copy()
    var inner = member_call_of(plan, slot[].states[0])
    inner.x = call[].x
    inner.y = call[].y
    inner.weights = call[].weights
    inner.dst = call[].dst
    inner.row_index = call[].row_index
    inner.column_index = call[].column_index
    inner.flags = call[].flags
    _ = member_call(plan.member, operation, inner)
    call[].value = inner.value
    call[].out_cols = inner.out_cols


def _import_search(slot: _SlotPtr, words: _F64P, count: Int) raises:
    """IMPORT `META_BLOCK_SEARCH` (`_fit_search`'s layout), after
    `META_BLOCK_NESTED`: the best candidate's plan becomes the one nested
    plan."""
    if count < META_SEARCH_HEAD:
        raise Error("member ABI: a search member's words are malformed")
    var best = Int(words[0])
    if best < 0 or best >= len(slot[].base_plans):
        raise Error("member ABI: a search member's best candidate is not one of its candidates")
    slot[].search_words = List[Float64](capacity=count)
    for i in range(count):  # small-loop(the imported search words)
        slot[].search_words.append(words[i])
    slot[].plans = [slot[].base_plans[best].copy()]
    slot[].masks = [slot[].base_masks[best]]


def _outputs_search(call: _CallPtr, first: MemberPlan, classes: Int) raises:
    """A search's outputs before a fit: its first candidate's (every
    candidate is the same estimator class)."""
    var plan = first.copy()
    plan.classes = classes
    var asked = call[].flags
    call[].out_cols = member_outputs(plan, asked, max(call[].x.cols, 1), max(call[].y.cols, 1))
    call[].out_dtype = member_output_dtype(plan, asked, max(call[].x.cols, 1), max(call[].y.cols, 1),
                                           MEMBER_DTYPE_UNSAID)
    call[].classes = classes


# ------------------------------------------------------------ THE MULTI-OUTPUT KIND (lane mf-search4)


def _fit_multi(slot: _SlotPtr, call: _CallPtr) raises:
    """`MultiOutputClassifier.fit` / `MultiOutputRegressor.fit`: one unit
    of the base per output (THE 2-D y: column j of the caller's rows x
    outputs targets, with output j's class count for a classifier), fitted
    as ONE member set, the sample weights handed to every unit."""
    var classifier = slot[].kind == META_MULTI_CLASSIFIER
    var n = call[].x.rows
    var d = call[].x.cols
    var outputs = call[].y.cols
    if outputs < 1:
        raise Error("member ABI: a " + _name(slot[].kind) + " member needs at least one output")
    var counts = List[Int](length=outputs, fill=0)
    if classifier:
        if call[].class_counts != 0:
            var given = MutPointer[Int, MutUntrackedOrigin](unsafe_from_address=call[].class_counts)
            for j in range(outputs):  # small-loop(the outputs' class counts)
                counts[j] = given[j]
        elif outputs == 1:
            counts[0] = call[].classes
        else:
            raise Error("member ABI: a MultiOutputClassifier member needs each output's class count at fit")
        for j in range(outputs):  # small-loop(the outputs' class counts)
            if counts[j] < 1:
                raise Error("member ABI: a MultiOutputClassifier member needs each output's class count at fit")
    _expand(slot, outputs)
    slot[].targets = List[List[Int32]](capacity=outputs)
    var columns = List[Int](length=outputs, fill=0)
    for j in range(outputs):  # driver-loop: over the outputs (their target columns, allocated once)
        slot[].targets.append(List[Int32](length=max(n, 1), fill=Int32(0)))
        columns[j] = Int(slot[].targets[j].unsafe_ptr())
    var columns_at = Int(columns.unsafe_ptr())
    var y = call[].y.data

    def _column(j: Int) {imm columns_at, imm y, imm n, imm outputs}:
        var dst = _I32P(unsafe_from_address=MutPointer[Int, MutUntrackedOrigin](unsafe_from_address=columns_at)[j])
        var src = _I32P(unsafe_from_address=y)
        for i in range(n):
            dst[i] = src[i * outputs + j]  # 4-byte words of either element type, as bits

    host_parallelize(_column, outputs)
    _ = len(columns)
    var weights = call[].weights.data if call[].weights.given() else 0
    # THE GIVEN y per output (lane mf-tail5): a regressor base that reads it
    # fits output j with column j of the given y (float64 words, one parallel
    # pass); none is made for a base that does not read it.
    var given_cols = List[List[Float64]]()
    var given_src = slot[].fit_given
    var given_kind = slot[].fit_given_dtype
    if (not classifier and given_src != 0 and len(slot[].base_masks) > 0
            and (slot[].base_masks[0] & MEMBER_READS_Y_GIVEN) != 0):
        var given_at = List[Int](length=outputs, fill=0)
        for j in range(outputs):  # driver-loop: over the outputs (their given columns, allocated once)
            given_cols.append(List[Float64](length=max(n, 1), fill=0.0))
            given_at[j] = Int(given_cols[j].unsafe_ptr())
        var given_at_at = Int(given_at.unsafe_ptr())

        def _given_column(j: Int) {imm given_at_at, imm given_src, imm given_kind, imm n, imm outputs}:
            var dst = _F64P(unsafe_from_address=MutPointer[Int, MutUntrackedOrigin](unsafe_from_address=given_at_at)[j])
            for i in range(n):
                var at = i * outputs + j
                if given_kind == MEMBER_I64:
                    dst[i] = Float64(MutPointer[Int64, MutUntrackedOrigin](unsafe_from_address=given_src)[at])
                elif given_kind == MEMBER_I32:
                    dst[i] = Float64(MutPointer[Int32, MutUntrackedOrigin](unsafe_from_address=given_src)[at])
                else:
                    dst[i] = _F64P(unsafe_from_address=given_src)[at]

        host_parallelize(_given_column, outputs)
        _ = len(given_at)
    _ = len(columns)  # keep-alive: _column read it through columns_at
    var run = _set_of(slot, call[].x.data, n, d, MEMBER_I32 if classifier else MEMBER_F32)
    for j in range(outputs):  # driver-loop: over the outputs (admission into the set, once each)
        _add(slot, run, j, counts[j], Int(slot[].targets[j].unsafe_ptr()), weights, 0, 0)
        if len(given_cols) > 0:
            run.given_target(Int(given_cols[j].unsafe_ptr()), MEMBER_F64)
    member_set_fit(run)
    _ = len(given_cols)  # read through their addresses by the fits above
    slot[].states = List[Int](length=outputs, fill=0)
    _keep_states(slot, run, 0)
    slot[].out_classes = counts^
    slot[].classes = 0


def _multi_dtype(slot: _SlotPtr) -> Int:
    """The element type of a multi-output predict: int32 codes for the
    classifier, the base's own float type for the regressor."""
    if slot[].kind == META_MULTI_CLASSIFIER:
        return MEMBER_I32
    if len(slot[].pred_dtype) > 0 and slot[].pred_dtype[0] == MEMBER_F64:
        return MEMBER_F64
    return MEMBER_F32


def _multi_outputs(slot: _SlotPtr, rows: VoteRows, dtype: Int, dst: Int) raises:
    """Each output's member predict over `rows`, column j of the dense
    rows x outputs `dst` (`dtype`), one task per output on the host workers
    when every unit is GIL-free, else one after the other."""
    var outputs = len(slot[].states)
    var n = rows.n
    var bufs = List[List[Int64]](capacity=outputs)
    var buf_at = List[Int](length=max(outputs, 1), fill=0)
    for j in range(outputs):  # driver-loop: over the outputs (their prediction buffers)
        bufs.append(List[Int64](length=max(n, 1), fill=Int64(0)))
        buf_at[j] = Int(bufs[j].unsafe_ptr())
    var failed = List[UInt8](length=max(outputs, 1), fill=UInt8(0))
    var errors = List[String](length=max(outputs, 1), fill=String(""))
    var slot_at = Int(slot)
    var bufs_list = Int(buf_at.unsafe_ptr())
    var failed_at = Int(failed.unsafe_ptr())
    var errors_at = Int(errors.unsafe_ptr())
    var x = rows.x
    var d = rows.d

    def _one(j: Int) {imm slot_at, imm bufs_list, imm failed_at, imm errors_at, imm x, imm d, imm n, imm dtype,
                       imm dst, imm outputs}:
        var s = _SlotPtr(unsafe_from_address=slot_at)
        var buf = MutPointer[Int, MutUntrackedOrigin](unsafe_from_address=bufs_list)[j]
        try:
            var pd = s[].pred_dtype[j]
            var inner = member_call_of(s[].plans[j], s[].states[j])
            inner.x = MemberMatrix(x, n, d)
            inner.dst = MemberMatrix(buf, n, 1, pd)
            _ = member_call(s[].plans[j].member, MEMBER_OP_PREDICT, inner)
            for i in range(n):
                var value: Float64
                if pd == MEMBER_I32:
                    value = Float64(_I32P(unsafe_from_address=buf)[i])
                elif pd == MEMBER_I64:
                    value = Float64(_I64P(unsafe_from_address=buf)[i])
                elif pd == MEMBER_F64:
                    value = _F64P(unsafe_from_address=buf)[i]
                else:
                    value = Float64(_F32P(unsafe_from_address=buf)[i])
                if dtype == MEMBER_I32:
                    _I32P(unsafe_from_address=dst)[i * outputs + j] = Int32(Int(value))
                elif dtype == MEMBER_F64:
                    _F64P(unsafe_from_address=dst)[i * outputs + j] = value
                else:
                    _F32P(unsafe_from_address=dst)[i * outputs + j] = Float32(value)
        except error:
            MutPointer[UInt8, MutUntrackedOrigin](unsafe_from_address=failed_at)[j] = UInt8(1)
            MutPointer[String, MutUntrackedOrigin](unsafe_from_address=errors_at)[j] = String(error)

    if _all(slot[].masks, MEMBER_GIL_FREE):
        host_parallelize(_one, outputs)
    else:
        for j in range(outputs):  # driver-loop: over the outputs (THE GIL RULE: kinds that are not GIL-free)
            _one(j)
    _ = len(bufs)
    _ = len(buf_at)
    for j in range(outputs):  # driver-loop: over the outputs (the first refusal, in output order)
        if failed[j] != UInt8(0):
            raise Error(errors[j])


def _multi_proba_width(slot: _SlotPtr) -> Int:
    """THE MULTI-OUTPUT PROBABILITIES (lane mf-tail5): the columns of one
    predict_proba row, every output's classes in turn (0: an output's
    member has none)."""
    var total = 0
    for j in range(len(slot[].proba_cols)):  # small-loop(the outputs)
        if slot[].proba_cols[j] <= 0:
            return 0
        total += slot[].proba_cols[j]
    return total


def _proba_multi(slot: _SlotPtr, call: _CallPtr) raises:
    """MultiOutputClassifier.predict_proba (lane mf-tail5): scikit-learn's
    list of per-output (n, classes_j) arrays as ONE block, output j's member
    proba in columns `sum(proba_cols[:j]) ..` of the dense host float64 rows
    x `_multi_proba_width` `dst` (each output's own classes, in its member's
    order); the caller's PACK cuts the list. One task per output when every
    member is GIL-free."""
    var name = _name(slot[].kind)
    var width = _multi_proba_width(slot)
    if width == 0:
        member_unsupported(name, MEMBER_OP_PREDICT_PROBA)
    var rows = _rows(call, name)
    var n = rows.n
    if not call[].dst.dense_host(MEMBER_F64) or call[].dst.rows != n or call[].dst.cols != width:
        raise Error("TypeError: a " + name + " member writes predict_proba as dense host float64 rows x "
                    + String(width) + " (every output's classes)")
    var outputs = len(slot[].states)
    var offsets = List[Int](length=max(outputs, 1), fill=0)
    for j in range(1, outputs):  # small-loop(the outputs' first columns)
        offsets[j] = offsets[j - 1] + slot[].proba_cols[j - 1]
    var bufs = List[List[Float64]](capacity=outputs)
    var buf_at = List[Int](length=max(outputs, 1), fill=0)
    for j in range(outputs):  # driver-loop: over the outputs (their probability buffers)
        bufs.append(List[Float64](length=max(n * slot[].proba_cols[j], 1), fill=0.0))
        buf_at[j] = Int(bufs[j].unsafe_ptr())
    var failed = List[UInt8](length=max(outputs, 1), fill=UInt8(0))
    var errors = List[String](length=max(outputs, 1), fill=String(""))
    var slot_at = Int(slot)
    var bufs_list = Int(buf_at.unsafe_ptr())
    var offsets_at = Int(offsets.unsafe_ptr())
    var failed_at = Int(failed.unsafe_ptr())
    var errors_at = Int(errors.unsafe_ptr())
    var x = rows.x
    var d = rows.d
    var dst = call[].dst.data

    def _one(j: Int) {imm slot_at, imm bufs_list, imm offsets_at, imm failed_at, imm errors_at, imm x, imm d, imm n,
                       imm dst, imm width}:
        var s = _SlotPtr(unsafe_from_address=slot_at)
        var buf = MutPointer[Int, MutUntrackedOrigin](unsafe_from_address=bufs_list)[j]
        var first = MutPointer[Int, MutUntrackedOrigin](unsafe_from_address=offsets_at)[j]
        try:
            var pd = s[].proba_dtype[j]
            var cols = s[].proba_cols[j]
            var inner = member_call_of(s[].plans[j], s[].states[j])
            inner.x = MemberMatrix(x, n, d)
            inner.dst = MemberMatrix(buf, n, cols, pd)
            _ = member_call(s[].plans[j].member, MEMBER_OP_PREDICT_PROBA, inner)
            var out = _F64P(unsafe_from_address=dst)
            for i in range(n):
                for c in range(cols):
                    var value = _F64P(unsafe_from_address=buf)[i * cols + c] if pd == MEMBER_F64 else Float64(
                        _F32P(unsafe_from_address=buf)[i * cols + c])
                    out[i * width + first + c] = value
        except error:
            MutPointer[UInt8, MutUntrackedOrigin](unsafe_from_address=failed_at)[j] = UInt8(1)
            MutPointer[String, MutUntrackedOrigin](unsafe_from_address=errors_at)[j] = String(error)

    if _all(slot[].masks, MEMBER_GIL_FREE):
        host_parallelize(_one, outputs)
    else:
        for j in range(outputs):  # driver-loop: over the outputs (THE GIL RULE: kinds that are not GIL-free)
            _one(j)
    _ = len(bufs)
    _ = len(buf_at)
    _ = len(offsets)
    for j in range(outputs):  # driver-loop: over the outputs (the first refusal, in output order)
        if failed[j] != UInt8(0):
            raise Error(errors[j])


def _predict_multi(slot: _SlotPtr, call: _CallPtr, operation: Int) raises:
    """predict: n x outputs (int32 codes per output, or the regressor's
    floats). predict_proba (MultiOutputClassifier, lane mf-tail5): every
    output's block in one float64 dst (`_proba_multi`). decision_function:
    scikit-learn's MultiOutputClassifier has none (refused)."""
    var name = _name(slot[].kind)
    if operation == MEMBER_OP_PREDICT_PROBA and slot[].kind == META_MULTI_CLASSIFIER:
        _fitted(slot)
        _proba_multi(slot, call)
        return
    if operation != MEMBER_OP_PREDICT:
        member_unsupported(name, operation)
    var rows = _rows(call, name)
    var outputs = len(slot[].states)
    var dtype = _multi_dtype(slot)
    if not call[].dst.dense_host(dtype) or call[].dst.rows != rows.n or call[].dst.cols != outputs:
        raise Error("TypeError: a " + name + " member writes predict as dense host rows x outputs")
    _multi_outputs(slot, rows, dtype, call[].dst.data)


def _score_multi(slot: _SlotPtr, call: _CallPtr) raises:
    """`score(X, y)`: MultiOutputClassifier the share of rows whose every
    output is predicted (`np.mean(np.all(y == y_pred, axis=1))`);
    MultiOutputRegressor `r2_score(y, y_pred)` averaged uniformly over the
    outputs (float64 sums; an output with no variance scores 1.0 when its
    residual is 0, else 0.0, scikit-learn's `force_finite`). Weights: the
    weighted forms of both."""
    _fitted(slot)
    var name = _name(slot[].kind)
    var rows = _rows(call, name)
    var n = rows.n
    var outputs = len(slot[].states)
    var classifier = slot[].kind == META_MULTI_CLASSIFIER
    var y_dtype = MEMBER_I32 if classifier else MEMBER_F32
    if not call[].y.dense_host(y_dtype) or call[].y.rows != n or call[].y.cols != outputs:
        raise Error("TypeError: a " + name + " member scores y as dense host rows x outputs")
    var has_w = call[].weights.given()
    if has_w and (not call[].weights.dense_host(MEMBER_F32) or call[].weights.rows != n):
        raise Error("TypeError: a " + name + " member reads the sample weights as one dense host float32 column")
    var pred = List[Float64](length=max(n * outputs, 1), fill=0.0)
    _multi_outputs(slot, rows, MEMBER_F64, Int(pred.unsafe_ptr()))
    var w = _F32P(unsafe_from_address=call[].weights.data)
    if classifier:
        var y = _I32P(unsafe_from_address=call[].y.data)
        var hit = 0.0
        var total = 0.0
        for i in range(n):  # small-loop(the rows of one score, after the outputs' parallel predicts)
            var all_equal = True
            for j in range(outputs):
                if Int(pred[i * outputs + j]) != Int(y[i * outputs + j]):
                    all_equal = False
            var weight = Float64(w[i]) if has_w else 1.0
            total += weight
            if all_equal:
                hit += weight
        call[].value = hit / total if total != 0.0 else 0.0
        return
    var scores = List[Float64](length=max(outputs, 1), fill=0.0)
    var scores_at = Int(scores.unsafe_ptr())
    var pred_at = Int(pred.unsafe_ptr())
    var y_at = call[].y.data
    var w_at = call[].weights.data

    def _r2(j: Int) {imm scores_at, imm pred_at, imm y_at, imm w_at, imm has_w, imm n, imm outputs}:
        var yy = _F32P(unsafe_from_address=y_at)
        var pp = _F64P(unsafe_from_address=pred_at)
        var ww = _F32P(unsafe_from_address=w_at)
        var sw = 0.0
        var mean = 0.0
        for i in range(n):
            var weight = Float64(ww[i]) if has_w else 1.0
            sw += weight
            mean += weight * Float64(yy[i * outputs + j])
        mean = mean / sw if sw != 0.0 else 0.0
        var num = 0.0
        var den = 0.0
        for i in range(n):
            var weight = Float64(ww[i]) if has_w else 1.0
            var t = Float64(yy[i * outputs + j])
            var r = t - pp[i * outputs + j]
            num += weight * r * r
            den += weight * (t - mean) * (t - mean)
        var value = 1.0
        if den != 0.0:
            value = 1.0 - num / den
        elif num != 0.0:
            value = 0.0
        _F64P(unsafe_from_address=scores_at)[j] = value

    host_parallelize(_r2, outputs)
    var total = 0.0
    for j in range(outputs):  # small-loop(the outputs' scores, averaged uniformly)
        total += scores[j]
    call[].value = total / Float64(outputs)
    _ = len(pred)
    _ = len(scores)


# ------------------------------------------------------------ THE RFE KIND (lane mf-meta5)


def _rfe_mask(slot: _SlotPtr) -> Int:
    """THE RFE KIND's kind-exact mask: fit, predict, transform, export and import, and what its base (the
    final estimator) answers: predict_proba, decision_function, score, sample weights (RFE.fit hands
    `sample_weight` to every fit). THE GIL RULE: MEMBER_GIL_DOOR at most, when the base has either bit (the
    rounds' x_prep programs run on the calling thread, as the naive Bayes members' do: the transports are
    not traced for parallel workers). MEMBER_CAN_SCOPE when the image keeps a pinned X resident in a
    caller's scope. No MEMBER_BORROWS_FIT: the rounds read X during the fit only and the final estimator
    owns its rows (`rfe_fit_run`)."""
    var mask = MEMBER_CAN_FIT | MEMBER_CAN_PREDICT | MEMBER_CAN_TRANSFORM | MEMBER_CAN_EXPORT | MEMBER_CAN_IMPORT
    if slot[].kind == META_RFE_CLASSIFIER:
        mask |= MEMBER_CAN_CLASSIFIER
    var base = slot[].base_masks[0] if len(slot[].base_masks) > 0 else 0
    mask |= base & (MEMBER_CAN_PREDICT_PROBA | MEMBER_CAN_DECISION | MEMBER_CAN_SCORE | MEMBER_CAN_SAMPLE_WEIGHT)
    if (base & (MEMBER_GIL_FREE | MEMBER_GIL_DOOR)) != 0:
        mask |= MEMBER_GIL_DOOR
    if len(slot[].words) == META_RFE_WORDS and slot[].words[META_RFE_SCOPE] != 0.0:
        mask |= MEMBER_CAN_SCOPE
    return mask


def _rfe_selected(words: List[Float64], d: Int) -> Int:
    """n_features_to_select resolved over `d` columns (`rfe_select_count`)."""
    return rfe_select_count(d, Int(words[META_RFE_SELECT_KIND]), words[META_RFE_SELECT_FRACTION],
                            Int(words[META_RFE_SELECT_COUNT]))


def _fit_rfe(slot: _SlotPtr, call: _CallPtr) raises:
    """RFE.fit(X, y[, sample_weight]) over its member, the door's RUN whole (bindings/prep_select_run.mojo
    `rfe_fit_run`, THE ONE BODY: the same rounds, programs and final fit as
    bindings/prep_select_native_door.mojo `_rfe_rounds`). The final estimator is the state's one nested
    member; the rounds' and the final fit's notices go to the PACK with the estimator's context; the
    support, ranking and kept columns are the state's (META_BLOCK_RFE)."""
    var classifier = slot[].kind == META_RFE_CLASSIFIER
    var n = call[].x.rows
    var d = call[].x.cols
    var words = slot[].words.copy()
    # The door's order: n_features_to_select, then step (its refusal), before any fit.
    var nsel = _rfe_selected(words, d)
    var step = rfe_step_count(d, words[META_RFE_STEP_FRACTION] != 0.0, words[META_RFE_STEP_VALUE],
                              Int(words[META_RFE_STEP_COUNT]))
    var member = slot[].base_plans[0].copy()
    member.classes = call[].classes if classifier else 0
    var plan = RfeFitPlan(member^, slot[].rfe_run, words[META_RFE_WORK_ON] != 0.0, call[].x.data, n, d,
                          call[].y.data, MEMBER_I32 if classifier else MEMBER_F32)
    plan.nsel = nsel
    plan.step = step
    plan.block = Int(words[META_RFE_BLOCK])
    plan.squared = words[META_RFE_SQUARED] != 0.0
    plan.importance_word = words[META_RFE_IMPORTANCE_WORD] != 0.0
    plan.refusal = slot[].refusal.copy()
    # A dotted importance_getter over a META base (lane mf-del5b): THE FORWARDED EXPORT's hops, as the
    # door's `_rfe_native` hands them (`plan.block` is then META_BLOCK_FORWARD).
    plan.hops = slot[].rfe_hops.copy()
    if slot[].fit_given != 0 and len(slot[].base_masks) > 0 and (slot[].base_masks[0] & MEMBER_READS_Y_GIVEN) != 0:
        # THE GIVEN y (lane mf-tail5): every round's and the final fit's (the same rows).
        plan.extras.y_given = slot[].fit_given
        plan.extras.y_given_dtype = slot[].fit_given_dtype
    if call[].weights.given():
        # RFE.fit's sample_weight, handed to every round's fit and the final fit (scikit-learn's
        # `**fit_params`): the caller's float32 column, as the base's own fit reads it.
        plan.extras.rows = n
        plan.extras.weights = call[].weights.data
    # THE RESIDENT ROUTE: a caller in a resident scope hands the rounds its scope and X's pin (THE PIN
    # RULE: the run writes this call's `pin_kept` the moment the scope's cache takes the reference).
    if (_rfe_mask(slot) & MEMBER_CAN_SCOPE) != 0 and call[].scope != MEMBER_SCOPE_NONE and call[].pin != 0:
        plan.scope = call[].scope
        plan.pin = call[].pin
        plan.kept_at = Int(UnsafePointer(to=call[].pin_kept))
    var res = rfe_fit_run(plan)
    slot[].plans = [res.final_plan.copy()]
    slot[].masks = [slot[].base_masks[0]]
    slot[].states = [res.final_state]
    for i in range(len(res.notices)):  # driver-loop: over the member fits' notices, in fit order
        slot[].notice_owner.append(0)
        slot[].notice_words.append(res.notices[i].copy())
    slot[].classes = call[].classes if classifier else 0
    slot[].rfe_support = res.support.copy()
    slot[].rfe_ranking = res.ranking.copy()
    slot[].rfe_columns = res.columns.copy()


def _rfe_view(slot: _SlotPtr):
    """THE RFE KIND's export view (META_BLOCK_RFE's layout)."""
    var d = len(slot[].rfe_support)
    var k = len(slot[].rfe_columns)
    var w = List[Float64](capacity=2 + 2 * d + k)
    w.append(Float64(d))
    w.append(Float64(k))
    for j in range(d):  # small-loop(the fitted support, feature count)
        w.append(Float64(slot[].rfe_support[j]))
    for j in range(d):  # small-loop(the fitted ranking, feature count)
        w.append(Float64(slot[].rfe_ranking[j]))
    for j in range(k):  # small-loop(the kept column ids)
        w.append(Float64(Int(slot[].rfe_columns[j])))
    slot[].rfe_words = w^


def _import_rfe(slot: _SlotPtr, words: _F64P, count: Int) raises:
    """IMPORT META_BLOCK_RFE (`_rfe_view`'s layout), after META_BLOCK_NESTED (the final estimator)."""
    var bad = String("member ABI: an RFE member's words are malformed")
    if count < 2:
        raise Error(bad)
    var d = Int(words[0])
    var k = Int(words[1])
    if d < 1 or k < 1 or k > d or count != 2 + 2 * d + k:
        raise Error(bad)
    slot[].rfe_support = List[Float32](length=d, fill=0.0)
    slot[].rfe_ranking = List[Float32](length=d, fill=0.0)
    slot[].rfe_columns = List[Int32](length=k, fill=Int32(0))
    for j in range(d):  # small-loop(the imported support and ranking, feature count)
        slot[].rfe_support[j] = Float32(words[2 + j])
        slot[].rfe_ranking[j] = Float32(words[2 + d + j])
    for j in range(k):  # small-loop(the imported kept column ids)
        var c = Int(words[2 + 2 * d + j])
        if c < 0 or c >= d:
            raise Error(bad)
        slot[].rfe_columns[j] = Int32(c)
    slot[].features = d


def _rfe_block(slot: _SlotPtr, call: _CallPtr, name: String) raises -> List[Float32]:
    """`self.transform(X)` (`_SelectorMixin.transform`): the kept columns of the caller's X, gathered into a
    new n x k float32 block (bit for bit the door's gather), after its feature check."""
    var rows = _rows(call, name)
    var d = len(slot[].rfe_support)
    if rows.d != d:
        raise Error("mojolearn: X has " + String(rows.d) + " features, RFE was fitted with " + String(d))
    var n = rows.n
    var k = len(slot[].rfe_columns)
    var block = List[Float32](length=max(n * k, 1), fill=0.0)
    if n == 0:
        return block^
    var order = List[Int32](length=n, fill=Int32(0))
    for i in range(n):  # small-loop(the row numbers 0 .. n-1 of the column gather, once per call)
        order[i] = Int32(i)
    _ = xt_gather_f32(slot[].glue, rows.x, Int(order.unsafe_ptr()), Int(slot[].rfe_columns.unsafe_ptr()),
                      Int(block.unsafe_ptr()), n, d, n, k)
    _ = len(order)  # read through the gather above
    return block^


def _call_rfe(slot: _SlotPtr, call: _CallPtr, operation: Int) raises:
    """RFE.predict / predict_proba / decision_function / score / transform: the kept columns of X, then the
    final estimator's own call on them (`self.estimator_.<op>(self.transform(X)[, y])`); transform writes
    the block. The caller's dst, y, weights and flags are the final call's."""
    _fitted(slot)
    var name = _name(slot[].kind)
    if call[].row_index.rows > 0 or call[].column_index.rows > 0:
        raise Error("TypeError: an RFE member reads whole host views (no row or column index)")
    var mask = _rfe_mask(slot)
    if operation == MEMBER_OP_PREDICT_PROBA and (mask & MEMBER_CAN_PREDICT_PROBA) == 0:
        member_unsupported(name, operation)
    if operation == MEMBER_OP_DECISION and (mask & MEMBER_CAN_DECISION) == 0:
        member_unsupported(name, operation)
    if operation == MEMBER_OP_SCORE and (mask & MEMBER_CAN_SCORE) == 0:
        member_unsupported(name, operation)
    var block = _rfe_block(slot, call, name)
    var n = call[].x.rows
    var k = len(slot[].rfe_columns)
    if operation == MEMBER_OP_TRANSFORM:
        if not call[].dst.dense_host(MEMBER_F32) or call[].dst.rows != n or call[].dst.cols != k:
            raise Error("TypeError: an RFE member writes transform as dense host float32 rows of its "
                        + String(k) + " kept columns")
        if n * k > 0:
            unsafe_memcpy(dest=_F32P(unsafe_from_address=call[].dst.data), src=block.unsafe_ptr(), count=n * k)
        call[].out_cols = k
        return
    var plan = slot[].plans[0].copy()
    var inner = member_call_of(plan, slot[].states[0])
    inner.x = MemberMatrix(Int(block.unsafe_ptr()), n, k)
    inner.y = call[].y
    inner.weights = call[].weights
    inner.dst = call[].dst
    inner.flags = call[].flags
    _ = member_call(plan.member, operation, inner)
    _ = len(block)  # read through the final estimator's call above
    call[].value = inner.value
    call[].out_cols = inner.out_cols


def _outputs_rfe_fitted(call: _CallPtr, slot: _SlotPtr) raises:
    """THE RFE KIND's outputs after a fit or an import: transform its kept columns (float32), every other
    operation its final estimator's."""
    var asked = call[].flags
    if asked == MEMBER_OP_TRANSFORM:
        call[].out_cols = len(slot[].rfe_columns)
        call[].out_dtype = MEMBER_F32
    else:
        var plan = slot[].plans[0].copy()
        var inner = member_call_of(plan, slot[].states[0])
        call[].out_cols = member_outputs_of(plan, inner, asked)
        call[].out_dtype = inner.out_dtype
    call[].classes = slot[].classes


def _outputs_rfe(call: _CallPtr, base: MemberPlan, words: List[Float64], classes: Int) raises:
    """THE RFE KIND's outputs before a fit: transform n_features_to_select over the caller's X width
    (float32), every other operation its estimator's plan's."""
    var asked = call[].flags
    if asked == MEMBER_OP_TRANSFORM:
        call[].out_cols = _rfe_selected(words, max(call[].x.cols, 1))
        call[].out_dtype = MEMBER_F32
        call[].classes = classes
        return
    var plan = base.copy()
    plan.classes = classes
    call[].out_cols = member_outputs(plan, asked, max(call[].x.cols, 1), max(call[].y.cols, 1))
    call[].out_dtype = member_output_dtype(plan, asked, max(call[].x.cols, 1), max(call[].y.cols, 1),
                                           MEMBER_DTYPE_UNSAID)
    call[].classes = classes


def _fit(slot: _SlotPtr, call: _CallPtr) raises:
    # THE DOOR'S REFUSAL (lane mf-refuse5, `meta_plan_refuse`): the door's words for a constructor value,
    # first, as the door's validation runs before it reads X.
    if _refused_first(slot):
        raise Error(slot[].refusal.copy())
    var name = _name(slot[].kind)
    if not call[].x.dense_host(MEMBER_F32):
        raise Error("TypeError: a " + name + " member reads X as a dense host float32 view")
    if call[].row_index.rows > 0 or call[].column_index.rows > 0:
        raise Error("TypeError: a " + name + " member reads whole host views (no row or column index)")
    var y_dtype = MEMBER_I32 if _classifier_kind(slot[].kind) else MEMBER_F32
    # THE 2-D y (lane mf-search4): a multi-output kind, or a search whose
    # every candidate is one, reads rows x outputs.
    var two_d = _multi_kind(slot[].kind) or (_search_kind(slot[].kind)
                                              and _all(slot[].base_masks, MEMBER_CAN_MULTI_OUTPUT))
    # A SEARCH WITHOUT A TARGET (lane mf-msel5b): an unsupervised base fitted
    # with y=None hands no y (the search RUN's target-free words).
    var y_free = (_search_kind(slot[].kind) and not _classifier_kind(slot[].kind) and call[].y.rows == 0
                  and call[].y.data == 0)
    if not y_free and (not call[].y.dense_host(y_dtype) or call[].y.rows != call[].x.rows
            or (call[].y.cols != 1 and not two_d) or call[].y.cols < 1):
        raise Error("TypeError: a " + name + " member reads y as one dense host column of "
                    + (String("int32 class codes") if y_dtype == MEMBER_I32 else String("float32 targets")))
    if call[].weights.given() and (not call[].weights.dense_host(MEMBER_F32) or call[].weights.rows != call[].x.rows):
        raise Error("TypeError: a " + name + " member reads the sample weights as one dense host float32 column")
    if _classifier_kind(slot[].kind) and call[].classes < 1 and call[].class_counts == 0:
        raise Error("member ABI: a " + name + " member needs its class count at fit")
    if call[].weights.given() and (slot[].kind == META_OVR or slot[].kind == META_CALIBRATED
                                   or _search_kind(slot[].kind)):
        raise Error("TypeError: a " + name + " member's fit takes no sample weights")
    if call[].weights.given() and _rfe_kind(slot[].kind) and (_rfe_mask(slot) & MEMBER_CAN_SAMPLE_WEIGHT) == 0:
        raise Error("TypeError: an RFE member's estimator fits no sample weights")
    _drop_states(slot)
    slot[].features = call[].x.cols
    # THE GIVEN y (lane mf-tail5): a regressor's, for this fit only.
    slot[].fit_y = call[].y.data
    slot[].fit_given = 0
    slot[].fit_given_dtype = MEMBER_F32
    var g = call[].y_given
    if (not _classifier_kind(slot[].kind) and g.given() and g.home == MEMBER_HOME_HOST and g.rows == call[].y.rows
            and g.cols == call[].y.cols and g.stride == g.cols and g.offset == 0
            and (g.dtype == MEMBER_F64 or g.dtype == MEMBER_I64 or g.dtype == MEMBER_I32)):
        slot[].fit_given = g.data
        slot[].fit_given_dtype = g.dtype
    try:
        if _stack_kind_code(slot[].kind):
            _fit_stack(slot, call)
        elif slot[].kind == META_OVR:
            _fit_ovr(slot, call)
        elif slot[].kind == META_CALIBRATED:
            _fit_cal(slot, call)
        elif _bag_kind(slot[].kind):
            _fit_bag(slot, call)   # its own `_nested_outputs` (the out-of-bag outputs read them)
        elif _ada_kind(slot[].kind):
            _fit_ada(slot, call)
        elif _search_kind(slot[].kind):
            _fit_search(slot, call)
        elif _multi_kind(slot[].kind):
            _fit_multi(slot, call)
        elif _rfe_kind(slot[].kind):
            _fit_rfe(slot, call)
        else:
            _fit_vote(slot, call)
        if slot[].refusal_late and slot[].refusal.byte_length() != 0 and not _rfe_kind(slot[].kind):
            # THE DOOR'S REFUSAL after the nested fits (lane mf-refuse5): the door checks it there.
            raise Error(slot[].refusal.copy())
        if not _bag_kind(slot[].kind):
            _nested_outputs(slot)
        _unit_views(slot)
        slot[].fitted = True
    except error:
        slot[].fit_y = 0
        slot[].fit_given = 0
        _drop_states(slot)
        raise error^
    slot[].fit_y = 0
    slot[].fit_given = 0  # read during the fit only (core/member_abi.mojo `MemberCall.y_given`)


# ------------------------------------------------------------------ PREDICT


def _fitted(slot: _SlotPtr) raises:
    if not slot[].fitted:
        raise Error("NotFittedError: this " + _name(slot[].kind) + " member is not fitted yet")


def _rows(call: _CallPtr, name: String) raises -> VoteRows:
    if not call[].x.dense_host(MEMBER_F32):
        raise Error("TypeError: a " + name + " member reads X as a dense host float32 view")
    return VoteRows(call[].x.data, call[].x.rows, call[].x.cols)


def _zero_dst(call: _CallPtr, count: Int):
    var p = _F64P(unsafe_from_address=call[].dst.data)
    for i in range(count):  # small-loop(the float64 accumulator the vote RUN sums into, zeroed once per call)
        p[i] = 0.0


def _predict_vote(slot: _SlotPtr, call: _CallPtr, operation: Int) raises:
    var name = _name(slot[].kind)
    var rows = _rows(call, name)
    var n = rows.n
    var m = len(slot[].plans)
    var soft = slot[].flag
    var k = slot[].classes
    var members = List[VoteMember](capacity=m)
    var op = MEMBER_OP_PREDICT
    if slot[].kind == META_VOTE_CLASSIFIER and soft:
        op = MEMBER_OP_PREDICT_PROBA
    for u in range(m):  # driver-loop: over the voting members
        members.append(_member_of(slot, u, slot[].states[u], op, slot[].weights[u]))
    var refused = VOTE_ALL_ANSWERED
    if slot[].kind == META_VOTE_REGRESSOR:
        if operation != MEMBER_OP_PREDICT:
            member_unsupported(name, operation)
        if not call[].dst.dense_host(MEMBER_F64) or call[].dst.rows != n:
            raise Error("TypeError: a VotingRegressor member writes predict as one dense host float64 column")
        _zero_dst(call, n)
        refused = vote_regress_run(slot[].glue, rows, members, call[].dst.data)
    elif operation == MEMBER_OP_PREDICT:
        if not call[].dst.dense_host(MEMBER_I32) or call[].dst.rows != n:
            raise Error("TypeError: a VotingClassifier member writes predict as one dense host int32 column")
        var acc = List[Float64](length=max(n * k, 1), fill=0.0)
        refused = vote_codes_run(slot[].glue, rows, members, k, soft, Int(acc.unsafe_ptr()), call[].dst.data)
        _ = len(acc)  # read through the RUN above
    elif operation == MEMBER_OP_PREDICT_PROBA and soft:
        if not call[].dst.dense_host(MEMBER_F64) or call[].dst.rows != n or call[].dst.cols != k:
            raise Error("TypeError: a VotingClassifier member writes predict_proba as dense host float64 rows")
        _zero_dst(call, n * k)
        refused = vote_soft_run(slot[].glue, rows, members, k, call[].dst.data)
    else:
        member_unsupported(name, operation)
    if refused != VOTE_ALL_ANSWERED:
        _raise_member(members[refused], rows)


def _predict_stack(slot: _SlotPtr, call: _CallPtr, operation: Int) raises:
    var name = _name(slot[].kind)
    var rows = _rows(call, name)
    var n = rows.n
    var d = rows.d
    var bases = len(slot[].stack_kinds)
    var k = slot[].classes
    var width = slot[].total + (d if slot[].flag else 0)
    var every = List[Int32]()
    var counts = _stack_counts(slot, k, every)
    var keep = List[List[Float32]]()
    var members = _stack_members(slot, slot[].states, 0, rows, every, keep)
    var meta = List[Float64](length=max(n * width, 1), fill=0.0)
    var order = List[Int32](length=max(n, 1), fill=Int32(0))
    for i in range(n):  # small-loop(the row numbers 0 .. n-1 the scatter places, once per call)
        order[i] = Int32(i)
    var got = List[Int]()
    var refused = stack_place_run(slot[].glue, rows, members, slot[].stack_kinds, counts, k, k == 2,
                                  Int(meta.unsafe_ptr()), n, width, Int(order.unsafe_ptr()), 0, got, 0)
    _ = len(keep)  # the label blocks are read through the RUN above
    if refused != VOTE_ALL_ANSWERED:
        _raise_member(members[refused], rows)
    if slot[].flag:
        _ = xt_scatter(slot[].glue, Int(meta.unsafe_ptr()), rows.x, Int(order.unsafe_ptr()), n, width, n, d,
                       slot[].total)
    var meta32 = List[Float32](length=max(n * width, 1), fill=0.0)
    for i in range(n * width):  # small-loop(one rounding pass of the meta features, as the door's float32 admission)
        meta32[i] = meta[i].cast[DType.float32]()
    _ = len(meta)
    _ = len(order)
    var plan = slot[].plans[bases].copy()
    var final = member_call_of(plan, slot[].states[bases])
    final.x = MemberMatrix(Int(meta32.unsafe_ptr()), n, width)
    final.dst = call[].dst
    _ = member_call(plan.member, operation, final)
    _ = len(meta32)  # read through the final member's call above


def _codes_dst(call: _CallPtr, n: Int, name: String) raises:
    if not call[].dst.dense_host(MEMBER_I32) or call[].dst.rows != n:
        raise Error("TypeError: a " + name + " member writes predict as one dense host int32 column")


def _proba_dst(call: _CallPtr, n: Int, k: Int, name: String) raises:
    if not call[].dst.dense_host(MEMBER_F64) or call[].dst.rows != n or call[].dst.cols != k:
        raise Error("TypeError: a " + name + " member writes predict_proba as dense host float64 rows")


def _predict_ovr(slot: _SlotPtr, call: _CallPtr, operation: Int) raises:
    """`_ovr_codes` / `_ovr_predict_proba` (module note)."""
    var name = _name(slot[].kind)
    var rows = _rows(call, name)
    var n = rows.n
    var k = slot[].classes
    var glue = slot[].glue
    if operation == MEMBER_OP_PREDICT:
        _codes_dst(call, n, name)
        if k == 2:
            _codes_into(slot, 0, slot[].states[0], rows, call[].dst.data)
            return
    elif operation == MEMBER_OP_PREDICT_PROBA:
        _proba_dst(call, n, k, name)
    else:
        member_unsupported(name, operation)
    if operation == MEMBER_OP_PREDICT_PROBA and k == 2:
        # `_proba_positive` then `_trees_binary_proba`: the unit's code-1
        # probabilities (float32, as the door's admission rounds them).
        var keep = List[List[Float32]]()
        var m = _member_of(slot, 0, slot[].states[0], MEMBER_OP_PREDICT_PROBA, 1.0)
        var block = member_f32_block(m, rows, keep)
        var col = _index_of_code(List[Int32](), 1)
        if col >= m.cols:
            col = -1
        var p = List[Float32](length=max(n, 1), fill=0.0)
        if col >= 0:
            var b = _F32P(unsafe_from_address=block)
            for i in range(n):  # small-loop(one column copy of one member's block)
                p[i] = b[i * m.cols + col]
        _ = len(keep)
        _ = xt_binary_proba(glue, Int(p.unsafe_ptr()), call[].dst.data, n)
        _ = len(p)
        return
    # k > 2: each unit's decision column (the base has decision_function and
    # predict asks), else its probability column of code 1.
    var decision = operation == MEMBER_OP_PREDICT and slot[].words[META_OVR_DECISION] != 0.0
    var op = MEMBER_OP_DECISION if decision else MEMBER_OP_PREDICT_PROBA
    var members = List[VoteMember](capacity=k)
    var at = List[Int](capacity=k)
    for u in range(k):  # driver-loop: over the units
        members.append(_member_of(slot, u, slot[].states[u], op, 1.0))
        at.append(0 if decision else 1)
    if operation == MEMBER_OP_PREDICT_PROBA:
        _zero_dst(call, n * k)
    var refused = ovr_assemble_run(glue, rows, members, at, k, 2 if operation == MEMBER_OP_PREDICT else 1,
                                   call[].dst.data)
    if refused != VOTE_ALL_ANSWERED:
        _raise_member(members[refused], rows)


def _predict_cal(slot: _SlotPtr, call: _CallPtr, operation: Int) raises:
    """`_calibrated_fused` over the kept pairs (`calibrated_pair_words`),
    then the row argmax for predict (`_calibrated_codes`)."""
    var name = _name(slot[].kind)
    var rows = _rows(call, name)
    var n = rows.n
    var k = slot[].classes
    var glue = slot[].glue
    if operation == MEMBER_OP_PREDICT:
        _codes_dst(call, n, name)
    elif operation == MEMBER_OP_PREDICT_PROBA:
        _proba_dst(call, n, k, name)
    else:
        member_unsupported(name, operation)
    var op = MEMBER_OP_DECISION if slot[].words[META_CAL_DECISION] != 0.0 else MEMBER_OP_PREDICT_PROBA
    var pairs = len(slot[].states)
    var members = List[VoteMember](capacity=pairs)
    var cal_pairs = List[CalPair](capacity=pairs)
    var cals = List[CalFit]()
    var placed = List[List[Int32]]()
    for i in range(pairs):  # driver-loop: over the (member, calibrators) pairs (their words, once per call)
        var m = _member_of(slot, i, slot[].states[i], op, 1.0)
        var cols = m.cols
        var at = 0
        var cols_at = 0
        var ks = 0
        var used = slot[].pair_count[i]
        if k == 2:
            if cols != 1:
                at = _index_of_code(slot[].unit_held[i], 1)
                meta_binary_column_check(at)
        elif len(slot[].unit_held[i]) == 0:
            cols_at = Int(slot[].iota.unsafe_ptr())
            ks = k
        else:
            # THE PLACEMENT RULE of a pair whose member lacks a class (lane mf-cal3b).
            var place = cal_fold_place(k, slot[].unit_held[i], op, cols, True, 0)
            # scikit-learn's predict zips the score columns with the calibrators: a pair fitted with more
            # calibrators than its member's columns place (ensemble=False's full fit with the unlabeled marker,
            # lane mf-meta5b: zipped there with the out-of-fold columns) reads the first of them.
            if len(place.calibrate) > slot[].pair_count[i]:
                raise Error("ValueError: mojolearn: CalibratedClassifierCV: a pair holds " + String(slot[].pair_count[i])
                            + " calibrators for a member placed at " + String(len(place.calibrate)) + " columns")
            used = len(place.calibrate)
            placed.append(place.place.copy())
            cols_at = Int(placed[len(placed) - 1].unsafe_ptr())
            ks = len(place.place)
        m.class_cols = cols_at
        members.append(m)
        cal_pairs.append(CalPair(at, cols_at, ks, len(cals), used))
        for j in range(slot[].pair_first[i], slot[].pair_first[i] + used):  # driver-loop: over the pair's calibrators
            cals.append(CalFit(slot[].cal_iso[j], slot[].cal_a[j], slot[].cal_b[j],
                               Int(slot[].cal_kx[j].unsafe_ptr()), Int(slot[].cal_ky[j].unsafe_ptr()), slot[].cal_m[j]))
    var scratch = List[Float64]()
    var acc = call[].dst.data
    if operation == MEMBER_OP_PREDICT:
        scratch = List[Float64](length=max(n * k, 1), fill=0.0)
        acc = Int(scratch.unsafe_ptr())
    else:
        _zero_dst(call, n * k)
    var refused = calibrated_proba_run(glue, rows, members, cal_pairs, cals, k, acc)
    _ = len(placed)  # the placed class codes are read through the RUN above
    if refused != VOTE_ALL_ANSWERED:
        _raise_member(members[refused], rows)
    if operation == MEMBER_OP_PREDICT:
        _ = xt_argmax_rows(glue, acc, call[].dst.data, n, k)
    _ = len(scratch)


def _predict_bag(slot: _SlotPtr, call: _CallPtr, operation: Int) raises:
    """`_bagging_proba` / `bagging_predict`: each unit over its feature
    subset of X (`_sub_x`), `bag_predict_run`, then the row argmax for a
    classifier's predict (`bagging_codes`)."""
    var name = _name(slot[].kind)
    var classifier = slot[].kind == META_BAG_CLASSIFIER
    var rows = _rows(call, name)
    var n = rows.n
    var d = rows.d
    var k = slot[].classes if classifier else 1
    var glue = slot[].glue
    if operation == MEMBER_OP_PREDICT:
        if classifier:
            _codes_dst(call, n, name)
        elif not call[].dst.dense_host(MEMBER_F64) or call[].dst.rows != n:
            raise Error("TypeError: a BaggingRegressor member writes predict as one dense host float64 column")
    elif operation == MEMBER_OP_PREDICT_PROBA and classifier:
        _proba_dst(call, n, k, name)
    else:
        member_unsupported(name, operation)
    if slot[].batched:
        _predict_batched(slot, call, operation, rows, k)
        return
    var count = len(slot[].states)
    var order = List[Int32](length=max(n, 1), fill=Int32(0))
    for i in range(n):  # small-loop(the row numbers 0 .. n-1 of the feature gathers, once per call)
        order[i] = Int32(i)
    var blocks = List[List[Float32]](capacity=count)
    var xs = List[Int](capacity=count)
    var ds = List[Int](capacity=count)
    var members = List[VoteMember](capacity=count)
    var unit_op = MEMBER_OP_PREDICT_PROBA if classifier else MEMBER_OP_PREDICT
    for u in range(count):  # driver-loop: over the member estimators (their feature subsets, `_sub_x`)
        var c = len(slot[].unit_cols[u])
        var block = List[Float32]()
        if c > 0 and not _all_columns(slot[].unit_cols[u], d):
            block = List[Float32](length=n * c, fill=0.0)
            _ = xt_gather_f32(glue, rows.x, Int(order.unsafe_ptr()), Int(slot[].unit_cols[u].unsafe_ptr()),
                              Int(block.unsafe_ptr()), n, d, n, c)
        blocks.append(block^)
        var m = _member_of(slot, u, slot[].states[u], unit_op, 1.0)
        if classifier and len(slot[].unit_held[u]) > 0:
            m.class_cols = Int(slot[].unit_held[u].unsafe_ptr())
        members.append(m)
    for u in range(count):  # driver-loop: over the member estimators (their rows' addresses, now final)
        if len(blocks[u]) > 0:
            xs.append(Int(blocks[u].unsafe_ptr()))
            ds.append(len(slot[].unit_cols[u]))
        else:
            xs.append(rows.x)
            ds.append(d)
    var scratch = List[Float64]()
    var acc = call[].dst.data
    if classifier and operation == MEMBER_OP_PREDICT:
        scratch = List[Float64](length=max(n * k, 1), fill=0.0)
        acc = Int(scratch.unsafe_ptr())
    else:
        _zero_dst(call, n * k)
    # ONE DEVICE COPY OF X PER PREDICT (lane mf-meta3c; the door's
    # `_staged_input`, bindings/trees_staged_input_door.mojo): when two or
    # more units read X itself (every feature) through a forest image on a
    # device, X is staged once in that image and every unit's resident
    # predict reads the staged rows (no upload or scan per unit). Units over
    # a feature subset read their own gathered block and upload it, as the
    # door's did.
    var staged = _stage_units(slot, rows.x, n, d, xs)
    var refused = VOTE_ALL_ANSWERED
    try:
        refused = bag_predict_run(glue, members, xs, ds, n, k, classifier, acc, 0)
    finally:
        if staged:
            xf_unstage_input(entry_ref_gil_free(slot[].base_plans[0].route.entry))
    _ = len(blocks)  # read through the RUN above
    _ = len(order)
    if refused != VOTE_ALL_ANSWERED:
        _raise_member(members[refused], VoteRows(xs[refused], n, ds[refused]))
    if classifier and operation == MEMBER_OP_PREDICT:
        _ = xt_argmax_rows(glue, acc, call[].dst.data, n, k)
    _ = len(scratch)


def _stage_units(slot: _SlotPtr, x: Int, n: Int, d: Int, xs: List[Int]) raises -> Bool:
    """Stage X (`x`, n x d float32 host rows) once in the forest image of a
    unit kind's base when two or more units read it whole and the base is a
    tree, forest or extra-trees member on a device route (the door's rule:
    staged in each binary two or more members predict through). True: the
    caller unstages (`xf_unstage_input`) after the units' predictions, also
    on an error. A host image or a build that cannot stage answers False and
    every unit uploads as before."""
    if len(slot[].base_plans) == 0 or n < 1 or d < 1:
        return False
    var base = slot[].base_plans[0].copy()
    if base.route.device == DEVICE_CPU:
        return False
    if (base.family != MEMBER_FAMILY_TREE and base.family != MEMBER_FAMILY_FOREST
            and base.family != MEMBER_FAMILY_EXTRA_TREES):
        return False
    var whole = 0
    for u in range(len(xs)):  # driver-loop: over the units (which read X itself)
        if xs[u] == x:
            whole += 1
    if whole < 2:
        return False
    return xf_stage_input(entry_ref_gil_free(base.route.entry), x, n, d)


def _all_columns(cols: List[Int32], d: Int) -> Bool:
    """`cols.tolist() == list(range(d))` (`_BaggingBase._sub_X`'s rule)."""
    if len(cols) != d:
        return False
    for j in range(d):  # small-loop(a member's feature subset)
        if Int(cols[j]) != j:
            return False
    return True


def _predict(slot: _SlotPtr, call: _CallPtr, operation: Int) raises:
    _fitted(slot)
    if _stack_kind_code(slot[].kind):
        if operation == MEMBER_OP_PREDICT_PROBA or operation == MEMBER_OP_DECISION:
            if slot[].kind != META_STACK_CLASSIFIER:
                member_unsupported(_name(slot[].kind), operation)
        _predict_stack(slot, call, operation)
    elif slot[].kind == META_OVR:
        _predict_ovr(slot, call, operation)
    elif slot[].kind == META_CALIBRATED:
        _predict_cal(slot, call, operation)
    elif _bag_kind(slot[].kind):
        _predict_bag(slot, call, operation)
    elif _ada_kind(slot[].kind):
        _predict_ada(slot, call, operation)
    elif _search_kind(slot[].kind):
        _forward_best(slot, call, operation)
    elif _multi_kind(slot[].kind):
        _predict_multi(slot, call, operation)
    elif _rfe_kind(slot[].kind):
        _call_rfe(slot, call, operation)
    else:
        _predict_vote(slot, call, operation)


# ------------------------------------------------------------------ OUTPUTS


def _outputs_vote(call: _CallPtr, kind: Int, soft: Bool, classes: Int) raises:
    if call[].flags == MEMBER_OP_PREDICT:
        call[].out_cols = 1
        call[].out_dtype = MEMBER_I32 if kind == META_VOTE_CLASSIFIER else MEMBER_F64
    elif call[].flags == MEMBER_OP_PREDICT_PROBA and kind == META_VOTE_CLASSIFIER and soft:
        call[].out_cols = classes
        call[].out_dtype = MEMBER_F64
    else:
        member_unsupported(_name(kind), call[].flags)
    call[].classes = classes if kind == META_VOTE_CLASSIFIER else 0


def _outputs_unit(call: _CallPtr, kind: Int, classes: Int) raises:
    """A unit kind's outputs: predict one column (int32 codes for a
    classifier, float64 for BaggingRegressor), predict_proba `classes`
    float64 columns (classifiers)."""
    var classifier = _classifier_kind(kind)
    if call[].flags == MEMBER_OP_PREDICT:
        call[].out_cols = 1
        call[].out_dtype = MEMBER_I32 if classifier else MEMBER_F64
    elif call[].flags == MEMBER_OP_PREDICT_PROBA and classifier:
        call[].out_cols = classes
        call[].out_dtype = MEMBER_F64
    else:
        member_unsupported(_name(kind), call[].flags)
    call[].classes = classes if classifier else 0


def _outputs(call: _CallPtr) raises:
    if call[].state == 0:
        if call[].param_count != META_PARAM_COUNT or not call[].params:
            raise Error("member ABI: a meta member takes six parameters")
        var words = _F64P(unsafe_from_address=call[].params)
        var kind = Int(words[0])
        if _ada_kind(kind):
            _outputs_ada(call, kind, call[].classes)
            return
        if _rfe_kind(kind):
            var entry_words = _plan_words(Int(words[1]))
            if len(entry_words) != META_RFE_WORDS:
                raise Error("member ABI: an RFE member needs its " + String(META_RFE_WORDS) + " settings words")
            _outputs_rfe(call, _plan_first(Int(words[1])), entry_words,
                         call[].classes if kind == META_RFE_CLASSIFIER else 0)
            return
        if _search_kind(kind):
            _outputs_search(call, _plan_first(Int(words[1])), call[].classes if kind == META_SEARCH_CLASSIFIER else 0)
            return
        if _multi_kind(kind):
            if call[].flags == MEMBER_OP_PREDICT_PROBA and kind == META_MULTI_CLASSIFIER:
                # (lane mf-tail5) every output's classes, float64 (`_proba_multi`)
                var total = 0
                if call[].class_counts != 0:
                    var given = MutPointer[Int, MutUntrackedOrigin](unsafe_from_address=call[].class_counts)
                    for j in range(max(call[].y.cols, 1)):  # small-loop(the outputs' class counts)
                        total += given[j]
                call[].out_cols = total
                call[].out_dtype = MEMBER_F64
                return
            if call[].flags != MEMBER_OP_PREDICT:
                member_unsupported(_name(kind), call[].flags)
            call[].out_cols = max(call[].y.cols, 1)
            if kind == META_MULTI_CLASSIFIER:
                call[].out_dtype = MEMBER_I32
            else:
                var base = _plan_final(Int(words[1]))
                var dtype = member_output_dtype(base, MEMBER_OP_PREDICT, max(call[].x.cols, 1), 1, MEMBER_F32)
                call[].out_dtype = MEMBER_F64 if dtype == MEMBER_F64 else MEMBER_F32
            call[].classes = 0
            return
        if _unit_kind(kind):
            _outputs_unit(call, kind, call[].classes)
            return
        if not _stack_kind_code(kind):
            _outputs_vote(call, kind, words[2] != 0.0, call[].classes)
            return
        var final = _plan_final(Int(words[1]))
        final.classes = call[].classes if kind == META_STACK_CLASSIFIER else 0
        var asked = call[].flags
        if asked != MEMBER_OP_PREDICT and kind != META_STACK_CLASSIFIER:
            member_unsupported(_name(kind), asked)
        # The final member's width does not depend on its input width.
        call[].out_cols = member_outputs(final, asked, max(call[].x.cols, 1), max(call[].y.cols, 1))
        call[].out_dtype = member_output_dtype(final, asked, max(call[].x.cols, 1), max(call[].y.cols, 1),
                                               MEMBER_DTYPE_UNSAID)
        if kind != META_STACK_CLASSIFIER:
            call[].classes = 0
        return
    var slot = _SlotPtr(unsafe_from_address=_lookup(call[].state, False))
    if _ada_kind(slot[].kind):
        _outputs_ada(call, slot[].kind, slot[].classes if slot[].fitted else call[].classes)
        return
    if _rfe_kind(slot[].kind):
        if slot[].fitted:
            _outputs_rfe_fitted(call, slot)
        else:
            _outputs_rfe(call, slot[].base_plans[0], slot[].words,
                         call[].classes if slot[].kind == META_RFE_CLASSIFIER else 0)
        return
    if _search_kind(slot[].kind):
        if not slot[].fitted:
            _outputs_search(call, slot[].base_plans[0], call[].classes if slot[].kind == META_SEARCH_CLASSIFIER else 0)
            return
        var best = slot[].plans[0].copy()
        var inner = member_call_of(best, slot[].states[0])
        call[].out_cols = member_outputs_of(best, inner, call[].flags)
        call[].out_dtype = inner.out_dtype
        call[].classes = slot[].classes
        return
    if _multi_kind(slot[].kind):
        if call[].flags == MEMBER_OP_PREDICT_PROBA and slot[].kind == META_MULTI_CLASSIFIER and slot[].fitted:
            # (lane mf-tail5) every output's own classes, float64 (`_proba_multi`)
            var width = _multi_proba_width(slot)
            if width == 0:
                member_unsupported(_name(slot[].kind), call[].flags)
            call[].out_cols = width
            call[].out_dtype = MEMBER_F64
            call[].classes = 0
            return
        if call[].flags != MEMBER_OP_PREDICT:
            member_unsupported(_name(slot[].kind), call[].flags)
        if not slot[].fitted:
            raise Error("NotFittedError: this " + _name(slot[].kind) + " member is not fitted yet")
        call[].out_cols = len(slot[].states)
        call[].out_dtype = _multi_dtype(slot)
        call[].classes = 0
        return
    if _unit_kind(slot[].kind):
        _outputs_unit(call, slot[].kind, slot[].classes if slot[].fitted else call[].classes)
        return
    if not _stack_kind_code(slot[].kind):
        _outputs_vote(call, slot[].kind, slot[].flag, slot[].classes if slot[].fitted else call[].classes)
        return
    if not slot[].fitted:
        raise Error("NotFittedError: this " + _name(slot[].kind) + " member is not fitted yet")
    var asked = call[].flags
    if asked != MEMBER_OP_PREDICT and slot[].kind != META_STACK_CLASSIFIER:
        member_unsupported(_name(slot[].kind), asked)
    var bases = len(slot[].stack_kinds)
    var plan = slot[].plans[bases].copy()
    var final = member_call_of(plan, slot[].states[bases])
    call[].out_cols = member_outputs_of(plan, final, asked)
    call[].out_dtype = final.out_dtype
    call[].classes = slot[].classes


# ------------------------------------------------------------------ EXPORT / IMPORT


#: LabelPropagation / LabelSpreading (core/xneighbors_member.mojo kinds 7 / 8, `LP_MARKER_WORD`).
comptime _LP_KIND_PROPAGATION = 7
comptime _LP_KIND_SPREADING = 8
comptime _LP_MARKER = 13


def _cal_unit_marker(plan: MemberPlan) -> Int:
    """The unlabeled marker a Calibrated unit's base reads (lane mf-meta5b): a LabelPropagation /
    LabelSpreading plan's marker word, else -1."""
    if (plan.family == MEMBER_FAMILY_XNEIGHBORS and (plan.kind == _LP_KIND_PROPAGATION or plan.kind == _LP_KIND_SPREADING)
            and len(plan.params) > _LP_MARKER):
        return Int(plan.params[_LP_MARKER])
    return -1


def _export_forward(slot: _SlotPtr, call: _CallPtr) raises:
    """EXPORT `META_BLOCK_FORWARD` (its note): the nested member the first word names, asked for the rest of
    the path (one more forwarded export when more than its block is left), its view handed back as this
    state's answer. A path step this state does not hold raises in the ABI's words."""
    var count = call[].param_count
    if count < 2 or call[].params == 0:
        raise Error("member ABI: a forwarded export names a nested member and a block")
    var words = _F64P(unsafe_from_address=call[].params)
    var nested = len(slot[].states)
    var at = Int(words[0])
    if at < 0:
        at = nested - 1
    if at < 0 or at >= nested or slot[].states[at] == 0:
        raise Error("member ABI: a meta member has no nested member " + String(Int(words[0])) + " of "
                    + String(nested))
    var inner = MemberCall(slot[].plans[at].route, slot[].states[at])
    inner.flags = 0
    if count == 2:
        inner.block = Int(words[1])
    else:
        inner.block = META_BLOCK_FORWARD
        inner.params = call[].params + 8
        inner.param_count = count - 1
    _ = member_call(slot[].plans[at].member, MEMBER_OP_EXPORT, inner)
    call[].dst = inner.dst
    call[].out_cols = inner.out_cols


def _export(slot: _SlotPtr, call: _CallPtr) raises:
    if call[].flags == MEMBER_EXPORT_OWN:
        for u in range(len(slot[].states)):  # driver-loop: over the nested members (each owns its rows)
            if slot[].states[u] != 0:
                member_own(slot[].plans[u], slot[].states[u])
        for u in range(len(slot[].fold_states)):  # driver-loop: over the cv tools' kept unit states (each owns its rows)
            if slot[].fold_states[u] != 0:
                member_own(slot[].base_plans[slot[].fold_owner[u]], slot[].fold_states[u])
        return
    _fitted(slot)
    var block = call[].block
    if block == META_BLOCK_FORWARD and call[].flags != MEMBER_BLOCK_BY_INDEX:
        _export_forward(slot, call)
        return
    if call[].flags == MEMBER_BLOCK_BY_INDEX:
        # vote / stack: nested, widths; a unit kind: nested, then its own words (calibrated pairs, bag draws),
        # then a bag's out-of-bag outputs.
        var blocks = 2  # an RFE state: nested (the final estimator), then its fitted words
        if slot[].kind == META_OVR or _multi_kind(slot[].kind):
            blocks = 1
        elif _bag_kind(slot[].kind):
            blocks = 3
        if block < 0 or block >= blocks:
            raise Error("member ABI: a meta member has " + String(blocks) + " blocks")
        if block == 0:
            block = META_BLOCK_NESTED
        elif _search_kind(slot[].kind):
            block = META_BLOCK_SEARCH
        elif _rfe_kind(slot[].kind):
            block = META_BLOCK_RFE
        elif not _unit_kind(slot[].kind):
            block = META_BLOCK_WIDTHS
        elif _ada_kind(slot[].kind):
            block = META_BLOCK_ADA
        elif block == 1:
            block = META_BLOCK_CAL if slot[].kind == META_CALIBRATED else META_BLOCK_DRAWS
        else:
            block = META_BLOCK_OOB
        call[].block = block
    call[].out_cols = 2
    if block == META_BLOCK_NESTED:
        call[].dst = MemberMatrix(Int(slot[].state_words.unsafe_ptr()), 1, len(slot[].state_words), MEMBER_I64)
    elif block == META_BLOCK_WIDTHS and not _unit_kind(slot[].kind):
        call[].dst = MemberMatrix(Int(slot[].width_words.unsafe_ptr()), 1 if len(slot[].width_words) > 0 else 0,
                                  len(slot[].width_words), MEMBER_I64)
    elif block == META_BLOCK_CAL and slot[].kind == META_CALIBRATED:
        call[].dst = MemberMatrix(Int(slot[].cal_words.unsafe_ptr()), 1, len(slot[].cal_words), MEMBER_F64)
    elif block == META_BLOCK_DRAWS and _bag_kind(slot[].kind):
        call[].dst = MemberMatrix(Int(slot[].draw_words.unsafe_ptr()), 1, len(slot[].draw_words), MEMBER_I32)
    elif block == META_BLOCK_OOB and _bag_kind(slot[].kind):
        call[].dst = MemberMatrix(Int(slot[].oob_words.unsafe_ptr()), 1 if len(slot[].oob_words) > 0 else 0,
                                  len(slot[].oob_words), MEMBER_F64)
    elif block == META_BLOCK_ADA and _ada_kind(slot[].kind):
        call[].dst = MemberMatrix(Int(slot[].ada_words.unsafe_ptr()), 1, len(slot[].ada_words), MEMBER_F64)
    elif block == META_BLOCK_SEARCH and _search_kind(slot[].kind):
        call[].dst = MemberMatrix(Int(slot[].search_words.unsafe_ptr()), 1, len(slot[].search_words), MEMBER_F64)
    elif block == META_BLOCK_FOLDS and _search_kind(slot[].kind):
        call[].dst = MemberMatrix(Int(slot[].fold_words.unsafe_ptr()), 1 if len(slot[].fold_words) > 0 else 0,
                                  len(slot[].fold_words), MEMBER_I64)
    elif block == META_BLOCK_RFE and _rfe_kind(slot[].kind):
        call[].dst = MemberMatrix(Int(slot[].rfe_words.unsafe_ptr()), 1, len(slot[].rfe_words), MEMBER_F64)
    elif block == MEMBER_BLOCK_SCALARS:
        call[].dst = MemberMatrix(Int(slot[].scalar_words.unsafe_ptr()), 1, len(slot[].scalar_words), MEMBER_F64)
    else:
        raise Error("member ABI: a meta member has no fitted-state block " + String(block))


def _import_cal(slot: _SlotPtr, words: _F64P, count: Int) raises:
    """IMPORT `META_BLOCK_CAL` (`_unit_views`' layout): each pair's held
    classes and calibrators."""
    var bad = String("member ABI: a CalibratedClassifierCV member's pair words are malformed")
    if count < 1:
        raise Error(bad)
    var pairs = Int(words[0])
    var at = 1
    slot[].unit_held = List[List[Int32]](capacity=pairs)
    slot[].cal_iso = List[Bool]()
    slot[].cal_a = List[Float64]()
    slot[].cal_b = List[Float64]()
    slot[].cal_m = List[Int]()
    slot[].cal_kx = List[List[Float64]]()
    slot[].cal_ky = List[List[Float64]]()
    slot[].pair_first = List[Int](capacity=pairs)
    slot[].pair_count = List[Int](capacity=pairs)
    for _ in range(pairs):  # driver-loop: over the pairs
        if at >= count:
            raise Error(bad)
        var h = Int(words[at])
        at += 1
        if at + h + 1 > count:
            raise Error(bad)
        var held = List[Int32](capacity=h)
        for c in range(h):  # small-loop(a pair's held classes)
            held.append(Int32(Int(words[at + c])))
        at += h
        slot[].unit_held.append(held^)
        var cals = Int(words[at])
        at += 1
        slot[].pair_first.append(len(slot[].cal_iso))
        slot[].pair_count.append(cals)
        for _ in range(cals):  # driver-loop: over the pair's calibrators
            if at + 4 > count:
                raise Error(bad)
            slot[].cal_iso.append(words[at] != 0.0)
            slot[].cal_a.append(words[at + 1])
            slot[].cal_b.append(words[at + 2])
            var m = Int(words[at + 3])
            at += 4
            if m < 0 or at + 2 * m > count:
                raise Error(bad)
            slot[].cal_m.append(m)
            var kx = List[Float64](capacity=m)
            var ky = List[Float64](capacity=m)
            for q in range(m):  # small-loop(one calibrator's knots)
                kx.append(words[at + q])
            for q in range(m):  # small-loop(one calibrator's knots)
                ky.append(words[at + m + q])
            at += 2 * m
            slot[].cal_kx.append(kx^)
            slot[].cal_ky.append(ky^)


def _import_draws(slot: _SlotPtr, words: _I32P, count: Int) raises:
    """IMPORT `META_BLOCK_DRAWS` (`_unit_views`' layout): each member's
    feature subset, held classes and out-of-bag rows."""
    var bad = String("member ABI: a Bagging member's draw words are malformed")
    if count < 1:
        raise Error(bad)
    var units = Int(words[0])
    var at = 1
    slot[].unit_cols = List[List[Int32]](capacity=units)
    slot[].unit_held = List[List[Int32]](capacity=units)
    slot[].unit_oob = List[List[Int32]](capacity=units)
    for _ in range(units):  # driver-loop: over the member estimators
        for part in range(3):  # small-loop(a member's three runs: columns, held classes, out-of-bag rows)
            if at >= count:
                raise Error(bad)
            var c = Int(words[at])
            at += 1
            if c < 0 or at + c > count:
                raise Error(bad)
            var run = List[Int32](capacity=c)
            for j in range(c):  # small-loop(one run of words)
                run.append(words[at + j])
            at += c
            if part == 0:
                slot[].unit_cols.append(run^)
            elif part == 1:
                slot[].unit_held.append(run^)
            else:
                slot[].unit_oob.append(run^)


def _import(slot: _SlotPtr, call: _CallPtr) raises:
    if call[].block == META_BLOCK_NESTED:
        var count = call[].x.rows * call[].x.cols
        var unit = _unit_kind(slot[].kind)
        var wrong = count < 1 if unit else count != len(slot[].plans)
        if _search_kind(slot[].kind) or _rfe_kind(slot[].kind):
            wrong = count != 1  # the refitted best only; an RFE's final estimator
        if wrong or call[].x.dtype != MEMBER_I64 or call[].x.data == 0:
            raise Error("member ABI: a meta member imports one int64 state handle per member")
        _drop_states(slot)
        if unit:
            _expand(slot, count)
        var words = _I64P(unsafe_from_address=call[].x.data)
        slot[].states = List[Int](length=count, fill=0)
        for u in range(count):  # driver-loop: over the nested members (their imported states change owner)
            slot[].states[u] = Int(words[u])
    elif call[].block == META_BLOCK_WIDTHS and not _unit_kind(slot[].kind):
        var count = call[].x.rows * call[].x.cols
        if count != len(slot[].stack_kinds) or (count > 0 and (call[].x.dtype != MEMBER_I64 or call[].x.data == 0)):
            raise Error("member ABI: a stacking member imports one int64 width per stacked member")
        var words = _I64P(unsafe_from_address=call[].x.data)
        slot[].widths = List[Int](length=count, fill=0)
        slot[].total = 0
        for j in range(count):  # driver-loop: over the stacked members
            slot[].widths[j] = Int(words[j])
            slot[].total += slot[].widths[j]
    elif call[].block == META_BLOCK_CAL and slot[].kind == META_CALIBRATED:
        if call[].x.dtype != MEMBER_F64 or call[].x.data == 0:
            raise Error("member ABI: a CalibratedClassifierCV member imports its pairs as float64 words")
        _import_cal(slot, _F64P(unsafe_from_address=call[].x.data), call[].x.rows * call[].x.cols)
    elif call[].block == META_BLOCK_DRAWS and _bag_kind(slot[].kind):
        if call[].x.dtype != MEMBER_I32 or call[].x.data == 0:
            raise Error("member ABI: a Bagging member imports its draws as int32 words")
        _import_draws(slot, _I32P(unsafe_from_address=call[].x.data), call[].x.rows * call[].x.cols)
    elif call[].block == META_BLOCK_SEARCH and _search_kind(slot[].kind):
        if call[].x.dtype != MEMBER_F64 or call[].x.data == 0:
            raise Error("member ABI: a search member imports its words as float64 words")
        _import_search(slot, _F64P(unsafe_from_address=call[].x.data), call[].x.rows * call[].x.cols)
    elif call[].block == META_BLOCK_RFE and _rfe_kind(slot[].kind):
        if call[].x.dtype != MEMBER_F64 or call[].x.data == 0:
            raise Error("member ABI: an RFE member imports its words as float64 words")
        _import_rfe(slot, _F64P(unsafe_from_address=call[].x.data), call[].x.rows * call[].x.cols)
    elif call[].block == META_BLOCK_ADA and _ada_kind(slot[].kind):
        if call[].x.dtype != MEMBER_F64 or call[].x.data == 0:
            raise Error("member ABI: an AdaBoost member imports its weights and errors as float64 words")
        _import_ada(slot, _F64P(unsafe_from_address=call[].x.data), call[].x.rows * call[].x.cols)
    elif call[].block == META_BLOCK_BATCHED and _bag_kind(slot[].kind):
        # THE BATCHED ROUTE: the one imported state is the batched forest's
        # (after META_BLOCK_NESTED; no draws follow).
        if len(slot[].states) != 1 or len(slot[].base_plans) != 2:
            raise Error("member ABI: a batched Bagging member imports one forest state, admitted batched")
        slot[].plans = [slot[].base_plans[1].copy()]
        slot[].masks = [slot[].base_masks[1]]
        slot[].batched = True
        slot[].unit_held = [List[Int32]()]
        slot[].unit_cols = [List[Int32]()]
        slot[].unit_oob = [List[Int32]()]
    elif call[].block != 0 or call[].flags != MEMBER_IMPORT_DONE:
        raise Error("member ABI: a meta member has no fitted-state block " + String(call[].block))
    if call[].flags == MEMBER_IMPORT_DONE:
        var kind = slot[].kind
        if len(slot[].states) != len(slot[].plans) or len(slot[].states) == 0:
            raise Error("member ABI: a meta member import needs its nested states")
        if _stack_kind_code(kind) and len(slot[].widths) != len(slot[].stack_kinds):
            raise Error("member ABI: a stacking member import needs its widths")
        if kind == META_CALIBRATED and len(slot[].pair_first) != len(slot[].states):
            raise Error("member ABI: a CalibratedClassifierCV member import needs one pair per member")
        if _bag_kind(kind) and len(slot[].unit_cols) != len(slot[].states):
            raise Error("member ABI: a Bagging member import needs one draw per member")
        if kind == META_OVR and len(slot[].states) != (1 if call[].classes == 2 else call[].classes):
            raise Error("member ABI: a OneVsRestClassifier member import needs one member per class (one for two)")
        # AdaBoost's predicts read one weight per member (the door's `IndexError` otherwise).
        if _ada_kind(kind) and len(slot[].ada_alphas) < len(slot[].states):
            raise Error("member ABI: an AdaBoost member import needs one weight per round member")
        if _search_kind(kind) and len(slot[].search_words) == 0:
            raise Error("member ABI: a search member import needs its search words")
        if _rfe_kind(kind) and len(slot[].rfe_columns) == 0:
            raise Error("member ABI: an RFE member import needs its fitted words")
        if _rfe_kind(kind):
            # The final estimator's plan: the rounds' plan with the importance word cleared, as its fit made it.
            slot[].plans = [rfe_final_plan(slot[].base_plans[0], slot[].words[META_RFE_IMPORTANCE_WORD] != 0.0)]
        var counts = List[Int]()
        if call[].class_counts != 0:
            var given = MutPointer[Int, MutUntrackedOrigin](unsafe_from_address=call[].class_counts)
            for j in range(len(slot[].states) if _multi_kind(kind) else 0):  # small-loop(the outputs' class counts)
                counts.append(given[j])
        if kind == META_MULTI_CLASSIFIER and len(counts) != len(slot[].states):
            raise Error("member ABI: a MultiOutputClassifier member import needs each output's class count")
        slot[].out_classes = counts.copy()
        slot[].classes = call[].classes if _classifier_kind(kind) and not _multi_kind(kind) else 0
        for u in range(len(slot[].plans)):  # driver-loop: over the nested members (their class count)
            var classes = slot[].classes
            if kind == META_OVR:
                classes = 2
            elif (kind == META_CALIBRATED or kind == META_BAG_CLASSIFIER) and u < len(slot[].unit_held):
                # (a batched forest's one held list is empty: every class)
                classes = _held_classes(slot[].unit_held[u], slot[].classes)
            elif kind == META_MULTI_CLASSIFIER:
                classes = counts[u]
            slot[].plans[u].classes = classes
        _nested_outputs(slot)
        _unit_views(slot)
        slot[].fitted = True


# ------------------------------------------------------------------ PACK reads


def meta_state_plans(state: Int) raises -> List[MemberPlan]:
    """PACK (the boundary): the nested plans of a META state, in member
    order (a stack's final last), copied. The state owns their states
    (`META_BLOCK_NESTED`)."""
    var slot = _SlotPtr(unsafe_from_address=_lookup(state, False))
    return slot[].plans.copy()


def meta_state_notices(state: Int, mut owners: List[Int], mut words: List[List[Float64]]) raises:
    """PACK: the nested fit notices of a META state, in fit order (each with
    the nested member it belongs to), handed over once."""
    var slot = _SlotPtr(unsafe_from_address=_lookup(state, False))
    owners = slot[].notice_owner.copy()
    words = slot[].notice_words.copy()
    slot[].notice_owner = List[Int]()
    slot[].notice_words = List[List[Float64]]()


# ------------------------------------------------------------------ the entry


def _ensemble_member(operation: Int, address: Int) raises -> Int:
    var call = _CallPtr(unsafe_from_address=address)
    if operation == MEMBER_OP_CAPABILITIES:
        if call[].state == 0:
            return _UNION
        var kind = _SlotPtr(unsafe_from_address=_lookup(call[].state, False))
        if _refused_first(kind):
            return _refused_mask(kind[].kind)
        if _rfe_kind(kind[].kind):
            return _rfe_mask(kind) | _reads_given(kind, kind[].base_masks)
        if _unit_kind(kind[].kind) or _search_kind(kind[].kind):
            return _kind_mask(kind[].kind, kind[].flag, kind[].base_masks) | _reads_given(kind, kind[].base_masks)
        return _kind_mask(kind[].kind, kind[].flag, kind[].masks) | _reads_given(kind, kind[].masks)
    if operation == MEMBER_OP_NEW:
        _new(call)
        return 0
    if operation == MEMBER_OP_FREE:
        var held = _SlotPtr(unsafe_from_address=_lookup(call[].state, False))
        _drop_states(held)
        _ = _lookup(call[].state, True)
        call[].state = 0
        return 0
    if operation == MEMBER_OP_OUTPUTS:
        _outputs(call)
        return 0
    var slot = _SlotPtr(unsafe_from_address=_lookup(call[].state, False))
    if operation == MEMBER_OP_FIT:
        _fit(slot, call)
    elif operation == MEMBER_OP_PREDICT or operation == MEMBER_OP_PREDICT_PROBA or operation == MEMBER_OP_DECISION:
        _predict(slot, call, operation)
    elif operation == MEMBER_OP_EXPORT:
        _export(slot, call)
    elif operation == MEMBER_OP_IMPORT:
        _import(slot, call)
    elif operation == MEMBER_OP_SCORE and _multi_kind(slot[].kind):
        _score_multi(slot, call)
    elif operation == MEMBER_OP_SCORE and _search_kind(slot[].kind):
        _fitted(slot)
        _forward_best(slot, call, operation)
    elif (operation == MEMBER_OP_SCORE or operation == MEMBER_OP_TRANSFORM) and _rfe_kind(slot[].kind):
        _call_rfe(slot, call, operation)
    else:
        member_unsupported(_name(slot[].kind), operation)
    return 0


def ensemble_member_v2(operation: Int, aa: Int, na: Int, pp: Int, np: Int, ff: Int, nf: Int,
    result: Int, context: Int, sink: NativeDescriptorError) abi("C") -> Int:
    """The member entry of the META kind (schema `MEMBER_ABI`)."""
    try:
        var word = _ensemble_member(operation, member_call_from_spans(operation, aa, na, np, nf))
        if result:
            MutPointer[Int, MutUntrackedOrigin](unsafe_from_address=result)[] = word
        return 0
    except error:
        native_descriptor_report(error, context, sink)
        return 1


def ensemble_member_publish(mode: Int, vendor: Int, publisher: Int) raises:
    """Publish this image's META member entry: one line in each x_trees
    image's module initialization, after its glue table (`vendor` -1: a build
    with no GPU vendor publishes nothing)."""
    if vendor < 0:
        return
    member_publish[MEMBER_FAMILY_ENSEMBLE](mode, vendor, publisher, ensemble_member_v2)


def ensemble_member_publish_image(vendor_word: Int) raises:
    """The one publication line of an x_trees image's module initialization,
    after its glue table: the image's tier and the vendor word of its glue
    table key (`x_trees_glue_vendor`; -1 publishes nothing)."""
    if vendor_word < 0:
        return
    var store = _STORE.get_or_create_ptr()
    ensemble_member_publish(Int(GLOBAL_NUMERIC_MODE), vendor_word, Int(store))
