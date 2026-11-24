# SPDX-License-Identifier: Apache-2.0
"""The XNeighbors family as MEMBERS (core/member_abi.mojo, MEMBER-ABI-V2.md).

One entry, `xneighbors_member_v2`, published by each x_neighbors image
(bindings/_mojolearn_x_neighbors.mojo: the device column, its
`COMPILED_VENDOR`; bindings/_mojolearn_x_neighbors_host.mojo: the CPU column,
vendor 0). It runs the family's existing typed RUN through the ONE route slot
it reads: `entry` = the x_neighbors native operations entry of the image the
route resolved to (`x_neighbors_native_api_v1`, the `XnDriver` entry every
x_neighbors door uses). No kernel is touched.

Kinds (MEMBER_OP_NEW word 0): 0 NearestCentroid, 1 LocalOutlierFactor,
2 OneClassSVM, 3 KernelPCA, 4 PolynomialCountSketch, 5 AdditiveChi2Sampler,
6 SkewedChi2Sampler, 7 LabelPropagation, 8 LabelSpreading, 9 KNNImputer,
10 SVGP. OUTPUTS with no state answers from the kind word and `call.classes`
(NearestCentroid: proba `classes`, decision 1 for two classes else
`classes`, predict 1; LabelPropagation / LabelSpreading: proba `classes`,
predict 1; the outlier detectors: 1 column).

NearestCentroid (kind 0). RUN: neighbors/nearest_centroid_run.mojo
(`nc_fit_count_run`, `nc_fit_priors_run`, `nc_fit_run`, `nc_labels_run`,
`nc_decision_run`, `nc_proba_run`), in the order the estimator's own door
(bindings/neighbors_local_native_door.mojo `_nc_fit`) runs them; empirical
priors are the host label ratio (core/array_helpers_impl.mojo
`class_ratio_f64` mode 1, count / rows: the host label kernel's own step).
MEMBER_OP_NEW parameters (Float64 words, count 6, or 6 + classes with given
priors): [0] kind 0, [1] metric (0 euclidean, 1 manhattan), [2] priors (0
empirical, 1 uniform, 2 given), [3] shrink_threshold (0 none), [4] the image's
fused capability (0 / 1), [5] the image's wide-probability capability (0 /
1), [6 ..] the given priors, one per class (checked against `call.classes`
at fit, normalized by the fit).
Views (all host, dense row-major; a device-home view, a row or column index,
a sample weight and a nonzero `fit_output` are refused in words): X float32
rows x features; y = class CODES 0 .. classes-1 as one int32 column
(`MEMBER_I32`), `call.classes` the class count (at least two, and every class
present in the fit rows: the estimator fits the classes it sees, so a fold
missing a class is refused in words). predict writes int32 codes (rows x 1);
decision (euclidean only) writes float32 rows x 1 for two classes (the second
class's score minus the first's), else rows x classes; predict_proba
(euclidean only) writes rows x classes as float32 or float64 (the dtype of
`dst`; the image computes one width and the other is cast from a slot-owned
scratch). Score is not published: the caller predicts and runs its own metric.
Fitted-state blocks (EXPORT / IMPORT), in index order: 64 centroids float32
classes x features, 65 within_class_std_dev float32 1 x features, 66
deviations float32 classes x features, 67 class_prior float64 1 x classes.
An import needs all four; `MEMBER_IMPORT_DONE` reads `call.classes`.

LocalOutlierFactor (kind 1). RUN: neighbors/lof_run.mojo (`lof_fit_run`,
`lof_fit_labels_run`, `lof_query_run`), as the estimator's door
(bindings/neighbors_local_native_door.mojo `_lof_fit`, `_lof_query`) runs
them. MEMBER_OP_NEW parameters (count 7): [0] kind 1, [1] n_neighbors, [2]
novelty (0 / 1), [3] contamination 'auto' (0 / 1), [4] contamination (when
not auto), [5] the unfused-kNN switch, [6] the old-items switch (the two
context flags the door reads). Views: X float32 rows x features (host,
dense; no y, no weights, no index, no device view). FIT with `fit_output =
MEMBER_OP_PREDICT` (novelty 0 only: fit_predict) also writes the fit rows'
labels to `dst` as int64 (-1 outlier, 1 inlier), rows x 1. With novelty 1:
PREDICT (int64 labels), DECISION and SCORE_SAMPLES (float32, rows x 1).
Blocks: 64 the fitted rows float32 rows x features, 65 their k-distances
float32 rows x k, 66 local reachability densities float32 1 x rows, 67
negative_outlier_factor float32 1 x rows, 68 the offset (float64 1 x 1).

OneClassSVM (kind 2). RUN: svm/svc_ocsvm_native_run.mojo (`oc_fit_run`,
`oc_query_run`, `oc_decision_run`, `oc_labels_run`) and
core/ocsvm_plan.mojo `oc_take_rows_into`, as the estimator's door
(bindings/svc_ocsvm_native_door.mojo) runs them. Route slots: `entry` = the
x_neighbors native operations entry; `second` = the image's resident
OneClassSVM solve (`ocsvm_resident_native_api_v1`, read only when word 9 is
the resident solve). MEMBER_OP_NEW parameters (count 14): [0] kind 2, [1]
kernel code (the family's `family_svc_xn_kernels` code), [2] nu, [3] degree,
[4] coef0, [5] gamma (0 the value in word 6, 1 'scale', 2 'auto' = 1 /
features), [6] gamma value, [7] the solve's iteration cap (max_iter, or
10000000 for -1), [8] tol, [9] the solve (svm/svc_ocsvm_native_run.mojo
OC_SOLVE_*), [10] the resident solve's fast tier (0 / 1), [11] the image's
fused labels (0 / 1), [12] the old-items switch (`family_svc_xn_old_items`),
[13] a precomputed kernel (lane mf-nbr6: X is the square Gram matrix, the
Gram solve, no support vectors kept (supports x 0), queries on the support
columns, OC_QUERY_GRAM_PRECOMPUTED). Views: X float32
rows x features (host, dense; no y, no weights: MEMBER_CAN_SAMPLE_WEIGHT is
not set, no index, no device view). FIT with `fit_output = MEMBER_OP_PREDICT`
also writes fit_predict's int64 labels. PREDICT int64 (-1 / 1), DECISION and
SCORE_SAMPLES float32, rows x 1. Blocks: 64 support vectors float32
supports x features, 65 dual coefficients float32 1 x supports, 66 support
indices int32 1 x supports, 67 scalars float64 1 x 4 (gamma used, offset
(rho), n_iter, features). A fit reads X only during the call (the support
vectors it keeps are gathered into the slot).

KernelPCA (kind 3). RUN: core/kpca_native_plan.mojo (`KPCADenseFit`,
`KPCANeighborDriver.alpha` / `.fused_query`, `KPCACenteredQuery`) and the
eigensolver `KPCANativeDriver.eigen`, as bindings/kpca_native_door.mojo
runs them. Route slots: `entry` = the x_neighbors native operations entry of
the estimator's decomposition route (its neighbors connection), `second` =
the eigensolver image's KernelPCA entry, `third` = that image's reduction
entry. MEMBER_OP_NEW parameters (count 11): [0] kind 3, [1] kernel code (the
`kpca_kernels` table), [2] n_components (-1 None), [3] eigen_solver 'auto'
(0 / 1), [4] the image's resident route (0 / 1), [5] gamma given (0: 1 /
features), [6] gamma, [7] degree, [8] coef0, [9] remove_zero_eig, [10] the
old-items switch (`kpca_old_items`). A precomputed kernel is the kernel word
KPCA_PRECOMPUTED (lane mf-nbr6): a square X at fit, queries of one column per
fitted row, in `kpca_fit` / `kpca_transform`'s words.
Views: X float32 rows x features (host, dense; no y, weights, index or
device view). FIT with `fit_output = MEMBER_OP_TRANSFORM` writes
fit_transform's rows x kept float32. TRANSFORM: rows x kept float32 (the
scaled alphas are computed once per fitted state). OUTPUTS needs a fitted
state (the kept count is the fit's). Blocks: 64 the fitted rows float32 rows
x features, 65 eigenvalues float32 1 x kept, 66 eigenvectors float32 rows x
kept, 67 the fitted column means float32 1 x rows, 68 their total float32
1 x 1, 69 gamma used (float64 1 x 1). A fit BORROWS X as its fitted rows
(LIFETIME below, as LocalOutlierFactor).

PolynomialCountSketch (kind 4). RUN: core/pcs_native_api.mojo
`PCSNativeDriver.draw` / `.query` over the x_neighbors entry (`entry`), as
bindings/neighbors_native_unpack.mojo `_pcs_fit` and
bindings/neighbors_native_pack.mojo `_pcs_transform` run them. MEMBER_OP_NEW
parameters (count 7): [0] kind 4, [1] degree, [2] n_components, [3] gamma,
[4] coef0, [5] the seed word (`pcs_seed_unpack` of the door's `pcs_seed`:
an exact int, or ONE native draw from a RandomState / None made at the
admission), [6] the old-items switch (`MOJOLEARN_XN_OLD_ITEMS`). Views: fit
reads only the width of X; transform X float32 rows x features -> float32
rows x n_components (host, dense). OUTPUTS answers n_components with no
state. Blocks: 64 indexHash int32 degree x augmented, 65 bitHash int32 degree
x augmented, 66 n_features_in (float64 1 x 1).

AdditiveChi2Sampler (kind 5). RUN: neighbors/chi2_run.mojo `achi2_fit_run`
(the negative-entry refusal), `achi2_interval_check`, `achi2_transform_run`
over the x_neighbors entry, as bindings/neighbors_semi_native_door.mojo runs
them. MEMBER_OP_NEW parameters (count 5): [0] kind 5, [1] sample_steps, [2]
the interval (given, or the steps' default), [3] interval valid (0: refused
at fit, as the estimator refuses), [4] the image's device scan (kfeat bit 1).
Views: X float32 rows x features (host, dense; sparse input is not a
member) -> float32 rows x features * (2 steps - 1). OUTPUTS answers from
`x.cols` with no state. Block: 64 n_features_in (float64 1 x 1). A fit reads
X only during the call.

SkewedChi2Sampler (kind 6). RUN: neighbors/chi2_run.mojo `schi2_fit_run`
(the counter-based draw of the weights and offsets), `schi2_check_run` and
`schi2_transform_run` (log(x + skewedness) into a slot scratch, then the
map) over the x_neighbors entry, as bindings/neighbors_semi_native_door.mojo
`_schi2_fit` / `_schi2_transform` run them with eager weights (the
`identical` tier), or the image's one-call transform over the stored
weights (kfeat bit 4, SCHI2_LAZYW: `schi2_lazy_transform`, word [4]).
MEMBER_OP_NEW parameters (count 5): [0] kind 6, [1] n_components, [2]
skewedness, [3] the seed word (the low 32 bits of `_semi_seed`'s int: an
exact int, or one native draw from a RandomState / None at the admission),
[4] the one-call transform (0 / 1).
Views: fit reads only the width of X; transform X float32 rows x features
(host, dense) -> float32 rows x n_components. OUTPUTS answers n_components
with no state. Blocks: 64 random_weights float32 features x n_components,
65 random_offset float32 1 x n_components (an import whose width differs from
n_components is refused).

KNNImputer (kind 9). RUN: neighbors/knn_imputer_run.mojo `imp_fit_run`
(the fit's column masks; the missing number's cells masked into the slot's
rows) and `imp_transform_run`, over the x_neighbors entry, as
bindings/neighbors_local_native_door.mojo `_imp_fit` / `_imp_transform`
run them; the kept-column plan (`imp_keep_plan`) is built once per fitted
state (the door builds it per transform). MEMBER_OP_NEW parameters (count
10): [0] kind 9, [1] n_neighbors, [2] weights 'distance' (0 / 1), [3]
missing_values is a number (0: NaN), [4] that number as a float32 word, [5]
keep_empty_features, [6] add_indicator, [7] the uncompact-impute switch, [8]
the old-items switch (the door's `family_xn_loc_flags`), [9] the image's lean
fit entry (`x_neighbors_nc_fit_lean`). An n_neighbors below 1 fits and is
refused at transform, as the door. Views: X float32 rows x features
(host, dense) -> float32 rows x (kept columns + one indicator column per
feature missing at fit). OUTPUTS with no state answers only when the width is
not chosen by the fit (keep_empty_features and no indicator: `x.cols`).
Blocks: 64 the fitted rows float32 rows x features, 65 the columns valid at
fit int32 1 x features (0 / 1), 66 the columns missing at fit int32 1 x count
(absent when none). A fit BORROWS X as its fitted rows when no number is
masked (as the estimator keeps the user's array), else the slot owns the
masked copy (the door allocates the same new array).

SVGP (kind 10, a regressor). RUN: neighbors/svgp_run.mojo `svgp_fit_run`
(the inducing rows: n_inducing evenly spaced rows of X; then the variational
fit) and `svgp_query_run` (the mean), over the x_neighbors entry, as
bindings/neighbors_graph_native_door.mojo `_svgp_fit` / `_svgp_query` run
them. MEMBER_OP_NEW parameters (count 10): [0] kind 10, [1] n_inducing, [2]
gamma (`_svgp_gamma`: 1 / (2 lengthscale^2)), [3] kernel_variance, [4]
noise_variance, [5] jitter, [6] the old-items switch
(`family_svc_xn_old_items`), [7] the address of caller-given inducing points
(float32 C order, admitted and pinned with the plan at UNPACK; 0: the rows
are picked from X), [8] their rows, [9] their columns (lane mf-nbr6). Views: X float32 rows x features, y float32
rows x 1 (host, dense; no weights, index or device view). PREDICT writes the
mean, float32 rows x 1. Score is not published (the caller predicts and runs
its own metric). Blocks: 64 Z float32 m x features, 65 alpha float32 1 x m,
66 C float32 m x m, 67 q_mu float32 1 x m, 68 q_sqrt float32 m x m, 69
scalars float64 1 x 2 (elbo, features). A fit reads X and y only during the
call.

LabelPropagation / LabelSpreading (kinds 7, 8, classifiers; lane
mf-entries5). RUN: neighbors/label_propagation_run.mojo `lp_fit_run` and
`lp_query_run`, the bodies the estimator's door
(bindings/neighbors_semi_native_door.mojo `_lp_fit`, `_lp_query`) runs.
The member fits the caller's class codes. THE UNLABELED MARKER (lane
mf-meta5, word [13]): the caller's code of the label -1, or -1 for none.
With a marker its rows take part in the graph and not in classes_, as the
door reads them (`lp_fit_run`'s marker word: the code itself; the fitted
classes are the caller's others, `call.classes - 1` of them); predict writes
the caller's codes (a fitted class c at or above the marker is the caller's
c + 1), predict_proba the fitted classes' columns. Without one the marker
word is the class count (every row labeled). MEMBER_OP_NEW parameters (count
14): [0] kind 7 / 8, [1] knn (0 rbf / 1 knn), [2] n_neighbors, [3] the
'rbf' kernel code (`family_svc_xn_kernels`), [4] gamma, [5] the image's
raw-graph switch (`x_neighbors_classical_graph_normalization`), [6] the
label step's clamp weight (1 for propagation, `xn_lp_static_weight(alpha)`
for spreading), [7] alpha as the float32 word (0 for propagation), [8] tol,
[9] max_iter, [10] the unfused-kNN arm, [11] the old-items arm, [12] the
image's dense-proba operation (`lp_dense_available`), [13] the marker code. Views: X float32 rows
x features, y int32 codes rows x 1 (host, dense; no weights, index or
device view; every class present). PREDICT int32 codes rows x 1;
PREDICT_PROBA float32 (or float64, cast) rows x classes. Blocks: 64 the
fitted rows float32 rows x features, 65 label_distributions float32 rows x
classes, 66 the transduction's codes int32 1 x rows (absent after an import
without it), 67 scalars float64 1 x 2 (n_iter, features). A fit BORROWS X
as its fitted rows (LIFETIME below, as LocalOutlierFactor).

LIFETIME: a NearestCentroid fit reads X and y only during the call (its
fitted state is the centroids it computes); nothing of X is kept or copied.
A LocalOutlierFactor fit BORROWS X as its fitted rows, as the estimator's
fit keeps the user's array (`_fit_X`): the caller keeps X alive and
unchanged until the state's next fit, import or free; the first export of
block 64 copies it into the slot once, and the state reads that copy from
then on.

The fitted state is OWNED here, in a process-wide store of fixed chunks (the
core/linear_member.mojo form): a slot never moves once minted; the store's
lock covers only mint, free and the handle lookup. Per-state scratch (the
predicted codes, a probability cast) is grown once and reused by every later
query of the state.
"""
from std.sys.info import size_of
from std.memory import unsafe_memcpy
from core.array_helpers_impl import class_ratio_f64
from core.host_parallel import host_parallelize
from core.host_predict_threads import host_predict_task_count_work, host_predict_chunk
from core.member_abi import (
    MemberCall, MemberMatrix, MEMBER_F32, MEMBER_F64, MEMBER_I32,
    MEMBER_OP_CAPABILITIES, MEMBER_OP_NEW, MEMBER_OP_FREE, MEMBER_OP_FIT, MEMBER_OP_PREDICT,
    MEMBER_OP_PREDICT_PROBA, MEMBER_OP_DECISION, MEMBER_OP_OUTPUTS, MEMBER_OP_EXPORT, MEMBER_OP_IMPORT,
    MEMBER_OP_SCORE_SAMPLES, MEMBER_I64, MEMBER_CAN_SCORE_SAMPLES, MEMBER_OP_TRANSFORM, MEMBER_CAN_TRANSFORM,
    MEMBER_CAN_FIT, MEMBER_CAN_PREDICT, MEMBER_CAN_PREDICT_PROBA, MEMBER_CAN_DECISION,
    MEMBER_CAN_EXPORT, MEMBER_CAN_IMPORT, MEMBER_CAN_CLASSIFIER, MEMBER_CAN_TEST_ROWS,
    MEMBER_BLOCK_FAMILY, MEMBER_BLOCK_BY_INDEX, MEMBER_IMPORT_DONE,
    member_publish, member_call_from_spans, member_unsupported,
)
from core.member_abi import MEMBER_BORROWS_FIT, MEMBER_EXPORT_OWN
from core.member_abi import MEMBER_CAN_ROW_INDEX, MEMBER_CAN_SCOPE
from core.device_route import DEVICE_GPU
# lane vdf-scope-cluster3: THE FOLD X WORD (core/fold_x_device.mojo), the member side (lane vdf-scope-linear).
from core.member_fold_x import (
    MemberFoldX, member_fold_x_open, member_fold_x_close, member_fold_take_view, member_fold_store_ready,
)
from core.member_families import MEMBER_FAMILY_XNEIGHBORS
from core.native_descriptor_abi import NativeDescriptorError
from core.native_descriptor_call import native_descriptor_report
from core.admitted_refusals import admitted_finite_refuse
from core.nearest_centroid_plan import NcLabelCodes, NcFitPlan
from neighbors.lof_run import (
    LofFitRequest, LofFitOutcome, LofQueryRequest, lof_fit_run, lof_fit_labels_run, lof_query_run,
)
from core.shared_slot import SharedSlot, shared_slot_lock, shared_slot_unlock
from core.ocsvm_resident_api import OcResidentDriver
from core.ocsvm_plan import oc_take_rows_into
from core.kpca_native_api import KPCAEigenRequest, KPCANativeDriver
from core.pcs_native_api import PCSFitPlan, PCSNativeDriver, pcs_options, pcs_feature_width
from neighbors.chi2_run import (
    Achi2FitRequest, Achi2Request, achi2_fit_run, achi2_interval_check, achi2_transform_run,
    Schi2FitRequest, Schi2Request, schi2_fit_run, schi2_check_run, schi2_transform_run,
)
from neighbors.neighbors_semi_run import xn_achi2_width, xn_schi2_missing_fit, xn_semi_knn_k
from neighbors.label_propagation_run import LpFitRequest, LpQueryRequest, lp_fit_run, lp_query_run
from core.kpca_native_plan import (
    KPCA_PRECOMPUTED, KPCAKernelOptions, KPCAFitRoute, KPCAEigenRoom, KPCADenseFit, KPCAFitPrepareRequest, KPCAAlphaRequest,
    KPCAQueryRequest, KPCACenteredQuery, KPCANeighborDriver,
)
from svm.svc_ocsvm_native_run import (
    OC_SOLVE_RESIDENT, OC_SOLVE_GRAM, OC_QUERY_FUSED_LABELS, OC_QUERY_KERNEL_PRODUCT, OC_QUERY_GRAM_KERNEL,
    OC_QUERY_GRAM_PRECOMPUTED,
    OcFitRequest, OcQueryRequest, oc_fit_run, oc_query_run, oc_decision_run, oc_labels_run,
)
from core.neighbors_native_driver import XnDriver
from neighbors.nearest_centroid_run import (
    NC_PRIORS_EMPIRICAL, NC_PRIORS_UNIFORM, NC_PRIORS_GIVEN,
    NcFitRequest, NcFitOutcome, NcQueryRequest,
    nc_fit_count_run, nc_fit_priors_run, nc_fit_run, nc_labels_run, nc_decision_run, nc_proba_wide, nc_proba_run,
)
from neighbors.neighbors_local_run import (
    LOF_RESULT_SCORES, LOF_RESULT_DECISION, LOF_RESULT_LABELS, lof_check_fit_rows, lof_fit_neighbors,
    lof_check_novelty, lof_refuse_fit_predict_novelty, local_check_n_neighbors,
    nc_check_classes, nc_check_priors_size, nc_check_variance, nc_refuse_decision_metric,
    ImpKeepPlan, imp_keep_plan, imp_check_features,
)
from neighbors.svgp_run import SVGP_VARIANCE_NONE, SvgpFitRequest, SvgpQueryRequest, svgp_fit_run, svgp_query_run
from neighbors.neighbors_graph_run import xn_rows_match_check, xn_svgp_definite_check
from neighbors.knn_imputer_run import ImpFitRequest, ImpTransformRequest, imp_fit_run, imp_transform_run, imp_result_columns

comptime _CHUNK = 256
comptime _INDEX_SPAN = 16777216
comptime XNEIGHBORS_KIND_NEAREST_CENTROID = 0
comptime XNEIGHBORS_KIND_LOF = 1
comptime XNEIGHBORS_KIND_OCSVM = 2
comptime XNEIGHBORS_KIND_KERNEL_PCA = 3
comptime XNEIGHBORS_KIND_PCS = 4
comptime XNEIGHBORS_KIND_ACHI2 = 5
comptime XNEIGHBORS_KIND_SCHI2 = 6
comptime XNEIGHBORS_KIND_LABEL_PROPAGATION = 7
comptime XNEIGHBORS_KIND_LABEL_SPREADING = 8
comptime XNEIGHBORS_KIND_IMPUTER = 9
comptime XNEIGHBORS_KIND_SVGP = 10
comptime XNEIGHBORS_KIND_COUNT = 11
comptime NC_BLOCK_CENTROIDS = MEMBER_BLOCK_FAMILY
comptime NC_BLOCK_STD = MEMBER_BLOCK_FAMILY + 1
comptime NC_BLOCK_DEVIATIONS = MEMBER_BLOCK_FAMILY + 2
comptime NC_BLOCK_PRIOR = MEMBER_BLOCK_FAMILY + 3
comptime _NC_BLOCKS = 4
comptime LOF_BLOCK_ROWS = MEMBER_BLOCK_FAMILY
comptime LOF_BLOCK_DISTANCES = MEMBER_BLOCK_FAMILY + 1
comptime LOF_BLOCK_DENSITIES = MEMBER_BLOCK_FAMILY + 2
comptime LOF_BLOCK_SCORES = MEMBER_BLOCK_FAMILY + 3
comptime LOF_BLOCK_OFFSET = MEMBER_BLOCK_FAMILY + 4
comptime _LOF_BLOCKS = 5
comptime OC_BLOCK_VECTORS = MEMBER_BLOCK_FAMILY
comptime OC_BLOCK_DUAL = MEMBER_BLOCK_FAMILY + 1
comptime OC_BLOCK_SUPPORT = MEMBER_BLOCK_FAMILY + 2
comptime OC_BLOCK_SCALARS = MEMBER_BLOCK_FAMILY + 3
comptime _OC_BLOCKS = 4
comptime KP_BLOCK_ROWS = MEMBER_BLOCK_FAMILY
comptime KP_BLOCK_VALUES = MEMBER_BLOCK_FAMILY + 1
comptime KP_BLOCK_VECTORS = MEMBER_BLOCK_FAMILY + 2
comptime KP_BLOCK_COLUMNS = MEMBER_BLOCK_FAMILY + 3
comptime KP_BLOCK_TOTAL = MEMBER_BLOCK_FAMILY + 4
comptime KP_BLOCK_GAMMA = MEMBER_BLOCK_FAMILY + 5
comptime _KP_BLOCKS = 6
comptime PC_BLOCK_INDEX_HASH = MEMBER_BLOCK_FAMILY
comptime PC_BLOCK_BIT_HASH = MEMBER_BLOCK_FAMILY + 1
comptime PC_BLOCK_FEATURES = MEMBER_BLOCK_FAMILY + 2
comptime _PC_BLOCKS = 3
comptime AC_BLOCK_FEATURES = MEMBER_BLOCK_FAMILY
comptime SC_BLOCK_WEIGHTS = MEMBER_BLOCK_FAMILY
comptime SC_BLOCK_OFFSET = MEMBER_BLOCK_FAMILY + 1
comptime _SC_BLOCKS = 2
comptime IM_BLOCK_ROWS = MEMBER_BLOCK_FAMILY
comptime IM_BLOCK_VALID = MEMBER_BLOCK_FAMILY + 1
comptime IM_BLOCK_MISSING = MEMBER_BLOCK_FAMILY + 2
comptime _IM_BLOCKS = 3
comptime SV_BLOCK_Z = MEMBER_BLOCK_FAMILY
comptime SV_BLOCK_ALPHA = MEMBER_BLOCK_FAMILY + 1
comptime SV_BLOCK_C = MEMBER_BLOCK_FAMILY + 2
comptime SV_BLOCK_QMU = MEMBER_BLOCK_FAMILY + 3
comptime SV_BLOCK_QSQRT = MEMBER_BLOCK_FAMILY + 4
comptime SV_BLOCK_SCALARS = MEMBER_BLOCK_FAMILY + 5
comptime _SV_BLOCKS = 6
comptime LP_BLOCK_ROWS = MEMBER_BLOCK_FAMILY
comptime LP_BLOCK_DISTRIBUTIONS = MEMBER_BLOCK_FAMILY + 1
comptime LP_BLOCK_TRANSDUCTION = MEMBER_BLOCK_FAMILY + 2
comptime LP_BLOCK_SCALARS = MEMBER_BLOCK_FAMILY + 3
comptime _LP_BLOCKS = 4
comptime _LP_WORDS = 14
#: LabelPropagation / LabelSpreading's word [13] (lane mf-meta5): THE UNLABELED MARKER, the caller's class
#: code of the label -1 (its rows take part in the graph, not in classes_; -1: no marker).
comptime LP_MARKER_WORD = 13
#: OneClassSVM gamma words (MEMBER_OP_NEW word 5).
comptime OC_GAMMA_VALUE = 0
comptime OC_GAMMA_SCALE = 1
comptime OC_GAMMA_AUTO = 2
comptime _I32P = MutPointer[Int32, MutUntrackedOrigin]
comptime _I64P = MutPointer[Int64, MutUntrackedOrigin]
comptime _F32P = MutPointer[Float32, MutUntrackedOrigin]
comptime _F64P = MutPointer[Float64, MutUntrackedOrigin]


struct XNeighborsMemberSlot(Copyable, Movable, Defaultable):
    var generation: Int
    var live: Bool
    var kind: Int
    var fitted: Bool
    var rows: Int
    var features: Int
    var classes: Int
    # NearestCentroid words (kind 0).
    var euclidean: Bool
    var priors: Int
    var shrink: Float64
    var fused: Bool
    var wide: Bool
    var given: List[Float64]
    var centroids: List[Float32]
    var std: List[Float32]
    var deviations: List[Float32]
    var prior: List[Float64]
    # Per-state query scratch, grown once: the predicted codes and the
    # probabilities of the other width when `dst` differs from the image's.
    var pred: List[Int32]
    var proba32: List[Float32]
    var proba64: List[Float64]
    # LocalOutlierFactor words (kind 1). `rows_addr` = the borrowed fit X,
    # or `fit_rows` after an import or the first export.
    var k: Int
    var novelty: Bool
    var auto: Bool
    var contamination: Float64
    var unfused: Bool
    var old_items: Bool
    var neighbors: Int
    var offset: Float64
    var rows_addr: Int
    var fit_rows: List[Float32]
    var distances: List[Float32]
    var densities: List[Float32]
    var scores: List[Float32]
    var offset_word: List[Float64]
    # OneClassSVM words (kind 2) and its fitted state (owned: the support
    # vectors are gathered from X by the fit, as the door's PACK allocates them).
    var oc_kernel: Int
    var oc_nu: Float64
    var oc_degree: Int
    var oc_coef0: Float64
    var oc_gamma_mode: Int
    var oc_gamma_given: Float64
    var oc_cap: Int
    var oc_tol: Float64
    var oc_solve: Int
    var oc_fast: Bool
    var oc_labels: Bool
    var oc_old_items: Bool
    var oc_precomputed: Bool
    var oc_gamma: Float64
    var oc_rho: Float64
    var oc_iterations: Int
    var oc_support: List[Int32]
    var oc_vectors: List[Float32]
    var oc_dual: List[Float32]
    var oc_scalars: List[Float64]
    # KernelPCA words (kind 3) and its fitted state; the fitted X is the
    # borrowed `rows_addr` (or `fit_rows`), as for LocalOutlierFactor.
    var kp_kernel: Int
    var kp_requested: Int
    var kp_auto: Bool
    var kp_resident: Bool
    var kp_gamma_given: Bool
    var kp_gamma_value: Float64
    var kp_degree: Int
    var kp_coef0: Float64
    var kp_remove_zero: Bool
    var kp_old_items: Bool
    var kp_gamma: Float64
    var kp_kept: Int
    var kp_values: List[Float32]
    var kp_vectors: List[Float32]
    var kp_columns: List[Float32]
    var kp_total: List[Float32]
    var kp_alphas: List[Float32]
    var kp_alphas_ready: Bool
    # PolynomialCountSketch words (kind 4) and its two hash tables.
    var pc_degree: Int
    var pc_components: Int
    var pc_gamma: Float64
    var pc_coef0: Float64
    var pc_seed: Int
    var pc_old_items: Bool
    var pc_augmented: Int
    var pc_indices: List[Int32]
    var pc_signs: List[Int32]
    # AdditiveChi2Sampler words (kind 5); its fitted state is the width.
    var ac_steps: Int
    var ac_interval: Float64
    var ac_valid: Bool
    var ac_scan: Bool
    # SkewedChi2Sampler words (kind 6), its fitted weights and offsets, and
    # the transform's log scratch (grown once, reused by every later call).
    var sc_components: Int
    var sc_skew: Float64
    var sc_seed: Int
    var sc_lazy: Bool
    var sc_weights: List[Float32]
    var sc_offset: List[Float32]
    var sc_scratch: List[Float32]
    # KNNImputer words (kind 9) and its fitted columns; the fitted rows are
    # `rows_addr` (borrowed X, or `fit_rows`), as for LocalOutlierFactor.
    var im_k: Int
    var im_distance: Bool
    var im_mask: Bool
    var im_missing: Float64
    var im_keep_empty: Bool
    var im_indicator: Bool
    var im_uncompact: Bool
    var im_old_items: Bool
    var im_lean: Bool
    var im_valid: List[Int32]
    var im_miss: List[Int]
    var im_miss32: List[Int32]
    var im_plan: ImpKeepPlan
    # SVGP words (kind 10) and its fitted arrays (m inducing rows).
    var sv_inducing: Int
    var sv_gamma: Float64
    var sv_kernel_variance: Float64
    var sv_noise_variance: Float64
    var sv_jitter: Float64
    var sv_old_items: Bool
    var sv_given: Int
    var sv_given_rows: Int
    var sv_given_cols: Int
    var sv_m: Int
    var sv_elbo: Float64
    var sv_z: List[Float32]
    var sv_alpha: List[Float32]
    var sv_c: List[Float32]
    var sv_qmu: List[Float32]
    var sv_qsqrt: List[Float32]
    var sv_scalars: List[Float64]
    # LabelPropagation / LabelSpreading words (kinds 7, 8) and the fitted
    # state; the fitted X is the borrowed `rows_addr` (or `fit_rows`), as for
    # LocalOutlierFactor. `lp_codes` are the transduction's class codes.
    var lp_knn: Bool
    var lp_neighbors: Int
    var lp_kernel: Int
    var lp_gamma: Float64
    var lp_raw: Bool
    var lp_weight: Float64
    var lp_alpha: Float64
    var lp_tol: Float64
    var lp_max_iter: Int
    var lp_unfused: Bool
    var lp_old: Bool
    var lp_fused: Bool
    var lp_marker: Int
    var lp_k: Int
    var lp_iterations: Int
    var lp_dist: List[Float32]
    var lp_codes: List[Int32]
    var lp_scalars: List[Float64]

    def __init__(out self):
        self.generation = 0
        self.live = False
        self.kind = 0
        self.fitted = False
        self.rows = 0
        self.features = 0
        self.classes = 0
        self.euclidean = True
        self.priors = NC_PRIORS_UNIFORM
        self.shrink = 0.0
        self.fused = False
        self.wide = False
        self.given = List[Float64]()
        self.centroids = List[Float32]()
        self.std = List[Float32]()
        self.deviations = List[Float32]()
        self.prior = List[Float64]()
        self.pred = List[Int32]()
        self.proba32 = List[Float32]()
        self.proba64 = List[Float64]()
        self.k = 20
        self.novelty = False
        self.auto = True
        self.contamination = 0.0
        self.unfused = False
        self.old_items = False
        self.neighbors = 0
        self.offset = 0.0
        self.rows_addr = 0
        self.fit_rows = List[Float32]()
        self.distances = List[Float32]()
        self.densities = List[Float32]()
        self.scores = List[Float32]()
        self.offset_word = List[Float64](length=1, fill=0.0)
        self.oc_kernel = 0
        self.oc_nu = 0.5
        self.oc_degree = 3
        self.oc_coef0 = 0.0
        self.oc_gamma_mode = OC_GAMMA_SCALE
        self.oc_gamma_given = 0.0
        self.oc_cap = 10000000
        self.oc_tol = 0.001
        self.oc_solve = 0
        self.oc_fast = False
        self.oc_labels = False
        self.oc_old_items = False
        self.oc_precomputed = False
        self.oc_gamma = 0.0
        self.oc_rho = 0.0
        self.oc_iterations = 0
        self.oc_support = List[Int32]()
        self.oc_vectors = List[Float32]()
        self.oc_dual = List[Float32]()
        self.oc_scalars = List[Float64](length=4, fill=0.0)
        self.kp_kernel = 0
        self.kp_requested = -1
        self.kp_auto = True
        self.kp_resident = False
        self.kp_gamma_given = False
        self.kp_gamma_value = 0.0
        self.kp_degree = 3
        self.kp_coef0 = 1.0
        self.kp_remove_zero = False
        self.kp_old_items = False
        self.kp_gamma = 0.0
        self.kp_kept = 0
        self.kp_values = List[Float32]()
        self.kp_vectors = List[Float32]()
        self.kp_columns = List[Float32]()
        self.kp_total = List[Float32]()
        self.kp_alphas = List[Float32]()
        self.kp_alphas_ready = False
        self.pc_degree = 2
        self.pc_components = 100
        self.pc_gamma = 1.0
        self.pc_coef0 = 0.0
        self.pc_seed = 0
        self.pc_old_items = False
        self.pc_augmented = 0
        self.pc_indices = List[Int32]()
        self.pc_signs = List[Int32]()
        self.ac_steps = 2
        self.ac_interval = 0.5
        self.ac_valid = True
        self.ac_scan = False
        self.sc_components = 100
        self.sc_skew = 1.0
        self.sc_seed = 0
        self.sc_lazy = False
        self.sc_weights = List[Float32]()
        self.sc_offset = List[Float32]()
        self.sc_scratch = List[Float32]()
        self.im_k = 5
        self.im_distance = False
        self.im_mask = False
        self.im_missing = 0.0
        self.im_keep_empty = False
        self.im_indicator = False
        self.im_uncompact = False
        self.im_old_items = False
        self.im_lean = False
        self.im_valid = List[Int32]()
        self.im_miss = List[Int]()
        self.im_miss32 = List[Int32]()
        self.im_plan = ImpKeepPlan(List[Int](), List[Int](), True)
        self.sv_inducing = 100
        self.sv_gamma = 0.5
        self.sv_kernel_variance = 1.0
        self.sv_noise_variance = 1.0
        self.sv_jitter = 0.0
        self.sv_old_items = False
        self.sv_given = 0
        self.sv_given_rows = 0
        self.sv_given_cols = 0
        self.sv_m = 0
        self.sv_elbo = 0.0
        self.sv_z = List[Float32]()
        self.sv_alpha = List[Float32]()
        self.sv_c = List[Float32]()
        self.sv_qmu = List[Float32]()
        self.sv_qsqrt = List[Float32]()
        self.sv_scalars = List[Float64](length=2, fill=0.0)
        self.lp_knn = False
        self.lp_neighbors = 7
        self.lp_kernel = 0
        self.lp_gamma = 20.0
        self.lp_raw = False
        self.lp_weight = 1.0
        self.lp_alpha = 0.0
        self.lp_tol = 0.001
        self.lp_max_iter = 1000
        self.lp_unfused = False
        self.lp_old = False
        self.lp_marker = -1
        self.lp_fused = False
        self.lp_k = 0
        self.lp_iterations = 0
        self.lp_dist = List[Float32]()
        self.lp_codes = List[Int32]()
        self.lp_scalars = List[Float64](length=2, fill=0.0)


struct XNeighborsMemberStore(Defaultable, Movable):
    var synchronization: List[Int64]
    var chunks: List[List[XNeighborsMemberSlot]]
    var free: List[Int]
    var count: Int

    def __init__(out self):
        # Side-effect-free even when the process slot's initializers race.
        self.synchronization = List[Int64](length=2, fill=0)
        self.chunks = List[List[XNeighborsMemberSlot]]()
        self.free = List[Int]()
        self.count = 0


comptime _STORE = SharedSlot[XNeighborsMemberStore, "MojolearnXNeighborsMemberStoreV2", 2]
comptime _SlotPtr = MutPointer[XNeighborsMemberSlot, MutUntrackedOrigin]
comptime _CallPtr = MutPointer[MemberCall, MutUntrackedOrigin]


def _slot_address(index: Int) raises -> Int:
    # Under the store's lock; a chunk's buffer never grows.
    var store = _STORE.get_or_create_ptr()
    return Int(store[].chunks[index // _CHUNK].unsafe_ptr()) + (index % _CHUNK) * size_of[XNeighborsMemberSlot]()


def _clear(slot: _SlotPtr):
    slot[].fitted = False
    slot[].given = List[Float64]()
    slot[].centroids = List[Float32]()
    slot[].std = List[Float32]()
    slot[].deviations = List[Float32]()
    slot[].prior = List[Float64]()
    slot[].pred = List[Int32]()
    slot[].proba32 = List[Float32]()
    slot[].proba64 = List[Float64]()
    slot[].rows_addr = 0
    slot[].fit_rows = List[Float32]()
    slot[].distances = List[Float32]()
    slot[].densities = List[Float32]()
    slot[].scores = List[Float32]()
    slot[].oc_support = List[Int32]()
    slot[].oc_vectors = List[Float32]()
    slot[].oc_dual = List[Float32]()
    slot[].kp_values = List[Float32]()
    slot[].kp_vectors = List[Float32]()
    slot[].kp_columns = List[Float32]()
    slot[].kp_total = List[Float32]()
    slot[].kp_alphas = List[Float32]()
    slot[].kp_alphas_ready = False
    slot[].pc_indices = List[Int32]()
    slot[].pc_signs = List[Int32]()
    slot[].sc_weights = List[Float32]()
    slot[].sc_offset = List[Float32]()
    slot[].sc_scratch = List[Float32]()
    slot[].im_valid = List[Int32]()
    slot[].im_miss = List[Int]()
    slot[].im_miss32 = List[Int32]()
    slot[].im_plan = ImpKeepPlan(List[Int](), List[Int](), True)
    slot[].sv_z = List[Float32]()
    slot[].sv_alpha = List[Float32]()
    slot[].sv_c = List[Float32]()
    slot[].sv_qmu = List[Float32]()
    slot[].sv_qsqrt = List[Float32]()
    slot[].lp_dist = List[Float32]()
    slot[].lp_codes = List[Int32]()
    slot[].lp_k = 0
    slot[].lp_iterations = 0


def _mint(words: _F64P, count: Int) raises -> Int:
    var store = _STORE.get_or_create_ptr()
    var lock = rebind[MutPointer[Int64, MutUntrackedOrigin]](store[].synchronization.unsafe_ptr())
    shared_slot_lock(lock)
    try:
        var index = -1
        if len(store[].free) > 0:
            index = store[].free.pop()
        else:
            if store[].count >= _INDEX_SPAN - 1:
                raise Error("RuntimeError: mojolearn: too many live XNeighbors member states")
            if store[].count == len(store[].chunks) * _CHUNK:
                store[].chunks.append(List[XNeighborsMemberSlot](length=_CHUNK, fill=XNeighborsMemberSlot()))
            index = store[].count
            store[].count += 1
        var slot = _SlotPtr(unsafe_from_address=_slot_address(index))
        slot[].generation += 1
        slot[].live = True
        _clear(slot)
        slot[].kind = Int(words[0])
        if slot[].kind == XNEIGHBORS_KIND_NEAREST_CENTROID:
            slot[].euclidean = words[1] == 0.0
            slot[].priors = Int(words[2])
            slot[].shrink = words[3]
            slot[].fused = words[4] != 0.0
            slot[].wide = words[5] != 0.0
            for i in range(6, count):  # driver-loop: over the given priors (one word per class)
                slot[].given.append(words[i])
        elif slot[].kind == XNEIGHBORS_KIND_OCSVM:
            slot[].oc_kernel = Int(words[1])
            slot[].oc_nu = words[2]
            slot[].oc_degree = Int(words[3])
            slot[].oc_coef0 = words[4]
            slot[].oc_gamma_mode = Int(words[5])
            slot[].oc_gamma_given = words[6]
            slot[].oc_cap = Int(words[7])
            slot[].oc_tol = words[8]
            slot[].oc_solve = Int(words[9])
            slot[].oc_fast = words[10] != 0.0
            slot[].oc_labels = words[11] != 0.0
            slot[].oc_old_items = words[12] != 0.0
            slot[].oc_precomputed = words[13] != 0.0
        elif slot[].kind == XNEIGHBORS_KIND_KERNEL_PCA:
            slot[].kp_kernel = Int(words[1])
            slot[].kp_requested = Int(words[2])
            slot[].kp_auto = words[3] != 0.0
            slot[].kp_resident = words[4] != 0.0
            slot[].kp_gamma_given = words[5] != 0.0
            slot[].kp_gamma_value = words[6]
            slot[].kp_degree = Int(words[7])
            slot[].kp_coef0 = words[8]
            slot[].kp_remove_zero = words[9] != 0.0
            slot[].kp_old_items = words[10] != 0.0
        elif slot[].kind == XNEIGHBORS_KIND_PCS:
            slot[].pc_degree = Int(words[1])
            slot[].pc_components = Int(words[2])
            slot[].pc_gamma = words[3]
            slot[].pc_coef0 = words[4]
            slot[].pc_seed = Int(words[5])
            slot[].pc_old_items = words[6] != 0.0
        elif slot[].kind == XNEIGHBORS_KIND_ACHI2:
            slot[].ac_steps = Int(words[1])
            slot[].ac_interval = words[2]
            slot[].ac_valid = words[3] != 0.0
            slot[].ac_scan = words[4] != 0.0
        elif slot[].kind == XNEIGHBORS_KIND_SCHI2:
            slot[].sc_components = Int(words[1])
            slot[].sc_skew = words[2]
            slot[].sc_seed = Int(words[3])
            slot[].sc_lazy = words[4] != 0.0
        elif slot[].kind == XNEIGHBORS_KIND_IMPUTER:
            slot[].im_k = Int(words[1])
            slot[].im_distance = words[2] != 0.0
            slot[].im_mask = words[3] != 0.0
            slot[].im_missing = words[4]
            slot[].im_keep_empty = words[5] != 0.0
            slot[].im_indicator = words[6] != 0.0
            slot[].im_uncompact = words[7] != 0.0
            slot[].im_old_items = (words[8] != 0.0) and not slot[].im_uncompact
            slot[].im_lean = words[9] != 0.0
        elif slot[].kind == XNEIGHBORS_KIND_LABEL_PROPAGATION or slot[].kind == XNEIGHBORS_KIND_LABEL_SPREADING:
            slot[].lp_knn = words[1] != 0.0
            slot[].lp_neighbors = Int(words[2])
            slot[].lp_kernel = Int(words[3])
            slot[].lp_gamma = words[4]
            slot[].lp_raw = words[5] != 0.0
            slot[].lp_weight = words[6]
            slot[].lp_alpha = words[7]
            slot[].lp_tol = words[8]
            slot[].lp_max_iter = Int(words[9])
            slot[].lp_unfused = words[10] != 0.0
            slot[].lp_old = words[11] != 0.0
            slot[].lp_fused = words[12] != 0.0
            slot[].lp_marker = Int(words[LP_MARKER_WORD])
        elif slot[].kind == XNEIGHBORS_KIND_SVGP:
            slot[].sv_inducing = Int(words[1])
            slot[].sv_gamma = words[2]
            slot[].sv_kernel_variance = words[3]
            slot[].sv_noise_variance = words[4]
            slot[].sv_jitter = words[5]
            slot[].sv_old_items = words[6] != 0.0
            slot[].sv_given = Int(words[7])
            slot[].sv_given_rows = Int(words[8])
            slot[].sv_given_cols = Int(words[9])
        else:
            slot[].k = Int(words[1])
            slot[].novelty = words[2] != 0.0
            slot[].auto = words[3] != 0.0
            slot[].contamination = words[4]
            slot[].unfused = words[5] != 0.0
            slot[].old_items = (words[6] != 0.0) and not slot[].unfused
        return slot[].generation * _INDEX_SPAN + index + 1
    finally:
        shared_slot_unlock(lock)


def _lookup(handle: Int, release: Bool) raises -> Int:
    """The slot address of a live handle; a released, foreign or made-up
    handle is refused (never a stale read)."""
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
            slot[].live = False
            _clear(slot)
            store[].free.append(index)
        return address
    finally:
        shared_slot_unlock(lock)


def _name(slot: _SlotPtr) -> String:
    if slot[].kind == XNEIGHBORS_KIND_NEAREST_CENTROID:
        return String("NearestCentroid")
    if slot[].kind == XNEIGHBORS_KIND_OCSVM:
        return String("OneClassSVM")
    if slot[].kind == XNEIGHBORS_KIND_KERNEL_PCA:
        return String("KernelPCA")
    if slot[].kind == XNEIGHBORS_KIND_PCS:
        return String("PolynomialCountSketch")
    if slot[].kind == XNEIGHBORS_KIND_ACHI2:
        return String("AdditiveChi2Sampler")
    if slot[].kind == XNEIGHBORS_KIND_SCHI2:
        return String("SkewedChi2Sampler")
    if slot[].kind == XNEIGHBORS_KIND_IMPUTER:
        return String("KNNImputer")
    if slot[].kind == XNEIGHBORS_KIND_SVGP:
        return String("SVGP")
    if slot[].kind == XNEIGHBORS_KIND_LABEL_PROPAGATION:
        return String("LabelPropagation")
    if slot[].kind == XNEIGHBORS_KIND_LABEL_SPREADING:
        return String("LabelSpreading")
    return String("LocalOutlierFactor")


def _whole(call: _CallPtr, estimator: String) raises:
    if call[].row_index.rows > 0 or call[].column_index.rows > 0:
        raise Error("TypeError: a " + estimator + " member reads whole host views (no row or column index)")


def _dense(view: MemberMatrix, dtype: Int, what: String, estimator: String) raises:
    if not view.dense_host(dtype):
        var word = String("int32") if dtype == MEMBER_I32 else (
            String("float64") if dtype == MEMBER_F64 else String("float32"))
        raise Error("TypeError: a " + estimator + " member reads " + what + " as a dense host " + word + " view")


# ------------------------------------------------------------ NearestCentroid


def _nc_fit(slot: _SlotPtr, call: _CallPtr) raises:
    var estimator = _name(slot)
    _whole(call, estimator)
    if call[].fit_output != 0:
        member_unsupported(estimator, call[].fit_output)
    if call[].weights.rows > 0:
        raise Error("TypeError: a " + estimator + " member fits no sample_weight")
    _dense(call[].x, MEMBER_F32, "X", estimator)
    _dense(call[].y, MEMBER_I32, "y (class codes)", estimator)
    var rows = call[].x.rows
    var features = call[].x.cols
    if call[].y.cols != 1 or call[].y.rows != rows:
        raise Error("X and y have different numbers of rows")
    var classes = call[].classes
    nc_check_classes(classes)
    if slot[].priors == NC_PRIORS_GIVEN:
        nc_check_priors_size(len(slot[].given), classes)
    var driver = XnDriver(call[].route.entry.call)
    slot[].fitted = False
    slot[].pred = List[Int32]()
    slot[].proba32 = List[Float32]()
    slot[].proba64 = List[Float64]()
    # The four fitted arrays are the state's own (the estimator's door
    # allocates the same four as its published attributes).
    slot[].prior = List[Float64](length=classes, fill=0.0)
    slot[].std = List[Float32](length=features, fill=Float32(0.0))
    slot[].centroids = List[Float32](length=classes * features, fill=Float32(0.0))
    slot[].deviations = List[Float32](length=classes * features, fill=Float32(0.0))
    if slot[].priors == NC_PRIORS_GIVEN:
        unsafe_memcpy(dest=slot[].prior.unsafe_ptr(), src=slot[].given.unsafe_ptr(), count=classes)
    var request = NcFitRequest(call[].x.data, call[].y.data, rows, features, classes,
        slot[].euclidean, slot[].priors, slot[].shrink)
    request.prior = Int(slot[].prior.unsafe_ptr())
    request.stats = Int(slot[].std.unsafe_ptr())
    request.centroids = Int(slot[].centroids.unsafe_ptr())
    request.deviations = Int(slot[].deviations.unsafe_ptr())
    var plan = NcFitPlan(rows, features, classes)
    nc_fit_count_run(driver, request, plan)
    for c in range(classes):  # driver-loop: over classes (the fit's class counts)
        if plan.wide_counts[c] == 0:
            raise Error("ValueError: a " + estimator + " member needs every class code 0 .. "
                + String(classes - 1) + " present in its fit rows (class " + String(c) + " is absent)")
    if slot[].priors == NC_PRIORS_EMPIRICAL:
        # The label family's ratio step, mode 1: Float64(count) / Float64(rows).
        _ = class_ratio_f64(_I64P(unsafe_from_address=Int(plan.wide_counts.unsafe_ptr())),
            _F64P(unsafe_from_address=request.prior), classes, rows, 1)
    else:
        nc_fit_priors_run(request)
    var outcome = nc_fit_run(driver, request, plan)
    if outcome.status != 0:
        admitted_finite_refuse(outcome.status, estimator)
    nc_check_variance(outcome.constant)
    _ = plan^
    slot[].rows = rows
    slot[].features = features
    slot[].classes = classes
    slot[].fitted = True


def _nc_request(slot: _SlotPtr, call: _CallPtr, scored: Bool) raises -> NcQueryRequest:
    var estimator = _name(slot)
    if not slot[].fitted:
        raise Error("NotFittedError: mojolearn: call fit before predict")
    _whole(call, estimator)
    if scored and not slot[].euclidean:
        nc_refuse_decision_metric()
    _dense(call[].x, MEMBER_F32, "X", estimator)
    if call[].x.cols != slot[].features:
        raise Error("ValueError: X has " + String(call[].x.cols) + " features, but " + estimator
            + " is expecting " + String(slot[].features) + " features as input.")
    return NcQueryRequest(call[].x.data, Int(slot[].centroids.unsafe_ptr()), Int(slot[].std.unsafe_ptr()),
        Int(slot[].prior.unsafe_ptr()), call[].x.rows, slot[].features, slot[].classes,
        slot[].euclidean, slot[].fused, slot[].wide)


def _nc_predict(slot: _SlotPtr, call: _CallPtr) raises:
    var estimator = _name(slot)
    var request = _nc_request(slot, call, False)
    var rows = request.rows
    _dense(call[].dst, MEMBER_I32, "the prediction buffer (class codes)", estimator)
    if call[].dst.rows != rows or call[].dst.cols != 1:
        raise Error("ValueError: the prediction buffer holds one class code per row of X")
    if rows == 0:
        return
    # The RUN writes into its code holder; the state's buffer (grown once)
    # stands in for it, so a query allocates one word, not one per row.
    if len(slot[].pred) < rows:
        slot[].pred = List[Int32](length=rows, fill=Int32(0))
    var codes = NcLabelCodes(1)
    swap(codes.values, slot[].pred)
    var status = 0
    try:
        status = nc_labels_run(XnDriver(call[].route.entry.call), request, codes)
    finally:
        swap(codes.values, slot[].pred)
    _ = codes^
    if status != 0:
        admitted_finite_refuse(status, estimator)
    unsafe_memcpy(dest=_I32P(unsafe_from_address=call[].dst.data), src=slot[].pred.unsafe_ptr(), count=rows)


def _nc_decision(slot: _SlotPtr, call: _CallPtr) raises:
    var estimator = _name(slot)
    var request = _nc_request(slot, call, True)
    var width = 1 if slot[].classes == 2 else slot[].classes
    _dense(call[].dst, MEMBER_F32, "the decision buffer", estimator)
    if call[].dst.rows != request.rows or call[].dst.cols != width:
        raise Error("ValueError: the decision buffer holds " + String(width) + " scores per row of X")
    var status = nc_decision_run(XnDriver(call[].route.entry.call), request, call[].dst.data)
    if status != 0:
        admitted_finite_refuse(status, estimator)


def _nc_proba(slot: _SlotPtr, call: _CallPtr) raises:
    var estimator = _name(slot)
    var request = _nc_request(slot, call, True)
    var rows = request.rows
    var classes = slot[].classes
    var wide = nc_proba_wide(request, False)
    var asked_wide = call[].dst.dtype == MEMBER_F64
    _dense(call[].dst, MEMBER_F64 if asked_wide else MEMBER_F32, "the probability buffer", estimator)
    if call[].dst.rows != rows or call[].dst.cols != classes:
        raise Error("ValueError: the probability buffer holds one row of " + String(classes)
            + " class probabilities per row of X")
    var count = rows * classes
    var target = call[].dst.data
    if wide != asked_wide:
        # The image computes the other width: into the state's scratch,
        # cast below (parallel over rows).
        if wide:
            if len(slot[].proba64) < count:
                slot[].proba64 = List[Float64](length=count, fill=0.0)
            target = Int(slot[].proba64.unsafe_ptr())
        else:
            if len(slot[].proba32) < count:
                slot[].proba32 = List[Float32](length=count, fill=Float32(0.0))
            target = Int(slot[].proba32.unsafe_ptr())
    var status = nc_proba_run(XnDriver(call[].route.entry.call), request, target, False)
    if status != 0:
        admitted_finite_refuse(status, estimator)
    if wide == asked_wide or count == 0:
        return
    var src = target
    var dst = call[].dst.data
    var tasks = max(host_predict_task_count_work(rows, count), 1)
    var chunk = host_predict_chunk(rows, tasks)

    def _cast(t: Int) {imm src, imm dst, imm rows, imm classes, imm chunk, imm wide}:
        var lo = t * chunk * classes
        var hi = min((t + 1) * chunk, rows) * classes
        if wide:
            var s = _F64P(unsafe_from_address=src)
            var d = _F32P(unsafe_from_address=dst)
            for i in range(lo, hi):  # member predict_proba: this task's words
                d.unsafe_store(i, Float32(s.unsafe_load(i)))
        else:
            var s = _F32P(unsafe_from_address=src)
            var d = _F64P(unsafe_from_address=dst)
            for i in range(lo, hi):  # member predict_proba: this task's words
                d.unsafe_store(i, Float64(s.unsafe_load(i)))

    if tasks <= 1:
        _cast(0)
    else:
        host_parallelize(_cast, tasks)


def _nc_block(slot: _SlotPtr, block: Int) raises -> MemberMatrix:
    if block == NC_BLOCK_CENTROIDS:
        return MemberMatrix(Int(slot[].centroids.unsafe_ptr()), slot[].classes, slot[].features)
    if block == NC_BLOCK_STD:
        return MemberMatrix(Int(slot[].std.unsafe_ptr()), 1, slot[].features)
    if block == NC_BLOCK_DEVIATIONS:
        return MemberMatrix(Int(slot[].deviations.unsafe_ptr()), slot[].classes, slot[].features)
    if block == NC_BLOCK_PRIOR:
        return MemberMatrix(Int(slot[].prior.unsafe_ptr()), 1, slot[].classes, MEMBER_F64)
    raise Error("member ABI: a NearestCentroid member has no fitted block " + String(block))


def _nc_export(slot: _SlotPtr, call: _CallPtr) raises:
    """A view of one fitted block; the slot owns the words until this
    state's next fit, import or free."""
    if not slot[].fitted:
        raise Error("NotFittedError: this NearestCentroid member is not fitted yet")
    var block = call[].block
    if (call[].flags & MEMBER_BLOCK_BY_INDEX) != 0:
        if block < 0 or block >= _NC_BLOCKS:
            raise Error("member ABI: a NearestCentroid member has four fitted blocks")
        block = NC_BLOCK_CENTROIDS + block
    call[].dst = _nc_block(slot, block)
    call[].block = block
    call[].out_cols = _NC_BLOCKS
    call[].classes = slot[].classes


def _nc_import(slot: _SlotPtr, call: _CallPtr) raises:
    """Copy one fitted block in; MEMBER_IMPORT_DONE completes the state."""
    var estimator = _name(slot)
    var block = call[].block
    var count = call[].x.rows * call[].x.cols
    if block == NC_BLOCK_PRIOR:
        _dense(call[].x, MEMBER_F64, "the class priors", estimator)
        slot[].fitted = False
        slot[].prior = List[Float64](length=count, fill=0.0)
        unsafe_memcpy(dest=slot[].prior.unsafe_ptr(), src=_F64P(unsafe_from_address=call[].x.data), count=count)
    elif block == NC_BLOCK_CENTROIDS or block == NC_BLOCK_STD or block == NC_BLOCK_DEVIATIONS:
        _dense(call[].x, MEMBER_F32, "a fitted block", estimator)
        slot[].fitted = False
        var words = List[Float32](length=count, fill=Float32(0.0))
        unsafe_memcpy(dest=words.unsafe_ptr(), src=_F32P(unsafe_from_address=call[].x.data), count=count)
        if block == NC_BLOCK_CENTROIDS:
            slot[].centroids = words^
            slot[].features = call[].x.cols
        elif block == NC_BLOCK_STD:
            slot[].std = words^
        else:
            slot[].deviations = words^
    elif block != 0 or (call[].flags & MEMBER_IMPORT_DONE) == 0:
        raise Error("member ABI: a NearestCentroid member has no fitted block " + String(block))
    if (call[].flags & MEMBER_IMPORT_DONE) == 0:
        return
    var classes = call[].classes
    var features = slot[].features
    nc_check_classes(classes)
    if (len(slot[].prior) != classes or len(slot[].centroids) != classes * features
            or len(slot[].deviations) != classes * features or len(slot[].std) != features or features == 0):
        raise Error("member ABI: a fitted NearestCentroid state needs its centroids, deviations, "
            + "standard deviations and one prior per class")
    slot[].classes = classes
    slot[].pred = List[Int32]()
    slot[].proba32 = List[Float32]()
    slot[].proba64 = List[Float64]()
    slot[].fitted = True


# ------------------------------------------------------------ LocalOutlierFactor


def _lof_fit(slot: _SlotPtr, call: _CallPtr) raises:
    var estimator = _name(slot)
    _whole(call, estimator)
    if call[].weights.rows > 0:
        raise Error("TypeError: a " + estimator + " member fits no sample_weight")
    var labels = call[].fit_output == MEMBER_OP_PREDICT
    if call[].fit_output != 0 and not labels:
        member_unsupported(estimator, call[].fit_output)
    if labels and slot[].novelty:
        lof_refuse_fit_predict_novelty()
    _dense(call[].x, MEMBER_F32, "X", estimator)
    var rows = call[].x.rows
    var features = call[].x.cols
    lof_check_fit_rows(rows, features)
    if labels:
        _dense(call[].dst, MEMBER_I64, "the label buffer", estimator)
        if call[].dst.rows != rows or call[].dst.cols != 1:
            raise Error("ValueError: the label buffer holds one label per row of X")
    var k = lof_fit_neighbors(slot[].k, rows)
    slot[].fitted = False
    # Borrow the rows (LIFETIME); a copy a previous import or export left is dropped.
    slot[].fit_rows = List[Float32]()
    slot[].rows_addr = call[].x.data
    slot[].distances = List[Float32](length=rows * k, fill=Float32(0.0))
    slot[].densities = List[Float32](length=rows, fill=Float32(0.0))
    slot[].scores = List[Float32](length=rows, fill=Float32(0.0))
    var request = LofFitRequest(call[].x.data, rows, features, k, slot[].unfused, slot[].old_items,
        slot[].auto, slot[].contamination)
    request.distances = Int(slot[].distances.unsafe_ptr())
    request.densities = Int(slot[].densities.unsafe_ptr())
    request.scores = Int(slot[].scores.unsafe_ptr())
    var driver = XnDriver(call[].route.entry.call)
    var outcome = lof_fit_run(driver, request)
    if outcome.status != 0:
        admitted_finite_refuse(outcome.status, estimator)
    if labels:
        lof_fit_labels_run(driver, request.scores, rows, outcome.offset, call[].dst.data)
    slot[].rows = rows
    slot[].features = features
    slot[].neighbors = k
    slot[].offset = outcome.offset
    slot[].fitted = True


def _lof_query(slot: _SlotPtr, call: _CallPtr, kind: Int, what: String) raises:
    var estimator = _name(slot)
    lof_check_novelty(slot[].novelty, what)
    if not slot[].fitted:
        raise Error("NotFittedError: mojolearn: call fit before " + what)
    _whole(call, estimator)
    _dense(call[].x, MEMBER_F32, "X", estimator)
    if call[].x.cols != slot[].features:
        raise Error("ValueError: X has " + String(call[].x.cols) + " features, but " + estimator
            + " is expecting " + String(slot[].features) + " features as input.")
    var rows = call[].x.rows
    _dense(call[].dst, MEMBER_I64 if kind == LOF_RESULT_LABELS else MEMBER_F32, "the result buffer", estimator)
    if call[].dst.rows != rows or call[].dst.cols != 1:
        raise Error("ValueError: the result buffer holds one value per row of X")
    var request = LofQueryRequest(call[].x.data, slot[].rows_addr, Int(slot[].distances.unsafe_ptr()),
        Int(slot[].densities.unsafe_ptr()), rows, slot[].rows, slot[].features, slot[].neighbors, slot[].rows,
        slot[].unfused, slot[].old_items, kind, 0.0 if kind == LOF_RESULT_SCORES else -slot[].offset)
    request.dst = call[].dst.data
    var status = lof_query_run(XnDriver(call[].route.entry.call), request)
    admitted_finite_refuse(status, estimator)


def _lof_own_rows(slot: _SlotPtr):
    """Copy the borrowed fit rows into the slot once (first export)."""
    var count = slot[].rows * slot[].features
    if slot[].rows_addr == Int(slot[].fit_rows.unsafe_ptr()) and len(slot[].fit_rows) == count:
        return
    var owned = List[Float32](length=count, fill=Float32(0.0))
    unsafe_memcpy(dest=owned.unsafe_ptr(), src=_F32P(unsafe_from_address=slot[].rows_addr), count=count)
    slot[].fit_rows = owned^
    slot[].rows_addr = Int(slot[].fit_rows.unsafe_ptr())


def _lof_export(slot: _SlotPtr, call: _CallPtr) raises:
    """A view of one fitted block; the slot owns the words until this
    state's next fit, import or free (the borrowed rows are copied here)."""
    if not slot[].fitted:
        raise Error("NotFittedError: this LocalOutlierFactor member is not fitted yet")
    if (call[].flags & MEMBER_EXPORT_OWN) != 0:
        # THE LIFETIME RULE: the borrowed fit rows, once; no view is answered.
        _lof_own_rows(slot)
        return
    var block = call[].block
    if (call[].flags & MEMBER_BLOCK_BY_INDEX) != 0:
        if block < 0 or block >= _LOF_BLOCKS:
            raise Error("member ABI: a LocalOutlierFactor member has five fitted blocks")
        block = LOF_BLOCK_ROWS + block
    if block == LOF_BLOCK_ROWS:
        _lof_own_rows(slot)
        call[].dst = MemberMatrix(slot[].rows_addr, slot[].rows, slot[].features)
    elif block == LOF_BLOCK_DISTANCES:
        call[].dst = MemberMatrix(Int(slot[].distances.unsafe_ptr()), slot[].rows, slot[].neighbors)
    elif block == LOF_BLOCK_DENSITIES:
        call[].dst = MemberMatrix(Int(slot[].densities.unsafe_ptr()), 1, slot[].rows)
    elif block == LOF_BLOCK_SCORES:
        call[].dst = MemberMatrix(Int(slot[].scores.unsafe_ptr()), 1, slot[].rows)
    elif block == LOF_BLOCK_OFFSET:
        slot[].offset_word[0] = slot[].offset
        call[].dst = MemberMatrix(Int(slot[].offset_word.unsafe_ptr()), 1, 1, MEMBER_F64)
    else:
        raise Error("member ABI: a LocalOutlierFactor member has no fitted block " + String(block))
    call[].block = block
    call[].out_cols = _LOF_BLOCKS


def _lof_import(slot: _SlotPtr, call: _CallPtr) raises:
    """Copy one fitted block in; MEMBER_IMPORT_DONE completes the state."""
    var estimator = _name(slot)
    var block = call[].block
    var count = call[].x.rows * call[].x.cols
    if block == LOF_BLOCK_OFFSET:
        _dense(call[].x, MEMBER_F64, "the offset", estimator)
        if count != 1:
            raise Error("member ABI: a LocalOutlierFactor offset is one float64 word")
        slot[].fitted = False
        slot[].offset = _F64P(unsafe_from_address=call[].x.data)[0]
    elif block >= LOF_BLOCK_ROWS and block <= LOF_BLOCK_SCORES:
        _dense(call[].x, MEMBER_F32, "a fitted block", estimator)
        slot[].fitted = False
        var words = List[Float32](length=count, fill=Float32(0.0))
        unsafe_memcpy(dest=words.unsafe_ptr(), src=_F32P(unsafe_from_address=call[].x.data), count=count)
        if block == LOF_BLOCK_ROWS:
            slot[].fit_rows = words^
            slot[].rows_addr = Int(slot[].fit_rows.unsafe_ptr())
            slot[].rows = call[].x.rows
            slot[].features = call[].x.cols
        elif block == LOF_BLOCK_DISTANCES:
            slot[].distances = words^
            slot[].neighbors = call[].x.cols
        elif block == LOF_BLOCK_DENSITIES:
            slot[].densities = words^
        else:
            slot[].scores = words^
    elif block != 0 or (call[].flags & MEMBER_IMPORT_DONE) == 0:
        raise Error("member ABI: a LocalOutlierFactor member has no fitted block " + String(block))
    if (call[].flags & MEMBER_IMPORT_DONE) == 0:
        return
    var rows = slot[].rows
    if (rows == 0 or len(slot[].fit_rows) != rows * slot[].features or slot[].neighbors < 1
            or len(slot[].distances) != rows * slot[].neighbors or len(slot[].densities) != rows
            or len(slot[].scores) != rows):
        raise Error("member ABI: a fitted LocalOutlierFactor state needs its rows, k-distances, densities, "
            + "scores and offset")
    slot[].fitted = True


# ------------------------------------------------------------ OneClassSVM


def _oc_fit(slot: _SlotPtr, call: _CallPtr) raises:
    """`OneClassSVM.fit` (bindings/svc_ocsvm_native_door.mojo `_ocsvm_fit`):
    the request the door's UNPACK builds, `oc_fit_run`, the support prefix
    and the support vectors gathered from X, in the door's order."""
    var estimator = _name(slot)
    _whole(call, estimator)
    if call[].weights.rows > 0:
        raise Error("TypeError: a " + estimator + " member fits no sample_weight")
    var labels = call[].fit_output == MEMBER_OP_PREDICT
    if call[].fit_output != 0 and not labels:
        member_unsupported(estimator, call[].fit_output)
    _dense(call[].x, MEMBER_F32, "X", estimator)
    var rows = call[].x.rows
    var features = call[].x.cols
    if labels:
        _dense(call[].dst, MEMBER_I64, "the label buffer", estimator)
        if call[].dst.rows != rows or call[].dst.cols != 1:
            raise Error("ValueError: the label buffer holds one label per row of X")
    var precomputed = slot[].oc_precomputed
    if precomputed and rows != features:
        # `oc_fit_unpack`'s words, after X's admission.
        raise Error("Precomputed matrix must be a square matrix.")
    var request = OcFitRequest()
    request.x = call[].x.data
    request.rows = rows
    request.features = features
    request.precomputed = precomputed
    request.nu = slot[].oc_nu
    request.degree = slot[].oc_degree
    request.coef0 = slot[].oc_coef0
    if slot[].oc_gamma_mode == OC_GAMMA_SCALE:
        request.gamma_scale = True  # 1 / (features * variance of X), read by RUN
    elif slot[].oc_gamma_mode == OC_GAMMA_AUTO:
        request.gamma = 1.0 / Float64(features)
    else:
        request.gamma = slot[].oc_gamma_given
    request.cap = slot[].oc_cap
    request.tol = slot[].oc_tol
    request.kernel = slot[].oc_kernel
    request.solve = slot[].oc_solve
    request.resident_fast = slot[].oc_fast
    var driver = XnDriver(call[].route.entry.call)
    var resident = Optional[OcResidentDriver]()
    if slot[].oc_solve == OC_SOLVE_RESIDENT:
        if not call[].route.has_second:
            raise Error("member ABI: a OneClassSVM member's resident solve needs the route's second entry")
        resident = Optional(OcResidentDriver(call[].route.second.call))
    slot[].fitted = False
    var run = oc_fit_run(driver, resident, request)
    var supports = run.state.supports
    # Unweighted: the support is the solve's compacted prefix (the door's
    # `svc_prefix`); the vectors are those rows of X (`oc_take_rows_into`).
    slot[].oc_support = List[Int32](length=supports, fill=Int32(0))
    slot[].oc_dual = List[Float32](length=supports, fill=Float32(0.0))
    if supports > 0:
        unsafe_memcpy(dest=slot[].oc_support.unsafe_ptr(), src=run.state.support_rows.unsafe_ptr(), count=supports)
        unsafe_memcpy(dest=slot[].oc_dual.unsafe_ptr(), src=run.state.support_values.unsafe_ptr(), count=supports)
    # A precomputed kernel keeps no support vectors (the door's PACK: supports x 0).
    var width = 0 if precomputed else features
    slot[].oc_vectors = List[Float32](length=supports * width, fill=Float32(0.0))
    if not precomputed:
        oc_take_rows_into(driver, call[].x.data, rows, features, Int(slot[].oc_support.unsafe_ptr()), supports,
            Int(slot[].oc_vectors.unsafe_ptr()))
    slot[].oc_gamma = run.gamma
    slot[].oc_rho = run.rho
    slot[].oc_iterations = Int(run.state.iterations[0])
    _ = run^
    slot[].rows = rows
    slot[].features = features
    slot[].fitted = True
    if labels:
        # fit_predict: the door's fit, then its predict on the same rows.
        _oc_query(slot, call, LOF_RESULT_LABELS, "predict")


def _oc_query(slot: _SlotPtr, call: _CallPtr, result: Int, what: String) raises:
    """score_samples / decision_function / predict, the request as the door's
    `oc_query_unpack` builds it (labels: predict; offset: decision and
    predict) and its RUN (`oc_query_run`, `oc_decision_run`,
    `oc_labels_run`). `result` is the LOF result word of the same meaning."""
    var estimator = _name(slot)
    if not slot[].fitted:
        raise Error("NotFittedError: mojolearn: call fit before " + what)
    _whole(call, estimator)
    _dense(call[].x, MEMBER_F32, "X", estimator)
    if call[].x.cols != slot[].features:
        raise Error("ValueError: X has " + String(call[].x.cols) + " features, but " + estimator
            + " is expecting " + String(slot[].features) + " features as input.")
    var rows = call[].x.rows
    var labels = result == LOF_RESULT_LABELS
    var offset = result != LOF_RESULT_SCORES
    _dense(call[].dst, MEMBER_I64 if labels else MEMBER_F32, "the result buffer", estimator)
    if call[].dst.rows != rows or call[].dst.cols != 1:
        raise Error("ValueError: the result buffer holds one value per row of X")
    var supports = len(slot[].oc_dual)
    var request = OcQueryRequest()
    request.rows = rows
    request.supports = supports
    request.kernel = slot[].oc_kernel
    request.degree = slot[].oc_degree
    request.gamma = slot[].oc_gamma
    request.coef0 = slot[].oc_coef0
    request.features = call[].x.cols
    request.query = call[].x.data
    request.references = Int(slot[].oc_vectors.unsafe_ptr())
    request.coefficients = Int(slot[].oc_dual.unsafe_ptr())
    if slot[].oc_precomputed:
        # `oc_query_unpack`'s precomputed path: the support columns of X.
        request.path = OC_QUERY_GRAM_PRECOMPUTED
        request.support = Int(slot[].oc_support.unsafe_ptr())
        request.support_count = len(slot[].oc_support)
    elif supports > 0 and rows > 0 and not slot[].oc_old_items:
        request.path = OC_QUERY_KERNEL_PRODUCT
        if labels and slot[].oc_labels:
            request.offset = slot[].oc_rho
            request.path = OC_QUERY_FUSED_LABELS
    else:
        request.path = OC_QUERY_GRAM_KERNEL
        request.reference_rows = supports
    if offset and request.path != OC_QUERY_FUSED_LABELS:
        request.offset = slot[].oc_rho
    var driver = XnDriver(call[].route.entry.call)
    if labels:
        oc_labels_run(driver, request, call[].dst.data)
    elif offset:
        oc_decision_run(driver, request, call[].dst.data)
    else:
        oc_query_run(driver, request, call[].dst.data)


def _oc_export(slot: _SlotPtr, call: _CallPtr) raises:
    """A view of one fitted block; the slot owns the words until this
    state's next fit, import or free."""
    if not slot[].fitted:
        raise Error("NotFittedError: this OneClassSVM member is not fitted yet")
    var block = call[].block
    if (call[].flags & MEMBER_BLOCK_BY_INDEX) != 0:
        if block < 0 or block >= _OC_BLOCKS:
            raise Error("member ABI: a OneClassSVM member has four fitted blocks")
        block = OC_BLOCK_VECTORS + block
    var supports = len(slot[].oc_dual)
    if block == OC_BLOCK_VECTORS:
        call[].dst = MemberMatrix(Int(slot[].oc_vectors.unsafe_ptr()), supports,
            0 if slot[].oc_precomputed else slot[].features)
    elif block == OC_BLOCK_DUAL:
        call[].dst = MemberMatrix(Int(slot[].oc_dual.unsafe_ptr()), 1, supports)
    elif block == OC_BLOCK_SUPPORT:
        call[].dst = MemberMatrix(Int(slot[].oc_support.unsafe_ptr()), 1, supports, MEMBER_I32)
    elif block == OC_BLOCK_SCALARS:
        slot[].oc_scalars[0] = slot[].oc_gamma
        slot[].oc_scalars[1] = slot[].oc_rho
        slot[].oc_scalars[2] = Float64(slot[].oc_iterations)
        slot[].oc_scalars[3] = Float64(slot[].features)
        call[].dst = MemberMatrix(Int(slot[].oc_scalars.unsafe_ptr()), 1, 4, MEMBER_F64)
    else:
        raise Error("member ABI: a OneClassSVM member has no fitted block " + String(block))
    call[].block = block
    call[].out_cols = _OC_BLOCKS


def _oc_import(slot: _SlotPtr, call: _CallPtr) raises:
    """Copy one fitted block in; MEMBER_IMPORT_DONE completes the state."""
    var estimator = _name(slot)
    var block = call[].block
    var count = call[].x.rows * call[].x.cols
    if block == OC_BLOCK_SCALARS:
        _dense(call[].x, MEMBER_F64, "the scalars (gamma, offset, n_iter, features)", estimator)
        if count != 4:
            raise Error("member ABI: a OneClassSVM member's scalars are four float64 words")
        slot[].fitted = False
        var words = _F64P(unsafe_from_address=call[].x.data)
        slot[].oc_gamma = words[0]
        slot[].oc_rho = words[1]
        slot[].oc_iterations = Int(words[2])
        slot[].features = Int(words[3])
    elif block == OC_BLOCK_SUPPORT:
        _dense(call[].x, MEMBER_I32, "the support indices", estimator)
        slot[].fitted = False
        slot[].oc_support = List[Int32](length=count, fill=Int32(0))
        if count > 0:
            unsafe_memcpy(dest=slot[].oc_support.unsafe_ptr(), src=_I32P(unsafe_from_address=call[].x.data),
                count=count)
    elif block == OC_BLOCK_VECTORS or block == OC_BLOCK_DUAL:
        _dense(call[].x, MEMBER_F32, "a fitted block", estimator)
        slot[].fitted = False
        var words = List[Float32](length=count, fill=Float32(0.0))
        if count > 0:
            unsafe_memcpy(dest=words.unsafe_ptr(), src=_F32P(unsafe_from_address=call[].x.data), count=count)
        if block == OC_BLOCK_VECTORS:
            slot[].oc_vectors = words^
        else:
            slot[].oc_dual = words^
    elif block != 0 or (call[].flags & MEMBER_IMPORT_DONE) == 0:
        raise Error("member ABI: a OneClassSVM member has no fitted block " + String(block))
    if (call[].flags & MEMBER_IMPORT_DONE) == 0:
        return
    var supports = len(slot[].oc_dual)
    var width = 0 if slot[].oc_precomputed else slot[].features
    if (slot[].features < 1 or len(slot[].oc_support) != supports
            or len(slot[].oc_vectors) != supports * width):
        raise Error("member ABI: a fitted OneClassSVM state needs its support vectors, dual coefficients, "
            + "support indices and scalars")
    slot[].fitted = True


# ------------------------------------------------------------ KernelPCA


def _f32_scalar(value: Float64) -> Float64:
    # x_decomp/kpca_run.mojo `_f32_scalar`: the float32 rounding the door applies.
    return Float64(Float32(value))


def _kp_neighbor(call: _CallPtr) -> KPCANeighborDriver:
    return KPCANeighborDriver(XnDriver(call[].route.entry.call))


def _kp_fit(slot: _SlotPtr, call: _CallPtr) raises:
    """`KernelPCA.fit` (bindings/kpca_native_door.mojo `kpca_fit`): the
    dense prepare (unless the resident route), the eigensolve, the kept
    eigenpairs, in the door's order. fit_transform: the door's unscaled
    alpha step into `dst`."""
    var estimator = _name(slot)
    _whole(call, estimator)
    if call[].weights.rows > 0:
        raise Error("TypeError: a " + estimator + " member fits no sample_weight")
    var transformed = call[].fit_output == MEMBER_OP_TRANSFORM
    if call[].fit_output != 0 and not transformed:
        member_unsupported(estimator, call[].fit_output)
    _dense(call[].x, MEMBER_F32, "X", estimator)
    var rows = call[].x.rows
    var features = call[].x.cols
    if slot[].kp_kernel == KPCA_PRECOMPUTED and rows != features:
        # `kpca_fit`'s words, after X's admission and before its route (lane mf-nbr6).
        raise Error("Precomputed matrix must be a square matrix.")
    if not call[].route.has_second or not call[].route.has_third:
        raise Error("member ABI: a KernelPCA member needs the route's eigensolver entries (second, third)")
    var gamma = _f32_scalar(1.0 / Float64(features) if not slot[].kp_gamma_given else slot[].kp_gamma_value)
    var near = _kp_neighbor(call)
    var decomp = KPCANativeDriver(call[].route.second.call)
    var reduction = call[].route.third.call
    var route = KPCAFitRoute(rows, slot[].kp_requested, slot[].kp_kernel, slot[].kp_auto, slot[].kp_resident)
    slot[].fitted = False
    slot[].kp_alphas_ready = False
    # The fitted column means and total (the door's `_fit_cols`, `_fit_all`).
    slot[].kp_columns = List[Float32](length=max(rows, 1), fill=Float32(0.0))
    slot[].kp_total = List[Float32](length=1, fill=Float32(0.0))
    var prepared: Optional[KPCADenseFit] = None
    var centered = Int(slot[].kp_columns.unsafe_ptr())  # the door's unread resident-route dummy
    if not route.resident:
        var options = KPCAKernelOptions(slot[].kp_kernel, slot[].kp_degree, gamma, _f32_scalar(slot[].kp_coef0))
        var prepare = KPCAFitPrepareRequest(call[].x.data, Int(slot[].kp_columns.unsafe_ptr()),
            Int(slot[].kp_total.unsafe_ptr()), rows, features, options^)
        prepared = Optional[KPCADenseFit](KPCADenseFit(near, reduction, prepare))
        centered = prepared.value().address()
    var room = KPCAEigenRoom(rows, route.room)
    var request = KPCAEigenRequest(centered, call[].x.data, room.values_address(), room.vectors_address(),
        Int(slot[].kp_columns.unsafe_ptr()), Int(slot[].kp_total.unsafe_ptr()), route.resident, rows, features,
        slot[].kp_requested, route.lanczos, slot[].kp_remove_zero, gamma)
    var kept = decomp.eigen(request)
    _ = prepared^
    slot[].kp_values = List[Float32](length=kept, fill=Float32(0.0))
    slot[].kp_vectors = List[Float32](length=rows * kept, fill=Float32(0.0))
    if kept:
        unsafe_memcpy(dest=slot[].kp_values.unsafe_ptr(), src=room.values.unsafe_ptr(), count=kept)
        unsafe_memcpy(dest=slot[].kp_vectors.unsafe_ptr(), src=room.vectors.unsafe_ptr(), count=rows * kept)
    _ = room^
    # Borrow the rows as the fitted X (LIFETIME); a copy a previous import or export left is dropped.
    slot[].fit_rows = List[Float32]()
    slot[].rows_addr = call[].x.data
    slot[].rows = rows
    slot[].features = features
    slot[].kp_gamma = gamma
    slot[].kp_kept = kept
    slot[].fitted = True
    if transformed:
        _dense(call[].dst, MEMBER_F32, "the transform buffer", estimator)
        if call[].dst.rows != rows or call[].dst.cols != kept:
            raise Error("ValueError: the transform buffer holds " + String(kept) + " components per row of X")
        near.alpha(KPCAAlphaRequest(Int(slot[].kp_vectors.unsafe_ptr()), Int(slot[].kp_values.unsafe_ptr()),
            call[].dst.data, rows, kept, False))


def _kp_transform(slot: _SlotPtr, call: _CallPtr) raises:
    """`KernelPCA.transform` (`kpca_transform`): the scaled alphas (computed
    once per fitted state here; the door recomputes them per call), then the
    fused query, or the centered query and its finish (the old-items switch,
    an empty query or no components)."""
    var estimator = _name(slot)
    if not slot[].fitted:
        raise Error("NotFittedError: mojolearn: call fit before transform")
    _whole(call, estimator)
    _dense(call[].x, MEMBER_F32, "X", estimator)
    if slot[].kp_kernel == KPCA_PRECOMPUTED:
        # `kpca_transform`: a precomputed query holds one column per fitted row.
        if call[].x.cols != slot[].rows:
            raise Error("KernelPCA: a precomputed kernel needs " + String(slot[].rows)
                + " columns (the fitted rows), got " + String(call[].x.cols))
    elif call[].x.cols != slot[].features:
        raise Error("KernelPCA: query and fitted input feature counts differ")
    var rows = call[].x.rows
    var count = slot[].kp_kept
    _dense(call[].dst, MEMBER_F32, "the transform buffer", estimator)
    if call[].dst.rows != rows or call[].dst.cols != count:
        raise Error("ValueError: the transform buffer holds " + String(count) + " components per row of X")
    var near = _kp_neighbor(call)
    if not slot[].kp_alphas_ready:
        slot[].kp_alphas = List[Float32](length=max(slot[].rows * count, 1), fill=Float32(0.0))
        near.alpha(KPCAAlphaRequest(Int(slot[].kp_vectors.unsafe_ptr()), Int(slot[].kp_values.unsafe_ptr()),
            Int(slot[].kp_alphas.unsafe_ptr()), slot[].rows, count, True))
        slot[].kp_alphas_ready = True
    var options = KPCAKernelOptions(slot[].kp_kernel, slot[].kp_degree, slot[].kp_gamma, slot[].kp_coef0)
    var alphas = Int(slot[].kp_alphas.unsafe_ptr())
    if rows != 0 and count != 0 and not slot[].kp_old_items:
        near.fused_query(KPCAQueryRequest(call[].x.data, slot[].rows_addr, Int(slot[].kp_columns.unsafe_ptr()),
            Int(slot[].kp_total.unsafe_ptr()), alphas, call[].dst.data, rows, slot[].rows, call[].x.cols, count,
            options^))
    else:
        var request = KPCAQueryRequest(call[].x.data, slot[].rows_addr, Int(slot[].kp_columns.unsafe_ptr()),
            Int(slot[].kp_total.unsafe_ptr()), alphas, 0, rows, slot[].rows, call[].x.cols, count, options^)
        var centered = KPCACenteredQuery(near, request)
        centered.finish(near, alphas, call[].dst.data, count)
        _ = centered^


def _kp_export(slot: _SlotPtr, call: _CallPtr) raises:
    """A view of one fitted block; the slot owns the words until this
    state's next fit, import or free (the borrowed rows are copied here)."""
    if not slot[].fitted:
        raise Error("NotFittedError: this KernelPCA member is not fitted yet")
    if (call[].flags & MEMBER_EXPORT_OWN) != 0:
        # THE LIFETIME RULE: the borrowed fit rows, once; no view is answered.
        _lof_own_rows(slot)
        return
    var block = call[].block
    if (call[].flags & MEMBER_BLOCK_BY_INDEX) != 0:
        if block < 0 or block >= _KP_BLOCKS:
            raise Error("member ABI: a KernelPCA member has six fitted blocks")
        block = KP_BLOCK_ROWS + block
    if block == KP_BLOCK_ROWS:
        _lof_own_rows(slot)
        call[].dst = MemberMatrix(slot[].rows_addr, slot[].rows, slot[].features)
    elif block == KP_BLOCK_VALUES:
        call[].dst = MemberMatrix(Int(slot[].kp_values.unsafe_ptr()), 1, slot[].kp_kept)
    elif block == KP_BLOCK_VECTORS:
        call[].dst = MemberMatrix(Int(slot[].kp_vectors.unsafe_ptr()), slot[].rows, slot[].kp_kept)
    elif block == KP_BLOCK_COLUMNS:
        call[].dst = MemberMatrix(Int(slot[].kp_columns.unsafe_ptr()), 1, slot[].rows)
    elif block == KP_BLOCK_TOTAL:
        call[].dst = MemberMatrix(Int(slot[].kp_total.unsafe_ptr()), 1, 1)
    elif block == KP_BLOCK_GAMMA:
        slot[].offset_word[0] = slot[].kp_gamma
        call[].dst = MemberMatrix(Int(slot[].offset_word.unsafe_ptr()), 1, 1, MEMBER_F64)
    else:
        raise Error("member ABI: a KernelPCA member has no fitted block " + String(block))
    call[].block = block
    call[].out_cols = _KP_BLOCKS


def _kp_import(slot: _SlotPtr, call: _CallPtr) raises:
    """Copy one fitted block in; MEMBER_IMPORT_DONE completes the state."""
    var estimator = _name(slot)
    var block = call[].block
    var count = call[].x.rows * call[].x.cols
    if block == KP_BLOCK_GAMMA:
        _dense(call[].x, MEMBER_F64, "gamma", estimator)
        if count != 1:
            raise Error("member ABI: a KernelPCA gamma is one float64 word")
        slot[].fitted = False
        slot[].kp_gamma = _F64P(unsafe_from_address=call[].x.data)[0]
    elif block >= KP_BLOCK_ROWS and block <= KP_BLOCK_TOTAL:
        _dense(call[].x, MEMBER_F32, "a fitted block", estimator)
        slot[].fitted = False
        slot[].kp_alphas_ready = False
        var words = List[Float32](length=count, fill=Float32(0.0))
        if count > 0:
            unsafe_memcpy(dest=words.unsafe_ptr(), src=_F32P(unsafe_from_address=call[].x.data), count=count)
        if block == KP_BLOCK_ROWS:
            slot[].fit_rows = words^
            slot[].rows_addr = Int(slot[].fit_rows.unsafe_ptr())
            slot[].rows = call[].x.rows
            slot[].features = call[].x.cols
        elif block == KP_BLOCK_VALUES:
            slot[].kp_values = words^
        elif block == KP_BLOCK_VECTORS:
            slot[].kp_vectors = words^
            slot[].kp_kept = call[].x.cols
        elif block == KP_BLOCK_COLUMNS:
            slot[].kp_columns = words^
        else:
            slot[].kp_total = words^
    elif block != 0 or (call[].flags & MEMBER_IMPORT_DONE) == 0:
        raise Error("member ABI: a KernelPCA member has no fitted block " + String(block))
    if (call[].flags & MEMBER_IMPORT_DONE) == 0:
        return
    var rows = slot[].rows
    if (rows == 0 or len(slot[].fit_rows) != rows * slot[].features or len(slot[].kp_values) < slot[].kp_kept
            or len(slot[].kp_vectors) != rows * slot[].kp_kept or len(slot[].kp_columns) < rows
            or len(slot[].kp_total) < 1):
        raise Error("KernelPCA: fitted query buffers do not cover the fitted row count")
    slot[].fitted = True


# ------------------------------------------------------------ PolynomialCountSketch


def _pc_plan(slot: _SlotPtr) raises -> PCSFitPlan:
    return PCSFitPlan(pcs_options(slot[].pc_degree, slot[].pc_components), slot[].features, slot[].pc_coef0)


def _pc_fit(slot: _SlotPtr, call: _CallPtr) raises:
    """`PolynomialCountSketch.fit` (bindings/neighbors_native_unpack.mojo
    `_pcs_fit`): the two hash tables drawn from the seed word (zeros when
    there is no augmented column), as the door draws them. The fit reads only
    the width of X. fit_transform: then the transform of the same rows."""
    var estimator = _name(slot)
    _whole(call, estimator)
    if call[].weights.rows > 0:
        raise Error("TypeError: a " + estimator + " member fits no sample_weight")
    var transformed = call[].fit_output == MEMBER_OP_TRANSFORM
    if call[].fit_output != 0 and not transformed:
        member_unsupported(estimator, call[].fit_output)
    slot[].fitted = False
    slot[].features = call[].x.cols
    var plan = _pc_plan(slot)
    var count = plan.options.degree * plan.augmented
    slot[].pc_indices = List[Int32](length=max(count, 1), fill=Int32(0))
    slot[].pc_signs = List[Int32](length=max(count, 1), fill=Int32(0))
    if plan.augmented > 0:
        var driver = PCSNativeDriver(XnDriver(call[].route.entry.call))
        driver.require_draw()
        driver.draw(plan.draw(slot[].pc_seed, Int(slot[].pc_indices.unsafe_ptr()), Int(slot[].pc_signs.unsafe_ptr())))
    slot[].pc_augmented = plan.augmented
    slot[].rows = call[].x.rows
    slot[].fitted = True
    if transformed:
        _pc_transform(slot, call)


def _pc_transform(slot: _SlotPtr, call: _CallPtr) raises:
    """`PolynomialCountSketch.transform` (bindings/neighbors_native_pack.mojo
    `_pcs_transform`): the query over the state's own hash tables."""
    var estimator = _name(slot)
    if not slot[].fitted:
        raise Error("NotFittedError: mojolearn: call fit before transform")
    _whole(call, estimator)
    _dense(call[].x, MEMBER_F32, "X", estimator)
    pcs_feature_width(call[].x.cols, slot[].features)
    var rows = call[].x.rows
    _dense(call[].dst, MEMBER_F32, "the transform buffer", estimator)
    if call[].dst.rows != rows or call[].dst.cols != slot[].pc_components:
        raise Error("ValueError: the transform buffer holds " + String(slot[].pc_components)
            + " components per row of X")
    var plan = _pc_plan(slot)
    var driver = PCSNativeDriver(XnDriver(call[].route.entry.call))
    driver.query(plan.query(call[].x.data, Int(slot[].pc_indices.unsafe_ptr()), Int(slot[].pc_signs.unsafe_ptr()),
        call[].dst.data, rows, slot[].pc_gamma, slot[].pc_coef0, slot[].pc_old_items))


def _pc_export(slot: _SlotPtr, call: _CallPtr) raises:
    """A view of one fitted block; the slot owns the words until this
    state's next fit, import or free."""
    if not slot[].fitted:
        raise Error("NotFittedError: this PolynomialCountSketch member is not fitted yet")
    var block = call[].block
    if (call[].flags & MEMBER_BLOCK_BY_INDEX) != 0:
        if block < 0 or block >= _PC_BLOCKS:
            raise Error("member ABI: a PolynomialCountSketch member has three fitted blocks")
        block = PC_BLOCK_INDEX_HASH + block
    if block == PC_BLOCK_INDEX_HASH:
        call[].dst = MemberMatrix(Int(slot[].pc_indices.unsafe_ptr()), slot[].pc_degree, slot[].pc_augmented,
            MEMBER_I32)
    elif block == PC_BLOCK_BIT_HASH:
        call[].dst = MemberMatrix(Int(slot[].pc_signs.unsafe_ptr()), slot[].pc_degree, slot[].pc_augmented,
            MEMBER_I32)
    elif block == PC_BLOCK_FEATURES:
        slot[].offset_word[0] = Float64(slot[].features)
        call[].dst = MemberMatrix(Int(slot[].offset_word.unsafe_ptr()), 1, 1, MEMBER_F64)
    else:
        raise Error("member ABI: a PolynomialCountSketch member has no fitted block " + String(block))
    call[].block = block
    call[].out_cols = _PC_BLOCKS


def _pc_import(slot: _SlotPtr, call: _CallPtr) raises:
    """Copy one fitted block in; MEMBER_IMPORT_DONE completes the state."""
    var estimator = _name(slot)
    var block = call[].block
    var count = call[].x.rows * call[].x.cols
    if block == PC_BLOCK_FEATURES:
        _dense(call[].x, MEMBER_F64, "n_features_in_", estimator)
        if count != 1:
            raise Error("member ABI: a PolynomialCountSketch feature count is one float64 word")
        slot[].fitted = False
        slot[].features = Int(_F64P(unsafe_from_address=call[].x.data)[0])
    elif block == PC_BLOCK_INDEX_HASH or block == PC_BLOCK_BIT_HASH:
        _dense(call[].x, MEMBER_I32, "a hash table", estimator)
        if call[].x.rows != slot[].pc_degree:
            raise Error("ValueError: PolynomialCountSketch hash tables must match degree")
        slot[].fitted = False
        var words = List[Int32](length=max(count, 1), fill=Int32(0))
        if count > 0:
            unsafe_memcpy(dest=words.unsafe_ptr(), src=_I32P(unsafe_from_address=call[].x.data), count=count)
        if block == PC_BLOCK_INDEX_HASH:
            slot[].pc_indices = words^
        else:
            slot[].pc_signs = words^
        slot[].pc_augmented = call[].x.cols
    elif block != 0 or (call[].flags & MEMBER_IMPORT_DONE) == 0:
        raise Error("member ABI: a PolynomialCountSketch member has no fitted block " + String(block))
    if (call[].flags & MEMBER_IMPORT_DONE) == 0:
        return
    var plan = _pc_plan(slot)
    var count_words = plan.options.degree * plan.augmented
    plan.admit_hash_widths(slot[].pc_augmented, slot[].pc_augmented)
    if len(slot[].pc_indices) < count_words or len(slot[].pc_signs) < count_words:
        raise Error("ValueError: PolynomialCountSketch hash tables do not match the fitted features")
    slot[].fitted = True


# ------------------------------------------------------------ AdditiveChi2Sampler


def _ac_fit(slot: _SlotPtr, call: _CallPtr) raises:
    """`AdditiveChi2Sampler.fit` (bindings/neighbors_semi_native_door.mojo
    `_achi2_fit`): the negative-entry refusal (the device scan when the image
    has it), the interval check, the width. fit_transform: then the map of the
    same rows without the transform's second scan (the door's `checked`)."""
    var estimator = _name(slot)
    _whole(call, estimator)
    if call[].weights.rows > 0:
        raise Error("TypeError: a " + estimator + " member fits no sample_weight")
    var transformed = call[].fit_output == MEMBER_OP_TRANSFORM
    if call[].fit_output != 0 and not transformed:
        member_unsupported(estimator, call[].fit_output)
    _dense(call[].x, MEMBER_F32, "X", estimator)
    var count = call[].x.rows * call[].x.cols
    slot[].fitted = False
    achi2_fit_run(XnDriver(call[].route.entry.call),
        Achi2FitRequest(call[].x.data, count, count != 0 and slot[].ac_scan))
    achi2_interval_check(slot[].ac_valid)
    slot[].features = call[].x.cols
    slot[].rows = call[].x.rows
    slot[].fitted = True
    if transformed:
        _ac_transform(slot, call, True)


def _ac_transform(slot: _SlotPtr, call: _CallPtr, checked: Bool) raises:
    """`_achi2_transform` of a dense X: rows x features * (2 steps - 1) float32."""
    var estimator = _name(slot)
    if not slot[].fitted:
        raise Error("NotFittedError: mojolearn: call fit before transform")
    _whole(call, estimator)
    _dense(call[].x, MEMBER_F32, "X", estimator)
    var rows = call[].x.rows
    var features = call[].x.cols
    var width = xn_achi2_width(features, slot[].ac_steps)
    _dense(call[].dst, MEMBER_F32, "the transform buffer", estimator)
    if call[].dst.rows != rows or call[].dst.cols != width:
        raise Error("ValueError: the transform buffer holds " + String(width) + " columns per row of X")
    _ = achi2_transform_run(XnDriver(call[].route.entry.call), Achi2Request(call[].x.data, call[].dst.data, rows,
        features, slot[].ac_steps, slot[].ac_interval, slot[].ac_valid, checked, False))


def _ac_export(slot: _SlotPtr, call: _CallPtr) raises:
    if not slot[].fitted:
        raise Error("NotFittedError: this AdditiveChi2Sampler member is not fitted yet")
    var block = call[].block
    if (call[].flags & MEMBER_BLOCK_BY_INDEX) != 0:
        if block != 0:
            raise Error("member ABI: an AdditiveChi2Sampler member has one fitted block")
        block = AC_BLOCK_FEATURES
    if block != AC_BLOCK_FEATURES:
        raise Error("member ABI: an AdditiveChi2Sampler member has no fitted block " + String(block))
    slot[].offset_word[0] = Float64(slot[].features)
    call[].dst = MemberMatrix(Int(slot[].offset_word.unsafe_ptr()), 1, 1, MEMBER_F64)
    call[].block = block
    call[].out_cols = 1


def _ac_import(slot: _SlotPtr, call: _CallPtr) raises:
    var estimator = _name(slot)
    if call[].block == AC_BLOCK_FEATURES:
        _dense(call[].x, MEMBER_F64, "n_features_in_", estimator)
        if call[].x.rows * call[].x.cols != 1:
            raise Error("member ABI: an AdditiveChi2Sampler feature count is one float64 word")
        slot[].fitted = False
        slot[].features = Int(_F64P(unsafe_from_address=call[].x.data)[0])
    elif call[].block != 0 or (call[].flags & MEMBER_IMPORT_DONE) == 0:
        raise Error("member ABI: an AdditiveChi2Sampler member has no fitted block " + String(call[].block))
    if (call[].flags & MEMBER_IMPORT_DONE) == 0:
        return
    achi2_interval_check(slot[].ac_valid)
    slot[].fitted = True


# ------------------------------------------------------------ SkewedChi2Sampler


def _sc_fit(slot: _SlotPtr, call: _CallPtr) raises:
    """`SkewedChi2Sampler.fit` (bindings/neighbors_semi_native_door.mojo
    `_schi2_fit`): the weights (features x n_components) and offsets
    (n_components) from the counter-based draw of the seed word, eager (the
    `identical` tier's form). The fit reads only the width of X.
    fit_transform: then the transform of the same rows (the door's order)."""
    var estimator = _name(slot)
    _whole(call, estimator)
    if call[].weights.rows > 0:
        raise Error("TypeError: a " + estimator + " member fits no sample_weight")
    var transformed = call[].fit_output == MEMBER_OP_TRANSFORM
    if call[].fit_output != 0 and not transformed:
        member_unsupported(estimator, call[].fit_output)
    slot[].fitted = False
    var features = call[].x.cols
    var components = slot[].sc_components
    if features < 1 or components < 1:
        xn_schi2_missing_fit(features, components)
    slot[].sc_weights = List[Float32](length=features * components, fill=Float32(0.0))
    slot[].sc_offset = List[Float32](length=components, fill=Float32(0.0))
    schi2_fit_run(XnDriver(call[].route.entry.call), Schi2FitRequest(slot[].sc_seed, features, components,
        Int(slot[].sc_weights.unsafe_ptr()), Int(slot[].sc_offset.unsafe_ptr())))
    slot[].features = features
    slot[].rows = call[].x.rows
    slot[].fitted = True
    if transformed:
        _sc_transform(slot, call)


def _sc_transform(slot: _SlotPtr, call: _CallPtr) raises:
    """`_schi2_transform` with eager weights: the refusal of an entry at or
    below -skewedness (non-empty X), log(x + skewedness) into the slot's
    scratch, the map into `dst` (rows x n_components float32)."""
    var estimator = _name(slot)
    if not slot[].fitted:
        raise Error("NotFittedError: mojolearn: call fit before transform")
    _whole(call, estimator)
    _dense(call[].x, MEMBER_F32, "X", estimator)
    var rows = call[].x.rows
    var features = call[].x.cols
    var components = slot[].sc_components
    if features != slot[].features:
        raise Error("ValueError: X has " + String(features) + " features, but SkewedChi2Sampler is expecting "
            + String(slot[].features) + " features as input.")
    _dense(call[].dst, MEMBER_F32, "the transform buffer", estimator)
    if call[].dst.rows != rows or call[].dst.cols != components:
        raise Error("ValueError: the transform buffer holds " + String(components) + " columns per row of X")
    var count = rows * features
    schi2_check_run(call[].x.data, count, slot[].sc_skew)
    if slot[].sc_lazy:
        # The image's one-call transform over the stored weights (kfeat bit 4,
        # SCHI2_LAZYW), as the door's `_schi2_transform` takes it (lane mf-nbr6).
        schi2_transform_run(XnDriver(call[].route.entry.call), Schi2Request(call[].x.data,
            Int(slot[].sc_weights.unsafe_ptr()), 0, Int(slot[].sc_offset.unsafe_ptr()), 0, call[].dst.data, rows,
            features, components, slot[].sc_skew, True, False))
        return
    if len(slot[].sc_scratch) < max(count, 1):
        slot[].sc_scratch.resize(max(count, 1), Float32(0.0))
    schi2_transform_run(XnDriver(call[].route.entry.call), Schi2Request(call[].x.data,
        Int(slot[].sc_weights.unsafe_ptr()), 0, Int(slot[].sc_offset.unsafe_ptr()), Int(slot[].sc_scratch.unsafe_ptr()),
        call[].dst.data, rows, features, components, slot[].sc_skew, False, False))


def _sc_export(slot: _SlotPtr, call: _CallPtr) raises:
    """A view of one fitted block; the slot owns the words until this
    state's next fit, import or free."""
    if not slot[].fitted:
        raise Error("NotFittedError: this SkewedChi2Sampler member is not fitted yet")
    var block = call[].block
    if (call[].flags & MEMBER_BLOCK_BY_INDEX) != 0:
        if block < 0 or block >= _SC_BLOCKS:
            raise Error("member ABI: a SkewedChi2Sampler member has two fitted blocks")
        block = SC_BLOCK_WEIGHTS + block
    if block == SC_BLOCK_WEIGHTS:
        call[].dst = MemberMatrix(Int(slot[].sc_weights.unsafe_ptr()), slot[].features, slot[].sc_components,
            MEMBER_F32)
    elif block == SC_BLOCK_OFFSET:
        call[].dst = MemberMatrix(Int(slot[].sc_offset.unsafe_ptr()), 1, slot[].sc_components, MEMBER_F32)
    else:
        raise Error("member ABI: a SkewedChi2Sampler member has no fitted block " + String(block))
    call[].block = block
    call[].out_cols = _SC_BLOCKS


def _sc_import(slot: _SlotPtr, call: _CallPtr) raises:
    """Copy one fitted block in; MEMBER_IMPORT_DONE completes the state (both
    blocks, n_components wide)."""
    var estimator = _name(slot)
    var block = call[].block
    var count = call[].x.rows * call[].x.cols
    if block == SC_BLOCK_WEIGHTS or block == SC_BLOCK_OFFSET:
        _dense(call[].x, MEMBER_F32, String("random_weights_") if block == SC_BLOCK_WEIGHTS
            else String("random_offset_"), estimator)
        if call[].x.cols != slot[].sc_components or (block == SC_BLOCK_OFFSET and call[].x.rows != 1):
            raise Error("member ABI: SkewedChi2Sampler fitted arrays must be n_components = "
                + String(slot[].sc_components) + " wide")
        slot[].fitted = False
        var words = List[Float32](length=max(count, 1), fill=Float32(0.0))
        if count > 0:
            unsafe_memcpy(dest=words.unsafe_ptr(), src=_F32P(unsafe_from_address=call[].x.data), count=count)
        if block == SC_BLOCK_WEIGHTS:
            slot[].sc_weights = words^
            slot[].features = call[].x.rows
        else:
            slot[].sc_offset = words^
    elif block != 0 or (call[].flags & MEMBER_IMPORT_DONE) == 0:
        raise Error("member ABI: a SkewedChi2Sampler member has no fitted block " + String(block))
    if (call[].flags & MEMBER_IMPORT_DONE) == 0:
        return
    if (slot[].features < 1 or len(slot[].sc_weights) < slot[].features * slot[].sc_components
            or len(slot[].sc_offset) < slot[].sc_components):
        raise Error("member ABI: a SkewedChi2Sampler import needs random_weights_ and random_offset_")
    slot[].fitted = True


# ------------------------------------------------------------ KNNImputer


def _im_columns(slot: _SlotPtr, valid: List[Bool], missing: List[Int]):
    """The fitted column words and the kept-column plan, once per fitted state."""
    slot[].im_valid = List[Int32](length=max(len(valid), 1), fill=Int32(0))
    for f in range(len(valid)):  # driver-loop: over the d feature columns (the fit's column masks)
        slot[].im_valid[f] = Int32(1) if valid[f] else Int32(0)
    slot[].im_miss = missing.copy()
    slot[].im_miss32 = List[Int32](length=max(len(missing), 1), fill=Int32(0))
    for i in range(len(missing)):  # driver-loop: over the columns missing at fit
        slot[].im_miss32[i] = Int32(missing[i])
    slot[].im_plan = imp_keep_plan(valid)


def _im_width(slot: _SlotPtr, rows: Int) -> Int:
    """The transform's columns: the kept ones, then one indicator column per
    feature missing at fit (none without add_indicator or without rows)."""
    var indicator = len(slot[].im_miss) if (slot[].im_indicator and rows != 0) else 0
    return imp_result_columns(slot[].features, slot[].im_keep_empty, slot[].im_plan, indicator)


def _im_fit(slot: _SlotPtr, call: _CallPtr) raises:
    """`KNNImputer.fit` (bindings/neighbors_local_native_door.mojo
    `_imp_fit`): the missing number's cells masked into the slot's rows (or
    X borrowed as the fitted rows), then the column masks. fit_transform:
    then the transform of the same rows (the door's order)."""
    var estimator = _name(slot)
    _whole(call, estimator)
    if call[].weights.rows > 0:
        raise Error("TypeError: a " + estimator + " member fits no sample_weight")
    var transformed = call[].fit_output == MEMBER_OP_TRANSFORM
    if call[].fit_output != 0 and not transformed:
        member_unsupported(estimator, call[].fit_output)
    _dense(call[].x, MEMBER_F32, "X", estimator)
    var rows = call[].x.rows
    var features = call[].x.cols
    var count = rows * features
    var mask = slot[].im_mask and count != 0
    slot[].fitted = False
    slot[].fit_rows = List[Float32]()
    var masked = 0
    if mask:
        slot[].fit_rows = List[Float32](length=count, fill=Float32(0.0))
        masked = Int(slot[].fit_rows.unsafe_ptr())
    var columns = imp_fit_run(XnDriver(call[].route.entry.call),
        ImpFitRequest(call[].x.data, masked, rows, features, slot[].im_missing, slot[].im_lean))
    slot[].rows_addr = masked if mask else call[].x.data
    slot[].rows = rows
    slot[].features = features
    _im_columns(slot, columns.valid, columns.missing)
    slot[].fitted = True
    if transformed:
        _im_transform(slot, call)


def _im_transform(slot: _SlotPtr, call: _CallPtr) raises:
    """`_imp_transform`: mask, impute from the fitted rows, select the kept
    columns, append the indicator block."""
    var estimator = _name(slot)
    if not slot[].fitted:
        raise Error("NotFittedError: mojolearn: call fit before transform")
    _whole(call, estimator)
    _dense(call[].x, MEMBER_F32, "X", estimator)
    var rows = call[].x.rows
    var features = call[].x.cols
    imp_check_features(features, slot[].features)
    # `_imp_transform`'s n_neighbors refusal, after the feature check (its fit took the value).
    local_check_n_neighbors(slot[].im_k)
    var width = _im_width(slot, rows)
    _dense(call[].dst, MEMBER_F32, "the transform buffer", estimator)
    if call[].dst.rows != rows or call[].dst.cols != width:
        raise Error("ValueError: the transform buffer holds " + String(width) + " columns per row of X")
    var request = ImpTransformRequest(call[].x.data, slot[].rows_addr, call[].dst.data, rows, slot[].rows, features,
        slot[].im_k, slot[].im_distance, slot[].im_mask and rows * features != 0, slot[].im_missing,
        slot[].im_uncompact, slot[].im_old_items, slot[].im_keep_empty)
    var driver = XnDriver(call[].route.entry.call)
    if slot[].im_indicator and rows != 0 and len(slot[].im_miss) > 0:
        imp_transform_run(driver, request, slot[].im_plan, slot[].im_miss)
    else:
        imp_transform_run(driver, request, slot[].im_plan, List[Int]())


def _im_export(slot: _SlotPtr, call: _CallPtr) raises:
    """A view of one fitted block (the borrowed rows are copied here once)."""
    if not slot[].fitted:
        raise Error("NotFittedError: this KNNImputer member is not fitted yet")
    if (call[].flags & MEMBER_EXPORT_OWN) != 0:
        # THE LIFETIME RULE: the borrowed fit rows, once; no view is answered.
        _lof_own_rows(slot)
        return
    var block = call[].block
    if (call[].flags & MEMBER_BLOCK_BY_INDEX) != 0:
        if block < 0 or block >= _IM_BLOCKS:
            raise Error("member ABI: a KNNImputer member has three fitted blocks")
        block = IM_BLOCK_ROWS + block
    if block == IM_BLOCK_ROWS:
        _lof_own_rows(slot)
        call[].dst = MemberMatrix(slot[].rows_addr, slot[].rows, slot[].features)
    elif block == IM_BLOCK_VALID:
        call[].dst = MemberMatrix(Int(slot[].im_valid.unsafe_ptr()), 1, slot[].features, MEMBER_I32)
    elif block == IM_BLOCK_MISSING:
        call[].dst = MemberMatrix(Int(slot[].im_miss32.unsafe_ptr()), 1, len(slot[].im_miss), MEMBER_I32)
    else:
        raise Error("member ABI: a KNNImputer member has no fitted block " + String(block))
    call[].block = block
    call[].out_cols = _IM_BLOCKS


def _im_import(slot: _SlotPtr, call: _CallPtr) raises:
    """Copy one fitted block in; MEMBER_IMPORT_DONE completes the state (the
    rows and the valid columns are required; no missing block = none)."""
    var estimator = _name(slot)
    var block = call[].block
    var count = call[].x.rows * call[].x.cols
    if block == IM_BLOCK_ROWS:
        _dense(call[].x, MEMBER_F32, "_fit_X", estimator)
        slot[].fitted = False
        var words = List[Float32](length=max(count, 1), fill=Float32(0.0))
        if count > 0:
            unsafe_memcpy(dest=words.unsafe_ptr(), src=_F32P(unsafe_from_address=call[].x.data), count=count)
        slot[].fit_rows = words^
        slot[].rows_addr = Int(slot[].fit_rows.unsafe_ptr())
        slot[].rows = call[].x.rows
        slot[].features = call[].x.cols
    elif block == IM_BLOCK_VALID or block == IM_BLOCK_MISSING:
        slot[].fitted = False
        var words = List[Int](capacity=count)
        if count > 0:
            _dense(call[].x, MEMBER_I32, "a column list", estimator)
            var source = _I32P(unsafe_from_address=call[].x.data)
            for i in range(count):  # driver-loop: over the feature columns of a fitted column list
                words.append(Int(source[i]))
        if block == IM_BLOCK_VALID:
            slot[].im_valid = List[Int32](length=max(count, 1), fill=Int32(0))
            for i in range(count):  # driver-loop: over the d feature columns
                slot[].im_valid[i] = Int32(1) if words[i] != 0 else Int32(0)
            slot[].classes = count   # the valid-column count, checked at DONE
        else:
            slot[].im_miss = words^
    elif block != 0 or (call[].flags & MEMBER_IMPORT_DONE) == 0:
        raise Error("member ABI: a KNNImputer member has no fitted block " + String(block))
    if (call[].flags & MEMBER_IMPORT_DONE) == 0:
        return
    var features = slot[].features
    if (slot[].rows < 1 or len(slot[].fit_rows) < slot[].rows * features or slot[].classes != features):
        raise Error("member ABI: a KNNImputer import needs _fit_X and one valid word per feature")
    var valid = List[Bool](capacity=features)
    for f in range(features):  # driver-loop: over the d feature columns
        valid.append(slot[].im_valid[f] != 0)
    for i in range(len(slot[].im_miss)):  # driver-loop: over the columns missing at fit
        if slot[].im_miss[i] < 0 or slot[].im_miss[i] >= features:
            raise Error("member ABI: a KNNImputer missing column is outside the fitted features")
    var missing = slot[].im_miss.copy()
    _im_columns(slot, valid, missing)
    slot[].classes = 0
    slot[].fitted = True


# ------------------------------------------------------------ SVGP


def _sv_fit(slot: _SlotPtr, call: _CallPtr) raises:
    """`SVGP.fit(X, y)` (bindings/neighbors_graph_native_door.mojo
    `_svgp_fit`) with the inducing rows taken from X: m = min(n_inducing,
    rows) evenly spaced rows, then the variational fit; the definite check
    after the RUN, as the door's PACK makes it."""
    var estimator = _name(slot)
    _whole(call, estimator)
    if call[].fit_output != 0:
        member_unsupported(estimator, call[].fit_output)
    if call[].weights.rows > 0:
        raise Error("TypeError: a " + estimator + " member fits no sample_weight")
    _dense(call[].x, MEMBER_F32, "X", estimator)
    _dense(call[].y, MEMBER_F32, "y", estimator)
    var rows = call[].x.rows
    var features = call[].x.cols
    if call[].y.cols != 1:
        raise Error("ValueError: an SVGP member fits one target column")
    xn_rows_match_check(call[].y.rows, rows)
    # Caller-given inducing points (lane mf-nbr6): the rows the plan pinned at
    # UNPACK, copied into the slot's Z (the door's fit reads them as given).
    var pick_rows = slot[].sv_given == 0
    var m = min(slot[].sv_inducing, rows)
    if not pick_rows:
        m = slot[].sv_given_rows
        if slot[].sv_given_cols != features:
            raise Error("ValueError: inducing_points has " + String(slot[].sv_given_cols) + " features, but X has "
                + String(features) + " features")
    elif m <= 0:
        raise Error("mojolearn: rows is empty, shape (0,): Found array with 0 sample(s) (shape=(0,)) "
            + "while a minimum of 1 is required.")
    slot[].fitted = False
    slot[].sv_z = List[Float32](length=max(m * features, 1), fill=Float32(0.0))
    if not pick_rows and m * features > 0:
        unsafe_memcpy(dest=slot[].sv_z.unsafe_ptr(), src=_F32P(unsafe_from_address=slot[].sv_given),
            count=m * features)
    slot[].sv_alpha = List[Float32](length=m, fill=Float32(0.0))
    slot[].sv_c = List[Float32](length=m * m, fill=Float32(0.0))
    slot[].sv_qmu = List[Float32](length=m, fill=Float32(0.0))
    slot[].sv_qsqrt = List[Float32](length=m * m, fill=Float32(0.0))
    var fitted = svgp_fit_run(XnDriver(call[].route.entry.call), SvgpFitRequest(call[].x.data, call[].y.data,
        Int(slot[].sv_z.unsafe_ptr()), Int(slot[].sv_alpha.unsafe_ptr()), Int(slot[].sv_c.unsafe_ptr()),
        Int(slot[].sv_qmu.unsafe_ptr()), Int(slot[].sv_qsqrt.unsafe_ptr()), rows, m, features, pick_rows,
        slot[].sv_gamma, slot[].sv_kernel_variance, slot[].sv_noise_variance, slot[].sv_jitter))
    xn_svgp_definite_check(fitted[1])
    slot[].sv_m = m
    slot[].sv_elbo = fitted[0]
    slot[].rows = rows
    slot[].features = features
    slot[].fitted = True


def _sv_predict(slot: _SlotPtr, call: _CallPtr) raises:
    """`SVGP.predict(X)`: the mean (`_svgp_query`, SVGP_VARIANCE_NONE)."""
    var estimator = _name(slot)
    if not slot[].fitted:
        raise Error("NotFittedError: mojolearn: call fit before predict")
    _whole(call, estimator)
    _dense(call[].x, MEMBER_F32, "X", estimator)
    var rows = call[].x.rows
    if call[].x.cols != slot[].features:
        raise Error("ValueError: X has " + String(call[].x.cols) + " features, but SVGP is expecting "
            + String(slot[].features) + " features as input.")
    _dense(call[].dst, MEMBER_F32, "the prediction buffer", estimator)
    if call[].dst.rows != rows or call[].dst.cols != 1:
        raise Error("ValueError: the prediction buffer holds one column per row of X")
    svgp_query_run(XnDriver(call[].route.entry.call), SvgpQueryRequest(call[].x.data, Int(slot[].sv_z.unsafe_ptr()),
        Int(slot[].sv_alpha.unsafe_ptr()), Int(slot[].sv_c.unsafe_ptr()), call[].dst.data, 0, rows, slot[].sv_m,
        slot[].features, slot[].sv_gamma, slot[].sv_kernel_variance, 0.0, SVGP_VARIANCE_NONE, slot[].sv_old_items))


def _sv_export(slot: _SlotPtr, call: _CallPtr) raises:
    if not slot[].fitted:
        raise Error("NotFittedError: this SVGP member is not fitted yet")
    var block = call[].block
    if (call[].flags & MEMBER_BLOCK_BY_INDEX) != 0:
        if block < 0 or block >= _SV_BLOCKS:
            raise Error("member ABI: an SVGP member has six fitted blocks")
        block = SV_BLOCK_Z + block
    var m = slot[].sv_m
    if block == SV_BLOCK_Z:
        call[].dst = MemberMatrix(Int(slot[].sv_z.unsafe_ptr()), m, slot[].features)
    elif block == SV_BLOCK_ALPHA:
        call[].dst = MemberMatrix(Int(slot[].sv_alpha.unsafe_ptr()), 1, m)
    elif block == SV_BLOCK_C:
        call[].dst = MemberMatrix(Int(slot[].sv_c.unsafe_ptr()), m, m)
    elif block == SV_BLOCK_QMU:
        call[].dst = MemberMatrix(Int(slot[].sv_qmu.unsafe_ptr()), 1, m)
    elif block == SV_BLOCK_QSQRT:
        call[].dst = MemberMatrix(Int(slot[].sv_qsqrt.unsafe_ptr()), m, m)
    elif block == SV_BLOCK_SCALARS:
        slot[].sv_scalars[0] = slot[].sv_elbo
        slot[].sv_scalars[1] = Float64(slot[].features)
        call[].dst = MemberMatrix(Int(slot[].sv_scalars.unsafe_ptr()), 1, 2, MEMBER_F64)
    else:
        raise Error("member ABI: an SVGP member has no fitted block " + String(block))
    call[].block = block
    call[].out_cols = _SV_BLOCKS


def _sv_import(slot: _SlotPtr, call: _CallPtr) raises:
    """Copy one fitted block in; MEMBER_IMPORT_DONE completes the state (Z,
    alpha, C and the scalars are required; q_mu / q_sqrt only round-trip)."""
    var estimator = _name(slot)
    var block = call[].block
    var count = call[].x.rows * call[].x.cols
    if block == SV_BLOCK_SCALARS:
        _dense(call[].x, MEMBER_F64, "the SVGP scalars", estimator)
        if count != 2:
            raise Error("member ABI: the SVGP scalars are two float64 words (elbo, features)")
        slot[].fitted = False
        var words = _F64P(unsafe_from_address=call[].x.data)
        slot[].sv_elbo = words[0]
        slot[].features = Int(words[1])
    elif block >= SV_BLOCK_Z and block <= SV_BLOCK_QSQRT:
        _dense(call[].x, MEMBER_F32, "a fitted SVGP array", estimator)
        slot[].fitted = False
        var words = List[Float32](length=max(count, 1), fill=Float32(0.0))
        if count > 0:
            unsafe_memcpy(dest=words.unsafe_ptr(), src=_F32P(unsafe_from_address=call[].x.data), count=count)
        if block == SV_BLOCK_Z:
            slot[].sv_z = words^
            slot[].sv_m = call[].x.rows
        elif block == SV_BLOCK_ALPHA:
            slot[].sv_alpha = words^
        elif block == SV_BLOCK_C:
            slot[].sv_c = words^
        elif block == SV_BLOCK_QMU:
            slot[].sv_qmu = words^
        else:
            slot[].sv_qsqrt = words^
    elif block != 0 or (call[].flags & MEMBER_IMPORT_DONE) == 0:
        raise Error("member ABI: an SVGP member has no fitted block " + String(block))
    if (call[].flags & MEMBER_IMPORT_DONE) == 0:
        return
    var m = slot[].sv_m
    if (m < 1 or slot[].features < 1 or len(slot[].sv_z) < m * slot[].features or len(slot[].sv_alpha) < m
            or len(slot[].sv_c) < m * m):
        raise Error("member ABI: an SVGP import needs Z_, _alpha, _C and the scalars of one fit")
    slot[].fitted = True


# ------------------------------------------------------------ LabelPropagation / LabelSpreading


def _lp_fit_old_items(slot: _SlotPtr) -> Bool:
    """The fit's old-items arm as the door's `_lp_fit` reads it: the compact
    kNN graph unless unfused; the RBF graph only for spreading's normalized
    Laplacian (not raw)."""
    if slot[].lp_knn:
        return slot[].lp_old and not slot[].lp_unfused
    return slot[].lp_old and slot[].kind == XNEIGHBORS_KIND_LABEL_SPREADING and not slot[].lp_raw


def _lp_codes_check(codes: Int, rows: Int, classes: Int, marker: Int, estimator: String) raises:
    """Every code in 0 .. classes-1 and every class but the unlabeled
    marker (`marker`, -1: none) present in the fit rows: the estimator fits
    the classes it sees, so a fold or a draw missing a class is refused in
    words and the caller's own loop fits it (as for NearestCentroid). One
    presence row per task, parallel over the rows, then one pass over the
    classes."""
    if rows == 0:
        raise Error("ValueError: a " + estimator + " member needs at least one fit row")
    var tasks = max(host_predict_task_count_work(rows, rows), 1)
    var chunk = host_predict_chunk(rows, tasks)
    # Row t of `seen`: task t's presence words (classes), then its foreign-code word.
    var width = classes + 1
    var seen = List[Int32](length=tasks * width, fill=Int32(0))
    var seen_at = Int(seen.unsafe_ptr())

    def _scan(t: Int) {imm codes, imm rows, imm classes, imm chunk, imm width, imm seen_at}:
        var lo = t * chunk
        var hi = min((t + 1) * chunk, rows)
        var c = _I32P(unsafe_from_address=codes)
        var mark = _I32P(unsafe_from_address=seen_at + t * width * 4)
        for i in range(lo, hi):  # member fit: this task's rows
            var code = Int(c.unsafe_load(i))
            if code < 0 or code >= classes:
                mark.unsafe_store(classes, Int32(1))
            else:
                mark.unsafe_store(code, Int32(1))

    if tasks <= 1:
        _scan(0)
    else:
        host_parallelize(_scan, tasks)
    var foreign = False
    for t in range(tasks):  # driver-loop: over the scan's tasks (their foreign-code words)
        if seen[t * width + classes] != 0:
            foreign = True
    if foreign:
        raise Error("ValueError: a " + estimator + " member reads class codes 0 .. " + String(classes - 1))
    for c in range(classes):  # driver-loop: over classes (the tasks' presence words)
        if c == marker:
            continue  # the unlabeled rows: none is a class
        var present = False
        for t in range(tasks):  # driver-loop: over the scan's tasks
            if seen[t * width + c] != 0:
                present = True
        if not present:
            raise Error("ValueError: a " + estimator + " member needs every class code 0 .. "
                + String(classes - 1) + " present in its fit rows (class " + String(c) + " is absent)")
    _ = seen^


def _lp_fit(slot: _SlotPtr, call: _CallPtr) raises:
    """`_LabelPropagationBase.fit(X, y)` (bindings/neighbors_semi_native_door.mojo
    `_lp_fit`, the same `lp_fit_run`) on the caller's class codes with every
    row labeled: the marker word is the class count (the admission answers
    False for a caller whose labels hold the -1 marker)."""
    var estimator = _name(slot)
    _whole(call, estimator)
    if call[].fit_output != 0:
        member_unsupported(estimator, call[].fit_output)
    if call[].weights.rows > 0:
        raise Error("TypeError: a " + estimator + " member fits no sample_weight")
    _dense(call[].x, MEMBER_F32, "X", estimator)
    _dense(call[].y, MEMBER_I32, "y (class codes)", estimator)
    var rows = call[].x.rows
    var features = call[].x.cols
    if call[].y.cols != 1:
        raise Error("ValueError: a " + estimator + " member reads one column of class codes")
    xn_rows_match_check(call[].y.rows, rows)
    var given = call[].classes
    if given < 1:
        raise Error("ValueError: a " + estimator + " member is a classifier: it reads class codes and a class count")
    # THE UNLABELED MARKER: the caller's code of -1 is not a class; `lp_fit_run` reads it as the door does
    # (its marker word is that code; without one, the class count).
    var marker = slot[].lp_marker
    if marker >= given or (marker >= 0 and given < 2):
        raise Error("ValueError: a " + estimator + " member's unlabeled marker is not one of its " + String(given)
                    + " class codes")
    var classes = given - 1 if marker >= 0 else given
    var marker_word = marker if marker >= 0 else given
    _lp_codes_check(call[].y.data, rows, given, marker, estimator)
    var k = xn_semi_knn_k(slot[].lp_neighbors, rows) if slot[].lp_knn else 0
    slot[].fitted = False
    slot[].pred = List[Int32]()
    slot[].proba32 = List[Float32]()
    slot[].proba64 = List[Float64]()
    # Borrow the rows (LIFETIME, as LocalOutlierFactor: the door publishes the
    # admitted X itself as X_); a copy a previous import or export left is dropped.
    slot[].fit_rows = List[Float32]()
    slot[].rows_addr = call[].x.data
    slot[].lp_dist = List[Float32](length=rows * classes, fill=Float32(0.0))
    slot[].lp_codes = List[Int32](length=rows, fill=Int32(0))
    var request = LpFitRequest(call[].x.data, call[].y.data, Int(slot[].lp_dist.unsafe_ptr()),
        Int(slot[].lp_codes.unsafe_ptr()), rows, features, classes, marker_word,
        slot[].kind == XNEIGHBORS_KIND_LABEL_SPREADING, slot[].lp_knn, k, slot[].lp_kernel, slot[].lp_gamma,
        slot[].lp_raw, slot[].lp_weight, slot[].lp_alpha, slot[].lp_tol, slot[].lp_max_iter, slot[].lp_unfused,
        _lp_fit_old_items(slot))
    var outcome = lp_fit_run(XnDriver(call[].route.entry.call), request)
    if outcome.status != 0:
        admitted_finite_refuse(outcome.status, estimator)
    slot[].rows = rows
    slot[].features = features
    slot[].classes = classes
    slot[].lp_k = k
    slot[].lp_iterations = outcome.iterations
    slot[].fitted = True


def _lp_request(slot: _SlotPtr, call: _CallPtr, want_labels: Bool) raises -> LpQueryRequest:
    """`lp_query_unpack`'s words (bindings/neighbors_lp_query_unpack.mojo)
    over the fitted state: the query reads the old-items arm and the unfused
    arm only on the compact kNN graph, the image's dense-proba operation only
    on the RBF kernel. `q`, `dst` / `labels` are set by the caller."""
    var estimator = _name(slot)
    if not slot[].fitted:
        raise Error("NotFittedError: mojolearn: call fit before predict")
    _whole(call, estimator)
    _dense(call[].x, MEMBER_F32, "X", estimator)
    if call[].x.cols != slot[].features:
        raise Error("ValueError: X has " + String(call[].x.cols) + " features, but " + estimator
            + " is expecting " + String(slot[].features) + " features as input.")
    var knn = slot[].lp_knn
    var unfused = knn and slot[].lp_unfused
    var old_items = knn and slot[].lp_old and not slot[].lp_unfused
    var kind = 0 if knn else slot[].lp_kernel
    var gamma = 0.0 if knn else slot[].lp_gamma
    var fused = (not knn) and slot[].lp_fused
    return LpQueryRequest(call[].x.data, slot[].rows_addr, Int(slot[].lp_dist.unsafe_ptr()), 0, 0, call[].x.rows,
        slot[].rows, slot[].features, slot[].classes, knn, slot[].lp_k, kind, gamma, fused, want_labels, unfused,
        old_items)


def _lp_predict(slot: _SlotPtr, call: _CallPtr) raises:
    var estimator = _name(slot)
    var request = _lp_request(slot, call, True)
    var rows = request.rows
    _dense(call[].dst, MEMBER_I32, "the prediction buffer (class codes)", estimator)
    if call[].dst.rows != rows or call[].dst.cols != 1:
        raise Error("ValueError: the prediction buffer holds one class code per row of X")
    if rows == 0:
        return
    request.labels = call[].dst.data
    var status = lp_query_run(XnDriver(call[].route.entry.call), request)
    if status != 0:
        admitted_finite_refuse(status, estimator)
    var marker = slot[].lp_marker
    if marker < 0:
        return
    # THE UNLABELED MARKER: a fitted class c at or above the marker is the caller's code c + 1 (parallel over
    # the rows).
    var dst = call[].dst.data
    var tasks = max(host_predict_task_count_work(rows, rows), 1)
    var chunk = host_predict_chunk(rows, tasks)

    def _shift(t: Int) {imm dst, imm rows, imm chunk, imm marker}:
        var d = _I32P(unsafe_from_address=dst)
        for i in range(t * chunk, min((t + 1) * chunk, rows)):  # member predict: this task's rows
            var code = Int(d.unsafe_load(i))
            if code >= marker:
                d.unsafe_store(i, Int32(code + 1))

    if tasks <= 1:
        _shift(0)
    else:
        host_parallelize(_shift, tasks)


def _lp_proba(slot: _SlotPtr, call: _CallPtr) raises:
    """predict_proba: float32 rows x classes (the image's width); a float64
    buffer is filled from the state's float32 scratch (grown once), cast in
    parallel over rows."""
    var estimator = _name(slot)
    var request = _lp_request(slot, call, False)
    var rows = request.rows
    var classes = slot[].classes
    var asked_wide = call[].dst.dtype == MEMBER_F64
    _dense(call[].dst, MEMBER_F64 if asked_wide else MEMBER_F32, "the probability buffer", estimator)
    if call[].dst.rows != rows or call[].dst.cols != classes:
        raise Error("ValueError: the probability buffer holds one row of " + String(classes)
            + " class probabilities per row of X")
    var count = rows * classes
    if count == 0:
        return
    var target = call[].dst.data
    if asked_wide:
        if len(slot[].proba32) < count:
            slot[].proba32 = List[Float32](length=count, fill=Float32(0.0))
        target = Int(slot[].proba32.unsafe_ptr())
    request.dst = target
    var status = lp_query_run(XnDriver(call[].route.entry.call), request)
    if status != 0:
        admitted_finite_refuse(status, estimator)
    if not asked_wide:
        return
    var src = target
    var dst = call[].dst.data
    var tasks = max(host_predict_task_count_work(rows, count), 1)
    var chunk = host_predict_chunk(rows, tasks)

    def _cast(t: Int) {imm src, imm dst, imm rows, imm classes, imm chunk}:
        var lo = t * chunk * classes
        var hi = min((t + 1) * chunk, rows) * classes
        var s = _F32P(unsafe_from_address=src)
        var d = _F64P(unsafe_from_address=dst)
        for i in range(lo, hi):  # member predict_proba: this task's words
            d.unsafe_store(i, Float64(s.unsafe_load(i)))

    if tasks <= 1:
        _cast(0)
    else:
        host_parallelize(_cast, tasks)


def _lp_export(slot: _SlotPtr, call: _CallPtr) raises:
    """A view of one fitted block; the slot owns the words until this
    state's next fit, import or free (the borrowed rows are copied here)."""
    var estimator = _name(slot)
    if not slot[].fitted:
        raise Error("NotFittedError: this " + estimator + " member is not fitted yet")
    if (call[].flags & MEMBER_EXPORT_OWN) != 0:
        # THE LIFETIME RULE: the borrowed fit rows, once; no view is answered.
        _lof_own_rows(slot)
        return
    var block = call[].block
    if (call[].flags & MEMBER_BLOCK_BY_INDEX) != 0:
        if block < 0 or block >= _LP_BLOCKS:
            raise Error("member ABI: a " + estimator + " member has four fitted blocks")
        block = LP_BLOCK_ROWS + block
    if block == LP_BLOCK_ROWS:
        _lof_own_rows(slot)
        call[].dst = MemberMatrix(slot[].rows_addr, slot[].rows, slot[].features)
    elif block == LP_BLOCK_DISTRIBUTIONS:
        call[].dst = MemberMatrix(Int(slot[].lp_dist.unsafe_ptr()), slot[].rows, slot[].classes)
    elif block == LP_BLOCK_TRANSDUCTION:
        if len(slot[].lp_codes) != slot[].rows:
            raise Error("member ABI: an imported " + estimator + " state holds no transduction codes")
        call[].dst = MemberMatrix(Int(slot[].lp_codes.unsafe_ptr()), 1, slot[].rows, MEMBER_I32)
    elif block == LP_BLOCK_SCALARS:
        slot[].lp_scalars[0] = Float64(slot[].lp_iterations)
        slot[].lp_scalars[1] = Float64(slot[].features)
        call[].dst = MemberMatrix(Int(slot[].lp_scalars.unsafe_ptr()), 1, 2, MEMBER_F64)
    else:
        raise Error("member ABI: a " + estimator + " member has no fitted block " + String(block))
    call[].block = block
    call[].out_cols = _LP_BLOCKS
    call[].classes = slot[].classes


def _lp_import(slot: _SlotPtr, call: _CallPtr) raises:
    """Copy one fitted block in (the rows, the label distributions, the
    optional transduction codes, the scalars); MEMBER_IMPORT_DONE completes
    the state (the class count is the distributions' width)."""
    var estimator = _name(slot)
    var block = call[].block
    var count = call[].x.rows * call[].x.cols
    if block == LP_BLOCK_SCALARS:
        _dense(call[].x, MEMBER_F64, "the scalars", estimator)
        if count != 2:
            raise Error("member ABI: a " + estimator + " member's scalars are two float64 words (n_iter, features)")
        slot[].fitted = False
        slot[].lp_iterations = Int(_F64P(unsafe_from_address=call[].x.data)[0])
    elif block == LP_BLOCK_TRANSDUCTION:
        _dense(call[].x, MEMBER_I32, "the transduction codes", estimator)
        slot[].fitted = False
        slot[].lp_codes = List[Int32](length=count, fill=Int32(0))
        unsafe_memcpy(dest=slot[].lp_codes.unsafe_ptr(), src=_I32P(unsafe_from_address=call[].x.data), count=count)
    elif block == LP_BLOCK_ROWS or block == LP_BLOCK_DISTRIBUTIONS:
        _dense(call[].x, MEMBER_F32, "a fitted block", estimator)
        slot[].fitted = False
        var words = List[Float32](length=count, fill=Float32(0.0))
        unsafe_memcpy(dest=words.unsafe_ptr(), src=_F32P(unsafe_from_address=call[].x.data), count=count)
        if block == LP_BLOCK_ROWS:
            slot[].fit_rows = words^
            slot[].rows_addr = Int(slot[].fit_rows.unsafe_ptr())
            slot[].rows = call[].x.rows
            slot[].features = call[].x.cols
        else:
            slot[].lp_dist = words^
            slot[].classes = call[].x.cols
    elif block != 0 or (call[].flags & MEMBER_IMPORT_DONE) == 0:
        raise Error("member ABI: a " + estimator + " member has no fitted block " + String(block))
    if (call[].flags & MEMBER_IMPORT_DONE) == 0:
        return
    var rows = slot[].rows
    if (rows == 0 or slot[].features == 0 or len(slot[].fit_rows) != rows * slot[].features
            or slot[].classes < 1 or len(slot[].lp_dist) != rows * slot[].classes):
        raise Error("member ABI: a fitted " + estimator + " state needs its rows and one label distribution per row")
    if len(slot[].lp_codes) != 0 and len(slot[].lp_codes) != rows:
        raise Error("member ABI: a " + estimator + " member holds one transduction code per fitted row")
    slot[].lp_k = xn_semi_knn_k(slot[].lp_neighbors, rows) if slot[].lp_knn else 0
    slot[].pred = List[Int32]()
    slot[].proba32 = List[Float32]()
    slot[].proba64 = List[Float64]()
    slot[].fitted = True


# ------------------------------------------------------------ dispatch


comptime _NC_EUCLIDEAN = (MEMBER_CAN_FIT | MEMBER_CAN_PREDICT | MEMBER_CAN_PREDICT_PROBA | MEMBER_CAN_DECISION
    | MEMBER_CAN_EXPORT | MEMBER_CAN_IMPORT | MEMBER_CAN_CLASSIFIER)
comptime _NC_MANHATTAN = MEMBER_CAN_FIT | MEMBER_CAN_PREDICT | MEMBER_CAN_EXPORT | MEMBER_CAN_IMPORT | MEMBER_CAN_CLASSIFIER


# MEMBER_CAN_TEST_ROWS, per kind (MEMBER-ABI-V2 item 25: set only where every
# published operation of the kind refuses non-finite words itself):
#   NearestCentroid: none: bit clear. Its fit and its fused queries return
#     the deferred finite word (`_nc_fit` / `_nc_predict` / `_nc_decision` /
#     `_nc_proba` below), but `nc_labels_run` on the unfused entries defers
#     nothing (neighbors/nearest_centroid_run.mojo:140-166: status 0), and
#     the fused choice depends on the image and the row count.
#   LocalOutlierFactor (both kinds): SET. Fit (`lof_fit_run`,
#     neighbors/lof_run.mojo:64) and every query (`lof_query_run`, :130) take
#     the finite word of the distance entry on BOTH the tiled and the
#     unfused path (core/lof_plan.mojo:73-94 `select`: the status word is a
#     parameter of `sqdist` and of `knn_sq` / `knn_sq_tiled`), refused by
#     `admitted_finite_refuse` in `_lof_fit` and `_lof_query`.
#   OneClassSVM, KernelPCA, PolynomialCountSketch, AdditiveChi2Sampler,
#   SkewedChi2Sampler, KNNImputer, SVGP: none: bit clear (their RUN calls
#     return no finite word; KNNImputer admits NaN as missing by design and
#     its RUN applies no infinity rule).
comptime _LOF_NOVELTY = (MEMBER_CAN_FIT | MEMBER_CAN_PREDICT | MEMBER_CAN_DECISION | MEMBER_CAN_SCORE_SAMPLES
    | MEMBER_CAN_EXPORT | MEMBER_CAN_IMPORT | MEMBER_CAN_TEST_ROWS)
comptime _LOF_OUTLIER = (MEMBER_CAN_FIT | MEMBER_CAN_EXPORT | MEMBER_CAN_IMPORT   # fit_predict: FIT + fit_output
    | MEMBER_CAN_TEST_ROWS)
comptime _KERNEL_PCA = MEMBER_CAN_FIT | MEMBER_CAN_TRANSFORM | MEMBER_CAN_EXPORT | MEMBER_CAN_IMPORT
comptime _TRANSFORMER = _KERNEL_PCA   # fit, transform, export, import (fit_output TRANSFORM)
comptime _SVGP = MEMBER_CAN_FIT | MEMBER_CAN_PREDICT | MEMBER_CAN_EXPORT | MEMBER_CAN_IMPORT
comptime _OCSVM = (MEMBER_CAN_FIT | MEMBER_CAN_PREDICT | MEMBER_CAN_DECISION | MEMBER_CAN_SCORE_SAMPLES
    | MEMBER_CAN_EXPORT | MEMBER_CAN_IMPORT)
# LabelPropagation / LabelSpreading: SET for MEMBER_CAN_TEST_ROWS. The fit
# (`lp_fit_run`, neighbors/label_propagation_run.mojo:121/130) and every query
# (`lp_query_run`: the dense proba, the kNN indices or the RBF weights, its
# first operation) return the finite word of X, refused by
# `admitted_finite_refuse` in `_lp_fit`, `_lp_predict` and `_lp_proba`. The
# fit BORROWS X as its fitted rows (LIFETIME, as LocalOutlierFactor).
comptime _LABEL_PROP = (MEMBER_CAN_FIT | MEMBER_CAN_PREDICT | MEMBER_CAN_PREDICT_PROBA | MEMBER_CAN_EXPORT
    | MEMBER_CAN_IMPORT | MEMBER_CAN_CLASSIFIER | MEMBER_CAN_TEST_ROWS)


def _fold_kind(kind: Int) -> Bool:
    """The kinds whose every read of X is an upload through the device ops'
    `_buf` (x_neighbors/device_ops.mojo, THE FOLD X WORD) during the call,
    with no host read of X and no borrow of X past the call (lane
    vdf-scope-cluster3): NearestCentroid. LocalOutlierFactor, KernelPCA,
    LabelPropagation / LabelSpreading and a KNNImputer borrow the fit X as
    their fitted rows; OneClassSVM gathers its support vectors from host X
    (core/ocsvm_plan.mojo `oc_take_rows_into`); the samplers and SVGP are
    left on the host gather (vdf-scope-cluster3-RESULT.md)."""
    return kind == XNEIGHBORS_KIND_NEAREST_CENTROID


def _fold_bits(call: _CallPtr) -> Int:
    """MEMBER_CAN_ROW_INDEX | MEMBER_CAN_SCOPE on a GPU route whose image
    ships the x_prep family store (the fold's rows taken there), else 0:
    the CPU column keeps its host path (the caller gathers)."""
    if call[].route.device == DEVICE_GPU and member_fold_store_ready(call):
        return MEMBER_CAN_ROW_INDEX | MEMBER_CAN_SCOPE
    return 0


def _nc_fold_op(slot: _SlotPtr, call: _CallPtr, operation: Int) raises:
    """A NearestCentroid FIT / PREDICT / DECISION / PREDICT_PROBA over the
    rows `call.row_index` of the WHOLE host X (MEMBER_CAN_ROW_INDEX |
    MEMBER_CAN_SCOPE, lane vdf-scope-cluster3; the pattern of
    core/logistic_member.mojo `_fold_op`): X found in the caller's scope's
    x_prep cache, or put ONCE and published with the pin (THE PIN RULE:
    `call.pin_kept`), or, outside a scope, put into a slot this call owns;
    the fold's rows taken ON THE DEVICE into a fresh slot
    (`member_fold_x_open`), and every device op of the RUN uploads them
    from there (THE FOLD X WORD at `_buf`): no host gather of X, no per-fold
    upload of X. The fit's class codes are y's rows taken through the same
    index on the host (4-byte words, bit for bit, row chunks across the
    host's cores). Same rows in the same order as the host gather: the same
    bytes as the host-index path and the CPU column. The call's views are
    restored and the fold slot freed whether the operation returns or
    raises. The fitted state keeps only its own words (centroids, std,
    deviations, priors): nothing points into the slot after the call."""
    if not _fold_kind(slot[].kind):
        raise Error("TypeError: a " + _name(slot) + " member reads whole host views (no row or column index)")
    var estimator = _name(slot)
    var x = call[].x
    var y = call[].y
    var index = call[].row_index
    var fold = MemberFoldX()
    var host_y = List[Int32]()
    var failure = String("")
    try:
        member_fold_x_open(call, fold, estimator)
        call[].x = MemberMatrix(fold.word(), fold.rows, fold.cols)
        if operation == MEMBER_OP_FIT:
            _dense(y, MEMBER_I32, "y (class codes)", estimator)
            if y.rows != x.rows or y.cols != 1:
                raise Error("X and y have different numbers of rows")
            host_y = List[Int32](length=max(fold.rows, 1), fill=Int32(0))
            member_fold_take_view(y, fold.index, fold.rows, Int(host_y.unsafe_ptr()))
            call[].y = MemberMatrix(Int(host_y.unsafe_ptr()), fold.rows, 1, MEMBER_I32)
        call[].row_index = MemberMatrix()
        if operation == MEMBER_OP_FIT:
            _nc_fit(slot, call)
        elif operation == MEMBER_OP_PREDICT:
            _nc_predict(slot, call)
        elif operation == MEMBER_OP_DECISION:
            _nc_decision(slot, call)
        else:
            _nc_proba(slot, call)
    except error:
        failure = String(error)
    call[].x = x
    call[].y = y
    call[].row_index = index
    try:
        member_fold_x_close(fold)
    except close_error:
        if failure.byte_length() == 0:
            failure = "RuntimeError: a " + estimator + " member's fold slot was not freed: " + String(close_error)
    _ = host_y^
    if failure.byte_length() > 0:
        raise Error(failure)


def _capabilities(slot: _SlotPtr) -> Int:
    # MEMBER_BORROWS_FIT, kind-exact: LocalOutlierFactor, KernelPCA and a
    # KNNImputer without a masked copy read the caller's fit X as their
    # fitted rows until an export of the rows block, or MEMBER_EXPORT_OWN,
    # copied them. Every other kind owns its fitted words.
    if slot[].kind == XNEIGHBORS_KIND_LOF:
        return (_LOF_NOVELTY if slot[].novelty else _LOF_OUTLIER) | MEMBER_BORROWS_FIT
    if slot[].kind == XNEIGHBORS_KIND_KERNEL_PCA or (slot[].kind == XNEIGHBORS_KIND_IMPUTER and not slot[].im_mask):
        return _TRANSFORMER | MEMBER_BORROWS_FIT
    if slot[].kind == XNEIGHBORS_KIND_OCSVM:
        return _OCSVM
    if slot[].kind == XNEIGHBORS_KIND_SVGP:
        return _SVGP
    if slot[].kind == XNEIGHBORS_KIND_LABEL_PROPAGATION or slot[].kind == XNEIGHBORS_KIND_LABEL_SPREADING:
        return _LABEL_PROP | MEMBER_BORROWS_FIT
    if (slot[].kind == XNEIGHBORS_KIND_KERNEL_PCA or slot[].kind == XNEIGHBORS_KIND_PCS
            or slot[].kind == XNEIGHBORS_KIND_ACHI2 or slot[].kind == XNEIGHBORS_KIND_SCHI2
            or slot[].kind == XNEIGHBORS_KIND_IMPUTER):
        return _TRANSFORMER
    return _NC_EUCLIDEAN if slot[].euclidean else _NC_MANHATTAN


def _new(call: _CallPtr) raises:
    if call[].param_count < 1 or not call[].params:
        raise Error("member ABI: an XNeighbors member takes its kind as the first parameter")
    var words = _F64P(unsafe_from_address=call[].params)
    var kind = Int(words[0])
    if Float64(kind) != words[0] or kind < 0 or kind >= XNEIGHBORS_KIND_COUNT:
        raise Error("member ABI: unknown XNeighbors member kind")
    if kind == XNEIGHBORS_KIND_NEAREST_CENTROID:
        if call[].param_count < 6:
            raise Error("member ABI: a NearestCentroid member takes six parameters (kind, metric, priors, "
                + "shrink_threshold, fused, wide) and the given priors")
        if words[1] != 0.0 and words[1] != 1.0:
            raise Error("member ABI: unknown NearestCentroid metric code")
        if words[2] != 0.0 and words[2] != 1.0 and words[2] != 2.0:
            raise Error("member ABI: unknown NearestCentroid priors code")
        if (words[2] == 2.0) != (call[].param_count > 6):
            raise Error("member ABI: NearestCentroid given priors come as words 6 .. 6 + classes")
    elif kind == XNEIGHBORS_KIND_OCSVM:
        if call[].param_count != 14:
            raise Error("member ABI: a OneClassSVM member takes fourteen parameters (kind, kernel, nu, degree, "
                + "coef0, gamma mode, gamma, max_iter cap, tol, solve, fast, fused labels, old_items, precomputed)")
        if words[13] != 0.0 and words[9] != Float64(OC_SOLVE_GRAM):
            raise Error("member ABI: a precomputed OneClassSVM member solves on its Gram matrix")
        if words[5] != 0.0 and words[5] != 1.0 and words[5] != 2.0:
            raise Error("member ABI: unknown OneClassSVM gamma code")
    elif kind == XNEIGHBORS_KIND_KERNEL_PCA:
        if call[].param_count != 11:
            raise Error("member ABI: a KernelPCA member takes eleven parameters (kind, kernel, n_components, "
                + "auto solver, resident, gamma given, gamma, degree, coef0, remove_zero_eig, old_items)")
    elif kind == XNEIGHBORS_KIND_PCS:
        if call[].param_count != 7:
            raise Error("member ABI: a PolynomialCountSketch member takes seven parameters (kind, degree, "
                + "n_components, gamma, coef0, seed word, old_items)")
        _ = pcs_options(Int(words[1]), Int(words[2]))
    elif kind == XNEIGHBORS_KIND_ACHI2:
        if call[].param_count != 5:
            raise Error("member ABI: an AdditiveChi2Sampler member takes five parameters (kind, sample_steps, "
                + "interval, interval valid, device scan)")
    elif kind == XNEIGHBORS_KIND_SCHI2:
        if call[].param_count != 5:
            raise Error("member ABI: a SkewedChi2Sampler member takes five parameters (kind, n_components, "
                + "skewedness, seed word, one-call transform)")
    elif kind == XNEIGHBORS_KIND_IMPUTER:
        if call[].param_count != 10:
            raise Error("member ABI: a KNNImputer member takes ten parameters (kind, n_neighbors, distance, "
                + "mask, missing value, keep_empty_features, add_indicator, uncompact, old_items, lean)")
        # n_neighbors below 1 fits and is refused at transform, as the door (lane mf-nbr6).
    elif kind == XNEIGHBORS_KIND_SVGP:
        if call[].param_count != 10:
            raise Error("member ABI: an SVGP member takes ten parameters (kind, n_inducing, gamma, "
                + "kernel_variance, noise_variance, jitter, old_items, inducing address, inducing rows, "
                + "inducing columns)")
    elif kind == XNEIGHBORS_KIND_LABEL_PROPAGATION or kind == XNEIGHBORS_KIND_LABEL_SPREADING:
        if call[].param_count != _LP_WORDS:
            raise Error("member ABI: a LabelPropagation / LabelSpreading member takes fourteen parameters (kind, "
                + "knn, n_neighbors, rbf kernel code, gamma, raw graph, clamp weight, alpha, tol, max_iter, "
                + "unfused, old_items, dense proba, unlabeled marker)")
        if words[1] != 0.0 and words[1] != 1.0:
            raise Error("member ABI: unknown LabelPropagation graph code")
    else:
        if call[].param_count != 7:
            raise Error("member ABI: a LocalOutlierFactor member takes seven parameters (kind, n_neighbors, "
                + "novelty, contamination auto, contamination, unfused, old_items)")
        local_check_n_neighbors(Int(words[1]))
    call[].state = _mint(words, call[].param_count)


def _outputs(kind: Int, classes: Int, call: _CallPtr) raises:
    """OUTPUTS of a kind: NearestCentroid answers `classes` (proba),
    1 or `classes` (decision), 1 (predict); the outlier detectors one column
    and no classes."""
    if kind == XNEIGHBORS_KIND_LABEL_PROPAGATION or kind == XNEIGHBORS_KIND_LABEL_SPREADING:
        # predict_proba `classes` columns, predict one code; no decision.
        call[].classes = classes
        call[].out_cols = classes if call[].flags == MEMBER_OP_PREDICT_PROBA else 1
        return
    if kind == XNEIGHBORS_KIND_NEAREST_CENTROID:
        call[].classes = classes
        if call[].flags == MEMBER_OP_PREDICT_PROBA:
            call[].out_cols = classes
        elif call[].flags == MEMBER_OP_DECISION:
            call[].out_cols = 1 if classes == 2 else classes
        else:
            call[].out_cols = 1
        return
    if kind < 0 or kind >= XNEIGHBORS_KIND_COUNT:
        raise Error("member ABI: unknown XNeighbors member kind")
    call[].classes = 0
    call[].out_cols = 1


def _out_dtype(kind: Int, asked: Int) -> Int:
    """The element type of `dst` for the asked operation of a kind:
    NearestCentroid's predict writes int32 class codes; LocalOutlierFactor's
    and OneClassSVM's predict (and fit_predict) int64 labels; every other
    output float32 (NearestCentroid's predict_proba also takes a float64
    buffer when the caller hands one)."""
    if asked == MEMBER_OP_PREDICT or asked == MEMBER_OP_FIT:
        if (kind == XNEIGHBORS_KIND_NEAREST_CENTROID or kind == XNEIGHBORS_KIND_LABEL_PROPAGATION
                or kind == XNEIGHBORS_KIND_LABEL_SPREADING):
            return MEMBER_I32
        if kind == XNEIGHBORS_KIND_LOF or kind == XNEIGHBORS_KIND_OCSVM:
            return MEMBER_I64
    return MEMBER_F32


def _xneighbors_member(operation: Int, address: Int) raises -> Int:
    var call = _CallPtr(unsafe_from_address=address)
    if operation == MEMBER_OP_CAPABILITIES:
        if call[].state == 0:
            return (_NC_EUCLIDEAN | _LOF_NOVELTY | _OCSVM | _TRANSFORMER | _SVGP | _LABEL_PROP | MEMBER_BORROWS_FIT
                | _fold_bits(call))
        var held = _SlotPtr(unsafe_from_address=_lookup(call[].state, False))
        return _capabilities(held) | (_fold_bits(call) if _fold_kind(held[].kind) else 0)
    if operation == MEMBER_OP_NEW:
        _new(call)
        return 0
    if operation == MEMBER_OP_FREE:
        _ = _lookup(call[].state, True)
        call[].state = 0
        return 0
    if operation == MEMBER_OP_OUTPUTS and call[].state == 0:
        # No state (member_outputs): answer from the NEW words (the kind)
        # and `call.classes`, as a state made from them and fitted would.
        if call[].param_count < 1 or not call[].params:
            raise Error("member ABI: an XNeighbors member takes its kind as the first parameter")
        var asked = Int(_F64P(unsafe_from_address=call[].params)[0])
        call[].out_dtype = _out_dtype(asked, call[].flags)
        if asked == XNEIGHBORS_KIND_KERNEL_PCA:
            # The kept component count is chosen by the fit (rank, zero eigenvalues).
            raise Error("member ABI: a KernelPCA member's transform width is chosen by its fit; ask a fitted state")
        if asked == XNEIGHBORS_KIND_ACHI2:
            if call[].param_count != 5:
                raise Error("member ABI: an AdditiveChi2Sampler member takes five parameters")
            call[].classes = 0
            call[].out_cols = xn_achi2_width(call[].x.cols, Int(_F64P(unsafe_from_address=call[].params)[1]))
            return 0
        if asked == XNEIGHBORS_KIND_IMPUTER:
            if call[].param_count != 10:
                raise Error("member ABI: a KNNImputer member takes ten parameters")
            var asked_words = _F64P(unsafe_from_address=call[].params)
            if asked_words[5] == 0.0 or asked_words[6] != 0.0:
                # The kept columns (and the indicator columns) are chosen by the fit.
                raise Error("member ABI: a KNNImputer member's transform width is chosen by its fit; ask a fitted state")
            call[].classes = 0
            call[].out_cols = call[].x.cols
            return 0
        if asked == XNEIGHBORS_KIND_SCHI2:
            if call[].param_count != 4:
                raise Error("member ABI: a SkewedChi2Sampler member takes four parameters")
            call[].classes = 0
            call[].out_cols = Int(_F64P(unsafe_from_address=call[].params)[1])
            return 0
        if asked == XNEIGHBORS_KIND_PCS:
            if call[].param_count != 7:
                raise Error("member ABI: a PolynomialCountSketch member takes seven parameters")
            call[].classes = 0
            call[].out_cols = Int(_F64P(unsafe_from_address=call[].params)[2])
            return 0
        _outputs(asked, call[].classes, call)
        return 0
    var slot = _SlotPtr(unsafe_from_address=_lookup(call[].state, False))
    if operation == MEMBER_OP_OUTPUTS:
        call[].out_dtype = _out_dtype(slot[].kind, call[].flags)
        if slot[].kind == XNEIGHBORS_KIND_KERNEL_PCA:
            if not slot[].fitted:
                raise Error("member ABI: a KernelPCA member's transform width is chosen by its fit; fit it first")
            call[].classes = 0
            call[].out_cols = slot[].kp_kept
            return 0
        if slot[].kind == XNEIGHBORS_KIND_PCS:
            call[].classes = 0
            call[].out_cols = slot[].pc_components
            return 0
        if slot[].kind == XNEIGHBORS_KIND_ACHI2:
            call[].classes = 0
            call[].out_cols = xn_achi2_width(slot[].features if slot[].fitted else call[].x.cols, slot[].ac_steps)
            return 0
        if slot[].kind == XNEIGHBORS_KIND_SCHI2:
            call[].classes = 0
            call[].out_cols = slot[].sc_components
            return 0
        if slot[].kind == XNEIGHBORS_KIND_IMPUTER:
            if not slot[].fitted and (not slot[].im_keep_empty or slot[].im_indicator):
                raise Error("member ABI: a KNNImputer member's transform width is chosen by its fit; fit it first")
            call[].classes = 0
            call[].out_cols = _im_width(slot, 1) if slot[].fitted else call[].x.cols
            return 0
        _outputs(slot[].kind, slot[].classes if slot[].fitted else call[].classes, call)
        return 0
    if slot[].kind == XNEIGHBORS_KIND_LABEL_PROPAGATION or slot[].kind == XNEIGHBORS_KIND_LABEL_SPREADING:
        if operation == MEMBER_OP_FIT:
            _lp_fit(slot, call)
        elif operation == MEMBER_OP_PREDICT:
            _lp_predict(slot, call)
        elif operation == MEMBER_OP_PREDICT_PROBA:
            _lp_proba(slot, call)
        elif operation == MEMBER_OP_EXPORT:
            _lp_export(slot, call)
        elif operation == MEMBER_OP_IMPORT:
            _lp_import(slot, call)
        else:
            member_unsupported(_name(slot), operation)
        return 0
    if slot[].kind == XNEIGHBORS_KIND_SVGP:
        if operation == MEMBER_OP_FIT:
            _sv_fit(slot, call)
        elif operation == MEMBER_OP_PREDICT:
            _sv_predict(slot, call)
        elif operation == MEMBER_OP_EXPORT:
            _sv_export(slot, call)
        elif operation == MEMBER_OP_IMPORT:
            _sv_import(slot, call)
        else:
            member_unsupported(_name(slot), operation)
        return 0
    if slot[].kind == XNEIGHBORS_KIND_IMPUTER:
        if operation == MEMBER_OP_FIT:
            _im_fit(slot, call)
        elif operation == MEMBER_OP_TRANSFORM:
            _im_transform(slot, call)
        elif operation == MEMBER_OP_EXPORT:
            _im_export(slot, call)
        elif operation == MEMBER_OP_IMPORT:
            _im_import(slot, call)
        else:
            member_unsupported(_name(slot), operation)
        return 0
    if slot[].kind == XNEIGHBORS_KIND_SCHI2:
        if operation == MEMBER_OP_FIT:
            _sc_fit(slot, call)
        elif operation == MEMBER_OP_TRANSFORM:
            _sc_transform(slot, call)
        elif operation == MEMBER_OP_EXPORT:
            _sc_export(slot, call)
        elif operation == MEMBER_OP_IMPORT:
            _sc_import(slot, call)
        else:
            member_unsupported(_name(slot), operation)
        return 0
    if slot[].kind == XNEIGHBORS_KIND_ACHI2:
        if operation == MEMBER_OP_FIT:
            _ac_fit(slot, call)
        elif operation == MEMBER_OP_TRANSFORM:
            _ac_transform(slot, call, False)
        elif operation == MEMBER_OP_EXPORT:
            _ac_export(slot, call)
        elif operation == MEMBER_OP_IMPORT:
            _ac_import(slot, call)
        else:
            member_unsupported(_name(slot), operation)
        return 0
    if slot[].kind == XNEIGHBORS_KIND_PCS:
        if operation == MEMBER_OP_FIT:
            _pc_fit(slot, call)
        elif operation == MEMBER_OP_TRANSFORM:
            _pc_transform(slot, call)
        elif operation == MEMBER_OP_EXPORT:
            _pc_export(slot, call)
        elif operation == MEMBER_OP_IMPORT:
            _pc_import(slot, call)
        else:
            member_unsupported(_name(slot), operation)
        return 0
    if slot[].kind == XNEIGHBORS_KIND_KERNEL_PCA:
        if operation == MEMBER_OP_FIT:
            _kp_fit(slot, call)
        elif operation == MEMBER_OP_TRANSFORM:
            _kp_transform(slot, call)
        elif operation == MEMBER_OP_EXPORT:
            _kp_export(slot, call)
        elif operation == MEMBER_OP_IMPORT:
            _kp_import(slot, call)
        else:
            member_unsupported(_name(slot), operation)
        return 0
    if slot[].kind == XNEIGHBORS_KIND_OCSVM:
        if operation == MEMBER_OP_FIT:
            _oc_fit(slot, call)
        elif operation == MEMBER_OP_PREDICT:
            _oc_query(slot, call, LOF_RESULT_LABELS, "predict")
        elif operation == MEMBER_OP_DECISION:
            _oc_query(slot, call, LOF_RESULT_DECISION, "decision_function")
        elif operation == MEMBER_OP_SCORE_SAMPLES:
            _oc_query(slot, call, LOF_RESULT_SCORES, "score_samples")
        elif operation == MEMBER_OP_EXPORT:
            _oc_export(slot, call)
        elif operation == MEMBER_OP_IMPORT:
            _oc_import(slot, call)
        else:
            member_unsupported(_name(slot), operation)
        return 0
    if slot[].kind == XNEIGHBORS_KIND_LOF:
        if operation == MEMBER_OP_FIT:
            _lof_fit(slot, call)
        elif operation == MEMBER_OP_PREDICT:
            _lof_query(slot, call, LOF_RESULT_LABELS, "predict")
        elif operation == MEMBER_OP_DECISION:
            _lof_query(slot, call, LOF_RESULT_DECISION, "decision_function")
        elif operation == MEMBER_OP_SCORE_SAMPLES:
            _lof_query(slot, call, LOF_RESULT_SCORES, "score_samples")
        elif operation == MEMBER_OP_EXPORT:
            _lof_export(slot, call)
        elif operation == MEMBER_OP_IMPORT:
            _lof_import(slot, call)
        else:
            member_unsupported(_name(slot), operation)
        return 0
    if (call[].row_index.rows > 0 and call[].column_index.rows == 0 and call[].route.device == DEVICE_GPU
            and (operation == MEMBER_OP_FIT or operation == MEMBER_OP_PREDICT or operation == MEMBER_OP_DECISION
            or operation == MEMBER_OP_PREDICT_PROBA)):
        _nc_fold_op(slot, call, operation)
        return 0
    if operation == MEMBER_OP_FIT:
        _nc_fit(slot, call)
    elif operation == MEMBER_OP_PREDICT:
        _nc_predict(slot, call)
    elif operation == MEMBER_OP_DECISION:
        _nc_decision(slot, call)
    elif operation == MEMBER_OP_PREDICT_PROBA:
        _nc_proba(slot, call)
    elif operation == MEMBER_OP_EXPORT:
        _nc_export(slot, call)
    elif operation == MEMBER_OP_IMPORT:
        _nc_import(slot, call)
    else:
        member_unsupported(_name(slot), operation)
    return 0


def xneighbors_member_v2(operation: Int, aa: Int, na: Int, pp: Int, np: Int, ff: Int, nf: Int,
    result: Int, context: Int, sink: NativeDescriptorError) abi("C") -> Int:
    """The member entry of the XNeighbors family (schema `MEMBER_ABI`)."""
    try:
        var word = _xneighbors_member(operation, member_call_from_spans(operation, aa, na, np, nf))
        if result:
            MutPointer[Int, MutUntrackedOrigin](unsafe_from_address=result)[] = word
        return 0
    except error:
        native_descriptor_report(error, context, sink)
        return 1


def xneighbors_member_publish(mode: Int, vendor: Int, publisher: Int) raises:
    """Publish this image's XNeighbors member entry. One line in each
    x_neighbors image's module initialization; a build with no GPU vendor
    (`vendor` -1) publishes nothing."""
    if vendor < 0:
        return
    member_publish[MEMBER_FAMILY_XNEIGHBORS](mode, vendor, publisher, xneighbors_member_v2)
