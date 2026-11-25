# SPDX-License-Identifier: Apache-2.0
# native-check: cpu
"""Lane mf-member8 (2026-11-23): the member kinds RFE with no elimination round fits natively, their words,
blocks and refusals before any image is called, and the RFE named-block rows this lane added. Written, not run
(WRITE CODE ONLY); IN_FILTER=mf_member8_members. The fits' equality with the estimators' own fits (OPTICS, MDS,
SpectralEmbedding run their image entries) is the Python test python/mojolearn/tests/test_mf_member8_members.py
and the digests tools/mf_member8_digests.py.

    test_nn_points_round_trip      NearestNeighbors (neighbors kind 2) FIT on X 4 x 3 with a y it does not read:
                                   kind mask FIT | EXPORT | IMPORT (no PREDICT), EXPORT of the points block equals
                                   X word for word (the public fit's `_index` is X), PREDICT refused, an IMPORT of
                                   the points alone completes a state that exports the same words
    test_nn_metric_input_checked   NearestNeighbors with jensenshannon (code 15) refuses negative X at FIT (the
                                   fit's `_check_metric_input`); RadiusNeighbors (kind 3, word [2] -1) fits it
    test_neighbors_kind_four       NEW kind 4 is refused ("unknown k-NN member kind")
    test_optics_words_and_sizes    OPTICS (x_cluster kind 3): 13 words refused; no `route.third` refused; a count
                                   min_samples 5 on 4 rows refused in `_optics_size`'s words before the entry is
                                   called; kind mask FIT | EXPORT | IMPORT | IGNORES_Y; PREDICT in scikit-learn's
                                   words; round b: an IMPORT of the arrays exports them back
    test_mds_words_and_mask        MDS (x_decomp kind 18): 13 words refused; 14 mint a state whose mask is
                                   FIT | EXPORT | IMPORT | IGNORES_Y; TRANSFORM in scikit-learn's words
    test_se_refusals_before_entry  SpectralEmbedding (spectral kind 1, 15 words): mask FIT | EXPORT; a non-square
                                   precomputed X refused in `_se_execute`'s words; n_components 3 of 4 rows with
                                   drop_first refused by `_se_check_shape`; PREDICT refused
    test_rfe_named_rows            `rfe_named_block`: AffinityPropagation cluster_centers_ (FAMILY+0),
                                   IncrementalPCA mean_ (FAMILY+2) and var_ (FAMILY+3); MeanShift unchanged

FAIL-PROOF (each edit makes the named case fail): in core/neighbors_member.mojo return `_REGRESSOR` for an
unsupervised slot -> test_nn_points_round_trip; drop the `knn_check_metric_input` call of `_fit_points` ->
test_nn_metric_input_checked; accept `words[0] == 4.0` in NEW -> test_neighbors_kind_four; in
core/xcluster_member.mojo make `_optics_size` return `Int(value)` without its range test ->
test_optics_words_and_sizes; set XDECOMP_MDS_PARAM_COUNT to 13 -> test_mds_words_and_mask; drop the square check
of `_fit_se` -> test_se_refusals_before_entry; drop `kind == 4` from the XCLUSTER row of `rfe_named_block` ->
test_rfe_named_rows; store the reachability block of `_optics_import` into `core_distances` ->
test_optics_words_and_sizes; drop MEMBER_IGNORES_Y from `_POINTS` -> test_nn_points_round_trip.
"""
from std.testing import TestSuite
from std.memory import bitcast
from core.entry_table import EntryRef
from core.device_route import DeviceRoute, DEVICE_CPU
from core.member_abi import (
    MemberCall, MemberMatrix, MemberRef, member_call, MEMBER_OP_NEW, MEMBER_OP_FREE, MEMBER_OP_FIT,
    MEMBER_OP_PREDICT, MEMBER_OP_TRANSFORM, MEMBER_OP_EXPORT, MEMBER_OP_IMPORT, MEMBER_OP_CAPABILITIES,
    MEMBER_CAN_FIT, MEMBER_CAN_EXPORT, MEMBER_CAN_IMPORT, MEMBER_CAN_PREDICT, MEMBER_CAN_TRANSFORM,
    MEMBER_BLOCK_FAMILY, MEMBER_IMPORT_DONE, MEMBER_IGNORES_Y, MEMBER_BLOCK_SCALARS, MEMBER_F64, MEMBER_I32,
    MEMBER_CAN_ROW_INDEX,
)
from core.member_families import (
    MEMBER_FAMILY_XCLUSTER, MEMBER_FAMILY_XDECOMP, MEMBER_FAMILY_DISCRIMINANT, MEMBER_FAMILY_PREP,
    MEMBER_FAMILY_XNEIGHBORS,
)
from core.neighbors_member import neighbors_member_v2, NEIGHBORS_BLOCK_POINTS
from core.xcluster_member import xcluster_member_v2
from core.xdecomp_member import xdecomp_member_v2, XDECOMP_KIND_MDS, XDECOMP_MDS_PARAM_COUNT
from core.spectral_member import spectral_member_v2
from bindings.prep_select_run import rfe_named_block
from checks.native_tests.testsuite_cases import require

comptime _F32P = MutPointer[Float32, MutUntrackedOrigin]


def _ref(which: Int) -> MemberRef:
    if which == 0:
        return MemberRef(EntryRef(neighbors_member_v2, 0), 0)
    if which == 1:
        return MemberRef(EntryRef(xcluster_member_v2, 0), 0)
    if which == 2:
        return MemberRef(EntryRef(xdecomp_member_v2, 0), 0)
    return MemberRef(EntryRef(spectral_member_v2, 0), 0)


def _new(which: Int, words: List[Float64], mut call: MemberCall) raises:
    call.params = Int(words.unsafe_ptr())
    call.param_count = len(words)
    _ = member_call(_ref(which), MEMBER_OP_NEW, call)
    call.params = 0
    call.param_count = 0


def _free(which: Int, mut call: MemberCall):
    try:
        _ = member_call(_ref(which), MEMBER_OP_FREE, call)
    except:
        pass


def _raises(which: Int, operation: Int, mut call: MemberCall, words: String) -> Bool:
    try:
        _ = member_call(_ref(which), operation, call)
    except error:
        return String(error).find(words) >= 0
    return False


def _x(rows: Int, cols: Int, shift: Float32) -> List[Float32]:
    var x = List[Float32](length=rows * cols, fill=0.0)
    for i in range(rows * cols):  # small-loop(a 4 x 3 fixture)
        x[i] = Float32(i) * 0.25 + shift
    return x^


def test_nn_points_round_trip() raises:
    var x = _x(4, 3, 0.5)
    var y = List[Float32](length=4, fill=1.0)
    var words: List[Float64] = [2.0, 5.0, 1.0, 2.0, 0.0, 0.0]
    var call = MemberCall(DeviceRoute(DEVICE_CPU))
    _new(0, words, call)
    try:
        var mask = member_call(_ref(0), MEMBER_OP_CAPABILITIES, call)
        require((mask & MEMBER_CAN_FIT) != 0 and (mask & MEMBER_CAN_EXPORT) != 0 and (mask & MEMBER_CAN_IMPORT) != 0,
                "NearestNeighbors kind mask lacks FIT / EXPORT / IMPORT")
        require((mask & MEMBER_CAN_PREDICT) == 0, "NearestNeighbors kind mask has PREDICT")
        require((mask & MEMBER_IGNORES_Y) != 0, "NearestNeighbors kind mask lacks MEMBER_IGNORES_Y")
        call.x = MemberMatrix(Int(x.unsafe_ptr()), 4, 3)
        call.y = MemberMatrix(Int(y.unsafe_ptr()), 4, 1)
        _ = member_call(_ref(0), MEMBER_OP_FIT, call)
        call.block = NEIGHBORS_BLOCK_POINTS
        call.flags = 0
        _ = member_call(_ref(0), MEMBER_OP_EXPORT, call)
        require(call.dst.rows == 4 and call.dst.cols == 3, "NearestNeighbors points block is not 4 x 3")
        var got = _F32P(unsafe_from_address=call.dst.data)
        for i in range(12):  # small-loop(the 4 x 3 points)
            require(got[i] == x[i], "NearestNeighbors points word " + String(i) + " differs from X")
        require(_raises(0, MEMBER_OP_PREDICT, call, "AttributeError: 'NearestNeighbors' object has no attribute 'predict'"),
                "NearestNeighbors PREDICT was not refused in scikit-learn's words")
    finally:
        _free(0, call)
    var back = MemberCall(DeviceRoute(DEVICE_CPU))
    _new(0, words, back)
    try:
        back.block = NEIGHBORS_BLOCK_POINTS
        back.x = MemberMatrix(Int(x.unsafe_ptr()), 4, 3)
        back.flags = MEMBER_IMPORT_DONE
        _ = member_call(_ref(0), MEMBER_OP_IMPORT, back)
        back.flags = 0
        back.block = NEIGHBORS_BLOCK_POINTS
        _ = member_call(_ref(0), MEMBER_OP_EXPORT, back)
        var again = _F32P(unsafe_from_address=back.dst.data)
        for i in range(12):  # small-loop(the 4 x 3 points)
            require(again[i] == x[i], "an imported NearestNeighbors state exports other points")
    finally:
        _free(0, back)
    _ = len(x)
    _ = len(y)


def test_nn_metric_input_checked() raises:
    var x = _x(4, 3, -1.0)
    var y = List[Float32](length=4, fill=0.0)
    var nn: List[Float64] = [2.0, 5.0, 15.0, 2.0, 0.0, 0.0]
    var call = MemberCall(DeviceRoute(DEVICE_CPU))
    _new(0, nn, call)
    try:
        call.x = MemberMatrix(Int(x.unsafe_ptr()), 4, 3)
        call.y = MemberMatrix(Int(y.unsafe_ptr()), 4, 1)
        var refused = False
        try:
            _ = member_call(_ref(0), MEMBER_OP_FIT, call)
        except:
            refused = True
        require(refused, "NearestNeighbors jensenshannon fitted a negative X")
    finally:
        _free(0, call)
    var radius: List[Float64] = [3.0, 0.0, -1.0, 2.0, 0.0, 0.0]
    var other = MemberCall(DeviceRoute(DEVICE_CPU))
    _new(0, radius, other)
    try:
        other.x = MemberMatrix(Int(x.unsafe_ptr()), 4, 3)
        other.y = MemberMatrix(Int(y.unsafe_ptr()), 4, 1)
        _ = member_call(_ref(0), MEMBER_OP_FIT, other)
    finally:
        _free(0, other)
    _ = len(x)
    _ = len(y)


def test_neighbors_kind_four() raises:
    var words: List[Float64] = [4.0, 5.0, 1.0, 2.0, 0.0, 0.0]
    var call = MemberCall(DeviceRoute(DEVICE_CPU))
    call.params = Int(words.unsafe_ptr())
    call.param_count = len(words)
    require(_raises(0, MEMBER_OP_NEW, call, "unknown k-NN member kind"), "neighbors kind 4 was not refused")
    _ = len(words)


def _optics_words(count: Int) -> List[Float64]:
    # [0] 3, min_samples a count of 5, min_cluster_size None, 'xi', predecessor_correction, euclidean (-1, 2),
    # max_eps, xi 0.05, eps = max_eps.
    var words: List[Float64] = [3.0, 0.0, 5.0, 0.0, -1.0, 0.0, 0.0, 0.0, 1.0, -1.0, 2.0, 1.0e30, 0.05, 1.0e30]
    while len(words) > count:  # small-loop(shorten a fixture by one word)
        _ = words.pop()
    return words^


def test_optics_words_and_sizes() raises:
    var short = _optics_words(13)
    var refused = MemberCall(DeviceRoute(DEVICE_CPU))
    refused.params = Int(short.unsafe_ptr())
    refused.param_count = len(short)
    require(_raises(1, MEMBER_OP_NEW, refused, "takes 14 parameter words"), "OPTICS with 13 words was not refused")
    _ = len(short)
    var words = _optics_words(14)
    var x = _x(4, 2, 0.0)
    var call = MemberCall(DeviceRoute(DEVICE_CPU))
    _new(1, words, call)
    try:
        var mask = member_call(_ref(1), MEMBER_OP_CAPABILITIES, call)
        require(mask == (MEMBER_CAN_FIT | MEMBER_CAN_EXPORT | MEMBER_CAN_IMPORT | MEMBER_IGNORES_Y),
                "OPTICS kind mask is not FIT | EXPORT | IMPORT | IGNORES_Y")
        call.x = MemberMatrix(Int(x.unsafe_ptr()), 4, 2)
        require(_raises(1, MEMBER_OP_FIT, call, "holds no OPTICS entry"), "OPTICS fit without route.third")
        # An entry that is never called: the size refusal comes first.
        call.route.third = EntryRef(xcluster_member_v2, 0)
        call.route.has_third = True
        require(_raises(1, MEMBER_OP_FIT, call,
                        "min_samples must be no greater than the number of samples (4) and >= 2, got 5"),
                "OPTICS min_samples 5 on 4 rows was not refused in _optics_size's words")
        require(_raises(1, MEMBER_OP_PREDICT, call, "AttributeError: 'OPTICS' object has no attribute 'predict'"),
                "OPTICS PREDICT was not refused in scikit-learn's words")
        # Round b: IMPORT of the five per-row arrays and the scalars completes a state that exports them.
        var labels: List[Int32] = [0, 0, -1, 1]
        var reach: List[Float32] = [1.0e30, 0.5, 0.25, 0.75]
        var ints = List[Int32](length=4, fill=Int32(1))
        var width: List[Float64] = [2.0]
        call.flags = 0
        call.block = MEMBER_BLOCK_FAMILY + 1
        call.x = MemberMatrix(Int(labels.unsafe_ptr()), 1, 4, MEMBER_I32)
        _ = member_call(_ref(1), MEMBER_OP_IMPORT, call)
        for blk in range(40, 44):  # small-loop(the four other per-row blocks)
            call.block = MEMBER_BLOCK_FAMILY + blk
            if blk <= 41:
                call.x = MemberMatrix(Int(reach.unsafe_ptr()), 1, 4)
            else:
                call.x = MemberMatrix(Int(ints.unsafe_ptr()), 1, 4, MEMBER_I32)
            _ = member_call(_ref(1), MEMBER_OP_IMPORT, call)
        call.block = MEMBER_BLOCK_SCALARS
        call.x = MemberMatrix(Int(width.unsafe_ptr()), 1, 1, MEMBER_F64)
        call.flags = MEMBER_IMPORT_DONE
        _ = member_call(_ref(1), MEMBER_OP_IMPORT, call)
        call.flags = 0
        call.block = MEMBER_BLOCK_FAMILY + 41
        _ = member_call(_ref(1), MEMBER_OP_EXPORT, call)
        require(call.dst.cols == 4 and _F32P(unsafe_from_address=call.dst.data)[2] == 0.25,
                "an imported OPTICS state exports other reachability")
        _ = len(labels)
        _ = len(reach)
        _ = len(ints)
        _ = len(width)
    finally:
        _free(1, call)
    _ = len(words)
    _ = len(x)


def test_mds_words_and_mask() raises:
    var short = List[Float64](length=XDECOMP_MDS_PARAM_COUNT - 1, fill=0.0)
    short[0] = Float64(XDECOMP_KIND_MDS)
    var refused = MemberCall(DeviceRoute(DEVICE_CPU))
    refused.params = Int(short.unsafe_ptr())
    refused.param_count = len(short)
    require(_raises(2, MEMBER_OP_NEW, refused, "an MDS member takes fourteen parameters"),
            "MDS with 13 words was not refused")
    _ = len(short)
    var words = List[Float64](length=XDECOMP_MDS_PARAM_COUNT, fill=0.0)
    words[0] = Float64(XDECOMP_KIND_MDS)
    words[8] = 2.0
    var call = MemberCall(DeviceRoute(DEVICE_CPU))
    _new(2, words, call)
    try:
        var mask = member_call(_ref(2), MEMBER_OP_CAPABILITIES, call)
        require(mask == (MEMBER_CAN_FIT | MEMBER_CAN_EXPORT | MEMBER_CAN_IMPORT | MEMBER_IGNORES_Y),
                "MDS kind mask is not FIT | EXPORT | IMPORT | IGNORES_Y")
        require((mask & MEMBER_CAN_TRANSFORM) == 0, "MDS kind mask has TRANSFORM")
        require(_raises(2, MEMBER_OP_TRANSFORM, call, "AttributeError: 'MDS' object has no attribute 'transform'"),
                "MDS TRANSFORM was not refused in scikit-learn's words")
    finally:
        _free(2, call)
    _ = len(words)


def _se_words(k: Int, affinity: Int) -> List[Float64]:
    # 15 words: kind 1, n_components k, n_neighbors None (auto), eigen_tol 1e-5, seed 0, the affinity code, gamma
    # 1.0, a dense fit entry, drop_first, norm_laplacian, automatic n_neighbors.
    var words: List[Float64] = [1.0, 0.0, Float64(k), 0.0, 0.0, 1.0e-5, bitcast[DType.float64](Int64(0)), -1.0,
        Float64(affinity), 0.0, 1.0, 1.0, 1.0, 1.0, 1.0]
    return words^


def test_se_refusals_before_entry() raises:
    var x = _x(4, 3, 0.0)
    var precomputed = _se_words(2, 1)
    var call = MemberCall(DeviceRoute(DEVICE_CPU))
    _new(3, precomputed, call)
    try:
        var mask = member_call(_ref(3), MEMBER_OP_CAPABILITIES, call)
        # 999e1d2475 (lane vdf-scope-cluster6): the kind reads a fold's row index (MEMBER_CAN_ROW_INDEX); the host
        # route has no SPECTRAL_CAP_DEV_STORE (core/spectral_native_device.mojo only), so no MEMBER_CAN_SCOPE
        require(mask == (MEMBER_CAN_FIT | MEMBER_CAN_EXPORT | MEMBER_CAN_IMPORT | MEMBER_IGNORES_Y | MEMBER_CAN_ROW_INDEX),
                "SpectralEmbedding kind mask is not FIT | EXPORT | IMPORT | IGNORES_Y | ROW_INDEX")
        call.x = MemberMatrix(Int(x.unsafe_ptr()), 4, 3)
        require(_raises(3, MEMBER_OP_FIT, call, "must be a square affinity matrix or a SciPy sparse matrix, "
                        + "got shape (4, 3)"), "SpectralEmbedding non-square precomputed X was not refused")
        require(_raises(3, MEMBER_OP_PREDICT, call, "AttributeError: 'SpectralEmbedding' object has no attribute 'predict'"),
                "SpectralEmbedding PREDICT was not refused in scikit-learn's words")
    finally:
        _free(3, call)
    var dataset = _se_words(3, 0)
    var other = MemberCall(DeviceRoute(DEVICE_CPU))
    _new(3, dataset, other)
    try:
        other.x = MemberMatrix(Int(x.unsafe_ptr()), 4, 3)
        require(_raises(3, MEMBER_OP_FIT, other, "needs 4 eigenpairs"),
                "SpectralEmbedding n_components 3 + drop_first on 4 rows was not refused")
    finally:
        _free(3, other)
    _ = len(precomputed)
    _ = len(dataset)
    _ = len(x)


def test_rfe_named_rows() raises:
    require(rfe_named_block(MEMBER_FAMILY_XCLUSTER, 4, "cluster_centers_") == MEMBER_BLOCK_FAMILY,
            "AffinityPropagation cluster_centers_ is not FAMILY+0")
    require(rfe_named_block(MEMBER_FAMILY_XCLUSTER, 2, "cluster_centers_") == MEMBER_BLOCK_FAMILY,
            "MeanShift cluster_centers_ moved")
    require(rfe_named_block(MEMBER_FAMILY_XCLUSTER, 3, "cluster_centers_") == 0, "OPTICS has cluster_centers_")
    require(rfe_named_block(MEMBER_FAMILY_XDECOMP, 0, "mean_") == MEMBER_BLOCK_FAMILY + 2,
            "IncrementalPCA mean_ is not FAMILY+2")
    require(rfe_named_block(MEMBER_FAMILY_XDECOMP, 0, "var_") == MEMBER_BLOCK_FAMILY + 3,
            "IncrementalPCA var_ is not FAMILY+3")
    require(rfe_named_block(MEMBER_FAMILY_XDECOMP, 3, "mean_") == 0, "NMF has mean_")
    # Round b: LDA covariance_, MinCovDet precision_, BayesianGaussianMixture means_, SimpleImputer statistics_,
    # NearestCentroid centroids_ / within_class_std_dev_ / deviations_.
    require(rfe_named_block(MEMBER_FAMILY_DISCRIMINANT, 0, "covariance_") == MEMBER_BLOCK_FAMILY + 7, "LDA covariance_")
    require(rfe_named_block(MEMBER_FAMILY_DISCRIMINANT, 1, "covariance_") == 0, "QDA has a 2-D covariance_")
    require(rfe_named_block(MEMBER_FAMILY_XDECOMP, 14, "precision_") == MEMBER_BLOCK_FAMILY + 1, "MinCovDet precision_")
    require(rfe_named_block(MEMBER_FAMILY_XCLUSTER, 5, "means_") == MEMBER_BLOCK_FAMILY + 9, "BGMM means_")
    require(rfe_named_block(MEMBER_FAMILY_PREP, 7, "statistics_") == MEMBER_BLOCK_FAMILY, "SimpleImputer statistics_")
    require(rfe_named_block(MEMBER_FAMILY_XNEIGHBORS, 0, "centroids_") == MEMBER_BLOCK_FAMILY, "NearestCentroid centroids_")
    require(rfe_named_block(MEMBER_FAMILY_XNEIGHBORS, 0, "within_class_std_dev_") == MEMBER_BLOCK_FAMILY + 1,
            "NearestCentroid within_class_std_dev_")
    require(rfe_named_block(MEMBER_FAMILY_XNEIGHBORS, 0, "deviations_") == MEMBER_BLOCK_FAMILY + 2,
            "NearestCentroid deviations_")


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
