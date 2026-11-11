# SPDX-License-Identifier: Apache-2.0
"""ANN fit/refine door RUN (`x_ann/neighbors_ann_run.mojo`) without a GPU.

Each fixture is one typed check, size plan or the candidate narrowing:
accepted and refused values, the refusal wording and the plan fields.
Independent fixtures run across the host's cores.
"""
from core.host_parallel import host_parallelize
from x_ann.neighbors_ann_run import (
    ann_check_dimension, ann_shape_size, ann_c_strides, ivf_metric_code, ivf_check_n_lists, ivf_fit_plan,
    pq_fit_plan, ann_check_n_lists, sq_check_train_per_list, rabitq_padded_dim, rabitq_code_words,
    cagra_degrees, cagra_copy_dataset, refine_metric_root, refine_check_features, refine_candidates_refusal,
    refine_candidates_k0, refine_check_candidate_kind, refine_narrow_ids, refine_check_status,
)

comptime NEIGHBORS_ANN_RUN_FIXTURES = 16


def require(ok: Bool, message: String) raises:
    if not ok:
        raise Error(message)


def refused(message: String, prefix: String) -> Bool:
    return message.startswith(prefix)


def fixture(unit: Int) raises:
    if unit == 0:
        ann_check_dimension(0)
        ann_check_dimension(7)
        try:
            ann_check_dimension(-1)
            raise Error("accepted")
        except e:
            require(String(e) == "mojolearn: negative dimension in shape (-1,)", "negative dimension wording")
    elif unit == 1:
        var shape: List[Int] = [2, 3, 4]
        require(ann_shape_size(shape) == 24, "shape size")
        var strides = ann_c_strides(shape, 4)
        require(len(strides) == 3 and strides[0] == 48 and strides[1] == 16 and strides[2] == 4, "C strides")
        var scalar = List[Int]()
        require(ann_shape_size(scalar) == 1 and len(ann_c_strides(scalar, 4)) == 0, "rank-0 layout")
    elif unit == 2:
        require(ivf_metric_code(0.0, "0") == 0 and ivf_metric_code(1.0, "1.0") == 1, "metric codes")
        try:
            _ = ivf_metric_code(2.0, "2")
            raise Error("accepted")
        except e:
            require(String(e) == "mojolearn IVFIndex: the metric codes are 0 (L2Expanded) and 1 "
                    + "(L2SqrtExpanded), got 2", "metric code wording")
    elif unit == 3:
        ivf_check_n_lists(1)
        try:
            ivf_check_n_lists(0)
            raise Error("accepted")
        except e:
            require(String(e) == "mojolearn IVFIndex: n_lists must be at least 1, got 0", "IVF n_lists wording")
    elif unit == 4:
        var p = ivf_fit_plan(10, 3, 4)
        require(p.centers == 12 and p.norms == 4 and p.offsets == 5 and p.indices == 10 and p.data == 30,
                "IVF fit plan sizes")
    elif unit == 5:
        var p = pq_fit_plan(100, 10, 4, 3, 8)
        require(p.pq_len == 4 and p.n_codes == 256, "PQ sub-vector length and codebook size")
        require(p.centers == 40 and p.offsets == 5 and p.indices == 100 and p.codebooks == 3 * 256 * 4
                and p.codes == 300, "PQ fit plan sizes")
    elif unit == 6:
        # pq_dim is refused before pq_bits and n_lists
        try:
            _ = pq_fit_plan(100, 10, 0, 11, 9)
            raise Error("accepted")
        except e:
            require(String(e) == "mojolearn IVFPQIndex: pq_dim must be in [1, 10], got 11", "pq_dim wording")
        try:
            _ = pq_fit_plan(100, 10, 0, 2, 9)
            raise Error("accepted")
        except e:
            require(String(e) == "mojolearn IVFPQIndex: pq_bits must be in [1, 8], got 9", "pq_bits wording")
        try:
            _ = pq_fit_plan(100, 10, 101, 2, 8)
            raise Error("accepted")
        except e:
            require(String(e) == "mojolearn IVFPQIndex: n_lists must be in [1, 100], got 101", "PQ n_lists wording")
    elif unit == 7:
        ann_check_n_lists("IVFSQIndex", 5, 5)
        try:
            ann_check_n_lists("IVFRaBitQIndex", 5, 6)
            raise Error("accepted")
        except e:
            require(String(e) == "mojolearn IVFRaBitQIndex: n_lists must be in [1, 5], got 6", "n_lists wording")
        sq_check_train_per_list(1)
        try:
            sq_check_train_per_list(0)
            raise Error("accepted")
        except e:
            require(String(e) == "mojolearn IVFSQIndex: max_train_points_per_cluster must be positive or None, got 0",
                    "train rows wording")
    elif unit == 8:
        require(rabitq_padded_dim(1) == 1 and rabitq_padded_dim(5) == 8 and rabitq_padded_dim(64) == 64,
                "power-of-two padding")
        require(rabitq_code_words(1) == 1 and rabitq_code_words(64) == 2 and rabitq_code_words(65) == 4,
                "RaBitQ code words")
    elif unit == 9:
        var small = cagra_degrees(10, 32, 64)
        require(small[0] == 9 and small[1] == 9, "degrees capped by n - 1")
        var large = cagra_degrees(100, 32, 16)
        require(large[0] == 32 and large[1] == 16, "degrees as named")
    elif unit == 10:
        var src: List[Float32] = [1.0, -2.5, 3.0, 4.25]
        var dst = List[Float32](length=4, fill=0.0)
        cagra_copy_dataset(Int(dst.unsafe_ptr()), Int(src.unsafe_ptr()), 4)
        require(dst[0] == 1.0 and dst[1] == -2.5 and dst[2] == 3.0 and dst[3] == 4.25, "dataset snapshot")
        _ = src
        _ = dst
    elif unit == 11:
        require(not refine_metric_root("sqeuclidean", "'sqeuclidean'"), "sqeuclidean keeps squares")
        require(refine_metric_root("euclidean", "'euclidean'"), "euclidean takes roots")
        try:
            _ = refine_metric_root("cosine", "'cosine'")
            raise Error("accepted")
        except e:
            require(String(e) == "mojolearn refine: metric must be 'sqeuclidean' or 'euclidean' "
                    + "(inner product is not implemented), got 'cosine'", "refine metric wording")
    elif unit == 12:
        refine_check_features(3, 3)
        try:
            refine_check_features(2, 3)
            raise Error("accepted")
        except e:
            require(String(e) == "mojolearn refine: queries have 2 features, the dataset has 3", "features wording")
        var shape: List[Int] = [4, 7]
        require(refine_candidates_k0(shape, 4) == 7, "candidate column count")
        refine_check_candidate_kind(True, 4)
        var wording = String("mojolearn refine: candidates must be an integer array of shape (4, k0)")
        require(refine_candidates_refusal(4) == wording, "candidates refusal wording")
        try:
            var rank1: List[Int] = [4]
            _ = refine_candidates_k0(rank1, 4)
            raise Error("accepted")
        except e:
            require(String(e) == wording, "candidate rank refusal")
        try:
            _ = refine_candidates_k0(shape, 5)
            raise Error("accepted")
        except e:
            require(String(e) == refine_candidates_refusal(5), "candidate rows refusal")
        try:
            refine_check_candidate_kind(False, 4)
            raise Error("accepted")
        except e:
            require(String(e) == wording, "candidate kind refusal")
    elif unit == 13:
        # (2, 3) int64 ids, C order: byte strides 24 and 8
        var ids: List[Int64] = [0, 1, 2, -1, 5, 2147483647]
        var dst = List[Int32](length=6, fill=Int32(-7))
        refine_narrow_ids(Int(ids.unsafe_ptr()), 24, 8, 2, 3, 8, True, False, Int(dst.unsafe_ptr()))
        require(dst[0] == 0 and dst[2] == 2 and dst[3] == -1 and dst[5] == 2147483647, "ids narrowed")
        # an empty candidate block is not read
        refine_narrow_ids(0, 0, 0, 0, 3, 8, True, False, 0)
        _ = ids
        _ = dst
    elif unit == 14:
        var ids: List[Int64] = [0, 2147483648]
        var dst = List[Int32](length=2, fill=Int32(0))
        try:
            refine_narrow_ids(Int(ids.unsafe_ptr()), 16, 8, 1, 2, 8, True, False, Int(dst.unsafe_ptr()))
            raise Error("accepted")
        except e:
            require(String(e) == "mojolearn refine: a candidate index exceeds int32", "int32 refusal wording")
        _ = ids
        _ = dst
    else:
        refine_check_status(0)
        try:
            refine_check_status(1)
            raise Error("accepted")
        except e:
            require(String(e) == "Input dataset or queries contains NaN or infinity (refine).", "status wording")


def run_checks() raises:
    var failures = List[Int](length=NEIGHBORS_ANN_RUN_FIXTURES, fill=0)
    var fp = failures.unsafe_ptr()
    def task(unit: Int) {imm fp}:
        try:
            fixture(unit)
        except error:
            fp.unsafe_store(unit, 1)
            print("FAIL neighbors_ann_run fixture", unit, String(error))
    host_parallelize(task, NEIGHBORS_ANN_RUN_FIXTURES)
    var failed = 0
    for value in failures:
        failed += value
    _ = failures^
    require(failed == 0, "neighbors_ann_run failed " + String(failed) + " fixtures")
    print("PASS neighbors_ann_run:", NEIGHBORS_ANN_RUN_FIXTURES, "typed check, plan and narrowing fixtures")
