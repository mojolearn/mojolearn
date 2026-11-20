# SPDX-License-Identifier: Apache-2.0
"""ANN search door RUN (`x_ann/neighbors_ann_query_run.mojo`) without a GPU.

Each fixture is one typed check, the seed fold, the filter preparation, the
search plan of one kind or the export address order: accepted and refused
values, the refusal wording and the request fields. Independent fixtures run
across the host's cores.
"""
from core.host_parallel import host_parallelize
from x_ann.neighbors_ann_query_run import (
    ANN_SEARCH_PQ, ANN_SEARCH_SQ, ANN_SEARCH_RABITQ, ANN_SEARCH_CAGRA, AnnSearchRequest, ann_search_owner,
    ann_search_entry, ann_check_search_dims, ann_check_query_features, cagra_check_search, ivf_check_search,
    pq_check_search, pq_check_code_dims, cagra_seed_word, ann_check_stored_shape, ann_check_stored_strides,
    ann_mask_refusal, ann_mask_fill_ones, ann_mask_from_bytes, ann_search_plan, ann_handle_addresses,
    ann_full_addresses, ann_filter_params,
)

comptime NEIGHBORS_ANN_QUERY_RUN_FIXTURES = 16


def require(ok: Bool, message: String) raises:
    if not ok:
        raise Error(message)


def same(values: List[Int], expected: List[Int]) -> Bool:
    if len(values) != len(expected):
        return False
    for i in range(len(values)):
        if values[i] != expected[i]:
            return False
    return True


def request(kind: Int, filtered: Bool, device_codes: Bool) -> AnnSearchRequest:
    # n=100, dim=10, k=5, m=3, status word 77; IVF: 8 lists, 2 probes,
    # PQ 4 x 6 bits (length 3), seed 9; Cagra: degree 16, itopk 64, width 2.
    return AnnSearchRequest(kind, 100, 10, 5, 3, 77, filtered, device_codes, 8, 2, 4, 6, 3, 9, 16, 64, 2, 0, 0)


def fixture(unit: Int) raises:
    if unit == 0:
        require(ann_search_owner(ANN_SEARCH_PQ) == "IVFPQIndex" and ann_search_owner(ANN_SEARCH_SQ) == "IVFSQIndex"
                and ann_search_owner(ANN_SEARCH_RABITQ) == "IVFRaBitQIndex"
                and ann_search_owner(ANN_SEARCH_CAGRA) == "CagraIndex", "owner names")
        require(ann_search_entry(ANN_SEARCH_PQ) == "x_ann_ivf_pq_search"
                and ann_search_entry(ANN_SEARCH_SQ) == "x_ann_ivf_sq_search"
                and ann_search_entry(ANN_SEARCH_RABITQ) == "x_ann_ivf_rabitq_search"
                and ann_search_entry(ANN_SEARCH_CAGRA) == "x_ann_cagra_search", "search entries")
        for kind in [-1, 4]:
            try:
                _ = ann_search_owner(kind)
                raise Error("accepted")
            except e:
                require(String(e) == "RuntimeError: unknown native ANN search kind", "unknown kind wording")
    elif unit == 1:
        ann_check_search_dims(1, 1, 1)
        try:
            ann_check_search_dims(1, 1, 0)
            raise Error("accepted")
        except e:
            require(String(e) == "ValueError: ANN rows, features and n_neighbors must be positive", "dims wording")
        ann_check_query_features("CagraIndex", 4, 4)
        try:
            ann_check_query_features("IVFSQIndex", 3, 4)
            raise Error("accepted")
        except e:
            require(String(e) == "ValueError: mojolearn IVFSQIndex: queries have 3 features, the index has 4",
                    "query features wording")
    elif unit == 2:
        cagra_check_search(1, 1, 1, 0, 0)
        try:
            cagra_check_search(1, 1, 1, -1, 0)
            raise Error("accepted")
        except e:
            require(String(e) == "ValueError: invalid CagraIndex search configuration", "Cagra config wording")
        ivf_check_search(10, 10, 10)
        try:
            ivf_check_search(10, 4, 5)
            raise Error("accepted")
        except e:
            require(String(e) == "ValueError: ANN n_lists must be in [1, n_rows] and n_probes in [1, n_lists]",
                    "IVF search wording")
    elif unit == 3:
        pq_check_search(10, 4, 8, 3)
        try:
            pq_check_search(10, 4, 8, 2)
            raise Error("accepted")
        except e:
            require(String(e) == "ValueError: invalid fitted IVFPQIndex dimensions", "PQ fitted wording")
        pq_check_code_dims(1, 1)
        try:
            pq_check_code_dims(0, 1)
            raise Error("accepted")
        except e:
            require(String(e) == "ValueError: invalid fitted IVFPQIndex code dimensions", "PQ code dims wording")
    elif unit == 4:
        var zero: List[Int] = [0]
        require(cagra_seed_word(zero) == 1, "seed 0")
        var one: List[Int] = [1]
        require(cagra_seed_word(one) == 2, "seed 1")
        # 2^31 - 2 folds to 0, 2^31 - 1 to 1; the bytes are signed big-endian
        var wrap: List[Int] = [0, 127, 255, 255, 254]
        require(cagra_seed_word(wrap) == 1, "seed modulus")
        var past: List[Int] = [0, 127, 255, 255, 255]
        require(cagra_seed_word(past) == 2, "seed past the modulus")
        var negative: List[Int] = [255]
        try:
            _ = cagra_seed_word(negative)
            raise Error("accepted")
        except e:
            require(String(e) == "ValueError: mojolearn CagraIndex: random_state must be None or >= 0", "seed wording")
    elif unit == 5:
        var shape: List[Int] = [3, 4]
        var expected: List[Int] = [3, 4]
        ann_check_stored_shape("centers_", shape, expected)
        var rank1: List[Int] = [12]
        try:
            ann_check_stored_shape("centers_", rank1, expected)
            raise Error("accepted")
        except e:
            require(String(e) == "ValueError: invalid fitted centers_ rank", "rank wording")
        var other: List[Int] = [3, 5]
        try:
            ann_check_stored_shape("centers_", shape, other)
            raise Error("accepted")
        except e:
            require(String(e) == "ValueError: invalid fitted centers_ shape", "shape wording")
    elif unit == 6:
        var shape: List[Int] = [3, 4]
        var c_order: List[Int] = [16, 4]
        ann_check_stored_strides("codes_", shape, c_order, -1)
        # a unit axis may carry any stride
        var column: List[Int] = [1, 4]
        var odd: List[Int] = [999, 4]
        ann_check_stored_strides("codes_", column, odd, -1)
        var fortran: List[Int] = [4, 12]
        try:
            ann_check_stored_strides("codes_", shape, fortran, -1)
            raise Error("accepted")
        except e:
            require(String(e) == "ValueError: fitted codes_ requires C-contiguous storage", "contiguity wording")
    elif unit == 7:
        # the last axis is walked first: its contiguity refusal precedes a
        # non-int stride on axis 0, and a non-int last stride is refused first
        var shape: List[Int] = [3, 4]
        var strides: List[Int] = [0, 8]
        try:
            ann_check_stored_strides("codes_", shape, strides, 0)
            raise Error("accepted")
        except e:
            require(String(e) == "ValueError: fitted codes_ requires C-contiguous storage", "walk order")
        var words: List[Int] = [16, 0]
        try:
            ann_check_stored_strides("codes_", shape, words, 1)
            raise Error("accepted")
        except e:
            require(String(e) == "TypeError: fitted array strides must be exact native ints", "non-int wording")
    elif unit == 8:
        require(ann_mask_refusal("CagraIndex", 5) == "ValueError: mojolearn CagraIndex: filter must be a boolean "
                + "array of shape (5,), one flag per indexed row", "filter refusal wording")
        var dest = List[Int32](length=4, fill=Int32(7))
        ann_mask_fill_ones(Int(dest.unsafe_ptr()), 4)
        require(dest[0] == 1 and dest[1] == 1 and dest[2] == 1 and dest[3] == 1, "all-ones filter")
        _ = dest
    elif unit == 9:
        # every other byte of a strided bool view
        var source: List[UInt8] = [1, 9, 0, 9, 2, 9]
        var dest = List[Int32](length=3, fill=Int32(7))
        ann_mask_from_bytes(Int(dest.unsafe_ptr()), Int(source.unsafe_ptr()), 3, 2)
        require(dest[0] == 1 and dest[1] == 0 and dest[2] == 1, "strided filter flags")
        _ = source
        _ = dest
    elif unit == 10:
        var p = ann_search_plan(request(ANN_SEARCH_PQ, False, False))
        require(len(p.names) == 5 and p.names[3] == "codebooks_" and p.names[4] == "codes_", "PQ schema names")
        var codebooks: List[Int] = [4, 64, 3]
        var codes: List[Int] = [100, 4]
        require(same(p.shapes[3], codebooks) and same(p.shapes[4], codes) and p.floating[3] and not p.floating[4],
                "PQ schema shapes")
        var prepare: List[Int] = [100, 10, 8, 4, 6]
        var full: List[Int] = [100, 10, 8, 4, 6, 3, 5, 2, 77]
        var search: List[Int] = [3, 5, 2, 77]
        require(same(p.prepare_params, prepare) and same(p.full_params, full) and same(p.search_params, search)
                and p.selected == 5, "PQ parameter words")
        var resident = ann_search_plan(request(ANN_SEARCH_PQ, False, True))
        require(len(resident.names) == 4, "device codes are not admitted from the host")
    elif unit == 11:
        var p = ann_search_plan(request(ANN_SEARCH_SQ, False, False))
        require(len(p.names) == 6 and p.names[0] == "centers_" and p.names[3] == "sq_vmin_"
                and p.names[5] == "codes_", "SQ schema names")
        var offsets: List[Int] = [9]
        require(same(p.shapes[1], offsets), "list offsets shape")
        var prepare: List[Int] = [100, 10, 8]
        var full: List[Int] = [100, 10, 8, 3, 5, 2, 77]
        require(same(p.prepare_params, prepare) and same(p.full_params, full), "SQ parameter words")
    elif unit == 12:
        var p = ann_search_plan(request(ANN_SEARCH_RABITQ, False, False))
        var codes: List[Int] = [100, 1]
        require(len(p.names) == 6 and p.names[3] == "codes_" and same(p.shapes[3], codes)
                and p.names[5] == "ip_factors_" and p.floating[5], "RaBitQ schema")
        var prepare: List[Int] = [100, 10, 8, 9]
        var full: List[Int] = [100, 10, 8, 9, 3, 5, 2, 77]
        require(same(p.prepare_params, prepare) and same(p.full_params, full), "RaBitQ parameter words")
    elif unit == 13:
        # iterations 0 -> itopk; seeds 0 -> min(itopk + width * degree, n)
        var p = ann_search_plan(request(ANN_SEARCH_CAGRA, False, False))
        require(len(p.names) == 2 and p.names[0] == "dataset_" and p.names[1] == "graph_", "Cagra schema")
        var search: List[Int] = [3, 5, 64, 2, 64, 96, 9, 77]
        var full: List[Int] = [100, 10, 16, 3, 5, 64, 2, 64, 96, 9, 77]
        var prepare: List[Int] = [100, 10, 16]
        require(p.selected == 5 and same(p.search_params, search) and same(p.full_params, full)
                and same(p.prepare_params, prepare), "Cagra parameter words")
        var r = request(ANN_SEARCH_CAGRA, True, False)
        r.iterations = 7
        r.seeds = 500
        var f = ann_search_plan(r)
        var filtered: List[Int] = [3, 64, 64, 2, 7, 100, 9, 77]
        require(f.selected == 64 and same(f.search_params, filtered), "filtered Cagra keeps itopk candidates")
    elif unit == 14:
        var ivf = ann_handle_addresses(ANN_SEARCH_SQ, 1, 2, 3, 4, 5, True)
        var ivf_expected: List[Int] = [1, 2, 3, 4, 5]
        var plain = ann_handle_addresses(ANN_SEARCH_PQ, 1, 2, 3, 4, 5, False)
        var plain_expected: List[Int] = [1, 2, 3, 4]
        var cagra = ann_handle_addresses(ANN_SEARCH_CAGRA, 1, 2, 3, 4, 5, True)
        var cagra_expected: List[Int] = [1, 2, 3]
        require(same(ivf, ivf_expected) and same(plain, plain_expected) and same(cagra, cagra_expected),
                "resident search addresses")
        var filter_words: List[Int] = [3, 100, 64, 5]
        require(same(ann_filter_params(3, 100, 64, 5), filter_words), "filter parameter words")
    else:
        var stored: List[Int] = [10, 11]
        var pq_expected: List[Int] = [10, 11, 1, 2, 3, 4, 5]
        var sq_expected: List[Int] = [10, 11, 5, 1, 2, 3, 4]
        var cagra_expected: List[Int] = [10, 11, 1, 2, 3]
        require(same(ann_full_addresses(ANN_SEARCH_PQ, stored, 5, 1, 2, 3, 4), pq_expected), "PQ addresses")
        require(same(ann_full_addresses(ANN_SEARCH_RABITQ, stored, 5, 1, 2, 3, 4), sq_expected), "RaBitQ addresses")
        require(same(ann_full_addresses(ANN_SEARCH_CAGRA, stored, 0, 1, 2, 3, 4), cagra_expected), "Cagra addresses")
        require(len(stored) == 2, "stored addresses are copied, not extended")


def run_checks() raises:
    var failures = List[Int](length=NEIGHBORS_ANN_QUERY_RUN_FIXTURES, fill=0)
    var fp = failures.unsafe_ptr()
    def task(unit: Int) {imm fp}:
        try:
            fixture(unit)
        except error:
            fp.unsafe_store(unit, 1)
            print("FAIL neighbors_ann_query_run fixture", unit, String(error))
    host_parallelize(task, NEIGHBORS_ANN_QUERY_RUN_FIXTURES)
    var failed = 0
    for value in failures:
        failed += value
    _ = failures^
    require(failed == 0, "neighbors_ann_query_run failed " + String(failed) + " fixtures")
    print("PASS neighbors_ann_query_run:", NEIGHBORS_ANN_QUERY_RUN_FIXTURES, "typed check, plan and address fixtures")
