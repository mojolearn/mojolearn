# SPDX-License-Identifier: Apache-2.0
"""The IVF-Flat operations table (core/ivf_entry_table.mojo) and its typed RUN
(ivf/ivf_query_run.mojo) without Python, on the CPU column.

The suite publishes the search-only CPU image's entry
(ivf/ops_entry_host.mojo, as `_mojolearn_ivf_search_host` does at its init)
and drives it with plain words over one small hand-built index: the span
checks of the eight operations, the borrowed-handle rule on two image keys,
the life of a prepared index (prepare, search by handle and by arrays with
the same words out, release, a released handle refused), the operations a CPU
image refuses, and extend. Independent fixtures run across the host's cores;
the handle registry is locked. The build and the GPU column are covered by
the installed-wheel IVF tests and the plan suites.
"""
from checks.numerics import GLOBAL_NUMERIC_MODE
from core.host_parallel import host_parallelize
from core.ivf_entry_table import (
    IVF_IMAGE_FULL, IVF_IMAGE_SEARCH, IVF_OPERATION_COUNT, IVF_OP_BUILD, IVF_OP_BUILD_RESIDENT, IVF_OP_EXTEND,
    IVF_OP_INDEX_EXPORT, IVF_OP_INDEX_PREPARE, IVF_OP_INDEX_RELEASE, IVF_OP_INDEX_SEARCH, IVF_OP_SEARCH,
    IvfEntryKey, IvfIndexSpans, IvfIndexWords, ivf_ops_check_spans, ivf_ops_entry,
)
from ivf.ivf_query_run import (
    IVF_NO_HANDLE, IvfBuildRequest, IvfExtendRequest, IvfSearchRequest, ivf_borrowed, ivf_build_run,
    ivf_export_run, ivf_extend_run, ivf_prepare_run, ivf_release_run, ivf_search_run,
)
from ivf.ops_entry_host import publish_ivf_ops_search_host

comptime IVF_OPS_TABLE_FIXTURES = 5


def require(ok: Bool, message: String) raises:
    if not ok:
        raise Error(message)


def _key() -> IvfEntryKey:
    return IvfEntryKey(Int(GLOBAL_NUMERIC_MODE), 0, IVF_IMAGE_SEARCH)


def _spans_refused(operation: Int, na: Int, np: Int, nf: Int, has_result: Bool) raises -> Bool:
    var words = List[Int](length=16, fill=0)
    var result = 0
    var refused = False
    try:
        ivf_ops_check_spans(operation, Int(words.unsafe_ptr()), na, Int(words.unsafe_ptr()), np, nf,
                            Int(MutPointer(to=result)) if has_result else 0)
    except:
        refused = True
    _ = words^
    return refused


struct TinyIndex(Movable):
    """Four rows in two features, two lists: rows 0 and 1 near (0, 0), rows
    2 and 3 near (10, 10)."""
    var centers: List[Float32]
    var norms: List[Float32]
    var offsets: List[Int32]
    var indices: List[Int32]
    var data: List[Float32]

    def __init__(out self):
        self.centers = List[Float32](length=4, fill=Float32(0.0))
        self.centers[2] = Float32(10.0)
        self.centers[3] = Float32(10.0)
        self.norms = List[Float32](length=2, fill=Float32(0.0))
        self.norms[1] = Float32(200.0)
        self.offsets = List[Int32](length=3, fill=Int32(0))
        self.offsets[1] = Int32(2)
        self.offsets[2] = Int32(4)
        self.indices = List[Int32](length=4, fill=Int32(0))
        self.indices[1] = Int32(1)
        self.indices[2] = Int32(2)
        self.indices[3] = Int32(3)
        self.data = List[Float32](length=8, fill=Float32(0.0))
        self.data[2] = Float32(1.0)
        self.data[4] = Float32(10.0)
        self.data[5] = Float32(10.0)
        self.data[6] = Float32(11.0)
        self.data[7] = Float32(10.0)

    def spans(self) -> IvfIndexSpans:
        return IvfIndexSpans(Int(self.centers.unsafe_ptr()), Int(self.norms.unsafe_ptr()),
                             Int(self.offsets.unsafe_ptr()), Int(self.indices.unsafe_ptr()),
                             Int(self.data.unsafe_ptr()))

    def words(self) -> IvfIndexWords:
        return IvfIndexWords(4, 2, 2, 0)


def _queries() -> List[Float32]:
    var q = List[Float32](length=4, fill=Float32(0.0))
    q[0] = Float32(0.25)
    q[2] = Float32(10.5)
    q[3] = Float32(10.0)
    return q^


def fixture(unit: Int) raises:
    if unit == 0:
        # every operation's layout is accepted; a wrong count, a float span,
        # a null result word and an unknown operation are refused
        require(not _spans_refused(IVF_OP_BUILD, 6, 6, 0, True), "build spans")
        require(not _spans_refused(IVF_OP_BUILD_RESIDENT, 5, 6, 0, True), "resident build spans")
        require(not _spans_refused(IVF_OP_SEARCH, 10, 8, 0, True), "search spans")
        require(not _spans_refused(IVF_OP_INDEX_PREPARE, 5, 5, 0, True), "prepare spans")
        require(not _spans_refused(IVF_OP_INDEX_SEARCH, 5, 9, 0, True), "index search spans")
        require(not _spans_refused(IVF_OP_INDEX_RELEASE, 0, 1, 0, True), "release spans")
        require(not _spans_refused(IVF_OP_INDEX_EXPORT, 1, 2, 0, True), "export spans")
        require(not _spans_refused(IVF_OP_EXTEND, 10, 5, 0, True), "extend spans")
        require(_spans_refused(IVF_OP_SEARCH, 9, 8, 0, True), "search with 9 addresses")
        require(_spans_refused(IVF_OP_INDEX_SEARCH, 5, 8, 0, True), "index search with 8 words")
        require(_spans_refused(IVF_OP_INDEX_RELEASE, 0, 1, 1, True), "a float span")
        require(_spans_refused(IVF_OP_INDEX_RELEASE, 0, 1, 0, False), "a null result word")
        require(_spans_refused(IVF_OPERATION_COUNT, 0, 0, 0, True), "an unknown operation")
        require(_spans_refused(-1, 0, 0, 0, True), "a negative operation")
    elif unit == 1:
        # a handle is borrowed across DEVICE columns only
        var cpu = IvfEntryKey(1, 0, IVF_IMAGE_FULL)
        var cpu_search = IvfEntryKey(1, 0, IVF_IMAGE_SEARCH)
        var cuda = IvfEntryKey(1, 2, IVF_IMAGE_FULL)
        var cuda_fast = IvfEntryKey(0, 2, IVF_IMAGE_FULL)
        require(ivf_borrowed(cuda, cpu) and ivf_borrowed(cpu, cuda), "across device columns")
        require(ivf_borrowed(cuda_fast, cpu_search), "across device columns, other tier and image")
        require(not ivf_borrowed(cpu, cpu_search) and not ivf_borrowed(cpu_search, cpu), "the two CPU images")
        require(not ivf_borrowed(cuda, cuda_fast) and not ivf_borrowed(cuda, cuda), "one device")
        require(cuda.on_device() and not cpu.on_device(), "device column")
        require(cpu.builds() and cuda.builds() and not cpu_search.builds(), "the search-only image has no build")
        require(cpu.same(cpu) and not cpu.same(cpu_search) and not cuda.same(cuda_fast), "key identity")
    elif unit == 2:
        # a prepared index answers as the arrays do; released, it is refused
        var key = _key()
        var index = TinyIndex()
        var q = _queries()
        var d_arrays = List[Float32](length=4, fill=Float32(-1.0))
        var i_arrays = List[Int32](length=4, fill=Int32(-7))
        var c_arrays = List[Int32](length=2, fill=Int32(-7))
        ivf_search_run(key, IvfSearchRequest(
            index.words(), IVF_NO_HANDLE, index.spans(), Int(q.unsafe_ptr()), Int(d_arrays.unsafe_ptr()),
            Int(i_arrays.unsafe_ptr()), Int(c_arrays.unsafe_ptr()), 0, 2, 2, 2, False))
        require(i_arrays[0] == 0 and i_arrays[1] == 1 and i_arrays[2] == 2 and i_arrays[3] == 3, "nearest ids")
        require(c_arrays[0] >= 2 and c_arrays[1] >= 2, "candidate counts are written")
        var handle = ivf_prepare_run(key, index.spans(), index.words(), False)
        require(handle >= 1 and handle != IVF_NO_HANDLE, "a minted handle")
        var d_handle = List[Float32](length=4, fill=Float32(-1.0))
        var i_handle = List[Int32](length=4, fill=Int32(-7))
        var c_handle = List[Int32](length=2, fill=Int32(-7))
        var by_handle = IvfSearchRequest(
            index.words(), handle, IvfIndexSpans(0, 0, 0, 0, 0), Int(q.unsafe_ptr()), Int(d_handle.unsafe_ptr()),
            Int(i_handle.unsafe_ptr()), Int(c_handle.unsafe_ptr()), 0, 2, 2, 2, False)
        ivf_search_run(key, by_handle)
        for slot in range(4):
            require(d_handle[slot] == d_arrays[slot] and i_handle[slot] == i_arrays[slot], "same words by handle")
        require(c_handle[0] == c_arrays[0] and c_handle[1] == c_arrays[1], "same candidate counts by handle")
        # the handle names another index than the call
        var other = IvfSearchRequest(
            IvfIndexWords(5, 2, 2, 0), handle, IvfIndexSpans(0, 0, 0, 0, 0), Int(q.unsafe_ptr()),
            Int(d_handle.unsafe_ptr()), Int(i_handle.unsafe_ptr()), Int(c_handle.unsafe_ptr()), 0, 2, 2, 2, False)
        var refused = False
        try:
            ivf_search_run(key, other)
        except:
            refused = True
        require(refused, "a handle of another index is refused")
        # a sample filter removes row 0 from the first query's answer
        var keep = List[Int32](length=4, fill=Int32(1))
        keep[0] = Int32(0)
        var filtered = IvfSearchRequest(
            index.words(), handle, IvfIndexSpans(0, 0, 0, 0, 0), Int(q.unsafe_ptr()), Int(d_handle.unsafe_ptr()),
            Int(i_handle.unsafe_ptr()), Int(c_handle.unsafe_ptr()), Int(keep.unsafe_ptr()), 2, 2, 2, False)
        ivf_search_run(key, filtered)
        require(i_handle[0] == 1, "the filtered row is not returned")
        ivf_release_run(key, handle)
        refused = False
        try:
            ivf_search_run(key, by_handle)
        except:
            refused = True
        require(refused, "a released handle is refused by a search")
        refused = False
        try:
            ivf_release_run(key, handle)
        except:
            refused = True
        require(refused, "a released handle is refused by a second release")
        _ = keep^
        _ = d_arrays^
        _ = i_arrays^
        _ = c_arrays^
        _ = d_handle^
        _ = i_handle^
        _ = c_handle^
        _ = q^
        _ = index^
    elif unit == 3:
        # what a CPU image does not carry is refused in words, and an image
        # that never published has no entry
        var key = _key()
        var index = TinyIndex()
        var x = List[Float32](length=8, fill=Float32(0.0))
        var refused = False
        try:
            _ = ivf_build_run(key, IvfBuildRequest(Int(x.unsafe_ptr()), index.spans(), index.words(), 2, 0))
        except:
            refused = True
        require(refused, "the search-only image refuses the build")
        refused = False
        try:
            ivf_export_run(key, 1, Int(x.unsafe_ptr()), 8)
        except:
            refused = True
        require(refused, "a CPU image refuses the device list-data export")
        refused = False
        try:
            _ = ivf_ops_entry(IvfEntryKey(Int(GLOBAL_NUMERIC_MODE), 0, 7))
        except:
            refused = True
        require(refused, "an unknown image word has no table")
        refused = False
        try:
            _ = ivf_prepare_run(key, index.spans(), IvfIndexWords(4, 2, 0, 0), False)
        except:
            refused = True
        require(refused, "an index with no list is refused before any buffer is read")
        _ = x^
        _ = index^
    else:
        # extend: one new row near the first centre joins the first list
        var key = _key()
        var index = TinyIndex()
        var new_x = List[Float32](length=2, fill=Float32(0.5))
        var offsets = List[Int32](length=3, fill=Int32(-1))
        var indices = List[Int32](length=5, fill=Int32(-1))
        var data = List[Float32](length=10, fill=Float32(-1.0))
        var labels = List[Int32](length=1, fill=Int32(-1))
        ivf_extend_run(key, IvfExtendRequest(
            index.words(), index.spans(), Int(new_x.unsafe_ptr()), 1, Int(offsets.unsafe_ptr()),
            Int(indices.unsafe_ptr()), Int(data.unsafe_ptr()), Int(labels.unsafe_ptr())))
        require(offsets[0] == 0 and offsets[1] == 3 and offsets[2] == 5, "extended offsets")
        require(labels[0] == 0, "the new row's list")
        require(indices[0] == 4 or indices[1] == 4 or indices[2] == 4, "the new row's id is in the first list")
        var refused = False
        try:
            ivf_extend_run(key, IvfExtendRequest(
                index.words(), index.spans(), Int(new_x.unsafe_ptr()), 0, Int(offsets.unsafe_ptr()),
                Int(indices.unsafe_ptr()), Int(data.unsafe_ptr()), Int(labels.unsafe_ptr())))
        except:
            refused = True
        require(refused, "an extend by no row is refused")
        _ = new_x^
        _ = offsets^
        _ = indices^
        _ = data^
        _ = labels^
        _ = index^


def run_checks() raises:
    # The search-only CPU image's entry, published once as its init does.
    var published = publish_ivf_ops_search_host()
    require(published.same(_key()), "the published key")
    var failures = List[Int](length=IVF_OPS_TABLE_FIXTURES, fill=0)
    var fp = failures.unsafe_ptr()
    def task(unit: Int) {imm fp}:
        try:
            fixture(unit)
        except error:
            fp.unsafe_store(unit, 1)
            print("FAIL ivf_ops_table fixture", unit, String(error))
    host_parallelize(task, IVF_OPS_TABLE_FIXTURES)
    var failed = 0
    for value in failures:
        failed += value
    _ = failures^
    require(failed == 0, "ivf_ops_table failed " + String(failed) + " fixtures")
    print("PASS ivf_ops_table:", IVF_OPS_TABLE_FIXTURES, "table, typed request and handle-lifetime fixtures")
