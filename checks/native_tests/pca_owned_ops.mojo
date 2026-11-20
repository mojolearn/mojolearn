# SPDX-License-Identifier: Apache-2.0
"""The PCA owned-fit table (core/decomp_entry_table_pca.mojo), the one entry
the TSNE owner's 'pca' start (bindings/mani_tsne_native.mojo) calls, with
expectations known by construction (all variance on the first feature, a
non-finite X), not by another call into the implementation. No Python
objects.
"""
from core.native_descriptor_abi import NativeDescriptorCall
from core.entry_table import entry_table_same_call
from core.native_descriptor_call import native_descriptor_call
from core.neighbors_entry_table import NeighborsEntryKey, neighbors_vendor_word
from core.decomp_entry_table_pca import PCA_OP_FIT_OWNED, pca_ops_publish, pca_ops_entry, pca_fit_owned_call
from checks.numerics import GLOBAL_NUMERIC_MODE
from core.host_parallel import host_parallelize
from decomposition.pca_ops_entry_host import pca_ops_host_v1, PCA_OPS_HOST_API
from checks.vendor import COMPILED_VENDOR
from decomposition.pca_ops_entry_device import pca_ops_device_v1, PCA_OPS_DEVICE_API
from checks.native_tests.runner import CPU_COLUMN_BUILD, Check, Column, Report, require


comptime PCA_OWNED_CASES = 3


def _f(values: List[Float32]) -> Int:
    return Int(values.unsafe_ptr())


def _near(a: Float64, b: Float64, tolerance: Float64) -> Bool:
    var gap = a - b
    return gap <= tolerance and gap >= -tolerance


def case_fit_first_axis(entry: NativeDescriptorCall) raises:
    # X = [[-1, 0], [1, 0], [-2, 0], [2, 0]]: mean 0, every variance on the
    # first feature (ddof 1: 10 / 3); one component, +-[1, 0]; the
    # projection is +-X's first column.
    var x: List[Float32] = [-1.0, 0.0, 1.0, 0.0, -2.0, 0.0, 2.0, 0.0]
    var components = List[Float32](length=2, fill=-7.0)
    var mean = List[Float32](length=2, fill=-7.0)
    var variance = List[Float32](length=1, fill=-7.0)
    var ratio = List[Float32](length=1, fill=-7.0)
    var singular = List[Float32](length=1, fill=-7.0)
    var z = List[Float32](length=4, fill=-7.0)
    var noise = List[Float64](length=1, fill=-7.0)
    var status = List[Int32](length=1, fill=-7)
    var kept = pca_fit_owned_call(entry, _f(x), _f(components), _f(mean), _f(variance), _f(ratio), _f(singular),
                                  _f(z), Int(noise.unsafe_ptr()), Int(status.unsafe_ptr()), 4, 2, 1, False, 0,
                                  False, 0.0)
    require(status[0] == 0, "pca: a finite X was flagged " + String(status[0]))
    require(kept == 1, "pca: kept " + String(kept))
    require(mean[0] == 0.0 and mean[1] == 0.0, "pca: mean differs from zero")
    var c0 = Float64(components[0])
    require(_near(c0 * c0, 1.0, 1e-5) and _near(Float64(components[1]), 0.0, 1e-5),
            "pca: the component is not +-[1, 0]")
    require(_near(Float64(variance[0]), 10.0 / 3.0, 1e-4), "pca: explained variance " + String(variance[0]))
    require(_near(Float64(ratio[0]), 1.0, 1e-5), "pca: explained ratio " + String(ratio[0]))
    for i in range(4):  # small-loop(no: the four fixture rows): test expectation
        require(_near(Float64(z[i]) * c0, Float64(x[2 * i]), 1e-4), "pca: projection row " + String(i))
    _ = x
    _ = components
    _ = mean
    _ = variance
    _ = ratio
    _ = singular
    _ = z
    _ = noise
    _ = status


def case_refuses_non_finite(entry: NativeDescriptorCall) raises:
    # A NaN in X with the status word asked: a nonzero status and kept 0.
    var x: List[Float32] = [-1.0, 0.0, 1.0, 0.0, -2.0, 0.0, 2.0, 0.0]
    x[5] = Float32(0.0) / Float32(0.0)
    var components = List[Float32](length=2, fill=-7.0)
    var mean = List[Float32](length=2, fill=-7.0)
    var variance = List[Float32](length=1, fill=-7.0)
    var ratio = List[Float32](length=1, fill=-7.0)
    var singular = List[Float32](length=1, fill=-7.0)
    var z = List[Float32](length=4, fill=-7.0)
    var noise = List[Float64](length=1, fill=-7.0)
    var status = List[Int32](length=1, fill=0)
    var kept = pca_fit_owned_call(entry, _f(x), _f(components), _f(mean), _f(variance), _f(ratio), _f(singular),
                                  _f(z), Int(noise.unsafe_ptr()), Int(status.unsafe_ptr()), 4, 2, 1, False, 0,
                                  False, 0.0)
    require(status[0] != 0, "pca: a NaN in X was not flagged")
    require(kept == 0, "pca: a flagged X kept " + String(kept))
    _ = x
    _ = components
    _ = mean
    _ = variance
    _ = ratio
    _ = singular
    _ = z
    _ = noise
    _ = status


def case_refuses_bad_spans(entry: NativeDescriptorCall) raises:
    var addrs: List[Int] = [8, 8, 8, 8, 8, 8, 8, 8]
    var words: List[Int] = [4, 2, 1, 0, 0, 0]
    var floats: List[Float64] = [0.0]
    var short = False
    try:
        _ = native_descriptor_call(entry, PCA_OP_FIT_OWNED, addrs, words, floats)
    except:
        short = True
    require(short, "spans: an eight-address request was not refused")


def run_case(entry: NativeDescriptorCall, case_id: Int) raises:
    if case_id == 0:
        case_fit_first_axis(entry)
    elif case_id == 1:
        case_refuses_non_finite(entry)
    elif case_id == 2:
        case_refuses_bad_spans(entry)
    else:
        raise Error("pca ops: unknown case " + String(case_id))


def check_pca_table(key: NeighborsEntryKey, publisher: Int, entry: NativeDescriptorCall) raises:
    """Publication then lookup returns the published pointer; a repeated
    identical publication is accepted; another tier refuses. Run first, in
    a fresh process (it fills the process slot)."""
    var missing = False
    try:
        _ = pca_ops_entry(key)
    except:
        missing = True
    require(missing, "pca table: lookup before publication did not refuse")
    pca_ops_publish(key, publisher, entry)
    pca_ops_publish(key, publisher, entry)
    require(entry_table_same_call(pca_ops_entry(key), entry), "pca table: lookup returned another pointer")
    var other = False
    try:
        _ = pca_ops_entry(NeighborsEntryKey(1 - key.mode, key.vendor))
    except:
        other = True
    require(other, "pca table: the other tier's unpublished slot did not refuse")


# ---- The host column (was pca_owned_ops_cpu_suite.mojo) ----
# The CPU column of the PCA owned-fit table: the table round trip first
# (ordered metadata), then the independent entry cases on the existing host
# pool.
def _run_cpu(mut report: Report) raises:
    var mode = Int(GLOBAL_NUMERIC_MODE)
    var api = PCA_OPS_HOST_API.get_or_create_ptr()
    check_pca_table(NeighborsEntryKey(mode, 0), Int(api[].entries.unsafe_ptr()), pca_ops_host_v1)
    var total = PCA_OWNED_CASES
    var cases = report.cases(total)
    def task(case_id: Int) {imm cases}:
        try:
            run_case(pca_ops_host_v1, case_id)
        except error:
            cases.fail(case_id, error)
    report.parallel(task, total)


# ---- The device column (was pca_owned_ops_gpu_suite.mojo) ----
# The device column of the PCA owned-fit table. The entry shares the
# estimators binding's device context, so the cases submit in order; the
# kernels keep their native parallelism (no concurrency claim).
def _run_gpu(mut report: Report) raises:
    var mode = Int(GLOBAL_NUMERIC_MODE)
    var vendor = neighbors_vendor_word(COMPILED_VENDOR)
    require(vendor > 0, "pca_owned_ops_gpu: this build has no GPU vendor")
    var api = PCA_OPS_DEVICE_API.get_or_create_ptr()
    check_pca_table(NeighborsEntryKey(mode, vendor), Int(api[].entries.unsafe_ptr()), pca_ops_device_v1)
    var cases = report.cases(PCA_OWNED_CASES)
    for case_id in range(PCA_OWNED_CASES):  # driver-loop: over the ordered device cases
        try:
            run_case(pca_ops_device_v1, case_id)
        except error:
            cases.fail(case_id, error)


def run_cases(column: Column, mut report: Report):
    try:
        comptime if CPU_COLUMN_BUILD:
            _run_cpu(report)
        else:
            _run_gpu(report)
    except error:
        report.abort(error)


def register() -> Check:
    return Check("pca_owned_ops", cpu=True, gpu=True, serial="", run=run_cases)
