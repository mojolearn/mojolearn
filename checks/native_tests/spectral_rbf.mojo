# SPDX-License-Identifier: Apache-2.0
"""Supplementary RBF numerical and ordered native-resource ownership checks.

Original Spectral files expose RBF boundary acceptance, not a numerical oracle.
These checks add independent formula coverage and do not claim a Python test
deletion. Native callback records are real owned storage pinned through unwind.
"""
from core.host_parallel import host_parallelize
from core.native_descriptor_abi import NativeDescriptorCall, NativeDescriptorError
from core.native_descriptor_call import native_descriptor_report
from core.spectral_affinity_plan import SpectralRBFRequest, SpectralRBFCells, spectral_rbf_affinity
from x_decomp.descriptor_schema import XD_DEV_ALLOC, XD_DEV_FREE
from checks.native_tests.spectral_native import source, source_unchanged, guarded, finite
from checks.native_tests.spectral_dense import clear
from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_IDENTICAL
from x_decomp.descriptor_host import x_decomp_host_v1
from x_decomp.descriptor_device import x_decomp_device_v1
from checks.native_tests.runner import CPU_COLUMN_BUILD, Check, Column, Report, require


comptime SPECTRAL_RBF_OWNERSHIP_FIXTURES = 4
comptime SPECTRAL_RBF_NUMERICAL_FIXTURES = 2


def resource_entry(operation: Int, aa: Int, na: Int, pp: Int, np: Int, ff: Int, nf: Int,
                   result: Int, context: Int, sink: NativeDescriptorError) abi("C") -> Int:
    try:
        require(na == 0 and np == 1 and nf == 0 and pp != 0 and result != 0,
            "RBF native allocation/free envelope")
        var handle = Pointer[Int, ImmUntrackedOrigin](unsafe_from_address=pp)[0]
        # Test allocation size is a real record address; no actual device
        # allocator is called. Each record owns a stats pointer, tag and flags.
        var record = Pointer[Int, ImmUntrackedOrigin](unsafe_from_address=handle)
        var stats = MutPointer[Int, MutUntrackedOrigin](unsafe_from_address=record[0])
        if operation == XD_DEV_ALLOC:
            stats[0] += 1
            if (record[2] & 1) != 0:
                raise Error("RBF prescribed allocation failure")
            MutPointer[Int, MutUntrackedOrigin](unsafe_from_address=result)[0] = handle
        elif operation == XD_DEV_FREE:
            stats[7] += 1
            var index = stats[1]
            require(index < 5,"RBF resource freed more than once")
            stats[index+2] = record[1]
            stats[1] += 1
            if (record[2] & 2) != 0:
                raise Error("RBF prescribed release failure")
            MutPointer[Int, MutUntrackedOrigin](unsafe_from_address=result)[0] = 0
        else:
            raise Error("RBF unexpected resource operation")
        return 0
    except error:
        native_descriptor_report(error,context,sink)
        return 1


def resource_scope(records: List[Int], unit: Int) raises:
    var owner = SpectralRBFCells(resource_entry)
    try:
        # Required allocation/free order is sequential resource ownership, not
        # independent numerical work. Four complete cases run in parallel.
        for index in range(5):
            var address = Int(records.unsafe_ptr()+index*3)
            require(owner.allocate(address) == address,"RBF resource handle retains its exact native word")
        if unit == 3:
            raise Error("RBF original operation failure")
    finally:
        _ = owner^
        _ = records


def ownership(unit: Int) raises:
    var stats = List[Int](length=8,fill=0)
    var records = List[Int](length=15,fill=0)
    var tags: Array[Int,5] = [4,1,3,0,2]
    try:
        for index in range(5):  # Fixed five resource records, independent of tensor values.
            records[index*3] = Int(stats.unsafe_ptr())
            records[index*3+1] = tags[index]
            if unit == 1 and (index == 0 or index == 2):
                records[index*3+2] = 2
            elif unit == 2 and index == 2:
                records[index*3+2] = 1
            elif unit == 3 and index == 1:
                records[index*3+2] = 2
        var refused = False
        try:
            resource_scope(records,unit)
        except error:
            var expected = "RBF prescribed allocation failure" if unit == 2 else "RBF original operation failure"
            require(unit >= 2 and String(error) == expected,"RBF quiet cleanup preserves the original error")
            refused = True
        require(refused == (unit >= 2),"RBF ownership refusal outcome")
        require(stats[0] == (3 if unit == 2 else 5),"RBF allocation attempts preserve original order")
        var released = 2 if unit == 2 else 5
        require(stats[1] == released and stats[7] == released,
            "RBF releases every successfully allocated resource exactly once")
        for index in range(released):
            require(stats[index+2] == tags[index],"RBF quiet release follows allocation order, not handle sort order")
    finally:
        _ = tags^
        _ = records^
        _ = stats^


def numerical(entry: NativeDescriptorCall, unit: Int, resident: Bool) raises:
    var values = source("embedding150-input.f32",150*2)
    var reference_name = "rbf150-half-reference.f32" if unit == 0 else "rbf150-one-reference.f32"
    var expected = source(reference_name,150*150)
    var destination = guarded(150*150)
    var gamma = Float64(0.5 if unit == 0 else 1.0)
    try:
        spectral_rbf_affinity(entry,SpectralRBFRequest(Int(values.unsafe_ptr()),
            Int(destination.unsafe_ptr()+1),150,2,gamma,resident))
        finite(destination)
        var failures = List[Int](length=150*150,fill=0)
        var fp = failures.unsafe_ptr()
        var dp = destination.unsafe_ptr()+1
        var ep = expected.unsafe_ptr()
        def cell(index: Int) {imm fp, imm dp, imm ep}:
            var got = Float64(dp[index])
            var want = Float64(ep[index])
            var okay = got >= 0.0 and got <= 1.00002
            okay = okay and abs(got-want) <= Float64(2e-6)+Float64(2e-5)*abs(want)
            fp.unsafe_store(index,Int(not okay))
        host_parallelize(cell,150*150)
        clear(failures,"Independent binary64 Gaussian affinity formula within supplementary Float32 bound")
        source_unchanged("embedding150-input.f32",values)
    finally:
        _ = values^
        _ = expected^
        _ = destination^


# ---- The host column (was spectral_rbf_cpu_suite.mojo) ----
# Independent RBF owner and actual host numerical graphs use all host cores.
comptime SPECTRAL_RBF_CPU_FIXTURES = SPECTRAL_RBF_OWNERSHIP_FIXTURES + SPECTRAL_RBF_NUMERICAL_FIXTURES


def _run_cpu(mut report: Report) raises:
    comptime assert GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL
    var failures = List[Int](length=SPECTRAL_RBF_CPU_FIXTURES,fill=0)
    var fp = failures.unsafe_ptr()
    def task(unit: Int) {imm fp}:
        try:
            if unit < SPECTRAL_RBF_OWNERSHIP_FIXTURES:
                ownership(unit)
            else:
                numerical(x_decomp_host_v1,unit-SPECTRAL_RBF_OWNERSHIP_FIXTURES,False)
        except error:
            fp.unsafe_store(unit,1)
            print("FAIL spectral_rbf_cpu fixture",unit,String(error))
    host_parallelize(task,SPECTRAL_RBF_CPU_FIXTURES)
    var failed = 0
    for value in failures:
        failed += value
    require(failed == 0,"spectral_rbf_cpu failed " + String(failed) + " fixtures")
    print("PASS spectral_rbf_cpu:",SPECTRAL_RBF_CPU_FIXTURES,"supplementary owner and RBF formula fixtures")


# ---- The device column (was spectral_rbf_gpu_suite.mojo) ----
# One RBF graph at a time on the physical GPU, with existing parallel kernels.
#
# Both existing immediate and resident cell sequences run. The separately
# recorded shared-context concurrent-submission failure remains open.
comptime SPECTRAL_RBF_GPU_FIXTURES = 2*SPECTRAL_RBF_NUMERICAL_FIXTURES


def _run_gpu(mut report: Report) raises:
    comptime assert GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL
    var cases = report.cases(SPECTRAL_RBF_GPU_FIXTURES)
    for unit in range(SPECTRAL_RBF_GPU_FIXTURES):
        try:
            numerical(x_decomp_device_v1,unit%2,unit>=2)
        except error:
            cases.fail(unit, error)


def run_cases(column: Column, mut report: Report):
    try:
        comptime if CPU_COLUMN_BUILD:
            _run_cpu(report)
        else:
            _run_gpu(report)
    except error:
        report.abort(error)


def register() -> Check:
    return Check("spectral_rbf", cpu=True, gpu=True, serial="", run=run_cases)
