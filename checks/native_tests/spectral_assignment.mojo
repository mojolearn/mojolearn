# SPDX-License-Identifier: Apache-2.0
# native-check: cpu gpu
"""Original180-row alternative assignment workloads, with independent partition.

The saved input graph has three connected components under both self-inclusive
and self-exclusive ten-neighbor conventions. The certificate is independent
NumPy input preparation, not a fitted-output hash or an eigensolver copy.
"""
from std.sys.compile import is_defined
from std.testing import TestSuite
from checks.native_tests.testsuite_cases import check_cases
from core.host_parallel import host_parallelize
from core.spectral_native_api import SpectralDriver, SpectralStateView
from core.spectral_assignment_api import (
    SpectralAssignmentDriver,
    SpectralAssignmentRequest,
    SpectralAssignmentOutput,
    SpectralAssignmentResult,
    spectral_assign_native_v1,
)
from checks.native_tests.spectral_native import source, source_unchanged, guarded, guards, fit
from checks.native_tests.spectral_dense import clear
from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_IDENTICAL
from core.spectral_native_provider import spectral_native_v1
from core.spectral_native_host import SpectralHostKernels
from core.spectral_assignment_host import SpectralAssignmentHost
from core.spectral_native_device import SpectralDeviceKernels
from core.spectral_assignment_device import SpectralAssignmentDevice


# The column this binary was built for (`-D MOJOLEARN_COLUMN_CPU=1` builds the host column).
comptime CPU_COLUMN_BUILD = is_defined["MOJOLEARN_COLUMN_CPU"]()


def require(ok: Bool, message: String) raises:
    if not ok:
        raise Error(message)


comptime SPECTRAL_ASSIGNMENT_FIXTURES = 2


def assigned(fitter: SpectralDriver, assignment: SpectralAssignmentDriver,
             method: Int) raises -> SpectralAssignmentResult:
    var values = source("assignment180-input.f32",180*3)
    var labels = List[Int32](length=182,fill=-31)
    var embedding = guarded(180*3)
    var output = SpectralAssignmentOutput()
    try:
        # Same default numerical fit and assignment settings as the original
        # SpectralClustering calls in test_x_cluster_twice's embedded child.
        fit(fitter,values,180,3,0,3,3,10,1,False,SpectralStateView(0,0,0,0),labels,embedding)
        assignment.assign(SpectralAssignmentRequest(Int(embedding.unsafe_ptr()+1),180*3,
            180,3,method,1,30,20),output)
        require(len(output.storage) == 1,"Spectral assignment returns one native owned result")
        var result = output.storage.pop()
        require(len(output.storage) == 0,"Native assignment result moves out of its output room")
        guards(embedding)
        source_unchanged("assignment180-input.f32",values)
        return result^
    finally:
        # Validation in the caller happens after the input, fit output and
        # descriptor output room are destroyed; labels must own their storage.
        _ = values^
        _ = labels^
        _ = embedding^
        _ = output^


def partition(labels: List[Int32]) raises:
    require(len(labels) == 180,"Original assignment label count")
    require(labels[0] != labels[60] and labels[0] != labels[120] and labels[60] != labels[120],
        "Independent neighbor graph components receive distinct labels")
    var failures = List[Int](length=180,fill=0)
    var fp = failures.unsafe_ptr()
    var lp = labels.unsafe_ptr()
    def row(index: Int) {imm fp, imm lp}:
        var value = lp[index]
        fp.unsafe_store(index,Int(value < Int32(0) or value > Int32(2)
            or value != lp[(index//60)*60]))
    host_parallelize(row,180)
    clear(failures,"Independent three-component partition, invariant to assignment label permutation")
    _ = labels


def fixture(fitter: SpectralDriver, assignment: SpectralAssignmentDriver, method: Int) raises:
    var result = assigned(fitter,assignment,method)
    partition(result.labels)
    require(result.iterations >= 0,"Native assignment iteration metadata is nonnegative")
    _ = result^


def _host_fixture(unit: Int) raises:
    var fitter = SpectralDriver(spectral_native_v1[SpectralHostKernels])
    var assignment = SpectralAssignmentDriver(spectral_assign_native_v1[SpectralAssignmentHost])
    fixture(fitter, assignment, unit)


def test_assignment() raises:
    comptime assert GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL
    comptime if CPU_COLUMN_BUILD:
        check_cases("assignment", SPECTRAL_ASSIGNMENT_FIXTURES, _host_fixture)
    else:
        # One complete assignment graph at a time on the physical GPU; the tuned fit/assignment kernels stay parallel.
        var fitter = SpectralDriver(spectral_native_v1[SpectralDeviceKernels])
        var assignment = SpectralAssignmentDriver(spectral_assign_native_v1[SpectralAssignmentDevice])
        for unit in range(SPECTRAL_ASSIGNMENT_FIXTURES):  # driver-loop: one graph at a time on the one device
            try:
                fixture(fitter, assignment, unit)
            except problem:
                raise Error("assignment row " + String(unit) + ": " + String(problem))


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
