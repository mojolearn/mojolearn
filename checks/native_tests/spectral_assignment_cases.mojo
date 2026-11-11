# SPDX-License-Identifier: Apache-2.0
"""Original180-row alternative assignment workloads, with independent partition.

The saved input graph has three connected components under both self-inclusive
and self-exclusive ten-neighbor conventions. The certificate is independent
NumPy input preparation, not a fitted-output hash or an eigensolver copy.
"""
from core.host_parallel import host_parallelize
from core.spectral_native_api import SpectralDriver, SpectralStateView
from core.spectral_assignment_api import SpectralAssignmentDriver, SpectralAssignmentRequest, SpectralAssignmentOutput, SpectralAssignmentResult
from checks.native_tests.spectral_native_cases import source, source_unchanged, guarded, guards, fit
from checks.native_tests.spectral_dense_cases import require, clear

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
