# SPDX-License-Identifier: Apache-2.0
"""The existing host Spectral assignment entry, without Python envelopes."""
from core.spectral_assignment_api import SpectralAssignmentBackend
from x_cluster.host.execution import cluster_host_execute
from x_cluster.entries import ENTRY_SPECTRAL_ASSIGN
from x_cluster.out import ClusterOut


struct SpectralAssignmentHost(SpectralAssignmentBackend):
    @staticmethod
    def execute(source: Int, count: Int, ints: List[Int], floats: List[Float64]) raises -> ClusterOut:
        return cluster_host_execute(ENTRY_SPECTRAL_ASSIGN, source, count, 0, 0, ints, floats)
