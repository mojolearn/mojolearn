# SPDX-License-Identifier: Apache-2.0
"""The existing device Spectral assignment entry, without Python envelopes."""
from core.spectral_assignment_api import SpectralAssignmentBackend
from x_cluster.device_execution import cluster_device_execute
from x_cluster.entries import ENTRY_SPECTRAL_ASSIGN
from x_cluster.out import ClusterOut


struct SpectralAssignmentDevice(SpectralAssignmentBackend):
    @staticmethod
    def execute(source: Int, count: Int, ints: List[Int], floats: List[Float64]) raises -> ClusterOut:
        return cluster_device_execute(ENTRY_SPECTRAL_ASSIGN, source, count, 0, 0, ints, floats)
