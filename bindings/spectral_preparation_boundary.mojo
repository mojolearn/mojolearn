# SPDX-License-Identifier: Apache-2.0
"""Spectral preparation boundary image pins around typed native requests."""
from std.python import Python, PythonObject
from std.python._cpython import GILReleased
from core.spectral_preparation_api import (
    SPECTRAL_PREPARATION_ABI, SPECTRAL_PREPARATION_COUNT, SpectralPreparationDriver,
    SpectralCOOCountRequest, SpectralCOOFillRequest, SpectralKNNRequest,
)
from bindings.spectral_umap_routes import spectral_umap_entry
from providers.python.neural_descriptor_pyapi import NeuralDescriptorProvider, neural_descriptor_connect_factory_once


struct SpectralPreparationConnection(Movable):
    var provider: PythonObject
    var driver: SpectralPreparationDriver

    def __init__(out self, binding: PythonObject, context: PythonObject) raises:
        var factory = spectral_umap_entry(binding, "spectral_preparation_native_api_v1", context)
        var provider = neural_descriptor_connect_factory_once(factory,
            String(SPECTRAL_PREPARATION_ABI), SPECTRAL_PREPARATION_COUNT)
        self.driver = SpectralPreparationDriver(provider.downcast_value_ptr[NeuralDescriptorProvider]()[].entry)
        self.provider = provider


def spectral_coo_count_run(connection: SpectralPreparationConnection, request: SpectralCOOCountRequest) raises -> Int:
    var count: Int
    try:
        with GILReleased(Python()):
            count = connection.driver.count(request)
    finally:
        _ = connection.provider
    return count


def spectral_coo_fill_run(connection: SpectralPreparationConnection, request: SpectralCOOFillRequest) raises -> Int:
    var count: Int
    try:
        with GILReleased(Python()):
            count = connection.driver.fill(request)
    finally:
        _ = connection.provider
    return count


def spectral_knn_run(connection: SpectralPreparationConnection, request: SpectralKNNRequest) raises:
    try:
        with GILReleased(Python()):
            connection.driver.knn(request)
    finally:
        _ = connection.provider
