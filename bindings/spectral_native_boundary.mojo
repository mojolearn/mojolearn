# SPDX-License-Identifier: Apache-2.0
"""Spectral UNPACK/PACK pins around one native request per numerical operation."""
from std.python import Python, PythonObject
from std.python._cpython import GILReleased
from core.spectral_native_api import (
    SPECTRAL_NATIVE_ABI, SPECTRAL_NATIVE_COUNT, SPECTRAL_CAPABILITIES,
    SpectralDriver, SpectralStateView, SpectralFitRequest, SpectralEmbeddingRequest, SpectralQueryRequest,
)
from bindings.classical_arrays import classical_address
from bindings.family_common import family_array
from bindings.spectral_umap_routes import spectral_umap_entry
from providers.python.neural_descriptor_pyapi import (
    NeuralDescriptorProvider, neural_descriptor_connect_factory_once,
)


@fieldwise_init
struct SpectralRoutePins(Movable):
    var metrics: PythonObject
    var decomp: PythonObject
    var base: PythonObject
    var cluster: PythonObject


@fieldwise_init
struct SpectralPredictionPins(Movable):
    var labels: PythonObject
    var embedding: PythonObject


struct SpectralConnection(Movable):
    var provider: PythonObject
    var driver: SpectralDriver
    var capabilities: Int

    def __init__(out self, binding: PythonObject, context: PythonObject) raises:
        # Captured namespaces retain exact compiled entries; the connection
        # resolves and pins the factory's actual extension image, not the copy.
        var factory = spectral_umap_entry(binding, "spectral_native_api_v1", context)
        var provider = neural_descriptor_connect_factory_once(factory,
            String(SPECTRAL_NATIVE_ABI), SPECTRAL_NATIVE_COUNT)
        self.driver = SpectralDriver(provider.downcast_value_ptr[NeuralDescriptorProvider]()[].entry)
        self.capabilities = self.driver.metadata(SPECTRAL_CAPABILITIES)
        self.provider = provider


struct SpectralStatePins(Copyable, Movable):
    # These are the observable fitted Arrays. Only their lifetime pins remain
    # Python; the four-address execution view is a native value.
    var eigenvalues: PythonObject
    var eigenvectors: PythonObject
    var diagonal: PythonObject
    var centroids: PythonObject

    def __init__(out self):
        self.eigenvalues = PythonObject(None)
        self.eigenvectors = PythonObject(None)
        self.diagonal = PythonObject(None)
        self.centroids = PythonObject(None)

    def allocate(mut self, rows: Int, components: Int, clusters: Int, context: PythonObject) raises:
        self.eigenvalues = family_array(Python.tuple(components), "<f4", context)
        self.eigenvectors = family_array(Python.tuple(rows, components), "<f4", context)
        self.diagonal = family_array(Python.tuple(rows), "<f4", context)
        self.centroids = family_array(Python.tuple(clusters, components), "<f4", context)

    def view(self) raises -> SpectralStateView:
        if self.eigenvalues is None:
            return SpectralStateView(0, 0, 0, 0)
        return SpectralStateView(Int(py=classical_address(self.eigenvalues)),
            Int(py=classical_address(self.eigenvectors)), Int(py=classical_address(self.diagonal)),
            Int(py=classical_address(self.centroids)))

    def same(self, other: Self) -> Bool:
        return (self.eigenvalues is other.eigenvalues and self.eigenvectors is other.eigenvectors
            and self.diagonal is other.diagonal and self.centroids is other.centroids)


def spectral_fit_run(connection: SpectralConnection, request: SpectralFitRequest) raises -> Int:
    var columns: Int
    try:
        with GILReleased():
            columns = connection.driver.fit(request)
    finally:
        _ = connection.provider
    return columns


def spectral_embedding_run(connection: SpectralConnection, request: SpectralEmbeddingRequest) raises -> Int:
    var columns: Int
    try:
        with GILReleased():
            columns = connection.driver.embed(request)
    finally:
        _ = connection.provider
    return columns


def spectral_query_run(connection: SpectralConnection, request: SpectralQueryRequest) raises:
    try:
        with GILReleased():
            _ = connection.driver.query(request)
    finally:
        _ = connection.provider
