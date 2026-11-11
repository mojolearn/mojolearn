# SPDX-License-Identifier: Apache-2.0
"""PolynomialCountSketch typed state and requests over the existing Xn ABI."""
from core.neighbors_native_driver import XnDriver


@fieldwise_init
struct PCSOptions(Copyable, Movable):
    var degree: Int
    var components: Int


def pcs_options(degree: Int, components: Int) raises -> PCSOptions:
    if degree < 1 or components < 1:
        raise Error("degree and n_components must be >= 1")
    return PCSOptions(degree, components)


def pcs_seed_word(raw: UInt64) -> Int:
    return Int(raw & 0xFFFFFFFF)


def pcs_feature_width(features: Int, fitted_features: Int) raises:
    if features != fitted_features:
        raise Error("Number of features of test samples does not match that of training samples.")


@fieldwise_init
struct PCSDrawRequest(Copyable, Movable):
    var seed: Int
    var components: Int
    var total: Int
    var indices: Int
    var signs: Int


@fieldwise_init
struct PCSQueryRequest(Copyable, Movable):
    var source: Int
    var indices: Int
    var signs: Int
    var dst: Int
    var rows: Int
    var features: Int
    var augmented: Int
    var components: Int
    var degree: Int
    var gamma: Float32
    var coef0: Float32
    var old_items: Bool


struct PCSFitPlan(Copyable, Movable):
    var options: PCSOptions
    var features: Int
    var augmented: Int

    def __init__(out self, options: PCSOptions, features: Int, coef0: Float64):
        self.options = options.copy()
        self.features = features
        # The original augmentation decision uses the unrounded scalar.
        # A nonzero Float64 that rounds to Float32 zero still adds a column.
        self.augmented = features + (1 if coef0 != 0.0 else 0)

    def hash_shape(self) -> List[Int]:
        var shape: List[Int] = [self.options.degree, self.augmented]
        return shape^

    def output_shape(self, rows: Int) -> List[Int]:
        var shape: List[Int] = [rows, self.options.components]
        return shape^

    def draw(self, seed: Int, indices: Int, signs: Int) -> PCSDrawRequest:
        return PCSDrawRequest(seed, self.options.components, self.options.degree * self.augmented, indices, signs)

    def admit_hash_widths(self, indices: Int, signs: Int) raises:
        if signs != indices or indices != self.augmented:
            raise Error("ValueError: PolynomialCountSketch hash tables do not match the fitted features")

    def query(self, source: Int, indices: Int, signs: Int, dst: Int, rows: Int,
              gamma: Float64, coef0: Float64, old_items: Bool) -> PCSQueryRequest:
        return PCSQueryRequest(source, indices, signs, dst, rows, self.features, self.augmented,
            self.options.components, self.options.degree, Float32(gamma), Float32(coef0), old_items)


@fieldwise_init
struct PCSNativeDriver(Copyable, Movable):
    var native: XnDriver

    def require_draw(self) raises:
        var a: InlineArray[Int, 0] = []
        var p: InlineArray[Int, 0] = []
        var f: InlineArray[Float64, 0] = []
        if (self.native.run("capabilities", a, p, f) & 8) == 0:
            raise Error("NotImplementedError: mojolearn PolynomialCountSketch: this binary lacks "
                + "x_neighbors_kfeat_pcs_draw_idn (rebuild without -D MOJOLEARN_IDN_XN_SKETCH_CTR_OFF)")

    def draw(self, request: PCSDrawRequest) raises:
        var a: InlineArray[Int, 2] = [request.indices, request.signs]
        var p: InlineArray[Int, 3] = [request.seed, request.components, request.total]
        var f: InlineArray[Float64, 0] = []
        _ = self.native.run("pcs_draw", a, p, f)

    def query(self, request: PCSQueryRequest) raises:
        var a: InlineArray[Int, 4] = [request.source, request.indices, request.signs, request.dst]
        var p: InlineArray[Int, 5] = [request.rows, request.features, request.augmented,
            request.components, request.degree]
        var f: InlineArray[Float64, 2] = [Float64(request.gamma), Float64(request.coef0)]
        var name = String("pcs") if request.old_items else String("pcs_resident")
        _ = self.native.run(name, a, p, f)
