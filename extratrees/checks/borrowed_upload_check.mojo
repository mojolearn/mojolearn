# SPDX-License-Identifier: Apache-2.0
"""DEVIATION 2481: borrowed input and legacy List paths keep full model bits."""
from std.testing import assert_equal
from max.gpu.host import DeviceContext
from std.memory import bitcast
from extratrees.checks.fixtures import hashed_classification
from extratrees.checks.bestfirst_fingerprint import (
    column_major, float_labels, forest_fingerprint,
)
from extratrees.estimator import (
    ExtraTreesConfig, fit_extra_trees_classifier_device,
    fit_extra_trees_regressor_device,
)
from extratrees.impl.decisiontree.batched_levelalgo.builder import upload_dataset


def main() raises:
    var ctx = DeviceContext()
    # Raw upload must preserve signed zero/subnormal bits as well as normal X.
    var raw = List[Float32]()
    var labels = List[Int32]()
    for i in range(514):
        var bits = UInt32(i % 257) | (UInt32(i % 2) << 31)
        raw.append(bitcast[DType.float32](bits))
    for i in range(257):
        labels.append(Int32(i % 2))
    var empty = List[Float32]()
    var borrowed = upload_dataset(
        ctx, empty, labels, 257, 2, 2, x_addr=Int(raw.unsafe_ptr())
    )
    var host = ctx.enqueue_create_host_buffer[DType.float32](514)
    ctx.enqueue_copy(dst_buf=host, src_buf=borrowed.d_data)
    ctx.synchronize()
    for i in range(514):
        assert_equal(host.unsafe_ptr().unsafe_load(i).to_bits(), raw[i].to_bits())
    _ = borrowed^
    _ = host^
    _ = raw^

    var fixture = hashed_classification(12345, 257, 5, 3)
    var x = column_major(fixture)
    var y = float_labels(fixture)
    var targets = List[Float32]()
    for i in range(257):
        targets.append(Float32(i % 17 - 8) / Float32(8))
    for sampling in range(2):
        var bootstrap = sampling == 1
        var config = ExtraTreesConfig()
        config.n_estimators = 3
        config.max_depth = 4
        config.bootstrap = bootstrap
        var owned_fit = fit_extra_trees_classifier_device(ctx, x, y, 257, 5, 3, config)
        var borrowed_fit = fit_extra_trees_classifier_device(
            ctx, empty, y, 257, 5, 3, config, x_addr=Int(x.unsafe_ptr())
        )
        assert_equal(forest_fingerprint(owned_fit.forest),
                     forest_fingerprint(borrowed_fit.forest))
        var regression = config.for_regression()
        var owned_reg = fit_extra_trees_regressor_device(ctx, x, targets, 257, 5, regression)
        var borrowed_reg = fit_extra_trees_regressor_device(
            ctx, empty, targets, 257, 5, regression, x_addr=Int(x.unsafe_ptr())
        )
        assert_equal(forest_fingerprint(owned_reg.forest),
                     forest_fingerprint(borrowed_reg.forest))
    _ = x^
    print("borrowed_upload_check: ALL OK (raw bits, classifier/regressor, bootstrap on/off)")
