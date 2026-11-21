# SPDX-License-Identifier: Apache-2.0
"""RUN half of `CNNClassifier` (`bindings/xcnn_classifier_native.mojo` is the
UNPACK/PACK boundary; lane doorsplit-cnn-xseq, 2026-11-11).

Typed values only: the constructor's number checks and refusal texts, the
layer-stack geometry `_build` derives (block planes, seeds, fan-ins, the
Kaiming bound), the image shape checks, and the CNN_FIT / CNN_OPEN /
CNN_PREDICT descriptor words (`CnnClfPlan`, `CnnClfFitSettings`). The
boundary reads the model's fields into these values, owns every array (their
addresses arrive as words), and calls the tier's whole-model door through its
provider, which pins the provider and releases the GIL around the native
call (docs/plans/PYTHON_AT_THE_BOUNDARY.md section 2.4).
"""
from std.math import sqrt
from x_cnn.door_run import xcnn_size
from sequence.xseq_neural_run import XseqNeuralEntry, xseq_run_uniform

#: rows per predict pass on the device (every op is per row, so the chunk
#: never changes a word)
comptime CNNCLF_PREDICT_ROWS = 2048


# ---------------------------------------------------------------- constructor checks

def cnnclf_run_check_input_rank(rank: Int) raises:
    if rank != 3:
        raise Error("ValueError: input_shape must contain C, H and W")


def cnnclf_run_check_input_dim(value: Int) raises:
    if value < 1:
        raise Error("ValueError: input_shape dimensions must be positive")


def cnnclf_run_check_channels(value: Int) raises:
    if value < 1:
        raise Error("ValueError: convolution channels must be positive")


def cnnclf_run_check_setting(position: Int, value: Int) raises:
    """kernel_size, pool_size and batch_size (positions 0-2) are at least 1;
    max_iter (position 3) is at least 0."""
    if value < (0 if position == 3 else 1):
        raise Error("ValueError: invalid classifier size or iteration setting")


def cnnclf_run_check_optimizer(name: String) raises:
    if name != "sgd" and name != "adam" and name != "adamw":
        raise Error("ValueError: optimizer must be sgd, adam or adamw")


def cnnclf_run_optimizer_word(name: String) -> Int:
    """The descriptor's optimizer word: 0 sgd, 1 adam, 2 adamw."""
    return 0 if name == "sgd" else (1 if name == "adam" else 2)


def cnnclf_run_check_nesterov(nesterov: Bool, momentum: Float64, dampening: Float64) raises:
    if nesterov and (momentum <= 0 or dampening != 0):
        raise Error("ValueError: Nesterov momentum requires a momentum and zero dampening")


# ---------------------------------------------------------------- layer geometry

@fieldwise_init
struct CnnClfBlockPlane(Copyable, Movable):
    """One conv block's output plane (`_build`): height, width, and whether
    the block keeps its MaxPool2d."""
    var h: Int
    var w: Int
    var pools: Bool


def cnnclf_run_block_plane(h: Int, w: Int, kernel: Int, pool: Int) -> CnnClfBlockPlane:
    """The 'same'-padded convolution (padding kernel // 2), then the pool when
    the plane holds one window and the pool is larger than one."""
    var ch = h + 2 * (kernel // 2) - kernel + 1
    var cw = w + 2 * (kernel // 2) - kernel + 1
    if ch >= pool and cw >= pool and pool > 1:
        return CnnClfBlockPlane(ch // pool, cw // pool, True)
    return CnnClfBlockPlane(ch, cw, False)


def cnnclf_run_conv_seed(seed: Int, block: Int) -> Int:
    return seed + 101 * block


def cnnclf_run_head_seed(seed: Int) -> Int:
    return seed + 997


def cnnclf_run_conv_fan_in(channels: Int, kernel: Int) -> Int:
    return channels * kernel * kernel


def cnnclf_run_kaiming_bound(fan_in: Int) -> Float64:
    """`_kaiming_uniform`'s bound: 1 / sqrt(fan_in), correctly rounded."""
    return 1.0 / sqrt(Float64(fan_in))


@fieldwise_init
struct CnnClfLayout(Copyable, Movable):
    """`_build(n_classes)`'s layer stack as typed words: per conv block its
    width and pool flag, the head's input width, and for every learned array
    in descriptor order (per block `weight_`, `bias_`, then the head's) its
    shape, float count, Kaiming fan-in and layer seed."""
    var widths: List[Int]
    var pools: List[Int]
    var flat: Int
    var shapes: List[List[Int]]
    var counts: List[Int]
    var fan_ins: List[Int]
    var seeds: List[Int]


def cnnclf_run_layout(c: Int, h: Int, w: Int, kernel: Int, pool: Int, channels: List[Int], classes: Int,
                      seed: Int) -> CnnClfLayout:
    """The stack `_build` makes: block i is Conv2d(c_i, channels[i],
    kernel, padding kernel // 2, seed + 101 i), ReLU, and MaxPool2d(pool)
    when its plane holds a window; the head is Linear(flat, classes,
    seed + 997). `seed` is `random_state` (0 when None)."""
    var widths = List[Int]()
    var pools = List[Int]()
    var shapes = List[List[Int]]()
    var counts = List[Int]()
    var fan_ins = List[Int]()
    var seeds = List[Int]()
    var ch = c
    var hh = h
    var ww = w
    for i in range(len(channels)):  # driver-loop: over the conv blocks (model structure)
        var oc = channels[i]
        var fan = cnnclf_run_conv_fan_in(ch, kernel)
        var layer_seed = cnnclf_run_conv_seed(seed, i)
        var weight_shape: List[Int] = [oc, ch, kernel, kernel]
        var bias_shape: List[Int] = [oc]
        shapes.append(weight_shape^)
        shapes.append(bias_shape^)
        counts.append(oc * ch * kernel * kernel)
        counts.append(oc)
        fan_ins.append(fan)
        fan_ins.append(fan)
        seeds.append(layer_seed)
        seeds.append(layer_seed)
        var plane = cnnclf_run_block_plane(hh, ww, kernel, pool)
        widths.append(oc)
        pools.append(1 if plane.pools else 0)
        hh = plane.h
        ww = plane.w
        ch = oc
    var flat = cnnclf_run_features([ch, hh, ww])
    var head_seed = cnnclf_run_head_seed(seed)
    var head_weight: List[Int] = [classes, flat]
    var head_bias: List[Int] = [classes]
    shapes.append(head_weight^)
    shapes.append(head_bias^)
    counts.append(classes * flat)
    counts.append(classes)
    fan_ins.append(flat)
    fan_ins.append(flat)
    seeds.append(head_seed)
    seeds.append(head_seed)
    return CnnClfLayout(widths^, pools^, flat, shapes^, counts^, fan_ins^, seeds^)


def cnnclf_run_init(entry: XseqNeuralEntry, layout: CnnClfLayout, params: List[Int]) raises:
    """The initial weights `_build` draws: every learned array (addresses
    in descriptor order) gets float32 U(-b, b), b = 1 / sqrt(fan_in), from
    its layer's counter-based stream (`uniform_init_f32`, MODEL_UNIFORM of
    the model's image: the same words on the host and every device). Each
    layer's stream starts at counter 0 of its seed; its `bias_` continues
    where `weight_` stopped. The draws of one array run in parallel on the
    image; the loop is over model structure."""
    if len(params) != len(layout.counts):
        raise Error("RuntimeError: CNNClassifier: one address per learned array")
    var offset = 0
    for i in range(len(layout.counts)):  # driver-loop: over the learned arrays (model structure)
        if i % 2 == 0:
            offset = 0  # a new layer: its own stream
        var bound = cnnclf_run_kaiming_bound(layout.fan_ins[i])
        offset = xseq_run_uniform(entry, params[i], layout.counts[i], UInt64(layout.seeds[i]), offset, -bound, bound)


# ---------------------------------------------------------------- images

def cnnclf_run_check_rows(shape: List[Int]) raises:
    if len(shape) < 1:
        raise Error("ValueError: classifier images require a row dimension")


@fieldwise_init
struct CnnClfImages(Copyable, Movable):
    """`_images(X)`'s answer: the (n, C, H, W) words and whether the flat
    rows are reshaped to them."""
    var reshape: Bool
    var full: List[Int]


def cnnclf_run_images(shape: List[Int], given: List[Int]) raises -> CnnClfImages:
    """`shape` is the admitted X shape (rank checked by
    `cnnclf_run_check_rows`), `given` the model's (C, H, W)."""
    var full: List[Int] = [shape[0], given[0], given[1], given[2]]
    if len(shape) == 2:
        return CnnClfImages(True, full^)
    if len(shape) != 4:
        raise Error("ValueError: classifier images must have rank two or four")
    for i in range(1, 4):  # small-loop(image channel/height/width metadata)
        if shape[i] != full[i]:
            raise Error("ValueError: images do not match the model input_shape")
    return CnnClfImages(False, full^)


def cnnclf_run_features(input_shape: List[Int]) -> Int:
    """`n_features_in_`: C * H * W."""
    return xcnn_size(input_shape)


def cnnclf_run_check_labels(labels: Int, n: Int) raises:
    if labels != n:
        raise Error("ValueError: X and y must contain the same number of rows")


# ---------------------------------------------------------------- descriptor words

@fieldwise_init
struct CnnClfPlan(Copyable, Movable):
    """The head of a CNN_FIT / CNN_OPEN / CNN_PREDICT descriptor (UNPACK reads
    it once from the model's fields): rows, the input plane, the class count,
    kernel and pool sizes, and per block its width and pool flag."""
    var n: Int
    var c: Int
    var h: Int
    var w: Int
    var k: Int
    var kernel: Int
    var pool: Int
    var widths: List[Int]
    var pools: List[Int]


@fieldwise_init
struct CnnClfFitSettings(Copyable, Movable):
    """The training settings CNN_FIT reads."""
    var batch: Int
    var max_iter: Int
    var shuffle: Bool
    var seed: UInt64
    var optimizer: Int          # cnnclf_run_optimizer_word
    var nesterov: Bool
    var learning_rate: Float64
    var momentum: Float64
    var beta1: Float64
    var beta2: Float64
    var eps: Float64
    var weight_decay: Float64
    var dampening: Float64


def _plan_head(plan: CnnClfPlan) -> List[Int]:
    """[n, C, H, W, k, blocks, kernel_size, pool_size]."""
    var out: List[Int] = [plan.n, plan.c, plan.h, plan.w, plan.k, len(plan.widths), plan.kernel, plan.pool]
    return out^


def _plan_blocks(plan: CnnClfPlan, mut ints: List[Int]):
    """Every block's width, then every block's pool flag."""
    for i in range(len(plan.widths)):  # driver-loop: over the conv blocks (model structure)
        ints.append(plan.widths[i])
    for i in range(len(plan.pools)):  # driver-loop: over the conv blocks (model structure)
        ints.append(plan.pools[i])


def cnnclf_run_fit_steps(n: Int, batch: Int) -> Int:
    """Optimizer steps per epoch (the last batch may be short)."""
    return (n + batch - 1) // batch


def cnnclf_run_fit_addresses(x: Int, y: Int, curve: Int, steps: Int, params: List[Int]) -> List[Int]:
    var out: List[Int] = [x, y, curve, steps]
    for i in range(len(params)):  # driver-loop: over the layer parameter arrays (model structure)
        out.append(params[i])
    return out^


def cnnclf_run_fit_ints(plan: CnnClfPlan, settings: CnnClfFitSettings) -> List[Int]:
    var ints = _plan_head(plan)
    ints.append(settings.batch)
    ints.append(settings.max_iter)
    ints.append(1 if settings.shuffle else 0)
    ints.append(Int(settings.seed & 0xFFFFFFFF))
    ints.append(Int(settings.seed >> 32))
    ints.append(settings.optimizer)
    ints.append(1 if settings.nesterov else 0)
    _plan_blocks(plan, ints)
    return ints^


def cnnclf_run_fit_floats(settings: CnnClfFitSettings) -> List[Float64]:
    var floats: List[Float64] = [
        settings.learning_rate, settings.momentum, settings.beta1, settings.beta2, settings.eps,
        settings.weight_decay, settings.dampening,
    ]
    return floats^


def cnnclf_run_open_addresses(scope: Int, params: List[Int]) -> List[Int]:
    var out: List[Int] = [scope]
    for i in range(len(params)):  # driver-loop: over the layer parameter arrays (model structure)
        out.append(params[i])
    return out^


def cnnclf_run_open_ints(plan: CnnClfPlan) -> List[Int]:
    """CNN_OPEN's words: the plan head (n is 1) and the block words."""
    var ints = _plan_head(plan)
    _plan_blocks(plan, ints)
    return ints^


def cnnclf_run_check_session(handle: Int, context_words: Int) raises:
    if handle == 0 or context_words != 2:
        raise Error("RuntimeError: CNN resident session is closed")


def cnnclf_run_predict_ints(context0: Int, context1: Int, handle: Int, n: Int, codes: Bool) -> List[Int]:
    var ints: List[Int] = [context0, context1, handle, n, 1 if codes else 0, CNNCLF_PREDICT_ROWS]
    return ints^
