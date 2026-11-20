# SPDX-License-Identifier: Apache-2.0
"""Write the model files and weights of checks/native_tests/core_import_lower.mojo.

The check replaced functions of python/mojolearn/tests/test_import_lower.py
that import an ONNX model and compare its program with the eager session
composition. The models are written by python/mojolearn/tests/_onnx_fixture.py
(pure Python: a protobuf encoder and fixed-seed weights, no ``onnx``, no
``torch``, no compiled Mojolearn binding; it is loaded here by path so that
``import mojolearn`` never runs). For each model this stores, under
``checks/native_tests/fixtures/import_lower/``:

* ``<model>.onnx``: the model file, the bytes the Python test wrote;
* ``<model>-<initializer>.f32``: the raw little-endian float32 words of each
  initializer the eager reference uploads (the test's ``_init(f, name)``).

The three models that lived in the test (``mixed``, ``grouped_conv``,
``padded_pools``) moved here unchanged. The models are independent and are
written one per process across the host cores (``os.cpu_count()`` workers).

    python tools/import_lower_native_fixtures.py
"""
from concurrent.futures import ProcessPoolExecutor
import importlib.util
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "checks" / "native_tests" / "fixtures" / "import_lower"
_FX_PATH = ROOT / "python" / "mojolearn" / "tests" / "_onnx_fixture.py"


def _fx():
    spec = importlib.util.spec_from_file_location("_onnx_fixture_standalone", _FX_PATH)
    mod = importlib.util.module_from_spec(spec)
    import sys
    sys.modules[spec.name] = mod  # dataclasses resolve the module by name
    spec.loader.exec_module(mod)
    return mod


def _mixed(fx, directory):
    """x [batch, 7] -> Slice [:, 1:6] -> Gather columns (0, 2, 3) -> Clip(-0.5, 0.5) -> Pow 2 -> Add a [3]
    bias -> ReduceMean over axis 1 (keepdims 0) [batch]; the clipped values also go through Softmax
    (axis 1) and ArgMax (axis 1, keepdims 0) [batch]."""
    inits = [fx.tensor("starts", "i64", (1,), (1,)), fx.tensor("ends", "i64", (1,), (6,)),
             fx.tensor("axes", "i64", (1,), (1,)), fx.tensor("cols", "i64", (3,), (0, 2, 3)),
             fx.tensor("lo", "f32", (), (-0.5,)), fx.tensor("hi", "f32", (), (0.5,)),
             fx.tensor("two", "i64", (), (2,)), fx.weight("bias", (3,), 41), fx.tensor("mean_axes", "i64", (1,), (1,))]
    nodes = [fx.node("Slice", ["x", "starts", "ends", "axes"], ["xs"], name="slice"),
             fx.node("Gather", ["xs", "cols"], ["xg"], name="gather", axis=1),
             fx.node("Clip", ["xg", "lo", "hi"], ["xc"], name="clip"),
             fx.node("Pow", ["xc", "two"], ["xp"], name="pow"),
             fx.node("Add", ["xp", "bias"], ["xb"], name="bias"),
             fx.node("ReduceMean", ["xb", "mean_axes"], ["m"], name="mean", keepdims=0),
             fx.node("Softmax", ["xc"], ["p"], name="softmax", axis=1),
             fx.node("ArgMax", ["xc"], ["a"], name="argmax", axis=1, keepdims=0)]
    return fx._write(directory, "mixed.onnx", nodes, [("x", "f32", ("batch", 7))],
                     [("m", "f32", ("batch",)), ("p", "f32", ("batch", 3)), ("a", "i64", ("batch",))], inits,
                     lambda b: {"m": (b,), "p": (b, 3), "a": (b,)}, opset=18)


def _grouped_conv(fx, directory):
    """x [batch, 4, 8, 9] -> Conv group=2, weight [6, 2, 3, 3], bias, strides 2, auto_pad
    SAME_UPPER (pads (0, 1, 1, 1): the odd total of the height at the end) [batch, 6, 4, 5] ->
    depthwise Conv group=6, weight [6, 1, 3, 3], pads 1, no bias [batch, 6, 4, 5]."""
    inits = [fx.weight("gw", (6, 2, 3, 3), 61), fx.weight("gb", (6,), 62), fx.weight("dw", (6, 1, 3, 3), 63)]
    nodes = [fx.node("Conv", ["x", "gw", "gb"], ["g"], name="conv_g2", kernel_shape=(3, 3), strides=(2, 2),
                     auto_pad="SAME_UPPER", group=2),
             fx.node("Conv", ["g", "dw"], ["y"], name="conv_dw", kernel_shape=(3, 3), pads=(1, 1, 1, 1), group=6)]
    return fx._write(directory, "grouped_conv.onnx", nodes, [("x", "f32", ("batch", 4, 8, 9))],
                     [("y", "f32", ("batch", 6, 4, 5))], inits, lambda b: {"y": (b, 6, 4, 5)})


def _padded_pools(fx, directory):
    """x [batch, 3, 7, 8] -> MaxPool 3 strides 2 pads 1 [batch, 3, 4, 4]; AveragePool 3 strides 2
    auto_pad SAME_LOWER (pads (1, 1, 1, 0)), count_include_pad 0 [batch, 3, 4, 4]; AveragePool 3
    strides 2 pads 1, count_include_pad 1 [batch, 3, 4, 4]."""
    nodes = [fx.node("MaxPool", ["x"], ["m"], name="max_p1", kernel_shape=(3, 3), strides=(2, 2),
                     pads=(1, 1, 1, 1)),
             fx.node("AveragePool", ["x"], ["a0"], name="avg_same", kernel_shape=(3, 3), strides=(2, 2),
                     auto_pad="SAME_LOWER"),
             fx.node("AveragePool", ["x"], ["a1"], name="avg_incl", kernel_shape=(3, 3), strides=(2, 2),
                     pads=(1, 1, 1, 1), count_include_pad=1)]
    outs = [(o, "f32", ("batch", 3, 4, 4)) for o in ("m", "a0", "a1")]
    return fx._write(directory, "padded_pools.onnx", nodes, [("x", "f32", ("batch", 3, 7, 8))], outs, [],
                     lambda b: {o: (b, 3, 4, 4) for o in ("m", "a0", "a1")})


# model name -> (builder, initializers the eager reference reads)
MODELS = {
    "mixed": (lambda fx, d: _mixed(fx, d), ("bias",)),
    "grouped_conv": (lambda fx, d: _grouped_conv(fx, d), ("gw", "gb", "dw")),
    "padded_pools": (lambda fx, d: _padded_pools(fx, d), ()),
    "embedding_linear": (lambda fx, d: fx.write_fixture("embedding_linear", d), ("table", "proj", "proj_b")),
    "batchnorm_relu": (lambda fx, d: fx.write_fixture("batchnorm_relu", d),
                       ("bn_scale", "bn_bias", "bn_mean", "bn_var")),
    "linear_ln_linear": (lambda fx, d: fx.write_fixture("linear_layernorm_linear", d), ("w1",)),
}


def _write_one(model):
    fx = _fx()
    build, wanted = MODELS[model]
    f = build(fx, str(OUT))
    written = [os.path.basename(f.path)]
    for name in wanted:
        t = f.initializers[name]
        assert t.dtype == "f32" and t.external is None, (model, name)
        out = OUT / f"{model}-{name}.f32"
        out.write_bytes(t.data)
        written.append(out.name)
    return written


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    with ProcessPoolExecutor(max_workers=min(len(MODELS), os.cpu_count() or 1)) as pool:
        for files in pool.map(_write_one, sorted(MODELS)):
            print(" ".join(files))


if __name__ == "__main__":
    main()
