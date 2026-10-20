# SPDX-License-Identifier: Apache-2.0
"""The operations a model import supports (``mojolearn.importers.supported_ops``)
and the session shape of a model value. The lowering itself is
autograd/import_lower.mojo; its table of ops and checks is the one listed here."""
from __future__ import annotations

# (op, the subset the import supports): the table mojolearn.importers.supported_ops() returns
SUPPORTED_OPS = (
    ("Abs", "any input"),
    ("Add", "inputs that broadcast"),
    ("And", "inputs that broadcast; nonzero is true"),
    ("ArgMax", "select_last_index=0; the positions come back as int32"),
    ("AveragePool", "2-D, square kernel_shape, equal strides, dilations 1; any pads or auto_pad, "
                    "count_include_pad 0 or 1"),
    ("BatchNormalization", "inference (training_mode=0) with constant scale, bias, mean and variance"),
    ("Cast", "to any float type; between integer and bool types; not from float to integer or bool"),
    ("Ceil", "any input"),
    ("Clip", "constant or computed min and max"),
    ("Concat", "any axis"),
    ("ConstantOfShape", "a static shape"),
    ("Conv", "1-D and 2-D, any group, dilations 1, equal strides; any pads or auto_pad"),
    ("Cos", "any input"),
    ("CumSum", "a static axis; exclusive and reverse"),
    ("Div", "inputs that broadcast; floating-point values"),
    ("Dropout", "inference: the input passes through; the mask output unused"),
    ("Elu", "any alpha"),
    ("Equal", "inputs that broadcast"),
    ("Erf", "any input"),
    ("Exp", "any input"),
    ("Expand", "a static shape"),
    ("Flatten", "any axis"),
    ("Floor", "any input"),
    ("Gather", "constant indices, or an integer model input read only as indices, each in [0, size)"),
    ("GatherND", "batch_dims=0; constant or computed indices, each in range"),
    ("Gelu", "approximate 'none' or 'tanh'"),
    ("Gemm", "any transA, transB, alpha and beta; a bias that broadcasts"),
    ("GlobalAveragePool", "rank 3 or more"),
    ("GlobalMaxPool", "rank 3 or more"),
    ("Greater", "inputs that broadcast"),
    ("GreaterOrEqual", "inputs that broadcast"),
    ("GroupNormalization", "rank 3 or more; scale and bias per channel or per group"),
    ("HardSigmoid", "any alpha and beta"),
    ("HardSwish", "any input"),
    ("Identity", "any input"),
    ("InstanceNormalization", "rank 3 or more"),
    ("LayerNormalization", "the last axis; the Mean and InvStdDev outputs unused"),
    ("LeakyRelu", "any alpha"),
    ("Less", "inputs that broadcast"),
    ("LessOrEqual", "inputs that broadcast"),
    ("Log", "any input"),
    ("LogSoftmax", "any axis"),
    ("MatMul", "any ranks that broadcast as NumPy's matmul"),
    ("Max", "inputs that broadcast"),
    ("MaxPool", "2-D, square kernel_shape, equal strides, dilations 1; any pads or auto_pad; "
                "the Indices output unused"),
    ("MaxPool", "2-D, square kernel_shape, equal strides, pads smaller than the kernel, dilations 1; the Indices output unused"),
    ("Min", "inputs that broadcast"),
    ("Mul", "inputs that broadcast"),
    ("Neg", "any input"),
    ("Not", "any input"),
    ("Or", "inputs that broadcast; nonzero is true"),
    ("Pad", "mode 'constant'; static pads and axes"),
    ("Pow", "a tensor or constant exponent"),
    ("Reciprocal", "any input"),
    ("ReduceL2", "static axes"),
    ("ReduceMax", "static axes; on bool values, any"),
    ("ReduceMean", "static axes"),
    ("ReduceMin", "static axes; on bool values, all"),
    ("ReduceSum", "static axes"),
    ("Relu", "any input"),
    ("Reshape", "a static target shape"),
    ("RMSNormalization", "the last axis"),
    ("Round", "any input; a half rounds to the even integer"),
    ("Shape", "folded at import"),
    ("Sigmoid", "any input"),
    ("Sign", "any input"),
    ("Sin", "any input"),
    ("Size", "folded at import"),
    ("Slice", "static starts, ends and axes; steps 1"),
    ("Softmax", "any axis"),
    ("Softplus", "any input"),
    ("Split", "static sizes; no empty piece"),
    ("Sqrt", "any input"),
    ("Squeeze", "static axes"),
    ("Sub", "inputs that broadcast"),
    ("Tanh", "any input"),
    ("Transpose", "any permutation"),
    ("Trilu", "rank 2 or more; a static k"),
    ("Unsqueeze", "static axes"),
    ("Where", "inputs that broadcast; a nonzero condition selects the first value"),
)


def session_shape(shape):
    """The shape a session holds for a model value: a session tensor has at
    least two axes, so a scalar is ``(1, 1)`` and a ``[n]`` vector ``(1, n)``
    (NumPy broadcasting pads on the left the same way)."""
    shape = tuple(shape)
    return shape if len(shape) >= 2 else (1,) * (2 - len(shape)) + shape
