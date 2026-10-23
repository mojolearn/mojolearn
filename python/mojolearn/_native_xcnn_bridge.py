# SPDX-License-Identifier: Apache-2.0
"""Import-time connection of the native owner of the CNN layers
(bindings/xcnn_native.mojo). A migrated layer method is one call,
`_XCNN(self, "<operation>", args, _XCNN_CONTEXT)`; Python here only connects
the compiled `_mojolearn_x_cnn` entries of both numeric modes, the switch
words the binary reports and the layer classes, once, at import. NumPy is
not imported here: the native call connects it when a layer runs."""

_ENTRY = None
_CONTEXT = None

#: the compiled entries the CNN layers run (those a binary lacks are absent)
_NAMES = (
    "x_cnn_adam_hyper_d",
    "x_cnn_adam_r",
    "x_cnn_adaptive_pool",
    "x_cnn_adaptive_pool_m",
    "x_cnn_add",
    "x_cnn_avgpool2d_backward",
    "x_cnn_avgpool2d_backward_m",
    "x_cnn_avgpool2d_forward",
    "x_cnn_avgpool2d_forward_m",
    "x_cnn_batchnorm_backward",
    "x_cnn_batchnorm_backward_m",
    "x_cnn_batchnorm_forward",
    "x_cnn_batchnorm_forward_m",
    "x_cnn_bounded_im2col",
    "x_cnn_chan_copy_m",
    "x_cnn_conv2d_backward",
    "x_cnn_conv2d_backward_m",
    "x_cnn_conv2d_forward",
    "x_cnn_conv2d_forward_m",
    "x_cnn_conv_block_backward_r",
    "x_cnn_conv_block_forward_r",
    "x_cnn_conv_shape",
    "x_cnn_csr_build",
    "x_cnn_csr_upload",
    "x_cnn_dropout2d",
    "x_cnn_dropout2d_m",
    "x_cnn_epoch_rows",
    "x_cnn_fit_epoch_d",
    "x_cnn_fit_epoch_r",
    "x_cnn_gcn_loops",
    "x_cnn_gcn_norm",
    "x_cnn_gcn_norm_m",
    "x_cnn_gemm",
    "x_cnn_gemm_m",
    "x_cnn_graph_op",
    "x_cnn_graph_op_m",
    "x_cnn_idn2_flags",
    "x_cnn_idn_flags",
    "x_cnn_linear_backward",
    "x_cnn_linear_backward_m",
    "x_cnn_linear_backward_r",
    "x_cnn_linear_forward",
    "x_cnn_linear_forward_m",
    "x_cnn_linear_forward_r",
    "x_cnn_map2_m",
    "x_cnn_maxpool2d_backward",
    "x_cnn_maxpool2d_backward_m",
    "x_cnn_maxpool2d_forward",
    "x_cnn_maxpool2d_forward_m",
    "x_cnn_mul",
    "x_cnn_neural_tape_budget_bytes",
    "x_cnn_numerical_profile",
    "x_cnn_pad2d_backward",
    "x_cnn_pad2d_forward",
    "x_cnn_pad2d_m",
    "x_cnn_pool_shape",
    "x_cnn_relu_backward",
    "x_cnn_relu_forward",
    "x_cnn_res_alloc",
    "x_cnn_res_alloc_nofill",
    "x_cnn_res_argmax",
    "x_cnn_res_download",
    "x_cnn_res_free",
    "x_cnn_res_gather",
    "x_cnn_res_upload",
    "x_cnn_sgd_r",
    "x_cnn_softmax_xent_r",
    "x_cnn_spmm",
    "x_cnn_spmm_m",
    "x_cnn_vendor",
)


def _unavailable(error):
    # glue: an install without the compiled owner keeps the package
    # importable; the layers then explain what is missing.
    message = f"mojolearn: the CNN layers need the compiled host binding: {error}"

    def unavailable(*args):
        raise ModuleNotFoundError(message, name=getattr(error, "name", None))
    return unavailable


def _connect_mode(module):
    # glue: the compiled entries and the constant switch words of one binary
    from types import BuiltinFunctionType
    symbols = {}
    for name in _NAMES:  # glue: fixed compiled entry names
        try:
            fn = getattr(module, name)
        except AttributeError:
            continue
        if not isinstance(fn, BuiltinFunctionType):
            raise TypeError(f"the CNN layers require a compiled {name}")
        symbols[name] = fn
    mixed = "x_cnn_map2_m" in symbols
    idn = int(symbols["x_cnn_idn_flags"]()) if mixed and "x_cnn_idn_flags" in symbols else 0
    idn2 = int(symbols["x_cnn_idn2_flags"]()) if "x_cnn_idn2_flags" in symbols else 0
    profile = str(symbols["x_cnn_numerical_profile"]()) if "x_cnn_numerical_profile" in symbols else None
    return {"module": module, "symbols": symbols, "mixed": mixed, "idn": idn, "idn2": idn2, "profile": profile}


def connect():
    """`(entry, context)`: the compiled owner and the context every layer
    shares. Connected once; later calls return the same pair."""
    global _ENTRY, _CONTEXT
    if _CONTEXT is not None:
        return _ENTRY, _CONTEXT
    from types import BuiltinFunctionType
    from . import _backend
    try:
        from .tokenizer import _binding
        entry = _binding()._module.xcnn_native_call
        if not isinstance(entry, BuiltinFunctionType):
            raise TypeError("the CNN layers require the compiled xcnn_native_call entry")
        from ._native_classical_bridge import _state_context
        context = _state_context()
    except (ImportError, AttributeError, TypeError) as error:
        _ENTRY, _CONTEXT = _unavailable(error), {"xcnn_unavailable": str(error)}
        return _ENTRY, _CONTEXT
    selected = _backend.default_mode()
    bindings = {}
    for mode in (selected, "fast" if selected == "identical" else "identical"):  # glue: the two numeric modes, connected once
        try:
            bindings[mode] = _connect_mode(_backend.binding("_mojolearn_x_cnn", mode))
        except (ImportError, AttributeError, ValueError, RuntimeError, TypeError, OSError) as error:
            bindings[mode] = type(error).__name__ + ": " + str(error)
    context.update(xcnn_bindings=bindings)
    _ENTRY, _CONTEXT = entry, context
    return _ENTRY, _CONTEXT


def register(**tables):
    """Connect the layer module's classes and switches (`xcnn_*` names the
    owner reads) at its definition."""
    _, context = connect()
    if "xcnn_unavailable" not in context:
        context.update(tables)
