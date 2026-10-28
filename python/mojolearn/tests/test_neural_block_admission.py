"""Public hook controls; source syntax checking is not runtime qualification."""
import pytest


@pytest.fixture(params=["transformer", "mamba1", "mamba2"])
def block_input(request):
    import numpy as np
    if request.param == "transformer":
        from mojolearn import TransformerBlock
        from mojolearn.tests.test_transformer_options import _weights, DM, NH, NKV, HD
        block = TransformerBlock(_weights(), n_heads=NH, n_kv_heads=NKV, head_dim=HD)
        width = DM
    else:
        from mojolearn.tests.test_mamba_native_public import _block
        block, width = _block(request.param)
    return block, np.zeros((1, 1, width), dtype=np.float32)


def test_replaced_dictionary_descriptor_is_refused_without_execution(block_input):
    block, x = block_input
    original = type(block)
    forward = block.forward
    calls = []

    class Changed(original):
        __slots__ = ()

        @property
        def __dict__(self):
            calls.append("dictionary")
            return {}

    block.__class__ = Changed
    try:
        with pytest.raises(TypeError):
            forward(x)
        assert calls == []
    finally:
        block.__class__ = original


@pytest.mark.parametrize("field", ["d_model", "numeric_mode", "_native_route", "_W_NAMES"])
def test_mutable_model_words_cannot_call_python_conversion(block_input, field):
    block, x = block_input
    calls = []

    class Hook:
        def __int__(self):
            calls.append("int")
            return 1

        def __index__(self):
            calls.append("index")
            return 1

        def __str__(self):
            calls.append("str")
            return "identical"

        def __len__(self):
            calls.append("len")
            return 0

        def strip(self):
            calls.append("strip")
            return "identical"

    fields = vars(block)
    missing = object()
    old = fields.get(field, missing)
    fields[field] = Hook()
    try:
        with pytest.raises(TypeError):
            block.forward(x)
        assert calls == []
    finally:
        if old is missing:
            del fields[field]
        else:
            fields[field] = old


def test_allocate_refuses_index_hook(block_input):
    block, _ = block_input
    calls = []

    class Count:
        def __index__(self):
            calls.append("index")
            return 1

    with pytest.raises(TypeError):
        block.allocate_state(Count())
    assert calls == []


def test_input_subclass_shape_hook_never_runs(block_input):
    import numpy as np
    block, x = block_input
    calls = []

    class HookArray(np.ndarray):
        @property
        def shape(self):
            calls.append("shape")
            return (1, 1, x.shape[-1])

    with pytest.raises(TypeError):
        block.forward(x.view(HookArray))
    assert calls == []
