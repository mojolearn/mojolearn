"""Actual installed neural dtype refusals; no conversion or fake providers."""
import numpy as np
import pytest
import mojolearn as ml


def _block(version):
    if version == 1:
        dm, di = 8, 16
        shapes = {"norm.weight": (dm,), "in_proj.weight": (2 * di, dm),
                  "conv1d.weight": (di, 1, 4), "conv1d.bias": (di,),
                  "x_proj.weight": (33, di), "dt_proj.weight": (di, 1),
                  "dt_proj.bias": (di,), "A_log": (di, 16), "D": (di,),
                  "out_proj.weight": (dm, di)}
        cls = ml.Mamba1Block
    else:
        dm, di, heads = 32, 64, 1
        shapes = {"block_norm.weight": (dm,),
                  "in_proj.weight": (2 * di + 256 + 3 * heads + 32, dm),
                  "dt_bias": (heads,), "B_norm.weight": (128,),
                  "C_norm.weight": (128,), "B_bias": (heads, 128),
                  "C_bias": (heads, 128), "D": (heads,),
                  "out_proj.weight": (dm, di)}
        cls = ml.Mamba3Block
    return cls({key: np.zeros(shape, np.float32) for key, shape in shapes.items()}), dm


@pytest.mark.parametrize("version", [1, 3])
@pytest.mark.parametrize("array", [False, True])
def test_float64_refusal_preserves_admission_order_and_dtype_policy(version, array):
    block, dm = _block(version)
    data = np.zeros((1, 2, dm), np.float64)
    value = ml.Array.from_list(data.tolist(), dtype="<f8") if array else data
    before = data.tobytes()
    with pytest.raises(TypeError) as caught:
        block.forward(value)
    message = str(caught.value)
    if array:
        # The existing restricted Array snapshot refuses f8 before the float32
        # check. Preserve that validation order rather than broaden admission
        # merely to publish a more specific downstream message.
        assert "neural Array storage dtype or rank is inconsistent" in message
    else:
        assert "has dtype float64" in message
        assert "float32 only" in message
        assert "bfloat16" in message  # Existing public policy-message contract.
    assert data.tobytes() == before


@pytest.mark.parametrize("array", [False, True])
def test_uint16_bits_are_named_and_refused_without_conversion(array):
    block, dm = _block(1)
    data = np.zeros((1, 2, dm), np.uint16)
    value = ml.Array.from_list(data.tolist(), dtype="<u2") if array else data
    with pytest.raises(TypeError, match=r"has dtype uint16 \(bfloat16 bits\)"):
        block.forward(value)


@pytest.mark.parametrize("version", [1, 3])
def test_float32_public_forward_still_returns_float32(version):
    block, dm = _block(version)
    result = block.forward(np.zeros((1, 2, dm), np.float32))
    view = np.asarray(result)
    assert view.shape == (1, 2, dm)
    assert view.dtype == np.float32
    assert np.isfinite(view).all()


def _transformer():
    dm, width, kv, hidden = 32, 32, 16, 64
    shapes = {"input_layernorm.weight": (dm,),
              "post_attention_layernorm.weight": (dm,),
              "q_proj.weight": (width, dm), "k_proj.weight": (kv, dm),
              "v_proj.weight": (kv, dm), "o_proj.weight": (dm, width),
              "gate_proj.weight": (hidden, dm), "up_proj.weight": (hidden, dm),
              "down_proj.weight": (dm, hidden)}
    return ml.TransformerBlock({k: np.zeros(s, np.float32) for k, s in shapes.items()},
                               n_heads=2, n_kv_heads=1)


def test_transformer_float64_forward_refusal_uses_same_admitted_dtype_message():
    with pytest.raises(TypeError) as caught:
        _transformer().forward(np.zeros((1, 2, 32), np.float64))
    assert "has dtype float64" in str(caught.value)
    assert "bfloat16" in str(caught.value)


def test_transformer_float64_gradient_is_named_before_execution():
    with pytest.raises(TypeError) as caught:
        _transformer().backward(np.zeros((1, 2, 32), np.float32),
                                np.zeros((1, 2, 32), np.float64))
    assert "grad_output has dtype float64" in str(caught.value)
