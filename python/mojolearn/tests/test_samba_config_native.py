# SPDX-License-Identifier: Apache-2.0
"""Independent Samba registry layout and complete-operation native audits."""
import struct

import pytest

from mojolearn import SambaConfig
from mojolearn._verify_native_runtime import audit_native_call


MAMBA = [
    ("block_norm.weight", (32,)), ("in_proj.weight", (419, 32)),
    ("dt_bias", (1,)), ("B_norm.weight", (128,)), ("C_norm.weight", (128,)),
    ("B_bias", (1, 128)), ("C_bias", (1, 128)), ("D", (1,)),
    ("out_proj.weight", (32, 64)),
]
ATTENTION = [
    ("input_layernorm.weight", (32,)), ("post_attention_layernorm.weight", (32,)),
    ("q_proj.weight", (32, 32)), ("k_proj.weight", (16, 32)),
    ("v_proj.weight", (16, 32)), ("o_proj.weight", (32, 32)),
    ("gate_proj.weight", (48, 32)), ("up_proj.weight", (48, 32)),
    ("down_proj.weight", (32, 48)),
]


def _blocked(*args, **kwargs):
    raise AssertionError("Python configuration orchestration was called")


@pytest.mark.parametrize("tie", [True, False])
def test_native_samba_registry_preserves_order_and_shapes(monkeypatch, tie):
    config = SambaConfig.__new__(SambaConfig)
    audit_native_call(SambaConfig.__init__, config, 37, 32,
                      ["mamba3", "attention"], n_heads=2, n_kv_heads=1,
                      intermediate=48, tie_embeddings=tie).assert_native()
    assert audit_native_call(config.block_shapes, "mamba3").assert_native() == MAMBA
    assert audit_native_call(config.block_shapes, "attention").assert_native() == ATTENTION
    expected = [("embed.weight", (37, 32))]
    expected += [("layers.0." + name, shape) for name, shape in MAMBA]
    expected += [("layers.1." + name, shape) for name, shape in ATTENTION]
    expected += [("norm_f.weight", (32,))]
    if not tie:
        expected += [("lm_head.weight", (37, 32))]
    with monkeypatch.context() as patch:
        patch.setattr(SambaConfig, "block_shapes", _blocked)
        assert audit_native_call(config.registry).assert_native() == expected


def test_native_samba_config_roundtrip_and_float32_semantics(monkeypatch):
    config = SambaConfig(37, 32, ("attention",), n_heads=2, intermediate=48,
                         dropout=0.1, norm_eps=1e-5)
    encoded = audit_native_call(config.to_dict).assert_native()
    f32 = lambda x: struct.unpack("<f", struct.pack("<f", x))[0]
    assert encoded == {"vocab": 37, "d_model": 32, "layers": ["attention"],
                       "n_heads": 2, "n_kv_heads": 2, "head_dim": 16,
                       "intermediate": 48, "tie_embeddings": True,
                       "norm_eps": f32(1e-5), "dropout": f32(0.1)}
    with monkeypatch.context() as patch:
        patch.setattr(SambaConfig, "__init__", _blocked)
        decoded = audit_native_call(SambaConfig.from_dict, encoded).assert_native()
    assert type(decoded) is SambaConfig
    assert decoded.to_dict() == encoded


@pytest.mark.parametrize("kwargs,match", [
    ({"vocab": 1}, "vocab"), ({"layers": []}, "at least one"),
    ({"layers": ["unknown"]}, "layer kind"),
    ({"d_model": 33}, "multiple of 32"),
    ({"dropout": float("nan")}, "dropout"),
    ({"norm_eps": -1.0}, "norm_eps"),
])
def test_native_samba_config_refusals_are_inside_boundary(kwargs, match):
    values = {"vocab": 37, "d_model": 32, "layers": ["mamba3"]}
    values.update(kwargs)
    result = audit_native_call(SambaConfig.__init__, SambaConfig.__new__(SambaConfig), **values)
    assert isinstance(result.exception, ValueError), result.report()
    assert match in str(result.exception)
    report = result.report()
    for key in ("python_frames", "forbidden_native_calls", "wrapper_work", "unobserved_existing_threads"):
        assert not report[key], report
