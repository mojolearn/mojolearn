# SPDX-License-Identifier: Apache-2.0
"""The per-part numeric profile at the Python boundary, and the case table of
the verify lane `numeric-profile-parts` (tools/identity_lanes/numeric_profile_parts.py).

The interface is docs/plans/NUMERIC_PROFILE_PARTS.md: a spec is a preset name
or an exact dict with "preset" and part overrides; an unknown part, a part the
family does not have (lm_head on a bare TransformerBlock), a format the part
cannot use (attn_context fixed15_v1) and an unregistered preset are refused by
name; `model.numeric_profile` is the canonical form. The model tests skip by
name when the package cannot build a model on this box.
"""
import os
from pathlib import Path

import pytest

PARTS = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj", "lm_head",
         "attn_scores", "attn_context")
FRAGMENT = Path(__file__).resolve().parents[3] / "tools" / "identity_lanes" / "numeric_profile_parts.py"


def _package():
    os.environ.setdefault("MOJOLEARN_NUMERIC_MODE", "identical")
    try:
        import mojolearn as ml
    except Exception as e:  # noqa: BLE001
        pytest.skip(f"the package does not import on this box: {e}")
    return ml


def _root(tmp_path):
    from mojolearn import _causal_lm_fixtures as fx
    cfg, tensors = fx.family_fixture("llama", False)
    return fx._write_checkpoint(str(tmp_path / "llama"), cfg, tensors)


def _load(root, **kw):
    from mojolearn.models import CausalLM
    try:
        return CausalLM.load(root, device="cpu", **kw)
    except (ImportError, OSError) as e:
        pytest.skip(f"the model's bindings are not built here: {e}")


def _block(ml, **kw):
    dm, it = 32, 64

    def f32(shape, seed):
        n = 1
        for s in shape:
            n *= s
        return ml.Array.from_list([((i * 2654435761 + seed) % 1000) / 4000.0 - 0.125 for i in range(n)],
                                  "<f4").reshape(shape)
    w = {"input_layernorm.weight": f32((dm,), 1), "post_attention_layernorm.weight": f32((dm,), 2),
         "q_proj.weight": f32((dm, dm), 3), "k_proj.weight": f32((dm, dm), 4),
         "v_proj.weight": f32((dm, dm), 5), "o_proj.weight": f32((dm, dm), 6),
         "gate_proj.weight": f32((it, dm), 7), "up_proj.weight": f32((it, dm), 8),
         "down_proj.weight": f32((dm, it), 9)}
    try:
        return ml.TransformerBlock(w, n_heads=2, device="cpu", **kw)
    except (ImportError, OSError) as e:
        pytest.skip(f"no transformer binding here: {e}")


# ------------------------------------------------------------ the verify lane's case table

def _fragment():
    """The lane fragment's module-level data, executed with the three harness
    names it calls at import replaced by no-ops (the lane bodies are not run)."""
    if not FRAGMENT.is_file():
        pytest.skip("the verify lane fragments are not in this tree (an installed wheel)")
    ns = {"lane": lambda name: (lambda fn: fn), "_one_sided": lambda *a, **k: None,
          "_batch_decl": lambda *a, **k: None}
    exec(compile(FRAGMENT.read_text(encoding="utf-8"), str(FRAGMENT), "exec"), ns)  # noqa: S102
    return ns


def test_the_lane_has_one_case_per_part_and_direction():
    ns = _fragment()
    assert ns["_NPP_PARTS"] == PARTS
    cases = {c["name"]: c for c in ns["_NPP_CASES"]}
    assert len(cases) == len(ns["_NPP_CASES"])
    for part in PARTS:
        if part == "attn_context":
            for preset in ("fp32_v1", "fixed15_v1"):
                c = cases[f"{preset}-{part}-refused"]
                assert c["spec"] == {"preset": preset, part: "fixed15_v1"} and c["refused"] == part
            assert cases[f"f15-{part}-same"]["canonical"] == "fixed15_v1"
            continue
        down, up = cases[f"f15-{part}"], cases[f"f32-{part}"]
        assert down["spec"] == down["canonical"] == {"preset": "fixed15_v1", part: "fp32_v1"}
        assert up["spec"] == up["canonical"] == {"preset": "fp32_v1", part: "fixed15_v1"}
        assert down["relation"] == up["relation"] == "differs"
    assert ns["_NPP_SAVED_SPEC"]["preset"] == "fixed15_v1" and len(ns["_NPP_SAVED_SPEC"]) == 3
    assert "lm_head" not in ns["_NPP_BLOCK_SPEC"]


def test_the_lane_parts_are_the_package_parts():
    _package()
    from mojolearn import _numeric_profile as npm
    assert tuple(npm.PARTS) == PARTS == _fragment()["_NPP_PARTS"]


# ------------------------------------------------------------ the boundary

def test_the_default_and_the_canonical_form(tmp_path):
    ml = _package()
    root = _root(tmp_path)
    assert _load(root).numeric_profile == "fixed15_v1"
    assert _load(root, numeric_profile="fp32_v1").numeric_profile == "fp32_v1"
    assert _load(root, numeric_profile={"preset": "fp32_v1"}).numeric_profile == "fp32_v1"
    # an override equal to the preset's own format is dropped; the others are kept
    spec = {"preset": "fixed15_v1", "lm_head": "fp32_v1", "q_proj": "fixed15_v1", "attn_context": "fp32_v1"}
    assert _load(root, numeric_profile=spec).numeric_profile == {"preset": "fixed15_v1", "lm_head": "fp32_v1"}
    assert _load(root, numeric_profile={"preset": "fixed15_v1", "attn_context": "fp32_v1"}).numeric_profile \
        == "fixed15_v1"
    up = {"preset": "fp32_v1", "attn_scores": "fixed15_v1"}
    assert _load(root, numeric_profile=up).numeric_profile == up
    # a bare TransformerBlock named without a profile stays fp32_v1; a block spec reads back canonical
    assert _block(ml).numeric_profile == "fp32_v1"
    assert _block(ml, numeric_profile=dict(up)).numeric_profile == up
    assert _block(ml, numeric_profile={"preset": "fp32_v1", "q_proj": "fp32_v1"}).numeric_profile == "fp32_v1"


@pytest.mark.parametrize("spec,word", [
    ({"preset": "fixed15_v1", "nonsense_proj": "fp32_v1"}, "nonsense_proj"),       # an unknown part
    ({"preset": "fp32_v1", "attn_context": "fixed15_v1"}, "attn_context"),         # a format the part cannot use
    ({"preset": "fixed15_v1", "attn_context": "fixed15_v1"}, "attn_context"),
    ({"preset": "nonsense_v9"}, "nonsense_v9"),                                     # a bad preset
    ({"preset": "fixed15_v1", "q_proj": "nonsense_v9"}, "nonsense_v9"),             # a bad part value
    ({"q_proj": "fp32_v1"}, "preset"),                                              # no preset
    ("nonsense_v9", "nonsense_v9"),
])
def test_a_bad_spec_is_refused_by_name(tmp_path, spec, word):
    _package()
    root = _root(tmp_path)
    _load(root, numeric_profile="fp32_v1")  # skips here when the bindings are not built
    from mojolearn.models import CausalLM
    with pytest.raises((ValueError, TypeError)) as e:
        CausalLM.load(root, device="cpu", numeric_profile=spec)
    assert word in str(e.value)


def test_lm_head_is_not_a_block_part():
    ml = _package()
    _block(ml)  # skips here when the binding is not built
    with pytest.raises((ValueError, TypeError)) as e:
        _block(ml, numeric_profile={"preset": "fixed15_v1", "lm_head": "fp32_v1"})
    assert "lm_head" in str(e.value)


def test_a_trainer_refuses_a_dict_spec():
    _package()
    from mojolearn import _numeric_profile as npm
    with pytest.raises((ValueError, TypeError, NotImplementedError)):
        npm.require_training("trainer", {"preset": "fp32_v1", "q_proj": "fixed15_v1"})


# ------------------------------------------------------------ generation route under the default (lane fixed15-decode)

@pytest.mark.parametrize("spec", [None, {"preset": "fixed15_v1", "lm_head": "fp32_v1", "attn_scores": "fp32_v1"}])
def test_cpu_forward_under_fixed15_runs_the_native_engine(tmp_path, spec):
    """Lane fixed15-decode (2026-11-25): on the CPU the fixed15_v1 default (and
    a per-part spec) runs the native generation engine (its route word on the
    state), not the slower per-layer route the engine used to fall back to;
    the logits equal a forward without a state."""
    ml = _package()
    root = _root(tmp_path)
    m = _load(root) if spec is None else _load(root, numeric_profile=spec)
    ids = ml.Array.from_list([1, 2, 3, 4, 5, 6], "<i4").reshape((2, 3))
    state = m.allocate_state(2, 8)
    y = m.forward(ids, state)
    assert getattr(state, "_native_execution_route", None) == "transformer-nonresident-native-engine-v1"
    assert getattr(state, "_native_engine_forwards", None) == 1
    import numpy as np
    assert np.asarray(y).tobytes() == np.asarray(m.forward(ids)).tobytes()


def test_cpu_stream_under_fixed15_matches_generate(tmp_path):
    """The default's stream and `generate` give the same continuation on the
    CPU (both the engine since lane fixed15-decode)."""
    ml = _package()
    root = _root(tmp_path)
    m = _load(root)
    ids = ml.Array.from_list([1, 2, 3, 4, 5, 6], "<i4").reshape((2, 3))
    import numpy as np
    tokens = np.asarray(m.stream(ids, 4).read(4))
    full = np.asarray(m.generate(ids, 4))[:, 3:]
    assert tokens.tobytes() == np.ascontiguousarray(full, dtype=tokens.dtype).tobytes()
