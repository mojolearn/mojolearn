# SPDX-License-Identifier: Apache-2.0
"""Per-part numeric profile specs at the Python boundary (lane fixed15-train,
2026-11-25; docs/plans/NUMERIC_PROFILE_PARTS.md section 2).

A spec is a preset name or an exact dict {"preset": name, part: name, ...}.
The compiled profile door validates it, refuses by name, and returns the
canonical form; these tests read only that boundary (the per-part bits are
the native checks' and verify's).
"""
import copy
import os
import warnings

import pytest


def _fresh(monkeypatch, env=None):
    if env is None:
        monkeypatch.delenv("MOJOLEARN_NUMERIC_PROFILE", raising=False)
    else:
        monkeypatch.setenv("MOJOLEARN_NUMERIC_PROFILE", env)
    os.environ.setdefault("MOJOLEARN_NUMERIC_MODE", "identical")
    try:
        from mojolearn import _numeric_profile as mod
    except Exception as e:  # noqa: BLE001
        pytest.skip(f"the package does not import on this box: {e}")
    for name in ("PROFILES", "REJECTED", "MEASURED", "PART_FORMATS", "FAMILY_PARTS"):
        monkeypatch.setattr(mod, name, copy.deepcopy(getattr(mod, name)))
    monkeypatch.setattr(mod, "_warned", set())
    for use in list(mod._defaults):
        monkeypatch.delitem(mod._defaults, use)
    return mod


def _quiet(call, *args, **kw):
    # the Apple speed warning, when this runs on a Mac
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return call(*args, **kw)


ALL = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj",
       "lm_head", "attn_scores", "attn_context")


def test_the_parts_table_matches_the_native_order(monkeypatch):
    g = _fresh(monkeypatch)
    assert g.PARTS == ALL
    assert set(g.PART_FORMATS) == set(ALL)
    assert g.PART_FORMATS["attn_context"] == ("fp32_v1",)
    for part in ALL[:-1]:
        assert set(g.PART_FORMATS[part]) == {"fp32_v1", "fixed15_v1"}, part
    assert g.FAMILY_PARTS["transformer"] == ALL
    assert "lm_head" not in g.FAMILY_PARTS["transformer_block"]
    assert set(g.FAMILY_PARTS["transformer_block"]) == set(ALL) - {"lm_head"}
    assert set(g.FAMILY_PARTS) == set(g.FAMILIES)
    # the native header lists the same order (core/numeric_profile_parts.mojo)
    here = os.path.dirname(os.path.abspath(__file__))
    header = os.path.join(here, "..", "..", "..", "core", "numeric_profile_parts.mojo")
    if os.path.exists(header):
        text = open(header, encoding="utf-8").read()
        for part in ALL:
            assert f'return "{part}"' in text, part
        positions = [text.index(f'return "{part}"') for part in ALL]
        assert positions == sorted(positions)


def test_defaults_inference_fixed15_training_fp32(monkeypatch):
    g = _fresh(monkeypatch)
    assert g.default_profile() == "fixed15_v1"
    assert g.default_profile("training") == "fp32_v1"
    assert _quiet(g.resolve, None, family="transformer") == "fixed15_v1"
    assert g.resolve(None, family="transformer_block") == "fp32_v1"
    assert g.require_training("a trainer") == "fp32_v1"


def test_the_opt_out_by_environment_and_by_call(monkeypatch):
    g = _fresh(monkeypatch, env="fp32_v1")
    assert g.default_profile() == "fp32_v1"
    assert g.resolve(None, family="transformer") == "fp32_v1"
    g = _fresh(monkeypatch)
    assert g.set_default_profile("fp32_v1") == "fixed15_v1"
    assert g.resolve(None, family="transformer") == "fp32_v1"
    assert g.default_profile("training") == "fp32_v1"


def test_canonicalisation(monkeypatch):
    g = _fresh(monkeypatch)
    r = lambda spec, **kw: _quiet(g.resolve, spec, family="transformer", **kw)  # noqa: E731
    # a dict with no override, or only overrides equal to the preset, is the preset name
    assert r({"preset": "fixed15_v1"}) == "fixed15_v1"
    assert r({"preset": "fp32_v1"}) == "fp32_v1"
    assert r({"preset": "fixed15_v1", "q_proj": "fixed15_v1"}) == "fixed15_v1"
    assert r({"preset": "fp32_v1", "lm_head": "fp32_v1"}) == "fp32_v1"
    # attn_context is fp32_v1 under the fixed15_v1 preset already
    assert r({"preset": "fixed15_v1", "attn_context": "fp32_v1"}) == "fixed15_v1"
    # differing overrides stay, preset first, then table order, whatever the input order
    got = r({"lm_head": "fp32_v1", "q_proj": "fp32_v1", "preset": "fixed15_v1", "k_proj": "fixed15_v1"})
    assert got == {"preset": "fixed15_v1", "q_proj": "fp32_v1", "lm_head": "fp32_v1"}
    assert list(got) == ["preset", "q_proj", "lm_head"]
    got = r({"preset": "fp32_v1", "attn_scores": "fixed15_v1"})
    assert got == {"preset": "fp32_v1", "attn_scores": "fixed15_v1"}
    # names are canonicalised like a preset name (whitespace)
    assert r({"preset": " fp32_v1 ", "q_proj": " fixed15_v1"}) == {"preset": "fp32_v1", "q_proj": "fixed15_v1"}
    # canonical() is the same form, no family
    assert g.canonical({"preset": "fixed15_v1", "lm_head": "fixed15_v1"}) == "fixed15_v1"
    assert g.canonical("fp32_v1") == "fp32_v1"
    # the canonical form resolves to itself
    spec = {"preset": "fixed15_v1", "down_proj": "fp32_v1"}
    assert r(r(spec)) == r(spec)
    # the caller's dict is not changed
    raw = {"preset": "fixed15_v1", "q_proj": "fixed15_v1"}
    r(raw)
    assert raw == {"preset": "fixed15_v1", "q_proj": "fixed15_v1"}


def test_the_whole_parts_table(monkeypatch):
    g = _fresh(monkeypatch)
    t = g.parts("fixed15_v1")
    assert list(t) == list(ALL)
    assert all(t[p] == "fixed15_v1" for p in ALL[:-1]) and t["attn_context"] == "fp32_v1"
    assert set(g.parts("fp32_v1").values()) == {"fp32_v1"}
    t = g.parts({"preset": "fp32_v1", "q_proj": "fixed15_v1"}, "transformer")
    assert [p for p in ALL if t[p] == "fixed15_v1"] == ["q_proj"]


@pytest.mark.parametrize("spec,family,error,word", [
    ({"preset": "fixed15_v1", "no_such_part": "fp32_v1"}, "transformer", ValueError, "no_such_part"),
    ({"preset": "fixed15_v1", "lm_head": "fp32_v1"}, "transformer_block", ValueError, "lm_head"),
    ({"preset": "fp32_v1", "q_proj": "fixed15_v1"}, "mamba1", ValueError, "q_proj"),
    ({"preset": "fp32_v1", "attn_context": "fixed15_v1"}, "transformer", ValueError, "attn_context"),
    ({"preset": "fixed15_v1", "attn_context": "fixed15_v1"}, None, ValueError, "attn_context"),
    ({"preset": "nonsense_v9"}, "transformer", ValueError, "nonsense_v9"),
    ({"preset": "fixed15_v1", "q_proj": "nonsense_v9"}, "transformer", ValueError, "nonsense_v9"),
    ({"preset": "fixed15_v1", "q_proj": "int8_v1"}, "transformer", ValueError, "not offered"),
    ({"preset": "int8_v1"}, "transformer", ValueError, "not offered"),
    ({"q_proj": "fp32_v1"}, "transformer", ValueError, "preset"),
    ({}, "transformer", ValueError, "preset"),
    ({"preset": "fixed15_v1", 3: "fp32_v1"}, "transformer", TypeError, "part names"),
    ({"preset": "fixed15_v1", "q_proj": 1}, "transformer", TypeError, "q_proj"),
    ({"preset": 15}, "transformer", TypeError, "preset"),
])
def test_each_refusal_by_name(monkeypatch, spec, family, error, word):
    g = _fresh(monkeypatch)
    with pytest.raises(error) as e:
        _quiet(g.resolve, spec, family=family)
    assert word in str(e.value)


def test_a_dict_is_refused_for_training(monkeypatch):
    g = _fresh(monkeypatch)
    for spec in ({"preset": "fp32_v1"}, {"preset": "fp32_v1", "q_proj": "fixed15_v1"}):
        with pytest.raises(ValueError) as e:
            g.resolve(spec, use="training")
        assert "inference only" in str(e.value)
        with pytest.raises(ValueError):
            g.require_training("a trainer", spec)
    # and set_numeric_profile takes a name, not a dict
    with pytest.raises(TypeError):
        g.set_default_profile({"preset": "fp32_v1"})


def test_a_dict_naming_a_profile_the_family_cannot_compute_is_refused(monkeypatch):
    g = _fresh(monkeypatch)
    # the mamba families have no parts; a bare preset dict is the preset's own refusal
    with pytest.raises(NotImplementedError) as e:
        g.resolve({"preset": "fixed15_v1"}, family="mamba2")
    assert "mamba2" in str(e.value)
    assert g.resolve({"preset": "fp32_v1"}, family="mamba2") == "fp32_v1"


def test_a_closed_profile_named_only_in_a_part_is_refused(monkeypatch):
    g = _fresh(monkeypatch)
    g.PROFILES["fixed15_v1"] = dict(g.PROFILES["fixed15_v1"], inference=False)
    with pytest.raises(NotImplementedError) as e:
        g.resolve({"preset": "fp32_v1", "q_proj": "fixed15_v1"}, family="transformer")
    assert "fixed15_v1" in str(e.value)
    # the sabotage arm: open it and the same spec passes
    g.PROFILES["fixed15_v1"] = dict(g.PROFILES["fixed15_v1"], inference=True)
    assert _quiet(g.resolve, {"preset": "fp32_v1", "q_proj": "fixed15_v1"}, family="transformer") == \
        {"preset": "fp32_v1", "q_proj": "fixed15_v1"}


def test_the_part_formats_table_decides(monkeypatch):
    """The sabotage arm of the attn_context refusal: the refusal is read from
    PART_FORMATS, not a name frozen in the door."""
    g = _fresh(monkeypatch)
    spec = {"preset": "fp32_v1", "attn_context": "fixed15_v1"}
    with pytest.raises(ValueError):
        g.resolve(spec, family="transformer")
    g.PART_FORMATS["attn_context"] = ("fp32_v1", "fixed15_v1")
    assert _quiet(g.resolve, spec, family="transformer") == spec


def test_state_round_trip_of_a_dict_spec(monkeypatch):
    g = _fresh(monkeypatch)
    spec = _quiet(g.resolve, {"preset": "fixed15_v1", "lm_head": "fp32_v1"}, family="transformer")
    field = g.state_field(spec)
    assert field == {g.STATE_KEY: {"preset": "fixed15_v1", "lm_head": "fp32_v1"}}
    assert field[g.STATE_KEY] is not spec
    state = dict(field, weights="...")
    # read back whole: adopted with no keyword, checked against the same spec
    assert g.adopt_saved(state) == spec
    assert g.check_saved(state, spec) == spec
    assert g.check_saved(state, {"lm_head": "fp32_v1", "preset": "fixed15_v1"}) == spec
    # a different spec, or the bare preset, is refused by name
    for other in ("fixed15_v1", "fp32_v1", {"preset": "fixed15_v1", "q_proj": "fp32_v1"}):
        with pytest.raises(ValueError) as e:
            g.check_saved(state, other)
        assert "lm_head" in str(e.value)
        with pytest.raises(ValueError):
            g.adopt_saved(state, other)
    # a dict that only restates its preset is stored as the name
    assert g.state_field({"preset": "fixed15_v1", "q_proj": "fixed15_v1"}) == {g.STATE_KEY: "fixed15_v1"}
    assert g.state_field({"preset": "fp32_v1"}) == {}
    # a saved dict is not adopted by a trainer, and a bad saved dict is refused by name
    with pytest.raises(ValueError):
        g.adopt_saved(state, use="training")
    with pytest.raises(ValueError) as e:
        g.adopt_saved({g.STATE_KEY: {"preset": "fixed15_v1", "no_such_part": "fp32_v1"}})
    assert "no_such_part" in str(e.value)


def test_the_package_reports_the_default(monkeypatch):
    monkeypatch.delenv("MOJOLEARN_NUMERIC_PROFILE", raising=False)
    try:
        import mojolearn as ml
        from mojolearn import _numeric_profile as g
    except Exception as e:  # noqa: BLE001
        pytest.skip(f"the package does not import on this box: {e}")
    for use in list(g._defaults):
        monkeypatch.delitem(g._defaults, use)
    assert ml.numeric_profile() == "fixed15_v1"
    assert ml.numeric_profile("training") == "fp32_v1"
    prev = ml.set_numeric_profile("fp32_v1")
    try:
        assert prev == "fixed15_v1" and ml.numeric_profile() == "fp32_v1"
    finally:
        ml.set_numeric_profile(prev)
