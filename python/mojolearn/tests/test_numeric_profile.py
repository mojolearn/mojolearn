# SPDX-License-Identifier: Apache-2.0
"""The `numeric_profile=` selector (`mojolearn._numeric_profile`): the
inference default is fp32_v1 for every family and every trainer;
fixed15_v1 requires opt-in; a named profile that cannot compute
is refused by name and never replaced; an arithmetic that failed quality is
not offered; a checkpoint carries its profile, and a state with none is
fp32_v1. Lanes lane/lowbit-flag and lane/lowbit-default, 2026-09-29.

The selector tests load the module BY PATH, so they need no binding and no
GPU. The model tests import the package and skip by name when it cannot
build a model on this box.
"""
import importlib.util
import os
import pathlib
import warnings

import pytest

_HERE = pathlib.Path(__file__).resolve().parent
_SRC = _HERE.parent / "_numeric_profile.py"


def _fresh(monkeypatch, env=None):
    """A private copy of the module, so a test's process default never
    leaks into another test or into the package."""
    if env is None:
        monkeypatch.delenv("MOJOLEARN_NUMERIC_PROFILE", raising=False)
    else:
        monkeypatch.setenv("MOJOLEARN_NUMERIC_PROFILE", env)
    spec = importlib.util.spec_from_file_location("_numeric_profile_under_test", _SRC)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _untrainable(g):
    names = [k for k, v in g.PROFILES.items() if v["inference"] and not v["training"]]
    assert names, "this test must have a profile inference takes and training refuses"
    return names


def _closed(g, name="fixed15_v1"):
    """Plant an unavailable row in the private copy: the refusal tests read
    the registry, not a name frozen in the test."""
    g.PROFILES[name] = dict(g.PROFILES[name], inference=False, training=False)
    return name


def test_the_defaults_per_use_and_family(monkeypatch):
    g = _fresh(monkeypatch)
    assert g.BASELINE == g.DEFAULT == g.TRAINING_DEFAULT == "fp32_v1"
    assert g.default_profile() == "fp32_v1"
    assert g.default_profile("inference") == "fp32_v1"
    assert g.default_profile("training") == "fp32_v1"
    # the families: the transformer models get the default, the rest fp32_v1, reported
    assert g.resolve(None, family="transformer") == "fp32_v1"
    for fam in ("mamba1", "mamba2", "transformer_block"):
        assert g.resolve(None, family=fam) == "fp32_v1", fam
    assert g.resolve(None) == "fp32_v1"
    # training never sees the inference default
    assert g.resolve(None, use="training") == "fp32_v1"
    assert g.require_training("a trainer") == "fp32_v1"
    for fam in g.FAMILIES:
        assert g.resolve(None, use="training", family=fam) == "fp32_v1"
    with pytest.raises(ValueError):
        g.resolve(None, family="gru")
    rows = {r["name"]: r for r in g.profiles()}
    assert [n for n, r in rows.items() if r["default"]] == ["fp32_v1"]
    assert [n for n, r in rows.items() if r["training_default"]] == ["fp32_v1"]
    assert list(g.PROFILES)[0] == "fp32_v1"
    assert g.PROFILES["fixed15_v1"]["inference"] is True and g.PROFILES["fixed15_v1"]["training"] is False
    assert g.PROFILES["fp32_v1"]["inference"] is True and g.PROFILES["fp32_v1"]["training"] is True


def test_the_escape_hatch_gives_fp32_v1_everywhere(monkeypatch):
    g = _fresh(monkeypatch, env="fp32_v1")
    for fam in (None,) + g.FAMILIES:
        assert g.resolve(None, family=fam) == "fp32_v1", fam
    g = _fresh(monkeypatch)
    assert g.set_default_profile("fp32_v1") == "fp32_v1"
    for fam in (None,) + g.FAMILIES:
        assert g.resolve(None, family=fam) == "fp32_v1", fam
    assert g.set_default_profile("fixed15_v1") == "fp32_v1"
    assert g.resolve(None, family="transformer") == "fixed15_v1"
    # the environment naming the default is the default
    g = _fresh(monkeypatch, env="fixed15_v1")
    assert g.resolve(None, family="transformer") == "fixed15_v1"


def test_a_named_profile_is_honored_or_refused_by_family(monkeypatch):
    g = _fresh(monkeypatch)
    assert g.resolve("fixed15_v1", family="transformer") == "fixed15_v1"
    assert g.resolve("fixed15_v1", family="transformer_block") == "fixed15_v1"
    for fam in ("mamba1", "mamba2"):
        with pytest.raises(NotImplementedError) as e:
            g.resolve("fixed15_v1", family=fam)
        assert "fixed15_v1" in str(e.value) and fam in str(e.value)
        assert g.resolve("fp32_v1", family=fam) == "fp32_v1"
    # the sabotage arm: a row that computes the family takes the name
    g.PROFILES["fixed15_v1"] = dict(g.PROFILES["fixed15_v1"], computes=g.FAMILIES)
    assert g.resolve("fixed15_v1", family="mamba1") == "fixed15_v1"
    # and a row the default reaches everywhere reaches the Mamba family
    g.PROFILES["fixed15_v1"] = dict(g.PROFILES["fixed15_v1"], default_for=g.FAMILIES)
    g.set_default_profile("fixed15_v1")
    assert g.resolve(None, family="mamba1") == "fixed15_v1"


def test_the_chosen_names_are_registered(monkeypatch):
    g = _fresh(monkeypatch)
    assert list(g.PROFILES) == ["fp32_v1", "fixed15_v1"]
    assert g.PROFILES["fixed15_v1"]["status"] == "experimental"
    assert g.PROFILES["fp32_v1"]["status"] == "baseline"
    # int8 and the int8-attention mix were dropped on 2026-09-29
    assert set(g.REJECTED) == {"int8_v1", "fixed15_int8_attention_v1"}


def test_every_name_carries_a_version_and_there_is_no_unversioned_alias(monkeypatch):
    g = _fresh(monkeypatch)
    for name in list(g.PROFILES) + list(g.REJECTED):
        stem, _, version = name.rpartition("_v")
        assert stem and version.isdigit(), name
    for bare in ("fp32", "fixed15", "fixed15_int8_attention", "int8", "fp32.v1", "FP32_V1"):
        with pytest.raises(ValueError):
            g.canonical(bare)
    assert g.canonical("  fp32_v1 ") == "fp32_v1"


def test_a_profile_says_which_gemm_each_product_family_runs(monkeypatch):
    g = _fresh(monkeypatch)
    for name, row in g.PROFILES.items():
        assert set(row["products"]) == {"projections", "attention"}, name
        for gemm in row["products"].values():
            assert gemm.startswith("mojolearn.identical.gemm.") and gemm.rsplit(".v", 1)[-1].isdigit()
        for key in ("computes", "default_for"):
            assert row[key] is None or set(row[key]) <= set(g.FAMILIES), (name, key)
        if row["default_for"] is not None:
            assert row["computes"] is not None and set(row["default_for"]) <= set(row["computes"])
    assert len(set(g.PROFILES["fp32_v1"]["products"].values())) == 1
    assert g.PROFILES["fp32_v1"]["computes"] is None
    # no offered profile runs an int8 GEMM as a model's arithmetic
    for row in g.PROFILES.values():
        assert not any("int8" in gemm for gemm in row["products"].values())


@pytest.mark.parametrize("bad", ["fp32", "fp32_v2", "bf16", "fast", "", "identical"])
def test_an_unregistered_name_is_refused_with_the_registered_names(monkeypatch, bad):
    g = _fresh(monkeypatch)
    with pytest.raises(ValueError) as e:
        g.resolve(bad)
    for name in g.PROFILES:
        assert name in str(e.value)


def test_what_failed_quality_is_not_offered_and_says_why(monkeypatch):
    g = _fresh(monkeypatch)
    assert "int8_v1" in g.REJECTED
    assert not set(g.REJECTED) & set(g.PROFILES)
    offered = [r["name"] for r in g.profiles()]
    for name, reason in g.REJECTED.items():
        assert name not in offered
        assert reason.startswith(("MEASURED", "DROPPED")), name
        for call in (g.resolve, g.canonical, g.set_default_profile, g.state_field):
            with pytest.raises(ValueError) as e:
                call(name)
            assert "not offered" in str(e.value) and reason in str(e.value)
    with pytest.raises(ValueError) as e:
        g.resolve("int8_v1")
    assert "32.2 percent" in str(e.value)
    with pytest.raises(ValueError):
        g.check_saved({g.STATE_KEY: "int8_v1"}, "fp32_v1")
    with pytest.raises(ValueError):
        g.adopt_saved({g.STATE_KEY: "int8_v1"})
    assert g.default_profile() == "fp32_v1"
    g2 = _fresh(monkeypatch, env="int8_v1")
    with pytest.raises(ValueError):
        g2.default_profile()


def test_a_measured_quality_is_a_number_per_text_and_under_the_bar(monkeypatch):
    g = _fresh(monkeypatch)
    for name, row in g.PROFILES.items():
        assert row["quality_note"], name
        for text, change in row["quality"].items():
            assert text and abs(change) < 0.01, (name, text, change)
    # the whole model under the profile, lane/lowbit-blocks (e): +0.0006% and +0.0033%
    assert g.PROFILES["fixed15_v1"]["quality"] == {"enwik8, last 1 MB": 0.000006, "pile_github": 0.000033}


def test_a_name_that_is_not_a_str_is_a_type_error(monkeypatch):
    g = _fresh(monkeypatch)
    for bad in (1, 1.0, b"fp32_v1", ("fp32_v1",)):
        with pytest.raises(TypeError):
            g.resolve(bad)


def test_an_unavailable_profile_is_refused_by_name_and_never_widened(monkeypatch):
    g = _fresh(monkeypatch)
    g.set_default_profile("fp32_v1")
    name = _closed(g)
    with pytest.raises(NotImplementedError) as e:
        g.resolve(name)
    assert name in str(e.value)
    with pytest.raises(NotImplementedError):
        g.resolve(name, family="transformer")
    with pytest.raises(NotImplementedError):
        g.set_default_profile(name)
    # a refused set left the default where it was
    assert g.default_profile() == "fp32_v1"


def test_refusal_follows_the_row_not_the_name(monkeypatch):
    """The sabotage arm of the test above: open the row again and the same
    call passes, so the refusal is read from the registry."""
    g = _fresh(monkeypatch)
    name = _closed(g)
    with pytest.raises(NotImplementedError):
        g.resolve(name)
    g.PROFILES[name] = dict(g.PROFILES[name], inference=True)
    assert g.resolve(name) == name
    prev = g.set_default_profile(name)
    assert g.default_profile() == name and g.resolve(None) == name


def test_a_closed_inference_default_is_refused_at_its_first_use_not_replaced(monkeypatch):
    g = _fresh(monkeypatch)
    g.set_default_profile("fixed15_v1")
    _closed(g, "fixed15_v1")
    with pytest.raises(NotImplementedError):
        g.resolve(None, family="transformer")
    # a family the default never reaches still gets fp32_v1
    assert g.resolve(None, family="mamba1") == "fp32_v1"


def test_training_is_its_own_gate(monkeypatch):
    g = _fresh(monkeypatch)
    name = _untrainable(g)[0]
    assert g.resolve(name) == name
    assert g.resolve(name, use="inference") == name
    with pytest.raises(NotImplementedError) as e:
        g.resolve(name, use="training")
    assert name in str(e.value) and "refused for training" in str(e.value)
    assert "available for inference" in str(e.value)
    # a trainer that NAMES it refuses it by name
    with pytest.raises(NotImplementedError) as e:
        g.require_training("a trainer", name)
    assert name in str(e.value) and "a trainer" in str(e.value)
    with pytest.raises(NotImplementedError):
        g.set_default_profile(name, use="training")
    assert g.default_profile("training") == "fp32_v1"
    # the inference default, set by name or by the environment, never reaches a trainer
    g.set_default_profile(name)
    assert g.require_training("a trainer") == "fp32_v1"
    g2 = _fresh(monkeypatch, env=name)
    assert g2.require_training("a trainer") == "fp32_v1"
    # the sabotage arm: open the training gate and the same named calls pass
    g.PROFILES[name] = dict(g.PROFILES[name], training=True)
    assert g.resolve(name, use="training") == name
    assert g.require_training("a trainer", name) == name
    assert g.set_default_profile(name, use="training") == "fp32_v1"
    assert g.require_training("a trainer") == name
    with pytest.raises(ValueError):
        g.resolve(name, use="fine-tuning")
    with pytest.raises(ValueError):
        g.default_profile("fine-tuning")
    for row in g.PROFILES.values():
        assert isinstance(row["inference"], bool) and isinstance(row["training"], bool)
        assert row["inference"] or not row["training"], "nothing trains that cannot infer"


def test_the_environment_sets_only_the_starting_value(monkeypatch):
    g = _fresh(monkeypatch, env="fp32_v1")
    assert g.default_profile() == "fp32_v1"
    g = _fresh(monkeypatch, env="nonsense_v9")
    with pytest.raises(ValueError):
        g.default_profile()


def test_an_fp32_checkpoint_gains_no_bytes_and_a_fixed15_one_carries_its_name(monkeypatch):
    g = _fresh(monkeypatch)
    assert g.state_field("fp32_v1") == {}
    assert g.state_field(g.DEFAULT) == {}
    assert g.state_field("fixed15_v1") == {g.STATE_KEY: "fixed15_v1"}
    assert g.STATE_KEY == "numeric_profile"


def test_a_state_is_read_only_under_the_profile_that_wrote_it(monkeypatch):
    g = _fresh(monkeypatch)
    other = "fixed15_v1"
    # no field: written under fp32_v1, by every version before this module
    assert g.check_saved({}, "fp32_v1") == "fp32_v1"
    assert g.check_saved({g.STATE_KEY: other}, other) == other
    for state, mine in (({g.STATE_KEY: other}, "fp32_v1"), ({}, other),
                        ({g.STATE_KEY: "fp32_v1"}, other)):
        with pytest.raises(ValueError) as e:
            g.check_saved(state, mine)
        assert "fp32_v1" in str(e.value) and other in str(e.value)
    with pytest.raises(ValueError):
        g.check_saved({g.STATE_KEY: "nonsense_v9"}, "fp32_v1")


def test_an_old_state_is_adopted_under_the_new_default(monkeypatch):
    g = _fresh(monkeypatch)
    g.set_default_profile("fixed15_v1")
    assert g.default_profile() == "fixed15_v1"
    # no field: fp32_v1, adopted, whatever the process default is now
    assert g.adopt_saved({}) == "fp32_v1"
    assert g.adopt_saved({}, None, use="training") == "fp32_v1"
    assert g.adopt_saved({}, "fp32_v1") == "fp32_v1"
    # a state written under the new default carries it and is adopted as such
    new = dict(g.state_field("fixed15_v1"))
    assert g.adopt_saved(new) == "fixed15_v1"
    # a caller who names a different profile is refused by name
    with pytest.raises(ValueError) as e:
        g.adopt_saved({}, "fixed15_v1")
    assert "fp32_v1" in str(e.value) and "fixed15_v1" in str(e.value)
    with pytest.raises(ValueError):
        g.adopt_saved(new, "fp32_v1")
    # a fixed15_v1 state cannot be adopted by a trainer
    with pytest.raises(NotImplementedError):
        g.adopt_saved(new, use="training")


def test_measured_rows_are_only_what_was_measured(monkeypatch):
    g = _fresh(monkeypatch)
    for profile, rows in g.MEASURED.items():
        assert profile in g.PROFILES
        for vendor, row in rows.items():
            assert vendor in ("cuda", "hip", "metal")
            lo, hi = row["over"]
            assert 0 < lo <= hi
            assert row["box"] and row["what"] and row["source"]
    assert g.measured("fp32_v1") == {}
    assert g.measured("fixed15_v1", "no-such-vendor") is None
    # quicker on NVIDIA and AMD, slower on Apple
    assert g.measured("fixed15_v1", "cuda")["over"][1] < 1
    assert g.measured("fixed15_v1", "hip")["over"][1] < 1
    assert g.measured("fixed15_v1", "metal")["over"][0] > 1
    by_name = {r["name"]: r for r in g.profiles()}
    assert by_name["fp32_v1"]["measured"] == {}


def test_a_profile_measured_slower_here_warns_once_and_still_resolves(monkeypatch):
    g = _fresh(monkeypatch)
    name = "fixed15_v1"
    g.set_default_profile(name)
    g.MEASURED[name] = {
        "cuda": {"over": (2.0, 4.0), "box": "a box", "what": "w", "source": "s"},
        "metal": {"over": (0.5, 0.7), "box": "a Mac", "what": "w", "source": "s"},
        "hip": {"over": (0.9, 1.3), "box": "straddles", "what": "w", "source": "s"},
    }
    monkeypatch.setattr(g, "_this_vendor", lambda: "cuda")
    with warnings.catch_warnings(record=True) as seen:
        warnings.simplefilter("always")
        assert g.resolve(name) == name
        assert g.resolve(None, family="transformer") == name
    slow = [w for w in seen if issubclass(w.category, g.NumericProfileSpeedWarning)]
    assert len(slow) == 1 and "2 to 4" in str(slow[0].message) and "a box" in str(slow[0].message)
    assert "fp32_v1" in str(slow[0].message)
    # quicker here, or a range that reaches below 1, or not measured here: silent
    for vendor in ("metal", "hip", None, "no-such-vendor"):
        monkeypatch.setattr(g, "_this_vendor", lambda v=vendor: v)
        with warnings.catch_warnings(record=True) as seen:
            warnings.simplefilter("always")
            assert g.resolve(name) == name
        assert not [w for w in seen if issubclass(w.category, g.NumericProfileSpeedWarning)], vendor
    # fp32_v1 never warns, and neither does a family the default does not reach
    monkeypatch.setattr(g, "_this_vendor", lambda: "cuda")
    g._warned.clear()
    with warnings.catch_warnings(record=True) as seen:
        warnings.simplefilter("always")
        assert g.resolve("fp32_v1") == "fp32_v1"
        assert g.resolve(None, family="mamba1") == "fp32_v1"
    assert not [w for w in seen if issubclass(w.category, g.NumericProfileSpeedWarning)]


def test_the_shipped_apple_row_warns_on_apple(monkeypatch):
    g = _fresh(monkeypatch)
    g.set_default_profile("fixed15_v1")
    monkeypatch.setattr(g, "_this_vendor", lambda: "metal")
    with warnings.catch_warnings(record=True) as seen:
        warnings.simplefilter("always")
        assert g.resolve(None, family="transformer") == "fixed15_v1"
    slow = [w for w in seen if issubclass(w.category, g.NumericProfileSpeedWarning)]
    assert len(slow) == 1 and "numeric_profile='fp32_v1'" in str(slow[0].message)


# --------------------------------------------------------------- the package


def _package():
    os.environ.setdefault("MOJOLEARN_NUMERIC_MODE", "identical")
    try:
        import mojolearn as ml
    except Exception as e:  # noqa: BLE001
        pytest.skip(f"the package does not import on this box: {e}")
    return ml


def _clean_default(ml):
    """The package's process default is the shipped one unless a test set
    it; every package test starts from the shipped default and the
    environment must not have changed it."""
    if os.environ.get("MOJOLEARN_NUMERIC_PROFILE", "").strip():
        pytest.skip("MOJOLEARN_NUMERIC_PROFILE is set in this environment; these tests read the shipped default")
    from mojolearn import _numeric_profile as g
    g._defaults.clear()
    return g


def test_the_package_exports_the_selector_and_it_is_not_the_mode():
    ml = _package()
    _clean_default(ml)
    assert ml.numeric_profile() == "fp32_v1"
    assert ml.numeric_profile("training") == "fp32_v1"
    assert [r["name"] for r in ml.numeric_profiles() if r["default"]] == ["fp32_v1"]
    for name in ("numeric_profile", "set_numeric_profile", "numeric_profiles", "numeric_profile_measured"):
        assert name in ml.__all__
    # its own parameter: choosing a profile does not touch the numeric mode
    before = ml.numeric_mode()
    try:
        assert ml.set_numeric_profile("fixed15_v1") == "fp32_v1"
        assert ml.numeric_mode() == before
    finally:
        ml.set_numeric_profile("fp32_v1")
    # and a profile name is not a mode, nor a mode a profile
    with pytest.raises(ValueError):
        ml.set_numeric_profile(before)


def test_every_trainer_resolves_fp32_v1_under_the_default_and_refuses_fixed15_by_name(monkeypatch):
    """The four trainers' one line (`require_training`) under the shipped
    inference default: each constructs, and each refuses fixed15_v1 named
    as the training default. That the trainers then TRAIN is the trainer
    test files' job, which the default lane's check runs unchanged."""
    ml = _package()
    g = _clean_default(ml)
    assert ml.numeric_profile() == "fp32_v1"
    from mojolearn import _training_impl as T
    # past the profile line: the next refusal is the empty registry's
    with pytest.raises(ValueError, match="params is empty"):
        T._Optimizer([], "AdamW")
    for where in ("mojolearn.SmallByteLanguageModelTrainer", "mojolearn.LanguageModelHostTrainer",
                  "mojolearn.AdamW", "mojolearn.SambaStack"):
        assert g.require_training(where) == "fp32_v1"
        with pytest.raises(NotImplementedError) as e:
            g.require_training(where, "fixed15_v1")
        assert "fixed15_v1" in str(e.value) and "refused for training" in str(e.value)
    with pytest.raises(NotImplementedError) as e:
        ml.set_numeric_profile("fixed15_v1", use="training")
    assert "fixed15_v1" in str(e.value)
    assert ml.numeric_profile("training") == "fp32_v1"
    # the sabotage arm: a training default that is not fp32_v1 reaches the trainers
    monkeypatch.setitem(g.PROFILES, "fixed15_v1", dict(g.PROFILES["fixed15_v1"], training=True))
    prev = ml.set_numeric_profile("fixed15_v1", use="training")
    try:
        assert g.require_training("x") == "fixed15_v1"
    finally:
        g._defaults["training"] = prev
    monkeypatch.setitem(g.PROFILES, "fixed15_v1", dict(g.PROFILES["fixed15_v1"], training=False))
    # and a trainer refuses by name when the training default is a profile that has not passed
    g._defaults["training"] = "fixed15_v1"
    try:
        with pytest.raises(NotImplementedError) as e:
            T._Optimizer([], "AdamW")
        assert "fixed15_v1" in str(e.value) and "refused for training" in str(e.value)
    finally:
        g._defaults["training"] = "fp32_v1"


def test_a_bare_block_stays_fp32_v1_and_its_backward_refuses_fixed15():
    ml = _package()
    _clean_default(ml)
    try:
        from mojolearn._transformer_impl import TransformerBlock
    except Exception as e:  # noqa: BLE001
        pytest.skip(f"no transformer module here: {e}")
    dm, it = 16, 32

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
        blk = TransformerBlock(w, n_heads=2)
    except (ImportError, OSError, RuntimeError) as e:
        pytest.skip(f"no transformer binding here: {e}")
    assert blk.numeric_profile == "fp32_v1" and blk._int15 is None
    blk15 = TransformerBlock(w, n_heads=2, numeric_profile="fixed15_v1")
    assert blk15.numeric_profile == "fixed15_v1"
    x = f32((1, 3, dm), 11)
    with pytest.raises(NotImplementedError) as e:
        blk15.backward(x, x)
    assert "fixed15_v1" in str(e.value)


def test_the_loader_refuses_a_profile_before_it_reads_the_path(tmp_path, monkeypatch):
    _package()
    from mojolearn import _numeric_profile as g
    from mojolearn.models import CausalLM
    missing = tmp_path / "no-such-model"
    monkeypatch.setitem(g.PROFILES, "fixed15_v1", dict(g.PROFILES["fixed15_v1"], inference=False))
    # a named profile is refused first: the path is never looked at
    with pytest.raises(NotImplementedError) as e:
        CausalLM.load(missing, numeric_profile="fixed15_v1")
    assert "fixed15_v1" in str(e.value)
    with pytest.raises(ValueError):
        CausalLM.load(missing, numeric_profile="nonsense_v9")
    with pytest.raises(ValueError) as e:
        CausalLM.load(missing, numeric_profile="int8_v1")
    assert "not offered" in str(e.value)
    # naming none, or fp32_v1, reaches the path check
    with pytest.raises(FileNotFoundError):
        CausalLM.load(missing)
    with pytest.raises(FileNotFoundError):
        CausalLM.load(missing, numeric_profile="fp32_v1")


# ------------------------------------------- every family under the default

_FAMILIES = (("llama", False), ("llama", True), ("mistral", False), ("qwen2", False),
             ("qwen3", False), ("phi3", False), ("mamba", True), ("mamba2", True))


def _load(root, **kw):
    from mojolearn.models import CausalLM
    try:
        return CausalLM.load(root, **kw)
    except (ImportError, OSError) as e:
        pytest.skip(f"the model's bindings are not built here: {e}")


def _bytes(a):
    from mojolearn._bufcheck import flat_view
    return bytes(flat_view(a, "f").cast("B"))


@pytest.mark.parametrize("device", ["auto", "cpu"])
@pytest.mark.parametrize("arch,tied", _FAMILIES)
def test_every_family_under_the_default(tmp_path, arch, tied, device):
    """Every model family `mojolearn.models` loads, under the shipped
    default, naming nothing: every family computes fp32_v1, reports it,
    and keeps decode == prefill and the batch invariant; a Mamba model
    computes fp32_v1 exactly as when it is named, reports fp32_v1, and
    refuses fixed15_v1 by name."""
    ml = _package()
    _clean_default(ml)
    from mojolearn import _causal_lm_fixtures as fx
    cfg, tensors = fx.family_fixture(arch, tied)
    root = fx._write_checkpoint(str(tmp_path / f"{arch}-{int(tied)}"), cfg, tensors)
    lm = _load(root, device=device)
    ref = _load(root, device=device, numeric_profile="fp32_v1")
    assert ref.numeric_profile == "fp32_v1"
    ids = ml.Array.from_list([[1, 7, 3, 11, 5], [2, 9, 2, 4, 8]], "<i4")
    full = lm.forward(ids)
    assert lm.numeric_profile == "fp32_v1"
    assert _bytes(full) == _bytes(ref.forward(ids))
    if lm.kind == "transformer":
        assert all(b.numeric_profile == "fp32_v1" for b in lm.blocks)
        assert lm._head_int15 is None
        opted = _load(root, device=device, numeric_profile="fixed15_v1")
        assert opted.numeric_profile == "fixed15_v1" and opted._head_int15 is not None
        assert _bytes(opted.forward(ids)) != _bytes(full)
        # the escape hatch in code: the process default back to fp32_v1 gives fp32_v1's bits
        prev = ml.set_numeric_profile("fp32_v1")
        try:
            old = _load(root, device=device)
        finally:
            ml.set_numeric_profile(prev)
        assert old.numeric_profile == "fp32_v1" and _bytes(old.forward(ids)) == _bytes(ref.forward(ids))
    else:
        assert lm.numeric_profile == "fp32_v1"
        assert _bytes(full) == _bytes(ref.forward(ids))
        with pytest.raises(NotImplementedError) as e:
            _load(root, device=device, numeric_profile="fixed15_v1")
        assert "fixed15_v1" in str(e.value)
    # decode == prefill and batch invariance under the default
    st = lm.allocate_state(2, 8)
    pre = lm.forward(ids[:, :3], st)
    steps = [lm.step(ids[:, i:i + 1], st) for i in (3, 4)]
    v = lm.vocab_size
    fb = _bytes(full)
    row = 5 * v * 4
    for b in range(2):
        want = fb[b * row:(b + 1) * row]
        got = _bytes(pre)[b * 3 * v * 4:(b + 1) * 3 * v * 4] + b"".join(
            _bytes(s)[b * v * 4:(b + 1) * v * 4] for s in steps)
        assert got == want, (arch, b)
        alone = _bytes(lm.forward(ids[b:b + 1]))
        assert alone == want, (arch, b)
    g1, g2 = lm.generate(ids[:, :3], 2), lm.generate(ids[:, :3], 2)
    assert g1.tobytes() == g2.tobytes()


def test_the_environment_escape_hatch_in_a_fresh_process(tmp_path):
    """MOJOLEARN_NUMERIC_PROFILE=fp32_v1, set before import: a model that
    names nothing computes fp32_v1's bits exactly."""
    ml = _package()
    _clean_default(ml)
    import subprocess
    import sys
    from mojolearn import _causal_lm_fixtures as fx
    cfg, tensors = fx.family_fixture("llama", False)
    root = fx._write_checkpoint(str(tmp_path / "llama"), cfg, tensors)
    code = ("import hashlib, sys\n"
            "from mojolearn.models import CausalLM\n"
            "from mojolearn._array import Array\n"
            "from mojolearn._bufcheck import flat_view\n"
            "kw = {} if sys.argv[2] == '-' else {'numeric_profile': sys.argv[2]}\n"
            "lm = CausalLM.load(sys.argv[1], **kw)\n"
            "x = lm.forward(Array.from_list([[1, 7, 3, 11, 5]], '<i4'))\n"
            "print(lm.numeric_profile, hashlib.sha256(bytes(flat_view(x, 'f').cast('B'))).hexdigest())\n")
    base = dict(os.environ, MOJOLEARN_NUMERIC_MODE="identical")
    base.pop("MOJOLEARN_NUMERIC_PROFILE", None)

    def run(env_profile, kw):
        env = dict(base)
        if env_profile:
            env["MOJOLEARN_NUMERIC_PROFILE"] = env_profile
        r = subprocess.run([sys.executable, "-c", code, root, kw], env=env, capture_output=True, text=True)
        if r.returncode != 0:
            pytest.skip(f"the model does not run in a child here: {r.stderr[-400:]}")
        return r.stdout.split()[-2:]

    hatch = run("fp32_v1", "-")
    named = run(None, "fp32_v1")
    default = run(None, "-")
    assert hatch[0] == named[0] == "fp32_v1" and hatch[1] == named[1]
    assert default == named
    opted = run("fixed15_v1", "-")
    assert opted == run(None, "fixed15_v1")
    assert opted[0] == "fixed15_v1" and opted[1] != named[1]


def test_a_trainer_state_with_no_field_loads_under_the_new_default():
    """SambaStack, the reader that checks the field: a state written with no
    field (every state before 2026-09-29, and every fp32_v1 state since)
    loads under the shipped default, after a real training step; a state
    carrying fixed15_v1 is refused by name."""
    ml = _package()
    _clean_default(ml)
    assert ml.numeric_profile() == "fp32_v1"
    try:
        cfg = ml.SambaConfig(vocab=256, d_model=32, layers=("mamba3", "attention"), n_heads=2,
                             intermediate=64, tie_embeddings=False)
        m = ml.SambaStack(cfg, generator=ml.training.Generator(1), lr=1e-3, max_norm=1.0)
        ids = ml.Array.from_list([[(i * 37 + r * 11) % 256 for i in range(17)] for r in range(2)], "<i4")
        out = m.train_step(ids[:, :-1], ids[:, 1:])
    except (ImportError, OSError) as e:
        pytest.skip(f"the Samba bindings are not built here: {e}")
    assert out["step"] == 1
    state = m.state_dict()
    assert "numeric_profile" not in state
    twin = ml.SambaStack(cfg, generator=ml.training.Generator(2), lr=1e-3, max_norm=1.0)
    twin.load_state_dict(state)
    assert _bytes(twin.flat) == _bytes(m.flat)
    with pytest.raises(ValueError) as e:
        twin.load_state_dict(dict(state, numeric_profile="fixed15_v1"))
    assert "fixed15_v1" in str(e.value) and "fp32_v1" in str(e.value)
