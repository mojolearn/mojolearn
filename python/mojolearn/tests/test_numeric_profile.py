# SPDX-License-Identifier: Apache-2.0
"""The `numeric_profile=` selector (`mojolearn._numeric_profile`): opt in, the
default does not move, an unavailable profile is refused by name and never
replaced by the default, an arithmetic that failed quality is not offered,
and a checkpoint carries its profile.
Lane lane/lowbit-flag, 2026-09-29.

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


def _unavailable(g):
    names = [k for k, v in g.PROFILES.items() if not v["inference"]]
    assert names, "this test must have a profile to refuse; when every profile is available, plant one"
    return names


def test_default_is_fp32_v1_and_is_available(monkeypatch):
    g = _fresh(monkeypatch)
    assert g.DEFAULT == "fp32_v1"
    assert g.default_profile() == "fp32_v1"
    assert g.resolve(None) == "fp32_v1"
    assert g.PROFILES["fp32_v1"]["inference"] is True and g.PROFILES["fp32_v1"]["training"] is True
    assert [r["name"] for r in g.profiles() if r["default"]] == ["fp32_v1"]
    assert list(g.PROFILES)[0] == "fp32_v1"


def test_the_names_andrew_chose_are_registered(monkeypatch):
    g = _fresh(monkeypatch)
    assert list(g.PROFILES) == ["fp32_v1", "fixed15_v1"]
    assert g.PROFILES["fixed15_v1"]["status"] == "experimental"
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
    assert len(set(g.PROFILES["fp32_v1"]["products"].values())) == 1
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
    assert len(g.PROFILES["fixed15_v1"]["quality"]) == 2


def test_a_name_that_is_not_a_str_is_a_type_error(monkeypatch):
    g = _fresh(monkeypatch)
    for bad in (1, 1.0, b"fp32_v1", ("fp32_v1",)):
        with pytest.raises(TypeError):
            g.resolve(bad)


def test_an_unavailable_profile_is_refused_by_name_and_never_widened(monkeypatch):
    g = _fresh(monkeypatch)
    for name in _unavailable(g):
        with pytest.raises(NotImplementedError) as e:
            g.resolve(name)
        assert name in str(e.value)
        with pytest.raises(NotImplementedError):
            g.set_default_profile(name)
    # a refused set left the default where it was
    assert g.default_profile() == "fp32_v1"


def test_refusal_follows_the_row_not_the_name(monkeypatch):
    """The sabotage arm of the test above: flip a row to available and the
    same call must now pass, so the refusal is read from the registry and
    is not a list of names frozen in the test."""
    g = _fresh(monkeypatch)
    name = _unavailable(g)[0]
    g.PROFILES[name] = dict(g.PROFILES[name], inference=True, training=True)
    assert g.resolve(name) == name
    prev = g.set_default_profile(name)
    assert prev == "fp32_v1" and g.default_profile() == name
    assert g.resolve(None) == name


def test_training_is_its_own_gate(monkeypatch):
    g = _fresh(monkeypatch)
    name = _unavailable(g)[0]
    # passed for inference, not for training: inference accepts, every trainer refuses
    g.PROFILES[name] = dict(g.PROFILES[name], inference=True, training=False)
    assert g.resolve(name) == name
    assert g.resolve(name, use="inference") == name
    with pytest.raises(NotImplementedError) as e:
        g.resolve(name, use="training")
    assert name in str(e.value) and "refused for training" in str(e.value)
    assert "available for inference" in str(e.value)
    # a process default that inference accepts still cannot train
    assert g.set_default_profile(name) == "fp32_v1"
    assert g.resolve(None) == name
    with pytest.raises(NotImplementedError):
        g.require_training("a trainer")
    # the default profile trains
    g.set_default_profile("fp32_v1")
    assert g.require_training("a trainer") == "fp32_v1"
    # the sabotage arm: open the training gate and the same call passes
    g.PROFILES[name] = dict(g.PROFILES[name], training=True)
    assert g.resolve(name, use="training") == name
    with pytest.raises(ValueError):
        g.resolve(name, use="fine-tuning")
    for row in g.PROFILES.values():
        assert isinstance(row["inference"], bool) and isinstance(row["training"], bool)
        assert row["inference"] or not row["training"], "nothing trains that cannot infer"


def test_the_environment_sets_only_the_starting_value(monkeypatch):
    g = _fresh(monkeypatch, env="fp32_v1")
    assert g.default_profile() == "fp32_v1"
    g = _fresh(monkeypatch, env="nonsense_v9")
    with pytest.raises(ValueError):
        g.default_profile()
    name = _unavailable(_fresh(monkeypatch))[0]
    g = _fresh(monkeypatch, env=name)
    with pytest.raises(NotImplementedError):
        g.default_profile()


def test_a_default_checkpoint_gains_no_bytes(monkeypatch):
    g = _fresh(monkeypatch)
    assert g.state_field("fp32_v1") == {}
    other = _unavailable(g)[0]
    assert g.state_field(other) == {g.STATE_KEY: other}
    assert g.STATE_KEY == "numeric_profile"


def test_a_state_is_read_only_under_the_profile_that_wrote_it(monkeypatch):
    g = _fresh(monkeypatch)
    other = _unavailable(g)[0]
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
    by_name = {r["name"]: r for r in g.profiles()}
    assert by_name["fp32_v1"]["measured"] == {}


def test_a_profile_measured_slower_here_warns_once_and_still_resolves(monkeypatch):
    g = _fresh(monkeypatch)
    name = _unavailable(g)[0]
    g.PROFILES[name] = dict(g.PROFILES[name], inference=True, training=True)
    g.MEASURED[name] = {
        "cuda": {"over": (2.0, 4.0), "box": "a box", "what": "w", "source": "s"},
        "metal": {"over": (0.5, 0.7), "box": "a Mac", "what": "w", "source": "s"},
        "hip": {"over": (0.9, 1.3), "box": "straddles", "what": "w", "source": "s"},
    }
    monkeypatch.setattr(g, "_this_vendor", lambda: "cuda")
    with warnings.catch_warnings(record=True) as seen:
        warnings.simplefilter("always")
        assert g.resolve(name) == name
        assert g.resolve(name) == name
    slow = [w for w in seen if issubclass(w.category, g.NumericProfileSpeedWarning)]
    assert len(slow) == 1 and "2 to 4" in str(slow[0].message) and "a box" in str(slow[0].message)
    # quicker here, or a range that reaches below 1, or not measured here: silent
    for vendor in ("metal", "hip", None, "no-such-vendor"):
        monkeypatch.setattr(g, "_this_vendor", lambda v=vendor: v)
        with warnings.catch_warnings(record=True) as seen:
            warnings.simplefilter("always")
            assert g.resolve(name) == name
        assert not [w for w in seen if issubclass(w.category, g.NumericProfileSpeedWarning)], vendor
    # the default never warns
    monkeypatch.setattr(g, "_this_vendor", lambda: "cuda")
    with warnings.catch_warnings(record=True) as seen:
        warnings.simplefilter("always")
        assert g.resolve(None) == "fp32_v1"
    assert not [w for w in seen if issubclass(w.category, g.NumericProfileSpeedWarning)]


# --------------------------------------------------------------- the package


def _package():
    os.environ.setdefault("MOJOLEARN_NUMERIC_MODE", "identical")
    try:
        import mojolearn as ml
    except Exception as e:  # noqa: BLE001
        pytest.skip(f"the package does not import on this box: {e}")
    return ml


def test_the_package_exports_the_selector_and_it_is_not_the_mode():
    ml = _package()
    assert ml.numeric_profile() == "fp32_v1"
    assert [r["name"] for r in ml.numeric_profiles() if r["default"]] == ["fp32_v1"]
    for name in ("numeric_profile", "set_numeric_profile", "numeric_profiles", "numeric_profile_measured"):
        assert name in ml.__all__
    # its own parameter: choosing a profile does not touch the numeric mode
    before = ml.numeric_mode()
    assert ml.set_numeric_profile("fp32_v1") == "fp32_v1"
    assert ml.numeric_mode() == before
    # and a profile name is not a mode, nor a mode a profile
    with pytest.raises(ValueError):
        ml.set_numeric_profile(before)


def test_a_trainer_refuses_a_profile_that_has_not_passed_for_training(monkeypatch):
    ml = _package()
    from mojolearn import _numeric_profile as g
    from mojolearn import _training_impl as T
    name = _unavailable(g)[0]
    monkeypatch.setitem(g.PROFILES, name, dict(g.PROFILES[name], inference=True, training=False))
    prev = ml.set_numeric_profile(name)
    try:
        with pytest.raises(NotImplementedError) as e:
            T._Optimizer([], "AdamW")
        assert name in str(e.value) and "refused for training" in str(e.value)
    finally:
        ml.set_numeric_profile(prev)
    assert ml.numeric_profile() == "fp32_v1"


def test_the_loader_refuses_a_profile_before_it_reads_the_path(tmp_path):
    _package()
    from mojolearn import _numeric_profile as g
    from mojolearn.models import CausalLM
    name = _unavailable(g)[0]
    missing = tmp_path / "no-such-model"
    # the profile is refused first: the path is never looked at
    with pytest.raises(NotImplementedError) as e:
        CausalLM.load(missing, numeric_profile=name)
    assert name in str(e.value)
    with pytest.raises(ValueError):
        CausalLM.load(missing, numeric_profile="nonsense_v9")
    with pytest.raises(ValueError) as e:
        CausalLM.load(missing, numeric_profile="int8_v1")
    assert "not offered" in str(e.value)
    # the default reaches the path check, as it did before the keyword existed
    with pytest.raises(FileNotFoundError):
        CausalLM.load(missing)
    with pytest.raises(FileNotFoundError):
        CausalLM.load(missing, numeric_profile="fp32_v1")
