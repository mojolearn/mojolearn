# SPDX-License-Identifier: Apache-2.0
"""The GEMM profile selector (`mojolearn._gemm_profile`): opt in, the default
does not move, an unavailable profile is refused by name and never replaced
by the default, and a checkpoint carries its profile.
Lane lane/lowbit-flag, 2026-09-29.

The selector tests load the module BY PATH, so they need no binding and no
GPU. The model tests import the package and skip by name when it cannot
build a model on this box.
"""
import importlib.util
import os
import pathlib

import pytest

_HERE = pathlib.Path(__file__).resolve().parent
_SRC = _HERE.parent / "_gemm_profile.py"


def _fresh(monkeypatch, env=None):
    """A private copy of the module, so a test's process default never
    leaks into another test or into the package."""
    if env is None:
        monkeypatch.delenv("MOJOLEARN_GEMM_PROFILE", raising=False)
    else:
        monkeypatch.setenv("MOJOLEARN_GEMM_PROFILE", env)
    spec = importlib.util.spec_from_file_location("_gemm_profile_under_test", _SRC)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_default_is_fp32_v1_and_is_available(monkeypatch):
    g = _fresh(monkeypatch)
    assert g.DEFAULT == "fp32.v1"
    assert g.default_profile() == "fp32.v1"
    assert g.resolve(None) == "fp32.v1"
    assert g.PROFILES["fp32.v1"]["models"] is True
    rows = g.profiles()
    assert [r["name"] for r in rows if r["default"]] == ["fp32.v1"]


def test_short_and_full_names_are_one_profile(monkeypatch):
    g = _fresh(monkeypatch)
    for short in g.PROFILES:
        assert g.canonical(short) == short
        assert g.canonical(g.PREFIX + short) == short
        assert g.canonical("  " + short + " ") == short


@pytest.mark.parametrize("bad", ["fp32", "fp32.v2", "bf16", "fast", "", "FP32.V1"])
def test_an_unregistered_name_is_refused_with_the_registered_names(monkeypatch, bad):
    g = _fresh(monkeypatch)
    with pytest.raises(ValueError) as e:
        g.resolve(bad)
    for short in g.PROFILES:
        assert short in str(e.value)


def test_a_name_that_is_not_a_str_is_a_type_error(monkeypatch):
    g = _fresh(monkeypatch)
    for bad in (1, 1.0, b"fp32.v1", ("fp32.v1",)):
        with pytest.raises(TypeError):
            g.resolve(bad)


def test_an_unavailable_profile_is_refused_by_name_and_never_widened(monkeypatch):
    g = _fresh(monkeypatch)
    unavailable = [k for k, v in g.PROFILES.items() if not v["models"]]
    assert unavailable, "this test must have a profile to refuse; when every profile is available, plant one"
    for short in unavailable:
        for spelled in (short, g.PREFIX + short):
            with pytest.raises(NotImplementedError) as e:
                g.resolve(spelled)
            assert short in str(e.value)
            with pytest.raises(NotImplementedError):
                g.set_default_profile(spelled)
    # a refused set left the default where it was
    assert g.default_profile() == "fp32.v1"


def test_refusal_follows_the_row_not_the_name(monkeypatch):
    """The sabotage arm of the test above: flip a row to available and the
    same call must now pass, so the refusal is read from the registry and
    is not a list of names frozen in the test."""
    g = _fresh(monkeypatch)
    short = next(k for k, v in g.PROFILES.items() if not v["models"])
    g.PROFILES[short] = dict(g.PROFILES[short], models=True)
    assert g.resolve(short) == short
    prev = g.set_default_profile(short)
    assert prev == "fp32.v1" and g.default_profile() == short
    assert g.resolve(None) == short


def test_the_environment_sets_only_the_starting_value(monkeypatch):
    g = _fresh(monkeypatch, env="mojolearn.identical.gemm.fp32.v1")
    assert g.default_profile() == "fp32.v1"
    g = _fresh(monkeypatch, env="nonsense.v9")
    with pytest.raises(ValueError):
        g.default_profile()
    unavailable = next(k for k, v in _fresh(monkeypatch).PROFILES.items() if not v["models"])
    g = _fresh(monkeypatch, env=unavailable)
    with pytest.raises(NotImplementedError):
        g.default_profile()


def test_a_default_checkpoint_gains_no_bytes(monkeypatch):
    g = _fresh(monkeypatch)
    assert g.state_field("fp32.v1") == {}
    assert g.state_field(g.PREFIX + "fp32.v1") == {}
    other = next(k for k in g.PROFILES if k != "fp32.v1")
    assert g.state_field(other) == {"gemm_profile": g.PREFIX + other}


def test_a_state_is_read_only_under_the_profile_that_wrote_it(monkeypatch):
    g = _fresh(monkeypatch)
    other = next(k for k in g.PROFILES if k != "fp32.v1")
    # no field: written under fp32.v1, by every version before this module
    assert g.check_saved({}, "fp32.v1") == "fp32.v1"
    assert g.check_saved({"gemm_profile": g.PREFIX + other}, other) == other
    for state, mine in (({"gemm_profile": g.PREFIX + other}, "fp32.v1"), ({}, other),
                        ({"gemm_profile": "fp32.v1"}, other)):
        with pytest.raises(ValueError) as e:
            g.check_saved(state, mine)
        assert "fp32.v1" in str(e.value) and other in str(e.value)
    with pytest.raises(ValueError):
        g.check_saved({"gemm_profile": "nonsense.v9"}, "fp32.v1")


def test_measured_rows_are_only_what_was_measured(monkeypatch):
    g = _fresh(monkeypatch)
    for profile, rows in g.MEASURED.items():
        assert profile in g.PROFILES
        for vendor, row in rows.items():
            assert vendor in ("cuda", "hip", "metal")
            lo, hi = row["over"]
            assert 0 < lo <= hi
            assert row["box"] and row["what"] and row["source"]
    assert g.measured("fp32.v1") == {}
    assert g.measured("int8i32.v1", "no-such-vendor") is None
    by_name = {r["name"]: r for r in g.profiles()}
    assert by_name["fp32.v1"]["measured"] == {}


def test_a_profile_measured_slower_here_warns_once_and_still_resolves(monkeypatch):
    import warnings
    g = _fresh(monkeypatch)
    g.PROFILES["int8i32.v1"] = dict(g.PROFILES["int8i32.v1"], models=True)
    g.MEASURED["int8i32.v1"] = {
        "cuda": {"over": (2.0, 4.0), "box": "a box", "what": "w", "source": "s"},
        "metal": {"over": (0.5, 0.7), "box": "a Mac", "what": "w", "source": "s"},
        "hip": {"over": (0.9, 1.3), "box": "straddles", "what": "w", "source": "s"},
    }
    monkeypatch.setattr(g, "_this_vendor", lambda: "cuda")
    with warnings.catch_warnings(record=True) as seen:
        warnings.simplefilter("always")
        assert g.resolve("int8i32.v1") == "int8i32.v1"
        assert g.resolve("int8i32.v1") == "int8i32.v1"
    slow = [w for w in seen if issubclass(w.category, g.GemmProfileSpeedWarning)]
    assert len(slow) == 1 and "2 to 4" in str(slow[0].message) and "a box" in str(slow[0].message)
    # quicker here, or a range that reaches below 1, or not measured here: silent
    for vendor in ("metal", "hip", None, "no-such-vendor"):
        monkeypatch.setattr(g, "_this_vendor", lambda v=vendor: v)
        with warnings.catch_warnings(record=True) as seen:
            warnings.simplefilter("always")
            assert g.resolve("int8i32.v1") == "int8i32.v1"
        assert not [w for w in seen if issubclass(w.category, g.GemmProfileSpeedWarning)], vendor
    # the default never warns
    monkeypatch.setattr(g, "_this_vendor", lambda: "cuda")
    with warnings.catch_warnings(record=True) as seen:
        warnings.simplefilter("always")
        assert g.resolve(None) == "fp32.v1"
    assert not seen


# --------------------------------------------------------------- the package


def _package():
    os.environ.setdefault("MOJOLEARN_NUMERIC_MODE", "identical")
    try:
        import mojolearn as ml
    except Exception as e:  # noqa: BLE001
        pytest.skip(f"the package does not import on this box: {e}")
    return ml


def test_the_package_exports_the_selector():
    ml = _package()
    assert ml.gemm_profile() == "fp32.v1"
    assert [r["name"] for r in ml.gemm_profiles() if r["default"]] == ["fp32.v1"]
    assert ml.set_gemm_profile("fp32.v1") == "fp32.v1"
    for name in ("gemm_profile", "set_gemm_profile", "gemm_profiles"):
        assert name in ml.__all__


def test_the_loader_refuses_a_profile_before_it_reads_the_path(tmp_path):
    ml = _package()
    from mojolearn import _gemm_profile as g
    from mojolearn.models import CausalLM
    unavailable = next(k for k, v in g.PROFILES.items() if not v["models"])
    missing = tmp_path / "no-such-model"
    # the profile is refused first: the path is never looked at
    with pytest.raises(NotImplementedError) as e:
        CausalLM.load(missing, gemm_profile=unavailable)
    assert unavailable in str(e.value)
    with pytest.raises(ValueError):
        CausalLM.load(missing, gemm_profile="nonsense.v9")
    # the default reaches the path check, as it did before the keyword existed
    with pytest.raises(FileNotFoundError):
        CausalLM.load(missing)
    with pytest.raises(FileNotFoundError):
        CausalLM.load(missing, gemm_profile="fp32.v1")
