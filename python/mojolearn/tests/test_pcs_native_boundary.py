# SPDX-License-Identifier: Apache-2.0
"""PCS Python admission and mutable public storage; numerical fixtures are native."""
import numpy as np
import pytest
import mojolearn as ml


def _data():
    return np.array([[1, 2, 1], [3, 1, 2], [1, 1, 4], [2, 3, 1]], dtype=np.float32)


def _model(**options):
    settings = dict(degree=3, n_components=16, gamma=0.5, coef0=1.0, random_state=7, numeric_mode="identical")
    settings.update(options)
    return ml.PolynomialCountSketch(**settings)


@pytest.mark.parametrize("old_items", [False, True])
def test_pcs_public_paths_do_not_call_legacy_python_entries(old_items, monkeypatch):
    from mojolearn import _backend

    monkeypatch.setenv("MOJOLEARN_XN_OLD_ITEMS", "1" if old_items else "0")
    calls = []

    def forbidden(*args, **kwargs):
        calls.append(True)
        raise AssertionError("legacy PCS argument-list dispatch called")

    binding = _backend.binding("_mojolearn_x_neighbors", "identical")
    for name in ("xn_pcs", "xn_pcs_resident", "x_neighbors_kfeat_pcs_draw_idn"):
        monkeypatch.setattr(binding, name, forbidden, raising=False)
    model = _model()
    assert np.asarray(model.fit_transform(_data())).shape == (4, 16)
    assert np.asarray(model.transform(_data()[:2])).shape == (2, 16)
    assert model.fit(_data()) is model
    assert calls == []


@pytest.mark.parametrize("seed", [0, 7, -1, (1 << 160) + 7, -((1 << 170) + 5)])
def test_pcs_arbitrary_precision_seed_uses_original_low_word(seed):
    model = _model(random_state=seed).fit(_data())
    reference = _model(random_state=seed & 0xFFFFFFFF).fit(_data())
    np.testing.assert_array_equal(np.asarray(model.indexHash_), np.asarray(reference.indexHash_))
    np.testing.assert_array_equal(np.asarray(model.bitHash_), np.asarray(reference.bitHash_))


def test_pcs_randomstate_consumes_exactly_one_seed_word():
    state = np.random.RandomState(4)
    reference = np.random.RandomState(4)
    word = int(reference.randint(0, 2**31, size=1)[0])
    model = _model(random_state=state).fit(_data())
    direct = _model(random_state=word).fit(_data())
    np.testing.assert_array_equal(np.asarray(model.indexHash_), np.asarray(direct.indexHash_))
    np.testing.assert_array_equal(np.asarray(model.bitHash_), np.asarray(direct.bitHash_))
    np.testing.assert_array_equal(state.randint(0, 2**31, size=5), reference.randint(0, 2**31, size=5))


@pytest.mark.parametrize("old_items", [False, True])
def test_pcs_rebound_readonly_and_mutated_public_hashes_remain_live(old_items, monkeypatch):
    monkeypatch.setenv("MOJOLEARN_XN_OLD_ITEMS", "1" if old_items else "0")
    model = _model().fit(_data())
    before = np.asarray(model.transform(_data())).copy()
    indices = np.asarray(model.indexHash_).copy()
    signs = np.asarray(model.bitHash_).copy()
    indices.flags.writeable = False
    signs.flags.writeable = False
    model.indexHash_, model.bitHash_ = indices, signs
    np.testing.assert_array_equal(np.asarray(model.transform(_data())), before)
    signs.flags.writeable = True
    signs *= -1
    # The odd-degree map reverses sign when every public sign table does.
    np.testing.assert_allclose(np.asarray(model.transform(_data())), -before, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("operation", ["fit", "transform"])
def test_pcs_replaced_descriptor_refuses_python_factory(operation, monkeypatch):
    from mojolearn import _backend

    model = _model().fit(_data())
    calls = []

    def forbidden():
        calls.append(True)
        raise AssertionError("Python descriptor factory called")

    binding = _backend.binding("_mojolearn_x_neighbors", "identical")
    monkeypatch.setattr(binding, "x_neighbors_native_api_v1", forbidden)
    with pytest.raises(TypeError, match="compiled extension entry"):
        getattr(model, operation)(_data())
    assert calls == []


def test_pcs_integer_seed_subclass_is_refused_before_conversion():
    calls = []

    class Seed(int):
        def __int__(self):
            calls.append("int")
            return 7

        def __index__(self):
            calls.append("index")
            return 7

        def __and__(self, other):
            calls.append("and")
            return 7

    with pytest.raises(TypeError, match="exact int|callbacks"):
        _model(random_state=Seed(7)).fit(_data())
    assert calls == []


def test_pcs_fit_ignores_gamma_conversion_until_query():
    model = _model(gamma="invalid")
    assert model.fit(_data()) is model
    with pytest.raises(TypeError, match="gamma.*native real"):
        model.transform(_data())


def test_pcs_augmentation_precedes_float32_rounding():
    model = _model(coef0=1e-300).fit(_data())
    assert np.asarray(model.indexHash_).shape == (3, 4)
    assert np.asarray(model.bitHash_).shape == (3, 4)
    assert np.asarray(model.transform(_data())).shape == (4, 16)


def test_pcs_invalid_size_precedes_coef_conversion_and_publication():
    model = _model(degree=0, coef0="invalid")
    with pytest.raises(ValueError, match="degree and n_components must be >= 1"):
        model.fit(_data())
    assert not hasattr(model, "indexHash_")
    assert not hasattr(model, "n_features_in_")
