"""Public JL operation controls; run after rebuilding the changed native owner."""
import math

import numpy as np
import pytest
import mojolearn as ml

__mojolearn_user_boundary__ = True


@pytest.mark.parametrize("n,eps", [
    (17, 0.25), (True, 0.5), (0.5, 0.25),
    (np.int64(17), np.float32(0.25)),
    (np.uint64(1000), np.float16(0.5)),
    (np.array(17, dtype=">i8"), np.array(0.25, dtype=">f8")),
    (np.longdouble(17), np.longdouble(0.25)),
])
def test_jl_native_scalar_storage_matches_public_bound(n, eps):
    count, distortion = float(n), float(eps)
    expected = int(4 * math.log(count) / (distortion**2 / 2 - distortion**3 / 3))
    result = ml.johnson_lindenstrauss_min_dim(n, eps=eps)
    assert type(result) is int
    assert result == expected


@pytest.mark.parametrize("n,eps,message", [
    (0, 0.25, "n_samples greater than zero"),
    (-(1 << 2000), 0.25, "n_samples greater than zero"),
    (17, 0.0, "eps in"), (17, 1.0, "eps in"),
    (17, np.nan, "eps in"), (17, np.array(np.inf), "eps in"),
])
def test_jl_native_preserves_bound_errors(n, eps, message):
    with pytest.raises(ValueError, match=message):
        ml.johnson_lindenstrauss_min_dim(n, eps=eps)


@pytest.mark.parametrize("argument", ["n_samples", "eps"])
@pytest.mark.parametrize("kind", ["object", "object_array", "int_subclass", "float_subclass"])
def test_jl_refuses_scalar_protocols_before_hooks(argument, kind):
    calls = []
    class Hook:
        def __lt__(self, other):
            calls.append("lt")
            return True
        def __gt__(self, other):
            calls.append("gt")
            return True
        def __le__(self, other):
            calls.append("le")
            return False
        def __bool__(self):
            calls.append("bool")
            return True
        def __float__(self):
            calls.append("float")
            return 0.25 if argument == "eps" else 17.0
    class IntHook(Hook, int):
        pass
    class FloatHook(Hook, float):
        pass
    if kind == "object_array":
        value = np.empty((), dtype=object)
        value[()] = Hook()
    elif kind == "int_subclass":
        value = IntHook(1)
    elif kind == "float_subclass":
        value = FloatHook(0.25)
    else:
        value = Hook()
    n, eps = (value, 0.25) if argument == "n_samples" else (17, value)
    with pytest.raises(TypeError):
        ml.johnson_lindenstrauss_min_dim(n, eps=eps)
    assert calls == []


def test_jl_preserves_nonfinite_integer_publication_errors():
    with pytest.raises(ValueError):
        ml.johnson_lindenstrauss_min_dim(np.nan, eps=0.25)
    with pytest.raises(OverflowError):
        ml.johnson_lindenstrauss_min_dim(np.inf, eps=0.25)
    with pytest.raises(OverflowError):
        ml.johnson_lindenstrauss_min_dim(1 << 2000, eps=0.25)


def test_jl_checks_extended_precision_before_binary64_conversion():
    # On hosts where longdouble is wider, this input is below one even though
    # its historical binary64 conversion rounds to one. The predicate must
    # run at its original precision. On binary64 hosts it remains below one.
    eps = np.nextafter(np.longdouble(1), np.longdouble(0))
    e = float(eps)
    expected = int(4 * math.log(17) / (e**2 / 2 - e**3 / 3))
    assert ml.johnson_lindenstrauss_min_dim(17, eps=eps) == expected
