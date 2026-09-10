"""Small sklearn protocol/parameter smoke; no GPU calls or mock kernels."""
import numpy as np
from sklearn.base import clone
from sklearn.utils import get_tags
from sklearn.utils.validation import check_is_fitted
from sklearn.exceptions import NotFittedError

from mojolearn import MinMaxScaler, preprocessing


def main():
    assert preprocessing.MinMaxScaler is MinMaxScaler
    bounds = [-2, 3]
    scaler = MinMaxScaler(feature_range=bounds, numeric_mode='identical')
    assert scaler.get_params()['feature_range'] is bounds
    copied = clone(scaler)
    assert copied.get_params() == scaler.get_params()
    assert not copied.__sklearn_is_fitted__()
    tags = get_tags(copied)
    assert tags.transformer_tags.preserves_dtype == ['float32']
    assert not tags.target_tags.required
    try:
        check_is_fitted(copied)
    except NotFittedError:
        pass
    else:
        raise AssertionError('unfitted scaler accepted')
    for options in (
        {'copy': False}, {'clip': 1}, {'numeric_mode': 'typo'},
        {'feature_range': [1, 1]}, {'feature_range': [2, 1]},
        {'feature_range': [0, np.inf]}, {'feature_range': [0, True]},
        {'feature_range': [0, np.nextafter(float(np.finfo(np.float32).max), np.inf)]},
        {'feature_range': (x for x in [0, 1])}, {'feature_range': np.array([[0, 1]])},
        {'unknown': 1},
    ):
        before = scaler.get_params()
        try:
            scaler.set_params(**options)
        except (ValueError, NotImplementedError):
            pass
        else:
            raise AssertionError(f'unsupported parameters accepted: {options}')
        assert scaler.get_params() == before
    assert scaler.set_params(clip=True, numeric_mode='fast') is scaler
    assert scaler.clip and scaler.numeric_mode == 'fast'
    print('PASS MinMaxScaler sklearn clone/tags/parameter smoke')


if __name__ == '__main__':
    main()
