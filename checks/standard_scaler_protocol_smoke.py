"""Focused optional sklearn StandardScaler protocol smoke; no GPU calls."""
from sklearn.base import clone
from sklearn.utils import get_tags
from mojolearn import StandardScaler, preprocessing


def main():
    assert preprocessing.StandardScaler is StandardScaler
    scaler = StandardScaler(with_mean=False, numeric_mode='identical')
    assert clone(scaler).get_params() == scaler.get_params()
    assert not clone(scaler).__sklearn_is_fitted__()
    assert get_tags(scaler).transformer_tags.preserves_dtype == ['float32']
    for options in ({'copy': False}, {'with_mean': 1}, {'with_std': 'yes'},
                    {'numeric_mode': 'wrong'}, {'unknown': 1}):
        before = scaler.get_params()
        try:
            scaler.set_params(**options)
        except (ValueError, NotImplementedError):
            pass
        else:
            raise AssertionError(f'unsupported parameters accepted: {options}')
        assert scaler.get_params() == before
    assert scaler.set_params(with_std=False) is scaler
    print('PASS StandardScaler sklearn clone/tags/parameter smoke')


if __name__ == '__main__':
    main()
