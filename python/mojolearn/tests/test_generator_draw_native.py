# SPDX-License-Identifier: Apache-2.0
"""Whole draw audits, shapes, stream counters and the retained public evaluation order.

The independent Philox oracle (uniform words, the dropout mask and scale,
chunk offsets, the last stream) is the native check
checks/native_tests/generator_draw.mojo.

The legacy driver below is test-only. Its source contract is the Generator
implementation at c9b65e2ddda3e8663bbaaf284eddfcdbf7275a5c; it drives the same
accepted RNG kernels independently of the new native preparation owner.
"""
import numbers

import numpy as np
import pytest

from mojolearn import Array
from mojolearn import _training_impl as training
from mojolearn._verify_native_runtime import audit_native_call


def _c32(a, name, where):
    """Frozen copy of the former `_training_impl._c32` (deleted from the
    package 2026-11-15 by lane py-buffer): a float32 input as a float32 Array."""
    from mojolearn.tests._typed_input import as_f32_c
    from mojolearn._bufcheck import is_native_f32, probe
    if not is_native_f32(probe(a).format):
        raise TypeError('mojolearn.%s: %s must be float32' % (where, name))
    return as_f32_c(a, ndim=None, name=name)[0]


class LegacyDraws:
    def __init__(self, seed=7, counter=0, mode=None):
        self.seed, self.counter, self.numeric_mode = seed, counter, mode

    def next(self):
        if self.counter > 0xffffffff:
            raise RuntimeError('the stream counter is exhausted')
        value = self.counter
        self.counter += 1
        return value

    def shape(self, shape):
        shape = (int(shape),) if isinstance(shape, numbers.Integral) else tuple(int(d) for d in shape)
        n = 1
        for d in shape:
            if d < 1:
                raise ValueError('shape must be positive')
            n *= d
        return shape, n

    def uniform(self, shape, low=0., high=1.):
        shape, n = self.shape(shape)
        lo, hi = float(training._round_f32(low)), float(training._round_f32(high))
        span = float(training._round_f32(hi-lo))
        if not span >= 0:
            raise ValueError('high must be >= low')
        out = training._rng_call(training._load(self.numeric_mode), 0, n, 0,
                                 self.seed, self.next(), lo, span)
        return out.reshape(shape)

    def normal(self, shape, mean=0., std=1.):
        shape, n = self.shape(shape)
        sd = float(training._round_f32(std))
        if not sd >= 0:
            raise ValueError('std must be >= 0')
        out = training._rng_call(training._load(self.numeric_mode), 1, n, 0,
                                 self.seed, self.next(), float(training._round_f32(mean)), sd)
        return out.reshape(shape)

    def dropout(self, x, p, offset=0, stream=None):
        p = float(training._round_f32(p))
        if not 0 <= p < 1:
            raise ValueError('p must be in [0, 1)')
        x = _c32(x, 'x', 'Generator.dropout')
        scale = float(training._round_f32(1/(1-p)))
        stream = self.next() if stream is None else int(stream)
        key = dict(seed=self.seed, stream=stream, p=p, scale=scale, offset=int(offset))
        y = training._rng_call(training._load(self.numeric_mode), 2, x.size,
                              key['offset'], self.seed, stream, p, scale, x.reshape(-1))
        return y.reshape(x.shape), key

    def dropout_backward(self, dy, key):
        dy = _c32(dy, 'dy', 'Generator.dropout_backward')
        dx = training._rng_call(training._load(self.numeric_mode), 3, dy.size,
                               key['offset'], key['seed'], key['stream'], key['p'], key['scale'], dy.reshape(-1))
        return dx.reshape(dy.shape)


def checked(function, *args, **kwargs):
    return audit_native_call(function, *args, **kwargs).assert_native()


def bits(actual, expected):
    a, b = np.asarray(actual), np.asarray(expected)
    assert a.shape == b.shape and a.dtype == b.dtype == np.float32
    np.testing.assert_array_equal(a.view('u4'), b.view('u4'))


@pytest.mark.parametrize('shape', [7, np.int64(9), (2, 3, 5), (), [3., 2]])
def test_uniform_shape_seed_and_flat_stream(shape):
    seed = 0xfedcba9876543210
    gen = training.Generator(seed)
    actual = checked(gen.uniform, shape, -2., 6.)
    expected_shape, count = LegacyDraws().shape(shape)
    # Every shape spelling draws the words of the flat stream (the words
    # themselves are the native check's Philox oracle).
    expected = np.asarray(training.Generator(seed).uniform(count, -2., 6.)).reshape(expected_shape)
    # Array's established rank-zero Python buffer exporter is flat. Inspect
    # product shape directly and compare all bytes, without letting NumPy's
    # conversion silently substitute shape (1,) for the requested scalar ().
    assert actual.shape == expected_shape and actual.ndim == len(expected_shape)
    assert actual.dtype == '<f4' and actual.tobytes() == expected.tobytes()
    assert gen.state_dict() == dict(seed=seed, counter=1)
    np.asarray(actual)[...] = 4  # The returned buffer remains writable.


@pytest.mark.parametrize('shape', [(1,), (3, 7), (2, 3, 4)])
@pytest.mark.parametrize('operation,kwargs', [('uniform', dict(low=-.17, high=2.03)),
                                             ('normal', dict(mean=.123456789, std=.432198765))])
def test_draw_matches_retained_driver_and_restores_exact_stream(shape, operation, kwargs):
    gen = training.Generator(0x12345678abcdef01)
    legacy = LegacyDraws(gen.seed)
    for _ in range(2):
        bits(checked(getattr(gen, operation), shape, **kwargs), getattr(legacy, operation)(shape, **kwargs))
    saved = checked(gen.state_dict)
    expected = checked(getattr(gen, operation), shape, **kwargs)
    checked(gen.load_state_dict, saved)
    bits(checked(getattr(gen, operation), shape, **kwargs), expected)


@pytest.mark.parametrize('layout', ['strided', 'fortran', 'readonly', 'array_fortran'])
def test_dropout_native_packing_and_retained_driver(layout):
    x = np.arange(-12, 12, dtype='f4').reshape(4, 6) / np.float32(8)
    if layout == 'strided':
        x = x[:, ::-2]
    elif layout == 'fortran':
        x = np.asfortranarray(x)
    elif layout == 'readonly':
        x.flags.writeable = False
    else:
        x = Array.from_buffer(np.asfortranarray(x))
    seed, stream, offset = 13, 0x80000003, 17
    gen = training.Generator(seed)
    actual, key = checked(gen.dropout, x, .3, offset=offset, stream=stream)
    old, old_key = LegacyDraws(seed).dropout(x, .3, offset=offset, stream=stream)
    bits(actual, old)
    assert key == old_key and gen.counter == 0
    # Every input layout gives the words of its C-contiguous copy.
    a = np.ascontiguousarray(np.asarray(x))
    bits(actual, training.Generator(seed).dropout(a, .3, offset=offset, stream=stream)[0])
    dy = np.full(a.shape, .125, 'f4')
    assert np.asarray(checked(gen.dropout_backward, dy, key)).shape == a.shape
    assert gen.counter == 0


def test_dropout_implicit_stream_and_explicit_offsets_keep_the_counter():
    # The chunk-offset words and the p 0 identity are the native check's.
    x = np.arange(24, dtype='f4').reshape(4, 6) / np.float32(8)
    gen = training.Generator(71)
    whole, key = checked(gen.dropout, x, .25)
    assert gen.counter == 1 and key['stream'] == 0
    for row in range(4):
        piece, piece_key = checked(gen.dropout, x[row:row+1], .25, offset=row*6, stream=key['stream'])
        assert np.asarray(piece).shape == (1, 6) and piece_key['offset'] == row * 6
    assert gen.counter == 1
    checked(gen.dropout, x, 0.)
    assert gen.counter == 2


@pytest.mark.parametrize('operation,args,kwargs,mode,advance', [
    ('uniform', ((0, 2),), {}, None, 0),
    ('uniform', ((2,),), dict(low=2, high=1), None, 0),
    ('uniform', ((2,),), {}, 'unknown', 0),
    ('normal', ((2,),), dict(std=-1), None, 0),
    ('normal', ((2,),), dict(mean='bad-number'), None, 1),
    ('normal', ((2,),), dict(mean=float('inf')), None, 1),
    ('dropout', (np.ones(3, 'f4'), 1.), {}, None, 0),
    ('dropout', (np.ones(3, 'f8'), .2), {}, None, 0),
    ('dropout', (np.ones(3, 'f4'), .2), dict(offset=-1), None, 1),
    ('dropout', (np.ones(3, 'f4'), .2), dict(offset='bad-number'), None, 1),
    ('dropout', (np.ones(3, 'f4'), .2), {}, 'unknown', 1),
    ('dropout', (np.ones(0, 'f4'), .2), {}, None, 0),
    ('dropout', (np.ones(3, 'f4'), .2), dict(stream=-1), None, 0),
    ('dropout', (np.array([float('nan')], 'f4'), .2), {}, None, 1),
])
def test_failed_calls_keep_the_established_stream_consumption(operation, args, kwargs, mode, advance):
    gen = training.Generator(7, numeric_mode=mode)
    old = LegacyDraws(7, mode=mode)
    with pytest.raises(Exception) as original:
        getattr(old, operation)(*args, **kwargs)
    audit = audit_native_call(getattr(gen, operation), *args, **kwargs)
    assert audit.exception is not None, audit.report()
    assert type(audit.exception) is type(original.value), audit.report()
    for key in ('python_frames', 'forbidden_native_calls', 'wrapper_work', 'unobserved_existing_threads'):
        assert not audit.report()[key], audit.report()
    assert gen.counter == old.counter == advance


def test_last_stream_draw_is_portable_then_exhaustion_is_unchanged():
    gen = training.Generator(19)
    checked(gen.load_state_dict, dict(seed=19, counter=0xffffffff))
    assert checked(gen.uniform, (11,)).shape == (11,)  # its words: the native check
    terminal = checked(gen.state_dict)
    assert terminal['counter'] == 1 << 32
    restored = training.Generator(0)
    checked(restored.load_state_dict, terminal)
    for function, args in ((restored.uniform, (3,)), (restored.normal, (3,)),
                           (restored.dropout, (np.ones(3, 'f4'), .2))):
        audit = audit_native_call(function, *args)
        assert isinstance(audit.exception, RuntimeError), audit.report()
        assert restored.state_dict() == terminal
        for key in ('python_frames', 'forbidden_native_calls', 'wrapper_work', 'unobserved_existing_threads'):
            assert not audit.report()[key], audit.report()
    # An explicit stream still replays after the allocator has been exhausted.
    result, key = checked(restored.dropout, np.ones(3, 'f4'), .2, stream=5)
    assert key['stream'] == 5 and np.asarray(result).shape == (3,)
    assert restored.state_dict() == terminal
