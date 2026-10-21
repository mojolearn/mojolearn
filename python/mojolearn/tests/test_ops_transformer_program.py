# SPDX-License-Identifier: Apache-2.0
"""Independent primitive derivatives and native whole-block capture checks."""
import numpy as np
import pytest

from mojolearn import ops
from mojolearn._verify_native_runtime import audit_native_call
from mojolearn import _verify_graph as verify


def native(call, *args, **kwargs):
    return audit_native_call(call, *args, **kwargs).assert_native()


def test_rms_norm_noncontiguous_view_and_both_derivatives():
    x = np.arange(24, dtype=np.float32).reshape(2, 4, 3) / 11 - 1
    w = np.array([.7, -1.1, .4, 1.3], np.float32)
    seed = np.linspace(-.4, .7, 24, dtype=np.float32).reshape(2, 3, 4)
    eps = .03
    with ops.Session(record=True) as s:
        hx, hw = s.tensor(x, requires_grad=True), s.tensor(w, requires_grad=True)
        view = s.permute(hx, (0, 2, 1))
        result = native(s.rms_norm, view, hw, eps)
        native(s.backward, s.sum(s.mul(result, s.tensor(seed))))
        z = x.transpose(0, 2, 1).astype(np.float64)
        inv = 1 / np.sqrt(np.mean(z*z, axis=-1, keepdims=True) + eps)
        expected = z * inv * w
        weighted = seed.astype(np.float64) * w
        dx = weighted * inv - z * inv**3 * np.mean(weighted*z, axis=-1, keepdims=True)
        dw = np.sum(seed.astype(np.float64) * z * inv, axis=(0, 1))
        np.testing.assert_allclose(native(s.numpy, result), expected, rtol=3e-6, atol=1e-7)
        np.testing.assert_allclose(native(s.grad, hx), dx.transpose(0, 2, 1), rtol=4e-6, atol=1e-7)
        np.testing.assert_allclose(native(s.grad, hw), dw, rtol=4e-6, atol=1e-7)


def test_rotary_absolute_offset_gradient_and_multiple_heads():
    x = np.linspace(-1, 1, 48, dtype=np.float32).reshape(2, 3, 8)
    seed = np.cos(np.arange(48)).astype(np.float32).reshape(x.shape)
    offset, hd, theta = 7, 4, 4000.
    angle = np.arange(offset, offset+3)[:, None] / theta ** (np.arange(hd//2)*2./hd)
    c = np.tile(np.concatenate([np.cos(angle)]*2, axis=-1), (1, 2))[None]
    si = np.tile(np.concatenate([np.sin(angle)]*2, axis=-1), (1, 2))[None]
    def rotate(values):
        z = values.reshape(2, 3, 2, hd)
        return np.concatenate([-z[..., 2:], z[..., :2]], axis=-1).reshape(x.shape)
    expected = x*c + rotate(x)*si
    expected_grad = seed*c - rotate(seed)*si
    with ops.Session(record=True) as s:
        h = s.tensor(x, requires_grad=True)
        y = native(s.rotary, h, hd, offset=offset, theta=theta)
        native(s.backward, s.sum(s.mul(y, s.tensor(seed))))
        np.testing.assert_allclose(native(s.numpy, y), expected, rtol=2e-6, atol=2e-6)
        np.testing.assert_allclose(native(s.grad, h), expected_grad, rtol=2e-6, atol=2e-6)


def test_projection_first_vjp_keeps_full_rank_and_parameter_shape():
    x = np.arange(24, dtype=np.float32).reshape(2, 3, 4) / 16
    w = np.arange(20, dtype=np.float32).reshape(5, 4) / 32 - .2
    seed = np.linspace(-.3, .7, 30, dtype=np.float32).reshape(2, 3, 5)
    with ops.Session(record=True) as s:
        hx, hw = s.tensor(x, requires_grad=True), s.tensor(w, requires_grad=True)
        y = native(s._transformer_linear, hx, hw)
        dx, dw = native(s.vjp, y, [hx, hw], seed)
        assert dx.shape == x.shape and dw.shape == w.shape
        np.testing.assert_allclose(dx, seed.astype(np.float64) @ w, rtol=2e-6, atol=2e-7)
        np.testing.assert_allclose(dw, seed.reshape(-1, 5).astype(np.float64).T @ x.reshape(-1, 4), rtol=2e-6, atol=2e-7)
        with pytest.raises(ValueError, match="higher|backward rule|unsupported"):
            s.vjp(y, [hx, hw], seed, create_graph=True)


@pytest.mark.parametrize("bad", ["rms_weight", "rms_epsilon", "rms_nan", "rotary_odd", "rotary_width", "rotary_domain", "rotary_theta", "integer"])
def test_native_primitive_refusals_and_recovery(bad):
    with ops.Session() as s:
        x = s.tensor(np.ones((1, 3, 8), np.float32))
        w = s.tensor(np.ones(8, np.float32))
        function, args, kwargs = s.rms_norm, (x, w), {}
        if bad == "rms_weight":
            args = (x, s.tensor(np.ones(7, np.float32)))
        elif bad in ("rms_epsilon", "rms_nan"):
            kwargs = {"eps": 0 if bad == "rms_epsilon" else float("nan")}
        elif bad == "integer":
            args = (s.ids(np.ones((1, 3, 8), np.int64)), w)
        else:
            function, args = s.rotary, (x, 3 if bad == "rotary_odd" else (6 if bad == "rotary_width" else 4))
            if bad == "rotary_domain":
                kwargs = {"offset": 8191}
            elif bad == "rotary_theta":
                kwargs = {"theta": -1.}
        audit = audit_native_call(function, *args, **kwargs)
        report = audit.report()
        assert isinstance(audit.exception, ValueError), report
        assert not any(report[k] for k in ("python_frames", "wrapper_work", "forbidden_native_calls", "unobserved_existing_threads")), report
        native(s.numpy, native(s.rms_norm, x, w))


@pytest.mark.parametrize("where", ["epsilon", "theta", "head_dim", "handle"])
def test_primitive_scalar_hooks_refused_before_dispatch(where):
    seen = []
    class FloatHook(float):
        def __float__(self):
            seen.append("float")
            raise AssertionError("Python scalar callback must not execute")
    class IntHook(int):
        def __int__(self):
            seen.append("int")
            raise AssertionError("Python scalar callback must not execute")
        def __index__(self):
            seen.append("index")
            raise AssertionError("Python scalar callback must not execute")
    with ops.Session() as s:
        x = s.tensor(np.ones((1, 3, 8), np.float32))
        w = s.tensor(np.ones(8, np.float32))
        call, args, kwargs = s.rms_norm, (x, w), {"eps": FloatHook(.01)}
        if where == "theta":
            call, args, kwargs = s.rotary, (x, 4), {"theta": FloatHook(10000.)}
        elif where == "head_dim":
            call, args, kwargs = s.rotary, (x, IntHook(4)), {}
        elif where == "handle":
            call, args, kwargs = s._transformer_linear, (IntHook(x), w), {}
        audit = audit_native_call(call, *args, **kwargs)
        report = audit.report()
        assert isinstance(audit.exception, TypeError), report
        assert not seen
        assert not any(report[k] for k in ("python_frames", "wrapper_work", "forbidden_native_calls", "unobserved_existing_threads")), report


@pytest.mark.parametrize("order", ["permute_swap", "swap_permute"])
def test_mixed_permutation_composition_rank_and_alias_guards(order):
    a = np.arange(48, dtype=np.float32).reshape(1, 3, 2, 8)
    with ops.Session() as s:
        x = s.tensor(a)
        if order == "permute_swap":
            intermediate = s.permute(x, (0, 2, 1, 3))
            result = s.transpose(intermediate, -1, -2)
        else:
            intermediate = s.transpose(x, -1, -2)
            result = s.permute(intermediate, (0, 2, 1, 3))
        p = s.capture(x, [intermediate, result])
        before = native(p.run, a)
        after = native(p.run, a, optimize=True)
        for actual, expected in zip(after, before):
            assert np.array_equal(actual.view(np.uint32), expected.view(np.uint32))
        counts = {r["name"]: r["rewrites"] for r in native(p.inspect, optimize=True)["pass_report"]["passes"]}
        assert counts["permutation_composition"] == 1
        assert counts["dead_intermediates"] == 0, "returned intermediate remains live"
        for optimize in (False, True):
            with pytest.raises(ValueError, match="rank|axes"):
                p.run_dynamic(a.reshape(3, 2, 8), optimize=optimize)
        native(p.run, a, optimize=True)


def test_complete_transformer_composition_and_checkpoint():
    outputs = verify.transformer_program_composition()
    assert len(outputs) == verify.OUTPUT_COUNTS["transformer_program_composition"]
