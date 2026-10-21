# SPDX-License-Identifier: Apache-2.0
# Whole sequence-optimizer updates, not a pointwise-kernel-only witness.
# The native opt_step owner advances its scalar state and launches the update;
# each case retains device parameter/moment buffers across six updates. Every
# returned parameter, moment, untouched gradient and scalar word is consumed.
# Public CPython adapters/checkpoints are deliberately outside this scope.
from std.memory import bitcast
from std.sys.compile import is_defined
from checks.numerics import ftz, identical_div, identical_mul, identical_pow64
from sequence.exec_trait import Exec
from sequence.ops import FP, OPT_ADAGRAD, OPT_ADAMAX, OPT_NADAM, OPT_RMSPROP, OPT_LION
from sequence.recurrent import OptConfig, OptState, opt_step
from sequence.checks.oracle import o_adagrad, o_adamax, o_nadam, o_rmsprop, o_lion

comptime PARAMS = 37
comptime STEPS = 6


def _ptr(mut values: List[Float32]) -> FP:
    return FP(unsafe_from_address=Int(values.unsafe_ptr()))


def _equal(got: Float32, expected: Float32, name: String, step: Int, index: Int) raises:
    if bitcast[DType.uint32](got) != bitcast[DType.uint32](expected):
        print("FAIL sequence_optimizer", name, "step", step, "word", index,
              "got", bitcast[DType.uint32](got), "expected", bitcast[DType.uint32](expected))
        raise Error("sequence optimizer differs from independent exact oracle")


def _oracle_scalars(cfg: OptConfig, mut words: List[Float32], step: Int, lr: Float32) -> List[Float32]:
    # Independent statement of the declared bias/momentum equations. Never
    # call product opt_scalars: a scalar-state bug must not move this oracle.
    var result: List[Float32] = [Float32(0.0), Float32(0.0), cfg.f7]
    if cfg.kind == OPT_ADAMAX or cfg.kind == OPT_NADAM:
        words[0] = ftz(identical_mul(words[0], cfg.f1))
        words[1] = ftz(identical_mul(words[1], cfg.f2))
    if cfg.kind == OPT_ADAMAX:
        result[0] = ftz(identical_div(lr, Float32(1.0) - words[0]))
    elif cfg.kind == OPT_ADAGRAD:
        result[0] = ftz(identical_div(lr, Float32(1.0) + ftz(identical_mul(Float32(step - 1), cfg.f1))))
    elif cfg.kind == OPT_NADAM:
        var beta = Float64(cfg.f1)
        var decay = Float64(cfg.f7)
        var mu = beta * (1.0 - 0.5 * identical_pow64(0.96, Float64(step) * decay))
        var next_mu = beta * (1.0 - 0.5 * identical_pow64(0.96, Float64(step + 1) * decay))
        words[2] = ftz(identical_mul(words[2], Float32(mu)))
        var product = Float64(words[2])
        result[0] = Float32(1.0) - words[1]
        result[1] = Float32(-Float64(lr) * (1.0 - mu) / (1.0 - product))
        result[2] = Float32(-Float64(lr) * next_mu / (1.0 - product * next_mu))
    return result^


def _run[E: Exec](mut ex: E, name: String, cfg: OptConfig) raises:
    var p = List[Float32](length=PARAMS, fill=0.0)
    var g = List[Float32](length=PARAMS, fill=0.0)
    var s1 = List[Float32](length=PARAMS, fill=0.0)
    var s2 = List[Float32](length=PARAMS, fill=0.0)
    var s3 = List[Float32](length=PARAMS, fill=0.0)
    for i in range(PARAMS):
        p[i] = Float32((i * 13) % 31 - 15) / Float32(17)
        # A nonzero initial accumulator is a supported Adagrad setting.
        if cfg.kind == OPT_ADAGRAD:
            s2[i] = Float32(0.125)
    var want_p = p.copy()
    var want_s1 = s1.copy()
    var want_s2 = s2.copy()
    var want_s3 = s3.copy()
    var want_scalars = List[Float32](length=3, fill=Float32(1.0))
    var scalar_state = OptState()
    var P = ex.alloc(PARAMS)
    var G = ex.alloc(PARAMS)
    var S1 = ex.alloc(PARAMS)
    var S2 = ex.alloc(PARAMS)
    var S3 = ex.alloc(PARAMS)
    ex.upload(P, _ptr(p), PARAMS)
    ex.upload(S1, _ptr(s1), PARAMS)
    ex.upload(S2, _ptr(s2), PARAMS)
    ex.upload(S3, _ptr(s3), PARAMS)
    var gradient_copy = List[Float32](length=PARAMS, fill=0.0)
    var digest = UInt64(1469598103934665603)
    for step in range(1, STEPS + 1):
        for i in range(PARAMS):
            g[i] = Float32((i * 7 + step * 11) % 29 - 14) / Float32(23)
        var lr = Float32(0.03125) / Float32(step + 1)
        var sc = _oracle_scalars(cfg, want_scalars, step, lr)
        var expected = List[Float32]()
        if cfg.kind == OPT_ADAGRAD:
            expected = o_adagrad(want_p, g, want_s2, sc[0], cfg.eps, cfg.wd, False)
        elif cfg.kind == OPT_ADAMAX:
            expected = o_adamax(want_p, g, want_s1, want_s2, cfg.f1, cfg.f2, cfg.eps, cfg.wd, sc[0], False)
        elif cfg.kind == OPT_NADAM:
            # This exact oracle covers no weight decay; no decoupled/L2
            # NAdam option coverage is attributed to these cases.
            expected = o_nadam(want_p, g, want_s1, want_s2, cfg.f1, cfg.f2, cfg.eps, sc[0], sc[1], sc[2], False)
        elif cfg.kind == OPT_RMSPROP:
            expected = o_rmsprop(want_p, g, want_s1, want_s2, want_s3, lr, cfg.f2, cfg.eps, cfg.wd, cfg.f7, False)
        else:
            expected = o_lion(want_p, g, want_s1, lr, cfg.f1, cfg.f2, cfg.wd, False)
        ex.upload(G, _ptr(g), PARAMS)
        opt_step(ex, cfg, scalar_state, step, lr, P, G, S1, S2, S3, PARAMS)
        ex.download_async(_ptr(p), P, PARAMS)
        ex.download_async(_ptr(s1), S1, PARAMS)
        ex.download_async(_ptr(s2), S2, PARAMS)
        ex.download_async(_ptr(s3), S3, PARAMS)
        ex.download_async(_ptr(gradient_copy), G, PARAMS)
        ex.sync()
        # A fixture-only negative control must fail the consumed-output
        # check, before any PASS marker. Ordinary arithmetic is unchanged.
        comptime if is_defined["MOJOLEARN_NATIVE_OPTIMIZER_BAD_REFERENCE"]():
            if step == 1:
                expected[0] = bitcast[DType.float32](bitcast[DType.uint32](expected[0]) ^ UInt32(1))
        for i in range(PARAMS):
            want_p[i] = expected[i]
            if cfg.kind == OPT_ADAGRAD:
                want_s2[i] = expected[PARAMS + i]
            else:
                want_s1[i] = expected[PARAMS + i]
                if cfg.kind != OPT_LION:
                    want_s2[i] = expected[2 * PARAMS + i]
                if cfg.kind == OPT_RMSPROP:
                    want_s3[i] = expected[3 * PARAMS + i]
            _equal(p[i], want_p[i], name, step, i)
            _equal(s1[i], want_s1[i], name, step, PARAMS + i)
            _equal(s2[i], want_s2[i], name, step, 2 * PARAMS + i)
            _equal(s3[i], want_s3[i], name, step, 3 * PARAMS + i)
            _equal(gradient_copy[i], g[i], name, step, 4 * PARAMS + i)
            for word in [p[i], s1[i], s2[i], s3[i], gradient_copy[i]]:
                digest = (digest ^ UInt64(bitcast[DType.uint32](word))) * UInt64(1099511628211)
        _equal(scalar_state.pw1, want_scalars[0], name, step, 5 * PARAMS)
        _equal(scalar_state.pw2, want_scalars[1], name, step, 5 * PARAMS + 1)
        _equal(scalar_state.mu_prod, want_scalars[2], name, step, 5 * PARAMS + 2)
        for word in [scalar_state.pw1, scalar_state.pw2, scalar_state.mu_prod]:
            digest = (digest ^ UInt64(bitcast[DType.uint32](word))) * UInt64(1099511628211)
    ex.free(P)
    ex.free(G)
    ex.free(S1)
    ex.free(S2)
    ex.free(S3)
    print("PASS sequence_optimizer", name, "parameters", PARAMS, "steps", STEPS, "digest", digest)


def check_sequence_optimizers[E: Exec](mut ex: E) raises:
    _run(ex, "adagrad_accumulator_decay", OptConfig(OPT_ADAGRAD, 0, Float32(0.037), Float32(0.0), Float32(1e-10), Float32(0.025), Float32(0.0)))
    _run(ex, "rmsprop_centered_momentum", OptConfig(OPT_RMSPROP, 1, Float32(0.0), Float32(0.9), Float32(1e-6), Float32(0.025), Float32(0.6)))
    _run(ex, "lion_decay", OptConfig(OPT_LION, 0, Float32(0.9), Float32(0.99), Float32(0.0), Float32(0.025), Float32(0.0)))
    # Exercise both sides of the independent two-branch lerp semantics.
    for beta in [Float32(0.25), Float32(0.9)]:
        _run(ex, "adamax_beta_" + String(beta), OptConfig(OPT_ADAMAX, 0, beta, Float32(0.99), Float32(1e-6), Float32(0.025), Float32(0.0)))
        _run(ex, "nadam_beta_" + String(beta), OptConfig(OPT_NADAM, 0, beta, Float32(0.99), Float32(1e-6), Float32(0.0), Float32(0.004)))
