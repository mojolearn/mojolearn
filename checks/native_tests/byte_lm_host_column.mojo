# SPDX-License-Identifier: Apache-2.0
# native-check: cpu
"""The byte LM host column's inference and training entry, ported from
`test_byte_lm_host.py` and `test_byte_lm_host_trainer.py` (lane tmig-lm-1,
2026-11-14).

Every unit calls `training/byte_lm_host_dispatch.mojo::byte_host_operation`,
the entry `_mojolearn_byte_lm_host` hands the inference and host-trainer
doors (operation 0 logits, 1 next bytes, 2 loss, 3 one training step), with
the Python tests' parameters, ids, thread words and optimizer scalars. What
stays in Python: returned types, the bits-to-float conversion, the byte-range
and shape refusals of the door, the threads/threaded admission, the trainer's
held scalars and the caller's ids copy.

Units (independent; the suite runs them across host cores):
  0  loss bits: the threaded path equals the unthreaded one (zero parameters)
  1  logits of [1, 4]: threaded bytes equal unthreaded bytes
  2  threads=3 loss equals the default (every core) loss; threads=2 logits
     write 1 * 4 * 256 cells
  3  zero parameters: logits of [2, 5] write 2 * 5 * 256 cells and every
     greedy tie goes to id 0
  4  next bytes on the threaded path equal the first-max argmax of the
     unthreaded full logits at each row's last position
  5  a step moves the parameters and both moments; the next step, from its
     outputs at completed 1, moves all three again
  6  a resumed step (m 0.25, v 0.75, completed 63) ends elsewhere than a
     fresh one from the same parameters
  7  `loss` is the step's loss bits at the current parameters and writes
     nothing; the step repeats its bits
  8  a four-row batch is a different gradient from the two-row one at the
     same parameter count

`test_routes` (lane py2mojo-D, 2026-11-22; ported from
`test_byte_lm_step_routes.py`): the original's inputs, `initial()`
(standard_normal(34944) * 0.02, seed 41) and batch 0 of `batches()` (seed
43), stored in `fixtures/byte_lm_routes/` by
`tools/py2mojo_native_fixtures_b7.py`. Every claim is bit for bit.
  0  `test_host_logits_reference_equals_threaded`: logits of batch 0's
     inputs ([2, 32]) on the reference path equal the threaded path's
  1  `test_host_loss_reference_equals_threaded`: the loss bits of batch 0
  2  `test_host_step_rows_equals_step_oracle` in one build: on batch 0's
     logits, `ce_host_rows` gives `ce_forward_oracle`'s loss and
     `ce_backward_oracle`'s dlogits; on the step's gradient, `adam_host_rows`
     gives `optimizer_step_oracle`'s parameters and both moments (the two arms
     `byte_host_gradient` / `byte_host_train_step` select between with
     -D MOJOLEARN_BYTE_LM_HOST_STEP_ROWS_OFF, training/byte_lm_host_backward.mojo)
  3  `test_host_step_oracle_loss_equals_reference_inference_loss`: the
     step's loss bits are the reference inference loss bits
"""
from std.memory import bitcast
from training.byte_lm_config import ByteConfig
from training.byte_lm_host_dispatch import byte_host_operation
from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_IDENTICAL
from training.byte_lm_host_execution import byte_host_optimizer
from training.checks.loss_contract import CeConfig
from training.checks.loss_oracle import ce_backward_oracle, ce_forward_oracle
from training.checks.optimizer_oracle import optimizer_step_oracle
from training.loss_host_rows import ce_host_rows
from training.optimizer_host_rows import adam_host_rows
from checks.native_tests.tsa_metrics_fixture_words import fixture_f32, fixture_i32
from std.testing import TestSuite, assert_true as require
from checks.native_tests.testsuite_cases import check_cases


comptime BYTE_LM_HOST_COLUMN_CASES = 9
comptime OP_LOGITS = 0
comptime OP_NEXT = 1
comptime OP_LOSS = 2
comptime OP_STEP = 3


def shape(batch: Int = 2) -> ByteConfig:
    """`ByteLanguageModelConfig(batch=batch)`: the Python default shape."""
    return ByteConfig(batch=batch, chunked_lm_head_v2=False)


def host_words(cfg: ByteConfig, threaded: Int, threads: Int, batch: Int = 0, length: Int = 0,
               completed: Int = 0) -> List[Int]:
    """The fifteen words the host doors hand the entry."""
    return [cfg.batch, cfg.length, cfg.d_model, cfg.n_heads, cfg.n_kv, cfg.head_dim, cfg.intermediate,
            cfg.n_layers, cfg.vocab_size, Int(cfg.chunked_lm_head_v2), threaded, threads, batch, length, completed]


def filled(n: Int, value: Float32) -> List[Float32]:
    return List[Float32](length=n, fill=value)


def step_ids(cfg: ByteConfig) -> List[Int32]:
    """`np.arange(batch * (length + 1)) % 256` as `[batch, length + 1]`."""
    var n = cfg.batch * (cfg.length + 1)
    var out = List[Int32](capacity=n)
    for i in range(n):  # driver-loop: over the fixture's token cells
        out.append(Int32(i % 256))
    return out^


def tokens(batch: Int, length: Int, scale: Int) -> List[Int32]:
    """`(np.arange(batch * length) * scale % 256)` as `[batch, length]`."""
    var out = List[Int32](capacity=batch * length)
    for i in range(batch * length):  # driver-loop: over the fixture's token cells
        out.append(Int32(i * scale % 256))
    return out^


def same_bits(a: List[Float32], b: List[Float32]) -> Bool:
    if len(a) != len(b):
        return False
    for i in range(len(a)):  # driver-loop: over one fixture's compared cells
        if bitcast[DType.uint32](a[i]) != bitcast[DType.uint32](b[i]):
            return False
    return True


def any_nonzero(a: List[Float32]) -> Bool:
    for i in range(len(a)):  # driver-loop: over one fixture's compared cells
        if a[i] != Float32(0):
            return True
    return False


def loss_bits(params: List[Float32], ids: List[Int32], cfg: ByteConfig, threaded: Int, threads: Int) raises -> Int:
    var a: List[Int] = [Int(params.unsafe_ptr()), Int(ids.unsafe_ptr())]
    var bits = byte_host_operation(OP_LOSS, a, host_words(cfg, threaded, threads), List[Float64]())
    _ = params  # read through its address by the synchronous call above
    _ = ids
    return bits


def logits(params: List[Float32], ids: List[Int32], batch: Int, length: Int, cfg: ByteConfig,
           threaded: Int, threads: Int) raises -> List[Float32]:
    var out = filled(batch * length * cfg.vocab_size, Float32(0))
    var a: List[Int] = [Int(params.unsafe_ptr()), Int(ids.unsafe_ptr()), Int(out.unsafe_ptr())]
    var written = byte_host_operation(OP_LOGITS, a, host_words(cfg, threaded, threads, batch, length), List[Float64]())
    _ = params
    _ = ids
    require(written == batch * length * cfg.vocab_size, "byte LM host logits wrote " + String(written) + " cells")
    return out^


def next_bytes(params: List[Float32], ids: List[Int32], batch: Int, length: Int, cfg: ByteConfig,
               threads: Int) raises -> List[Int32]:
    var out = List[Int32](length=batch, fill=Int32(-7))
    var a: List[Int] = [Int(params.unsafe_ptr()), Int(ids.unsafe_ptr()), Int(out.unsafe_ptr())]
    var written = byte_host_operation(OP_NEXT, a, host_words(cfg, 1, threads, batch, length), List[Float64]())
    _ = params
    _ = ids
    require(written == batch, "byte LM host next bytes wrote " + String(written) + " rows")
    return out^


def first_max(values: List[Float32], base: Int, count: Int) -> Int:
    """Test oracle: the first maximum of `count` cells (ties go low)."""
    var best = 0
    for j in range(1, count):  # driver-loop: over one row's vocabulary cells
        if values[base + j] > values[base + best]:
            best = j
    return best


@fieldwise_init
struct HostStep(Movable):
    var bits: Int
    var grad: List[Float32]
    var p: List[Float32]
    var m: List[Float32]
    var v: List[Float32]


def default_scalars(lr: Float64 = 1e-3) -> List[Float64]:
    """`LanguageModelHostTrainer`'s admitted optimizer: lr, beta1, beta2,
    eps, weight_decay."""
    return [lr, 0.9, 0.999, 1e-8, 1e-2]


def host_step(p: List[Float32], m: List[Float32], v: List[Float32], ids: List[Int32], cfg: ByteConfig,
              scalars: List[Float64], completed: Int) raises -> HostStep:
    var n = cfg.n_total()
    var grad = filled(n, Float32(0))
    var post_p = filled(n, Float32(0))
    var post_m = filled(n, Float32(0))
    var post_v = filled(n, Float32(0))
    var a: List[Int] = [Int(p.unsafe_ptr()), Int(m.unsafe_ptr()), Int(v.unsafe_ptr()), Int(ids.unsafe_ptr()),
                        Int(grad.unsafe_ptr()), Int(post_p.unsafe_ptr()), Int(post_m.unsafe_ptr()),
                        Int(post_v.unsafe_ptr())]
    var bits = byte_host_operation(OP_STEP, a, host_words(cfg, 0, 0, completed=completed), scalars)
    _ = p
    _ = m
    _ = v
    _ = ids
    return HostStep(bits, grad^, post_p^, post_m^, post_v^)


def finite_bits(bits: Int) -> Bool:
    return (UInt32(bits) & UInt32(0x7F800000)) != UInt32(0x7F800000)


def run_case(unit: Int) raises:
    var cfg = shape()
    var n = cfg.n_total()
    var zeros = filled(n, Float32(0))
    var ones = filled(n, Float32(1))
    if unit == 0:
        # test_threaded_flag_is_a_bool_and_both_paths_agree (loss half)
        var ids = step_ids(cfg)
        require(loss_bits(zeros, ids, cfg, 1, 0) == loss_bits(zeros, ids, cfg, 0, 0),
                "threaded loss bits differ from the unthreaded loss bits")
    elif unit == 1:
        # test_threaded_flag_is_a_bool_and_both_paths_agree (logits half): batch[:1, :4]
        var ids = tokens(1, 4, 1)
        require(same_bits(logits(zeros, ids, 1, 4, cfg, 1, 0), logits(zeros, ids, 1, 4, cfg, 0, 0)),
                "threaded logits bytes differ from the unthreaded logits bytes")
    elif unit == 2:
        # test_thread_count_is_kept_and_refused_out_of_range (numeric half)
        var ids = step_ids(cfg)
        require(loss_bits(zeros, ids, cfg, 1, 3) == loss_bits(zeros, ids, cfg, 1, 0),
                "threads=3 loss bits differ from the default threaded loss bits")
        var out = logits(zeros, List[Int32](length=4, fill=Int32(0)), 1, 4, cfg, 1, 2)
        require(len(out) == 1 * 4 * 256, "threads=2 logits are not [1, 4, 256]")
    elif unit == 3:
        # test_logits_shape_and_greedy_ties_go_low
        var ids = List[Int32](length=10, fill=Int32(0))
        var out = logits(zeros, ids, 2, 5, cfg, 1, 0)
        require(len(out) == 2 * 5 * 256, "zero-parameter logits are not [2, 5, 256]")
        var picked = next_bytes(zeros, ids, 2, 5, cfg, 0)
        require(picked[0] == 0 and picked[1] == 0, "a zero-parameter greedy tie did not go to id 0")
    elif unit == 4:
        # test_unthreaded_next_bytes_uses_full_logits_fallback
        var ids = tokens(2, 5, 7)
        var full = logits(zeros, ids, 2, 5, cfg, 0, 0)
        var picked = next_bytes(zeros, ids, 2, 5, cfg, 0)
        for row in range(2):  # driver-loop: over the fixture's two rows
            var expected = first_max(full, (row * 5 + 4) * cfg.vocab_size, cfg.vocab_size)
            require(Int(picked[row]) == expected, "row " + String(row) + ": threaded next byte "
                    + String(picked[row]) + " is not the unthreaded logits argmax " + String(expected))
    elif unit == 5:
        # test_a_step_reaches_the_binding_and_advances_the_state
        var ids = step_ids(cfg)
        var first = host_step(ones, zeros, zeros, ids, cfg, default_scalars(), 0)
        require(finite_bits(first.bits), "the step's loss is not finite")
        require(any_nonzero(first.grad), "the step's gradient is all zero")
        require(not same_bits(first.p, ones), "the step did not replace the parameters")
        require(any_nonzero(first.m) and any_nonzero(first.v), "the step left a moment all zero")
        var second = host_step(first.p, first.m, first.v, ids, cfg, default_scalars(), 1)
        require(not same_bits(second.p, first.p), "the second step did not move the parameters")
        require(not same_bits(second.m, first.m), "the second step did not move the first moment")
        require(not same_bits(second.v, first.v), "the second step did not move the second moment")
    elif unit == 6:
        # test_from_state_carries_the_moments_and_the_step_count (numeric half)
        var ids = step_ids(cfg)
        var resumed = host_step(ones, filled(n, Float32(0.25)), filled(n, Float32(0.75)), ids, cfg,
                                default_scalars(3e-3), 63)
        var fresh = host_step(ones, zeros, zeros, ids, cfg, default_scalars(3e-3), 0)
        require(not same_bits(resumed.p, fresh.p), "a resumed step ended where a fresh step ends")
    elif unit == 7:
        # test_loss_returns_the_float_of_the_bits
        var ids = step_ids(cfg)
        var stepped = host_step(ones, zeros, zeros, ids, cfg, default_scalars(), 0)
        var params = ones.copy()
        require(loss_bits(params, ids, cfg, 1, 0) == stepped.bits, "loss bits are not the step's loss bits")
        require(same_bits(params, ones), "loss wrote the parameters")
        require(host_step(ones, zeros, zeros, ids, cfg, default_scalars(), 0).bits == stepped.bits,
                "a repeated step returned other loss bits")
    elif unit == 8:
        # test_a_second_shape_reaches_the_binding_as_that_shape (numeric half)
        var four = shape(4)
        require(four.n_total() == n, "the four-row shape changed the parameter count")
        var four_step = host_step(ones, zeros, zeros, step_ids(four), four, default_scalars(), 0)
        var two_step = host_step(ones, zeros, zeros, step_ids(cfg), cfg, default_scalars(), 0)
        require(not same_bits(four_step.grad, two_step.grad), "four rows and two rows gave one gradient")
    else:
        raise Error("unknown byte LM host column case")


def test_cases() raises:
    comptime assert GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL
    check_cases("byte_lm_host_column", BYTE_LM_HOST_COLUMN_CASES, run_case)


comptime ROUTES_DIR = "checks/native_tests/fixtures/byte_lm_routes/"
comptime BYTE_LM_ROUTES_CASES = 4


def routes_params(cfg: ByteConfig) raises -> List[Float32]:
    """`initial()` of test_byte_lm_step_routes.py."""
    return fixture_f32(ROUTES_DIR + "initial.f32", cfg.n_total())


def routes_batch0(cfg: ByteConfig) raises -> List[Int32]:
    """`batches()[0]`: the first `[batch, length + 1]` of the stored `[3, 2, 33]`."""
    var stack = fixture_i32(ROUTES_DIR + "batches.i32", 3 * cfg.batch * (cfg.length + 1))
    var out = List[Int32](capacity=cfg.batch * (cfg.length + 1))
    for i in range(cfg.batch * (cfg.length + 1)):  # driver-loop: over one batch's token cells
        out.append(stack[i])
    return out^


def split_rows(ids: List[Int32], cfg: ByteConfig, shift: Int) -> List[Int32]:
    """`ids[:, :-1]` (shift 0) or `ids[:, 1:]` (shift 1), row-major."""
    var out = List[Int32](capacity=cfg.batch * cfg.length)
    for b in range(cfg.batch):  # driver-loop: over the batch rows
        for l in range(cfg.length):
            out.append(ids[b * (cfg.length + 1) + l + shift])
    return out^


def routes_case(unit: Int) raises:
    var cfg = shape()
    var params = routes_params(cfg)
    var ids = routes_batch0(cfg)
    if unit == 0:
        var inputs = split_rows(ids, cfg, 0)
        require(same_bits(logits(params, inputs, cfg.batch, cfg.length, cfg, 0, 0),
                          logits(params, inputs, cfg.batch, cfg.length, cfg, 1, 0)),
                "random-init logits: the reference path's bytes differ from the threaded path's")
    elif unit == 1:
        var reference = loss_bits(params, ids, cfg, 0, 0)
        var threaded = loss_bits(params, ids, cfg, 1, 0)
        require(reference == threaded, "random-init loss bits: reference " + String(reference) + ", threaded "
                + String(threaded))
    elif unit == 2:
        var targets = split_rows(ids, cfg, 1)
        var head = logits(params, split_rows(ids, cfg, 0), cfg.batch, cfg.length, cfg, 0, 0)
        var ce_cfg = CeConfig.causal_lm(cfg.vocab_size)
        var rows = ce_host_rows(head, targets, ce_cfg)
        var oracle = ce_forward_oracle(head, targets, ce_cfg)
        ce_backward_oracle(oracle, targets, ce_cfg)
        require(bitcast[DType.uint32](rows[0]) == bitcast[DType.uint32](oracle.loss[0]),
                "ce_host_rows loss " + String(rows[0]) + ", ce_forward_oracle " + String(oracle.loss[0]))
        require(same_bits(rows[1], oracle.dlogits), "ce_host_rows dlogits differ from ce_backward_oracle's")
        var n = cfg.n_total()
        var zeros = filled(n, Float32(0))
        var step = host_step(params, zeros, zeros, ids, cfg, default_scalars(), 0)
        var opt = byte_host_optimizer(default_scalars())
        var p_rows = params.copy()
        var m_rows = zeros.copy()
        var v_rows = zeros.copy()
        adam_host_rows(p_rows, step.grad, m_rows, v_rows, opt, 1)
        var p_oracle = params.copy()
        var g_oracle = step.grad.copy()
        var m_oracle = zeros.copy()
        var v_oracle = zeros.copy()
        var initialized = List[Bool](length=cfg.n_tensors(), fill=True)
        _ = optimizer_step_oracle(p_oracle, g_oracle, m_oracle, v_oracle, initialized, cfg.offsets(), opt, 1)
        require(same_bits(p_rows, p_oracle), "adam_host_rows parameters differ from optimizer_step_oracle's")
        require(same_bits(m_rows, m_oracle), "adam_host_rows first moment differs from optimizer_step_oracle's")
        require(same_bits(v_rows, v_oracle), "adam_host_rows second moment differs from optimizer_step_oracle's")
        require(same_bits(step.p, p_rows), "the step's parameters differ from adam_host_rows on its gradient")
    elif unit == 3:
        var n = cfg.n_total()
        var zeros = filled(n, Float32(0))
        var step_bits = host_step(params, zeros, zeros, ids, cfg, default_scalars(), 0).bits
        var reference = loss_bits(params, ids, cfg, 0, 0)
        require(step_bits == reference, "step loss bits " + String(step_bits) + ", reference inference loss bits "
                + String(reference))
    else:
        raise Error("unknown byte LM routes case")


def test_routes() raises:
    comptime assert GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL
    check_cases("byte_lm_host_routes", BYTE_LM_ROUTES_CASES, routes_case)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
