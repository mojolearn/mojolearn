# SPDX-License-Identifier: Apache-2.0
"""`CausalLM.generate` as ONE `Program.decode` over ONE operation session
(lane x45c-causal-generate-program-decode, 2026-11-09; plan
docs/plans/EXECUTOR_CONSOLIDATION_AND_CLEANUP.md row X4 "CausalLM.generate as
Program.decode", API docs/plans/EXECUTOR_X1_API.md section 4).

`CausalLMProgram` is one inference `OpsSession` on the device holding

- the embedding table, the final norm weight and the head (or the tied
  table) as uploaded parameters, read in place at every run;
- every layer's weights as ONE packed parameter (`upload_block`), its carried
  float state as ONE declared state tensor and, where the block carries
  integers (attention: cached tokens; Mamba-2: buffer length; Mamba-3:
  buffer length and pending), one int32-word state tensor: the layers of a
  stack of any mix of attention, Mamba-1, Mamba-2 and Mamba-3 blocks are
  state tensors of this one session, not one executor per layer;
- the rows state (autograd/generation_ops.mojo: row status, lengths, eos
  ids, the alive word and the sampler's position).

One decode step is the captured body

    ids (b, 1) -> embedding -> step op per layer -> rms_norm -> head -> lm_pick -> ids (b, 1)

and the condition `lm_alive` (some row active, no row failed). A read runs
the prompt (L > 1) eagerly through the same ops, captures the body at its
first L = 1 step, and runs every further token as the executor's bounded
loop (`control_run`): the chunk rule enqueues several bodies between two
status reads, the gate undoes the writes of bodies past the stop, and the
token record is kept on the device. Nothing crosses to the host between
tokens: no per-layer state round trip, no host token loop, no host logits.

Weights are keyed by the caller's addresses and the process weight epoch
(core/weight_epoch.mojo); the state is loaded from the caller's arrays at
every read and exported once at its end (the caller's arrays are the truth
between public calls).
"""
from std.memory import bitcast
from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_IDENTICAL
from core.cast_in import CastIn, CAST_KIND_SIGNED
from core.neural_context import neural_ctx
from core.weight_epoch import weight_epoch
from sequence.ops import FP
from sequence.exec_device import DeviceExec
from autograd.rankn import ShapeN
from autograd.ops_api import (
    OpsSession, ops_session_open, ops_session, ops_session_close, OPS_DEVICE, OPS_EMBEDDING, OPS_RMS_NORM,
    OPS_TRANSFORMER_LINEAR,
)
from autograd.generation_ops import OPS_LM_PICK, OPS_LM_ALIVE, lm_rows_words, LM_PICK_GREEDY, LM_PICK_SAMPLE
from autograd.block_step_ops import (
    OPS_M1_STEP, OPS_M2_STEP, OPS_M3_STEP, OPS_ATTN_STEP, BLOCK_PART_WEIGHTS, BLOCK_PART_STATE,
    BLOCK_PART_WORDS, block_weight_sizes, block_state_sizes, block_int_words, block_step_reports_to_host,
    block_offsets, block_guard,
)
from autograd.program_control import control_prepare, control_run
from autograd.program_io import ProgramInput
from autograd.program_loop import LOOP_CHECK_AUTO

comptime _CTX = "MojoNeuralTransformerContextIdentical" if GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL else "MojoNeuralTransformerContextFast"


def _pieces(addrs: List[Int], sizes: List[Int], what: String) raises -> List[CastIn]:
    """One float32 C-order source per piece; an absent piece (size 0) is an
    empty source. A null address for a present piece is refused."""
    if len(addrs) != len(sizes):
        raise Error("CausalLM: " + what + ": expected " + String(len(sizes)) + " arrays, got " + String(len(addrs)))
    var out = List[CastIn](capacity=len(sizes))
    for i in range(len(sizes)):  # driver-loop: over a block's weight or state pieces (model structure)
        if sizes[i] == 0:
            out.append(CastIn.f32_c(0, 0, 1))
        elif addrs[i] == 0:
            raise Error("CausalLM: " + what + ": null buffer address at piece " + String(i))
        else:
            out.append(CastIn.f32_c(addrs[i], sizes[i], 1))
    return out^


def _ids_src(addr: Int, rows: Int, cols: Int) -> CastIn:
    """A C-contiguous int32 (rows, cols) host array."""
    return CastIn(addr, CAST_KIND_SIGNED, 4, rows, cols, cols * 4, 4)


struct CausalLMLayerSpec(Copyable, Movable):
    """One layer of the stack: its step op, the op's integer and float
    arguments, and the host addresses of its weight pieces in
    `block_weight_sizes` order (0 for an absent optional piece)."""
    var code: Int
    var ints: List[Int]
    var fs: List[Float32]
    var weights: List[Int]

    def __init__(out self, code: Int, var ints: List[Int], var fs: List[Float32], var weights: List[Int]):
        self.code = code
        self.ints = ints^
        self.fs = fs^
        self.weights = weights^


struct CausalLMDecodeResult(Movable):
    """What one read produced: the picked tokens step-major (`wrote` steps of
    b ids), the forwards run (a failed pick still ran its forward) and the
    first failing token (-1 none)."""
    var tokens: List[Int32]
    var wrote: Int
    var forwards: Int
    var bad_token: Int

    def __init__(out self):
        self.tokens = List[Int32]()
        self.wrote = 0
        self.forwards = 0
        self.bad_token = -1


struct CausalLMProgram(Movable):
    var sid: Int
    var b: Int
    var v: Int
    var d: Int
    var eps: Float32
    var embed: Int
    var norm: Int
    var head: Int
    var codes: List[Int]
    var ints: List[List[Int]]
    var fs: List[List[Float32]]
    var w: List[Int]
    var st: List[Int]
    var q: List[Int]
    var rows: Int
    var n_rows: Int
    var n_eos: Int
    #: the captured decode body and condition (0 until the first L = 1 step)
    var body: Int
    var cond: Int
    #: the pick's arguments the body was captured with
    var pick_ints: List[Int]
    var pick_fs: List[Float32]
    #: the version key: every weight address and the weight epoch at load
    var key: List[Int]
    var epoch: Int
    var closed: Bool

    def __init__(out self, pe: Int, v: Int, d: Int, pn: Int, eps: Float32, ph: Int, b: Int,
                 layers: List[CausalLMLayerSpec], n_eos: Int) raises:
        """Upload the weights (`pe` the (v, d) table, `pn` the (d) final norm
        or 0 for none, `ph` the (v, d) head or 0 for tied) and allocate the
        state of every layer for `b` rows (loaded at each read)."""
        if b < 1 or v < 1 or d < 1:
            raise Error("CausalLM: B, vocab and d_model must be positive")
        if pe == 0:
            raise Error("CausalLM: null embedding address")
        if pn != 0 and not (eps > 0):
            raise Error("CausalLM: the decode program needs a positive rms_norm eps")
        self.epoch = weight_epoch()
        var ex = DeviceExec(neural_ctx[_CTX]())
        self.sid = ops_session_open[DeviceExec, OPS_DEVICE](ex^, False)
        self.b = b
        self.v = v
        self.d = d
        self.eps = eps
        self.embed = 0
        self.norm = 0
        self.head = 0
        self.codes = List[Int]()
        self.ints = List[List[Int]]()
        self.fs = List[List[Float32]]()
        self.w = List[Int]()
        self.st = List[Int]()
        self.q = List[Int]()
        self.rows = 0
        self.n_eos = n_eos
        self.n_rows = lm_rows_words(b, n_eos)
        self.body = 0
        self.cond = 0
        self.pick_ints = List[Int]()
        self.pick_fs = List[Float32]()
        self.key = [pe, pn, ph, b]
        self.closed = False
        try:
            var s = ops_session[DeviceExec, OPS_DEVICE](self.sid)
            self.embed = s[].upload(CastIn.f32_c(pe, v, d), False)
            if pn != 0:
                self.norm = s[].upload(CastIn.f32_c(pn, 1, d), False)
            self.head = self.embed if ph == 0 else s[].upload(CastIn.f32_c(ph, v, d), False)
            for i in range(len(layers)):  # driver-loop: over the model's layers (one packed weight, state and words tensor each)
                ref spec = layers[i]
                var code = spec.code
                var ints = spec.ints.copy()
                var wsizes = block_weight_sizes(code, ints)
                var wh = s[].upload_block(code, ints, BLOCK_PART_WEIGHTS, _pieces(spec.weights, wsizes, "weights"),
                                          List[Int]())
                # The state tensor is allocated here and loaded at every read
                # (`load_layer`); its initial contents are the device's zeros.
                var ssizes = block_state_sizes(code, ints)
                var zsrc = List[CastIn](capacity=len(ssizes))
                var most = 1
                for k in range(len(ssizes)):  # small-loop(ssizes: a block's state pieces): layout arithmetic, no data
                    most = max(most, ssizes[k])
                var zero = List[Float32](length=most, fill=Float32(0))
                for k in range(len(ssizes)):  # small-loop(ssizes: a block's state pieces): one source each
                    zsrc.append(CastIn.f32_c(Int(zero.unsafe_ptr()), ssizes[k], 1) if ssizes[k] > 0
                                else CastIn.f32_c(0, 0, 1))
                var sh = s[].upload_block(code, ints, BLOCK_PART_STATE, zsrc, List[Int]())
                _ = len(zero)  # keep-alive past the staging copies
                var qh = 0
                var nk = block_int_words(code)
                if nk > 0:
                    qh = s[].upload_block(code, ints, BLOCK_PART_WORDS, List[CastIn](), List[Int](length=nk, fill=0))
                if code == OPS_ATTN_STEP and ints[7] == 0:
                    # A full-causal cache is append-only behind its position
                    # word: the loop's gate restores the word, not the cache.
                    s[].declare_append(sh, qh)
                self.codes.append(code)
                self.ints.append(ints^)
                self.fs.append(spec.fs.copy())
                self.w.append(wh)
                self.st.append(sh)
                self.q.append(qh)
                for k in range(len(spec.weights)):  # small-loop(spec.weights: a block's weight addresses): key words, no data
                    self.key.append(spec.weights[k])
            var zeros = List[Int32](length=self.n_rows, fill=Int32(0))
            self.rows = s[].upload_ids(_ids_src(Int(zeros.unsafe_ptr()), 1, self.n_rows),
                                       ShapeN.matrix(1, self.n_rows))
            _ = len(zeros)  # keep-alive past the staging copy
            s[].declare_state(self.rows)
            s[].ex.sync()
        except error:
            self.close()
            raise error

    def __del__(deinit self):
        if not self.closed:
            try:
                ops_session_close[DeviceExec, OPS_DEVICE](self.sid)
            except:
                pass

    def close(mut self):
        if self.closed:
            return
        self.closed = True
        try:
            ops_session_close[DeviceExec, OPS_DEVICE](self.sid)
        except:
            pass

    def matches(self, key: List[Int], n_eos: Int) raises -> Bool:
        """Whether this program serves a model whose weights sit at `key`
        ([embedding, norm, head, b, every layer's weight addresses]) at the
        current weight epoch."""
        return not self.closed and self.epoch == weight_epoch() and self.key == key and self.n_eos == n_eos

    # ------------------------------------------------------------ state
    def load_layer(mut self, i: Int, pieces: List[Int], words: List[Int]) raises:
        """Layer i's carried state from the caller's arrays (`block_state_sizes`
        order) and its carried integers, in place."""
        var s = ops_session[DeviceExec, OPS_DEVICE](self.sid)
        var code = self.codes[i]
        var ints = self.ints[i].copy()
        if self.q[i] != 0:
            s[].load_block(self.q[i], code, ints, BLOCK_PART_WORDS, List[CastIn](), words)
        s[].load_block(self.st[i], code, ints, BLOCK_PART_STATE,
                       _pieces(pieces, block_state_sizes(code, ints), "state"), List[Int]())

    def export_layers(mut self, pieces: List[List[Int]], mut words: List[List[Int]], reports: List[List[Int]]) raises:
        """Every layer's state into the caller's arrays and its carried
        integers into `words`, behind one wait; Mamba reports (`reports[i]`,
        0 addresses skipped, empty for none) of the last run too."""
        var s = ops_session[DeviceExec, OPS_DEVICE](self.sid)
        # The carried integers come back from their word tensors (two words
        # at most a layer): a resident entry dropped after a loop's overrun
        # rebuilds from the same words.
        var host = List[Float32](length=max(1, 2 * len(self.codes)), fill=Float32(0))
        words.clear()
        for i in range(len(self.codes)):  # driver-loop: over the model's layers (state export, queued)
            var code = self.codes[i]
            var ints = self.ints[i].copy()
            var sizes = block_state_sizes(code, ints)
            var offs = block_offsets(sizes, block_guard(code))
            var buf = s[].buffer(self.st[i])
            for k in range(len(sizes)):  # small-loop(sizes: a block's state pieces): one queued copy each
                if sizes[k] > 0 and pieces[i][k] != 0:
                    s[].ex.download_async(FP(unsafe_from_address=pieces[i][k]), buf + offs[k], sizes[k])
            var nk = block_int_words(code)
            if nk > 0:
                s[].ex.download_async(FP(unsafe_from_address=Int(host.unsafe_ptr()) + 8 * i), s[].buffer(self.q[i]), nk)
            if len(reports[i]) > 0:
                var ctx = s[].ex.ctx.copy()
                block_step_reports_to_host(ctx, s[].steps, code, Int(s[].buffer(self.w[i])), Int(buf), reports[i])
        s[].ex.sync()
        for i in range(len(self.codes)):  # driver-loop: over the model's layers (carried integers)
            var mine = List[Int]()
            for k in range(block_int_words(self.codes[i])):  # small-loop(at most two state words): word decode, no data
                mine.append(Int(bitcast[DType.int32](host[2 * i + k])))
            words.append(mine^)

    def load_rows(mut self, finished: List[Bool], lengths: List[Int], eos: List[Int], position: Int) raises:
        """The rows state: status, lengths, the eos ids and the sampler's next
        output position; alive = some row unfinished."""
        var b = self.b
        if len(finished) != b or len(lengths) != b or len(eos) != self.n_eos:
            raise Error("CausalLM: generation rows do not match the decode program")
        var w = List[Int32](length=self.n_rows, fill=Int32(0))
        var alive = 0
        for r in range(b):  # driver-loop: over batch rows (state words, no data)
            w[r] = Int32(1 if finished[r] else 0)
            w[b + r] = Int32(lengths[r])
            w[2 * b + r] = Int32(-1)
            if not finished[r]:
                alive = 1
        w[3 * b] = Int32(alive)
        w[3 * b + 1] = Int32(-1)
        w[3 * b + 2] = Int32(position)
        w[3 * b + 3] = Int32(self.n_eos)
        for i in range(self.n_eos):  # small-loop(eos: the stream's end ids): state words, no data
            w[3 * b + 4 + i] = Int32(eos[i])
        var s = ops_session[DeviceExec, OPS_DEVICE](self.sid)
        s[].load_into(self.rows, _ids_src(Int(w.unsafe_ptr()), 1, self.n_rows))
        s[].ex.sync()
        _ = len(w)

    def read_rows(mut self, mut finished: List[Bool], mut lengths: List[Int]) raises -> Tuple[Int, Int]:
        """The rows state back: status and lengths into the caller's lists;
        returns (alive, first failing token or -1)."""
        var b = self.b
        var host = List[Float32](length=self.n_rows, fill=Float32(0))
        var s = ops_session[DeviceExec, OPS_DEVICE](self.sid)
        s[].download(self.rows, FP(unsafe_from_address=Int(host.unsafe_ptr())))
        for r in range(b):  # driver-loop: over batch rows (state words, no data)
            finished[r] = bitcast[DType.int32](host[r]) != 0
            lengths[r] = Int(bitcast[DType.int32](host[b + r]))
        return (Int(bitcast[DType.int32](host[3 * b])), Int(bitcast[DType.int32](host[3 * b + 1])))

    # ------------------------------------------------------------ runs
    def _chain(mut self, ids: Int) raises -> Int:
        """embedding -> every layer's step op -> norm -> head -> pick."""
        var s = ops_session[DeviceExec, OPS_DEVICE](self.sid)
        var pad_none: List[Int] = [-1]
        var x = s[].op(OPS_EMBEDDING, [self.embed, ids], pad_none, List[Float32]())
        for i in range(len(self.codes)):  # driver-loop: over the model's layers, one step op each
            var hs: List[Int] = [x, self.w[i], self.st[i]]
            if self.q[i] != 0:
                hs.append(self.q[i])
            x = s[].op(self.codes[i], hs, self.ints[i].copy(), self.fs[i].copy())
        if self.norm != 0:
            var eps: List[Float32] = [self.eps]
            x = s[].op(OPS_RMS_NORM, [x, self.norm], List[Int](), eps)
        var logits = s[].op(OPS_TRANSFORMER_LINEAR, [x, self.head], List[Int](), List[Float32]())
        return s[].op(OPS_LM_PICK, [logits, self.rows], self.pick_ints.copy(), self.pick_fs.copy())

    def set_policy(mut self, pad: Int, greedy: Bool, top_k: Int, seed: UInt64, temperature: Float32,
                   top_p: Float32) raises:
        """The pick's arguments; a change drops the captured body."""
        var ints: List[Int] = [self.b, self.v, pad, LM_PICK_GREEDY if greedy else LM_PICK_SAMPLE,
                               0 if greedy else top_k, Int(seed & UInt64(0xFFFFFFFF)), Int(seed >> 32)]
        var fs: List[Float32] = [Float32(1.0) if greedy else temperature, Float32(1.0) if greedy else top_p]
        if ints == self.pick_ints and len(fs) == len(self.pick_fs) and (
                bitcast[DType.uint32](fs[0]) == bitcast[DType.uint32](self.pick_fs[0])
                and bitcast[DType.uint32](fs[1]) == bitcast[DType.uint32](self.pick_fs[1])):
            return
        var s = ops_session[DeviceExec, OPS_DEVICE](self.sid)
        if self.body != 0:
            s[].program_free(self.body)
            s[].program_free(self.cond)
            self.body = 0
            self.cond = 0
        self.pick_ints = ints^
        self.pick_fs = fs^

    def step(mut self, ids_addr: Int, l: Int, mut out: List[Int32]) raises -> Bool:
        """One forward of `l` tokens per row (the prompt, or one token) and
        its pick, eagerly; the picked ids go into `out` (b) and the result is
        the alive word. At its first L = 1 step the program captures the
        decode body and the condition from this very step."""
        var b = self.b
        var s = ops_session[DeviceExec, OPS_DEVICE](self.sid)
        s[].reset()
        var ids = s[].upload_ids(_ids_src(ids_addr, b, l), ShapeN.matrix(b, l))
        var picked = self._chain(ids)
        var bl: List[Int] = [b]
        var alive = s[].op(OPS_LM_ALIVE, [ids, self.rows], bl, List[Float32]())
        if l == 1 and self.body == 0:
            var ins: List[Int] = [ids]
            var po: List[Int] = [picked]
            var ao: List[Int] = [alive]
            self.body = s[].capture(ins, po)
            self.cond = s[].capture(ins, ao)
        var host = List[Float32](length=b, fill=Float32(0))
        s[].download(picked, FP(unsafe_from_address=Int(host.unsafe_ptr())))
        var flag = s[].scalar(alive)
        out.resize(b, Int32(0))
        for r in range(b):  # driver-loop: over batch rows (id words, no data)
            out[r] = bitcast[DType.int32](host[r])
        s[].free(ids)
        s[].reset()
        return flag != Float32(0)

    def decode(mut self, first: List[Int32], n: Int, mut result: CausalLMDecodeResult) raises:
        """Up to n further tokens from the ids `first` (b), as the executor's
        bounded loop over the captured body with the alive condition; the
        tokens of every committed iteration are appended to `result`."""
        if n <= 0:
            return
        if self.body == 0:
            raise Error("CausalLM: the decode body is captured at the first one-token step")
        var b = self.b
        var plan = control_prepare[DeviceExec, OPS_DEVICE](
            self.sid, self.body, self.cond, 0, self.sid, self.sid, n, False, LOOP_CHECK_AUTO, False, True)
        var inputs = List[ProgramInput]()
        inputs.append(ProgramInput(_ids_src(Int(first.unsafe_ptr()), b, 1), ShapeN.matrix(b, 1),
                                   List[Float64](), List[Int64]()))
        var r = control_run[DeviceExec, OPS_DEVICE](self.sid, plan, inputs^)
        _ = len(first)
        var iters = r.iterations
        result.forwards += iters
        if len(r.history) > 0 and iters > 0:
            ref h = r.history[0]
            for k in range(iters * b):  # driver-loop: over the device token record (id words, no data)
                result.tokens.append(bitcast[DType.int32](h.data[k]))
            result.wrote += iters

