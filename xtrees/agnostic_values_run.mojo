# SPDX-License-Identifier: Apache-2.0
"""KernelExplainer and PermutationExplainer as typed RUNs (lane bm-rest-shap, 2026-11-23).

The chunk loops that bindings/trees_agnostic_native_door.mojo drove over Python objects: the coalition schedule,
the synthetic rows, the explained model on them, the background means, the weighted solve (Kernel SHAP) or the
permutation fold (Permutation SHAP) and the logit link. Every stage is the x_trees glue operation the door called
as a Python entry before (xtrees/glue_runs.mojo, `XT_KSHAP_*`, `XT_PSHAP_*`, `XT_AGN_*`, `XT_BLOCK_MEAN`,
`XT_LOGIT`), reached here through the resolved glue entry: no Python object, import or callback in this file.

THE MODEL is an `AgnModel`: a typed evaluator of the explained estimator on a float32 row buffer.
`AgnMember` is the NATIVE MODEL HANDLE: the estimator's held member state (core/member_abi.mojo) and the member
operation of the explained method (predict of a regressor kind, predict_proba of a classifier). The families
with a published member entry serve it: the forests (RandomForest, ExtraTrees, the decision trees), DART and
the GBDT host and device images, and every other family that publishes one. The door keeps one more evaluator
for a compiled estimator whose class has no member entry (its compiled door per chunk, with the GIL held).

PARALLEL. Chunks are independent (each row's coalitions and permutations are drawn from its own stream
(seed, row): chunk boundaries move no bit), and each chunk writes its own rows of phi. On the CPU column with a
member whose kind is `MEMBER_GIL_FREE`, the chunks run across the allocation's cores (`host_predict_workers`),
one evaluator, one synthetic buffer and one member call per worker; the in-flight synthetic words stay under the
one-chunk budget (`AGN_BUDGET`). A GPU glue image (one device context, the pooled device buffer of
MOJOLEARN_AGN_IDN_SYN_POOL) and every other evaluator run the chunks in order on the calling thread.
"""
from std.math import min, max
from core.entry_table import EntryRef
from core.host_parallel import host_parallelize
from core.host_predict_threads import host_predict_workers
from core.member_abi import MemberRef, MemberPlan, MemberCall, MemberMatrix, member_call, member_call_of, MEMBER_F64
from core.hotpath_impl import cast_elements, HP_F32, HP_F64
from core.admitted_cast_elements import admitted_cast_elements
from xtrees.trees_agnostic_native_run import _chunk
from xtrees.glue_calls import (
    xt_kshap_schedule, xt_kshap_synth, xt_kshap_solve, xt_kshap_means, xt_kshap_solve_ey, xt_kshap_solve_model,
    xt_pshap_synth, xt_pshap_values, xt_pshap_dsynth, xt_pshap_dvalues, xt_pshap_values_linear,
    xt_pshap_values_model, xt_agn_model_load, xt_agn_model_release, xt_agn_pool_release, xt_block_mean, xt_logit,
)

comptime _I64 = MutPointer[Int64, MutUntrackedOrigin]
comptime _F32 = MutPointer[Float32, MutUntrackedOrigin]
comptime _INT = MutPointer[Int, MutUntrackedOrigin]

#: xtrees/agnostic.mojo AGN_BUDGET: synthetic words per explainer chunk (trees_agnostic_native_run `_chunk`).
comptime AGN_BUDGET = 1 << 25
#: MOJOLEARN_IDN_PSHAP_RESIDENT: 2^28 float32 words (1 GiB of synthetic rows on the device), a memory bound that
#: fits every supported GPU.
comptime AGN_BUDGET_RESIDENT = 1 << 28
#: x_trees_fast_switches bits (providers/python/xtrees_api_pyapi_door.mojo XTREES_FAST_SWITCHES).
comptime AGN_SW_KSHAP_FAST_BATCH = 8
comptime AGN_SW_IDN_SYN_POOL = 16
comptime AGN_SW_PSHAP_DELTA = 32


trait AgnModel(Movable & Deinitable):
    """The explained estimator on float32 C-order rows of `d` features, answering float32 (rows, k)."""

    def explained(mut self) raises -> Int:
        """The model on the explained rows (the request's `x`, `n` rows): the address of float32 (n, k) words,
        valid until the evaluator is destroyed."""
        ...

    def synthetic(mut self, count: Int, filled: Bool) raises -> Int:
        """A float32 input buffer of `count` words (zeroed when `filled`) for the next `rows` call; valid until
        the next `synthetic` call. A buffer of the same size is reused."""
        ...

    def rows(mut self, n: Int) raises -> Int:
        """The model on the first `n` rows of the synthetic buffer: the address of float32 (n, k) words, valid
        until the next `rows` or `synthetic` call."""
        ...

    def workers(self) -> Int:
        """How many evaluators may run at once (1: the calling thread only)."""
        ...

    def worker(self) raises -> Self:
        """A parallel worker's own evaluator over the same fitted state (only asked when `workers() > 1`)."""
        ...


struct AgnMember(AgnModel):
    """THE NATIVE MODEL HANDLE: one held member state of the explained estimator and the member operation of the
    explained method. `plan` is the address of the caller's `MemberPlan` and `refused` of one Int word; both
    outlive the RUN (the caller's UNPACK holds them). A member refusal writes 1 to `refused` and raises the
    member's words: the caller then answers through the model's own door, which raises its own words.
    `wide` (lane mf-shaprfe, 2026-11-23): the member writes the operation's output as float64 (the predict_proba
    of LogisticRegression, SVC, DART and GP classifiers, the naive Bayes and discriminant members on a
    float64-proba route): it writes into a float64 buffer of the chunk and the RUN reads that narrowed to
    float32, the conversion the door's `_evaluate` made of the public float64 output (core/hotpath_impl.mojo
    `cast_elements`, element by element, so no bit differs from the door's). `host`: the glue image is the CPU
    column (the request's `host`). Before, such a class was
    evaluated through the door, serially with the GIL held."""
    var member: MemberRef
    var plan: Int
    var state: Int
    var operation: Int
    var x: Int
    var n: Int
    var d: Int
    var k: Int
    var parallel: Bool
    var refused: Int
    var wide: Bool
    var host: Bool
    var call: MemberCall
    var fx: List[Float32]
    var syn: List[Float32]
    var out: List[Float32]
    var wide_out: List[Float64]

    def __init__(out self, member: MemberRef, plan: Int, state: Int, operation: Int, x: Int, n: Int, d: Int, k: Int,
                 parallel: Bool, refused: Int, wide: Bool, host: Bool) raises:
        self.member = member
        self.plan = plan
        self.state = state
        self.operation = operation
        self.x = x
        self.n = n
        self.d = d
        self.k = k
        self.parallel = parallel
        self.refused = refused
        self.wide = wide
        self.host = host
        self.call = member_call_of(MutPointer[MemberPlan, MutUntrackedOrigin](unsafe_from_address=plan)[], state)
        self.fx = List[Float32]()
        self.syn = List[Float32]()
        self.out = List[Float32]()
        self.wide_out = List[Float64]()

    def _run(mut self, x: Int, n: Int, dst: Int) raises:
        var d = self.d
        var k = self.k
        var member = self.member
        var operation = self.operation
        self.call.x = MemberMatrix(x, n, d)
        if not self.wide:
            self.call.dst = MemberMatrix(dst, n, k)
        else:
            if len(self.wide_out) < n * k:
                self.wide_out = List[Float64]()
                # Unfilled: the member writes every cell of dst before the narrowing reads it.
                self.wide_out = List[Float64](unsafe_uninit_length=n * k)
            self.call.dst = MemberMatrix(Int(self.wide_out.unsafe_ptr()), n, k, MEMBER_F64)
        try:
            _ = member_call(member, operation, self.call)
            if self.wide:
                var src = Int(self.wide_out.unsafe_ptr())
                # The CPU column's serial chunk across the host's cores, as the door's cast did; a parallel
                # worker's chunk (the workers are the split) and a GPU image's (no host threads in a GPU
                # binding) on the calling thread.
                var bad = (admitted_cast_elements(src, HP_F64, dst, HP_F32, n * k) if self.host and not self.parallel
                           else cast_elements(src, HP_F64, dst, HP_F32, n * k))
                if bad != 0:
                    raise Error("ValueError: the explained model output is not representable as float32")
        except problem:
            if self.refused != 0:
                _INT(unsafe_from_address=self.refused)[] = 1
            raise problem^

    def explained(mut self) raises -> Int:
        if len(self.fx) != self.n * self.k:
            # Unfilled: the member writes every cell of dst before anything reads it.
            self.fx = List[Float32](unsafe_uninit_length=self.n * self.k)
        var dst = Int(self.fx.unsafe_ptr())
        var x = self.x
        var n = self.n
        self._run(x, n, dst)
        return dst

    def synthetic(mut self, count: Int, filled: Bool) raises -> Int:
        if len(self.syn) != count:
            self.syn = List[Float32]()
            if filled:
                self.syn = List[Float32](length=count, fill=Float32(0))
            else:
                self.syn = List[Float32](unsafe_uninit_length=count)
        # A reused buffer is not zeroed again: every synthetic writer writes each cell the model then reads.
        return Int(self.syn.unsafe_ptr())

    def rows(mut self, n: Int) raises -> Int:
        if n * self.d > len(self.syn):
            raise Error("RuntimeError: agnostic explainer: synthetic rows exceed their buffer")
        if len(self.out) < n * self.k:
            self.out = List[Float32]()
            self.out = List[Float32](unsafe_uninit_length=n * self.k)
        var dst = Int(self.out.unsafe_ptr())
        var src = Int(self.syn.unsafe_ptr())
        self._run(src, n, dst)
        return dst

    def workers(self) -> Int:
        return host_predict_workers() if self.parallel else 1

    def worker(self) raises -> Self:
        return Self(self.member, self.plan, self.state, self.operation, self.x, self.n, self.d, self.k,
                    self.parallel, self.refused, self.wide, self.host)


struct AgnRequest(Copyable, Movable):
    """One shap_values call, typed. `x` float32 n x d (the explained rows), `bg` float32 nb x d (the background),
    `phi` float64 n x d x k (the caller's output, every cell written). `switches`: the glue image's
    x_trees_fast_switches; `fast`: the FAST tier; `host`: the glue image is the CPU column. `forest`: the five
    flat-forest array addresses of MOJOLEARN_IDN_SHAP_DEVICE_MODEL (empty: not that route) with
    `forest_words` = [trees, nodes, k, d, nb, rf_input]."""
    var x: Int
    var bg: Int
    var phi: Int
    var n: Int
    var nb: Int
    var d: Int
    var k: Int
    var seed: Int
    var switches: Int
    var fast: Bool
    var host: Bool
    var forest: List[Int]
    var forest_words: List[Int]

    def __init__(out self, x: Int, bg: Int, phi: Int, n: Int, nb: Int, d: Int, k: Int, seed: Int, switches: Int,
                 fast: Bool, host: Bool):
        self.x = x
        self.bg = bg
        self.phi = phi
        self.n = n
        self.nb = nb
        self.d = d
        self.k = k
        self.seed = seed
        self.switches = switches
        self.fast = fast
        self.host = host
        self.forest = List[Int]()
        self.forest_words = List[Int]()

    def device_forest(self) -> Bool:
        return len(self.forest) == 5


def agn_expected_run(glue: EntryRef, y: Int, nb: Int, k: Int, logit: Bool, ev: Int) raises:
    """The explainer's expected value: ev (float64 k) = the background mean of the model outputs y (float32
    nb x k), then (KernelExplainer, link 'logit') the link: shap's `expected_value = link(fnull)`."""
    _ = xt_block_mean(glue, y, ev, 1, nb, k)
    if logit:
        _ = xt_logit(glue, ev, k)


struct _KshapSchedule(Movable):
    """`x_trees_kshap_schedule` for M features and the resolved sample count, and its words."""
    var size_off: List[Int64]
    var size_w: List[Float64]
    var cdf: List[Float64]
    var m: Int
    var nfixed: Int
    var nfull: Int
    var npaired: Int
    var L: Int
    var wbits: Int

    def __init__(out self, glue: EntryRef, M: Int, ns: Int) raises:
        var h = max(1, M // 2)
        self.size_off = List[Int64](length=h + 1, fill=Int64(0))
        self.size_w = List[Float64](length=h, fill=0.0)
        self.cdf = List[Float64](length=h, fill=0.0)
        var sched = List[Int64](length=6, fill=Int64(0))
        if M > 1:
            _ = xt_kshap_schedule(glue, Int(self.size_off.unsafe_ptr()), Int(self.size_w.unsafe_ptr()),
                                  Int(self.cdf.unsafe_ptr()), Int(sched.unsafe_ptr()), M, max(ns, 0))
        self.m = Int(sched[0])
        self.nfixed = Int(sched[1])
        self.nfull = Int(sched[2])
        self.npaired = Int(sched[3])
        self.L = Int(sched[4])
        self.wbits = Int(sched[5])

    def so(self) -> Int:
        return Int(self.size_off.unsafe_ptr())

    def sw(self) -> Int:
        return Int(self.size_w.unsafe_ptr())

    def cd(self) -> Int:
        return Int(self.cdf.unsafe_ptr())

    def synth_words(self, rows: Int, nb: Int, d: Int, r0: Int, seed: Int) -> List[Int]:
        var p: List[Int] = [rows, nb, d, self.nfixed, self.m, self.nfull, self.L, self.npaired, r0, seed, self.wbits]
        return p^

    def solve_words(self, rows: Int, nb: Int, d: Int, r0: Int, seed: Int, k: Int, link: Int) -> List[Int]:
        var p: List[Int] = [rows, nb, d, self.nfixed, self.m, self.nfull, self.L, self.npaired, r0, seed, self.wbits,
                            k, link]
        return p^


def _join(failed: List[Int], messages: List[String]) raises:
    for w in range(len(failed)):  # driver-loop: over the workers, after the join
        if failed[w] != 0:
            raise Error(messages[w])


def _plan_workers(n: Int, per_row: Int, available: Int) -> Tuple[Int, Int]:
    """(workers, rows per task) for a parallel chunk loop: at most `available` workers, never more than one chunk
    budget of synthetic words in flight across them (each task holds rows * per_row words), at least one row per
    task. A worker count moves no bit."""
    var w = min(available, max(1, AGN_BUDGET // max(1, per_row)))
    w = max(1, min(w, n))
    var rows = max(1, min((n + w - 1) // w, AGN_BUDGET // max(1, w * per_row)))
    var tasks = (n + rows - 1) // rows
    return (max(1, min(w, tasks)), rows)


def agn_kernel_values_run[M: AgnModel](glue: EntryRef, req: AgnRequest, ns: Int, fnull: Int, logit: Bool,
                                       mut model: M) raises:
    """KernelExplainer values of `req.n > 0` rows into req.phi. `ns`: the requested sample count ('auto' already
    resolved to 2 M + 2048), capped at 2^M - 2 for M <= 30 as shap does; `fnull` float64 k (linked); `logit`: the
    link meets every background mean and the model on the rows before the solve."""
    var n = req.n
    var d = req.d
    var k = req.k
    var nb = req.nb
    var Mf = d
    var nsc = ns
    if Mf <= 30:
        nsc = min(nsc, (1 << Mf) - 2)
    var sc = _KshapSchedule(glue, Mf, nsc)
    var m = sc.m
    var link = 1 if logit else 0
    var fx = model.explained()   # the explained rows, one model call (the device forest route needs only this)
    var R = _chunk(m * nb * d, n)
    var bits = req.switches
    if m > 0 and req.fast and (bits & AGN_SW_KSHAP_FAST_BATCH) != 0 and not req.device_forest():
        # MOJOLEARN_KSHAP_FAST_BATCH: one host buffer for every chunk's synthetic rows, each chunk's linked
        # background means into one float64 (n, m, k) block, then the solve over as many rows per call as the
        # budget holds (the same words as the per-chunk solve).
        var ey = List[Float64](length=n * m * k, fill=0.0)
        var e0 = Int(ey.unsafe_ptr())
        var r0 = 0
        while r0 < n:  # driver-loop: one glue call per row chunk
            var rows = min(R, n - r0)
            var syn = model.synthetic(rows * m * nb * d, True)
            _ = xt_kshap_synth(glue, req.x + 4 * r0 * d, req.bg, sc.so(), sc.sw(), sc.cd(), syn,
                               sc.synth_words(rows, nb, d, r0, req.seed))
            var out = model.rows(rows * m * nb)
            var means: List[Int] = [rows, nb, m, k, link]
            _ = xt_kshap_means(glue, out, e0 + 8 * r0 * m * k, means)
            r0 += rows
        var S = _chunk(m * d, n)
        var s0 = 0
        while s0 < n:  # driver-loop: solve-batch loop (one glue call per batch)
            var rows = min(S, n - s0)
            _ = xt_kshap_solve_ey(glue, e0 + 8 * s0 * m * k, fx + 4 * s0 * k, fnull, sc.so(), sc.sw(), sc.cd(),
                                  req.phi + 8 * s0 * d * k, sc.solve_words(rows, nb, d, s0, req.seed, k, link))
            s0 += rows
        _ = ey^
    elif req.device_forest():
        # The forest on the device: masks -> model -> means -> solve per chunk, nothing crossing but the
        # chunk's rows and phi.
        _ = xt_agn_model_load(glue, req.forest, req.bg, req.forest_words)
        try:
            var r0 = 0
            while r0 < n:  # driver-loop: one glue call per row chunk
                var rows = min(R, n - r0)
                _ = xt_kshap_solve_model(glue, req.x + 4 * r0 * d, fx + 4 * r0 * k, fnull, sc.so(), sc.sw(),
                                         sc.cd(), req.phi + 8 * r0 * d * k,
                                         sc.solve_words(rows, nb, d, r0, req.seed, k, link))
                r0 += rows
        finally:
            _ = xt_agn_model_release(glue)
    elif m > 0 and req.host and model.workers() > 1 and n > 1:
        # The CPU column, a member safe on parallel workers: the chunks across the allocation's cores.
        var per_row = m * nb * d
        var plan = _plan_workers(n, per_row, model.workers())
        var W = plan[0]
        var Rp = plan[1]
        var tasks = (n + Rp - 1) // Rp
        var failed = List[Int](length=W, fill=0)
        var messages = List[String](length=W, fill=String(""))
        var fp = failed.unsafe_ptr()
        var mp = messages.unsafe_ptr()
        var x0 = req.x
        var bg0 = req.bg
        var p0 = req.phi
        var seed = req.seed
        var so = sc.so()
        var sw = sc.sw()
        var cd = sc.cd()
        var nfixed = sc.nfixed
        var nfull = sc.nfull
        var L = sc.L
        var npaired = sc.npaired
        var wbits = sc.wbits

        def task(w: Int) {imm fp, imm mp, imm model, imm glue, imm x0, imm bg0, imm p0, imm seed, imm so, imm sw,
                          imm cd, imm nfixed, imm nfull, imm L, imm npaired, imm wbits, imm fx, imm fnull, imm n,
                          imm d, imm k, imm nb, imm m, imm link, imm Rp, imm tasks, imm W}:
            try:
                var mine = model.worker()
                var t = w
                while t < tasks:  # driver-loop: this worker's chunks (round robin)
                    var r0 = t * Rp
                    var rows = min(Rp, n - r0)
                    var syn = mine.synthetic(rows * m * nb * d, False)
                    var sp: List[Int] = [rows, nb, d, nfixed, m, nfull, L, npaired, r0, seed, wbits]
                    _ = xt_kshap_synth(glue, x0 + 4 * r0 * d, bg0, so, sw, cd, syn, sp)
                    var out = mine.rows(rows * m * nb)
                    var vp: List[Int] = [rows, nb, d, nfixed, m, nfull, L, npaired, r0, seed, wbits, k, link]
                    _ = xt_kshap_solve(glue, out, fx + 4 * r0 * k, fnull, so, sw, cd, p0 + 8 * r0 * d * k, vp)
                    t += W
                _ = mine^
            except problem:
                mp[unsafe_offset=w] = String(problem)
                fp.unsafe_store(w, 1)

        host_parallelize(task, W)
        _join(failed, messages)
        _ = failed^
        _ = messages^
    else:
        var reuse = (bits & AGN_SW_IDN_SYN_POOL) != 0   # the pooled device synthetic buffer (GPU glue)
        try:
            var r0 = 0
            while r0 < n:  # driver-loop: one glue call per row chunk
                var rows = min(R, n - r0)
                var y_addr = 0
                if m > 0:
                    # Unfilled (fb-infer/C, 2026-11-17): x_trees_kshap_synth writes every cell.
                    var syn = model.synthetic(rows * m * nb * d, False)
                    _ = xt_kshap_synth(glue, req.x + 4 * r0 * d, req.bg, sc.so(), sc.sw(), sc.cd(), syn,
                                       sc.synth_words(rows, nb, d, r0, req.seed))
                    y_addr = model.rows(rows * m * nb)
                _ = xt_kshap_solve(glue, y_addr, fx + 4 * r0 * k, fnull, sc.so(), sc.sw(), sc.cd(),
                                   req.phi + 8 * r0 * d * k, sc.solve_words(rows, nb, d, r0, req.seed, k, link))
                r0 += rows
        finally:
            if reuse:
                _ = xt_agn_pool_release(glue)   # the pooled device buffer does not outlive the explanation
    _ = sc^


def agn_permutation_values_run[M: AgnModel](glue: EntryRef, req: AgnRequest, npm: Int, linear_wb: Int,
                                            linear_link: Int, mut model: M) raises:
    """PermutationExplainer values of `req.n > 0` rows into req.phi with `npm >= 1` permutations per row.
    `linear_wb` (nonzero): MOJOLEARN_IDN_PSHAP_RESIDENT, the [w | b] float32 block of this library's linear
    predict `link(X W^T + b)` with link word `linear_link`: synthetic rows, the linear model, means and marginals
    on the device; the evaluator is not called."""
    var n = req.n
    var d = req.d
    var k = req.k
    var nb = req.nb
    var mm = npm * (2 * d + 1)
    var R = _chunk(mm * nb * d, n)
    if req.device_forest():
        # The forest on the device: permutations -> model -> means -> marginals per chunk.
        _ = xt_agn_model_load(glue, req.forest, req.bg, req.forest_words)
        try:
            var r0 = 0
            while r0 < n:  # driver-loop: one glue call per row chunk
                var rows = min(R, n - r0)
                var p: List[Int] = [rows, nb, d, npm, r0, req.seed, k]
                _ = xt_pshap_values_model(glue, req.x + 4 * r0 * d, req.phi + 8 * r0 * d * k, p)
                r0 += rows
        finally:
            _ = xt_agn_model_release(glue)
        return
    if linear_wb != 0:
        var Rl = max(1, min(n, AGN_BUDGET_RESIDENT // max(1, mm * nb * d)))
        var r0 = 0
        while r0 < n:  # driver-loop: one glue call per row chunk
            var rows = min(Rl, n - r0)
            var p: List[Int] = [rows, nb, d, npm, r0, req.seed, k, linear_link]
            _ = xt_pshap_values_linear(glue, req.x + 4 * r0 * d, req.bg, linear_wb, req.phi + 8 * r0 * d * k, p)
            r0 += rows
        return
    var bits = req.switches
    var idn_pool = (bits & AGN_SW_IDN_SYN_POOL) != 0
    if req.fast and (bits & AGN_SW_PSHAP_DELTA) != 0:
        # MOJOLEARN_PSHAP_DELTA: per chunk, the device builds only the synthetic rows that differ from their
        # previous coalition's into the front of one reused buffer (their count into `tot`), the model runs on
        # that prefix, and the device expands the outputs back to every (coalition, background row). Three Int64
        # words per (coalition, background row): bound them as if d were at least 16.
        var Rd = min(R, _chunk(mm * nb * max(d, 16), n))
        var tot = List[Int64](length=1, fill=Int64(0))
        var t0 = Int(tot.unsafe_ptr())
        var syn = model.synthetic(Rd * mm * nb * d, True)
        var r0 = 0
        while r0 < n:  # driver-loop: one model call per row chunk
            var rows = min(Rd, n - r0)
            var sp: List[Int] = [rows, nb, d, npm, r0, req.seed]
            _ = xt_pshap_dsynth(glue, req.x + 4 * r0 * d, req.bg, syn, t0, sp)
            var nv = Int(_I64(unsafe_from_address=t0)[0])
            if nv < 0 or nv > rows * mm * nb:
                raise Error("RuntimeError: x_trees_pshap_dsynth returned an invalid row count")
            # The model on the first nv rows of the buffer (read in place).
            var out = model.rows(nv)
            var vp: List[Int] = [rows, nb, d, npm, r0, req.seed, k]
            _ = xt_pshap_dvalues(glue, req.x + 4 * r0 * d, req.bg, out, req.phi + 8 * r0 * d * k, t0, vp)
            r0 += rows
        _ = tot^
        return
    if req.host and model.workers() > 1 and n > 1:
        # The CPU column, a member safe on parallel workers: the chunks across the allocation's cores.
        var per_row = mm * nb * d
        var plan = _plan_workers(n, per_row, model.workers())
        var W = plan[0]
        var Rp = plan[1]
        var tasks = (n + Rp - 1) // Rp
        var failed = List[Int](length=W, fill=0)
        var messages = List[String](length=W, fill=String(""))
        var fp = failed.unsafe_ptr()
        var mp = messages.unsafe_ptr()
        var x0 = req.x
        var bg0 = req.bg
        var p0 = req.phi
        var seed = req.seed

        def task(w: Int) {imm fp, imm mp, imm model, imm glue, imm x0, imm bg0, imm p0, imm seed, imm n, imm d,
                          imm k, imm nb, imm npm, imm mm, imm Rp, imm tasks, imm W}:
            try:
                var mine = model.worker()
                var t = w
                while t < tasks:  # driver-loop: this worker's chunks (round robin)
                    var r0 = t * Rp
                    var rows = min(Rp, n - r0)
                    var syn = mine.synthetic(rows * mm * nb * d, True)
                    var sp: List[Int] = [rows, nb, d, npm, r0, seed]
                    _ = xt_pshap_synth(glue, x0 + 4 * r0 * d, bg0, syn, sp)
                    var out = mine.rows(rows * mm * nb)
                    var vp: List[Int] = [rows, nb, d, npm, r0, seed, k]
                    _ = xt_pshap_values(glue, out, p0 + 8 * r0 * d * k, vp)
                    t += W
                _ = mine^
            except problem:
                mp[unsafe_offset=w] = String(problem)
                fp.unsafe_store(w, 1)

        host_parallelize(task, W)
        _join(failed, messages)
        _ = failed^
        _ = messages^
        return
    try:
        var r0 = 0
        while r0 < n:  # driver-loop: one model call per row chunk
            var rows = min(R, n - r0)
            var syn = model.synthetic(rows * mm * nb * d, True)
            var sp: List[Int] = [rows, nb, d, npm, r0, req.seed]
            _ = xt_pshap_synth(glue, req.x + 4 * r0 * d, req.bg, syn, sp)
            var out = model.rows(rows * mm * nb)
            var vp: List[Int] = [rows, nb, d, npm, r0, req.seed, k]
            _ = xt_pshap_values(glue, out, req.phi + 8 * r0 * d * k, vp)
            r0 += rows
    finally:
        if idn_pool:
            _ = xt_agn_pool_release(glue)   # the pooled device buffer does not outlive the explanation
