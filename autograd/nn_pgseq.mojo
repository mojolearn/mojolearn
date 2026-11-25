# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Parameter groups of `Sequential.compile` (lane nn-blocks-pgseq, 2026-11-25).

`compile(..., param_groups=[{"layers": [...]} | {"params": [...]}, "lr": ..., "weight_decay": ..., ...],
no_decay=...)`: torch.optim's `param_groups` for a Sequential model (a new head at a higher rate, a pretrained
body lower, biases without weight decay). The per-group keys and their refusal words are the operation session's
(autograd/nn_pg.mojo pg_admit_key); the membership refusals (an empty group, a tensor listed twice, a tensor in
two groups) are its pg_check_membership over tensor indices.

THE TABLE. compile admits each group's keys and computes each group's optimizer words with compile's own words
body (bindings/nn_optim_compile_pack.mojo nn_optim_compile_words: one admission, one set of defaults). The fit
door resolves the membership against the model's tensor layout (autograd/model_checkpoint.mojo model_layout, the
state_dict names) ONCE per fit call and hands this module the words below after the schedule's argument words
(autograd/model_execution.mojo model_fit_native):

    [G, T, D, then G groups of PGSEQ_GROUP_WORDS words (lr, flags, f1, f2, eps, weight_decay, f7, 0),
     then T group indices, one per layout tensor]

G groups, T layout tensors, D the default group (the tensors no group lists, with compile's own settings).
`pgseq_table` turns them into a `PgSeqTable` held by FitConfig.groups: one OptConfig and base rate per group and
the maximal RUNS of consecutive tensors of one group (element ranges of the flat parameter vector).

THE STEP (`pgseq_step`, model_fit_state's grouped branch; the ungrouped fit runs its single launch as before, the
branch one Bool per batch). Per run, the group's own update over the run's slice of P, G and the state buffers
with the group's scalars: sgd / momentum / adam / adamw / rmsprop / adagrad one OP_OPT launch a run
(autograd/nn_optim.mojo nn_opt_step, sequence/recurrent.mojo opt_step), the group's running products (beta1^t,
beta2^t) advanced once a step. LAMB: the global clip ONCE over every tensor of every group (lamb_clip, three
launches; its coefficient is optimizer-wide), then per run the moments and update direction (lamb_upd), the trust
ratios once over every tensor when any group needs them (lamb_trust; a run whose group has no weight decay and no
always_adapt gets ratio one), and per run the update (lamb_apply over the run's own table). Launches per step:
one a run (+3 clip, +2 or +1 trust for LAMB, +one ratio fill per untrusting run). Nothing is read back.

Two groups whose scalars equal the ungrouped settings give, word for word, the ungrouped fit: every launch is
elementwise (or per tensor, over the same blocks), and the global clip and trust ratios are the same launches
over the same table.

SCHEDULES. One schedule scales every group, as a torch scheduler scales each `param_group["lr"]` from its own
initial rate: each group's rate is the schedule's value at that group's base rate (autograd/model.mojo
model_epoch_lr; per-step schedules filled once per fit by nn_step_lrs per group). reduce_on_plateau is refused
with groups (its rate is one float64 state word of the fit state; a rate per group is not carried).

STATE. The moment buffers are per element, so a group's state is its slice of optimizer_m_ / optimizer_v_. A
group's running products are a function of the step count (one identical_mul a step from 1), so a resumed fit
rebuilds them from the saved step count (`pgseq_state`): the same words as carrying them.
"""
from std.sys import is_defined
from checks.numerics import ftz, identical_mul
from sequence.exec_trait import Exec
from sequence.ops import FP, Args, OP_FILL, dummy_ptr
from sequence.recurrent import OptConfig, OptState, opt_scalars
from sequence.lamb_core import lamb_apply, lamb_clip, lamb_table, lamb_trust, lamb_unit_ratio, lamb_upd
from autograd.nn_optim import (
    NN_LAMB_ALWAYS_ADAPT, NN_LAMB_BIAS_CORRECTION, NN_LAMB_CLIP, NN_OPT_ADAGRAD, NN_OPT_LAMB, NN_SCHED_PLATEAU,
    NnLamb, nn_opt_step,
)

comptime PGSEQ_GROUP_WORDS = 8   # lr, flags, f1, f2, eps, weight_decay, f7, reserved (0)
comptime PGSEQ_HEADER = 3        # G, T, D

# Sabotage arms (tools/identity_lanes/coverage_sabotage.tsv coverage-autograd-pgseq_*, lane
# sequential-param-groups). Off in every build; an arm's patch switches one on. GROUP_LR: each run steps at the
# NEXT group's rate (the neighbouring group's lr). CLIP_PER_GROUP: LAMB's clip coefficient is computed over each
# run's own tensors before its update (a clip per group, not one global clip). The plain name acts on the device
# executor only, _HOST on the host executor only.
comptime PGSEQ_SABOTAGE_GROUP_LR = is_defined["MOJOLEARN_PGSEQ_SABOTAGE_GROUP_LR"]()
comptime PGSEQ_SABOTAGE_GROUP_LR_HOST = is_defined["MOJOLEARN_PGSEQ_SABOTAGE_GROUP_LR_HOST"]()
comptime PGSEQ_SABOTAGE_CLIP_PER_GROUP = is_defined["MOJOLEARN_PGSEQ_SABOTAGE_CLIP_PER_GROUP"]()
comptime PGSEQ_SABOTAGE_CLIP_PER_GROUP_HOST = is_defined["MOJOLEARN_PGSEQ_SABOTAGE_CLIP_PER_GROUP_HOST"]()


struct PgSeqTable(Copyable, Movable):
    """The groups of one Sequential fit (empty: no groups, the ungrouped path).

    cfgs, lrs: each group's optimizer words and base learning rate. default: the default group.
    toffs: the tensor offsets the runs are cut from (NnLamb's rule: 0, the layout's rising ends, n).
    run_lo, run_hi: each run's element range; run_group its group; run_t0, run_nt its first tensor (an index
    into toffs) and tensor count."""

    var cfgs: List[OptConfig]
    var lrs: List[Float32]
    var default: Int
    var toffs: List[Int]
    var run_lo: List[Int]
    var run_hi: List[Int]
    var run_group: List[Int]
    var run_t0: List[Int]
    var run_nt: List[Int]

    def __init__(out self):
        self.cfgs = List[OptConfig]()
        self.lrs = List[Float32]()
        self.default = 0
        self.toffs = List[Int]()
        self.run_lo = List[Int]()
        self.run_hi = List[Int]()
        self.run_group = List[Int]()
        self.run_t0 = List[Int]()
        self.run_nt = List[Int]()

    def on(self) -> Bool:
        return len(self.cfgs) > 0

    def groups(self) -> Int:
        return len(self.cfgs)

    def runs(self) -> Int:
        return len(self.run_lo)


def pgseq_table(fp: List[Float64], at: Int, kind: Int, offsets: List[Int], n: Int) raises -> PgSeqTable:
    """The group table from the fit's words fp[at:] (layout above), the layout's tensor `offsets` (model_layout:
    0 then each tensor's end) and the parameter count n. Refuses damaged words by name."""
    var table = PgSeqTable()
    if len(fp) < at + PGSEQ_HEADER:
        raise Error("model fit: the parameter-group words are damaged; call compile again")
    var G = Int(fp[at])
    var T = Int(fp[at + 1])
    var D = Int(fp[at + 2])
    if G < 1 or T != len(offsets) - 1 or D < 0 or D >= G or len(fp) != at + PGSEQ_HEADER + G * PGSEQ_GROUP_WORDS + T:
        raise Error("model fit: the parameter groups do not match this model's " + String(len(offsets) - 1)
                    + " parameter tensors; call compile again")
    table.default = D
    var w = at + PGSEQ_HEADER
    for g in range(G):  # driver-loop: over parameter groups (once per fit)
        var lr = fp[w]
        if not (lr > 0.0):
            raise Error("ValueError: param_groups[" + String(g) + "]: the learning rate must be positive")
        table.lrs.append(Float32(lr))
        table.cfgs.append(OptConfig(kind, Int(fp[w + 1]), Float32(fp[w + 2]), Float32(fp[w + 3]),
                                    Float32(fp[w + 4]), Float32(fp[w + 5]), Float32(fp[w + 6])))
        w += PGSEQ_GROUP_WORDS
    # the tensor offsets by NnLamb's rule (autograd/nn_optim.mojo: rising ends, then n), each kept tensor's group
    var groups_of = List[Int]()
    table.toffs.append(0)
    for k in range(1, len(offsets)):  # driver-loop: over parameter tensors (description metadata, once per fit)
        var gk = Int(fp[w + k - 1])
        if gk < 0 or gk >= G:
            raise Error("model fit: the parameter-group words are damaged; call compile again")
        if offsets[k] > table.toffs[len(table.toffs) - 1] and offsets[k] <= n:
            table.toffs.append(offsets[k])
            groups_of.append(gk)
    if table.toffs[len(table.toffs) - 1] < n:
        table.toffs.append(n)
        groups_of.append(D)
    # maximal runs of consecutive tensors of one group
    for j in range(len(groups_of)):  # driver-loop: over parameter tensors (once per fit)
        var r = len(table.run_lo)
        if r > 0 and table.run_group[r - 1] == groups_of[j]:
            table.run_hi[r - 1] = table.toffs[j + 1]
            table.run_nt[r - 1] += 1
        else:
            table.run_lo.append(table.toffs[j])
            table.run_hi.append(table.toffs[j + 1])
            table.run_group.append(groups_of[j])
            table.run_t0.append(j)
            table.run_nt.append(1)
    return table^


def pgseq_admit_fit(table: PgSeqTable, sched: Int) raises:
    """The fit-time refusal: reduce_on_plateau keeps one rate in the fit state, not one a group."""
    if table.on() and sched == NN_SCHED_PLATEAU:
        raise Error("ValueError: param_groups with schedule='reduce_on_plateau' is not available: its rate is "
                    + "one value of the training state; use another schedule, or no param_groups")


struct PgSeqState(Movable):
    """Each group's running products (beta1^t, beta2^t, NAdam's mu product), one triple a group, and LAMB's
    bias corrections of the current step (allocated once per fit, written each step)."""

    var pw1: List[Float32]
    var pw2: List[Float32]
    var mu: List[Float32]
    var bc1: List[Float32]
    var bc2: List[Float32]

    def __init__(out self, table: PgSeqTable, step: Int):
        """The products after `step` optimizer steps, replayed one multiplication a step from 1 (the words the
        steps themselves carried; once per fit)."""
        self.pw1 = List[Float32]()
        self.pw2 = List[Float32]()
        self.mu = List[Float32]()
        self.bc1 = List[Float32](length=table.groups(), fill=Float32(1.0))
        self.bc2 = List[Float32](length=table.groups(), fill=Float32(1.0))
        for g in range(table.groups()):  # driver-loop: over parameter groups (once per fit)
            var cfg = table.cfgs[g]
            var st = OptState()
            if cfg.kind == NN_OPT_LAMB:
                if (cfg.flags & NN_LAMB_BIAS_CORRECTION) != 0:
                    for _ in range(step):  # driver-loop: over the steps already run (once per fit, host scalars)
                        st.pw1 = ftz(identical_mul(st.pw1, cfg.f1))
                        st.pw2 = ftz(identical_mul(st.pw2, cfg.f2))
            else:
                for s in range(1, step + 1):  # driver-loop: over the steps already run (once per fit, host scalars)
                    _ = opt_scalars(cfg, st, s, table.lrs[g])
            self.pw1.append(st.pw1)
            self.pw2.append(st.pw2)
            self.mu.append(st.mu_prod)

    def of(self, g: Int) -> OptState:
        var st = OptState()
        st.pw1 = self.pw1[g]
        st.pw2 = self.pw2[g]
        st.mu_prod = self.mu[g]
        return st^

    def keep(mut self, g: Int, st: OptState):
        self.pw1[g] = st.pw1
        self.pw2[g] = st.pw2
        self.mu[g] = st.mu_prod


struct PgSeqLamb(Movable):
    """LAMB's per-run tables (each run's tensors from its own start, `lamb_table`'s format), uploaded once per fit
    into one buffer; run r's table starts at TABS + tab_at[r]. Empty for every other optimizer."""

    var on: Bool
    var TABS: FP
    var tab_at: List[Int]
    var run_nb: List[Int]
    var trust: List[Bool]
    var any_trust: Bool

    def __init__[E: Exec](out self, mut ex: E, table: PgSeqTable) raises:
        self.on = table.on() and table.cfgs[0].kind == NN_OPT_LAMB
        self.TABS = dummy_ptr()
        self.tab_at = List[Int]()
        self.run_nb = List[Int]()
        self.trust = List[Bool]()
        self.any_trust = False
        if not self.on:
            return
        for g in range(table.groups()):  # driver-loop: over parameter groups (once per fit)
            var c = table.cfgs[g]
            var needs = c.wd != Float32(0.0) or (c.flags & NN_LAMB_ALWAYS_ADAPT) != 0
            self.trust.append(needs)
            if needs:
                self.any_trust = True
        var words = List[Float32]()
        for r in range(table.runs()):  # driver-loop: over runs of one group's tensors (once per fit)
            var offs = List[Int]()
            var t0 = table.run_t0[r]
            for k in range(table.run_nt[r] + 1):  # driver-loop: over the run's tensor offsets (once per fit)
                offs.append(table.toffs[t0 + k] - table.run_lo[r])
            var tab = List[Float32]()
            self.run_nb.append(lamb_table(offs, tab))
            self.tab_at.append(len(words))
            for i in range(len(tab)):  # driver-loop: over the run table's words (once per fit)
                words.append(tab[i])
        self.TABS = ex.alloc(len(words))
        ex.upload(self.TABS, FP(unsafe_from_address=Int(words.unsafe_ptr())), len(words))
        ex.sync()
        _ = words^

    def release[E: Exec](mut self, mut ex: E) raises:
        if not self.on:
            return
        ex.sync()
        ex.free(self.TABS)
        self.on = False


@always_inline
def _rate[E: Exec](table: PgSeqTable, lrs: List[Float32], g: Int) -> Float32:
    """Group g's rate this step (the sabotage arm takes the neighbouring group's)."""
    comptime if (PGSEQ_SABOTAGE_GROUP_LR and E.IS_DEVICE) or (PGSEQ_SABOTAGE_GROUP_LR_HOST and not E.IS_DEVICE):
        return lrs[(g + 1) % table.groups()]
    return lrs[g]


def pgseq_fill_accumulators[E: Exec](mut ex: E, table: PgSeqTable, s2: FP) raises:
    """Adagrad's initial_accumulator_value per group: one fill a run whose group's value is not 0 (a fresh fit)."""
    for r in range(table.runs()):  # driver-loop: over runs of one group's tensors (once per fit)
        var c = table.cfgs[table.run_group[r]]
        if c.kind == NN_OPT_ADAGRAD and c.f2 != Float32(0):
            var a = Args()
            a.p0 = s2 + table.run_lo[r]
            a.f0 = c.f2
            ex.launch[OP_FILL](a, table.run_hi[r] - table.run_lo[r])


def pgseq_step[E: Exec](
    mut ex: E, table: PgSeqTable, mut state: PgSeqState, t: Int, lrs: List[Float32],
    P: FP, Gr: FP, s1: FP, s2: FP, s3: FP, lamb: NnLamb, plamb: PgSeqLamb,
) raises:
    """One optimizer step of a grouped Sequential fit (step t, one-based); lrs: each group's rate this step."""
    var G = table.groups()
    if not plamb.on:
        for r in range(table.runs()):  # driver-loop: over runs of one group's tensors (a few a model)
            var g = table.run_group[r]
            var lo = table.run_lo[r]
            var st = state.of(g)
            nn_opt_step(ex, table.cfgs[g], st, t, _rate[E](table, lrs, g), P + lo, Gr + lo, s1 + lo, s2 + lo,
                        s3 + lo, table.run_hi[r] - lo, lamb)
        for g in range(G):  # driver-loop: over parameter groups (each group's products advance once a step)
            var st = state.of(g)
            _ = opt_scalars(table.cfgs[g], st, t, lrs[g])
            state.keep(g, st)
        return
    # LAMB: flags and max_grad_norm are optimizer-wide (every group's words carry compile's)
    var c0 = table.cfgs[0]
    var clip = (c0.flags & NN_LAMB_CLIP) != 0
    var per_group_clip = False
    comptime if (PGSEQ_SABOTAGE_CLIP_PER_GROUP and E.IS_DEVICE) or (PGSEQ_SABOTAGE_CLIP_PER_GROUP_HOST and not E.IS_DEVICE):
        per_group_clip = True
    if clip and not per_group_clip:
        # ONE clip over every tensor of every group, before any update
        lamb_clip(ex, Gr, lamb.TAB, lamb.partsA, lamb.nrm, lamb.scal, lamb.nt, lamb.nb, c0.f7)
    if (c0.flags & NN_LAMB_BIAS_CORRECTION) != 0:
        for g in range(G):  # driver-loop: over parameter groups (one product a group a step)
            var c = table.cfgs[g]
            state.pw1[g] = ftz(identical_mul(state.pw1[g], c.f1))
            state.pw2[g] = ftz(identical_mul(state.pw2[g], c.f2))
            state.bc1[g] = Float32(1.0) - state.pw1[g]
            state.bc2[g] = Float32(1.0) - state.pw2[g]
    for r in range(table.runs()):  # driver-loop: over runs of one group's tensors (a few a model)
        var g = table.run_group[r]
        var c = table.cfgs[g]
        var lo = table.run_lo[r]
        if clip and per_group_clip:
            # SABOTAGE: the coefficient from this run's tensors only
            lamb_clip(ex, Gr + lo, plamb.TABS + plamb.tab_at[r], lamb.partsA, lamb.nrm, lamb.scal, table.run_nt[r],
                      plamb.run_nb[r], c0.f7)
        lamb_upd(ex, P + lo, Gr + lo, s1 + lo, s2 + lo, lamb.U + lo, lamb.scal, table.run_hi[r] - lo, clip,
                 c0.flags, c.f1, c.f2, c.eps, c.wd, state.bc1[g], state.bc2[g])
    if plamb.any_trust:
        lamb_trust(ex, P, lamb.U, lamb.TAB, lamb.partsA, lamb.partsB, lamb.ratio, lamb.nt, lamb.nb, c0.flags)
        for r in range(table.runs()):  # driver-loop: over runs (ratio one where the group takes no trust ratio)
            if not plamb.trust[table.run_group[r]]:
                lamb_unit_ratio(ex, lamb.ratio + table.run_t0[r], table.run_nt[r])
    else:
        lamb_unit_ratio(ex, lamb.ratio, lamb.nt)
    for r in range(table.runs()):  # driver-loop: over runs of one group's tensors (a few a model)
        var lo = table.run_lo[r]
        lamb_apply(ex, P + lo, lamb.U + lo, lamb.ratio + table.run_t0[r], plamb.TABS + plamb.tab_at[r],
                   table.run_nt[r], table.run_hi[r] - lo, _rate[E](table, lrs, table.run_group[r]))
