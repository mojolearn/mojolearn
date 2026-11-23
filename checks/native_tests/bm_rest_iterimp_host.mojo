# SPDX-License-Identifier: Apache-2.0
"""Host-column driver of the IterativeImputer RUN for checks/native_tests/prep_iterative_imputer.mojo (lane
bm-rest-iterimp, 2026-11-23). Importable; no `main()`, no `test_*`.

`HostIterRunner` runs the RUN's x_prep programs (x_prep/iterimp_run.mojo) over the host x_prep transport
(x_prep/transport_host.mojo `x_prep_transport_host_v1`, the route `py2mojo_host_members.host_route` names: no route
flags, so no range runner, device scratch or output region): owner 0 is the caller's X, owner k >= 1 the address
`seconds[k - 1]`, -1 the program's own words, each copied into a zeroed host arena the runner keeps until
`release`. `ii_request` writes the request words the door's UNPACK writes for the public keywords on the host
identical image; `ii_output` and `ii_transform` read what PACK reads. Nothing here computes a statistic.
"""
from std.memory import bitcast, unsafe_memcpy
from core.entry_table import EntryRef
from core.member_abi import MemberPlan, MemberCall, member_call
from bindings.x_prep_native_owner_run import XprepRunInput, XprepRoute, xprep_execute, xprep_release, xprep_r3
from bindings.prep_encode_prog_run import EncProgram
from bindings.prep_encode_imputer_run import ImputerFit, ImputerApply
from x_prep.iterimp_run import (
    IterImpRunner, IterImpFit, IterImpFitted, IterImpApply, IterImpStep, iterimp_transform, II_NONE,
)
from checks.native_tests.py2mojo_host_members import host_route, host_idn_fam

comptime PREP_ITERIMP_DIR = "checks/native_tests/fixtures/prep_iterimp/"
comptime _F32P = MutPointer[Float32, MutUntrackedOrigin]


def pos_inf() -> Float64:
    return bitcast[DType.float64](UInt64(0x7FF0000000000000))


def neg_inf() -> Float64:
    return bitcast[DType.float64](UInt64(0xFFF0000000000000))


struct HostIterRunner(IterImpRunner):
    var key: EntryRef
    var x: Int
    var arenas: List[List[Float32]]

    def __init__(out self, x: Int):
        self.key = host_route().second
        self.x = x
        self.arenas = List[List[Float32]]()

    def run(mut self, mut prog: EncProgram, seconds: List[Int]) raises -> Int:
        prog.check_bound()
        if prog.out_words > 0:
            raise Error("RuntimeError: the host column has no x_prep output region")
        var inputs = List[XprepRunInput](capacity=len(prog.inputs))
        for i in range(len(prog.inputs)):  # driver-loop: over the program's input records
            var record = prog.inputs[i]
            var address = record.address
            if record.owner == 0:
                address = self.x
            elif record.owner >= 1:
                address = seconds[record.owner - 1]
            inputs.append(XprepRunInput(record.offset, record.words, address, -1, False, 0, False, 0, 0, 0))
        var arena = List[Float32](length=max(prog.host + prog.scratch_size, 1), fill=0.0)
        var owned = List[Int](capacity=2 * len(inputs) + 1)
        try:
            xprep_execute(self.key, Int(arena.unsafe_ptr()), prog.host, prog.scratch_size, 0, 0, prog.stage_words(),
                          inputs, prog.inout_ranges, XprepRoute(False, False, False), owned)
        except error:
            _ = xprep_release(self.key, owned)
            raise error^
        var cleanup = xprep_release(self.key, owned)
        if cleanup.byte_length() > 0:
            raise Error("RuntimeError: x_prep slot cleanup failed: " + cleanup)
        self.arenas.append(arena^)
        return len(self.arenas) - 1

    def words(self, handle: Int) -> Int:
        return Int(self.arenas[handle].unsafe_ptr())

    def release(mut self, handle: Int):
        self.arenas[handle] = List[Float32]()

    def member(mut self, plan: MemberPlan, operation: Int, mut call: MemberCall) raises -> Int:
        # A check's process holds no interpreter: the member call as it is.
        return member_call(plan.member, operation, call)


def ii_request(n: Int, d: Int, strategy: Int, order: Int, max_iter: Int, nnf: Int, rng: UInt64, lo: Float64,
               hi: Float64, posterior: Bool, add_indicator: Bool) -> IterImpFit:
    """The door's `_fit_request` for IterativeImputer(missing_values=nan, initial_strategy, imputation_order,
    max_iter, n_nearest_features (II_NONE: None), random_state (None: 0), min_value, max_value, sample_posterior,
    add_indicator), the other keywords at their defaults (tol 1e-3, skip_complete False), on the host identical
    image: blocked statistics from its family word, no radix median (the fast Metal route), no output region."""
    var blocked = (host_idn_fam() & 1) != 0
    var init = ImputerFit(n, d, 0, False, 0.0, strategy, False, blocked, 0.0, False)
    var los = List[Float64](length=d, fill=lo)
    var his = List[Float64](length=d, fill=hi)
    return IterImpFit(init^, n, d, False, 0.0, blocked, xprep_r3("work"), False, order, False, posterior,
                      add_indicator, max_iter, nnf, 1e-3, rng, los^, his^)


def ii_output(runner: HostIterRunner, fitted: IterImpFitted, n: Int) -> List[Float32]:
    """fit_transform's (n, cols) result, as PACK reads it."""
    var count = n * fitted.cols
    var values = List[Float32](length=count, fill=0.0)
    if count > 0:
        unsafe_memcpy(dest=_F32P(unsafe_from_address=Int(values.unsafe_ptr())),
                      src=_F32P(unsafe_from_address=runner.words(fitted.out_handle)) + fitted.out_offset, count=count)
    return values^


def ii_transform(fit_runner: HostIterRunner, fitted: IterImpFitted, r: IterImpFit, xh: List[Float32],
                 n: Int) raises -> List[Float32]:
    """transform(Xh) of a fit, with the fitted fields UNPACK reads (`_steps`): each step's vectors where the fit's
    rounds arena holds them, the initial imputer's fill vector."""
    var dk = len(fitted.keep)
    var rounds = fit_runner.words(fitted.rounds)
    var steps = List[IterImpStep]()
    for s in range(len(fitted.seq_j)):  # driver-loop: over the fitted imputation steps
        var has = False
        var nbl = List[Int]()
        var sig = 0
        var sig_words = 0
        var al = 0
        var means = 0
        if r.posterior:
            has = fitted.post_has_nbl[s]
            nbl = fitted.post_nbl[s].copy()
            sig_words = fitted.post_side[s] * fitted.post_side[s]
            sig = rounds + 4 * fitted.post_sig[s]
            al = rounds + 4 * fitted.post_al[s]
            means = rounds + 4 * fitted.post_means[s]
        steps.append(IterImpStep(fitted.seq_j[s], rounds + 4 * fitted.seq_coef[s], rounds + 4 * fitted.seq_inter[s],
                                 has, nbl^, sig, sig_words, al, 2 if r.posterior else 0, means,
                                 dk if r.posterior else 0))
    var init = ImputerApply(n, r.d, 0, r.init.mark, r.init.missing, 1, len(fitted.fill), fitted.keep.copy(),
                            List[Int]())
    var request = IterImpApply(init^, Int(fitted.fill.unsafe_ptr()), n, r.d, r.mark, r.missing, r.work_on, False,
                               fitted.keep.copy(), fitted.bounds.copy(), fitted.indicator.copy(), r.posterior,
                               Int(fitted.rng & 0x7FFFFFFF), fitted.post_step, steps^)
    var runner = HostIterRunner(Int(xh.unsafe_ptr()))
    var applied = iterimp_transform(runner, request)
    var count = n * applied.cols
    var values = List[Float32](length=count, fill=0.0)
    if count > 0:
        unsafe_memcpy(dest=_F32P(unsafe_from_address=Int(values.unsafe_ptr())),
                      src=_F32P(unsafe_from_address=runner.words(applied.out_handle)) + applied.out_offset,
                      count=count)
    _ = len(xh)  # X and the fitted vectors the programs read stay alive past them
    _ = len(fitted.fill)
    return values^
