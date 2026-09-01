# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Entry point for the SVC gates and the card.

    pixi run mojo run -I . svm/svc_main.mojo -- oracle        # host only, no lock needed
    pixi run mojo run -I . svm/svc_main.mojo -- svr-oracle    # host, regression only
    tools/with_build_lock.sh     pixi run mojo run -I . svm/svc_main.mojo
    tools/with_identical_mode.sh pixi run mojo run -I . svm/svc_main.mojo
    MOJOLEARN_IDENTITY_TRACE=/tmp/svm.card \\
        tools/with_identical_mode.sh pixi run mojo run -I . svm/svc_main.mojo -- card

Every printed line carries the mode the binary COMPILED in. `oracle` runs
the host gates only (no GPU), CLASSIFICATION AND REGRESSION; `svr-oracle`
runs the regression half of that on its own, which is the cheap loop while
the SVR lane is being worked; the default runs everything; `card` fits F2
once with the stage recorder pointed at `MOJOLEARN_IDENTITY_TRACE` so a
cross-vendor leg can diff stage by stage with
`tools/identity_trace_diff.py`. That leg ran at `a0a0eee` on 2026-08-28 and
the 32-stage card is byte-identical on Apple M4, NVIDIA H100 and AMD MI325X.

THE REGRESSION GATES
--------------------
`SmoSolver.solve` no longer raises for EPSILON_SVR. The refusal came out on
2026-08-31 (`fea6becc`) and the regression gates are what removed it. The
four HOST gates below (the eps-insensitive dual's monotone descent, the KKT
gap off an independently recomputed gradient, the eps tube, and the two trap
measurements) do not touch the device and are the ones written from the
FORMULATION rather than from this solver. The two DEVICE regression gates run
beside them. Both device gates keep a BLOCKED branch that FAILS rather than
passing quietly, so that putting the refusal back can never turn a gate green
on a comparison that did not happen.

`-- svr-oracle` runs the host half on its own, which is the cheap loop; the
default run runs everything.
"""

from std.sys import argv
from std.os import getenv
from max.gpu.host import DeviceContext

from core.identity_trace import IdentityTrace
from svm.checks.svc_check import (
    _mode_name,
    _run_device,
    all_fixtures,
    all_reg_fixtures,
    check_block_solve_signed_zero_tie,
    check_card_is_emitted,
    check_device_is_launch_invariant,
    check_device_matches_oracle,
    check_nan_never_recorded,
    check_oracle_f32_matches_f64_reference,
    check_oracle_kkt_and_accuracy,
    check_oracle_objective_decreases,
    check_rbf_float_vs_double_reference,
    check_refusals,
    check_svr_device_is_launch_invariant,
    check_svr_device_matches_oracle,
    check_svr_eps_tube,
    check_svr_fold_order,
    check_svr_kkt,
    check_svr_objective_decreases,
    check_svr_sabotage_reach,
    check_svr_signed_zero,
    check_ws_sequence_is_pure_in_f_and_index,
    fixture_blobs,
    fixture_xor,
    svr_oracle_fit,
)


def _gate(name: String, mut ran: Int, mut failed: Int, e: String):
    ran += 1
    if e == "":
        print("PASS " + name + " [" + _mode_name() + "]")
    else:
        failed += 1
        print("FAIL " + name + " [" + _mode_name() + "]: " + e)


def run_oracle_gates(mut ran: Int, mut failed: Int) raises:
    var fixtures = all_fixtures()
    for i in range(len(fixtures)):
        var e = String("")
        try:
            check_oracle_kkt_and_accuracy(fixtures[i])
        except err:
            e = String(err)
        _gate("oracle_kkt_and_accuracy " + fixtures[i].name, ran, failed, e)
    for i in range(2):
        var e = String("")
        try:
            check_oracle_objective_decreases(fixtures[i])
        except err:
            e = String(err)
        _gate("oracle_objective_decreases " + fixtures[i].name, ran, failed, e)
    var e1 = String("")
    try:
        check_oracle_f32_matches_f64_reference(fixtures[0])
    except err:
        e1 = String(err)
    _gate("oracle_f32_matches_f64_reference F1.blobs", ran, failed, e1)
    var e2 = String("")
    try:
        check_rbf_float_vs_double_reference()
    except err:
        e2 = String(err)
    _gate("rbf_float_vs_double_reference (DEVIATION 630)", ran, failed, e2)


def run_svr_oracle_gates(mut ran: Int, mut failed: Int) raises:
    """The EPSILON_SVR host gates. ONE oracle solve per fixture feeds all of
    that fixture's gates; `smo_oracle_fit` is O(n_ws^2 k) per outer iteration
    on one host thread, so R5.big and R6.zero are the expensive rows and
    both are capped at 8 and 6 outer iterations respectively.

    The order is deliberate: the KKT gate first, because `max|f_ref - f|` is
    the single number that fires loudest on a wrong `SvrInit`, and reading a
    failure there first saves reading three more."""
    var fixtures = all_reg_fixtures()
    for i in range(len(fixtures)):
        var name = String(fixtures[i].name)
        var res = svr_oracle_fit(fixtures[i])
        var e = String("")
        try:
            check_svr_kkt(fixtures[i], res)
        except err:
            e = String(err)
        _gate("svr_kkt " + name, ran, failed, e)
        if fixtures[i].do_objective:
            e = ""
            try:
                check_svr_objective_decreases(fixtures[i], res)
            except err:
                e = String(err)
            _gate("svr_objective_decreases " + name, ran, failed, e)
        if fixtures[i].do_tube:
            e = ""
            try:
                check_svr_eps_tube(fixtures[i], res)
            except err:
                e = String(err)
            _gate("svr_eps_tube " + name, ran, failed, e)
        e = ""
        try:
            check_svr_fold_order(fixtures[i], res)
        except err:
            e = String(err)
        _gate("svr_fold_order_collisions " + name, ran, failed, e)
        e = ""
        try:
            check_svr_sabotage_reach(fixtures[i], res)
        except err:
            e = String(err)
        _gate("svr_sabotage_reach " + name, ran, failed, e)
        if name == "R6.zero":
            e = ""
            try:
                check_svr_signed_zero(fixtures[i], res)
            except err:
                e = String(err)
            _gate("svr_signed_zero_from_ordinary_input " + name, ran, failed, e)
        _ = res^


def run_svr_device_gates(mut ran: Int, mut failed: Int) raises:
    """The two device regression gates. BOTH FAIL, naming BLOCKED, while
    `SmoSolver.solve`'s UNGATED clause is in the tree."""
    var ctx = DeviceContext()
    var e = String("")
    try:
        check_svr_device_matches_oracle(ctx, all_reg_fixtures())
    except err:
        e = String(err)
    _gate("svr_device_matches_oracle (6 regression fixtures)", ran, failed, e)
    e = ""
    try:
        check_svr_device_is_launch_invariant(ctx)
    except err:
        e = String(err)
    _gate("svr_device_is_launch_invariant (R4, R6 x 5 arms)", ran, failed, e)


def run_device_gates(mut ran: Int, mut failed: Int) raises:
    var ctx = DeviceContext()
    var e = String("")
    try:
        check_refusals(ctx)
    except err:
        e = String(err)
    _gate("refusals_by_name", ran, failed, e)
    e = ""
    try:
        check_device_matches_oracle(ctx, all_fixtures())
    except err:
        e = String(err)
    _gate("device_matches_oracle (7 fixtures)", ran, failed, e)
    e = ""
    try:
        check_ws_sequence_is_pure_in_f_and_index(ctx)
    except err:
        e = String(err)
    _gate("ws_sequence_is_pure_in_f_and_index (DEVIATION 631, F3.dup)", ran, failed, e)
    e = ""
    try:
        check_device_is_launch_invariant(ctx)
    except err:
        e = String(err)
    _gate("device_is_launch_invariant (F2, F6, F7 x 7 arms)", ran, failed, e)
    e = ""
    try:
        check_card_is_emitted(ctx, "/tmp/svm_card_a.trace", "/tmp/svm_card_b.trace")
    except err:
        e = String(err)
    _gate("card_is_emitted", ran, failed, e)
    e = ""
    try:
        check_block_solve_signed_zero_tie(ctx)
    except err:
        e = String(err)
    _gate("block_solve_signed_zero_tie (row 39, DEVIATIONS 633/635, Z x 2 orders x 2 blocks)", ran, failed, e)
    e = ""
    try:
        check_nan_never_recorded(ctx)
    except err:
        e = String(err)
    _gate("nan_never_recorded (row 39 FACT 2, DEVIATIONS 636/637)", ran, failed, e)


def emit_card() raises:
    var path = String(getenv("MOJOLEARN_IDENTITY_TRACE"))
    if path == "":
        raise Error("card: set MOJOLEARN_IDENTITY_TRACE=<path>")
    var ctx = DeviceContext()
    var card = IdentityTrace.to_path(path)
    card.header("svm card [" + _mode_name() + "] fixture F2.xor rbf gamma=0.5 C=10 n=240 k=2")
    var fx = fixture_xor(10.0, "F2.xor")
    var run = _run_device(ctx, fx, 0, 1 << 30, 200.0, 0, 0.0, card)
    print("card written to " + path + " [" + _mode_name() + "] n_support=" + String(run.model.n_support))


def main() raises:
    var what = String("all")
    var args = argv()
    if len(args) > 1:
        what = String(args[len(args) - 1])
    print("== svm/svc_main.mojo [" + _mode_name() + "] " + what + " ==")
    print(
        "C-SVC certified Apple M4, NVIDIA H100 and AMD MI325X at a0a0eee"
        " (E3 round 13, 2026-08-28); SVR gated 2026-08-31 (fea6becc, 44 of 44)"
        " and not yet in a cross-vendor round."
    )
    var ran = 0
    var failed = 0
    if what == "card":
        emit_card()
        return
    if what == "svr-oracle":
        run_svr_oracle_gates(ran, failed)
    else:
        run_oracle_gates(ran, failed)
        run_svr_oracle_gates(ran, failed)
        if what != "oracle":
            run_device_gates(ran, failed)
            run_svr_device_gates(ran, failed)
    print(
        "== svm/svc_main.mojo [" + _mode_name() + "] " + String(ran - failed)
        + "/" + String(ran) + " gates passed =="
    )
    if failed > 0:
        raise Error(String(failed) + " gate(s) FAILED")
