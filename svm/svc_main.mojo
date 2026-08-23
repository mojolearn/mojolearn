"""Entry point for the SVC gates and the card.

    pixi run mojo run -I . svm/svc_main.mojo -- oracle        # host only, no lock needed
    tools/with_build_lock.sh     pixi run mojo run -I . svm/svc_main.mojo
    tools/with_identical_mode.sh pixi run mojo run -I . svm/svc_main.mojo
    MOJOLEARN_IDENTITY_TRACE=/tmp/svm.card \\
        tools/with_identical_mode.sh pixi run mojo run -I . svm/svc_main.mojo -- card

Every printed line carries the mode the binary COMPILED in. `oracle` runs
the host gates only (no GPU); the default runs everything; `card` fits F2
once with the stage recorder pointed at `MOJOLEARN_IDENTITY_TRACE` so a
future cross-vendor leg can diff stage by stage with
`tools/identity_trace_diff.py`.
"""

from std.sys import argv
from std.os import getenv
from max.gpu.host import DeviceContext

from core.identity_trace import IdentityTrace
from svm.mojo_only.svc_check import (
    _mode_name,
    _run_device,
    all_fixtures,
    check_card_is_emitted,
    check_device_is_launch_invariant,
    check_device_matches_oracle,
    check_oracle_f32_matches_f64_reference,
    check_oracle_kkt_and_accuracy,
    check_oracle_objective_decreases,
    check_rbf_float_vs_double_reference,
    check_refusals,
    check_ws_sequence_is_pure_in_f_and_index,
    fixture_blobs,
    fixture_xor,
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
    print("CONSTRUCTION plus one Apple device's gates; no second vendor has run this.")
    var ran = 0
    var failed = 0
    if what == "card":
        emit_card()
        return
    run_oracle_gates(ran, failed)
    if what != "oracle":
        run_device_gates(ran, failed)
    print(
        "== svm/svc_main.mojo [" + _mode_name() + "] " + String(ran - failed)
        + "/" + String(ran) + " gates passed =="
    )
    if failed > 0:
        raise Error(String(failed) + " gate(s) FAILED")
