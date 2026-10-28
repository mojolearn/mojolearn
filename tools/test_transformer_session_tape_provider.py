"""Budget-aware session-ticket source contracts; no binary/runtime qualification."""
from pathlib import Path
from check_generation_native_boundary import audit

ROOT = Path(__file__).resolve().parents[1]
GPU = (ROOT / "transformer/session_tape_dispatch.mojo").read_text()
CPU = (ROOT / "transformer/host/session_tape_dispatch.mojo").read_text()
PROVIDER = (ROOT / "providers/python/transformer_session_tape_pyapi.mojo").read_text()
PUBLIC = (ROOT / "bindings/transformer_public_native.mojo").read_text()


def test_both_original_session_numerical_closures_stay_native():
    for module in ("transformer.session_tape_dispatch", "transformer.host.session_tape_dispatch"):
        _, failures = audit(ROOT, (module,))
        assert not failures, failures
    for source in (GPU, CPU):
        assert "_session_forward_tape_run(words[0], inputs, p)" in source
        assert "_session_backward_tape_run(words[0], a, generation)" in source
        assert "for i in range(1, 11):" in source and "p.append(words[i])" in source
        assert "_transformer_run_session[True, True]" not in source
        assert "_transformer_saved_backward" not in source


def test_ticket_reports_keep_original_cookie_epoch_retention_and_stage_bytes():
    for source, epoch in ((GPU, "tape_generation"), (CPU, "generation")):
        assert "len(a) != 12 or len(words) != 11 or a[11] == 0" in source
        assert "report[0] = words[0]" in source
        assert f"report[1] = owner[].{epoch}" in source
        assert "report[2] = Int(owner[].tape.value().retained)" in source
        assert "report[3] = owner[].tape.value().retained_bytes" in source
        assert "p[8] < 0 or p[9] < 0" in source
    assert "var report = List[Int](length=4, fill=0)" in PROVIDER
    assert "with_report.append(Int(report.unsafe_ptr()))" in PROVIDER
    assert PROVIDER.index("with_report.append") < PROVIDER.index("36, 37, 38,") < PROVIDER.index("return report^")


def test_failure_release_and_closed_or_stale_discard_preserve_existing_lifecycle():
    for source, epoch, cleanup in ((GPU, "tape_generation", "clear"), (CPU, "generation", "invalidate_tape")):
        assert "if owner[].busy:" in source and "owner[].busy = False" in source
        discard = source.split("    if op == 4:", 1)[1].split("    if owner[].closed:", 1)[0]
        assert "if words[1] != words[0]:" in discard
        assert f"if not owner[].closed and words[2] == owner[].{epoch}:" in discard
        assert "owner[].invalidate_tape()" in discard
        assert f"except problem:\n            owner[].{cleanup}()" in source
        assert f"generation != owner[].{epoch} or not owner[].tape" in source
    for operation in (38, 39, 40):
        assert f"36, 37, {operation}," in PROVIDER
    assert 'py_call(_transformer_native_entry(previous, "transformer_session_close"), session)' in PUBLIC
    assert "idempotent" in PUBLIC


def test_append_only_table_wiring_and_cpu_compile_probe_reach_numerical_dispatch():
    for path, helper, args in (
        ("transformer/native_block_dispatch.mojo", "transformer_session_tape_operation", "a, words"),
        ("transformer/host/native_block_dispatch.mojo", "transformer_host_session_tape_operation", "a, p"),
    ):
        source = (ROOT / path).read_text()
        assert "if op >= 36 and op < 41:" in source
        assert f"return {helper}(op - 36, {args})" in source
        assert "op >= 41" in source
        assert "TRANSFORMER_BLOCK_ABI_NAME" in source
    probe = (ROOT / "experiments/peer3_neural_tokenizer_20261027/compile_transformer_session_tape_host.mojo").read_text()
    assert "from transformer.host.native_block_dispatch import transformer_host_block_v1" in probe
    assert "return transformer_host_block_v1(op, aa, na, pp, np, ff, nf, result, context, sink)" in probe


def test_public_tickets_keep_provider_and_handle_without_old_numerical_python_doors():
    owner = PUBLIC.split("struct TransformerPublicTapeOwner", 1)[1].split("def _transformer_configuration", 1)[0]
    assert "var provider: PythonObject" in owner and "var handle: Int" in owner
    assert "var backward: PythonObject" not in owner and "var discard: PythonObject" not in owner
    for spelling in ('"transformer_session_forward_tape"', '"transformer_session_backward_tape"', '"transformer_session_discard_tape"'):
        assert spelling not in PUBLIC
    assert "String(TRANSFORMER_BLOCK_ABI_NAME), 41)" in PUBLIC
    assert "neural_descriptor_construct(provider," in PUBLIC
    backward = PUBLIC.split("def transformer_backward_from_tape", 1)[1].split("def transformer_tape_close", 1)[0]
    assert "transformer_session_tape_backward(owner[].provider, owner[].session, owner[].handle," in backward
    assert "neural_descriptor_handle(" not in backward
    close = PUBLIC.split("def transformer_tape_close", 1)[1].split("def transformer_public_call", 1)[0]
    assert "transformer_session_tape_discard(owner[].provider, owner[].session, owner[].handle," in close
    assert "neural_descriptor_handle(" not in close
