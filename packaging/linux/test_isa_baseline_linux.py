# SPDX-License-Identifier: Apache-2.0
"""The CPU ISA baseline gate reads opcodes, never symbol names (2026-10-24)."""
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent))
import isa_baseline_linux as isa  # noqa: E402

KNOTS = ('  aa5bc9:\tcall   f21560 <std::python::python_object::PythonObject::__init__'
         '[!kgen.string](::StringLiteral[$0])_REMOVED_ARG,value.value`2x="knots">')


def _scan_with(monkeypatch, listing):
    monkeypatch.setattr(isa.subprocess, "run",
                        lambda *a, **k: SimpleNamespace(returncode=0, stdout=listing, stderr=""))
    return isa.scan(Path("fake.so"))


def test_a_symbol_named_like_an_opmask_instruction_is_not_a_finding(monkeypatch):
    listing = "0000000000aa5b00 <bindings::prep_encode_native::_spline_fit>:\n" + KNOTS + "\n  aa5bce:\tret\n"
    findings, ninsn, ncpuid = _scan_with(monkeypatch, listing)
    assert findings == {} and ninsn == 2 and ncpuid == 0


def test_a_real_opmask_instruction_is_still_refused(monkeypatch):
    listing = "0000000000aa5b00 <f>:\n  aa5b00:\tkmovw  %k1,%eax\n  aa5b04:\tvpternlogd $0x96,%zmm2,%zmm1,%zmm0\n"
    findings, ninsn, _ = _scan_with(monkeypatch, listing)
    assert ninsn == 2
    assert "AVX-512 opcode" in findings and findings["AVX-512 opcode"]["count"] == 2
    assert "zmm register" in findings and "opmask register k0-k7" in findings


def test_an_approximate_reciprocal_estimate_is_refused_and_exact_ops_are_not(monkeypatch):
    listing = ("0000000000aa5b00 <f>:\n  aa5b00:\tvrcpss %xmm1,%xmm1,%xmm0\n  aa5b04:\trsqrtps %xmm0,%xmm1\n"
               "  aa5b08:\tvsqrtss %xmm0,%xmm0,%xmm0\n  aa5b0c:\tvdivss %xmm1,%xmm0,%xmm0\n")
    findings, ninsn, _ = _scan_with(monkeypatch, listing)
    assert ninsn == 4
    assert list(findings) == ["approximate reciprocal estimate (CPU-model-dependent bits)"]
    assert findings["approximate reciprocal estimate (CPU-model-dependent bits)"]["count"] == 2


def test_the_scan_is_parallel_by_cores_and_memory_and_reports_in_file_order(tmp_path, monkeypatch, capsys):
    # lane build-reuse-1, 2026-11-26: one process per binary up to the cores and the memory a disassembly needs
    sos = [tmp_path / f"b{i}.so" for i in range(3)]
    for p in sos:
        p.write_bytes(b"\0" * 10)
    assert isa.workers_for(sos, {"ISA_BASELINE_WORKERS": "2"}) == 2
    assert 1 <= isa.workers_for(sos, {}) <= 3
    listing = "0000000000000000 <f>:\n  0:\tret\n"
    monkeypatch.setattr(isa.subprocess, "run",
                        lambda *a, **k: SimpleNamespace(returncode=0, stdout=listing, stderr=""))
    monkeypatch.setattr(isa.shutil, "which", lambda name: "/usr/bin/objdump")
    monkeypatch.setenv("ISA_BASELINE_WORKERS", "1")
    monkeypatch.setattr(sys, "argv", ["isa_baseline_linux.py", str(tmp_path)])
    assert isa.main() == 0
    out = capsys.readouterr().out
    assert [l.split()[1] for l in out.splitlines() if l.startswith("  ok")] == ["b0.so", "b1.so", "b2.so"]
