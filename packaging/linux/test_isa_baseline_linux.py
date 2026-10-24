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
