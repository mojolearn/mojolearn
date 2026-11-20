# SPDX-License-Identifier: Apache-2.0
"""packaging/macos/air_contract.py on small AIR listings (no GPU, no build, no llvm-dis needed)."""
from pathlib import Path
import struct
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import air_contract as ac  # noqa: E402

OPTS = ('!air.compile_options = !{!1, !2}\n!1 = !{!"air.compile.denorms_disable"}\n'
        '!2 = !{!"air.compile.fast_math_disable"}\n')


def module(name, *body, opts=OPTS):
    return (f"define void @{name}(ptr addrspace(1) %0) {{\n" + "".join(f"  {b}\n" for b in body)
            + "  ret void\n}\n" + opts)


def test_clean_identical_module_passes():
    ll = module("x_prep_device_p2m_kernel_SIM6A_819a", "%3 = fmul contract float %1, %2",
                "%4 = fadd contract float %3, %1", "%5 = call float @air.sqrt.f32(float %4)",
                "%6 = call float @air.fmax.f32(float %5, float %4)")
    assert ac.audit_text(ll) == ([], ["air.compile.denorms_disable", "air.compile.fast_math_disable"])


def test_fast_math_and_approximate_intrinsics_are_refused():
    ll = module("k", "%3 = fdiv arcp contract float %1, %2", "%4 = call fast float @air.sqrt.f32(float %3)",
                "%5 = call float @air.exp2.f32(float %4)", "%6 = call float @air.fast_sqrt.f32(float %5)",
                "%7 = call float @air.log2.f32(float %6)", "%8 = call float @air.tanh.f32(float %7)")
    refused, _ = ac.audit_text(ll)
    assert refused == [("k", "fast-math flag arcp"), ("k", "fast-math flag fast"), ("k", "air.exp2.f32"),
                       ("k", "air.fast_sqrt.f32"), ("k", "air.log2.f32"), ("k", "air.tanh.f32")]


def test_module_without_fast_math_disable_is_refused():
    refused, _ = ac.audit_text(module("k", opts='!air.compile_options = !{!1}\n!1 = !{!"air.compile.denorms_disable"}\n'))
    assert refused == [("k", "module lacks air.compile.fast_math_disable")]


def test_sabotage_kernels_are_allowed_and_modules_are_found(tmp_path, monkeypatch):
    def wrapped(n):
        body = b"BC\xc0\xde" + bytes([n]) * 12
        return struct.pack("<IIIII", 0x0B17C0DE, 0, 20, len(body), 0) + body
    so = tmp_path / "_mojolearn_native_trees.so"
    so.write_bytes(b"\0" * 32 + wrapped(1) + b"pad" + wrapped(2))
    assert len(ac.modules(so.read_bytes())) == 2
    lls = iter([module("kernel_methods_checks_km_sabot6A_1c77", "%3 = call float @air.cos.f32(float %1)"),
                module("gbdt_methods_kernel_pointwise6A_f604", "%3 = call float @air.log2.f32(float %1)")])
    monkeypatch.setattr(ac, "_disassemble", lambda job: (ac.audit_text(next(lls)), ""))
    r = ac.audit_file(so, "llvm-dis")
    assert r["air_modules"] == 2 and r["errors"] == []
    assert r["refused"] == ["gbdt_methods_kernel_pointwise6A_f604: air.log2.f32"]
    assert list(r["allowed"]) == ["kernel_methods_checks_km_sabot6A_1c77"]
