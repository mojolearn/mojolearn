# SPDX-License-Identifier: Apache-2.0
"""packaging/linux/hip_contract.py on small fake listings and ELFs (no GPU, no build, no disassembler needed)."""
from pathlib import Path
import struct
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import hip_contract as hc  # noqa: E402


def listing(name, *body):
    return "\n".join([f"0000000000001900 <{name}>:"] + [f"\t{b}    // 000000001B94: 7E084504" for b in body]) + "\n"


def test_approximate_ops_are_refused_by_kernel():
    # the 0.8.67-dev AMD defects, by shape
    asm = (listing("x_prep_device_p2m_kernel_SIM6A6A_819a", "v_mul_f32_e32 v1, v2, v3", "v_sqrt_f32_e32 v4, v1")
           + listing("gbdt_methods_kernel_pointwise6A6A_f604", "v_log_f32_e32 v4, v1", "v_exp_f32_e64 v5, v1"))
    assert hc.refused_sites(asm) == [("x_prep_device_p2m_kernel_SIM6A6A_819a", "v_sqrt_f32"),
                                     ("gbdt_methods_kernel_pointwise6A6A_f604", "v_log_f32"),
                                     ("gbdt_methods_kernel_pointwise6A6A_f604", "v_exp_f32")]
    for op in ("v_rsq_f32_e32 v1, v2", "v_sin_f32_e32 v1, v2", "v_cos_f32_e32 v1, v2", "v_sqrt_f64_e32 v[0:1], v[2:3]"):
        assert len(hc.refused_sites(listing("k", op))) == 1, op


def test_rcp_inside_a_division_expansion_is_exact():
    fdiv = listing("k", "v_div_scale_f32 v1, s[0:1], v2, v2, v3", "v_rcp_f32_e32 v4, v1",
                   "v_fma_f32 v5, -v1, v4, 1.0", "v_div_fmas_f32 v6, v5, v4, v1", "v_div_fixup_f32 v7, v6, v2, v3")
    intdiv64 = listing("k", "v_fmac_f32_e32 v4, 0x4f800000, v14", "v_rcp_f32_e32 v4, v4", "v_mul_lo_u32 v14, v13, s16")
    div24 = listing("k", "v_cvt_f32_i32_e32 v12, v10", "v_rcp_f32_e32 v13, 0x40a00000", "v_add_u32_e32 v12, 1, v12",
                    "v_mul_f32_e32 v12, v12, v13", "v_trunc_f32_e32 v12, v12", "v_cvt_i32_f32_e32 v12, v12")
    for asm in (fdiv, intdiv64, div24):
        assert hc.refused_sites(asm) == []
    bare = listing("k", "v_mul_f32_e32 v1, v2, v3", "v_rcp_f32_e32 v4, v1", "v_mul_f32_e32 v5, v4, v6")
    assert hc.refused_sites(bare) == [("k", "v_rcp_f32 (outside a division expansion)")]


def test_sabotage_kernels_share_the_ptx_allow_list(tmp_path, monkeypatch):
    blob = elf()
    so = tmp_path / "_mojolearn_native_trees.so"
    so.write_bytes(b"\0" * 64 + blob + b"\0" * 16 + blob)
    asm = (listing("kernel_methods_checks_km_sabot6A6A6A6A_bfa0", "v_exp_f32_e32 v1, v2")
           + listing("gbdt_methods_pointwise_kernels6A6A6A6A_f4d1", "v_log_f32_e32 v1, v2"))
    monkeypatch.setattr(hc, "_disassemble", lambda job: (hc.refused_sites(asm), ""))
    r = hc.audit_file(so, "llvm-objdump")
    assert r["code_objects"] == 2 and r["errors"] == []
    assert r["refused"] == ["gbdt_methods_pointwise_kernels6A6A6A6A_f4d1: v_log_f32"]
    assert list(r["allowed"]) == ["kernel_methods_checks_km_sabot6A6A6A6A_bfa0"]


def test_a_binary_without_code_objects_is_an_error(tmp_path):
    so = tmp_path / "x.so"
    so.write_bytes(b"\x7fELF" + b"\2" + b"\0" * 59)
    assert hc.audit_file(so, "llvm-objdump")["errors"] == ["no AMDGPU code object embedded"]


def elf(payload=b"\x00" * 32):
    """A minimal 64-bit AMDGPU ELF: header, one PROGBITS section holding `payload`, section table after it."""
    shoff = 64 + len(payload)
    head = bytearray(64)
    head[0:4] = b"\x7fELF"
    head[4] = 2
    struct.pack_into("<H", head, 18, hc.EM_AMDGPU)
    struct.pack_into("<Q", head, 40, shoff)
    struct.pack_into("<HH", head, 58, 64, 2)
    null = bytes(64)
    sec = bytearray(64)
    struct.pack_into("<I", sec, 4, 1)
    struct.pack_into("<QQ", sec, 24, 64, len(payload))
    return bytes(head) + payload + null + bytes(sec)


def test_code_objects_are_cut_at_their_own_extent():
    a, b = elf(b"\x01" * 40), elf(b"\x02" * 8)
    data = b"host code" + a + b"\0\0" + b + b"tail"
    assert hc.code_objects(data) == [a, b]
    # an x86-64 ELF (the host .so itself) is not a code object
    x86 = bytearray(a)
    struct.pack_into("<H", x86, 18, 62)
    assert hc.code_objects(bytes(x86)) == []
