"""GPU_INSTALL_CPU_NA (lane gs-fix row G1, decided 2026-11-25): the eight
SambaStack lanes read n/a on the CPU side of a GPU installation only."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import identity_break as ib  # noqa: E402

SAMBA = {"samba", "samba-bf16w", "samba-int8w", "samba-untied-dropout-accum", "samba-decode-session",
         "samba-optimizer-state", "nn-ep-samba-stack", "nn-ep-samba-inference"}


def test_the_declared_lanes_are_the_eight_samba_lanes():
    assert set(ib.GPU_INSTALL_CPU_NA) == SAMBA
    assert SAMBA <= set(ib.LANES)


def test_na_only_on_the_cpu_side_of_a_gpu_install(monkeypatch):
    monkeypatch.setattr(ib, "gpu_side_available", lambda ml: True)
    assert ib.gpu_install_na("samba", None, "cpu").startswith("n/a:gpu-install: ")
    assert ib.gpu_install_na("samba", None, "gpu") is None
    assert ib.gpu_install_na("transformer", None, "cpu") is None
    monkeypatch.setattr(ib, "gpu_side_available", lambda ml: False)
    assert ib.gpu_install_na("samba", None, "cpu") is None
