"""Only lanes that require a training stack need a CPU-only installation."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import identity_break as ib  # noqa: E402

SAMBA = {"samba", "samba-bf16w", "samba-int8w", "samba-untied-dropout-accum", "samba-decode-session",
         "samba-optimizer-state", "nn-ep-samba-stack"}


def test_only_training_stack_lanes_need_a_cpu_only_installation():
    assert set(ib.GPU_INSTALL_CPU_NA) == SAMBA
    assert SAMBA <= set(ib.LANES)


def test_na_only_on_the_cpu_side_of_a_gpu_install(monkeypatch):
    monkeypatch.setattr(ib, "gpu_side_available", lambda ml: True)
    assert ib.gpu_install_na("samba", None, "cpu").startswith("n/a:gpu-install: ")
    assert ib.gpu_install_na("nn-ep-samba-inference", None, "cpu") is None
    assert ib.one_sided_na("nn-ep-samba-inference", "cpu") is None
    assert ib.one_sided_na("nn-ep-samba-inference", "gpu")
    assert ib.gpu_install_na("samba", None, "gpu") is None
    assert ib.gpu_install_na("transformer", None, "cpu") is None
    monkeypatch.setattr(ib, "gpu_side_available", lambda ml: False)
    assert ib.gpu_install_na("samba", None, "cpu") is None
