"""Supported environment setup for Torch's bundled ROCm runtime assets."""
import os
from pathlib import Path


def configure_hipblaslt(torch, environ=None):
    """Locate Tensile assets from the actual worker's Torch, preserving overrides."""
    env = os.environ if environ is None else environ
    key = "HIPBLASLT_TENSILE_LIBPATH"
    if key in env or not getattr(torch.version, "hip", None):
        return
    # FAIRNESS FIX (2026-10-13 audit): AMD smoke BF16 workers loaded Torch's
    # bundled libhipblaslt.so but searched the system ROCm 6.4 asset directory,
    # then segfaulted before ready. Use hipBLASLt's documented relocation variable
    # with the assets accompanying that Torch runtime; do not change its backend.
    # Reproduction and passing five-arm GEMM probe: 2026-10-27, retained under
    # ~/mojolearn-evidence/amd-loader-diagnosis-20261027/. The cause of the
    # campaign/direct-worker search-path difference remains unconfirmed.
    # https://rocm.docs.amd.com/projects/hipBLASLt/en/docs-7.0.2/conceptual/hipblaslt-clients.html
    lib = Path(torch.__file__).resolve().parent / "lib"
    assets = lib / "hipblaslt" / "library"
    if (lib / "libhipblaslt.so").is_file() and assets.is_dir():
        env[key] = str(assets)
