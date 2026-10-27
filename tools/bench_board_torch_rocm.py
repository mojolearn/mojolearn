"""Supported environment setup for Torch's bundled ROCm runtime assets."""
import json
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


_CALLER_OVERRIDE = "MOJOLEARN_SMOKE_HIPBLASLT_CALLER_OVERRIDE"


def capture_caller_override(environ=None):
    """Save the caller setting before optional libraries mutate the smoke environment."""
    env = os.environ if environ is None else environ
    env.setdefault(_CALLER_OVERRIDE, json.dumps(env.get("HIPBLASLT_TENSILE_LIBPATH")))


def torch_worker_environment(env):
    # pylibraft.__init__ sets a system ROCm path on import. The shared smoke
    # loads pylibraft for HipVS, but that library's default is not a user
    # override for a separate Torch worker with its own bundled runtime.
    if _CALLER_OVERRIDE in env:
        original = json.loads(env.pop(_CALLER_OVERRIDE))
        if original is None:
            env.pop("HIPBLASLT_TENSILE_LIBPATH", None)
        else:
            env["HIPBLASLT_TENSILE_LIBPATH"] = original
    return env
