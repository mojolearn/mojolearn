"""Shared policy for future pinned-runner submissions and metadata preflight.
Importing this module performs no builds, imports of native modules, or jobs.
"""
import re

POLICIES = {
    "tools/arima_k3_df_gpu_quality.py": "verified-k3-gpu-quality",
    "tools/scoped_gemm_quality.py": "verified-scoped-adapter-quality",
    "tools/resample_gpu_gather_pair.py": "verified-standalone-quality",
    "tools/arima_assoc_scan_oracle.py": "reference",
    "tools/arima_gaussian_scan_oracle.py": "reference",
    "tools/arima_k1_diagnostics.py": "reference",
    "tools/arima_k3_df_reference.py": "reference",
    "tools/mcd_saved_oracle.py": "reference",
    "tools/catalog_gemm_quality.py": "verified-standalone-quality",
    "tools/callpath_probe_pair.py": "verified-standalone-quality",
    "tools/shared_gemm_quality.py": "verified-standalone-quality",
    "tools/catalog_resident_quality.py": "verified-standalone-quality",
    "tools/shared_gemm_downstream_pair.py": "verified-downstream-quality",
    "tools/shared_gemm_scoped.py": "verified-scoped-caller",
    "tools/catalog_gemm_matrix_timing.py": "verified-matrix-timing",
    "tools/catalog_resident_timing.py": "verified-resident-timing",
}


def policy_for(source, script, args):
    if not isinstance(source,str) or not re.fullmatch(r"[0-9a-f]{40}",source):
        raise ValueError("harness source must be an exact lowercase commit SHA")
    if not isinstance(script,str) or script not in POLICIES:
        raise ValueError("script is not allowed by pinned-runner policy: "+str(script))
    if not isinstance(args,list) or any(not isinstance(a,str) or '\0' in a for a in args):
        raise ValueError("script args must be a list of strings without NUL")
    policy=POLICIES[script]
    if policy in ("verified-standalone-quality","verified-resident-timing"):
        if not args or args[0]!=source:
            raise ValueError("this probe must verify the exact harness source")
    elif policy in ("verified-downstream-quality","verified-matrix-timing","verified-scoped-caller"):
        if not args or not re.fullmatch(r"[0-9a-f]{40}",args[0]):
            raise ValueError("this helper requires an exact compiled source first")
    if policy == "verified-scoped-adapter-quality":
        if (len(args) != 3 or not re.fullmatch(r"[0-9a-f]{40}", args[0])
                or not re.fullmatch(r"[A-Za-z0-9_.-]+", args[1])
                or args[2] not in ("tall", "dense", "gram", "narrow", "gram-split", "pca", "all")):
            raise ValueError("scoped quality requires COMPILED_SOURCE TAG PROFILE")
    if policy == "verified-k3-gpu-quality":
        required = ["--baseline-report", "--binding", "--binding-sha256", "--source", "--output"]
        if len(args) != 10 or args[::2] != required or args[7] != source or not re.fullmatch(r"[0-9a-f]{64}", args[5]):
            raise ValueError("K3 requires exact source, binary hash, baseline and fresh output")
    return policy
