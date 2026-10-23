#!/usr/bin/env python3
"""Build validated RunPod requests without renting or accessing credentials.

NVIDIA development pods use v2's open-ended CUDA floor. The legacy v1 builder
remains available for the unchanged AMD route and retained request fixtures.
https://github.com/runpod/runpod-plugins-official/blob/main/plugins/runpod/skills/runpod-usage/reference/gpu-selection.md
"""
import json
import os
from pathlib import Path
import re
import sys

CUDA_FLOOR = (13, 0)
OVERRIDE = "MOJOLEARN_DEVPOD_ALLOWED_CUDA_VERSIONS"


def cuda_version(value):
    if not isinstance(value, str) or not re.fullmatch(r"(?:0|[1-9][0-9]*)\.(?:0|[1-9][0-9]*)", value):
        raise ValueError(f"invalid CUDA version {value!r}; expected major.minor")
    return tuple(int(part) for part in value.split("."))


def allowed_cuda_versions(gpu_ids, catalog=None, override=None):
    """Return exact-match versions >=13.0; availability never lowers the floor."""
    versions = set()
    if override is not None:
        values = [part.strip() for part in override.split(",")]
        if not values or any(not value for value in values):
            raise ValueError(f"{OVERRIDE} must be a nonempty comma-separated version list")
        for value in values:
            parsed = cuda_version(value)
            if parsed < CUDA_FLOOR:
                raise ValueError(f"{OVERRIDE} cannot allow CUDA {value}; pinned MAX requires >=13.0")
            versions.add(parsed)
    else:
        if not isinstance(catalog, dict) or not isinstance(catalog.get("gpus"), list):
            raise ValueError("RunPod CUDA catalog has no GPU inventory")
        selected = {}
        for row in catalog["gpus"]:
            if not isinstance(row, dict) or not isinstance(row.get("id"), str):
                raise ValueError("RunPod CUDA catalog contains a malformed GPU row")
            if row["id"] in gpu_ids:
                if row["id"] in selected:
                    raise ValueError(f"RunPod CUDA catalog repeats {row['id']}")
                selected[row["id"]] = row
        for gpu in gpu_ids:
            row = selected.get(gpu)
            if row is None or row.get("manufacturer") != "NVIDIA":
                raise ValueError(f"RunPod CUDA catalog has no NVIDIA metadata for {gpu}")
            if not isinstance(row.get("cudaVersions"), list) or not row["cudaVersions"]:
                raise ValueError(f"RunPod CUDA catalog has no version metadata for {gpu}")
            for item in row["cudaVersions"]:
                if not isinstance(item, dict) or type(item.get("available")) is not bool:
                    raise ValueError(f"RunPod CUDA catalog has malformed version metadata for {gpu}")
                parsed = cuda_version(item.get("version"))
                if parsed >= CUDA_FLOOR:
                    # Keep temporarily unavailable compatible versions: the
                    # existing create loop can retry when capacity returns.
                    versions.add(parsed)
    if not versions:
        raise ValueError("no advertised CUDA version meets the pinned >=13.0 floor")
    return [f"{major}.{minor}" for major, minor in sorted(versions)]


def build_request(name, image, gpus, disk, amd, count, *, env, catalog=None, bootstrap=""):
    """Pure request builder; retains the existing placement and cleanup contract."""
    gpu_ids = [gpu.strip() for gpu in gpus.split(",") if gpu.strip()]
    if not gpu_ids:
        raise ValueError("GPU selection is empty")
    req = {"name": name, "imageName": image, "gpuTypeIds": gpu_ids, "gpuCount": int(count),
           "cloudType": "SECURE", "containerDiskInGb": int(disk), "volumeInGb": 0,
           "ports": ["22/tcp"], "supportPublicIp": True, "interruptible": False}
    cloud = env.get("MOJOLEARN_DEVPOD_CLOUD_TYPE", "SECURE")
    if cloud not in ("SECURE", "COMMUNITY"):
        raise ValueError("MOJOLEARN_DEVPOD_CLOUD_TYPE must be SECURE or COMMUNITY")
    req["cloudType"] = cloud
    if env.get("MOJOLEARN_DEVPOD_MIN_VCPU"):
        minimum = int(env["MOJOLEARN_DEVPOD_MIN_VCPU"])
        if minimum < 1:
            raise ValueError("MOJOLEARN_DEVPOD_MIN_VCPU must be positive")
        req["minVCPUPerGPU"] = minimum
    if not amd:
        req["allowedCudaVersions"] = allowed_cuda_versions(gpu_ids, catalog, env.get(OVERRIDE))
    if amd and image.startswith("rocm/"):
        req["dockerEntrypoint"] = ["/bin/bash", "-lc"]
        req["dockerStartCmd"] = [bootstrap]
    return req


def build_v2_request(name, image, gpus, disk, count, attempt, *, env):
    """Select one GPU per attempt; the caller advances only after proven stock refusal.

    v2 accepts one gpu.id, not v1's gpuTypeIds. Rotate the requested alternatives
    in order, with the same count/resources and >=13.0 driver requirement.
    Schema: runpod/docs/api-reference-v2/openapi.json, CreatePodRequest and
    CreateGpuConfig. v2 has no interruptible placement field; this is the
    standard on-demand create endpoint, with no bid/spot request.
    """
    gpu_ids = [gpu.strip() for gpu in gpus.split(",")]
    if not gpu_ids or any(not gpu.startswith("NVIDIA ") for gpu in gpu_ids):
        raise ValueError("v2 NVIDIA selection requires nonempty NVIDIA GPU identifiers")
    count, disk, attempt = int(count), int(disk), int(attempt)
    if not 1 <= count <= 8 or disk < 1 or attempt < 0:
        raise ValueError("invalid GPU count, disk size or attempt index")
    cloud = env.get("MOJOLEARN_DEVPOD_CLOUD_TYPE", "SECURE")
    if cloud not in ("SECURE", "COMMUNITY"):
        raise ValueError("MOJOLEARN_DEVPOD_CLOUD_TYPE must be SECURE or COMMUNITY")
    gpu = {"id": gpu_ids[attempt % len(gpu_ids)], "count": count}
    if OVERRIDE in env:
        gpu["allowedCudaVersions"] = allowed_cuda_versions(gpu_ids, override=env[OVERRIDE])
    else:
        gpu["minCudaVersion"] = "13.0"
    if env.get("MOJOLEARN_DEVPOD_MIN_VCPU"):
        minimum = int(env["MOJOLEARN_DEVPOD_MIN_VCPU"])
        if minimum < 1:
            raise ValueError("MOJOLEARN_DEVPOD_MIN_VCPU must be positive")
        gpu["minVcpuCountPerGpu"] = minimum
    return {"name": name, "image": image, "cloud": cloud, "disk": disk,
            "startSsh": True, "ports": ["22/tcp"], "gpu": gpu}


def main(argv=None):
    args = sys.argv[1:] if argv is None else argv
    if args and args[0] == "v2":
        if len(args) != 8:
            raise SystemExit("usage: runpod_gpu_request.py v2 OUT NAME IMAGE GPUS DISK COUNT ATTEMPT")
        _, out, name, image, gpus, disk, count, attempt = args
        try:
            req = build_v2_request(name, image, gpus, disk, count, attempt, env=os.environ)
            Path(out).write_text(json.dumps(req, indent=2) + "\n")
        except (OSError, ValueError, TypeError) as error:
            raise SystemExit("RunPod request refused before create: " + str(error)) from None
        return 0
    if len(args) != 9:
        raise SystemExit("usage: runpod_gpu_request.py OUT NAME IMAGE GPUS DISK AMD BOOTSTRAP COUNT CATALOG")
    out, name, image, gpus, disk, amd, bootstrap, count, catalog_path = args
    try:
        catalog = None
        if amd != "1" and OVERRIDE not in os.environ:
            catalog = json.loads(Path(catalog_path).read_text())
        bootstrap_text = Path(bootstrap).read_text() if amd == "1" and image.startswith("rocm/") else ""
        req = build_request(name, image, gpus, disk, amd == "1", count, env=os.environ,
                            catalog=catalog, bootstrap=bootstrap_text)
        Path(out).write_text(json.dumps(req, indent=2) + "\n")
    except (OSError, ValueError, TypeError) as error:
        raise SystemExit("RunPod request refused before create: " + str(error)) from None
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
