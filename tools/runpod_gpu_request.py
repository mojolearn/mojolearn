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


# CPU pods (tools/cpu_central.sh through tools/dev_pod.sh up --vendor cpu, lane fx-cpu-box-checks, 2026-11-19). Every
# call is RunPod API v2 (https://api.runpod.io/v2, the same API the NVIDIA pods use): the v1 host rest.runpod.io stopped
# resolving on 2026-11-20 and v2 refused the old flat v1 CPU body with 422 (lane fx-cpu-pod-v2; evidence
# ~/mojolearn-evidence/devpods/cpc1.requests-20261120T171733Z-96913). The v2 body, field by field, from RunPod's own
# reference (runpod/docs api-reference-v2/openapi.json, rendered at docs.runpod.io/api-reference-v2/pods/create-a-pod):
#   name, cpu          CreatePodRequest: `name` required; exactly one of `gpu` or `cpu`; CPU memory is derived by the
#                      API from the flavor's RAM multiplier, so the client sends only the flavor and the vCPU count
#   cpu.id, cpu.vcpuCount  CreateCpuConfig/BaseCpuConfig: ONE flavor id (as GET /v2/catalog/cpus lists it) and a vCPU
#                      count >= 2 that is valid for the flavor and a power of two
#   image, disk, ports BaseContainerConfig: the image (required unless templateId), container disk GB, "22/tcp"
#   cloud              CreatePodRequest: SECURE (default) or COMMUNITY
#   startSsh           CreatePodRequest: inject the account's registered keys; the pod's ssh.direct (PodSsh) needs 22/tcp
# v2 takes one flavor per create, not v1's ordered cpuFlavorIds list: the create loop tries OUR order (the best for the
# price first) one flavor per attempt, advancing only after a proven no-capacity refusal, as the NVIDIA v2 path rotates
# its GPU ids. No `queue` field: the spec refuses `queue` with `cpu` (PodQueueOptions), and a queued create is a pod
# that appears later. Flavors: cpu<gen><class>, class c (2 GB a vCPU), g (4 GB), m (8 GB); gen 3 or 5. Only these
# RunPod Linux CPU flavors are accepted, so no Apple machine can be named here (AGENTS.md hard rule).
CPU_FLAVOR = re.compile(r"cpu[35][cgm]")
NOT_LINUX_IMAGE = re.compile(r"(?i)apple|macos|darwin|osx")
# RAM per vCPU of a class, when the catalog does not say (RunPod's compute / general / memory classes)
CPU_CLASS_RAM = {"c": 2, "g": 4, "m": 8}
CPU_FALLBACK = ["cpu5c", "cpu3c", "cpu5g", "cpu3g", "cpu5m", "cpu3m"]


def is_power_of_two(n):
    return n >= 1 and n & (n - 1) == 0


def build_cpu_request(name, image, flavors, vcpu, disk, attempt=0, *, env):
    """The v2 CreatePodRequest for one CPU pod: flavor `attempt` (mod the list) of the comma-separated order."""
    flavor_ids = [f.strip() for f in flavors.split(",") if f.strip()]
    if not flavor_ids or any(not CPU_FLAVOR.fullmatch(f) for f in flavor_ids):
        raise ValueError("CPU flavors must be cpu3c/cpu3g/cpu3m/cpu5c/cpu5g/cpu5m, comma separated (RunPod Linux CPU only)")
    if NOT_LINUX_IMAGE.search(image):
        raise ValueError("the CPU box is a RunPod Linux pod; no Apple image, ever (AGENTS.md hard rule)")
    vcpu, disk, attempt = int(vcpu), int(disk), int(attempt)
    if not 2 <= vcpu <= 512 or not is_power_of_two(vcpu) or disk < 1 or attempt < 0:
        raise ValueError("invalid vCPU count (2..512, a power of two: RunPod v2 BaseCpuConfig), disk size or attempt")
    cloud = env.get("MOJOLEARN_DEVPOD_CLOUD_TYPE", "SECURE")
    if cloud not in ("SECURE", "COMMUNITY"):
        raise ValueError("MOJOLEARN_DEVPOD_CLOUD_TYPE must be SECURE or COMMUNITY")
    return {"name": name, "image": image, "cloud": cloud, "disk": disk, "startSsh": True, "ports": ["22/tcp"],
            "cpu": {"id": flavor_ids[attempt % len(flavor_ids)], "vcpuCount": vcpu}}


def pick_cpu_instance(catalog, min_vcpu=32, min_ram_per_vcpu=2):
    """(flavor ids in order, vCPU count) from RunPod's CPU catalog: the most vCPUs for the price.

    catalog: the GraphQL `cpuFlavors` rows ({id, maxVcpu, ramMultiplier[, pricePerVcpu]}) or None. Only flavors with
    at least min_vcpu vCPUs and min_ram_per_vcpu GB a vCPU qualify. The vCPU count is the largest power of two (RunPod
    v2 BaseCpuConfig.vcpuCount) the best flavor offers; ranked by price a vCPU when every row has one, else by vCPUs,
    then the cheaper RAM class, then the newer generation. With no usable catalog: the fallback order at the smallest
    power of two at or above min_vcpu.
    """
    rows = []
    for row in catalog or []:
        if not isinstance(row, dict) or not isinstance(row.get("id"), str) or not CPU_FLAVOR.fullmatch(row["id"]):
            continue
        try:
            maxv = int(row.get("maxVcpu") or 0)
            ram = float(row.get("ramMultiplier") or CPU_CLASS_RAM[row["id"][-1]])
            price = row.get("pricePerVcpu")
            price = float(price) if price not in (None, "") else None
        except (TypeError, ValueError):
            continue
        if maxv >= min_vcpu and ram >= min_ram_per_vcpu:
            rows.append((row["id"], maxv, ram, price))
    if not rows:
        order = [f for f in CPU_FALLBACK if CPU_CLASS_RAM[f[-1]] >= min_ram_per_vcpu]
        if not order:
            raise ValueError(f"no CPU flavor has {min_ram_per_vcpu} GB a vCPU")
        # the smallest power of two at or above the minimum (v2 BaseCpuConfig.vcpuCount)
        return order, 1 << (max(int(min_vcpu), 2) - 1).bit_length()
    # v2 takes only a power-of-two vCPU count (BaseCpuConfig.vcpuCount): each flavor offers the largest power of two
    # within its maximum
    rows = [(f, 1 << (maxv.bit_length() - 1), ram, price) for f, maxv, ram, price in rows]
    rows = [r for r in rows if r[1] >= min_vcpu]
    if not rows:
        raise ValueError(f"no CPU flavor offers a power-of-two vCPU count of at least {min_vcpu}")
    priced = all(r[3] is not None for r in rows)
    if priced:
        rows.sort(key=lambda r: (r[3], -r[1], r[2], -int(r[0][3])))
    else:
        rows.sort(key=lambda r: (-r[1], r[2], -int(r[0][3])))
    vcpu = rows[0][1]
    return [r[0] for r in rows if r[1] >= vcpu], vcpu


def check_cpu_vcpu(catalog, flavors, vcpu):
    """Refuse a vCPU count outside a flavor's catalog range BEFORE any provider call (lane fx-cpu-box-wait, 2026-11-20:
    cpu5m takes 2..32). catalog: the GraphQL `cpuFlavors` rows or None (then nothing is known and nothing is refused
    here; the API refuses with nothing created). Every flavor in the comma-separated order must be in the catalog and
    allow `vcpu`."""
    vcpu = int(vcpu)
    rows = {r["id"]: r for r in catalog or [] if isinstance(r, dict) and isinstance(r.get("id"), str)}
    if not rows:
        return
    for f in [f.strip() for f in str(flavors).split(",") if f.strip()]:
        row = rows.get(f)
        if row is None:
            raise ValueError(f"CPU flavor {f} is not in RunPod's CPU catalog ({', '.join(sorted(rows))})")
        try:
            lo, hi = int(row.get("minVcpu") or 2), int(row.get("maxVcpu") or 0)
        except (TypeError, ValueError):
            continue
        if hi and not lo <= vcpu <= hi:
            raise ValueError(f"CPU flavor {f} takes {lo}..{hi} vCPU (RunPod's CPU catalog); {vcpu} asked")


def main(argv=None):
    args = sys.argv[1:] if argv is None else argv
    if args and args[0] == "cpu":
        if len(args) not in (7, 8):
            raise SystemExit("usage: runpod_gpu_request.py cpu OUT NAME IMAGE FLAVORS VCPU DISK [ATTEMPT]")
        _, out, name, image, flavors, vcpu, disk = args[:7]
        attempt = args[7] if len(args) == 8 else 0
        try:
            req = build_cpu_request(name, image, flavors, vcpu, disk, attempt, env=os.environ)
            Path(out).write_text(json.dumps(req, indent=2) + "\n")
        except (OSError, ValueError, TypeError) as error:
            raise SystemExit("RunPod request refused before create: " + str(error)) from None
        return 0
    if args and args[0] == "cpu-pick":
        # cpu-pick CATALOG_JSON MIN_VCPU MIN_RAM_PER_VCPU [FLAVORS VCPU] -> "<flavor,flavor,...> <vcpu>"; a missing or
        # unreadable catalog picks from the fallback order. FLAVORS / VCPU (each '-' for the pick's own) are the
        # caller's pins: they replace the pick and are checked against the catalog's vCPU range, as the pick is
        if len(args) not in (4, 6):
            raise SystemExit("usage: runpod_gpu_request.py cpu-pick CATALOG_JSON MIN_VCPU MIN_RAM_PER_VCPU [FLAVORS VCPU]")
        try:
            data = json.loads(Path(args[1]).read_text())
            catalog = (data.get("data") or {}).get("cpuFlavors") if isinstance(data, dict) else data
        except (OSError, ValueError, AttributeError):
            catalog = None
        try:
            order, vcpu = pick_cpu_instance(catalog, int(args[2]), float(args[3]))
            flavors = ",".join(order)
            if len(args) == 6:
                flavors = flavors if args[4] in ("", "-") else args[4]
                vcpu = vcpu if args[5] in ("", "-") else int(args[5])
            check_cpu_vcpu(catalog, flavors, vcpu)
        except ValueError as error:
            raise SystemExit("RunPod request refused before create: " + str(error)) from None
        print(flavors, vcpu)
        return 0
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
