#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn
"""tools/ops/host_probe.py -- what the measurement box is, so a time is only ever compared with a time taken on the
same box (AGENTS.md "Times are compared only on the same box", Andrew, 2026-11-14: two L40S hosts differed 1.2-2.2x
on unchanged code and produced 56 false regression flags).

    python3 tools/ops/host_probe.py --vendor nvidia|amd|none [--out FILE] [--mb 880]

Runs ON THE BOX (tools/ops/paper_compare_job.sh, phase probe, before any board), prints one JSON document and writes
it to --out. Fields (schema mojolearn-host-probe/1):
  host_id            "<RunPod pod id, else the hostname>/<GPU 0 uuid>": the key every compare matches on
  hostname, pod_id, gpu_uuid
  cpu_model, numa_nodes, nproc, affinity_cores, cgroup_cores (cgroup v2 cpu.max, else v1 cfs quota), mem_total_gb
  memcpy_gbps        numpy copy of --mb MB host to host (best of 3)
  h2d_pageable_gbps  host-to-device copy of --mb MB from pageable memory (best of 3)
  h2d_pinned_gbps    the same from pinned memory (cuMemHostAlloc / hipHostMalloc)
  h2d_api            the driver the copies went through: the CUDA driver (libcuda) or HIP (libamdhip64) through
                     ctypes. No torch, no mojolearn import: the probe must be the same code for every arm.
  pcie               generation and width read WHILE pinned copies run (an idle link downshifts): NVIDIA through
                     `nvidia-smi --query-gpu=pcie.link.gen.current,pcie.link.width.current`, AMD from the GPU's sysfs
                     current_link_speed / current_link_width; plus the maximum the link supports
  gpu                name and driver
  errors             every step that failed (the probe never fails the job; a missing field is shown as such)
Tests: tools/test_paper_compare_same_box.py.
"""
import argparse
import ctypes
import glob
import json
import os
import platform
import re
import subprocess
import sys
import threading
import time

SCHEMA = "mojolearn-host-probe/1"
MB = 880        # the brief's copy size: about one full-size taxi block


def _read(path):
    try:
        with open(path) as fh:
            return fh.read().strip()
    except OSError:
        return None


def _run(argv, timeout=30):
    try:
        p = subprocess.run(argv, capture_output=True, text=True, timeout=timeout)
        return p.stdout.strip() if p.returncode == 0 else None
    except (OSError, subprocess.SubprocessError):
        return None


def cpu_model():
    text = _read("/proc/cpuinfo") or ""
    m = re.search(r"^model name\s*:\s*(.+)$", text, re.M)
    if m:
        return m.group(1).strip()
    return _run(["sysctl", "-n", "machdep.cpu.brand_string"]) or platform.processor() or None


def numa_nodes():
    nodes = glob.glob("/sys/devices/system/node/node[0-9]*")
    return len(nodes) or None


def cgroup_cores():
    """The pod's CPU allocation: cgroup v2 cpu.max, else v1 cfs quota/period (None when unlimited or unreadable)."""
    v2 = _read("/sys/fs/cgroup/cpu.max")
    if v2:
        q, _, per = v2.partition(" ")
        if q != "max" and per.isdigit() and int(per) > 0:
            return -(-int(q) // int(per))
        return None
    q, per = _read("/sys/fs/cgroup/cpu/cpu.cfs_quota_us"), _read("/sys/fs/cgroup/cpu/cpu.cfs_period_us")
    try:
        if q and per and int(q) > 0 and int(per) > 0:
            return -(-int(q) // int(per))
    except ValueError:
        pass
    return None


def mem_total_gb():
    m = re.search(r"^MemTotal:\s*(\d+)", _read("/proc/meminfo") or "", re.M)
    return round(int(m.group(1)) / 1048576, 1) if m else None


def memcpy_gbps(nbytes, reps=3):
    import numpy as np
    src = np.ones(nbytes, dtype=np.uint8)
    dst = np.empty_like(src)
    best = None
    for _ in range(reps):
        t = time.perf_counter()
        np.copyto(dst, src)
        dt = time.perf_counter() - t
        best = dt if best is None or dt < best else best
    return round(nbytes / best / 1e9, 2)


class _Cuda:
    """The CUDA driver through ctypes: GPU 0's primary context, one device buffer."""
    name = "cuda-driver"

    def __init__(self):
        self.lib = ctypes.CDLL("libcuda.so.1")
        self._ok(self.lib.cuInit(0), "cuInit")
        self.dev = ctypes.c_int()
        self._ok(self.lib.cuDeviceGet(ctypes.byref(self.dev), 0), "cuDeviceGet")
        self.ctx = ctypes.c_void_p()
        self._ok(self.lib.cuDevicePrimaryCtxRetain(ctypes.byref(self.ctx), self.dev), "cuDevicePrimaryCtxRetain")
        self._ok(self.lib.cuCtxSetCurrent(self.ctx), "cuCtxSetCurrent")

    @staticmethod
    def _ok(rc, what):
        if rc != 0:
            raise RuntimeError("%s returned %d" % (what, rc))

    def alloc(self, n):
        p = ctypes.c_uint64()
        self._ok(self.lib.cuMemAlloc_v2(ctypes.byref(p), ctypes.c_size_t(n)), "cuMemAlloc")
        return p

    def host_alloc(self, n):
        p = ctypes.c_void_p()
        self._ok(self.lib.cuMemHostAlloc(ctypes.byref(p), ctypes.c_size_t(n), 0), "cuMemHostAlloc")
        return p

    def h2d(self, dst, src_ptr, n):
        self._ok(self.lib.cuMemcpyHtoD_v2(dst, ctypes.c_void_p(src_ptr), ctypes.c_size_t(n)), "cuMemcpyHtoD")
        self._ok(self.lib.cuCtxSynchronize(), "cuCtxSynchronize")

    def free(self, dptr, hptr):
        if hptr is not None:
            self.lib.cuMemFreeHost(hptr)
        if dptr is not None:
            self.lib.cuMemFree_v2(dptr)

    def close(self):
        self.lib.cuDevicePrimaryCtxRelease(self.dev)


class _Hip:
    """HIP through ctypes: device 0, one device buffer."""
    name = "hip"

    def __init__(self):
        lib = None
        for cand in ("libamdhip64.so", "libamdhip64.so.6", "libamdhip64.so.7", "/opt/rocm/lib/libamdhip64.so"):
            try:
                lib = ctypes.CDLL(cand)
                break
            except OSError:
                continue
        if lib is None:
            raise RuntimeError("libamdhip64 not found")
        self.lib = lib
        self._ok(self.lib.hipSetDevice(0), "hipSetDevice")

    _ok = staticmethod(_Cuda._ok)

    def alloc(self, n):
        p = ctypes.c_void_p()
        self._ok(self.lib.hipMalloc(ctypes.byref(p), ctypes.c_size_t(n)), "hipMalloc")
        return p

    def host_alloc(self, n):
        p = ctypes.c_void_p()
        self._ok(self.lib.hipHostMalloc(ctypes.byref(p), ctypes.c_size_t(n), 0), "hipHostMalloc")
        return p

    def h2d(self, dst, src_ptr, n):
        self._ok(self.lib.hipMemcpy(dst, ctypes.c_void_p(src_ptr), ctypes.c_size_t(n), 1), "hipMemcpy")
        self._ok(self.lib.hipDeviceSynchronize(), "hipDeviceSynchronize")

    def free(self, dptr, hptr):
        if hptr is not None:
            self.lib.hipHostFree(hptr)
        if dptr is not None:
            self.lib.hipFree(dptr)

    def close(self):
        pass


def _amd_pci_dir():
    for card in sorted(glob.glob("/sys/class/drm/card[0-9]*/device")):
        if (_read(os.path.join(card, "vendor")) or "").lower() == "0x1002" and os.path.exists(
                os.path.join(card, "current_link_speed")):
            return card
    return None


def _gen_of_speed(text):
    """sysfs link speed ('32.0 GT/s PCIe') -> PCIe generation."""
    m = re.match(r"\s*([\d.]+)\s*GT/s", text or "")
    if not m:
        return None
    return {2.5: 1, 5.0: 2, 8.0: 3, 16.0: 4, 32.0: 5, 64.0: 6}.get(float(m.group(1)))


def read_pcie(vendor):
    """{gen, width, max_gen, max_width, source} of GPU 0's link, now."""
    if vendor == "nvidia":
        s = _run(["nvidia-smi", "--query-gpu=pcie.link.gen.current,pcie.link.width.current,pcie.link.gen.max,"
                  "pcie.link.width.max", "--format=csv,noheader,nounits"])
        if not s:
            return {"source": "nvidia-smi", "error": "nvidia-smi gave nothing"}
        f = [x.strip() for x in s.splitlines()[0].split(",")]
        num = lambda x: int(x) if x.isdigit() else x  # noqa: E731
        return {"gen": num(f[0]), "width": num(f[1]), "max_gen": num(f[2]) if len(f) > 2 else None,
                "max_width": num(f[3]) if len(f) > 3 else None, "source": "nvidia-smi"}
    d = _amd_pci_dir()
    if not d:
        return {"source": "sysfs", "error": "no AMD GPU link in /sys/class/drm"}
    w, mw = _read(os.path.join(d, "current_link_width")), _read(os.path.join(d, "max_link_width"))
    return {"gen": _gen_of_speed(_read(os.path.join(d, "current_link_speed"))),
            "width": int(w) if w and w.isdigit() else w,
            "max_gen": _gen_of_speed(_read(os.path.join(d, "max_link_speed"))),
            "max_width": int(mw) if mw and mw.isdigit() else mw, "source": "sysfs " + d}


def h2d_probe(vendor, nbytes, reps=3):
    """(pageable GB/s, pinned GB/s, api, pcie read during pinned copies)."""
    import numpy as np
    api = _Cuda() if vendor == "nvidia" else _Hip()
    dptr = hptr = None
    try:
        dptr = api.alloc(nbytes)
        src = np.ones(nbytes, dtype=np.uint8)
        best = None
        for _ in range(reps):
            t = time.perf_counter()
            api.h2d(dptr, src.ctypes.data, nbytes)
            dt = time.perf_counter() - t
            best = dt if best is None or dt < best else best
        pageable = round(nbytes / best / 1e9, 2)
        del src
        hptr = api.host_alloc(nbytes)
        ctypes.memset(hptr, 1, nbytes)
        # the link is read WHILE pinned copies run: an idle GPU link downshifts to gen 1
        box = {}
        reader = threading.Thread(target=lambda: box.update(pcie=read_pcie(vendor)), daemon=True)
        best, count, t_end = None, 0, time.perf_counter() + 20.0
        reader.start()
        while count < reps or (reader.is_alive() and time.perf_counter() < t_end):
            t = time.perf_counter()
            api.h2d(dptr, hptr.value, nbytes)
            dt = time.perf_counter() - t
            best = dt if best is None or dt < best else best
            count += 1
        reader.join(timeout=5)
        pcie = box.get("pcie") or {"error": "the link read did not finish during the copies"}
        pcie["read_during_copy"] = "pcie" in box
        return pageable, round(nbytes / best / 1e9, 2), api.name, pcie
    finally:
        api.free(dptr, hptr)
        api.close()


def gpu_info(vendor):
    if vendor == "nvidia":
        s = _run(["nvidia-smi", "--query-gpu=name,driver_version,uuid", "--format=csv,noheader"])
        if s:
            f = [x.strip() for x in s.splitlines()[0].split(",")]
            return {"name": f[0], "driver": f[1] if len(f) > 1 else None, "uuid": f[2] if len(f) > 2 else None,
                    "count": len(s.splitlines())}
        return {}
    if vendor == "amd":
        out = {}
        s = _run(["rocm-smi", "--showproductname"]) or ""
        m = re.search(r"Card Series\s*:\s*(.+)", s)
        if m:
            out["name"] = m.group(1).strip()
        d = _run(["rocm-smi", "--showdriverversion"]) or ""
        m = re.search(r"Driver version:\s*(\S+)", d)
        out["driver"] = m.group(1) if m else (_read("/sys/module/amdgpu/version") or None)
        u = _run(["rocm-smi", "--showuniqueid"]) or ""
        m = re.search(r"Unique ID:\s*(\S+)", u)
        if m:
            out["uuid"] = m.group(1)
        return out
    return {}


def probe(vendor, mb=MB):
    out = {"schema": SCHEMA, "taken": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "vendor": vendor,
           "hostname": platform.node(), "pod_id": os.environ.get("RUNPOD_POD_ID") or None, "errors": []}

    def step(name, fn):
        try:
            return fn()
        except Exception as exc:  # noqa: BLE001 -- the probe records every failure and goes on
            out["errors"].append("%s: %s" % (name, str(exc)[:300]))
            return None

    out["cpu_model"] = step("cpu_model", cpu_model)
    out["numa_nodes"] = step("numa_nodes", numa_nodes)
    out["nproc"] = os.cpu_count()
    out["affinity_cores"] = step("affinity", lambda: len(os.sched_getaffinity(0)))
    out["cgroup_cores"] = step("cgroup_cores", cgroup_cores)
    out["mem_total_gb"] = step("mem_total", mem_total_gb)
    nbytes = int(mb * 1e6)
    out["copy_mb"] = mb
    out["memcpy_gbps"] = step("memcpy", lambda: memcpy_gbps(nbytes))
    gpu = step("gpu", lambda: gpu_info(vendor)) or {}
    out["gpu"] = {k: gpu.get(k) for k in ("name", "driver", "count") if gpu.get(k) is not None}
    out["gpu_uuid"] = gpu.get("uuid")
    out["h2d_pageable_gbps"] = out["h2d_pinned_gbps"] = out["h2d_api"] = None
    out["pcie"] = None
    if vendor in ("nvidia", "amd"):
        got = step("h2d", lambda: h2d_probe(vendor, nbytes))
        if got:
            out["h2d_pageable_gbps"], out["h2d_pinned_gbps"], out["h2d_api"], out["pcie"] = got
        else:
            out["pcie"] = step("pcie", lambda: dict(read_pcie(vendor), read_during_copy=False))
    out["host_id"] = host_id(out)
    return out


def host_id(p):
    """'<pod id or hostname>/<GPU uuid or ->': what makes two timings comparable (paper_compare matches on it)."""
    return "%s/%s" % (p.get("pod_id") or p.get("hostname") or "?", p.get("gpu_uuid") or "-")


def load(path):
    with open(path) as fh:
        doc = json.load(fh)
    if doc.get("schema") != SCHEMA:
        raise ValueError("%s is not a %s document" % (path, SCHEMA))
    return doc


def write(doc, path):
    path = os.path.abspath(path)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = "%s.tmp-%d" % (path, os.getpid())
    with open(tmp, "w") as fh:
        fh.write(json.dumps(doc, indent=1, sort_keys=True) + "\n")
    os.replace(tmp, path)


def main(argv=None):
    p = argparse.ArgumentParser(prog="host_probe", description=__doc__.split("\n")[0])
    p.add_argument("--vendor", required=True, choices=("nvidia", "amd", "none"))
    p.add_argument("--out", default=None, help="also write the JSON here")
    p.add_argument("--mb", type=float, default=MB, help="copy size in MB (default %d)" % MB)
    a = p.parse_args(argv)
    doc = probe(a.vendor, a.mb)
    if a.out:
        write(doc, a.out)
    sys.stdout.write(json.dumps(doc, indent=1, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
