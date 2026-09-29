# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Two things every benchmark-board driver needs inside its worker processes:
PER-ARM MEMORY and OUR CPU ARM. Standard library only at import time, so the
conductor (tools/bench_board.py) can read the tables without numpy.

MEMORY (`MemProbe`)
-------------------
`start()` runs just before an arm's timed call and `stop()` just after it,
both OUTSIDE the clock. They reset and read back a high-water mark, so each
round's number is that round's peak, not the process's whole life:

  host, Linux   /proc/self/status VmHWM after writing 5 to
                /proc/self/clear_refs (the kernel's resettable peak RSS).
                When the reset is refused the lifetime VmHWM is read and the
                method says so.
  host, macOS   proc_pid_rusage RUSAGE_INFO_V4 ri_interval_max_phys_footprint
                after proc_reset_footprint_interval: the peak physical
                footprint over the round. On Apple silicon Metal buffers are
                charged to the footprint, so the GPU side of an Apple arm is
                INSIDE this number (unified memory).
  host, other   getrusage(RUSAGE_SELF).ru_maxrss, lifetime.
  host children the resident size of the process's descendants (joblib or
                loky pools, torch.compile workers) at the round's end, from
                `ps`, reported separately as children_mb (not a peak).

  GPU           CPU arms: none. torch on CUDA/ROCm: torch.cuda
                max_memory_allocated, reset before the round (the caching
                allocator's peak; the context is not in it). torch on MPS:
                torch.mps.driver_allocated_memory at the round's end (not a
                peak). Every other GPU arm (ours, cuML, cuVS, XGBoost,
                CatBoost, LightGBM): the driver's per-process figure at the
                round's end, `nvidia-smi --query-compute-apps` or `rocm-smi
                --showpids`, the process total (context and every pool; a
                buffer freed inside the round is not seen). On Apple there is
                no per-process GPU counter: the figure is None and the method
                points at the footprint.

Every sample carries its method string; the board prints it beside the
number.

OUR CPU ARM (`ours-cpu`)
------------------------
The wheel's public CPU switch is MOJOLEARN_VENDOR=cpu before import
(python/mojolearn/_backend.py): the GPU set is not loaded and the host
bindings under mojolearn/host/ answer (IDENTICAL only, the only tier they
build). `ours_cpu_env` sets it for a worker; `ours_cpu_check` reads back
`mojolearn.vendor()` and refuses BY NAME when the installed wheel did not
honour the switch (the macOS wheel read it only on the Linux vendor layout
through 0.8.22), so a Metal or CUDA fit is never timed under the CPU label.
"""
import os
import subprocess
import sys

OURS_CPU_ARM = "ours-cpu"
#: set in an ours-cpu worker's environment; the drivers' readbacks read it
OURS_CPU_FLAG = "MOJOLEARN_BOARD_OURS_CPU"
#: the conductor tells workers which vendor the box is (apple, nvidia, amd)
VENDOR_ENV = "MOJOLEARN_BOARD_VENDOR"
CPU_SWITCH = "MOJOLEARN_VENDOR=cpu (set before import; the wheel's public CPU switch)"


def ours_cpu_env(env):
    """`env` (a dict) made into an ours-cpu worker's environment."""
    env["MOJOLEARN_VENDOR"] = "cpu"
    env["MOJOLEARN_NUMERIC_MODE"] = "identical"
    env[OURS_CPU_FLAG] = "1"
    env.pop("MOJOLEARN_VENDOR_FORCE", None)
    return env


def ours_cpu_requested():
    return os.environ.get(OURS_CPU_FLAG, "").strip() == "1"


def ours_cpu_check(ml):
    """The readback of an ours-cpu worker: {} when this is not one, else the
    info fields to merge. Raises (a by-name refusal) unless the wheel loaded
    its CPU set."""
    if not ours_cpu_requested():
        return {}
    try:
        said = ml.vendor()
    except Exception as exc:  # noqa: BLE001
        said = "unavailable (%r)" % (exc,)
    how = None
    try:
        how = ml._backend.vendor_how()
    except Exception:  # noqa: BLE001
        pass
    if said != "cpu":
        raise RuntimeError(
            "REFUSED: ours-cpu: the installed wheel did not load its CPU tier under "
            "MOJOLEARN_VENDOR=cpu (mojolearn.vendor() = %r; %s). The macOS wheel through "
            "0.8.22 reads the switch only on the Linux vendor layout." % (said, how))
    return {"device": "cpu", "vendor_used": "cpu", "cpu_switch": CPU_SWITCH,
            "vendor_how": how}


def bits_equal(a, b):
    """True when two output dicts (name -> ndarray-like) are the same bytes,
    same keys, same shapes and dtypes; None when either is missing."""
    if not a or not b:
        return None
    if sorted(a) != sorted(b):
        return False
    import numpy as np
    for k in a:
        x, y = np.asarray(a[k]), np.asarray(b[k])
        if x.shape != y.shape or x.dtype != y.dtype:
            return False
        if np.ascontiguousarray(x).tobytes() != np.ascontiguousarray(y).tobytes():
            return False
    return True


# ---------------------------------------------------------------------------
# Memory
# ---------------------------------------------------------------------------

_MB = 1024.0 * 1024.0


class _RusageV4(object):
    """ctypes access to proc_pid_rusage(RUSAGE_INFO_V4) on macOS."""
    # uint8 uuid[16], then uint64 fields; the indices below are into the
    # uint64 array (sys/resource.h, struct rusage_info_v4)
    LIFETIME_MAX = 28
    INTERVAL_MAX = 33
    N = 40

    def __init__(self):
        import ctypes
        import ctypes.util
        self.ct = ctypes
        self.lib = ctypes.CDLL(ctypes.util.find_library("proc") or "/usr/lib/libSystem.B.dylib")
        self.lib.proc_pid_rusage.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_void_p]
        self.lib.proc_pid_rusage.restype = ctypes.c_int
        self.reset_fn = getattr(self.lib, "proc_reset_footprint_interval", None)
        if self.reset_fn is not None:
            self.reset_fn.argtypes = [ctypes.c_int]
            self.reset_fn.restype = ctypes.c_int

    def read(self):
        ct = self.ct

        class Info(ct.Structure):
            _fields_ = [("uuid", ct.c_uint8 * 16), ("v", ct.c_uint64 * self.N)]
        info = Info()
        if self.lib.proc_pid_rusage(os.getpid(), 4, ct.byref(info)) != 0:
            return None
        return list(info.v)

    def reset(self):
        return self.reset_fn is not None and self.reset_fn(os.getpid()) == 0


#: libraries whose GPU buffers live in torch's caching allocator
#: libraries whose GPU buffers live in torch's caching allocator (mamba-ssm's
#: kernels allocate through torch)
TORCH_LIBRARIES = ("torch", "gpytorch", "torch_geometric", "torch-geometric", "mamba-ssm",
                   "mamba_ssm")


class MemProbe(object):
    """Per-round peak memory of this process, sampled around the timed call.

    device: 'cpu' or 'gpu' (the arm's device); vendor: apple, nvidia or amd
    (default: MOJOLEARN_BOARD_VENDOR, else guessed from the platform);
    shared: True when several arms share this process (the trees driver), so
    a process-level GPU figure is labelled as the process total. library: the
    arm's library (runner.info["library"]); torch's allocator counter is read
    only for a torch arm. Before 2026-09-29 it was read for ANY arm in a
    process where torch had initialized CUDA/ROCm, so on do-amd our trees arm
    (and XGBoost's) reported peak_gpu_mb 0.0 from torch's allocator, which
    never sees their buffers. None (a caller that does not say) keeps that
    older behavior."""

    def __init__(self, device="gpu", vendor=None, shared=False, library=None):
        self.device = device
        self.library = library
        self.vendor = (vendor or os.environ.get(VENDOR_ENV)
                       or ("apple" if sys.platform == "darwin" else None))
        self.shared = shared
        self._mac = None
        self._linux_reset = None
        if sys.platform == "darwin":
            try:
                self._mac = _RusageV4()
            except Exception:  # noqa: BLE001
                self._mac = None

    # -- host ---------------------------------------------------------------
    def _reset_host(self):
        if self._mac is not None:
            self._mac_reset_ok = self._mac.reset()
            return
        if sys.platform.startswith("linux"):
            try:
                with open("/proc/self/clear_refs", "w") as fh:
                    fh.write("5")
                self._linux_reset = True
            except OSError:
                self._linux_reset = False

    def _read_host(self):
        if self._mac is not None:
            v = self._mac.read()
            if v is not None:
                if getattr(self, "_mac_reset_ok", False):
                    return (v[_RusageV4.INTERVAL_MAX] / _MB,
                            "macOS proc_pid_rusage ri_interval_max_phys_footprint (peak physical "
                            "footprint over the round; Metal buffers are inside it)")
                return (v[_RusageV4.LIFETIME_MAX] / _MB,
                        "macOS proc_pid_rusage ri_lifetime_max_phys_footprint (the process's "
                        "lifetime peak: the interval reset was refused)")
        if sys.platform.startswith("linux"):
            try:
                with open("/proc/self/status") as fh:
                    for line in fh:
                        if line.startswith("VmHWM:"):
                            kb = float(line.split()[1])
                            if self._linux_reset:
                                return (kb / 1024.0, "Linux VmHWM after clear_refs 5 (peak RSS "
                                                     "over the round)")
                            return (kb / 1024.0, "Linux VmHWM, lifetime peak RSS (clear_refs "
                                                 "refused)")
            except OSError:
                pass
        try:
            import resource
            r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            mb = r / _MB if sys.platform == "darwin" else r / 1024.0
            return mb, "getrusage ru_maxrss (lifetime peak RSS)"
        except Exception:  # noqa: BLE001
            return None, "unavailable"

    def _children_mb(self):
        """Resident MB of this process's descendants now, or None."""
        try:
            out = subprocess.run(["ps", "-A", "-o", "pid=,ppid=,rss="], capture_output=True,
                                 text=True, timeout=20, check=False).stdout
        except (OSError, subprocess.SubprocessError):
            return None
        kids, rss = {}, {}
        for line in out.splitlines():
            f = line.split()
            if len(f) != 3:
                continue
            try:
                pid, ppid, kb = int(f[0]), int(f[1]), float(f[2])
            except ValueError:
                continue
            kids.setdefault(ppid, []).append(pid)
            rss[pid] = kb
        total, todo, seen = 0.0, list(kids.get(os.getpid(), [])), set()
        while todo:
            p = todo.pop()
            if p in seen:
                continue
            seen.add(p)
            total += rss.get(p, 0.0)
            todo.extend(kids.get(p, []))
        return total / 1024.0

    # -- GPU ----------------------------------------------------------------
    def _torch(self):
        t = sys.modules.get("torch")
        if t is None:
            return None, None
        try:
            if t.cuda.is_available() and t.cuda.is_initialized():
                return t, "cuda"
        except Exception:  # noqa: BLE001
            pass
        try:
            mps = getattr(t.backends, "mps", None)
            if mps is not None and mps.is_available() and hasattr(t, "mps") \
                    and t.mps.driver_allocated_memory() > 0:
                return t, "mps"
        except Exception:  # noqa: BLE001
            pass
        return None, None

    def _reset_gpu(self):
        if self.device != "gpu":
            return
        t, kind = self._torch()
        if kind == "cuda":
            try:
                t.cuda.reset_peak_memory_stats()
            except Exception:  # noqa: BLE001
                pass

    def _smi(self):
        pid = os.getpid()
        total = " (the process total: every arm in this one process)" if self.shared else ""
        if self.vendor == "nvidia":
            try:
                out = subprocess.run(
                    ["nvidia-smi", "--query-compute-apps=pid,used_memory",
                     "--format=csv,noheader,nounits"], capture_output=True, text=True,
                    timeout=30, check=False).stdout
            except (OSError, subprocess.SubprocessError) as exc:
                return None, "nvidia-smi unavailable (%s)" % exc.__class__.__name__
            for line in out.splitlines():
                f = [x.strip() for x in line.split(",")]
                if len(f) >= 2 and f[0] == str(pid):
                    try:
                        return float(f[1]), ("nvidia-smi --query-compute-apps used_memory for this "
                                             "pid at the round's end (context and pools; not a "
                                             "peak)" + total)
                    except ValueError:
                        break
            return None, ("nvidia-smi lists no compute app with this pid (a container's pid "
                          "namespace hides it, or no context was opened)")
        if self.vendor == "amd":
            try:
                out = subprocess.run(["rocm-smi", "--showpids"], capture_output=True, text=True,
                                     timeout=30, check=False).stdout
            except (OSError, subprocess.SubprocessError) as exc:
                return None, "rocm-smi unavailable (%s)" % exc.__class__.__name__
            for line in out.splitlines():
                f = line.split()
                # PID  PROCESS NAME  GPU(s)  VRAM USED  SDMA USED  CU OCCUPANCY
                if len(f) >= 4 and f[0] == str(pid):
                    nums = [x for x in f[1:] if x.isdigit()]
                    if len(nums) >= 2:
                        return (float(nums[1]) / _MB,
                                "rocm-smi --showpids VRAM USED for this pid at the round's end "
                                "(not a peak)" + total)
            return None, "rocm-smi --showpids lists no VRAM for this pid"
        return None, ("Apple unified memory: no per-process GPU counter; Metal buffers are "
                      "inside peak_host_mb (phys_footprint)")

    def _read_gpu(self):
        if self.device != "gpu":
            return None, "cpu arm: no device memory"
        if self.library is not None and str(self.library) not in TORCH_LIBRARIES:
            return self._smi()
        t, kind = self._torch()
        if kind == "cuda":
            try:
                return (t.cuda.max_memory_allocated() / _MB,
                        "torch.cuda.max_memory_allocated, reset before the round (caching "
                        "allocator peak; the context is not in it)")
            except Exception:  # noqa: BLE001
                pass
        if kind == "mps":
            try:
                return (t.mps.driver_allocated_memory() / _MB,
                        "torch.mps.driver_allocated_memory at the round's end (not a peak; "
                        "unified memory, also inside peak_host_mb)")
            except Exception:  # noqa: BLE001
                pass
        return self._smi()

    # -- the round ----------------------------------------------------------
    def start(self):
        try:
            self._reset_gpu()
            self._reset_host()
        except Exception:  # noqa: BLE001  (memory never breaks a timing)
            pass

    def stop(self):
        """{host_mb, host_method, gpu_mb, gpu_method, children_mb}."""
        out = {}
        try:
            out["host_mb"], out["host_method"] = self._read_host()
        except Exception as exc:  # noqa: BLE001
            out["host_mb"], out["host_method"] = None, "error %r" % (exc,)
        try:
            out["gpu_mb"], out["gpu_method"] = self._read_gpu()
        except Exception as exc:  # noqa: BLE001
            out["gpu_mb"], out["gpu_method"] = None, "error %r" % (exc,)
        try:
            out["children_mb"] = self._children_mb()
        except Exception:  # noqa: BLE001
            out["children_mb"] = None
        for k in ("host_mb", "gpu_mb", "children_mb"):
            if isinstance(out.get(k), float):
                out[k] = round(out[k], 1)
        return out


def summarize(samples):
    """Per-round samples (warm-up first, then the timed rounds) -> the cell's
    memory record: peaks over the TIMED rounds, the warm-up apart."""
    samples = list(samples or [])
    timed = [s for s in samples[1:] if isinstance(s, dict)]
    warm = samples[0] if samples and isinstance(samples[0], dict) else None

    def peak(key, rows):
        vals = [s.get(key) for s in rows if isinstance(s.get(key), (int, float))]
        return max(vals) if vals else None
    last = (timed or ([warm] if warm else [{}]))[-1]
    return {"peak_host_mb": peak("host_mb", timed), "peak_gpu_mb": peak("gpu_mb", timed),
            "children_mb": peak("children_mb", timed),
            "warmup_host_mb": (warm or {}).get("host_mb"),
            "warmup_gpu_mb": (warm or {}).get("gpu_mb"),
            "host_method": last.get("host_method"), "gpu_method": last.get("gpu_method"),
            "rounds_sampled": len(timed)}


# ---------------------------------------------------------------------------
# The arm's own library identity (the opponent store's key; tools/bench_board_store.py)
# ---------------------------------------------------------------------------

#: a board library name -> the module it imports as
IMPORT_NAME = {"scikit-learn": "sklearn", "umap-learn": "umap", "hf": "tokenizers",
               "mamba-ssm": "mamba_ssm",
               "torch-geometric": "torch_geometric", "faiss-cpu": "faiss", "cuml-cu12": "cuml"}


def library_version(library):
    """The exact version of the library this process imports: module.__version__,
    else importlib.metadata.version of its distribution; None if not importable."""
    if not library or library in ("mojolearn",):
        return None
    name = IMPORT_NAME.get(library, library).replace("-", "_")
    try:
        import importlib
        mod = sys.modules.get(name) or importlib.import_module(name)
        v = getattr(mod, "__version__", None)
        if v:
            return str(v)
    except Exception:  # noqa: BLE001
        pass
    try:
        import importlib.metadata as md
        for dist in md.packages_distributions().get(name, []) + [library]:
            try:
                return md.version(dist)
            except md.PackageNotFoundError:
                continue
    except Exception:  # noqa: BLE001
        pass
    return None


def gpu_device_name():
    """The GPU this process sees: torch or CuPy when already imported, else the
    vendor tool; None when there is none."""
    torch = sys.modules.get("torch")
    try:
        if torch is not None and torch.cuda.is_available():
            return torch.cuda.get_device_name(0)
    except Exception:  # noqa: BLE001
        pass
    cupy = sys.modules.get("cupy")
    try:
        if cupy is not None:
            name = cupy.cuda.runtime.getDeviceProperties(0)["name"]
            return name.decode() if isinstance(name, bytes) else str(name)
    except Exception:  # noqa: BLE001
        pass
    for cmd in (["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                ["rocm-smi", "--showproductname"]):
        try:
            out = subprocess.run(cmd, capture_output=True, text=True, timeout=30).stdout
        except (OSError, subprocess.SubprocessError):
            continue
        if not out.strip():
            continue
        if cmd[0] == "nvidia-smi":
            return out.splitlines()[0].strip()
        for line in out.splitlines():
            if "Card Series" in line and ":" in line:
                return line.rsplit(":", 1)[1].strip()
    if sys.platform == "darwin":
        try:
            return subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True,
                                  text=True, timeout=10).stdout.strip() or None
        except (OSError, subprocess.SubprocessError):
            return None
    return None


def library_identity(info):
    """Fill `version` and, on a GPU arm, `device_name` in a worker's info when the
    arm did not set them (the store keys an opponent by what it imported and the
    device it ran on)."""
    info = info if isinstance(info, dict) else {}
    out = {}
    if not info.get("version"):
        out["version"] = library_version(info.get("library"))
    if info.get("device") == "gpu" and not info.get("device_name"):
        out["device_name"] = gpu_device_name()
    return out
