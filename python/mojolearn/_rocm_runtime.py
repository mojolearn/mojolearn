# SPDX-License-Identifier: Apache-2.0
"""One ROCm runtime per process on AMD (loader glue, no computation).

The MAX runtime opens the HIP runtime and the HSA runtime with `dlopen` by bare
name (`libamdhip64.so`, `libhsa-runtime64.so`) the first time a device is used.
PyTorch's ROCm wheels carry their own copies in `torch/lib` and load them when
`torch` is imported. glibc reuses an already-loaded library for a bare-name
`dlopen` only when the name matches one the object was loaded under, its
DT_SONAME, or a file on the search path with the same inode. When none of those
hold, a second HIP and a second HSA runtime are mapped into the process, and two
copies of either cannot share one: the MAX device query fails with "Failed to
initialize HSA runtime", or the next HIP fat binary registration aborts in
`hip.cpp` ("__hipPopCallConfiguration has non-null function pointer ... first
instance of the library being copies").

Modular exposes no supported setting for the HIP library path (2026-10-26; the
ask is docs/plans/MODULAR_ASK_AMD_HIP_RUNTIME.md), so this module does the two
things a process can do on its own, before the first device use:

* No HIP runtime loaded yet (mojolearn imported before torch): load, by the
  same bare names MAX uses, the copies MAX would load. A later `import torch`
  then resolves its `libamdhip64` / `libhsa-runtime64` dependencies to these
  objects by name or DT_SONAME instead of mapping its own, and the process keeps
  one runtime.
* A HIP runtime already loaded (torch imported first): ask glibc, with
  RTLD_NOLOAD, whether MAX's bare-name `dlopen` reaches that loaded object. If it
  does, nothing to do. If it would map a second copy, refuse at import with the
  loaded path and the two setups that work, instead of failing at the first fit.

Only the device probe in `_backend` and the HIP-vendor selection call this. It
never loads a library just to look: `locate` searches the paths without opening
anything, so the probe cannot add a runtime copy of its own (until 2026-10-26 it
dlopened every `libamdhip64` candidate, which could map ROCm 7 and ROCm 6 copies
side by side).
"""
import os
import sys

#: The bare names the MAX runtime passes to dlopen, in dependency order (HIP
#: needs HSA). Changing these means MAX changed what it loads.
RUNTIME_NAMES = ("libhsa-runtime64.so", "libamdhip64.so")

#: Set to 0 to skip `prepare` (diagnosis only: with it off, the two-copy
#: failure comes back at the first device use).
ENV_SWITCH = "MOJOLEARN_ROCM_SINGLE_RUNTIME"


class RocmRuntimeConflict(ImportError):
    """torch (or another library) loaded a HIP runtime that the MAX runtime
    would not reuse, so the first device use would map a second copy."""


def _stem(name):
    # "libamdhip64.so" -> "libamdhip64"; also matches hashed wheel copies
    # such as "libamdhip64-1a2b3c4d.so.7" (auditwheel naming).
    return name.split(".so", 1)[0]


def loaded_copies():
    """{stem: [real paths]} of the ROCm runtime files mapped into this process,
    read from /proc/self/maps. Empty off Linux."""
    stems = {_stem(n) for n in RUNTIME_NAMES}  # glue: the two runtime names
    out = {s: [] for s in stems}  # glue: the two runtime names
    try:
        with open("/proc/self/maps") as fh:
            lines = fh.read().splitlines()
    except OSError:
        return out
    for line in lines:  # glue: loader loop over mapped files of this process
        parts = line.split(None, 5)
        path = parts[5] if len(parts) == 6 else ""
        if not path.startswith("/"):
            continue
        base = os.path.basename(path)
        for s in stems:  # glue: loader loop over the two runtime names
            if base == s + ".so" or base.startswith(s + ".so.") or base.startswith(s + "-"):
                real = os.path.realpath(path)
                if real not in out[s]:
                    out[s].append(real)
    return out


def _conf_dirs(path, seen):
    """Directories listed in an ld.so.conf file, following `include` globs."""
    import glob
    dirs = []
    if path in seen:
        return dirs
    seen.add(path)
    try:
        with open(path) as fh:
            lines = fh.read().splitlines()
    except OSError:
        return dirs
    for raw in lines:  # glue: loader loop over ld.so.conf lines
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        if line.startswith("include"):
            for inc in sorted(glob.glob(line.split(None, 1)[1].strip())):  # glue: loader loop over conf includes
                dirs += _conf_dirs(inc, seen)
        else:
            dirs.append(line)
    return dirs


def search_dirs():
    """The directories a bare-name dlopen searches, in glibc's order, without
    the caller's RPATH/RUNPATH: LD_LIBRARY_PATH, ld.so.conf, the defaults."""
    dirs = [d for d in os.environ.get("LD_LIBRARY_PATH", "").split(":") if d]  # glue: library path entries
    dirs += _conf_dirs("/etc/ld.so.conf", set())
    dirs += ["/lib64", "/usr/lib64", "/lib", "/usr/lib",
             "/lib/x86_64-linux-gnu", "/usr/lib/x86_64-linux-gnu"]
    seen, out = set(), []
    for d in dirs:  # glue: loader loop over search directories
        if d not in seen:
            seen.add(d)
            out.append(d)
    return out


def locate(name):
    """The file a bare-name dlopen of `name` would most likely open, or None.
    Opens nothing. A copy already mapped into the process counts as found."""
    mapped = loaded_copies().get(_stem(name))
    for d in search_dirs():  # glue: loader loop over search directories
        p = os.path.join(d, name)
        if os.path.exists(p):
            return os.path.realpath(p)
    if mapped:
        return mapped[0]
    return None


def _reaches_loaded(name):
    """True when dlopen(name) resolves to an object already in the process
    (RTLD_NOLOAD: glibc answers without loading anything)."""
    import ctypes
    try:
        ctypes.CDLL(name, mode=os.RTLD_NOLOAD | os.RTLD_LAZY)
        return True
    except OSError:
        return False


def prepare():
    """Make the MAX runtime and any other HIP user in this process share one
    HIP and one HSA runtime. Called once, when the HIP set is selected and
    before any HIP binding loads. Raises RocmRuntimeConflict when a runtime
    that MAX would not reuse is already loaded."""
    if not sys.platform.startswith("linux"):
        return
    if os.environ.get(ENV_SWITCH, "1").strip().lower() in ("0", "false", "no", "off"):
        return
    import ctypes
    copies = loaded_copies()
    conflicts = []
    for name in RUNTIME_NAMES:  # glue: loader loop over the two runtime names
        have = copies[_stem(name)]
        if not have:
            # Nothing loaded yet: load what MAX will load, under the name MAX
            # will ask for, so a later torch import reuses it. A missing file
            # is left to MAX's own error at the first device use.
            try:
                ctypes.CDLL(name)
            except OSError:
                pass
            continue
        if not _reaches_loaded(name):
            conflicts.append((name, have, locate(name)))
    if conflicts:
        raise RocmRuntimeConflict(conflict_message(conflicts))


def conflict_message(conflicts):
    lines = ["mojolearn: this process already has a ROCm runtime loaded (usually by "
             "`import torch`) that the MAX runtime would not reuse; its first GPU call "
             "would load a second copy, and two HIP or HSA runtimes cannot share a process."]
    for name, have, would in conflicts:  # glue: loader loop over at most two runtime names
        lines.append(f"  {name}: loaded {', '.join(have)}; MAX would open {would or 'nothing found'}")
    lib_dir = os.path.dirname(conflicts[0][1][0])
    lines += ["Either of these works:",
              "  1. import mojolearn before torch, so torch reuses the runtime mojolearn loads; or",
              f"  2. start Python with LD_LIBRARY_PATH={lib_dir}:$LD_LIBRARY_PATH so both "
              "use the copy already loaded.",
              f"{ENV_SWITCH}=0 skips this check (the second copy then fails at the first fit)."]
    return "\n".join(lines)
