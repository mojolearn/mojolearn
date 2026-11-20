# SPDX-License-Identifier: Apache-2.0
"""Refuse, at import and with a clear ImportError, a host the compiled files cannot run on.

A compiled binding that executes an instruction the CPU lacks dies by SIGILL, which kills the process with no
Python traceback (the 0.3.0 Linux wheel did exactly that on an EPYC without AVX-512). The Linux files are built
for x86-64-v3 (packaging/linux/build_sets.sh pins `--target-cpu x86-64-v3` and refuses a set with anything above
it), so a CPU without those features is refused here, before any binding loads. The interpreter floor is the
package's `Requires-Python`; an older interpreter that reached this file anyway (a copied site-packages) is
refused the same way. Runs once per process at `import mojolearn`; reads the first `flags` line of
/proc/cpuinfo and nothing else. An unreadable /proc/cpuinfo is not a refusal (containers may hide it).
"""
import sys

#: x86-64-v3 as /proc/cpuinfo names it (`abm` carries LZCNT, `pni` is SSE3, `bmi1` is BMI)
_X86_64_V3 = ("avx", "avx2", "bmi1", "bmi2", "f16c", "fma", "abm", "movbe", "xsave", "popcnt", "sse4_1", "sse4_2",
              "ssse3", "pni", "cx16")
_MIN_PYTHON = (3, 10)


def check():
    if sys.version_info[:2] < _MIN_PYTHON:
        raise ImportError(f"Mojolearn needs Python {_MIN_PYTHON[0]}.{_MIN_PYTHON[1]} or newer; this is "
                          f"{sys.version_info[0]}.{sys.version_info[1]}")
    if not sys.platform.startswith("linux"):
        return
    import os
    if os.uname().machine.lower() not in ("x86_64", "amd64"):
        return
    try:
        with open("/proc/cpuinfo", encoding="ascii", errors="replace") as fh:
            for line in fh:
                if line.startswith("flags"):
                    flags = set(line.partition(":")[2].split())
                    break
            else:
                return
    except OSError:
        return
    missing = [f for f in _X86_64_V3 if f not in flags]
    if missing:
        raise ImportError("Mojolearn's Linux files are built for x86-64-v3 CPUs (AVX2, FMA, BMI2, F16C, MOVBE); "
                          "this CPU lacks " + ", ".join(missing) + ". Running them would stop the process with an "
                          "illegal instruction, so the import is refused.")
