# SPDX-License-Identifier: Apache-2.0
"""Known allocator diagnostics the Mojo runtime prints on its own.

The Mojo runtime's tcmalloc is built NUMA-aware. In a container that does not
permit NUMA memory policy (seccomp/CAP_SYS_NICE; seen on the RunPod L4 pod) its
`mbind` fails with EPERM and tcmalloc writes one warning line to stderr per
process, e.g.::

    127075 external/tcmalloc+/tcmalloc/internal/system_allocator.h:713] Warning:
    Unable to mbind memory (errno=1, base=0x70f1c0000000, nodemask=1)

That line is the toolchain's, not mojolearn's (the ask is recorded for
Modular). Tests that require an empty stderr drop exactly this line and keep
every other byte, so our own output is still asserted in full.
"""
import re

_TCMALLOC_MBIND = re.compile(
    r"^\d+ external/tcmalloc\+?/tcmalloc/internal/system_allocator\.h:\d+\] "
    r"Warning: Unable to mbind memory \(errno=1, base=0x[0-9a-fA-F]+, nodemask=\d+\)\n?$")


def product_stderr(text):
    """`text` without the tcmalloc NUMA mbind warning lines."""
    return "".join(line for line in text.splitlines(keepends=True)
                   if not _TCMALLOC_MBIND.match(line))
