#!/usr/bin/env python3
"""The defines the build refuses as removed, read from the guard modules.

A deleted experiment's define keeps a refusal in its family guard module
(`core/experiment_guards_<family>.mojo`, or `core/six_lane_experiment_guards.mojo`
for shared ones): `comptime assert not is_defined["NAME"](), "removed ..."`. Old
grid plans (experiments/six_lane_integration/grid/*.json) still name such defines;
they are history and stay as they are. A tool that turns those plans into builds
or queue lines asks this module first, so it skips (and reports) a removed define
instead of compiling it into the guard's refusal.

Planning only: a text scan of the guard modules, no compile, no import of the
package.
"""
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SHARED = 'core/six_lane_experiment_guards.mojo'
FAMILY_GLOB = 'core/experiment_guards_*.mojo'

# comptime assert not is_defined["NAME"](), "..." (one refusal per define: "removed ...", "promoted ...", every
# message the build stops on); the two-define conflict checks `not (A and B)` are not refusals of one define
_REFUSAL = re.compile(r'comptime\s+assert\s+not\s+is_defined\["(MOJOLEARN_[A-Z0-9_]+)"\]\(\)\s*,')


def guard_files(root=ROOT):
    root = Path(root)
    files = sorted(root.glob(FAMILY_GLOB))
    shared = root / SHARED
    if shared.is_file():
        files.append(shared)
    return files


def removed_defines(root=ROOT):
    """The set of define names (no value) the guard modules refuse outright (removed or promoted)."""
    names = set()
    for path in guard_files(root):
        names.update(_REFUSAL.findall(path.read_text(errors='replace')))
    return names


def define_name(define):
    """'-D MOJOLEARN_X=1', 'MOJOLEARN_X=1' or 'MOJOLEARN_X' -> 'MOJOLEARN_X'."""
    d = define.strip()
    if d.startswith('-D'):
        d = d[2:].strip()
    return d.split('=', 1)[0]


def removed_in(defines, removed):
    """The sorted removed names among `defines` (any of the spellings above)."""
    return sorted({define_name(d) for d in defines or []} & removed)


if __name__ == '__main__':
    for name in sorted(removed_defines()):
        print(name)
