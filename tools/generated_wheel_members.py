"""Wheel Python members the builders generate from files outside python/.

packaging/macos/build_release_wheel.sh and packaging/linux/pack_wheel.py copy
these sources into the package under new names (tools/release_rehearsal.py
stages the same layout). None of them is tracked under python/mojolearn/
(python/.gitignore), so a wheel member is checked against its SOURCE path at
the build commit, never against python/<member>.

Keep this table in step with the copies in those three builders.
"""

from pathlib import PurePosixPath

#: Fixed member -> source path (repository-relative).
FIXED = {
    'mojolearn/_identity_break.py': 'tools/identity_break.py',
    'mojolearn/_identity_trace_diff.py': 'tools/identity_trace_diff.py',
}
#: `mojolearn/_identity_lane_<stem>.py` <- `tools/identity_lanes/<stem>.py`.
LANE_PREFIX = 'mojolearn/_identity_lane_'
LANE_SOURCE_DIR = 'tools/identity_lanes'


def source_path(member):
    """The repository-relative source of generated wheel member `member`,
    or None when `member` is not a generated Python member."""
    if member in FIXED:
        return FIXED[member]
    if member.startswith(LANE_PREFIX) and member.endswith('.py'):
        stem = member[len(LANE_PREFIX):-len('.py')]
        if stem and PurePosixPath(stem).name == stem and stem not in ('.', '..'):
            return f'{LANE_SOURCE_DIR}/{stem}.py'
    return None
