#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Print the queue lines that A/B one build-time define across synthetic shapes.

    afc_shape_sweep.py --define MOJOLEARN_X --lane L --binding B --family F
        [--legacy] [--branch BR] [--shapes s-r200k-f50,...] [--box apple]

A FAST speed claim needs evidence on neighboring shapes and on data that is
not on the board (no benchmark-shape tuning, Oct 4). Each printed line is one
`lq add <box> CMD <branch> <tag> ...` job running tools/afc_ab_def.sh on one
shape of tools/afc_shape_data.py, one run per arm; the default spread is
features {50, 400, 1000} x rows {200k, 1M}. Every line names the same
AFC_DEF_BUILD_TAG, so the two arm builds are made once and reused by the
other shapes while the branch head is unchanged.

Arms: A = default build (""), B = "-D <define>". Without --legacy the define
is the candidate (B is the new route, keep it when B beats A on the spread
with equal quality). With --legacy the define restores an old narrow window
(MOJOLEARN_LEGACY_NARROW_<X>): A is the general rule under test and B the old
window, and the general rule holds when A is no slower than B on the spread.
"""
import argparse
import hashlib
import os
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import afc_shape_data as SD   # noqa: E402


def _slug(define):
    d = re.sub(r"^MOJOLEARN_", "", define.split("=")[0]).lower().replace("_", "-")
    return d[:16].strip("-") + "-" + hashlib.sha1(define.encode()).hexdigest()[:4]


def _branch():
    try:
        return subprocess.run(["git", "-C", HERE, "rev-parse", "--abbrev-ref", "HEAD"],
                              capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "<branch>"


def lines(define, lane, binding, family, legacy=False, branch=None, shapes=None, box="apple"):
    branch = branch or _branch()
    if shapes is None:
        shapes = [SD.shape_name(r, f) for r, f in SD.SPREAD]
    blk, why = SD.lane_block(family, lane, shapes[0])
    if blk is None:
        raise SystemExit("REFUSED: %s lane %s cannot take synthetic shapes: %s" % (family, lane, why))
    kind = "ssl" if legacy else "ss"
    build_tag = "%s-%s-%s" % (kind, lane, _slug(define))
    out = ["# %s A/B of -D %s, lane=%s family=%s binding=%s block=%s; %s"
           % ("LEGACY_NARROW" if legacy else "candidate", define, lane, family, binding, blk,
              "A = general rule (default) under test, B = old window"
              if legacy else "A = default, B = candidate define")]
    for s in shapes:
        rs = SD.parse_shape(s)
        if rs is None:
            raise SystemExit("REFUSED: %r is not a shape name" % s)
        if not SD.fits(*rs):
            out.append("# SKIPPED %s: over the 4 GiB cap" % s)
            continue
        tag = "%s-%s" % (build_tag, s[2:])
        out.append("lq add %s CMD %s %s 'AFC_FAMILY=%s AFC_DEF_BUILD_TAG=%s bash tools/afc_ab_def.sh "
                   "%s %s %s %s 1 1 \"\" \"-D %s\"'"
                   % (box, branch, tag, family, build_tag, tag, binding, lane, s, define))
    return out


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--define", required=True, help="the -D name (or NAME=VALUE)")
    p.add_argument("--lane", required=True)
    p.add_argument("--binding", required=True, help="afc_ab_def.sh binding (base, x_linear, auto, ...)")
    p.add_argument("--family", required=True, choices=sorted(SD.DRIVER))
    p.add_argument("--legacy", action="store_true", help="the define restores an old narrow window")
    p.add_argument("--branch", help="default: this checkout's branch")
    p.add_argument("--shapes", help="comma list of s-r<rows>-f<features> (default: the 6-shape spread)")
    p.add_argument("--box", default="apple")
    a = p.parse_args(argv)
    shapes = [s for s in a.shapes.split(",") if s] if a.shapes else None
    print("\n".join(lines(a.define, a.lane, a.binding, a.family, a.legacy, a.branch, shapes, a.box)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
