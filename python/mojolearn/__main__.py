# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""`python -m mojolearn <subcommand>`.

    python -m mojolearn verify           check this build against the
                                         reference card shipped in the wheel
    python -m mojolearn verify --json    the same, machine readable
    python -m mojolearn env              what this process loaded, no GPU
    python -m mojolearn check-fixture    rebuild and hash the pinned fixture,
                                         no GPU and no extension call

The logic is in `_verify.py`; this file is argument parsing and nothing else,
so a new subcommand is a parser entry and a function rather than a rewrite.

WHY NOT A CONSOLE SCRIPT AS THE PRIMARY SPELLING. `python -m mojolearn` needs
no entry point, works from a source checkout with PYTHONPATH set, and cannot
be shadowed by a stale script on PATH from an earlier install. The
distribution already installs a `mojolearn` console script pointing at
`mojolearn_diagnostics`, which deliberately lives OUTSIDE this package so it
still runs when importing the extensions is the thing being diagnosed. This
module is the opposite case and must import them, so the two are separate on
purpose.

EXIT CODES, because people put this in continuous integration.

    0  VERIFIED       the card matched the reference, stage for stage
    1  MISMATCH       the fit ran, the card differs, a stage is named
    2  USAGE          bad arguments
    3  REFUSED        this process loaded the FAST binaries, which make no
                      identity claim; nothing was judged
    4  CANNOT RUN     no GPU, no identical binaries, the fit raised, the
                      trace never reached the binary, or no comparator
    5  NO REFERENCE   this install ships no usable reference card, or the
                      one it ships is still the placeholder

docs/VERIFY.md is the human document for all of it.
"""

import argparse
import sys

from . import _verify


def build_parser():
    parser = argparse.ArgumentParser(
        prog="python -m mojolearn",
        description=(
            "Check mojolearn's cross-vendor bitwise identity claim on this "
            "machine. See docs/VERIFY.md for what a local run does and does "
            "not prove."),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "exit codes\n"
            "  0 verified   1 mismatch   2 usage   3 refused (fast build)\n"
            "  4 cannot run 5 no reference\n"),
    )
    sub = parser.add_subparsers(dest="command", metavar="<subcommand>")

    v = sub.add_parser(
        "verify",
        help="run the pinned fixture and compare its stage card to the "
             "reference shipped in this install",
        description=(
            "Runs one pinned k-means fit with the identity trace enabled, "
            "then compares the stage card it emits against the reference "
            "card in mojolearn/reference_cards/ using "
            "tools/identity_trace_diff.py. Refuses unless "
            "MOJOLEARN_NUMERIC_MODE=identical was set before import, because "
            "the FAST arm makes no cross-vendor claim."),
        formatter_class=argparse.RawDescriptionHelpFormatter)
    v.add_argument("--json", action="store_true",
                   help="emit one JSON object instead of the human report")
    v.add_argument("--all", action="store_true",
                   help="on a mismatch, list every diverging stage rather "
                        "than only the first")
    v.add_argument("--keep", action="store_true",
                   help="keep the card this run produced even when it "
                        "matched (a mismatched card is always kept)")
    v.add_argument("--reference-name", dest="reference_name",
                   default=_verify.REFERENCE_NAME,
                   help="which reference card in reference_cards/ to compare "
                        "against (default %(default)s)")
    v.add_argument("--emit-reference", metavar="PATH", default=None,
                   help="MAINTAINER PATH. Run the fixture and write a "
                        "provenance-stamped candidate reference card to "
                        "PATH instead of comparing. Refuses on a FAST build.")
    v.set_defaults(func=_verify.cmd_verify)

    e = sub.add_parser(
        "env",
        help="print what this process loaded, without touching the GPU",
        description=(
            "The block `verify` prints above its verdict: version, numeric "
            "mode and how it was established, the extension that would run "
            "the fixture, host, device and commit. Nothing is fitted."))
    e.add_argument("--json", action="store_true")
    e.set_defaults(func=_verify.cmd_env)

    f = sub.add_parser(
        "check-fixture",
        help="rebuild the pinned fixture and check it against the input "
             "hashes three vendors recorded",
        description=(
            "No GPU, no extension call. Rebuilds the E1U k-means fixture in "
            "numpy and hashes it, then compares against the input.x and "
            "input.centroids values every E1U leg printed. Run this first "
            "after any edit to the fixture generator: a card comparison "
            "against different input bytes measures nothing."))
    f.add_argument("--json", action="store_true")
    f.set_defaults(func=_verify.cmd_check_fixture)

    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    if getattr(args, "func", None) is None:
        # NO DEFAULT SUBCOMMAND. `python -m mojolearn` with no argument must
        # not quietly run the check: a bare invocation that fits on a GPU is
        # a surprise, and a bare invocation that prints a verdict nobody
        # asked for is how a green line ends up quoted out of context.
        parser.print_help()
        return _verify.EXIT_USAGE
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
