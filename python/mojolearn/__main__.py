# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""`python -m mojolearn <subcommand>`.

    python -m mojolearn verify           check this build against the
                                         reference card shipped in the wheel
    python -m mojolearn verify --json    the same, machine readable
    python -m mojolearn env              what this process loaded, no GPU
    python -m mojolearn check-fixture    rebuild and hash the pinned fixture,
                                         no GPU and no extension call
    python -m mojolearn install-reference PRODUCED CONFIRMED
                                         install an agreed pair of candidate
                                         cards as the reference; host-side,
                                         refusal-first (docs/VERIFY.md)
    python -m mojolearn conformance {export,validate,diff}
                                         the identity claim as a portable
                                         bundle an external implementation
                                         can check itself against
                                         (docs/CONFORMANCE.md)

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

from . import _conformance
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
    v.add_argument("--confirm-reference", metavar="PEER_CARD", default=None,
                   help="MAINTAINER PATH, second box. Run the fixture here, "
                        "stamp a local candidate (at --emit-reference PATH "
                        "if given), and compare it against PEER_CARD with "
                        "the one comparator. Exit 0 = the pair is ready for "
                        "install-reference.")
    v.set_defaults(func=_verify.cmd_verify)

    i = sub.add_parser(
        "install-reference",
        help="install an agreed pair of candidate cards as the shipped "
             "reference (host-side, no GPU)",
        description=(
            "Takes the producing vendor's candidate card and the confirming "
            "vendor's, refuses anything dishonest (FILL-IN token, profile "
            "mismatch, missing provenance, differing or unknown commits, a "
            "divergent pair), installs the produced card into "
            "mojolearn/reference_cards/ with the confirmation's provenance "
            "appended, removes the placeholder, and prints the filled "
            "docs/VERIFY.md provenance block for a human to paste. "
            "DEVIATION 927; the procedure is docs/VERIFY.md, 'Regenerating "
            "the reference card'."))
    i.add_argument("produced", help="the producing vendor's candidate card")
    i.add_argument("confirmed",
                   help="the confirming vendor's candidate card")
    i.add_argument("--reference-name", dest="reference_name",
                   default=_verify.REFERENCE_NAME,
                   help="install under this name in reference_cards/ "
                        "(default %(default)s)")
    i.set_defaults(func=_verify.cmd_install_reference)

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

    c = sub.add_parser(
        "conformance",
        help="export, validate and diff conformance bundles "
             "(docs/CONFORMANCE.md)",
        description=(
            "The identity claim as a portable artifact. `export` packages "
            "the pinned fixture -- frozen inputs, expected stage bytes, the "
            "identity-trace card, a SHA-256 manifest -- into a bundle an "
            "external implementation can check itself against without "
            "running Mojo or Python. `validate` structurally checks a "
            "bundle, or grades an implementation-report.json against one. "
            "`diff` localizes the first diverging stage through "
            "tools/identity_trace_diff.py, the repository's one comparator. "
            "Exit codes: 0 pass, 1 fail (named), 2 could-not-judge."))
    csub = c.add_subparsers(dest="conf_command", metavar="<subcommand>")

    ce = csub.add_parser(
        "export",
        help="produce a format v1 bundle directory",
        description=(
            "Default: run the pinned fixture on this machine (needs an "
            "identical-mode build and a GPU; refuses on FAST) with raw "
            "stage dumps, and assemble the bundle. --from-card assembles "
            "host-side from an existing candidate card whose .bin dumps "
            "sit beside it; no GPU, no extension import."))
    ce.add_argument("out", help="bundle directory to create")
    ce.add_argument("--from-card", metavar="CARD", default=None,
                    help="assemble from this card (its <card>.<seq>.<tag>"
                         ".bin dumps must sit beside it) instead of "
                         "running the fixture")
    ce.add_argument("--force", action="store_true",
                    help="write into a non-empty directory")
    ce.set_defaults(func=_conformance.cmd_export)

    cv = csub.add_parser(
        "validate",
        help="structural check of a bundle, or grade an external "
             "implementation report against it",
        description=(
            "Without --report: every listed file present and SHA-256-clean, "
            "every stage's expected bytes/digest consistent with the card "
            "and the manifest; a wrong hash names the file, a missing stage "
            "names the stage. With --report: grades an external "
            "implementation-report.json; a profile mismatch refuses before "
            "any byte is compared."))
    cv.add_argument("bundle", help="bundle directory")
    cv.add_argument("--report", metavar="REPORT.json", default=None,
                    help="an external implementation's result file "
                         "(format: docs/CONFORMANCE.md section 5)")
    cv.set_defaults(func=_conformance.cmd_validate)

    cd = csub.add_parser(
        "diff",
        help="first-divergence localization through the one comparator",
        description=(
            "Synthesizes a card from --report's per-stage FNV-1a64 column "
            "(or, with --self, from the bundle's own expected/ bytes) and "
            "hands it with the bundle's card to "
            "tools/identity_trace_diff.py. Same verdict vocabulary: "
            "RESULT: IDENTICAL / DIVERGENT; FIRST DIVERGENCE names the "
            "stage."))
    cd.add_argument("bundle", help="bundle directory")
    cd.add_argument("--report", metavar="REPORT.json", default=None,
                    help="diff the bundle's card against this report")
    cd.add_argument("--self", dest="self", action="store_true",
                    help="diff the bundle's card against its own expected/ "
                         "bytes (the corruption witness)")
    cd.add_argument("--all", action="store_true",
                    help="list every diverging stage, not only the first")
    cd.set_defaults(func=_conformance.cmd_diff)

    # `python -m mojolearn conformance` bare: help, exit 2, same reasoning
    # as the top level -- no default subcommand.
    c.set_defaults(func=lambda _args: (c.print_help(),
                                       _verify.EXIT_USAGE)[1])

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
