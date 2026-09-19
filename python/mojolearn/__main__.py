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
    python -m mojolearn identity         run the identity_break lanes on
                                         this box and diff them against the
                                         three GPU columns shipped in the
                                         wheel; --check resolves the files
                                         and runs nothing (_identity.py)

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
import json
from pathlib import Path
import sys

from . import _conformance
from . import _identity
from . import _verify
from . import _verify_all


def _wants_suite(args):
    """`verify` runs the identity suite (`_verify_all.py`) when any of its
    flags is given, else the pinned k-means card as before."""
    return bool(getattr(args, "all", False) or getattr(args, "quick", False)
                or getattr(args, "full", False) or getattr(args, "lanes", "")
                or getattr(args, "emit_models", None)
                or getattr(args, "self_test", False)
                or getattr(args, "cross_check", None)
                or getattr(args, "compare", None)
                # sealing a document is a pure function over one JSON file, the
                # same as comparing two, and must not fall through to the
                # pinned k-means card, which needs a reference this party has
                # no reason to own
                or getattr(args, "commitment", None)
                or getattr(args, "commitment_a", None)
                or getattr(args, "commitment_b", None)
                or getattr(args, "coverage", False)
                or getattr(args, "include_pending", False)
                or getattr(args, "models_only", False)
                or getattr(args, "training_only", False)
                or getattr(args, "batch_checks", False))


def _verify_dispatch(args):
    if getattr(args, "training_only", False):
        args.no_models = True
    if _wants_suite(args):
        return _verify_all.cmd_verify_all(args)
    return _verify.cmd_verify(args)


def _causal_lm_dispatch(args):
    from . import _verify_causal_lm as proof
    if args.compare:
        left, right = (json.loads(Path(p).read_text()) for p in args.compare)
        equal = proof.compare(left, right)
        print(json.dumps({'status': 'NUMERICAL_MATCH_UNQUALIFIED' if equal else 'DIVERGENT',
                          'release_qualified': False}))
        return 0 if equal else 1
    path = Path(args.output)
    if path.exists():
        raise ValueError('capture output already exists; choose a new path to preserve evidence')
    options = {'layer_devices': args.layer_devices} if args.layer_devices is not None else {}
    result = proof.capture(args.device, tuple(args.formats), **options)
    with path.open('x') as stream:
        json.dump(result, stream, indent=2)
        stream.write('\n')
    print(json.dumps({'status': result['status'], 'output': str(path), 'release_qualified': False}))
    return 0 if result['status'] == 'CAPTURED_UNQUALIFIED' else 1


def _distributed_dispatch(args):
    from ._verify_distributed import main as distributed_main
    if args.compare:
        argv = ['--compare', *args.compare]
        if args.devices or args.out or args.require_installed:
            raise ValueError('--compare cannot be combined with capture options')
    else:
        argv = []
        if args.devices is not None:
            argv += ['--devices', args.devices]
        if args.out is not None:
            argv += ['--out', args.out]
        if args.require_installed:
            argv += ['--require-installed']
    return distributed_main(argv)


def _cross_validation_dispatch(args):
    from ._verify_parallel_cv import main as cv_main
    if args.compare:
        if args.devices or args.out or args.require_installed or args.require_backend:
            raise ValueError('--compare cannot be combined with capture options')
        argv = ['--compare', *args.compare]
    else:
        argv = []
        for flag, value in (('--devices', args.devices), ('--out', args.out),
                            ('--require-backend', args.require_backend)):
            if value is not None:
                argv += [flag, value]
        if args.require_installed:
            argv += ['--require-installed']
    return cv_main(argv)


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

    cv = sub.add_parser('verify-cross-validation',
        help='checkpoint small GPU cross-validation scheduling checks')
    cv.add_argument('--devices', help='two distinct GPU indices, e.g. 0,1')
    cv.add_argument('--out', metavar='DIRECTORY', help='new evidence directory')
    cv.add_argument('--require-installed', action='store_true')
    cv.add_argument('--require-backend', choices=('cuda', 'hip'))
    cv.add_argument('--compare', nargs=2, metavar=('LEFT', 'RIGHT'))
    cv.add_argument('--cpu-threads', type=int, default=1)
    cv.set_defaults(func=_cross_validation_dispatch)

    distributed = sub.add_parser('verify-distributed',
        help='checkpoint small two-GPU forecast, classifier and sharded-index checks',
        description='Numerical and device-placement checks with transport fault controls. '
                    'Independent GPU execution traces and release qualification remain separate.')
    distributed.add_argument('--devices', help='two distinct GPU indices, e.g. 0,1')
    distributed.add_argument('--out', metavar='PATH', help='new checkpoint JSON path')
    distributed.add_argument('--require-installed', action='store_true',
                             help='require package and native bytes to match installed wheel RECORD')
    distributed.add_argument('--compare', nargs=2, metavar=('LEFT', 'RIGHT'))
    distributed.add_argument('--cpu-threads', type=int, default=1)
    distributed.set_defaults(func=_distributed_dispatch)

    lm = sub.add_parser('verify-causal-lm',
        help='capture tiny loaded-model inference properties or compare two captures',
        description='End-to-end loaded-model logits, stateful decode, batch, reset and reload checks. '
                    'A successful capture or numerical comparison is not release qualification.')
    action = lm.add_mutually_exclusive_group(required=True)
    action.add_argument('--output', metavar='PATH', help='write a fresh capture; never overwrite')
    action.add_argument('--compare', nargs=2, metavar=('LEFT', 'RIGHT'),
                        help='compare matching captures without executing models')
    lm.add_argument('--device', choices=('cpu', 'gpu'), default='cpu')
    lm.add_argument('--formats', nargs='+', choices=('float32', 'bfloat16', 'int8'),
                    default=['float32', 'bfloat16', 'int8'])
    lm.add_argument('--cpu-threads', type=int, default=1)
    lm.add_argument('--layer-devices', nargs='+', type=int,
                    help='experimental GPU layer-owner map, one device index per fixture layer')
    lm.set_defaults(func=_causal_lm_dispatch)

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
                   help="check every identity cell this install can run: the "
                        "identity_break lanes (every lane on a GPU install, the "
                        "public CPU reference lanes on a CPU-only one) plus the "
                        "portable GPU-trained models, each cell part compared "
                        "with the reference table shipped in the wheel "
                        "(docs/VERIFY.md, python/mojolearn/_verify_all.py)")
    v.add_argument("--batch-checks", action="store_true",
                   help="also run gradient, batch-size, ragged and sampler/replay probes; missing references read OWED")
    v.add_argument("--coverage", action="store_true",
                   help="inspect all appendix variants, lane availability and batch contracts without fitting")
    v.add_argument("--include-pending", action="store_true",
                   help="also execute unqualified CPU routes and supported logical-shard drivers; stale references read OWED, and missing routes remain scope gaps")
    scope = v.add_mutually_exclusive_group()
    scope.add_argument("--models-only", "--inference", dest="models_only", action="store_true",
                   help="check bundled GPU-trained models through the saved-model loader, including HostForest and HostGBDT, without training")
    scope.add_argument("--training", dest="training_only", action="store_true",
                   help="run fit-based algorithm verification and learned-model properties, excluding the separate bundled-model suite; narrow with --lanes and --fixtures")
    v.add_argument("--cpu-threads", type=int, default=1, metavar="N",
                   help="thread setting for supported CPU libraries (default: 1); capped below the available logical CPU count where possible; not a hard CPU or memory limit")
    v.add_argument("--quick", action="store_true",
                   help="implies --all: one lane per family on the base "
                        "fixture")
    v.add_argument("--full", action="store_true",
                   help="implies --all: every lane on every fixture (the "
                        "default depth of --all)")
    v.add_argument("--lanes", default="",
                   help="implies --all: only these comma separated lanes")
    v.add_argument("--fixtures", default="",
                   help="with --all: only these comma separated fixtures")
    v.add_argument("--repeats", type=int, default=1,
                   help="with --all: fits per cell; two or more also catch a "
                        "cell that moves on this box (default %(default)s)")
    v.add_argument("--no-models", dest="no_models", action="store_true",
                   help="with --all: skip the portable models")
    v.add_argument("--self-test", dest="self_test", action="store_true",
                   help="SHOW THAT THIS VERIFIER CAN FAIL. Runs one lane twice "
                        "through the ordinary comparison, once untouched and "
                        "once with every value of the input's first column "
                        "moved up by one ULP, and requires the first to read "
                        "IDENTICAL and the second DIVERGENT. The perturbation "
                        "is real arithmetic at run time, not a printed verdict, "
                        "and it needs no sabotage build. Exit 0 only if the "
                        "comparison both reproduced the reference and caught "
                        "the wrong answer")
    v.add_argument("--cross-check", dest="cross_check", nargs="?", const="default",
                   choices=("quick", "default", "all"), default=None,
                   help="COMPARE YOUR GPU AGAINST YOUR CPU, on this machine. Fits "
                        "each lane once on the GPU, then asks the same fitted "
                        "model for the same held-out answer twice: from the GPU "
                        "estimator, and from the saved model reloaded through the "
                        "CPU host binding. It requires trusting nobody, because "
                        "you generated both sides on two different pieces of "
                        "hardware. It compares the infer part (cross-vendor "
                        "identity) and, where the lane has one, the batch part "
                        "(batch invariance, a different axis). 'quick' is one "
                        "lane per family on the base fixture, seconds; the "
                        "default is up to 24 lanes, minutes, capped because one "
                        "Apple Metal process may not run a full column outside "
                        "a release; 'all' is the whole 79-lane intersection and "
                        "is refused on Apple by that same rule. --lanes and "
                        "--fixtures widen or narrow any of them. On a CPU-only "
                        "install it says so rather than silently skipping")
    v.add_argument("--compare", nargs=2, metavar=("A", "B"), default=None,
                   help="DIFF TWO EVIDENCE DOCUMENTS, with us out of the loop. "
                        "Two people on different hardware each run "
                        "`verify --all --json-out mine.json`, swap files, and "
                        "run this: it compares every cell hash, reports where "
                        "they agree and differ, shows the two provenance blocks "
                        "side by side, and says whether the machines were "
                        "genuinely different. A cell present in only one "
                        "document is INCOMPARABLE, never a match. Needs no GPU, "
                        "no bindings and no network")
    v.add_argument("--commitment", metavar="DOC", default=None,
                   help="COMMIT TO YOUR OWN EVIDENCE DOCUMENT BEFORE YOU SEE "
                        "THEIRS. Writes a random nonce into DOC and prints a "
                        "64-character commitment over the document's cells, "
                        "its provenance block, its verification contract, its "
                        "binding digests and its own verdict. Publish that "
                        "line anywhere, by any means, BEFORE the two parties "
                        "exchange documents; then exchange them, nonces "
                        "included, and pass both lines back to --compare. "
                        "Without this step nothing stops whoever receives the "
                        "other file first from pasting its numbers into a "
                        "document carrying their own hardware. Adds no network "
                        "code and needs no GPU, no bindings and no repo")
    v.add_argument("--commitment-a", dest="commitment_a", metavar="C", default=None,
                   help="with --compare: the commitment published for the FIRST "
                        "document before the exchange, as the 64-character line "
                        "itself or a path to the .commitment file. A comparison "
                        "without commitments is not an error; it is labelled a "
                        "weaker result")
    v.add_argument("--commitment-b", dest="commitment_b", metavar="C", default=None,
                   help="with --compare: the same, for the SECOND document")
    v.add_argument("--json-out", dest="json_out", metavar="PATH", default=None,
                   help="with --all: also write the full evidence document "
                        "(per-cell hashes computed here and expected, per-lane "
                        "timings, the sha256 of every binding loaded, and the "
                        "committed column each reference came from) to PATH")
    v.add_argument("--reference-table", dest="reference_table", metavar="PATH",
                   default=None,
                   help="with --all: compare against this table instead of "
                        "the one shipped in the wheel; with --emit-reference "
                        "and --lanes, update only those lanes in this base table")
    v.add_argument("--records", action="append", metavar="DIR", default=None,
                   help="MAINTAINER PATH, with --all --emit-reference: the "
                        "identity_break record directories or JSONs to build "
                        "the table from (default: the checkout's "
                        "bench/results/identity_break)")
    v.add_argument("--emit-models", dest="emit_models", metavar="DIR",
                   default=None,
                   help="MAINTAINER PATH, GPU install: save the portable "
                        "models and their manifest to DIR, keeping only "
                        "models whose file bytes equal the table's model "
                        "reference")
    v.add_argument("--all-stages", dest="all_stages", action="store_true",
                   help="without --all: on a mismatch of the pinned k-means "
                        "card, list every diverging stage rather than only "
                        "the first")
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
    v.set_defaults(func=_verify_dispatch)

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
    d = sub.add_parser(
        "identity",
        help="run the identity_break lanes on this box and diff them "
             "against the three GPU columns shipped in this install",
        description=(
            "Runs tools/identity_break.py (the wheel's copy, or the "
            "checkout's) on this box under the identical tier, then diffs "
            "the column it wrote against the three training GPU columns the "
            "manifest names (Apple M4, NVIDIA H100, AMD MI300X at the "
            "recorded commit), requiring IDENTICAL x4 on every cell this box "
            "ran. On a CPU-only install only the lanes with a CPU training "
            "path run. Needs numpy. The whole record takes minutes to an "
            "hour depending on the box; --fixtures base --repeats 1 is the "
            "quick pass and is judged cell by cell rather than through "
            "--require-columns."),
        formatter_class=argparse.RawDescriptionHelpFormatter)
    d.add_argument("--check", action="store_true",
                   help="resolve the harness, the columns and the commit "
                        "witness and run nothing (exit 0 ready, 5 missing)")
    d.add_argument("--lanes", default="",
                   help="comma separated subset of the record's lanes "
                        "(default: every lane this box can run)")
    d.add_argument("--fixtures", default="",
                   help="comma separated subset of the record's fixtures "
                        "(default: all nine)")
    d.add_argument("--repeats", type=int, default=1,
                   help="fits per cell (default %(default)s); a hash equal to "
                        "another box's is already stable, so raise it only to "
                        "classify a cell that diverged")
    d.add_argument("--vendor", default=None,
                   help="the box label written into the local column "
                        "(default: the harness's own, cpu-<model> or the "
                        "machine architecture)")
    d.add_argument("--keep", metavar="PATH", default=None,
                   help="write the local column here instead of a temporary "
                        "directory (it is always kept)")
    d.add_argument("--json", action="store_true",
                   help="emit one JSON object as the verdict instead of the "
                        "human line (the harness's own output still streams)")
    d.set_defaults(func=_identity.cmd_identity)

    c.set_defaults(func=lambda _args: (c.print_help(),
                                       _verify.EXIT_USAGE)[1])

    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    # kept so `verify --all` with no numeric mode chosen can run itself again
    # under the identical tier with the same arguments
    args.argv = list(sys.argv[1:] if argv is None else argv)
    if getattr(args, "func", None) is None:
        # NO DEFAULT SUBCOMMAND. `python -m mojolearn` with no argument must
        # not quietly run the check: a bare invocation that fits on a GPU is
        # a surprise, and a bare invocation that prints a verdict nobody
        # asked for is how a green line ends up quoted out of context.
        parser.print_help()
        return _verify.EXIT_USAGE
    from ._verify_resources import run_with_budget
    try:
        resource_exit = run_with_budget(args)
    except ValueError as exc:
        parser.error(str(exc))
    if resource_exit is not None:
        return resource_exit
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
