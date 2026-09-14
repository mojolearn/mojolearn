# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The shared shape of the workstream D surface tests (2026-09-14): run as
modules, never by path (`cd python && python3 -m mojolearn.tests.<name>`),
exit 2 by name when the binding is unbuilt, assert bits only under the
identical tier and REPORT them otherwise ([[fast-is-not-identical]]).

`test_gp_surface.py`'s Report, reduced to what six sibling files share.
"""
import sys

from mojolearn import _backend


class Report:
    def __init__(self, name):
        self.name = name
        self.rows = []
        self.failures = []

    def check(self, arm, cond, what, detail=""):
        ok = bool(cond)
        self.rows.append((ok, arm, what + ("" if ok or not detail else " -- " + str(detail))))
        if not ok:
            self.failures.append((arm, what))
        return ok

    def report_only(self, arm, cond, what):
        self.rows.append((True, arm, "[REPORT, not asserted] %s: %s" % (what, "same" if cond else "MOVED")))
        return bool(cond)

    def raises(self, arm, exc_types, needle, what, fn, *a, **kw):
        try:
            fn(*a, **kw)
        except exc_types as exc:
            return self.check(arm, needle in str(exc), what, "message: %s" % exc)
        except Exception as exc:  # noqa: BLE001
            return self.check(arm, False, what, "raised %s: %s" % (type(exc).__name__, exc))
        return self.check(arm, False, what, "ACCEPTED")

    def render(self, out=sys.stdout):
        for ok, arm, what in self.rows:
            out.write("  %s  %-10s %s\n" % ("ok  " if ok else "FAIL", arm, what))


def mode():
    return _backend.default_mode()


def bind_or_exit(binding, build_script, out=sys.stderr):
    """The binding, or exit 2 naming the build command."""
    try:
        return _backend.binding(binding)
    except ImportError as exc:
        out.write("%s is not built (%s); build it with\n    bash bindings/%s\n" % (binding, exc, build_script))
        sys.exit(2)


def run(name, arms, rep, out=sys.stdout):
    """`arms` is a list of (label, callable(rep)); an arm that raises is
    recorded as DID NOT RUN, never swallowed."""
    aborted = []
    for label, fn in arms:
        try:
            fn(rep)
        except SystemExit:
            raise
        except Exception as exc:  # noqa: BLE001
            aborted.append((label, "%s: %s" % (type(exc).__name__, exc)))
    rep.render(out)
    for label, why in aborted:
        out.write("\n  %s ARM DID NOT RUN\n    %s\n" % (label, why))
    out.write("\n")
    if rep.failures or aborted:
        out.write("%s: RED. %d checks failed, %d arms did not run.\n" % (name, len(rep.failures), len(aborted)))
        return 1
    out.write("%s: GREEN under numeric_mode=%s, %d checks. One box, one vendor; no bit is claimed for any other.\n"
              % (name, mode(), len(rep.rows)))
    return 0
