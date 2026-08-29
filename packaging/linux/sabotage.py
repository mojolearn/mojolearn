#!/usr/bin/env python3
"""THE NEGATIVE GATES for the vendor selector. Verify reach, not output.

    python3 packaging/linux/sabotage.py --pkgroot <dir> --vendor cuda|hip \\
        --python "<interpreter command>" --json <out.json>

`<dir>/mojolearn/` is a good, assembled package (the `.py` files plus this
box's `<vendor>/` set). Every case below builds a scratch copy of it with
hard links, breaks ONE thing, runs `import mojolearn` in a fresh
interpreter, and checks BOTH the outcome and the message. A selector that
happens to pick the right set on a healthy tree proves nothing; these are
the runs where picking wrong is possible and the wrong pick must be
refused BY NAME.

The cases, with `V` the box's vendor and `W` the other one:

  wrong_label_ignored   W/ is created and filled with V's own binaries (a
                        set under the wrong label). The probe finds V's
                        device, picks V/, never opens W/. Import succeeds,
                        vendor() == V.
  wrong_label_refused   the same tree with MOJOLEARN_VENDOR=W forcing the
                        selector into W/. The binaries there answer V. The
                        import must RAISE naming the mismatch. This is the
                        read-back beating the label.
  poisoned_ignored      W/ holds files named like a set but full of random
                        bytes. Import succeeds through V/; W/ was never
                        loaded.
  poisoned_forced       MOJOLEARN_VENDOR=W on that tree. The import must
                        raise (a load failure, from the loader), never
                        fall through to V/.
  correct_set_removed   V/ deleted, W/ present and well formed (V's
                        binaries under the W label, so a fallback would
                        even WORK on this box). The import must RAISE the
                        no-GPU refusal naming: the V device evidence that
                        was found, the W set that is the only one carried,
                        and MOJOLEARN_VENDOR. It must not load W/.
  tier_mislabeled       V/deterministic/_mojolearn_gbdt.so replaced by the
                        FAST binary. The existing tier read-back must
                        refuse it on the new layout too.

Each case records the exit status, the last line of stderr and whether
the expected substrings appeared. Exit 0 only if every case behaved.
"""

import argparse
import json
import os
import pathlib
import shlex
import shutil
import subprocess
import sys

OTHER = {"cuda": "hip", "hip": "cuda"}
PROBE_SNIPPET = (
    "import json, mojolearn; from mojolearn import _backend; "
    "print(json.dumps({'vendor': mojolearn.vendor(), 'how': _backend.vendor_how(), "
    "'mode': mojolearn.numeric_mode()}))"
)


def link_tree(src, dst):
    shutil.copytree(src, dst, copy_function=os.link, symlinks=True)


def run_import(python, pkgroot, env_extra):
    env = dict(os.environ)
    env["PYTHONPATH"] = str(pkgroot)
    env.pop("MOJOLEARN_VENDOR", None)
    env.update(env_extra)
    cmd = shlex.split(python) + ["-c", PROBE_SNIPPET]
    r = subprocess.run(cmd, env=env, capture_output=True, text=True,
                       cwd=str(pkgroot), timeout=600)
    err_tail = "\n".join(r.stderr.strip().splitlines()[-6:])
    out_last = r.stdout.strip().splitlines()[-1] if r.stdout.strip() else ""
    return r.returncode, out_last, err_tail


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pkgroot", required=True)
    ap.add_argument("--vendor", required=True, choices=("cuda", "hip"))
    ap.add_argument("--python", default=sys.executable)
    ap.add_argument("--json", default="")
    ap.add_argument("--scratch", default="")
    a = ap.parse_args()
    V, W = a.vendor, OTHER[a.vendor]
    good = pathlib.Path(a.pkgroot).resolve()
    scratch = pathlib.Path(a.scratch or (good.parent / "sabotage_scratch")).resolve()
    if scratch.exists():
        shutil.rmtree(scratch)
    scratch.mkdir(parents=True)
    results = {}

    def case(name, build, env_extra, expect_ok, expect_substrings=(), expect_vendor=None):
        root = scratch / name
        link_tree(good, root)
        pkg = root / "mojolearn"
        build(pkg)
        rc, out, err = run_import(a.python, root, env_extra)
        ok = (rc == 0) == expect_ok
        found = {s: (s in err or s in out) for s in expect_substrings}
        ok = ok and all(found.values())
        got_vendor = None
        if rc == 0 and out:
            try:
                got_vendor = json.loads(out).get("vendor")
            except ValueError:
                pass
        if expect_vendor is not None:
            ok = ok and got_vendor == expect_vendor
        results[name] = {"pass": ok, "returncode": rc, "expected_import_ok": expect_ok,
                         "stdout": out, "stderr_tail": err, "substrings": found,
                         "vendor_reported": got_vendor}
        last = out or (err.splitlines()[-1] if err else "")
        print(f"  {name:<22} {'PASS' if ok else 'FAIL'}  rc={rc}  {last}"[:200])

    def fill_w_with_v(pkg):
        link_tree(pkg / V, pkg / W)

    def poison_w(pkg):
        (pkg / W / "deterministic").mkdir(parents=True)
        (pkg / W / "identical").mkdir(parents=True)
        for so in (pkg / V).glob("_mojolearn*.so"):
            for d in (pkg / W, pkg / W / "deterministic", pkg / W / "identical"):
                (d / so.name).write_bytes(os.urandom(4096))

    def remove_v(pkg):
        link_tree(pkg / V, pkg / W)
        shutil.rmtree(pkg / V)

    def mislabel_tier(pkg):
        t = pkg / V / "deterministic" / "_mojolearn_gbdt.so"
        t.unlink()
        os.link(pkg / V / "_mojolearn_gbdt.so", t)

    case("baseline", lambda pkg: None, {}, True, expect_vendor=V)
    case("wrong_label_ignored", fill_w_with_v, {}, True, expect_vendor=V)
    case("wrong_label_refused", fill_w_with_v, {"MOJOLEARN_VENDOR": W}, False,
         (f"compiled for {V}", "wrong vendor directory"))
    case("poisoned_ignored", poison_w, {}, True, expect_vendor=V)
    case("poisoned_forced", poison_w, {"MOJOLEARN_VENDOR": W}, False)
    case("correct_set_removed", remove_v, {}, False,
         ("NO SUPPORTED GPU FOUND", f"['{W}']", "MOJOLEARN_VENDOR"))
    case("tier_mislabeled", mislabel_tier, {"MOJOLEARN_NUMERIC_MODE": "deterministic"},
         False, ("compiled fast",))

    allok = all(r["pass"] for r in results.values())
    if a.json:
        pathlib.Path(a.json).write_text(json.dumps(
            {"vendor": V, "other": W, "python": a.python, "cases": results,
             "ok": allok}, indent=2, sort_keys=True))
    print("sabotage", "PASS" if allok else "FAIL")
    shutil.rmtree(scratch, ignore_errors=True)
    return 0 if allok else 1


if __name__ == "__main__":
    sys.exit(main())
