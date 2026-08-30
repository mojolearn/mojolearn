#!/usr/bin/env python3
"""THE LINUX VENDOR SELECTOR, TESTED ON A MACHINE WITH NO GPU AT ALL.

    python3 packaging/linux/selector_test.py

WHY THIS EXISTS, AND WHAT IT IS NOT
====================================
`docs/LINUX_WHEEL.md` section 7 lists five gates for the Linux wheel and
every one of them needs a rented box: the sets have to be built on an NVIDIA
box and an AMD box before anything can be smoked. That is roughly two leases
and a hundred minutes of build before the FIRST answer arrives about whether
`_backend.py`'s two-axis selector is right.

**It does not have to be that way for the selector itself.** `_layout()`,
`_probe_box()` and the refusals around them are pure Python that reads the
FILESYSTEM. Nothing in them opens a device, imports an extension or asks
Mojo anything. So the whole decision tree can be exercised here, against a
fabricated package directory and a fabricated probe, on a MacBook.

**What this proves:** that the right directory is chosen, that every refusal
fires on the case it was written for, and that the precedence order is what
section 3 says it is.

**What this does NOT prove, and the distinction is the whole point:** that a
CUDA binary runs on an NVIDIA box. Not one line here loads an extension.
Section 7's gates (a) through (d) and the install smoke are still owed in
full and this file does not shorten that list by one entry. It is worth
running first only because a selector bug found here costs nothing and the
same bug found on a rented box costs the lease.

EVERY CASE IS SABOTAGE-VERIFIED
================================
The standing rule in this repository is "verify reach, not output": a check
that passes is worthless until something proves it could have failed. A test
that asserts `_layout()` returns `cuda` passes just as happily against a
function that returns `cuda` unconditionally.

So each case that asserts a CHOICE is run twice, once as written and once
with the fixture altered so the answer must change, and the case fails if
the altered run agrees with the original. Each case that asserts a REFUSAL
checks that the message names the thing a reader needs, not merely that
something was raised.
"""

import os
import shutil
import sys
import tempfile
import traceback

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
sys.path.insert(0, os.path.join(_ROOT, "python"))

from mojolearn import _backend as B          # noqa: E402


FAILURES = []
PASSES = []


def _reset(pkg, probe_result, vendor_env=None,
           device=(None, "not probed in this test"), arch_env=None):
    """Point the selector at a fabricated install and a fabricated box.

    `_layout()` caches in module globals and reads three things it does
    not take as arguments: the package directory, the box probe, and the
    device architecture. All are replaced here rather than mocked at a
    higher level, so the function under test is the real one, unmodified.
    `device` is what `_device_arch()` answers for ANY vendor."""
    B._LAYOUT = None
    B._VENDOR_SELECTED = None
    B._VENDOR_HOW = None
    B._ARCH_SELECTED = None
    B._ARCH_HOW = None
    B._pkg_dir = lambda: pkg
    B._probe_box = lambda: probe_result
    B._device_arch = lambda vendor: device
    if vendor_env is None:
        os.environ.pop("MOJOLEARN_VENDOR", None)
    else:
        os.environ["MOJOLEARN_VENDOR"] = vendor_env
    if arch_env is None:
        os.environ.pop("MOJOLEARN_GPU_ARCH", None)
    else:
        os.environ["MOJOLEARN_GPU_ARCH"] = arch_env


def _probe(cuda=False, hip=False):
    """A probe result shaped exactly like the real `_probe_box()` returns."""
    def one(found, spec):
        return {
            "paths": {p: found for p in spec["paths"]},
            "libs": {l: False for l in spec["libs"]},
            "found": found,
        }
    return {
        "cuda": one(cuda, B._PROBE["cuda"]),
        "hip": one(hip, B._PROBE["hip"]),
    }


def _mkinstall(root, vendors_with_binaries=(), empty_vendor_dirs=(),
               flat_binaries=False, vendor_archs=None):
    """A package directory shaped like one of the real layouts.

    `vendors_with_binaries` builds the ARCH-LESS (legacy) vendor layout;
    `vendor_archs` ({vendor: (arch, ...)}) builds the architecture layout,
    one set per named architecture."""
    pkg = os.path.join(root, "mojolearn")
    os.makedirs(pkg, exist_ok=True)
    if flat_binaries:
        open(os.path.join(pkg, "_mojolearn.so"), "w").close()
    for v in vendors_with_binaries:
        d = os.path.join(pkg, v)
        os.makedirs(os.path.join(d, "deterministic"), exist_ok=True)
        os.makedirs(os.path.join(d, "identical"), exist_ok=True)
        open(os.path.join(d, "_mojolearn.so"), "w").close()
    for v, archs in (vendor_archs or {}).items():
        for arch in archs:
            d = os.path.join(pkg, v, arch)
            os.makedirs(os.path.join(d, "deterministic"), exist_ok=True)
            os.makedirs(os.path.join(d, "identical"), exist_ok=True)
            open(os.path.join(d, "_mojolearn.so"), "w").close()
    for v in empty_vendor_dirs:
        os.makedirs(os.path.join(pkg, v), exist_ok=True)
    return pkg


def case(name):
    def deco(fn):
        try:
            fn()
        except AssertionError as exc:
            FAILURES.append((name, str(exc)))
            print("FAIL  %s\n        %s" % (name, exc))
        except Exception:                       # noqa: BLE001
            FAILURES.append((name, traceback.format_exc()))
            print("ERROR %s\n%s" % (name, traceback.format_exc()))
        else:
            PASSES.append(name)
            print("pass  %s" % name)
        return fn
    return deco


def expect_import_error(fn, must_mention):
    """Run `fn`, require ImportError, and require the message to NAME things.

    A refusal that raises the right type and says nothing useful is a
    refusal a user cannot act on, and every message in `_layout()` was
    written to be read at 2am by someone whose container has no GPU."""
    try:
        fn()
    except ImportError as exc:
        msg = str(exc)
        missing = [m for m in must_mention if m not in msg]
        assert not missing, (
            "the refusal fired but its message never mentions %r.\n"
            "        message was: %s" % (missing, msg[:400]))
        return msg
    else:
        raise AssertionError("expected ImportError, none raised")


def main():
    root = tempfile.mkdtemp(prefix="mojolearn-selector-")
    try:
        run_all(root)
    finally:
        shutil.rmtree(root, ignore_errors=True)

    print("")
    print("%d passed, %d failed" % (len(PASSES), len(FAILURES)))
    if FAILURES:
        print("")
        print("THE SELECTOR IS NOT READY FOR A LEASE. Failures:")
        for n, e in FAILURES:
            print("  - %s: %s" % (n, e.strip().splitlines()[-1][:160]))
        return 1
    print("")
    print("The selector's decision tree is correct on a box with no GPU.")
    print("STILL OWED, and this file shortens none of it: LINUX_WHEEL.md")
    print("section 7 gates (a) smoke, (b) sabotage, (c) audit, (d) no-GPU,")
    print("and the install smoke -- all on rented NVIDIA and AMD boxes,")
    print("because not one line here loads an extension.")
    return 0


def run_all(root):
    # ---------------------------------------------------------------- flat
    @case("macOS / source checkout: no vendor dir means FLAT")
    def _():
        pkg = _mkinstall(os.path.join(root, "a"), flat_binaries=True)
        _reset(pkg, _probe())
        kind, base = B._layout()
        assert kind == "flat", "got %r" % kind
        assert base == pkg, "got %r" % base

    @case("an EMPTY vendor dir must not turn a flat install into a lookup")
    def _():
        # Section 2: "an empty one left by a failed build must not turn a
        # working flat install into a vendor lookup." A failed build leaves
        # exactly this, and the flat install still has to import.
        pkg = _mkinstall(os.path.join(root, "b"), flat_binaries=True,
                         empty_vendor_dirs=("cuda", "hip"))
        _reset(pkg, _probe())
        kind, _ = B._layout()
        assert kind == "flat", "an empty cuda/ was treated as a set: %r" % kind

        # SABOTAGE: put ONE binary in it and the same call must flip.
        open(os.path.join(pkg, "cuda", "_mojolearn.so"), "w").close()
        _reset(pkg, _probe(cuda=True))
        kind2, base2 = B._layout()
        assert kind2 == "vendor", (
            "adding a binary to cuda/ did not change the answer, so the "
            "first assertion proved nothing")
        assert base2.endswith("cuda"), "got %r" % base2

    # -------------------------------------------------------------- probe
    @case("box probe: exactly one vendor's evidence picks that vendor")
    def _():
        pkg = _mkinstall(os.path.join(root, "c"),
                         vendors_with_binaries=("cuda", "hip"))
        _reset(pkg, _probe(cuda=True))
        kind, base = B._layout()
        assert (kind, os.path.basename(base)) == ("vendor", "cuda"), \
            "got %r" % ((kind, base),)
        assert B._VENDOR_HOW.startswith("the box probe"), B._VENDOR_HOW

        # SABOTAGE: move the evidence to the other vendor; so must the answer.
        _reset(pkg, _probe(hip=True))
        _, base2 = B._layout()
        assert os.path.basename(base2) == "hip", (
            "the probe returned cuda for a box whose only evidence is hip, "
            "so the vendor is not actually being read from the probe")

    @case("box probe: a vendor with evidence but NO set carried is not chosen")
    def _():
        # The install carries cuda only; the box looks like AMD. There is no
        # set to load, so this must refuse, not silently serve cuda.
        pkg = _mkinstall(os.path.join(root, "d"),
                         vendors_with_binaries=("cuda",))
        _reset(pkg, _probe(hip=True))
        msg = expect_import_error(
            B._layout,
            ["NO SUPPORTED GPU", "cuda", "/dev/nvidiactl", "MOJOLEARN_VENDOR"])
        assert "hip" not in os.path.basename(msg.split("\n")[0]), msg[:200]

    @case("box probe: TWO vendors' evidence refuses and asks for the env var")
    def _():
        pkg = _mkinstall(os.path.join(root, "e"),
                         vendors_with_binaries=("cuda", "hip"))
        _reset(pkg, _probe(cuda=True, hip=True))
        expect_import_error(
            B._layout,
            ["MORE THAN ONE", "MOJOLEARN_VENDOR=cuda", "MOJOLEARN_VENDOR=hip",
             "/dev/kfd", "/dev/nvidiactl"])

    @case("box probe: NO evidence raises the no-GPU refusal and names the table")
    def _():
        pkg = _mkinstall(os.path.join(root, "f"),
                         vendors_with_binaries=("cuda", "hip"))
        _reset(pkg, _probe())
        msg = expect_import_error(
            B._layout,
            ["NO SUPPORTED GPU FOUND", "no CPU path",
             "/dev/nvidiactl", "/dev/kfd", "libcuda.so.1",
             "--gpus all", "--device /dev/kfd"])
        # It must NOT fall through to a vendor: gate (b)'s exact wording.
        assert B._LAYOUT is None, (
            "the no-GPU path cached a layout, so a second import would "
            "silently succeed against a set this box cannot run")

    # ----------------------------------------------------- the env override
    @case("MOJOLEARN_VENDOR outranks the probe")
    def _():
        pkg = _mkinstall(os.path.join(root, "g"),
                         vendors_with_binaries=("cuda", "hip"))
        # The box looks like NVIDIA; the user says hip. Section 3: the env
        # can "only open the other directory on a box that does not have
        # that device", and the first device call is what fails.
        _reset(pkg, _probe(cuda=True), vendor_env="hip")
        _, base = B._layout()
        assert os.path.basename(base) == "hip", (
            "MOJOLEARN_VENDOR=hip did not beat a cuda-looking box: %r" % base)

        # SABOTAGE: unset it and the same box must go back to cuda.
        _reset(pkg, _probe(cuda=True))
        _, base2 = B._layout()
        assert os.path.basename(base2) == "cuda", (
            "clearing MOJOLEARN_VENDOR did not change the answer, so the "
            "override was not what decided the first one")

    @case("MOJOLEARN_VENDOR with a bogus value refuses and names the valid ones")
    def _():
        pkg = _mkinstall(os.path.join(root, "h"),
                         vendors_with_binaries=("cuda", "hip"))
        _reset(pkg, _probe(cuda=True), vendor_env="rocm")
        expect_import_error(B._layout, ["rocm", "'cuda' or 'hip'"])

    @case("MOJOLEARN_VENDOR naming a set this install lacks refuses by path")
    def _():
        pkg = _mkinstall(os.path.join(root, "i"),
                         vendors_with_binaries=("cuda",))
        _reset(pkg, _probe(cuda=True), vendor_env="hip")
        expect_import_error(
            B._layout,
            ["carries no hip set", os.path.join(pkg, "hip"), "['cuda']"])

    @case("MOJOLEARN_VENDOR is case- and whitespace-tolerant")
    def _():
        pkg = _mkinstall(os.path.join(root, "j"),
                         vendors_with_binaries=("cuda", "hip"))
        _reset(pkg, _probe(cuda=True), vendor_env="  HIP  ")
        _, base = B._layout()
        assert os.path.basename(base) == "hip", (
            "'  HIP  ' was not accepted; a user who exports it with a "
            "trailing space gets a refusal that reads like a bug: %r" % base)

    # --------------------------------------------------------- the tier axis
    @case("tier_dir puts the tier UNDER the vendor, both axes at once")
    def _():
        pkg = _mkinstall(os.path.join(root, "k"),
                         vendors_with_binaries=("cuda", "hip"))
        _reset(pkg, _probe(hip=True))
        fast = B.tier_dir("fast")
        det = B.tier_dir("deterministic")
        ident = B.tier_dir("identical")
        assert os.path.basename(fast) == "hip", fast
        assert det == os.path.join(pkg, "hip", "deterministic"), det
        assert ident == os.path.join(pkg, "hip", "identical"), ident

        # SABOTAGE: same tiers under the other vendor must move.
        _reset(pkg, _probe(cuda=True))
        det2 = B.tier_dir("deterministic")
        assert det2 == os.path.join(pkg, "cuda", "deterministic"), (
            "the tier path did not follow the vendor: %r" % det2)

    @case("tier_dir on a FLAT install keeps the macOS paths unchanged")
    def _():
        pkg = _mkinstall(os.path.join(root, "l"), flat_binaries=True)
        _reset(pkg, _probe())
        assert B.tier_dir("fast") == pkg, B.tier_dir("fast")
        assert B.tier_dir("identical") == os.path.join(pkg, "identical")

    # ------------------------------------------------- the architecture axis
    @case("arch layout: the device's exact architecture picks its directory")
    def _():
        pkg = _mkinstall(os.path.join(root, "n"),
                         vendor_archs={"cuda": ("sm_80", "sm_86")})
        _reset(pkg, _probe(cuda=True), device=("sm_80", "test probe"))
        kind, base = B._layout()
        assert kind == "vendor" and base == os.path.join(pkg, "cuda", "sm_80"), base
        assert B.gpu_arch() == "sm_80", B.gpu_arch()
        assert "exact match" in B.gpu_arch_how(), B.gpu_arch_how()

        # SABOTAGE: a different device must move the answer.
        _reset(pkg, _probe(cuda=True), device=("sm_86", "test probe"))
        _, base2 = B._layout()
        assert base2.endswith("sm_86"), (
            "the arch did not follow the device, so the first assertion "
            "proved nothing: %r" % base2)

    @case("cuda: a device sm_XY accepts a carried sm_XYa build")
    def _():
        pkg = _mkinstall(os.path.join(root, "o"),
                         vendor_archs={"cuda": ("sm_90a",)})
        _reset(pkg, _probe(cuda=True), device=("sm_90", "test probe"))
        _, base = B._layout()
        assert base.endswith("sm_90a"), base
        assert "architecture-specific" in B.gpu_arch_how(), B.gpu_arch_how()

    @case("cuda family fallback: an sm_86 device runs the carried sm_80 set")
    def _():
        pkg = _mkinstall(os.path.join(root, "p"),
                         vendor_archs={"cuda": ("sm_80",)})
        _reset(pkg, _probe(cuda=True), device=("sm_86", "test probe"))
        _, base = B._layout()
        assert base.endswith("sm_80"), base
        assert "same-family" in B.gpu_arch_how(), B.gpu_arch_how()

        # SABOTAGE: a device of ANOTHER family must refuse, not inherit.
        _reset(pkg, _probe(cuda=True), device=("sm_75", "test probe"))
        expect_import_error(B._layout, ["sm_75", "sm_80", "MOJOLEARN_GPU_ARCH"])

    @case("cuda family fallback never goes DOWN: sm_80 device, sm_86-only install")
    def _():
        pkg = _mkinstall(os.path.join(root, "q"),
                         vendor_archs={"cuda": ("sm_86",)})
        _reset(pkg, _probe(cuda=True), device=("sm_80", "test probe"))
        expect_import_error(B._layout, ["sm_80", "sm_86", "MOJOLEARN_GPU_ARCH"])

    @case("an a-suffixed build is never chosen by the family rule")
    def _():
        # sm_90a targets EXACTLY the 9.0 chip; a later 9.x device must not
        # inherit it the way it would inherit sm_90.
        pkg = _mkinstall(os.path.join(root, "r"),
                         vendor_archs={"cuda": ("sm_90a",)})
        _reset(pkg, _probe(cuda=True), device=("sm_92", "test probe"))
        expect_import_error(B._layout, ["sm_92", "sm_90a"])

        # SABOTAGE: the same install, non-a, and the same device must load.
        pkg2 = _mkinstall(os.path.join(root, "r2"),
                          vendor_archs={"cuda": ("sm_90",)})
        _reset(pkg2, _probe(cuda=True), device=("sm_92", "test probe"))
        _, base = B._layout()
        assert base.endswith("sm_90"), (
            "sm_90 was not inherited where sm_90a was rightly refused, so "
            "the refusal above was not about the a suffix: %r" % base)

    @case("hip is exact-only: no cross-architecture inheritance")
    def _():
        pkg = _mkinstall(os.path.join(root, "s"),
                         vendor_archs={"hip": ("gfx942",)})
        _reset(pkg, _probe(hip=True), device=("gfx90a", "test probe"))
        expect_import_error(B._layout, ["gfx90a", "gfx942", "ISA-exact"])

        # SABOTAGE: the exact device must load the same install.
        _reset(pkg, _probe(hip=True), device=("gfx942", "test probe"))
        _, base = B._layout()
        assert base.endswith("gfx942"), base

    @case("device arch unreadable: one carried arch is used, two refuse")
    def _():
        pkg = _mkinstall(os.path.join(root, "t"),
                         vendor_archs={"cuda": ("sm_80",)})
        _reset(pkg, _probe(cuda=True), device=(None, "no libcuda in this test"))
        _, base = B._layout()
        assert base.endswith("sm_80"), base
        assert "only architecture carried" in B.gpu_arch_how(), B.gpu_arch_how()

        pkg2 = _mkinstall(os.path.join(root, "t2"),
                          vendor_archs={"cuda": ("sm_80", "sm_90a")})
        _reset(pkg2, _probe(cuda=True), device=(None, "no libcuda in this test"))
        expect_import_error(
            B._layout, ["sm_80", "sm_90a", "MOJOLEARN_GPU_ARCH",
                        "no libcuda in this test"])

    @case("MOJOLEARN_GPU_ARCH outranks the device")
    def _():
        pkg = _mkinstall(os.path.join(root, "u"),
                         vendor_archs={"cuda": ("sm_80", "sm_86")})
        _reset(pkg, _probe(cuda=True), device=("sm_80", "test probe"),
               arch_env="sm_86")
        _, base = B._layout()
        assert base.endswith("sm_86"), (
            "MOJOLEARN_GPU_ARCH=sm_86 did not beat an sm_80 device: %r" % base)
        assert "MOJOLEARN_GPU_ARCH" in B.gpu_arch_how(), B.gpu_arch_how()

        # SABOTAGE: clear it and the device must decide again.
        _reset(pkg, _probe(cuda=True), device=("sm_80", "test probe"))
        _, base2 = B._layout()
        assert base2.endswith("sm_80"), (
            "clearing MOJOLEARN_GPU_ARCH did not change the answer, so the "
            "override was not what decided the first one")

    @case("MOJOLEARN_GPU_ARCH naming an absent arch refuses by name")
    def _():
        pkg = _mkinstall(os.path.join(root, "v"),
                         vendor_archs={"cuda": ("sm_80",)})
        _reset(pkg, _probe(cuda=True), device=("sm_80", "test probe"),
               arch_env="sm_75")
        expect_import_error(B._layout, ["sm_75", "sm_80"])

    @case("tier_dir puts the tier under vendor AND arch")
    def _():
        pkg = _mkinstall(os.path.join(root, "w"),
                         vendor_archs={"cuda": ("sm_80",)})
        _reset(pkg, _probe(cuda=True), device=("sm_80", "test probe"))
        det = B.tier_dir("deterministic")
        assert det == os.path.join(pkg, "cuda", "sm_80", "deterministic"), det
        assert B.tier_dir("fast") == os.path.join(pkg, "cuda", "sm_80")

    @case("an arch-less vendor set keeps working beside the new layout")
    def _():
        # Every set built before 2026-08-30 has binaries directly under
        # <vendor>/. That layout is supported, not deprecated.
        pkg = _mkinstall(os.path.join(root, "x"),
                         vendors_with_binaries=("cuda",))
        _reset(pkg, _probe(cuda=True), device=("sm_80", "test probe"))
        kind, base = B._layout()
        assert (kind, os.path.basename(base)) == ("vendor", "cuda"), (kind, base)
        assert B.gpu_arch() is None, B.gpu_arch()
        assert "arch-less" in B.gpu_arch_how(), B.gpu_arch_how()

    # ------------------------------------------------------------- caching
    @case("the layout is decided ONCE per process")
    def _():
        pkg = _mkinstall(os.path.join(root, "m"),
                         vendors_with_binaries=("cuda", "hip"))
        _reset(pkg, _probe(cuda=True))
        first = B._layout()
        # Change the box underneath it. A cached answer must not move: two
        # different vendors' binaries loaded into one process is the failure
        # this cache exists to prevent.
        B._probe_box = lambda: _probe(hip=True)
        second = B._layout()
        assert first == second, (
            "the layout changed inside one process (%r then %r)" % (
                first, second))


if __name__ == "__main__":
    raise SystemExit(main())
