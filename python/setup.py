"""Wheel shape for the prebuilt Mojo extension.

setuptools compiles nothing. `bindings/build.sh` builds the extension with the
Mojo toolchain and `packaging/macos/build_release_wheel.sh` stages it, plus the
MAX runtime dylibs it links through @rpath, into python/mojolearn/ before this
file runs. There is no source build and no way to make one that works without
the Mojo toolchain, which is why no sdist is published.

All distribution metadata is in pyproject.toml. This file carries only the two
facts PEP 621 has no field for.

1. **The wheel is not pure Python.** `has_ext_modules` is forced True so the
   wheel is tagged for a platform rather than `py3-none-any`.

2. **The interpreter tag is `py3`, not `cp3xx`, and that is a measured claim.**
   `otool -L` on the extension shows no libpython: only the two MAX runtime
   dylibs and libSystem. It resolves CPython by name at runtime, so one
   artifact serves several interpreters without being built against the
   limited API -- abi3 exists to solve a problem this binary does not have.
   `packaging/macos/verify_wheel.sh` installs the built wheel into a clean
   venv under every python3.N present and runs a real fit. **If that script
   has not been run, this tag is a guess and should be reverted to the
   default.**

3. On macOS the platform tag is pinned to the Mojo compile step's deployment
   target, not the OS of the build machine. The `minos` in the extension's
   LC_BUILD_VERSION decides where the binary loads; the tag is what pip
   compares before it tries. A tag that disagrees with the binary is a
   published lie in one of two directions: too low and it installs onto Macs
   where it cannot load, too high and it is refused by Macs that could have
   run it.
"""

import os
import platform
import sys

from setuptools import setup
from setuptools.dist import Distribution

# Must equal the `minos` that `otool -l python/mojolearn/_mojolearn.so`
# reports for LC_BUILD_VERSION. Nothing here reads the Mach-O header, so the
# two are kept in step by the release procedure rather than by this file.
DEFAULT_MACOS_TARGET = "26.0"
TARGET_ENV_VAR = "MOJOLEARN_MACOS_DEPLOYMENT_TARGET"

# MACOSX_DEPLOYMENT_TARGET is deliberately NOT consulted. Conda-style
# environments, which is what pixi gives this build, export it for their own
# compilers at values unrelated to what the Mojo toolchain emitted.
# Inheriting it would tag a wheel with a floor its binary does not honor, and
# that failure only appears on someone else's machine.


class BinaryDistribution(Distribution):
    def has_ext_modules(self):
        return True


try:
    from wheel.bdist_wheel import bdist_wheel as _bdist_wheel
except ImportError:  # setuptools >= 70 vendors it
    from setuptools.command.bdist_wheel import bdist_wheel as _bdist_wheel


class bdist_wheel(_bdist_wheel):
    """Tag the interpreter `py3` rather than `cp3xx`.

    Setting `python_tag` in the options dict is not enough: `has_ext_modules`
    returning True makes bdist_wheel derive the implementation tag from the
    running interpreter and overwrite the option. Overriding `get_tag` is the
    only place the decision is actually made. The platform half is left as the
    base class computed it so `plat_name` below keeps working.
    """

    def get_tag(self):
        _, _, plat = _bdist_wheel.get_tag(self)
        return ("py3", "none", plat)


def macos_plat_name():
    arch = platform.machine().lower()
    if arch != "arm64":
        raise SystemExit(
            "mojolearn builds no macOS wheel for {!r}. pixi.toml declares no "
            "osx-64 platform and the pinned channel ships no Intel macOS "
            "toolchain, so there is nothing to build with. If this is Apple "
            "silicon, the build is running under Rosetta; use a native arm64 "
            "interpreter.".format(arch)
        )
    target = os.environ.get(TARGET_ENV_VAR, DEFAULT_MACOS_TARGET)
    return "macosx_{}_{}".format(target.replace(".", "_"), arch)


options = {}
if sys.platform == "darwin":
    options["bdist_wheel"] = {"plat_name": macos_plat_name()}

setup(
    distclass=BinaryDistribution,
    options=options,
    cmdclass={"bdist_wheel": bdist_wheel},
)
