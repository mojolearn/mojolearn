"""Release tier selection, separate from the set of historical valid modes."""
import argparse
import os

LEGACY_MODES = ("fast", "identical")
ENV = "MOJOLEARN_RELEASE_MODES"


def parse_modes(value):
    modes = value.split() if isinstance(value, str) else value
    if (not isinstance(modes, (list, tuple)) or not modes
            or any(mode not in LEGACY_MODES for mode in modes)
            or len(modes) != len(set(modes))):
        raise ValueError(f"{ENV} must name distinct supported modes: fast, identical")
    # Canonical order makes equivalent requests share one recorded identity.
    return tuple(mode for mode in LEGACY_MODES if mode in modes)


def release_modes(environ=None, default=LEGACY_MODES):
    return parse_modes((os.environ if environ is None else environ).get(ENV, default))


def record_modes(record):
    """Read artifact metadata; older artifacts explicitly keep their old scope.

    Never reinterpret an old proof using the caller's environment.
    """
    return parse_modes(record.get("release_modes", LEGACY_MODES))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--default", default="fast identical")
    args = parser.parse_args()
    try:
        print(" ".join(release_modes(default=parse_modes(args.default))))
    except ValueError as exc:
        parser.error(str(exc))
