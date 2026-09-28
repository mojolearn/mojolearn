"""NumPy support for optional APIs; importing this module needs only Python."""


def require_numpy(feature):
    """Return NumPy unchanged, or explain how to install optional support.

    Do not disguise a broken NumPy installation as an absent dependency.
    Keeping the actual module preserves its array and random-generator behavior.
    """
    try:
        import numpy
    except ModuleNotFoundError as exc:
        if exc.name != "numpy":
            raise
        raise ModuleNotFoundError(
            f"mojolearn: {feature} requires optional NumPy support. "
            'Install it with: python -m pip install "mojolearn[numpy]" '
            '("mojolearn[verify]" also includes NumPy).', name="numpy"
        ) from exc
    return numpy
