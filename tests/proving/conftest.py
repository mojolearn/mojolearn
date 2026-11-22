# SPDX-License-Identifier: Apache-2.0
"""Proving suite configuration: the `gpu` marker and its clean skip on an install without a GPU.

CPU:  python -m pytest tests/proving -x -q -n auto      (pytest-xdist splits the files across every core;
                                                       drop -n auto when xdist is not installed)
GPU:  python -m pytest tests/proving -x -q -m gpu -n auto
"""
import pytest


def pytest_configure(config):
    config.addinivalue_line("markers", "gpu: needs a GPU device in the installed wheel; skipped cleanly without one")


def pytest_collection_modifyitems(config, items):
    try:
        import mojolearn

        gpu = mojolearn.vendor() != "cpu"
    except Exception:
        gpu = False
    if gpu:
        return
    skip = pytest.mark.skip(reason="no GPU device in this installation (mojolearn.vendor() == 'cpu')")
    for item in items:
        if "gpu" in item.keywords:
            item.add_marker(skip)
