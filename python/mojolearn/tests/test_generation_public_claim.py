"""Admission controls supplement the retained actual two-thread GPU probes."""
import os

import numpy as np
import pytest

from mojolearn.tests.test_generation_native import load_fixture

pytestmark = pytest.mark.skipif(
    os.environ.get("MOJOLEARN_GENERATION_DEVICE") != "gpu",
    reason="explicit installed GPU generation run required",
)


@pytest.mark.parametrize("operation", ["read", "info", "state_dict"])
def test_other_public_stream_owner_is_not_captured_or_released(tmp_path, operation):
    model = load_fixture(tmp_path, "llama", uniform=False)
    stream = model.stream(np.array([[1, 3]], dtype="i4"), 4)
    before = stream.info()
    fields = stream.__dict__
    # Deterministic admission negative control; actual overlapping calls are
    # separately exercised by the retained full-checkpoint concurrency harness.
    fields["_native_public_busy"] = True
    try:
        with pytest.raises(Exception, match="stream is busy"):
            getattr(stream, operation)(1) if operation == "read" else getattr(stream, operation)()
        assert fields["_native_public_busy"] is True
    finally:
        fields["_native_public_busy"] = False
    assert stream.info() == before
    assert np.asarray(stream.read(4)).shape == (1, 4)
    assert fields["_native_public_busy"] is False
