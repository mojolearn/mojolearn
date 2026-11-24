"""`verify --all` runs its cell batches across the core allocation (w3-integrate-8)."""
from mojolearn import _verify_all as va


def test_requested_count_pins_the_workers():
    assert va.verify_cell_workers(3) == 3
    assert va.verify_cell_workers(1) == 1


def test_default_is_bounded_by_the_allocation():
    import os
    got = va.verify_cell_workers()
    assert got >= 1
    try:
        assert got <= len(os.sched_getaffinity(0))
    except AttributeError:
        assert got <= (os.cpu_count() or 1)
