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


def test_cost_hints_ship_and_the_last_run_overrides_them(monkeypatch, tmp_path):
    """rel-fix-1 (2026-11-26): the pool's order comes from the shipped hints, then this machine's last run."""
    from mojolearn import _verify_all as va
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    hints, source = va.cell_cost_hints()
    assert source == "shipped" and hints["generic-pickle-trees"] > hints["pca"]
    assert {"@cross-check", "@models", "@public"} <= set(hints)
    va.save_cell_cost_hints(hints, {("pca", "base"): 500.0, ("pca", "ties"): 300.0}, {"@models": [7.0]})
    again, source = va.cell_cost_hints()
    assert source == "shipped+last run" and again["pca"] == 400.0 and again["@models"] == 7.0
    assert again["generic-pickle-trees"] == hints["generic-pickle-trees"]


def test_a_timed_out_cell_never_raises_the_next_limit(monkeypatch, tmp_path):
    """rel-fix-2 (2026-11-26): a killed cell's seconds are its limit, not its cost; the saved hints leave it out, so a
    lane's limit does not grow by CELL_LIMIT_FACTOR on every run that times it out."""
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    hints = {"pca": 10.0}
    va.save_cell_cost_hints(hints, {("pca", "base"): 1800.0, ("pca", "ties"): 12.0}, {},
                            skip={("pca", "base")})
    again, _ = va.cell_cost_hints()
    assert again["pca"] == 12.0
    assert va.cell_limit(again, "pca", 120.0) == 120.0
    assert va.cell_limit({"slow": 100.0}, "slow", 120.0) == 4 * 100.0
    assert va.cell_limit({"hang": 1e9}, "hang", 120.0) == va.CELL_LIMIT_MAX_S
