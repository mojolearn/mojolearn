"""`verify --all` runs its cell batches across the core allocation (w3-integrate-8)."""
from mojolearn import _verify_all as va


def test_requested_count_pins_the_workers():
    assert va.verify_cell_workers(3) == 3
    assert va.verify_cell_workers(1) == 1


def test_default_is_bounded_by_the_allocation():
    from mojolearn import _allocation
    got = va.verify_cell_workers()
    assert 1 <= got <= _allocation.cores()[0]


def test_the_jobs_measured_allocation_sets_the_workers_and_the_reason_is_printed(monkeypatch):
    """rel-fix-3 (2026-11-26): the L40S record job measured 14 cores and verify ran 128 workers (it read cgroup v2
    only; the pod's quota is v1). The job's MOJOLEARN_ALLOC_CORES is taken as given, still bounded by memory, the
    GPU and the cgroup's tasks; the count comes with its reason."""
    from mojolearn import _allocation
    monkeypatch.setenv(_allocation.ENV_CORES, "3")
    monkeypatch.setattr(_allocation, "gpu_free_mb", lambda timeout=30: 45000)
    n, reason = va.verify_cell_workers(with_reason=True)
    assert n <= 3 and "MOJOLEARN_ALLOC_CORES=3" in reason
    monkeypatch.setattr(_allocation, "gpu_free_mb", lambda timeout=30: 4096)
    n, reason = va.verify_cell_workers(with_reason=True)
    assert n == 1 and "gpu memory 1" in reason
    n, reason = va.verify_cell_workers(with_reason=True, gpu=False)
    assert "gpu memory" not in reason
    assert va.verify_cell_workers(7, with_reason=True) == (7, "7 requested")


def test_a_lane_without_a_measured_cost_is_limited_at_its_familys_and_the_load():
    """rel-fix-3: a lane missing from the hints took the bare --cell-timeout; it now takes its family's cost, and a
    pinned worker count above the cores raises every limit by that ratio."""
    hints = {"gbdt-a": 50.0, "pca": 1.0}
    assert va.cell_limit(hints, "gbdt-multiclass", 120.0) == 4 * 50.0
    assert va.cell_limit(hints, "pca", 120.0) == 120.0
    assert va.cell_limit(hints, "pca", 10.0, load=9.0) == 4 * 1.0 * 9.0
    assert va.cell_limit(hints, "gbdt-a", 120.0, load=0.5) == 4 * 50.0


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
