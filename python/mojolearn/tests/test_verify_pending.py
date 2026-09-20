import types
import json

import pytest

from mojolearn import __main__ as cli, _verify_all as va, _verify_reference as vr
from mojolearn import _verification_coverage as coverage


def test_pending_selection_runs_declared_cpu_shards_without_admitting_gpu_only_drivers():
    h = va.load_harness()
    table = vr.load_table()
    normal, _ = va.select_lanes(h, table, 'cpu', 'full', [])
    broad, _ = va.select_lanes(h, table, 'cpu', 'full', [], include_pending=True)
    assert set(normal) < set(broad)
    # A PENDING LANE IS SELECTABLE UNLESS RUNNING IT WOULD RAISE (2026-09-20).
    # The `no cpu route` lanes refuse BY NAME on a CPU-only install, and a
    # raise is a REFUSED part, which still gates. Selecting them would turn a
    # structural fact about the box into a failure of the run. They are still
    # PUBLIC and still reported, as NOT APPLICABLE with the reason.
    surface = va.host_surface()
    blocked = {l for l in surface.PUBLIC_PENDING_LANES if surface.pending_blocks_comparison(l)}
    assert set(surface.PUBLIC_PENDING_LANES) - blocked <= set(broad)
    assert blocked, "the case this test distinguishes has to exist"
    exposure = surface.lane_exposure(list(h.LANES))
    for lane in blocked:
        assert exposure[lane]["exposed"], f"{lane} was hidden rather than reported"
        assert exposure[lane]["status"] in (surface.LANE_OWED, surface.LANE_NOT_APPLICABLE,
                                            surface.LANE_HELD)
    assert set(va.host_surface().PUBLIC_REFERENCE_CANDIDATES) <= set(broad)
    for lane in ('par-scaler-minmax', 'par-queries-nn', 'par-forest-et-clf', 'par-forest-reg'):
        assert lane in broad and lane not in normal
        assert va.select_lanes(h, table, 'cpu', 'full', [lane], True)[0] == [lane]
    assert 'par-gram' not in broad
    with pytest.raises(ValueError, match='does not run'):
        va.select_lanes(h, table, 'cpu', 'full', ['par-gram'], True)


def test_stale_reference_cannot_hide_a_local_failure_or_claim_a_false_match():
    table = dict(cells={'x/base': {'batch': {'ref': 'a'*16, 'cols': {}}}})
    def judged(value, error=None):
        return va.judge_rows([dict(lane='x', fixture='base', part='batch', value=value,
                                  error=error)], table, unreferenced_lanes={'x'})[0]
    row = judged('a'*16)
    assert row['state'] == vr.OWED and row['reference'] is None
    assert row['value'] == 'a'*16 and row['local_check'] == 'passed'
    assert judged('BATCH_MOVED: real mismatch')['state'] == vr.DIVERGENT
    assert judged(None, 'native operation unavailable')['state'] == vr.REFUSED
    counts = va.lane_counts([row], ['x'], stale=['x'])
    assert counts['requested'] == counts['attempted'] == counts['skipped'] == 1
    assert counts['checked'] == 0 and counts['not_run'] == []


def test_saved_model_entries_have_installed_commands_and_parallel_scope_is_explicit():
    r = coverage.inventory(va.load_harness(), vr.load_table(), 'cpu')
    entries = {e.get('api'): e for e in r['entries'] if e.get('api')}
    for name in ('HostForest', 'HostGBDT'):
        check = entries[name]['installed_check']
        assert check['status'] == 'available'
        assert check['command'] == 'verify --models-only' and check['run_by_default']
        assert check['models']
    execution = r['lanes']['par-scaler-minmax']['execution']
    assert execution['cpu_logical_shards'] and not execution['requires_gpu_for_execution']
    assert not execution['physical_multi_gpu_measured_by_cpu']
    assert r['lanes']['par-gram']['execution']['requires_gpu_for_execution']
    assert not any(row['release_qualified'] for row in r['lanes'].values())


def test_new_selection_flags_dispatch_and_reject_ambiguous_model_scope():
    for flag in ('--models-only', '--include-pending'):
        args = cli.build_parser().parse_args(['verify', flag])
        assert cli._wants_suite(args)
        assert va._depth(args) == 'full'
    for flag in ('no_models', 'lanes', 'fixtures', 'include_pending', 'batch_checks', 'quick'):
        with pytest.raises(ValueError, match='models-only'):
            va._depth(types.SimpleNamespace(models_only=True, **{flag: True}))


def test_models_only_uses_cpu_on_gpu_install_and_detects_repeat_instability(tmp_path):
    base = tmp_path / 'verify_reference' / 'models'
    base.mkdir(parents=True)
    (base / 'model.npz').write_bytes(b'fixture model bytes')
    (base / 'models.json').write_text(json.dumps({'models': [dict(
        lane='rf-clf', fixture='base', file='model.npz', **{'class': 'MustNotUseGpu'})]}))
    loaded = []
    ml = types.SimpleNamespace(vendor=lambda: 'cuda', host_model=lambda p: loaded.append(p))
    values = iter(['a'*16, 'b'*16])
    h = types.SimpleNamespace(Fit=lambda _: types.SimpleNamespace(),
        heldout=lambda _: [], BATCH_ALONE=1,
        _probe_batch=lambda *args: (next(values), None))
    rows = va.run_models(h, ml, {}, pkg_dir=str(tmp_path), repeats=2, host_only=True)
    assert len(loaded) == 2
    batch = next(row for row in rows if row['part'] == 'batch')
    assert va.judge_rows([batch], {'cells': {}})[0]['state'] == vr.DIVERGENT
