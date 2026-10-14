from types import SimpleNamespace
import json
import time

import pytest
import gpu_gap_campaign as C


def test_manifest_only_selects_requested_supported_gpu_recipes():
    rid = "algos/adam/synthetic/rows=full"
    plan = C.selected_plan("amd", {"amd": [rid]})
    assert [r["id"] for r in plan] == [rid]
    assert plan[0]["our_arms"] == {"ours": "identical"}
    assert all(a.startswith("torch-") for a in plan[0]["opponents"])
    for vendor, manifest in (("apple", {"apple": [rid]}), ("amd", {"amd": [rid, rid]}),
                             ("amd", {"amd": ["not-a-recipe"]})):
        with pytest.raises(ValueError):
            C.selected_plan(vendor, manifest)


def test_synthetic_recipe_is_not_passed_to_tabular_dataset_selector(tmp_path):
    args = SimpleNamespace(python_env="python", vendor="amd", cache=str(tmp_path))
    for dataset in ("synthetic", "gaussian", "taxi", "istella"):
        cmd = C.command(args, dict(family="algos", lane="adam", dataset=dataset), tmp_path)
        assert ("--datasets" in cmd) == (dataset in ("taxi", "istella"))
        assert cmd[cmd.index("--rows") + 1] == "full"
        assert cmd[cmd.index("--rounds") + 1] == "1"
        assert cmd[cmd.index("--round-seconds") + 1] == "0"
        assert "--arm-budget-s" not in cmd
        assert "--race-deadline-s" not in cmd


def test_explicit_wall_time_ceiling_does_not_change_workload(tmp_path):
    args = SimpleNamespace(python_env="python", vendor="nvidia", cache=str(tmp_path), round_seconds=1800)
    cmd = C.command(args, dict(family="classical", lane="dbscan", dataset="taxi"), tmp_path)
    assert cmd[cmd.index("--round-seconds") + 1] == "1800"
    assert cmd[cmd.index("--rows") + 1] == "full"
    assert cmd[cmd.index("--rounds") + 1] == "1"


def test_repair_can_reuse_a_verified_opponent_without_forcing_a_retime(tmp_path):
    args = SimpleNamespace(python_env="python", vendor="nvidia", cache=str(tmp_path),
                           opponent_store=str(tmp_path / "retained-opponents.jsonl"))
    cmd = C.command(args, dict(family="classical", lane="dbscan", dataset="taxi"), tmp_path)
    assert "--retime-opponents" not in cmd
    assert "--with-opponents" in cmd
    assert cmd[cmd.index("--opponent-store") + 1] == args.opponent_store


@pytest.mark.parametrize("preflight,returncode,races,expected_calls,expected_exit", [
    ("MATCHED", 1, {}, 1, 1),
    ("MATCHED", 0, {}, 1, 1),
    (None, 1, {}, 1, 1),
    ("MATCHED", 0, {"r": {"evidence_failure": "R2 retention failed"}}, 1, 1),
    ("REFUSED", 1, {}, 2, 0),
])
def test_campaign_stops_on_unsealed_work_but_continues_explicit_preflight_refusals(
        tmp_path, monkeypatch, preflight, returncode, races, expected_calls, expected_exit):
    ids = ['algos/adam/synthetic/rows=full', 'algos/adamw/synthetic/rows=full']
    manifest = tmp_path / 'manifest.json'
    manifest.write_text(json.dumps({'amd': ids}))
    monkeypatch.setattr(C, 'prepare_fixtures', lambda *a: {})
    calls = []
    def execute(cmd, log):
        calls.append(cmd)
        out = C.Path(cmd[cmd.index('--out') + 1]); out.mkdir(parents=True)
        # A retained raw scored output must survive an unexpected driver exit.
        (out / 'scored-output.bin').write_bytes(b'original scored bytes')
        C.E.write(out / 'board.json', {'preflight': {'verdict': preflight}, 'races': races})
        return returncode
    monkeypatch.setattr(C, 'logged', execute)
    monkeypatch.setattr(C, 'retain', lambda archive, members: {'sha256': 'fixture', 'status': 'verified'})
    wheel = tmp_path / 'wheel-manifest.json'
    wheel.write_text('{}')
    monkeypatch.setattr(C.B.PYPI, 'verify', lambda path, py: ({'vendor': 'amd', 'version': '0.8.44'}, None))
    out = tmp_path / 'campaign'
    rc = C.main(['--vendor', 'amd', '--manifest', str(manifest), '--out', str(out),
                 '--cache', str(tmp_path / 'cache'), '--python-env', 'fixture-python',
                 '--wheel-manifest', str(wheel), '--stop-at', str(time.time() + 3600)])
    assert rc == expected_exit and len(calls) == expected_calls
    state = json.loads((out / 'campaign.json').read_text())
    if expected_exit:
        assert state['stop_reason']
        assert state['recipes'][ids[1]]['status'] == 'pending'
    assert (out / 'measured' / ids[0].replace('/', '.') / 'scored-output.bin').read_bytes() == b'original scored bytes'


def _campaign(tmp_path, monkeypatch, extra, verify):
    rid = 'algos/adam/synthetic/rows=full'
    tmp_path.mkdir(parents=True, exist_ok=True)
    manifest = tmp_path / 'manifest.json'
    manifest.write_text(json.dumps({'amd': [rid]}))
    monkeypatch.setattr(C, 'prepare_fixtures', lambda *a: {})
    calls = []
    monkeypatch.setattr(C, 'logged', lambda cmd, log: calls.append(cmd) or 1)
    monkeypatch.setattr(C, 'retain', lambda archive, members: {'sha256': 'fixture', 'status': 'verified'})
    monkeypatch.setattr(C.B.PYPI, 'verify', verify)
    out = tmp_path / 'campaign'
    rc = C.main(['--vendor', 'amd', '--manifest', str(manifest), '--out', str(out),
                 '--cache', str(tmp_path / 'cache'), '--python-env', 'fixture-python',
                 '--stop-at', str(time.time() + 3600)] + extra)
    return rc, calls, json.loads((out / 'campaign.json').read_text())


def test_campaign_refuses_to_time_without_a_wheel_manifest(tmp_path, monkeypatch):
    rc, calls, state = _campaign(tmp_path, monkeypatch, [], lambda p, py: ({}, None))
    assert rc == 1 and calls == []
    assert 'wheel-manifest' in state['stop_reason']


def test_campaign_refuses_when_ours_did_not_load_from_the_pypi_files(tmp_path, monkeypatch):
    wheel = tmp_path / 'w.json'
    wheel.write_text('{}')
    rc, calls, state = _campaign(tmp_path, monkeypatch, ['--wheel-manifest', str(wheel)],
                                 lambda p, py: ({'vendor': 'amd'}, 'mojolearn loaded from /src, not site-packages'))
    assert rc == 1 and calls == []
    assert 'site-packages' in state['stop_reason']
    rc, calls, state = _campaign(tmp_path / 'b', monkeypatch, ['--wheel-manifest', str(wheel)],
                                 lambda p, py: ({'vendor': 'nvidia'}, None))
    assert rc == 1 and calls == [] and 'nvidia' in state['stop_reason']


def test_command_passes_the_wheel_manifest_to_every_recipe(tmp_path):
    args = SimpleNamespace(python_env="python", vendor="nvidia", cache=str(tmp_path),
                           wheel_manifest=str(tmp_path / "wheel-manifest.json"))
    cmd = C.command(args, dict(family="algos", lane="adam", dataset="synthetic"), tmp_path)
    assert cmd[cmd.index("--wheel-manifest") + 1] == str((tmp_path / "wheel-manifest.json").resolve())


# ---------------------------------------------------------------------------
# preflight-smoke lane (2026-10-14): log as we go, --smoke, Apple smoke
# ---------------------------------------------------------------------------

def _timed(tmp_path, monkeypatch, retain):
    ids = ['algos/adam/synthetic/rows=full', 'algos/adamw/synthetic/rows=full']
    manifest = tmp_path / 'manifest.json'
    manifest.write_text(json.dumps({'amd': ids}))
    monkeypatch.setattr(C, 'prepare_fixtures', lambda *a: {})

    def execute(cmd, log):
        out = C.Path(cmd[cmd.index('--out') + 1])
        (out / 'evidence').mkdir(parents=True)
        C.E.write(out / 'evidence' / 'preflight.json', {'verdict': 'REFUSED'})
        C.E.write(out / 'board.json', {'preflight': {'verdict': 'REFUSED'}, 'races': {}})
        C.Path(log).write_text('bench_board: plan preflight REFUSED\n')
        return 1
    monkeypatch.setattr(C, 'logged', execute)
    monkeypatch.setattr(C, 'retain', retain)
    monkeypatch.setattr(C.B.PYPI, 'verify', lambda path, py: ({'vendor': 'amd', 'version': '0.8.44'}, None))
    wheel = tmp_path / 'wheel-manifest.json'
    wheel.write_text('{}')
    out = tmp_path / 'campaign'
    rc = C.main(['--vendor', 'amd', '--manifest', str(manifest), '--out', str(out),
                 '--cache', str(tmp_path / 'cache'), '--python-env', 'fixture-python',
                 '--wheel-manifest', str(wheel), '--stop-at', str(time.time() + 3600)])
    return rc, ids, out


def test_every_recipe_uploads_its_evidence_and_campaign_json_as_it_finishes(tmp_path, monkeypatch):
    uploads = []
    rc, ids, out = _timed(tmp_path, monkeypatch,
                          lambda archive, members: uploads.append((C.Path(archive).name, sorted(members)))
                          or {'sha256': str(len(uploads)), 'status': 'verified'})
    assert rc == 0
    names = [u[0] for u in uploads]
    # recipe archive, then campaign.json, after EACH recipe (refused recipes included)
    assert names == [ids[0].replace('/', '.') + '.tar.gz', 'campaign.tar.gz',
                     ids[1].replace('/', '.') + '.tar.gz', 'campaign.tar.gz']
    recipe_members = uploads[0][1]
    assert {'campaign-row.json', 'recipe.log', 'out/board.json', 'out/evidence/preflight.json'} <= set(recipe_members)
    assert uploads[1][1] == ['campaign.json']
    state = json.loads((out / 'campaign.json').read_text())
    assert all(state['recipes'][rid]['retention']['status'] == 'verified' for rid in ids)


def test_a_failed_recipe_upload_stops_before_the_next_recipe(tmp_path, monkeypatch):
    def fail(archive, members):
        raise ValueError('R2 evidence request failed; local bytes retained')
    rc, ids, out = _timed(tmp_path, monkeypatch, fail)
    state = json.loads((out / 'campaign.json').read_text())
    assert rc == 1 and 'upload failed' in state['stop_reason']
    assert state['recipes'][ids[1]]['status'] == 'pending'
    assert 'retention_failure' in state['recipes'][ids[0]]


def test_apple_is_smoke_only_and_uses_both_modes():
    ident = C.selected_plan('apple', {}, 'identical', smoke=True)
    fast = C.selected_plan('apple', {}, 'fast', smoke=True)
    assert ident and isinstance(fast, list)
    assert all(r['opponents'] for r in ident + fast)
    with pytest.raises(ValueError):
        C.selected_plan('apple', {'apple': [ident[0]['id']]})
    with pytest.raises(ValueError):
        C.selected_plan('nvidia', {}, 'fast', smoke=True)
    with pytest.raises(SystemExit):
        C.main(['--vendor', 'apple', '--manifest', 'm.json', '--out', 'x', '--cache', 'c',
                '--stop-at', str(time.time() + 60)])


def test_smoke_selects_the_whole_plan_when_the_manifest_has_no_list_for_the_vendor():
    whole = C.selected_plan('amd', {}, smoke=True)
    rid = 'algos/adam/synthetic/rows=full'
    assert rid in [r['id'] for r in whole]
    assert [r['id'] for r in C.selected_plan('amd', {'amd': [rid]}, smoke=True)] == [rid]


def test_smoke_command_is_preflight_only_with_the_scorer_and_one_race_id(tmp_path):
    args = SimpleNamespace(python_env="python", vendor="amd", cache=str(tmp_path), wheel_manifest=None)
    race = dict(id='classical/dbscan/taxi/rows=full', family='classical', lane='dbscan', dataset='taxi')
    cmd = C.smoke_command(args, race, tmp_path, 'identical', {'source': 's.json', 'box_checked': 'smoke.json'})
    for flag in ('--preflight-only', '--preflight-scorer', '--gpu-opponents-only'):
        assert flag in cmd
    assert cmd[cmd.index('--race-ids') + 1] == race['id']
    assert cmd[cmd.index('--measurement-source') + 1] == 's.json'
    assert cmd[cmd.index('--box-checked') + 1] == 'smoke.json'
    assert '--retime-opponents' not in cmd and '--verify-data' not in cmd


def test_run_bounded_kills_the_whole_process_group_at_the_timeout(tmp_path):
    marker = tmp_path / 'child-survived'
    t0 = time.time()
    rc, seconds = C.run_bounded(['sh', '-c', '(sleep 3; touch %s) & sleep 30' % marker], tmp_path / 'log', 0.5)
    assert rc is None and time.time() - t0 < 10
    time.sleep(3.5)
    assert not marker.exists()
    rc, _ = C.run_bounded(['sh', '-c', 'exit 3'], tmp_path / 'log2', 5)
    assert rc == 3


def _write_board(directory, rid, audit):
    C.E.write(C.Path(directory) / 'board.json', {'preflight': {'races': {rid: {'audit': audit}}}})


def test_smoke_rows_matched_refused_started_failed(tmp_path):
    race = dict(id='algos/adam/synthetic/rows=full', family='algos', lane='adam', dataset='synthetic')
    d = tmp_path / 'r'
    _write_board(d, race['id'], {'verdict': 'REFUSED', 'problems': ['torch-eager-fp32: amsgrad missing from ours read-back; explicit semantic mapping required'],
                                 'scorer_preflight': {'status': 'ok'}})
    row = C.smoke_row(race, 'identical', 1, 2.0, d, tmp_path / 'none.log', 5)
    assert row['status'] == 'REFUSED' and 'amsgrad' in row['problems'][0]
    log = tmp_path / 'started.log'
    log.write_text('bench_board: algos prep rows=full lanes=1 datasets=taxi\n')
    row = C.smoke_row(race, 'identical', None, 5.0, tmp_path / 'empty', log, 5)
    assert row['status'] == 'STARTED' and row['reached']['last_board_line'].startswith('bench_board: algos prep')
    log.write_text('Traceback\nImportError: no module named torch\n')
    row = C.smoke_row(race, 'identical', 1, 1.0, tmp_path / 'empty', log, 5)
    assert row['status'] == 'FAILED' and 'ImportError' in row['problems'][0]


def test_smoke_writes_rows_text_and_uploads_after_every_recipe(tmp_path, monkeypatch):
    ids = ['algos/adam/synthetic/rows=full', 'algos/adamw/synthetic/rows=full', 'algos/sgd/synthetic/rows=full']
    manifest = tmp_path / 'manifest.json'
    manifest.write_text(json.dumps({'amd': ids}))
    monkeypatch.setattr(C, 'smoke_stage', lambda *a, **k: {'status': 'ok', 'problems': []})
    monkeypatch.setattr(C, 'smoke_checks', lambda args, plans, out, warm=None: (
        {'wheel': {'status': 'ok', 'problems': []}, 'data': {'status': 'failed', 'problems': ['dataset taxi missing']}},
        {}))
    seen = []

    def bounded(cmd, log, timeout, env=None):
        rid = cmd[cmd.index('--race-ids') + 1]
        seen.append((rid, timeout))
        C.Path(log).write_text('bench_board: setup\n')
        if rid == ids[0]:
            _write_board(cmd[cmd.index('--out') + 1], rid, {'verdict': 'MATCHED', 'problems': []})
            return 0, 1.0
        if rid == ids[1]:
            _write_board(cmd[cmd.index('--out') + 1], rid, {'verdict': 'REFUSED', 'problems': [
                'torch-eager-fp32: maximize missing from ours read-back; explicit semantic mapping required']})
            return 1, 2.0
        return None, timeout
    monkeypatch.setattr(C, 'run_bounded', bounded)
    uploads = []
    monkeypatch.setattr(C, 'retain', lambda archive, members: uploads.append(sorted(members)) or {'sha256': 'x'})
    out = tmp_path / 'campaign'
    rc = C.main(['--vendor', 'amd', '--manifest', str(manifest), '--out', str(out), '--cache', str(tmp_path),
                 '--smoke', '--smoke-warm', 'off'])
    assert rc == 0
    assert [t for _, t in seen] == [5.0, 5.0, 5.0]            # the default per-recipe timeout
    assert len(uploads) == 4 and 'smoke.json' in uploads[0]   # after every recipe, then the final report
    doc = json.loads((out / 'smoke' / 'smoke.json').read_text())
    status = {r['id']: r['status'] for r in doc['rows'].values()}
    assert status == {ids[0]: 'MATCHED', ids[1]: 'REFUSED', ids[2]: 'STARTED'}
    text = (out / 'smoke' / 'smoke.txt').read_text()
    assert 'CHECK data failed: dataset taxi missing' in text
    assert '== maximize missing from ours read-back (1)' in text and ids[1] in text
    started = text.split('STARTED, NO VERDICT')[1]
    assert ids[2] in started and ids[0] not in started
    assert not (out / 'campaign.json').exists()                # a smoke never starts a timed campaign
    with pytest.raises(ValueError):
        C.main(['--vendor', 'amd', '--manifest', str(manifest), '--out', str(out), '--cache', str(tmp_path),
                '--smoke'])


def test_smoke_timeout_flag(tmp_path, monkeypatch):
    manifest = tmp_path / 'manifest.json'
    manifest.write_text(json.dumps({'amd': ['algos/adam/synthetic/rows=full']}))
    monkeypatch.setattr(C, 'smoke_stage', lambda *a, **k: {'status': 'ok', 'problems': []})
    monkeypatch.setattr(C, 'smoke_checks', lambda args, plans, out, warm=None: ({}, {}))
    monkeypatch.setattr(C, 'smoke_stage', lambda *a, **k: {'status': 'ok', 'problems': []})
    seen = []
    monkeypatch.setattr(C, 'run_bounded', lambda cmd, log, timeout, env=None: seen.append(timeout) or (None, timeout))
    monkeypatch.setattr(C, 'retain', lambda archive, members: {'sha256': 'x'})
    C.main(['--vendor', 'amd', '--manifest', str(manifest), '--out', str(tmp_path / 'o'), '--cache', str(tmp_path),
            '--smoke', '--smoke-timeout', '12', '--smoke-warm', 'off'])
    assert seen == [12.0]


def test_problem_kinds_group_by_problem_not_by_arm():
    assert C.problem_kind('torch-eager-fp32: amsgrad missing from ours read-back; explicit semantic mapping '
                          'required') == 'amsgrad missing from ours read-back'
    assert C.problem_kind('missing parameter/version probe: torch-compile-bf16') == \
        'missing parameter/version probe: <arm>'
    assert C.problem_kind('quality scorer unavailable: reference data: the block holds no yq') == \
        'quality scorer unavailable: reference data'


def test_smoke_stages_every_recipes_block_prep_once_before_the_loop(tmp_path, monkeypatch):
    ids = ['algos/adam/synthetic/rows=full', 'classical/knn/taxi/rows=full']
    plans = {'identical': [r for r in C.selected_plan('amd', {'amd': ids}, smoke=True)]}
    args = SimpleNamespace(python_env='py', vendor='amd', cache=str(tmp_path), wheel_manifest=None,
                           smoke_stage_timeout=99.0)
    seen = []
    monkeypatch.setattr(C, 'run_bounded', lambda cmd, log, timeout, env=None: seen.append((cmd, timeout)) or (0, 1.0))
    chk = C.smoke_stage(args, plans, tmp_path, {}, python='/warm/bin/python')
    assert chk['status'] == 'ok' and len(seen) == 1
    cmd, timeout = seen[0]
    assert timeout == 99.0 and cmd[0] == '/warm/bin/python' and '--prep-only' in cmd and '--preflight-only' in cmd
    assert cmd[cmd.index('--families') + 1] == 'algos,classical'
    assert cmd[cmd.index('--race-ids') + 1].startswith('@')
    assert sorted((tmp_path / 'stage-race-ids.identical.txt').read_text().split()) == sorted(ids)
    assert '--datasets' not in cmd
