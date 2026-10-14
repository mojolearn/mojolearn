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


class FakeWorker:
    """Stands in for C.SmokeWorker: `behave(argv, log, timeout)` -> (reply or None, restart or None)."""
    behave = None
    instances = []

    def __init__(self, args, out, env, worker_cmd=None):
        self.report = dict(status='serving', import_seconds=12.5, gpu_init_seconds=1.5, restarts=[],
                           starts=[{'import_seconds': 12.5, 'gpu_init_seconds': 1.5, 'report': {'imports': {}}}])
        self.recipes, self.asked, self.staged = [], [], []
        FakeWorker.instances.append(self)

    def start(self):
        return self.report

    def alive(self):
        return True

    def ask(self, request, timeout):
        self.asked.append(request['op'])
        return {'event': request['op'], 'facts': {'box_fingerprint': {'status': 'ok', 'seconds': 2.0}}}

    def warm(self, request, timeout):
        return self.ask(request, timeout)['facts']

    def stage(self, request, timeout):
        # the next recipe's inputs staged before its timer (smoke-fixes r12): order recorded with the recipes
        self.staged.append((request['key'], len(self.recipes)))
        return {'items': [{'lane': i['lane'], 'status': 'ok', 'seconds': 0.1} for i in request['items']]}

    def recipe(self, key, argv, log, timeout):
        self.recipes.append((key, argv, timeout))
        reply, restart = FakeWorker.behave(argv, log, timeout)
        if restart:
            self.report['restarts'].append(restart)
        return reply, restart

    def kill(self):
        pass

    def stop(self):
        pass


def _fake_smoke(monkeypatch, behave):
    FakeWorker.behave = staticmethod(behave)
    FakeWorker.instances = []
    monkeypatch.setattr(C, 'SmokeWorker', FakeWorker)
    monkeypatch.setattr(C, 'worker_env', lambda args: {})
    monkeypatch.setattr(C, 'smoke_stage', lambda *a, **k: {'status': 'ok', 'problems': []})
    monkeypatch.setattr(C, 'smoke_fixtures', lambda *a, **k: {'status': 'ok', 'problems': []})


def test_smoke_writes_rows_text_and_uploads_after_every_recipe(tmp_path, monkeypatch):
    ids = ['algos/adam/synthetic/rows=full', 'algos/adamw/synthetic/rows=full', 'algos/sgd/synthetic/rows=full',
           'algos/rmsprop/synthetic/rows=full', 'algos/adagrad/synthetic/rows=full']
    manifest = tmp_path / 'manifest.json'
    manifest.write_text(json.dumps({'amd': ids}))

    def behave(argv, log, timeout):
        rid = argv[argv.index('--race-ids') + 1]
        out = argv[argv.index('--out') + 1]
        C.Path(log).write_text('bench_board: setup\n')
        if rid == ids[0]:
            _write_board(out, rid, {'verdict': 'MATCHED', 'problems': []})
            return dict(status='done', rc=0, seconds=1.0), None
        if rid == ids[1]:
            _write_board(out, rid, {'verdict': 'REFUSED', 'problems': [
                'torch-eager-fp32: maximize missing from ours read-back; explicit semantic mapping required']})
            return dict(status='done', rc=1, seconds=2.0), None
        if rid == ids[2]:          # the recipe's own alarm fired: STARTED, and the loop went on
            C.Path(log).write_text('bench_board: setup\nbench_board: [1/1] params probe\n')
            return dict(status='timeout', rc=None, seconds=timeout,
                        stopped_at=['tools/bench_board.py:2700 params_probe']), None
        if rid == ids[3]:          # the check raised: FAILED with the exception, and the loop went on
            return dict(status='error', rc=None, error="KeyError('lr')", seconds=0.3), None
        # the worker never answered: the watchdog restarted it; the row is STARTED with the restart
        return None, dict(after='identical|' + rid, reason='no answer within 35.0s', started=True)
    _fake_smoke(monkeypatch, behave)
    monkeypatch.setattr(C, 'smoke_checks', lambda args, plans, out, worker=None: (
        {'wheel': {'status': 'ok', 'problems': []}, 'data': {'status': 'failed', 'problems': ['dataset taxi missing']}},
        {}))
    uploads = []
    monkeypatch.setattr(C, 'retain', lambda archive, members: uploads.append(sorted(members)) or {'sha256': 'x'})
    out = tmp_path / 'campaign'
    rc = C.main(['--vendor', 'amd', '--manifest', str(manifest), '--out', str(out), '--cache', str(tmp_path),
                 '--smoke'])
    assert rc == 0
    worker = FakeWorker.instances[-1]
    assert [t for _, _, t in worker.recipes] == [5.0] * 5           # the default per-recipe timer
    assert all(argv[0] == '--vendor' for _, argv, _ in worker.recipes)  # bench_board argv, no interpreter
    assert worker.asked == ['warm-box']                            # box facts once, before the loop
    # every algos recipe's inputs staged right before its own timer starts (smoke-fixes r12)
    assert worker.staged == [(rid, i) for i, rid in enumerate(ids)]
    assert len(uploads) == 6 and 'smoke.json' in uploads[0]        # after every recipe, then the final report
    doc = json.loads((out / 'smoke' / 'smoke.json').read_text())
    status = {r['id']: r['status'] for r in doc['rows'].values()}
    assert status == {ids[0]: 'MATCHED', ids[1]: 'REFUSED', ids[2]: 'STARTED', ids[3]: 'FAILED', ids[4]: 'STARTED'}
    failed = next(r for r in doc['rows'].values() if r['id'] == ids[3])
    assert "KeyError('lr')" in failed['problems'][0]
    hung = next(r for r in doc['rows'].values() if r['id'] == ids[4])
    assert hung['worker']['answered'] is False and hung['worker']['restart']['started']
    assert doc['worker']['import_seconds'] == 12.5 and doc['worker']['gpu_init_seconds'] == 1.5
    assert len(doc['worker']['restarts']) == 1
    text = (out / 'smoke' / 'smoke.txt').read_text()
    assert 'WORKER serving import_s=12.5 gpu_init_s=1.5 starts=1 restarts=1' in text
    assert 'restarted after identical|' + ids[4] in text
    assert 'CHECK data failed: dataset taxi missing' in text
    assert '== maximize missing from ours read-back (1)' in text and ids[1] in text
    started = text.split('STARTED, NO VERDICT')[1].split('FAILED TO START')[0]
    assert ids[2] in started and ids[4] in started and ids[0] not in started
    assert 'params probe' in started                                # how far it got
    assert 'stopped in tools/bench_board.py:2700 params_probe' in started
    assert ids[3] in text.split('FAILED TO START')[1]
    assert not (out / 'campaign.json').exists()                     # a smoke never starts a timed campaign
    with pytest.raises(ValueError):
        C.main(['--vendor', 'amd', '--manifest', str(manifest), '--out', str(out), '--cache', str(tmp_path),
                '--smoke'])


def test_smoke_timeout_flag(tmp_path, monkeypatch):
    manifest = tmp_path / 'manifest.json'
    manifest.write_text(json.dumps({'amd': ['algos/adam/synthetic/rows=full']}))
    _fake_smoke(monkeypatch, lambda argv, log, timeout: (dict(status='timeout', rc=None, seconds=timeout), None))
    monkeypatch.setattr(C, 'smoke_checks', lambda args, plans, out, worker=None: ({}, {}))
    monkeypatch.setattr(C, 'retain', lambda archive, members: {'sha256': 'x'})
    C.main(['--vendor', 'amd', '--manifest', str(manifest), '--out', str(tmp_path / 'o'), '--cache', str(tmp_path),
            '--smoke', '--smoke-timeout', '12'])
    assert [t for _, _, t in FakeWorker.instances[-1].recipes] == [12.0]


def test_smoke_row_reads_the_preflight_checkpoint_when_the_board_json_is_not_written(tmp_path):
    race = dict(id='algos/adam/synthetic/rows=full', family='algos', lane='adam', dataset='synthetic')
    d = tmp_path / 'r'
    C.E.write(d / 'evidence' / 'preflight.json', {'races': {race['id']: {'audit': {'verdict': 'MATCHED',
                                                                                      'problems': []}}}})
    row = C.smoke_row(race, 'identical', None, 5.0, d, tmp_path / 'none.log', 5)
    assert row['status'] == 'MATCHED'


def test_worker_check_classifies_the_load_report():
    report = {'starts': [{'import_seconds': 30.0, 'gpu_init_seconds': 2.0, 'report': {
        'imports': {'torch': {'status': 'ok'}, 'cuml': {'status': 'absent', 'error': "No module named 'cuml'"},
                    'cuvs': {'status': 'failed', 'error': 'libopenblas.so.0: cannot open'},
                    'bench_board_algos.py': {'status': 'ok', 'kind': 'driver'}},
        'gpu_init': {'torch': {'status': 'ok'}, 'cupy': {'status': 'failed', 'error': 'no device'}}}}]}
    chk = C.worker_check(report)
    assert chk['status'] == 'failed' and chk['absent'] == ['cuml']
    assert any(p.startswith('cuvs does not import') for p in chk['problems'])
    assert any('cupy did not initialize' in p for p in chk['problems'])
    assert not any('cuml' in p for p in chk['problems'])
    ok = C.worker_check({'starts': [{'report': {'imports': {'torch': {'status': 'ok'}}, 'gpu_init': {}}}]})
    assert ok['status'] == 'ok' and ok['problems'] == []


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
    chk = C.smoke_stage(args, plans, tmp_path, {})
    assert chk['status'] == 'ok' and len(seen) == 1
    cmd, timeout = seen[0]
    assert timeout == 99.0 and cmd[0] == 'py' and '--prep-only' in cmd and '--preflight-only' in cmd
    assert cmd[cmd.index('--families') + 1] == 'algos,classical'
    assert cmd[cmd.index('--race-ids') + 1].startswith('@')
    assert sorted((tmp_path / 'stage-race-ids.identical.txt').read_text().split()) == sorted(ids)
    assert '--datasets' not in cmd


def test_smoke_generates_missing_neural_fixtures_once_then_verifies(tmp_path, monkeypatch):
    # smoke-fixes (2026-10-14): the Apple smoke had no fixture set, so every layer recipe refused; the smoke
    # now stages the set itself (the timed campaign's fixture_commands), outside every recipe's timer
    monkeypatch.delenv('MOJOLEARN_NEURAL_FIXTURES', raising=False)
    races = [dict(family='algos', lane='layernorm', dataset='synthetic'),
             dict(family='algos', lane='gcn', dataset='taxi'),
             dict(family='algos', lane='adam', dataset='synthetic')]
    args = SimpleNamespace(python_env='py', cache=str(tmp_path), smoke_stage_timeout=99.0)
    fixdir = tmp_path / 'algos-data' / 'neural-fixtures'
    ran, verified = [], []

    def run(cmd, log, timeout, env=None):
        ran.append((cmd[1].rsplit('/', 1)[-1], cmd[2], timeout))
        if cmd[2] == 'generate':
            fixdir.mkdir(parents=True, exist_ok=True)
            (fixdir / 'manifest.json').write_text(json.dumps({'torch_version': 't', 'entries': {
                'layernorm/synthetic/nb64': {'lane': 'layernorm', 'dataset': 'synthetic'},
                'gcn/taxi/n9': {'lane': 'gcn', 'dataset': 'taxi'}}}))
        return 0, 1.0
    monkeypatch.setattr(C, 'run_bounded', run)
    monkeypatch.setattr(C, 'bounded_check', lambda cmd, timeout=120: verified.append(cmd) or (True, 'NF-VERIFY ok'))
    chk = C.smoke_fixtures(args, {'identical': races, 'fast': races}, tmp_path)
    assert chk['status'] == 'ok' and chk['generated'] and chk['entries'] == 2, chk
    assert ran == [('bench_board_algos.py', 'prep', 99.0), ('neural_fixtures.py', 'generate', 99.0)]
    gen = [c for c in C.fixture_commands(args, chk['generated_lanes'], fixdir) if c[2] == 'generate'][0]
    assert gen[gen.index('--lanes') + 1] == 'gcn,layernorm'
    assert verified and verified[-1][2] == 'verify' and verified[-1][-1] == str(fixdir)
    ran.clear()
    chk = C.smoke_fixtures(args, {'identical': races}, tmp_path)   # present and complete: verify only
    assert chk['status'] == 'ok' and not chk['generated'] and ran == []
    chk = C.smoke_fixtures(args, {'identical': races + [dict(family='algos', lane='conv1d', dataset='synthetic')]},
                           tmp_path)                                # an entry missing: generate again (refuses: the
    assert chk['generated'] and chk['status'] == 'failed'          # fake generate cannot write conv1d)
    assert any('conv1d/synthetic' in p for p in chk['problems'])


def test_stage_request_names_the_algos_block_dir_and_the_race_arm_interpreters(tmp_path):
    """smoke-fixes r12: an algos recipe's stage-data request (its block dir under --cache, rows=full, and the arm
    servers of its own arms only); other families stage nothing."""
    args = SimpleNamespace(cache=str(tmp_path / "cache"))
    race = dict(id='algos/als/text/rows=full', family='algos', lane='als', dataset='text',
                arms=['ours', 'implicit-gpu'])
    req = C.stage_request(args, race, {'implicit-gpu': '/v/implicit/bin/python', 'cuml-gpu': '/v/cuml/bin/python'})
    assert req['op'] == 'stage-data' and req['key'] == race['id']
    assert req['arm_pythons'] == ['/v/implicit/bin/python']
    item, = req['items']
    assert item['lane'] == 'als' and item['dataset'] == 'text'
    assert item['data'] == str(tmp_path / 'cache' / 'algos-data' / 'rows-full')
    assert item['driver'].endswith('tools/bench_board_algos.py')
    assert C.stage_request(args, dict(race, family='neural'), {}) is None


def test_stage_request_stages_classical_classical2_and_trees_through_their_drivers(tmp_path):
    """smoke-fixes r14: the classical, classical2 and trees recipes are staged too, each by its own driver's
    stage_inputs, from the block dir the board's command reads (rows=full) or the forest driver's dataset."""
    args = SimpleNamespace(cache=str(tmp_path / "cache"))
    cache = tmp_path / 'cache'
    for family, driver, data in (('classical', 'tools/classical_two_datasets.py', cache / 'ctd-data' / 'rows-full'),
                                 ('classical2', 'tools/bench_board_more.py', cache / 'more-data' / 'rows-full')):
        race = dict(id='%s/x/istella/rows=full' % family, family=family, lane='kmeans', dataset='istella',
                    arms=['ours', 'cuml-gpu'])
        item, = C.stage_request(args, race, {})['items']
        assert item['driver'].endswith(driver) and item['data'] == str(data) and item['dataset'] == 'istella'
    race = dict(id='trees/gbdt-multiclass/istella/rows=full', family='trees', lane='gbdt-multiclass',
                dataset='istella', arms=['ours', 'catboost-gpu'], opponents=['catboost-gpu'])
    item, = C.stage_request(args, race, {})['items']
    assert item['driver'].endswith('bench/speed/forest_speed_arm.py')
    assert item['dataset'] == C.B.tree_driver_dataset('gbdt-multiclass', 'istella')
    assert item['data'] == dict(size='shipped', rows=None, arms=['catboost-gpu'])


def test_smoke_worker_stage_restarts_a_worker_that_does_not_answer(tmp_path):
    """A stage-data request with no answer kills and restarts the worker (recorded), and the recipe goes on."""
    args = SimpleNamespace(smoke_load_timeout=5, smoke_grace=1)
    w = C.SmokeWorker(args, tmp_path, {}, worker_cmd=['true'])
    calls = []
    w.ask = lambda request, timeout: None
    w.kill = lambda: calls.append('kill')
    w.start = lambda: calls.append('start')
    w.alive = lambda: True
    got = w.stage(dict(op='stage-data', key='algos/adam/synthetic/rows=full', items=[]), 30)
    assert got['status'] == 'failed' and 'no stage-data answer within 30s' in got['problem']
    assert calls == ['kill', 'start'] and w.report['restarts'][0]['after'].startswith('stage-data ')


def test_only_the_slow_table_recipes_get_a_longer_timer_and_rows_record_it():
    # Andrew 2026-10-14: default 5 s; a longer timer only for the recipes in SMOKE_SLOW, never global
    args = SimpleNamespace(smoke_timeout=5.0)
    slow = {'id': 'algos/moe/synthetic/rows=full'}
    plain = {'id': 'algos/adam/synthetic/rows=full'}
    assert C.recipe_timeout(args, plain) == (5.0, 'default --smoke-timeout')
    seconds, why = C.recipe_timeout(args, slow)
    assert seconds == C.SMOKE_SLOW[slow['id']][0] > 5.0 and why
    assert all(isinstance(v, tuple) and v[0] > 5.0 and v[1] for v in C.SMOKE_SLOW.values())
    # an explicit longer --smoke-timeout is never shortened by the table
    assert C.recipe_timeout(SimpleNamespace(smoke_timeout=60.0), slow) == (60.0, 'default --smoke-timeout')


def test_mamba_ssm_is_prepared_once_per_box_only_when_a_planned_arm_needs_it(tmp_path, monkeypatch):
    import install_mamba_ssm as M
    calls = []
    monkeypatch.setattr(M, 'install', lambda python, vendor, log, wheels=None: calls.append((python, vendor)) or
                        {'status': 'failed', 'problem': 'no hipcc under /opt/rocm'})
    args = SimpleNamespace(python_env='py', vendor='amd')
    neural = [{'id': 'neural/mamba1-forward/gaussian/shape=full', 'family': 'neural', 'lane': 'mamba1-forward',
               'arms': ['ours', 'mamba-ssm-fp32']}]
    algos = [{'id': 'algos/adam/synthetic/rows=full', 'family': 'algos', 'lane': 'adam', 'arms': ['ours']}]
    assert C.prepare_mamba_ssm(args, algos, tmp_path)['status'] == 'ok' and calls == []
    rec = C.prepare_mamba_ssm(args, neural, tmp_path)
    assert calls == [('py', 'amd')] and rec['status'] == 'failed' and rec['problems'] == ['no hipcc under /opt/rocm']
    assert C.prepare_mamba_ssm(SimpleNamespace(python_env='py', vendor='apple'), neural, tmp_path)['status'] == 'ok'


def test_install_mamba_ssm_does_nothing_when_both_import(monkeypatch, tmp_path):
    import install_mamba_ssm as M
    monkeypatch.setattr(M, 'imports', lambda python: (True, '2.3.0'))
    monkeypatch.setattr(M.B, 'run_logged', lambda *a, **k: (_ for _ in ()).throw(AssertionError('no build')))
    assert M.install('py', 'amd', str(tmp_path / 'log'))['installed'] is False
