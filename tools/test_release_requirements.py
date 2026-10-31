# SPDX-License-Identifier: Apache-2.0
import copy
import json
import subprocess
from concurrent.futures import ThreadPoolExecutor

import pytest

import release_requirements as R
PRODUCTION_PRODUCER_ADMISSION = R.producer_admission_available


@pytest.fixture(autouse=True)
def synthetic_producer_only(monkeypatch):
    # These unit tests exercise ledger/proof mechanics with synthetic reports.
    # Production has no fixture producer exemption; its producer parsers have
    # their own tests. All receipt/input/evidence/coverage checks remain real.
    import release_requirement_receipts as P
    monkeypatch.setattr(P, "validate_producer_receipt", lambda *args, **kwargs: [], raising=False)
    monkeypatch.setattr(R, "producer_admission_available", lambda wanted: True)


def manifest(revision="a", arithmetic="one", compiler="1", flags="identical"):
    result = dict(schema=R.SCHEMA, version="0.8.57", source_revision=revision, requirements={})
    for name, source in (("identity", arithmetic), ("metadata", "constant")):
        inputs = dict(arithmetic={name + ".mojo": source}, compiler=compiler, flags=flags)
        result["requirements"][name] = dict(source_revision=revision, fingerprint=R.digest(inputs),
                                           inputs=inputs, scope=name, coverage=["whole operation"],
                                           kind="source", stage="pre", missing_inputs=[], required=True)
    return result


@pytest.fixture
def ledger(tmp_path):
    led = R.RequirementsLedger(tmp_path / "requirements.json")
    led.sync(manifest())
    return led


def record(ledger, tmp_path, name="identity", status="passed", reason=None):
    invalidation = [invalidation_evidence(tmp_path, reason)] if reason and ledger.plan()[name]['action'] == 'reuse' else []
    run = ledger.begin(name, reason, invalidation_paths=invalidation)
    path = tmp_path / (run + ".log")
    path.write_text(status)
    ledger.record(run, status, [path], "whole operation checked", 0 if status == "passed" else 1,
                  2, 2 if status == "passed" else 1, failed=int(status == "failed"),
                  receipt=receipt_for(ledger, run, path, expected=2) if status == "passed" else None)
    return path


def invalidation_evidence(tmp_path, reason):
    path = tmp_path / ('invalidation-' + R.digest(reason)[:12] + '.log')
    path.write_text(reason)
    return path


def receipt_for(ledger, run_id, path, expected=1):
    import release_requirement_receipts as P
    run = next(r for r in ledger.read()["runs"] if r["id"] == run_id)
    wanted = run["manifest"]
    return dict(schema=P.SCHEMA, requirement_id=run["requirement"], kind=wanted["kind"],
                source_revision=wanted["source_revision"], input_fingerprint=wanted["fingerprint"],
                producer=dict(tool="fixture", report_type="source" if wanted["kind"] == "source" else "build" if wanted["kind"] == "compile" else "execution"), status="passed", exit_code=0,
                coverage=[dict(id=name, status="passed", exit_code=0, expected=expected,
                               completed=expected, failed=0, skipped=0) for name in wanted["coverage"]],
                evidence=R.evidence_files([path]))


def test_new_commit_only_reuses_pass_and_preserves_tested_revision(ledger, tmp_path):
    record(ledger, tmp_path)
    ledger.sync(manifest(revision="new"))
    row = ledger.plan()["identity"]
    assert row["status"] == "passed" and row["action"] == "reuse"
    assert row["tested_revision"] == "a" and row["source_revision"] == "new"
    assert ledger.begin("identity") is None
    with pytest.raises(ValueError, match='new commit or free-text reason alone'):
        ledger.begin("identity", reason="a new commit exists")
    assert ledger.begin("identity", reason="   ") is None
    assert len(ledger.read()["runs"]) == 1


@pytest.mark.parametrize("field,value,reason", [("arithmetic", "two", "arithmetic/identity.mojo"),
                                               ("compiler", "2", "compiler"),
                                               ("flags", "other", "flags")])
def test_relevant_inputs_invalidate_with_specific_reason(ledger, tmp_path, field, value, reason):
    record(ledger, tmp_path)
    record(ledger, tmp_path, "metadata")
    ledger.sync(manifest(**{field: value}))
    row = ledger.plan()["identity"]
    assert row["status"] == "untested" and row["reasons"] == ["changed input: " + reason]
    if field == "arithmetic":
        assert ledger.plan()["metadata"]["status"] == "passed"
    assert len(ledger.read()["attempts"]) == 2


def test_missing_and_corrupt_evidence_require_restore_not_blind_rerun(ledger, tmp_path):
    path = record(ledger, tmp_path)
    path.write_text("tampered")
    assert ledger.plan()["identity"]["action"] == "restore-evidence"
    with pytest.raises(ValueError, match="restore provenance"):
        ledger.begin("identity")
    path.unlink()
    assert "evidence unavailable" in ledger.plan()["identity"]["reasons"][0]
    assert not ledger.gate()[0]


def test_later_failure_cannot_be_hidden_by_old_pass(ledger, tmp_path):
    record(ledger, tmp_path)
    record(ledger, tmp_path, status="failed", reason="negative control exposed wrong-device routing")
    assert ledger.plan()["identity"]["status"] == "failed"
    assert ledger.begin("identity") is not None
    assert ledger.read()["runs"][-1]["reasons"][0].startswith("previous attempt failed")


@pytest.mark.parametrize("kwargs", [dict(exit_code=1), dict(expected=0, completed=0),
                                    dict(completed=1), dict(failed=1), dict(skipped=1)])
def test_incomplete_or_failed_coverage_never_passes(ledger, tmp_path, kwargs):
    run = ledger.begin("identity")
    path = tmp_path / "report.log"
    path.write_text("summary")
    values = dict(exit_code=0, expected=2, completed=2)
    values.update(kwargs)
    with pytest.raises(ValueError, match="complete coverage"):
        ledger.record(run, "passed", [path], "checks", **values)


def test_inflight_old_source_result_does_not_qualify_changed_source(ledger, tmp_path):
    run = ledger.begin("identity")
    ledger.sync(manifest(revision="b", arithmetic="two"))
    path = tmp_path / "old.log"
    path.write_text("passed old source")
    ledger.record(run, "passed", [path], "old source", 0, 1, 1, receipt=receipt_for(ledger, run, path))
    assert ledger.plan()["identity"]["status"] == "untested"


def test_readonly_plan_does_not_rewrite_or_create_ledger(ledger, tmp_path):
    before = ledger.path.read_bytes()
    ledger.plan(manifest(revision="b", arithmetic="two"))
    assert ledger.path.read_bytes() == before
    absent = R.RequirementsLedger(tmp_path / "absent.json")
    assert not absent.gate()[0] and not absent.path.exists()


def test_parallel_producers_preserve_both_attempts(ledger, tmp_path):
    with ThreadPoolExecutor(2) as pool:
        list(pool.map(lambda name: record(ledger, tmp_path, name), ["identity", "metadata"]))
    assert len(ledger.read()["attempts"]) == 2
    assert ledger.gate() == (True, [])


def test_missing_configuration_cannot_admit_a_pass(ledger, tmp_path):
    target = manifest()
    target["requirements"]["identity"]["missing_inputs"] = ["compiler"]
    ledger.sync(target)
    assert ledger.plan()["identity"]["action"] == "complete-provenance"
    with pytest.raises(ValueError, match="missing input"):
        ledger.begin("identity")


def test_git_snapshot_ignores_docs_and_tracks_additions_deletions(tmp_path):
    def git(*args):
        return subprocess.check_output(["git", "-C", str(tmp_path), *args], text=True).strip()
    git("init", "-q")
    git("config", "user.name", "Test")
    git("config", "user.email", "test@example.invalid")
    (tmp_path / "a.mojo").write_text("a")
    (tmp_path / "README").write_text("old")
    git("add", ".")
    git("commit", "-qm", "initial")
    catalog = {"requirements": [dict(id="id", scope="source", coverage=["source"],
                                     source_inputs={"arithmetic": ["*.mojo"]})]}
    a = R.requirements(tmp_path, git("rev-parse", "HEAD"), "v", catalog)
    (tmp_path / "README").write_text("new docs")
    git("commit", "-qam", "docs")
    b = R.requirements(tmp_path, git("rev-parse", "HEAD"), "v", catalog)
    assert a["requirements"]["id"]["fingerprint"] == b["requirements"]["id"]["fingerprint"]
    (tmp_path / "b.mojo").write_text("b")
    git("add", ".")
    git("commit", "-qm", "new arithmetic")
    c = R.requirements(tmp_path, "HEAD", "v", catalog)
    assert R.changes(b["requirements"]["id"]["inputs"], c["requirements"]["id"]["inputs"]) == ["added input: arithmetic/b.mojo"]


def test_record_is_immutable(ledger, tmp_path):
    path = record(ledger, tmp_path)
    run = ledger.read()["runs"][0]["id"]
    with pytest.raises(ValueError, match="already recorded"):
        ledger.record(run, "passed", [path], "second result", 0, 1, 1)


def test_flags_use_existing_cache_policy_and_include_linker_options():
    flags = R.build_flags({"MOJOLEARN_COMMIT": "new", "MOJOLEARN_COMPILE_JOBS": "2",
                           "MOJOLEARN_BUILD_JOBS": "16", "CXXFLAGS": "-DNEW=1"})
    assert flags == {"MOJOLEARN_COMPILE_JOBS": "2", "CXXFLAGS": "-DNEW=1"}


def test_provisioning_credentials_never_enter_build_fingerprints():
    secret_names = ["MOJOLEARN_RUNPOD_KEY_FILE", "MOJOLEARN_API_TOKEN", "MOJOLEARN_PASSWORD",
                    "MOJOLEARN_SECRET", "MOJOLEARN_ACCESS_KEY", "MOJOLEARN_PRIVATE_KEY",
                    "MOJOLEARN_AUTH_TOKEN_FILE", "MOJOLEARN_PRIVATE_KEY_PATH"]
    environment = dict.fromkeys(secret_names, "credential must not be retained")
    environment.update(MOJOLEARN_NUMERIC_MODE="identical", MOJOLEARN_GPU_ARCHS="sm_89",
                       CXXFLAGS="-DREDUCTION_ORDER=1")
    assert R.build_flags(environment) == dict(MOJOLEARN_NUMERIC_MODE="identical",
        MOJOLEARN_GPU_ARCHS="sm_89", CXXFLAGS="-DREDUCTION_ORDER=1")


def test_explicit_redo_requires_a_specific_reason(tmp_path):
    import release
    from test_release import args
    ctx = release.Release(args(state_dir=str(tmp_path), redo="rehearsal", only="rehearsal"), runner=lambda *a: 0)
    with pytest.raises(SystemExit, match="--rerun-reason"):
        ctx.go()
    ctx.args.rerun_reason = "fixed receipt parser rejected the retained complete report"
    ctx.schedule = lambda steps: {"rehearsal": "done"}
    ctx.report = lambda status: True
    ctx.go()
    assert json.loads(ctx.state_path.read_text())["rerun_requests"][0]["reason"] == ctx.args.rerun_reason


def test_raw_log_and_aggregate_counts_cannot_create_a_pass(ledger, tmp_path):
    run = ledger.begin("identity")
    log = tmp_path / "source-only.log"
    log.write_text("one source test passed")
    with pytest.raises(ValueError, match="structured release requirement"):
        ledger.record(run, "passed", [log], "not an execution receipt", 0, 1, 1)
    assert ledger.plan()["identity"]["status"] == "untested"


def test_prepublication_does_not_require_future_postpublication_checks(ledger, tmp_path):
    target = manifest()
    target["requirements"]["metadata"]["stage"] = "post"
    ledger.sync(target)
    record(ledger, tmp_path)
    assert ledger.gate(stage="pre") == (True, [])
    assert not ledger.gate()[0] and not ledger.gate(stage="post")[0]


def test_producer_evidence_is_revalidated_when_reusing_a_pass(ledger, tmp_path):
    path = record(ledger, tmp_path)
    doc = ledger.read()
    doc["attempts"][0]["receipt"]["producer"]["report_type"] = "execution"
    ledger.path.write_text(json.dumps(doc))
    assert ledger.plan()["identity"]["status"] == "untested"


def test_unfinished_run_prevents_duplicate_launches(ledger):
    run = ledger.begin("identity")
    assert ledger.plan()["identity"]["active_run"] == run
    with pytest.raises(ValueError, match="duplicate work"):
        ledger.begin("identity")
    assert len(ledger.read()["runs"]) == 1


def test_explicit_rerun_holds_readiness_until_its_result(ledger, tmp_path):
    record(ledger, tmp_path)
    reason = "new negative control may expose invalid retained result"
    proof = invalidation_evidence(tmp_path, reason)
    ledger.begin("identity", reason=reason, invalidation_paths=[proof])
    assert ledger.plan()["identity"]["status"] == "untested"
    assert ledger.plan()["identity"]["action"] == "resume"


def test_rerun_retains_invalidation_bytes_and_refuses_changed_diagnostic(ledger, tmp_path):
    record(ledger, tmp_path)
    reason = 'negative control exposed a missed routing failure'
    proof = invalidation_evidence(tmp_path, reason)
    run = ledger.begin('identity', reason=reason, invalidation_paths=[proof])
    diagnostic = ledger.read()['runs'][-1]['invalidation_evidence']
    assert diagnostic == R.evidence_files([proof])
    report = tmp_path / 'rerun-failure.log'
    report.write_text('failure reproduced')
    proof.write_text('replaced original diagnostic')
    with pytest.raises(ValueError, match='invalidation evidence requires restoration'):
        ledger.record(run, 'failed', [report], 'routing failure reproduced', 1, 1, 0, failed=1)
    proof.write_text(reason)
    attempt = ledger.record(run, 'failed', [report], 'routing failure reproduced', 1, 1, 0, failed=1)
    assert diagnostic[0] in attempt['evidence']


def test_explicit_reason_does_not_hide_computed_invalidation(ledger, tmp_path):
    record(ledger, tmp_path)
    ledger.sync(manifest(arithmetic='changed'))
    ledger.begin('identity', reason='verify corrected arithmetic')
    reasons = ledger.read()['runs'][-1]['reasons']
    assert 'changed input: arithmetic/identity.mojo' in reasons
    assert 'verify corrected arithmetic' in reasons


def native_manifest(tmp_path, revision='a' * 40, source='old', flags=None):
    inputs = dict(native_sources={'core/shared.mojo': source},
        unit=dict(vendor='nvidia', lanes=['two'], build_flags=flags or {}),
        execution_environment={'target': 'linux-64', 'compiler': 'fixed'},
        contract=dict(id='identity', kind='identity', stage='pre', scope='identity', coverage=['whole operation']))
    wanted = dict(source_revision=revision, fingerprint=R.digest(inputs), inputs=inputs,
        scope='identity', coverage=['whole operation'], kind='identity', stage='pre', missing_inputs=[], required=True,
        parent_requirement='identity-nvidia', dependency_limitations={}, provenance_sources=['core/shared.mojo'])
    return dict(schema=R.SCHEMA, version='0.8.57', source_revision=revision, source_root=str(tmp_path),
                requirements={'identity': wanted})


@pytest.fixture
def native_ledger(tmp_path, monkeypatch):
    import release_requirement_dependencies as D
    calls, selected = [], dict(lanes=['one'], unattributed=[], precision_limitations=[])
    def snapshot(root, revision, request, **kwargs):
        calls.append((revision, copy.deepcopy(request)))
        selection = dict(copy.deepcopy(selected), base_revision=request['base_revision'], source_revision=revision,
                         changed_paths=['core/shared.mojo'], selector_sources={'tools/lane_select.py': 'frozen-selector'})
        return {'selections': {'nvidia': selection}}
    monkeypatch.setattr(D, 'frozen_snapshot', snapshot)
    monkeypatch.setattr(D, '_revision', lambda root, revision: revision)
    led = R.RequirementsLedger(tmp_path / 'native-ledger.json')
    led.sync(native_manifest(tmp_path))
    return led, calls, selected


def test_default_catalog_expands_units_and_retains_context(tmp_path, monkeypatch):
    import release_requirement_dependencies as D
    raw = {'requirements': [dict(id='compile-cpu', scope='compile', kind='compile', coverage=['build'])]}
    (tmp_path / 'tools').mkdir()
    (tmp_path / 'tools/release_requirements.json').write_text(json.dumps(raw))
    calls = []
    def expand(root, revision, catalog, configuration, **kwargs):
        calls.append((catalog, configuration, kwargs))
        return dict(requirements=[dict(id='compile-cpu/host', scope='compile', kind='compile', coverage=['host'],
            parent_requirement='compile-cpu', inherit_source_inputs=False,
            resolved_inputs={'binding': {'target': 'osx-arm64'}}, provenance_sources=['a.mojo'],
            dependency_limitations={'host': ['conservative']})],
            dependency_provenance=dict(schema=D.SCHEMA, source_revision='f' * 40))
    monkeypatch.setattr(D, 'expand_catalog', expand)
    monkeypatch.setattr(R, 'source_files', lambda root, revision: {'a.mojo': 'blob'})
    config = dict(build_flags={}, dependency_context=dict(cpu_target='osx-arm64', apple_toolchain={'clang': 'pin'}, workloads={}))
    result = R.requirements(tmp_path, 'HEAD', 'v', configuration=config, evidence_log_dir=tmp_path / 'logs')
    row = result['requirements']['compile-cpu/host']
    assert len(calls) == 1 and calls[0][0] == raw and calls[0][1] == config
    assert row['parent_requirement'] == 'compile-cpu'
    assert row['dependency_limitations'] == {'host': ['conservative']}
    assert row['provenance_sources'] == ['a.mojo']
    assert result['source_revision'] == 'f' * 40
    assert result['source_root'] == str(tmp_path)
    assert result['dependency_log_dir'] == str(tmp_path / 'logs')
    result['configuration']['dependency_context']['cpu_target'] = 'changed'
    assert config['dependency_context']['cpu_target'] == 'osx-arm64'
    R.requirements(tmp_path, 'HEAD', 'v', catalog={'requirements': []}, configuration=config)
    R.requirements(tmp_path, 'HEAD', 'v', expand_dependencies=False, configuration=config)
    assert len(calls) == 1  # explicit fixture catalog and explicit opt-out never analyze a real tree
    R.requirements(tmp_path, 'HEAD', 'v', catalog=raw, expand_dependencies=True, configuration=config)
    assert len(calls) == 2


def test_equivalence_reuses_actual_pass_without_fabricating_new_execution(native_ledger, tmp_path):
    led, calls, _ = native_ledger
    record(led, tmp_path)
    original = copy.deepcopy(led.read()['attempts'][0])
    led.sync(native_manifest(tmp_path, 'b' * 40, 'edited one definition'))
    row = led.plan()['identity']
    assert calls
    assert row['status'] == 'passed' and row['action'] == 'reuse'
    assert row['tested_revision'] == 'a' * 40 and row['source_revision'] == 'b' * 40
    assert row['receipt'] == original['receipt']
    assert row['equivalence_proof']['old_fingerprint'] == original['fingerprint']
    assert row['parent_requirement'] == 'identity-nvidia'
    assert row['provenance_sources'] == ['core/shared.mojo']
    assert led.begin('identity') is None
    assert led.read()['attempts'] == [original] and len(led.read()['runs']) == 1


def test_legacy_run_manifest_reconstructs_inputs_for_proof(native_ledger, tmp_path):
    led, _, _ = native_ledger
    record(led, tmp_path)
    with led.edit() as doc:
        for entry in (doc['runs'][0], doc['attempts'][0]):
            entry['manifest'] = {k: entry['manifest'][k] for k in
                                 ('source_revision', 'fingerprint', 'coverage', 'kind', 'stage')}
    led.sync(native_manifest(tmp_path, 'b' * 40, 'edited'))
    assert led.plan()['identity']['status'] == 'passed'


@pytest.mark.parametrize('tamper', ['digest', 'rehashed-selector', 'missing'])
def test_equivalence_proof_tampering_cannot_qualify(native_ledger, tmp_path, tamper):
    import release_requirement_dependencies as D
    led, _, _ = native_ledger
    record(led, tmp_path)
    led.sync(native_manifest(tmp_path, 'b' * 40, 'edited'))
    with led.edit() as doc:
        proof = doc['status']['identity']['equivalence_proof']
        if tamper == 'digest':
            proof['digest'] = 'forged'
        elif tamper == 'missing':
            doc['status']['identity']['equivalence_proof'] = None
        else:
            proof['selection']['lanes'] = []
            proof.pop('digest')
            proof['digest'] = D.digest(proof)
    row = led.plan()['identity']
    assert row['status'] == 'untested' and row['action'] == 'complete-provenance'
    assert row['dependency_decision'] == 'blocked'
    assert 'proof invalid' in row['reasons'][0]


def test_equivalence_revalidates_frozen_selector_not_just_saved_digest(native_ledger, tmp_path):
    led, _, selected = native_ledger
    record(led, tmp_path)
    led.sync(native_manifest(tmp_path, 'b' * 40, 'edited'))
    selected['lanes'].append('two')
    row = led.plan()['identity']
    assert row['status'] == 'untested' and row['dependency_decision'] == 'blocked'
    assert 'does not reproduce' in row['reasons'][0]


@pytest.mark.parametrize('changed', ['flags', 'coverage', 'environment', 'recipe'])
def test_non_source_inputs_cannot_use_source_equivalence(native_ledger, tmp_path, changed):
    led, calls, _ = native_ledger
    record(led, tmp_path)
    target = native_manifest(tmp_path, 'b' * 40, 'edited')
    wanted = target['requirements']['identity']
    if changed == 'flags':
        wanted['inputs']['unit']['build_flags'] = {'CXXFLAGS': '-changed'}
    elif changed == 'coverage':
        wanted['inputs']['contract']['coverage'] = wanted['coverage'] = ['different actual coverage']
    elif changed == 'environment':
        wanted['inputs']['execution_environment']['compiler'] = 'new toolchain'
    else:
        wanted['inputs']['recipe'] = {'dataset': 'changed'}
    wanted['fingerprint'] = R.digest(wanted['inputs'])
    led.sync(target)
    row = led.plan()['identity']
    assert row['status'] == 'untested' and row['dependency_decision'] == 'affected'
    assert not calls


def test_no_selector_work_for_untested_or_only_failed_rows(native_ledger, tmp_path):
    led, calls, _ = native_ledger
    led.sync(native_manifest(tmp_path, 'b' * 40, 'edited'))
    led.plan()
    assert not calls
    record(led, tmp_path, status='failed')
    led.sync(native_manifest(tmp_path, 'c' * 40, 'edited again'))
    assert led.plan()['identity']['status'] == 'untested'
    assert not calls


@pytest.mark.parametrize('missing', [False, True])
def test_equivalent_source_missing_bytes_requires_restore_not_rerun(native_ledger, tmp_path, missing):
    led, calls, _ = native_ledger
    path = record(led, tmp_path)
    led.sync(native_manifest(tmp_path, 'b' * 40, 'edited'))
    calls.clear()
    if missing:
        path.unlink()
    else:
        path.write_text('corrupt retained report')
    row = led.plan()['identity']
    assert row['status'] == 'untested' and row['action'] == 'restore-evidence'
    assert not calls
    with pytest.raises(ValueError, match='restore provenance'):
        led.begin('identity')
    assert len(led.read()['runs']) == 1


@pytest.mark.parametrize('limited', [False, True])
def test_unresolved_selection_is_blocked_not_a_blanket_rerun(native_ledger, tmp_path, limited):
    led, _, selected = native_ledger
    record(led, tmp_path)
    if limited:
        selected['precision_limitations'] = [{'lane': 'two', 'paths': ['core/shared.mojo']}]
    else:
        selected['unattributed'] = ['new/unknown.mojo']
    led.sync(native_manifest(tmp_path, 'b' * 40, 'edited'))
    row = led.plan()['identity']
    assert row['dependency_decision'] == 'blocked' and row['action'] == 'complete-provenance'
    with pytest.raises(ValueError, match='selector provenance unresolved'):
        led.begin('identity')


def test_proven_source_change_reaching_lane_is_explicitly_affected(native_ledger, tmp_path):
    led, _, selected = native_ledger
    record(led, tmp_path)
    selected['lanes'].append('two')
    led.sync(native_manifest(tmp_path, 'b' * 40, 'edited'))
    row = led.plan()['identity']
    assert row['dependency_decision'] == 'affected' and row['action'] == 'run'
    assert any('source changes reach this unit' in reason for reason in row['reasons'])


def test_later_equivalent_failure_or_pending_cannot_hide_behind_exact_old_pass(native_ledger, tmp_path):
    led, _, _ = native_ledger
    record(led, tmp_path)
    led.sync(native_manifest(tmp_path, 'b' * 40, 'edited'))
    reason = 'new negative control is required'
    run = led.begin('identity', reason=reason, invalidation_paths=[invalidation_evidence(tmp_path, reason)])
    led.sync(native_manifest(tmp_path))
    assert led.plan()['identity']['action'] == 'resume'
    assert led.plan()['identity']['active_revision'] == 'b' * 40
    assert 'different inputs' in led.plan()['identity']['reasons'][0]
    path = tmp_path / 'new-failure.log'
    path.write_text('failed')
    led.record(run, 'failed', [path], 'negative control failed on later run', 1, 1, 0, failed=1)
    row = led.plan()['identity']
    assert row['status'] == 'untested' and row['action'] == 'repair'
    assert 'later failure on equivalent inputs' in row['reasons'][0]
    assert row['failure_equivalence_proof']['old_revision'] == 'b' * 40
    assert not led.gate()[0]


def test_revert_to_qualified_inputs_ignores_failed_changed_arithmetic(native_ledger, tmp_path):
    led, _, selected = native_ledger
    record(led, tmp_path)
    original = led.read()['attempts'][0]
    selected['lanes'].append('two')
    led.sync(native_manifest(tmp_path, 'b' * 40, 'changed arithmetic'))
    failed = record(led, tmp_path, status='failed')
    failed.unlink()  # Unrelated historical bytes do not force this valid cell to rerun.
    led.sync(native_manifest(tmp_path, 'c' * 40))
    row = led.plan()['identity']
    assert row['status'] == 'passed' and row['action'] == 'reuse'
    assert row['tested_revision'] == original['source_revision']
    assert row['attempt'] == original['id'] and row['receipt'] == original['receipt']
    assert row['unrelated_failures'][0]['reason'] == 'source changes reach this unit'
    assert len(led.read()['attempts']) == 2 and led.begin('identity') is None


def test_revert_to_qualified_flags_ignores_other_configuration_failure(native_ledger, tmp_path):
    led, calls, _ = native_ledger
    record(led, tmp_path)
    led.sync(native_manifest(tmp_path, flags={'CXXFLAGS': '-changed'}))
    record(led, tmp_path, status='failed')
    led.sync(native_manifest(tmp_path))
    assert led.plan()['identity']['status'] == 'passed'
    assert not calls


def test_later_failure_with_unresolved_equivalence_blocks_reuse(native_ledger, tmp_path):
    led, _, selected = native_ledger
    record(led, tmp_path)
    led.sync(native_manifest(tmp_path, 'b' * 40, 'edited'))
    record(led, tmp_path, status='failed', reason='new failure')
    selected['unattributed'] = ['unknown/source.mojo']
    led.sync(native_manifest(tmp_path))
    row = led.plan()['identity']
    assert row['status'] == 'untested' and row['action'] == 'complete-provenance'
    assert 'cannot scope later failed run' in row['reasons'][0]


def test_invalid_original_receipt_cannot_acquire_an_equivalence_pass(native_ledger, tmp_path):
    led, calls, _ = native_ledger
    record(led, tmp_path)
    with led.edit() as doc:
        doc['attempts'][0]['receipt']['source_revision'] = 'b' * 40
    led.sync(native_manifest(tmp_path, 'b' * 40, 'edited'))
    row = led.plan()['identity']
    assert row['status'] == 'untested' and row['action'] == 'complete-provenance'
    assert not calls


def test_begin_retains_expanded_requirement_metadata(native_ledger):
    led, _, _ = native_ledger
    run_id = led.begin('identity')
    run = next(run for run in led.read()['runs'] if run['id'] == run_id)
    assert run['manifest']['parent_requirement'] == 'identity-nvidia'
    assert run['manifest']['dependency_limitations'] == {}
    assert run['manifest']['provenance_sources'] == ['core/shared.mojo']
    assert R.RequirementsLedger._attempt_requirement(run)['inputs'] == run['inputs']


def test_invalid_proof_stays_blocked_across_sync(native_ledger, tmp_path):
    led, _, _ = native_ledger
    record(led, tmp_path)
    target = native_manifest(tmp_path, 'b' * 40, 'edited')
    led.sync(target)
    with led.edit() as doc:
        doc['status']['identity']['equivalence_proof']['digest'] = 'tampered'
    led.sync(target)
    led.sync(target)
    assert led.plan()['identity']['status'] == 'untested'
    assert led.read()['status']['identity']['equivalence_proof']['digest'] == 'tampered'


def test_restoration_and_missing_inputs_cannot_be_bypassed_with_rerun_reason(ledger, tmp_path):
    path = record(ledger, tmp_path)
    path.unlink()
    with pytest.raises(ValueError, match='restore provenance'):
        ledger.begin('identity', reason='a reason cannot recover missing evidence bytes')
    assert len(ledger.read()['runs']) == 1


def test_later_failed_evidence_also_needs_restoration(native_ledger, tmp_path):
    led, _, _ = native_ledger
    record(led, tmp_path)
    led.sync(native_manifest(tmp_path, 'b' * 40, 'edited'))
    path = record(led, tmp_path, status='failed', reason='negative control')
    path.unlink()
    led.sync(native_manifest(tmp_path))
    assert led.plan()['identity']['action'] == 'restore-evidence'
    with pytest.raises(ValueError, match='restore provenance'):
        led.begin('identity')


def test_old_late_finish_cannot_replace_newer_run_result(ledger, tmp_path):
    old = ledger.begin('identity')
    # Model a ledger produced before cross-source pending-run protection.
    with ledger.edit() as doc:
        newer = copy.deepcopy(doc['runs'][0])
        newer.update(id='newer-run', started_at='later')
        doc['runs'].append(newer)
    path = tmp_path / 'newer-pass.log'
    path.write_text('newer source passed')
    ledger.record('newer-run', 'passed', [path], 'newer run', 0, 1, 1,
                  receipt=receipt_for(ledger, 'newer-run', path))
    failure = tmp_path / 'older-failure.log'
    failure.write_text('older in-flight run finished later')
    ledger.record(old, 'failed', [failure], 'older run failed', 1, 1, 0, failed=1)
    assert ledger.plan()['identity']['status'] == 'passed'
    assert ledger.plan()['identity']['attempt'] == ledger.read()['attempts'][0]['id']


def test_original_attempt_outcome_tampering_cannot_reuse(native_ledger, tmp_path):
    led, calls, _ = native_ledger
    record(led, tmp_path)
    with led.edit() as doc:
        doc['attempts'][0]['coverage']['skipped'] = 1
    led.sync(native_manifest(tmp_path, 'b' * 40, 'edited'))
    assert led.plan()['identity']['action'] == 'complete-provenance'
    assert not calls


def test_original_contract_metadata_tampering_cannot_reuse(native_ledger, tmp_path):
    led, calls, _ = native_ledger
    record(led, tmp_path)
    with led.edit() as doc:
        doc['attempts'][0]['manifest']['coverage'] = ['invented easier coverage']
    led.sync(native_manifest(tmp_path, 'b' * 40, 'edited'))
    assert led.plan()['identity']['action'] == 'complete-provenance'
    assert not calls


def test_record_producer_validation_receives_original_input_configuration(ledger, tmp_path, monkeypatch):
    import release_requirement_receipts as P
    seen = []
    monkeypatch.setattr(P, 'validate_producer_receipt', lambda *args, **kwargs: seen.append((args, kwargs)) or [])
    record(ledger, tmp_path)
    # The producer adapter must receive the begun contract and artifact/config
    # inputs, even when the current desired source has moved on.
    assert any(isinstance(value, dict) and value.get('inputs') == ledger.read()['runs'][0]['inputs']
               for args, kwargs in seen for value in (*args, *kwargs.values()))


@pytest.mark.parametrize('kind', ['compile', 'runtime', 'identity', 'adapters', 'performance', 'published'])
def test_unsupported_producers_require_evidence_admission_before_new_execution(tmp_path, monkeypatch, kind):
    monkeypatch.setattr(R, 'producer_admission_available', PRODUCTION_PRODUCER_ADMISSION)
    target = manifest()
    target['requirements']['identity']['kind'] = kind
    ledger = R.RequirementsLedger(tmp_path / 'ledger.json')
    ledger.sync(target)
    row = ledger.plan()['identity']
    assert row['status'] == 'untested' and row['action'] == 'complete-provenance'
    assert row['producer_admission'] == 'unsupported'
    assert 'reconcile retained evidence' in row['reasons'][-1]
    with pytest.raises(ValueError, match='producer report adapter unavailable'):
        ledger.begin('identity')
    assert not ledger.read()['runs']


def test_unsupported_producer_keeps_exact_input_change_reasons(native_ledger, tmp_path, monkeypatch):
    ledger, _, _ = native_ledger
    record(ledger, tmp_path, status='failed')
    monkeypatch.setattr(R, 'producer_admission_available', PRODUCTION_PRODUCER_ADMISSION)
    ledger.sync(native_manifest(tmp_path, 'b' * 40, 'edited', flags={'CXXFLAGS': '-new'}))
    row = ledger.plan()['identity']
    assert row['action'] == 'complete-provenance'
    assert 'changed input: native_sources/core/shared.mojo' in row['reasons']
    assert 'added input: unit/build_flags/CXXFLAGS' in row['reasons']
    assert any('previous attempt failed' in reason for reason in row['reasons'])


def test_unadapted_source_audit_requires_admission_before_new_execution(tmp_path, monkeypatch):
    monkeypatch.setattr(R, 'producer_admission_available', PRODUCTION_PRODUCER_ADMISSION)
    target = manifest()
    wanted = target['requirements'].pop('identity')
    wanted['inputs']['contract'] = {'id': 'coverage-source', 'kind': 'source'}
    wanted['fingerprint'] = R.digest(wanted['inputs'])
    target['requirements'] = {'coverage-source': wanted}
    ledger = R.RequirementsLedger(tmp_path / 'ledger.json')
    ledger.sync(target)
    row = ledger.plan()['coverage-source']
    assert row['action'] == 'complete-provenance'
    assert row['producer_admission'] == 'unsupported'
    with pytest.raises(ValueError, match='producer report adapter unavailable'):
        ledger.begin('coverage-source')
    assert not ledger.read()['runs']

    # A real supported source producer can still start its first assessment.
    wanted['inputs']['contract']['id'] = 'runtime-source'
    wanted['fingerprint'] = R.digest(wanted['inputs'])
    target['requirements'] = {'runtime-source': wanted}
    ledger.sync(target)
    assert ledger.plan()['runtime-source']['action'] == 'run'
    assert ledger.begin('runtime-source')


def test_interrupted_launch_keeps_history_and_allows_changed_source(ledger, tmp_path):
    run = ledger.begin('identity')
    original = copy.deepcopy(ledger.read()['runs'])
    ledger.sync(manifest(revision='new', arithmetic='two'))
    proof = tmp_path / 'producer-stopped.json'
    proof.write_text('{"pod_absent": true, "controller_stopped": true}')
    closed = ledger.interrupt(run, [proof], 'lease expired; no final producer report', producer_stopped=True)
    doc = ledger.read()
    assert doc['runs'] == original and doc['attempts'] == []
    assert closed['source_revision'] == 'a' and closed['qualification_claim'] is False
    assert 'coverage' not in closed and 'exit_code' not in closed
    row = ledger.plan()['identity']
    assert row['status'] == 'untested' and row['action'] == 'run'
    assert not ledger.gate()[0]
    assert ledger.begin('identity') != run
    assert ledger.read()['runs'][-1]['source_revision'] == 'new'
    assert ledger.plan()['identity']['action'] == 'resume'


def test_interruption_cannot_revive_pass_invalidated_by_required_rerun(ledger, tmp_path):
    record(ledger, tmp_path)
    attempts = copy.deepcopy(ledger.read()['attempts'])
    reason = 'routing negative control requires a new execution'
    proof = invalidation_evidence(tmp_path, reason)
    run = ledger.begin('identity', reason=reason, invalidation_paths=[proof])
    stopped = tmp_path / 'stopped.log'
    stopped.write_text('controller stopped before qualification completed')
    ledger.interrupt(run, [stopped], 'controller interrupted', producer_stopped=True)
    assert ledger.read()['attempts'] == attempts
    row = ledger.plan()['identity']
    assert row['status'] == 'untested' and row['action'] == 'run'
    assert ledger.begin('identity') is not None


def test_interruption_preserves_prior_algorithm_failure(ledger, tmp_path):
    record(ledger, tmp_path, status='failed')
    attempts = copy.deepcopy(ledger.read()['attempts'])
    run = ledger.begin('identity')
    proof = tmp_path / 'stopped.log'
    proof.write_text('repair producer stopped')
    ledger.interrupt(run, [proof], 'repair execution interrupted', producer_stopped=True)
    assert ledger.read()['attempts'] == attempts
    assert ledger.read()['attempts'][0]['status'] == 'failed'
    assert ledger.plan()['identity']['status'] == 'untested'


def test_interruption_requires_stopped_confirmation_reason_and_evidence(ledger, tmp_path):
    run = ledger.begin('identity')
    proof = tmp_path / 'stopped.log'
    proof.write_text('producer stopped')
    original = ledger.path.read_bytes()
    for paths, reason, stopped, match in [
        ([proof], 'stopped', False, 'confirm producers stopped'),
        ([proof], ' ', True, 'concrete interruption reason'),
        ([], 'stopped', True, 'retain producer stop/absence evidence'),
    ]:
        with pytest.raises(ValueError, match=match):
            ledger.interrupt(run, paths, reason, producer_stopped=stopped)
        assert ledger.path.read_bytes() == original


def test_interruption_batch_is_atomic_and_rejects_finished_or_unknown_runs(ledger, tmp_path):
    first, second = ledger.begin('identity'), ledger.begin('metadata')
    proof = tmp_path / 'stopped.log'
    proof.write_text('both controllers stopped')
    original = ledger.path.read_bytes()
    for ids, match in [([first, 'unknown'], 'unknown interrupted'), ([first, first], 'unique interrupted')]:
        with pytest.raises(ValueError, match=match):
            ledger.interrupt_many(ids, [proof], 'batch stopped', producer_stopped=True)
        assert ledger.path.read_bytes() == original
    closed = ledger.interrupt_many([first, second], [proof], 'batch stopped', producer_stopped=True)
    assert len(closed) == 2 and len(ledger.read()['interruptions']) == 2
    after = ledger.path.read_bytes()
    with pytest.raises(ValueError, match='already recorded or interrupted'):
        ledger.interrupt(first, [proof], 'again', producer_stopped=True)
    assert ledger.path.read_bytes() == after


def test_late_results_cannot_rewrite_interrupted_or_completed_run(ledger, tmp_path):
    run = ledger.begin('identity')
    proof = tmp_path / 'stopped.log'
    proof.write_text('producer stopped')
    ledger.interrupt(run, [proof], 'interrupted producer', producer_stopped=True)
    with pytest.raises(ValueError, match='run was interrupted'):
        ledger.record(run, 'failed', [proof], 'late output', 1, 1, 0, failed=1)
    record(ledger, tmp_path, name='metadata', status='failed')
    completed = ledger.read()['attempts'][-1]['run_id']
    with pytest.raises(ValueError, match='already recorded or interrupted'):
        ledger.interrupt(completed, [proof], 'cannot rewrite failure', producer_stopped=True)


def test_interruption_evidence_is_revalidated_before_relaunch(ledger, tmp_path):
    run = ledger.begin('identity')
    proof = tmp_path / 'stopped.log'
    proof.write_text('stopped producer')
    ledger.interrupt(run, [proof], 'stopped', producer_stopped=True)
    proof.write_text('changed proof')
    row = ledger.plan()['identity']
    assert row['status'] == 'untested' and row['action'] == 'restore-evidence'
    with pytest.raises(ValueError, match='restore provenance before rerunning'):
        ledger.begin('identity')


def test_interrupt_cli_bulk_preserves_unexecuted_status(ledger, tmp_path, capsys):
    run = ledger.begin('identity')
    ids = tmp_path / 'run-ids.json'
    ids.write_text(json.dumps([run]))
    proof = tmp_path / 'stopped.log'
    proof.write_text('producer absent')
    assert R.main(['interrupt', '--ledger', str(ledger.path), '--run-ids-file', str(ids),
                   '--producer-stopped', '--reason', 'producer stopped', '--evidence', str(proof)]) == 0
    assert json.loads(capsys.readouterr().out) == {'interrupted': 1, 'qualification_claim': False}
    assert ledger.plan()['identity']['status'] == 'untested'
