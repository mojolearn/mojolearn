from run_full_do_measurements import campaign_complete, state_file


def test_runpod_transport_and_lease_preserve_owned_port_and_key():
    from run_full_do_measurements import transport, provider_command
    assert transport({'SSH_TARGET': '-p 12345 root@192.0.2.1'}) == (
        'root@192.0.2.1', ['-p', '12345'], ['-P', '12345'], '-p 12345 root@192.0.2.1')
    assert provider_command({'provider': 'runpod', 'state_key': 'measure-nv'}, 'extend', 480) == [
        'bash', 'tools/dev_pod.sh', 'extend', 'measure-nv', '480']
    assert transport({'IP': '192.0.2.2'}) == ('root@192.0.2.2', [], [], 'root@192.0.2.2')


def test_failures_do_not_prevent_completion_but_pending_or_unretained_work_does():
    doc = {'finished': 1, 'expected': 3, 'failed_recipes': ['crash'], 'recipes': {
        key: {'status': status, 'retention': {'status': 'verified'}}
        for key, status in [('good', 'process finished'), ('skip', 'skipped'), ('crash', 'failed')]}}
    assert campaign_complete(doc)
    doc['recipes']['crash']['status'] = 'pending'
    assert not campaign_complete(doc)
    doc['recipes']['crash']['status'] = 'failed'
    doc['stop_reason'] = 'retention failed'
    assert not campaign_complete(doc)
    del doc['stop_reason']
    doc['recipes']['skip']['retention']['status'] = 'local'
    assert not campaign_complete(doc)


def test_state_is_read_without_executing_shell(tmp_path):
    p = tmp_path/'state.env'
    p.write_text("DROPLET_ID=123\nRENTAL_OWNER='owner $(false)'\n")
    assert state_file(p) == {'DROPLET_ID': '123', 'RENTAL_OWNER': 'owner $(false)'}


def test_retention_dedup_preserves_every_source_path_and_byte(tmp_path):
    from full_do_measurement_worker import deduplicate_sources
    paths = []
    for name, content in [('a', b'same'), ('b', b'same'), ('c', b'different')]:
        p = tmp_path/'measured'/name/'evidence/source/source.tar.gz'
        p.parent.mkdir(parents=True)
        p.write_bytes(content)
        paths.append(p)
    assert len(deduplicate_sources(tmp_path)) == 1
    assert paths[0].stat().st_ino == paths[1].stat().st_ino
    assert [p.read_bytes() for p in paths] == [b'same', b'same', b'different']


def test_candidate_switches_are_explicit_and_do_not_replace_other_environment():
    import pytest
    from full_do_measurement_worker import apply_runtime_environment
    environment = {'UNCHANGED': 'value'}
    switches = {'MOJOLEARN_NEURAL_NATIVE_DISPATCH': '1', 'MOJOLEARN_GENERATION_NATIVE_SESSION': '1'}
    assert apply_runtime_environment({'runtime_env': switches}, environment) == switches
    assert environment == dict(switches, UNCHANGED='value')
    with pytest.raises(ValueError, match='declared neural'):
        apply_runtime_environment({'runtime_env': {'UNDECLARED': '1'}}, environment)
    with pytest.raises(ValueError, match='string 0 or 1'):
        apply_runtime_environment({'runtime_env': {'MOJOLEARN_NEURAL_NATIVE_DISPATCH': True}}, environment)


def test_existing_box_requirement_refuses_missing_state_without_provisioning(tmp_path, monkeypatch):
    import pytest
    import run_full_do_measurements as controller
    def forbidden(*args, **kwargs):
        raise AssertionError('must not launch provider or SSH commands')
    monkeypatch.setattr(controller.subprocess, 'run', forbidden)
    config = dict(repo=str(tmp_path), out=str(tmp_path / 'out'), source_commit='test',
                  vendors=[dict(vendor='amd', state_dir=str(tmp_path / 'missing-state'), require_existing_box=True)])
    with pytest.raises(RuntimeError, match='existing owned box state is missing'):
        controller.run(config)
