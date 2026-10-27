from run_full_do_measurements import campaign_complete, state_file


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
