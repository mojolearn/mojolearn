from gpu_gap_campaign import configure_checkpoint_cache


def test_generation_uses_campaign_cache_and_keeps_explicit_root(tmp_path):
    env = {}
    key = 'MOJOLEARN_BOARD_LM_CHECKPOINTS'
    assert configure_checkpoint_cache(tmp_path, env) == str(tmp_path / 'hf-checkpoints')
    env[key] = '/explicit/checkpoints'
    assert configure_checkpoint_cache(tmp_path, env) == '/explicit/checkpoints'
    env[key] = ' '
    assert configure_checkpoint_cache(tmp_path, env) == str(tmp_path / 'hf-checkpoints')
