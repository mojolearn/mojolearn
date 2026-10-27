"""CPU doubles check generation startup without GPU, fitting or token execution."""
import json
import sys
from types import SimpleNamespace as NS
from unittest.mock import patch

import numpy as np
import pytest

import bench_board_neural as N


@pytest.mark.parametrize('bf16', [False, True])
@pytest.mark.parametrize('byte_lm', [False, True])
def test_generation_ready_uses_checkpoint_inputs(tmp_path, bf16, byte_lm):
    cfg = NS(rope_theta=10000., rms_norm_eps=1e-5, head_dim=4,
             hidden_size=8, num_attention_heads=2, num_key_value_heads=1,
             num_hidden_layers=2, vocab_size=256, tie_word_embeddings=False,
             _attn_implementation='sdpa')
    events = []

    class Model:
        def __init__(self, config):
            self.config = config
            self.model = NS(rotary_emb=None)
            events.append('constructed')

        @classmethod
        def from_pretrained(cls, path, **kwargs):
            events.append(('checkpoint', path, kwargs))
            return cls(cfg)

        def eval(self):
            events.append('eval')

        def to(self, dtype):
            events.append(('dtype', dtype))
            return self

        def generate(self, **kwargs):
            raise AssertionError('startup must never generate tokens')

    torch = NS(__version__='stub', version=NS(hip='stub', cuda=None),
               manual_seed=lambda seed: None, from_numpy=lambda x: x,
               ones_like=np.ones_like, float32='float32', bfloat16='bfloat16',
               get_num_threads=lambda: 1, nn=NS(Identity=lambda: 'identity'))
    twin = NS(COLUMNS={'eager_fp32': dict(tf32=False, autocast=None, compile=False),
                      'eager_bf16': dict(tf32=False, autocast='bfloat16', compile=False)},
              set_precision=lambda *args: {'tf32': False})
    transformers = NS(__version__='stub', GenerationConfig=lambda **kw: NS(**kw))
    checkpoint = {'format': 'hf'}
    if byte_lm:
        checkpoint = dict(format=N.LM_CHECKPOINTS['mojolearn-162m']['fmt'],
                          vocab_size=256, d_model=8, n_heads=2, n_kv=1, head_dim=4,
                          intermediate=16, n_layers=2, max_position_embeddings=128,
                          rms_norm_eps=1e-5, rope_theta=10000.)
    (tmp_path / 'config.json').write_text(json.dumps(checkpoint))
    # Only generation inputs: any accidental MLP/program-run lookup fails here.
    data = dict(prompt_ids=np.array([[1, 2]], dtype=np.int32),
                new_tokens=np.array([3]), checkpoint_dir=np.array(str(tmp_path)),
                checkpoint_key=np.array('stub'))
    lane = 'causal-lm-generate' if byte_lm else 'causal-lm-generate-smollm2'
    with patch.object(N, '_load', return_value=twin), \
         patch.object(N, '_torch_device', return_value=(torch, 'cuda', 'cuda', 'stub', lambda: None)), \
         patch.object(N, 'DeviceMemory', return_value=NS(method=lambda: 'stub')), \
         patch.object(N, '_transformers', return_value=(transformers, None, NS(LlamaForCausalLM=Model))), \
         patch.object(N, 'hf_llama_config', return_value=(None, cfg)), \
         patch.object(N, 'hf_load_exact', side_effect=lambda *args: events.append('weights_checked')), \
         patch.dict(sys.modules, {'transformers': transformers,
                                  'safetensors.torch': NS(load_file=lambda path: {})}):
        runner = N.TorchGenerateArm(lane, 'full', data,
                                    'torch-eager-bf16' if bf16 else 'torch-eager-fp32')
    assert runner.model.config is cfg
    assert runner.record['d_model'] == 8
    assert runner.record['prompt_tokens'] == 2
    assert runner.record['new_tokens'] == 3
    assert runner.record['__implementation__'] is Model
    assert runner._on_device is False
    assert runner.gc[3].max_new_tokens == 3
    assert runner.gc[3].do_sample is False and runner.gc[3].eos_token_id is None
    assert ('dtype', 'bfloat16') in events if bf16 else not any(isinstance(e, tuple) and e[0] == 'dtype' for e in events)
    assert ('weights_checked' in events) == byte_lm
    if byte_lm:
        assert runner.model.model.norm == 'identity'
    runner.out, runner.first = np.array([[1, 2, 3]]), np.array([[1, 2]])
    assert runner.outputs() == {'ids': runner.out, 'first': runner.first}
    assert len(runner.digest()) == 16
