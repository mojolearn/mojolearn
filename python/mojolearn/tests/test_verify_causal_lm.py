# SPDX-License-Identifier: Apache-2.0
import copy
import pytest
from mojolearn._verify_causal_lm import ARCHITECTURES, CHECKS, PARTS, PROFILE, compare
from mojolearn.models.causal_lm import CausalLM, CausalLMState


def record():
    return dict(profile=PROFILE, formats=['float32'], repeats=2, source_sha256='source', status='CAPTURED_UNQUALIFIED', cases=[
        dict(architecture=arch, tied=tied, weight_format='float32',
             checks=dict.fromkeys(CHECKS, True), parts=dict.fromkeys(PARTS, 'a'*64),
             checkpoint_sha256={'model': 'b'*64}) for arch,tied in ARCHITECTURES])


def test_comparator_compares_logits_and_input_bytes():
    a=record(); b=copy.deepcopy(a)
    assert compare(a,b)
    b['cases'][0]['parts']['decode_2']='c'*64
    assert not compare(a,b)
    b=copy.deepcopy(a); b['cases'][0]['checkpoint_sha256']['model']='d'*64
    assert not compare(a,b)



def test_host_vs_device_state_is_recorded_not_required():
    """Andrew 2026-10-07: host digests are recorded, never required; device vs device stays exact."""
    a=record(); b=copy.deepcopy(a)
    a['device']='cpu'; b['device']='gpu'
    b['cases'][0]['parts']['state']='e'*64
    notes=[]
    assert compare(a,b,notes)
    assert notes[0]['host_device_state']=='DIFFERS' and len(notes[0]['state_differs'])==1
    b['cases'][0]['parts']['logits']='f'*64
    assert not compare(a,b)
    c=copy.deepcopy(a); d=copy.deepcopy(a); c['device']=d['device']='gpu'
    d['cases'][0]['parts']['state']='e'*64
    assert not compare(c,d)

@pytest.mark.parametrize('fault', ['missing_part','missing_check','failed','duplicate','empty','source'])
def test_incomplete_records_refused(fault):
    a=record(); b=copy.deepcopy(a)
    if fault=='missing_part': b['cases'][0]['parts'].pop('logits')
    if fault=='missing_check': b['cases'][0]['checks'].pop('batch')
    if fault=='failed': b['cases'][0]['checks']['batch']=False
    if fault=='duplicate': b['cases']*=2
    if fault=='empty': b['cases']=[]
    if fault=='source': b['source_sha256']='other'
    with pytest.raises(ValueError): compare(a,b)


def test_reset_refuses_foreign_state_before_native_work():
    model=object.__new__(CausalLM)
    with pytest.raises(ValueError, match='another model'):
        model.reset_state(CausalLMState(1,8,[],owner=object()))


def test_both_truncated_records_are_not_a_pass():
    a=record(); a['cases'].pop()
    with pytest.raises(ValueError, match='missing requested'): compare(a,a)


@pytest.mark.parametrize('architecture,tied', ARCHITECTURES)
def test_fixture_names_exactly_cover_supported_family(architecture,tied):
    from mojolearn._causal_lm_fixtures import family_fixture
    from mojolearn.models import HFConfig,plan_for
    cfg,tensors=family_fixture(architecture,tied)
    plan=plan_for(HFConfig(cfg))
    assert set(tensors)==set(plan.checkpoint_names())
    if architecture=='mistral': assert plan.block_options['window']==3
    if architecture=='qwen2': assert plan.block_options['qkv_bias']
    if architecture=='qwen3': assert plan.block_options['qk_norm']
    if architecture=='phi3': assert any(rows is not None for _,_,rows in plan.layer_weights(0))


def test_state_digest_is_canonical_kv_state():
    # Host and device states with equal written positions digest equal: the
    # capacity tail and the device-only arithmetic_profile label are not state.
    from mojolearn import Array
    from mojolearn._transformer_impl import TransformerState
    from mojolearn._verify_causal_lm import state_digest
    def layer(tail, written=1.0, label=None):
        # B=1, KV=1, HD=2, max_tokens=4, cached_tokens=2: four written floats.
        k = Array.from_list([written, 2.0, 3.0, 4.0] + [tail] * 4, '<f4')
        v = Array.from_list([5.0, 6.0, 7.0, 8.0] + [tail] * 4, '<f4')
        st = TransformerState(1, 1, 2, 4, k, v, cached_tokens=2)
        if label is not None:
            st.arithmetic_profile = label
        return st
    def lm_state(st):
        s = CausalLMState(1, 4, [st])
        s.positions = 2
        return s
    host = state_digest(lm_state(layer(0.0)))
    assert state_digest(lm_state(layer(9.0, label='mojolearn.identical.transformer.fp32'))) == host
    assert state_digest(lm_state(layer(0.0, written=-1.0))) != host
