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
