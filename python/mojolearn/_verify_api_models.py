"""One-layer, two-token loaded-model and continuation fixtures; no downloads."""
__mojolearn_user_boundary__ = True

import tempfile
from pathlib import Path
from ._verify_api import resolve, encode


def applicable(public,method):return public.startswith('mojolearn.models.')


def model_checks(recorder,public,methods):
    import numpy as np
    from . import models, _backend
    from ._causal_lm_fixtures import _llama_config,_llama_tensors,_write_checkpoint
    name=public.rsplit('.',1)[-1]
    def check(method):
        if name=='catalog':
            entries=models.catalog();assert isinstance(entries,(dict,list))
            return entries
        if name=='verify_catalog_entry':
            # This command can download multi-GB models. The normal verifier
            # checks its admission boundary; model arithmetic is covered below.
            try:models.verify_catalog_entry('__verifier_unknown_model__')
            except KeyError:return {'unknown_catalog_name_refused':True}
            raise AssertionError('unknown catalog name was admitted')
        if name=='Tokenizer':
            import unicodedata
            assert models.Tokenizer.unicode_version==unicodedata.unidata_version
            return {'unicode_version_matches_runtime':True}
        cfg=_llama_config(num_hidden_layers=1,vocab_size=256,max_position_embeddings=4)
        tensors=_llama_tensors(cfg)
        with tempfile.TemporaryDirectory() as tmp:
            _write_checkpoint(tmp,cfg,tensors,shards=1)
            if name=='HFConfig':
                m=models.HFConfig.from_json(tmp)
                if method=='__contains__':assert 'hidden_size' in m and 'missing_verifier_key' not in m;return True
                return m['hidden_size']
            if name in ('Checkpoint','SafetensorsFile'):
                m=models.Checkpoint.open(tmp) if name=='Checkpoint' else models.SafetensorsFile(str(Path(tmp)/'model.safetensors'))
                assert 'model.embed_tokens.weight' in m and 'missing_verifier_tensor' not in m
                return {'present':True,'missing':False}
            if name=='ParallelCausalLM':
                m=models.ParallelCausalLM.load(tmp,layer_devices=(0,))
            else:m=models.CausalLM.load(tmp)
            try:
                ids=np.array([[1,2]],np.int32)
                if name=='GenerationStream':
                    stream=m.stream(ids,max_new_tokens=1)
                    stream.read(1)
                    return getattr(stream,method)()
                if name=='CausalLMState':
                    state=m.allocate_state(1,4)
                    clone=models.CausalLMState(state.batch_size,state.max_tokens,state.layers,owner=m)
                    return m.forward(ids,clone)
                if method=='__call__':return m(ids)
                if method=='blocks:get':assert len(m.blocks)==1;return m.forward(ids)
                state=m.allocate_state(1,4)
                if method=='allocate_state':return m.forward(ids,state)
                if method=='reset_state':
                    first=m.forward(ids,state);m.reset_state(state)
                    second=m.forward(ids,state);assert encode(first)==encode(second)
                    return second
                if method=='step':m.forward(ids[:,:1],state);return m.step(ids[:,1:],state)
                if method=='generate':return m.generate(ids,1)
                if method in ('stream','resume_generation'):
                    stream=m.stream(ids,max_new_tokens=1)
                    if method=='resume_generation':stream=m.resume_generation(stream.state_dict())
                    return stream.read(1)
                from .models.tokenizer import Tokenizer
                tok=Tokenizer([bytes([i]) for i in range(256)],[],pattern='gpt2')
                if method=='generate_text':return m.generate_text('ab',1,tokenizer=tok)
                return m.stream_text('ab',1,tokenizer=tok).read(1)
            finally:
                if name=='ParallelCausalLM':m.close()
    for method in methods:
        kind='admission only' if name=='verify_catalog_entry' else 'numerical'
        recorder.check(public+'.'+method,lambda method=method:check(method),kind=kind)
