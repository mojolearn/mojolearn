"""One-step portable training/state fixtures, with one logical shard."""
__mojolearn_user_boundary__ = True

from pathlib import Path
import tempfile
from ._verify_api import resolve, encode

NAMES=frozenset(('LanguageModelConfig','LanguageModelHostTrainer','LanguageModelInference',
 'LanguageModelTrainer','OffloadedByteLanguageModelTrainer','ParallelByteLanguageModelTrainer',
 'PooledByteLanguageModelTrainer','ParallelNeuralTrainer','SmallMLPTrainer'))


def applicable(public,method):
    return public.rsplit('.',1)[-1] in NAMES and method not in ('get_params','set_params','__getattribute__')


def tiny_lm():
    import numpy as np
    from . import LanguageModelConfig, LanguageModelTrainer
    shape=LanguageModelConfig(batch=1,length=2,d_model=4,n_heads=1,n_kv=1,head_dim=4,
                              intermediate=4,n_layers=1,vocab_size=4)
    named={entry['name']:((np.arange(entry['size'],dtype=np.float32).reshape(entry['shape'])%7-3)/32)
           for entry in LanguageModelTrainer.parameter_registry(shape)}
    # Unequal norm weights make swapped-layer/normalizer faults visible on
    # the very first step; no extra warmup or training iterations.
    for i,(key,value) in enumerate(named.items()):
        if 'norm' in key:value[:]=1+i/32
    model=LanguageModelTrainer(named,shape=shape,data_schedule={'fixture':'public-api-v1'},lr=.125)
    return model,shape,np.array([[0,1,2]],np.int32)


def tiny_mlp():
    import numpy as np
    from . import SmallMLPTrainer
    weights=[(np.arange(n,dtype=np.float32)%7-3).reshape(shape)/32 for n,shape in
             [(128,(16,8)),(16,(16,)),(48,(3,16)),(3,(3,))]]
    return SmallMLPTrainer(*weights,data_schedule={'fixture':'public-api-v1'},lr=.125)


def training_checks(recorder,public,methods):
    import numpy as np
    cls=resolve(public);name=public.rsplit('.',1)[-1]
    def check(method):
        if name in ('SmallMLPTrainer','ParallelNeuralTrainer'):
            m=tiny_mlp();x=np.arange(16,dtype=np.float32).reshape(2,8)/32;y=np.array([0,2],np.int32)
            if name=='ParallelNeuralTrainer':
                p=cls(m,devices=(0,),logical_shards=1)
                try:
                    other=cls.from_checkpoint(p.checkpoint(),devices=(0,))
                    try:return other.checkpoint()
                    finally:other.close()
                finally:p.close()
            if method=='train_steps':
                try:return m.train_steps(x,y,1)
                except RuntimeError as error:
                    assert 'Apple FAST training binding' in str(error)
                    return {'availability':'fast-only multistep refused in identical','numerical_execution':False}
            m.train_step(x,y)
            if method=='predict_logits_sessions':return m.predict_logits_sessions([x[:1],x[1:]])
            if method=='step_:get':assert m.step_==1;return m.step_
            if method=='load_state_dict':
                other=tiny_mlp();other.load_state_dict(m.state_dict())
                assert encode(other.state_dict())==encode(m.state_dict())
                return other.predict_logits_sessions([x])
            return m.state_dict()
        source,shape,ids=tiny_lm()
        try:
            if name=='LanguageModelConfig':
                if method=='__post_init__':shape.__post_init__()
                return {'config':shape.to_dict(),'native_shape':shape.native_shape,'registry':shape.parameter_shapes}
            if name in ('LanguageModelHostTrainer','LanguageModelInference'):
                params=source.state_dict()['parameters']
                m=cls(params,shape=shape)
                if name=='LanguageModelInference':return m.parameters_sha256()
                m.train_step(ids)
                if method=='from_state':
                    other=cls.from_state(m.parameters_,m.m_,m.v_,shape=shape,completed_steps=1,lr=.125)
                    assert other.parameters_sha256()==m.parameters_sha256()
                    return other.loss(ids)
                if method=='loss':return m.loss(ids)
                value=getattr(m,method.split(':')[0]);value=value() if callable(value) else value
                return value.to_dict() if method=='shape:get' else value
            if name=='LanguageModelTrainer':
                source.train_step(ids)
                if method=='evaluate':return source.evaluate(ids)
                if method=='parameter_registry':return source.parameter_registry(shape)
                if method=='data_schedule:get':return source.data_schedule
                if method=='step_:get':assert source.step_==1;return source.step_
                if method=='load_state_dict':
                    other,_,_=tiny_lm()
                    try:
                        other.load_state_dict(source.state_dict())
                        assert encode(other.state_dict())==encode(source.state_dict())
                        return other.evaluate(ids)
                    finally:other.close()
                if method=='export_state':return source.export_state()
                if method=='run_metadata':
                    info=source.run_metadata();assert info['completed_steps']==1
                    return {k:info[k] for k in ('profile','config','data_schedule','completed_steps')}
                if method=='attention_stage_report':
                    report=source.attention_stage_report()
                    return report
                with tempfile.TemporaryDirectory() as tmp:
                    path=Path(tmp)/'checkpoint'
                    if method in ('export_checkpoint_binary','from_checkpoint_binary'):
                        source.export_checkpoint_binary(str(path));other=cls.from_checkpoint_binary(str(path))
                    else:
                        getattr(source,method if method in ('save_checkpoint','export_checkpoint') else 'export_checkpoint')(path)
                        other=cls.from_checkpoint_bytes(path.read_bytes()) if method=='from_checkpoint_bytes' else cls.from_checkpoint(path)
                    try:
                        assert encode(other.state_dict())==encode(source.state_dict())
                        return other.evaluate(ids)
                    finally:other.close()
            model=cls(source.state_dict(),devices=(0,),logical_shards=1,
                      **({'pool_optimizer':False} if name=='ParallelByteLanguageModelTrainer' else {}))
            try:
                if name!='ParallelByteLanguageModelTrainer' and method in ('apply_gradient','shard_gradient','fold_reset','fold_add','fold_export','shard_gradient_fold'):
                    args=() if method in ('fold_reset','fold_export') else ((ids,) if method in ('shard_gradient','shard_gradient_fold') else (np.zeros(shape.n_total,np.float32),))
                    try:getattr(model,method)(*args)
                    except ValueError as error:
                        assert 'requires ParallelByteLanguageModelTrainer' in str(error)
                        return {'unsupported_schedule_refused':True,'numerical_execution':False}
                    raise AssertionError('unsupported split-step schedule admitted')
                if method=='from_checkpoint':
                    other=cls.from_checkpoint(model.checkpoint(),devices=(0,))
                    try:return other.state_dict()
                    finally:other.close()
                if method=='checkpoint':return model.checkpoint()
                if method=='export_raw':
                    raw=model.export_raw();state=model.state_dict()
                    assert all(encode(raw[k])==encode(state[k]) for k in raw)
                    return raw
                if method=='set_lr':model.set_lr(.25);assert model.state_dict()['config']['lr']==.25;return .25
                if method in ('optimizer_ownership','model_ownership'):
                    report=getattr(model,method)();assert isinstance(report,(dict,list,tuple))
                    return report
                if method=='has_device_fold:get':return model.has_device_fold
                if method.startswith('fold_') or method=='shard_gradient_fold':
                    grad=np.arange(shape.n_total,dtype=np.float32)/max(shape.n_total,1)/32
                    model.fold_reset(grad)
                    if method=='fold_reset':return model.fold_export()
                    if method=='shard_gradient_fold':loss=model.shard_gradient_fold(ids);return {'loss':loss,'gradient':model.fold_export()}
                    model.fold_add(grad)
                    result=model.fold_export();assert np.array_equal(np.frombuffer(result,dtype='<f4'),grad+grad)
                    return result
                if method=='shard_gradient':return model.shard_gradient(ids)
                if method=='apply_gradient':
                    _,gradient=model.shard_gradient(ids);model.apply_gradient(gradient)
                    assert model.step_==1
                    return model.export_raw()
                model.train_step([ids]);assert model.step_==1
                return model.step_
            finally:model.close()
        finally:source.close()
    for method in methods:
        recorder.check(public+'.'+method,lambda method=method:check(method))
