"""Imported-program inherited doors on a single scalar captured graph."""
__mojolearn_user_boundary__ = True

import tempfile
from pathlib import Path


def applicable(public,method):
    return public=='mojolearn.importers._program.ImportedProgram'


def program_checks(recorder,public,methods):
    import numpy as np
    from . import ops
    from .importers._program import ImportedProgram
    def check(method):
        with ops.Session() as s, tempfile.TemporaryDirectory() as tmp:
            a=np.array([[.25]],np.float32);x=s.tensor(a);w=s.tensor([[2.]])
            captured=s.capture(x,s.matmul(x,w))
            info=(captured._id,captured.input_shapes,captured.output_shapes,captured.params,captured.input_kinds)
            p=ImportedProgram(s,info,[('x',(1,1),'float32')],[('out',(1,1),'float32')])
            condition=s.capture(x,s.less(x,s.full((1,1),1)))
            other=s.capture(x,s.add(x,s.full((1,1),.25)))
            if method in ('__init__','run'):return p.run(a)
            if method=='run_dynamic':return p.run_dynamic(np.array([[.25],[.5]],np.float32))
            if method=='inspect':
                report=p.inspect();assert report
                return {'input_shapes':p.input_shapes,'output_shapes':p.output_shapes,'output':p.run(a)}
            if method=='profile':
                report=p.profile(a)
                # Timings are incidental API output, never stored as identity.
                return report['outputs']
            if method=='run_if':return [p.run_if(condition,other,a),p.run_if(condition,other,np.array([[2]],np.float32))]
            if method=='while_loop':return p.while_loop(a,condition=condition,max_iterations=2)
            if method in ('vjp_if','vjp_while'):
                kwargs=dict(seeds=np.ones_like(a),parameters=[w])
                result=p.vjp_if(condition,other,a,**kwargs) if method=='vjp_if' else p.vjp_while(a,condition=condition,max_iterations=2,**kwargs)
                return {'outputs':result['outputs'],'input_gradients':result['input_gradients'],
                        'parameter_gradient':result['parameter_gradients'][w]}
            if method in ('save','save_control'):
                path=str(Path(tmp)/'program')
                if method=='save':p.save(path);restored=s.load_program(path);return restored.run(a)
                p.save_control(path,condition=condition,otherwise=other)
                restored=s.load_control(path)
                return restored['body'].run_if(restored['condition'],restored['otherwise'],a)
            opt=s.optimizer('sgd',lr=.125)
            if method in ('fit','fit_if'):
                kwargs=dict(targets=np.ones_like(a),optimizer=opt,epochs=1,batch_size=1,shuffle=False)
                result=p.fit(a,**kwargs) if method=='fit' else p.fit_if(condition,other,a,**kwargs)
                assert result is p
                return {'output':p.run(a),'parameter':s.numpy(w)}
            kwargs=dict(devices=(0,),optimizer=opt,parameters=[w])
            if method=='partition':kwargs.update(stages=1,tensor_shards=1,microbatches=1)
            else:kwargs['logical_shards']=1
            with getattr(p,method)(**kwargs) as owner:
                owner.fit(a,targets=np.ones_like(a),epochs=1,batch_size=1,shuffle=False)
                return owner.run(a)
    for method in methods:recorder.check(public+'.'+method,lambda method=method:check(method))
