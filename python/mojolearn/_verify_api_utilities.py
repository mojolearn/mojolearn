"""Small array, scalar, random-stream and configuration endpoint checks."""
__mojolearn_user_boundary__ = True

import inspect
import struct
from ._verify_api import resolve, encode
from ._verify_api_metadata import factory

CLASSES = frozenset(('Array','ConstantKernel','RBF','Matern','WhiteKernel',
    'ParameterGrid','ParameterSampler','ConstantLR','WarmupLinearLR','WarmupCosineLR',
    'StepLR','ExponentialLR','OneCycleLR','Generator','BootstrapResult',
    'MonteCarloResult','PermutationTestResult'))
FUNCTIONS = frozenset(('gpu_arch_how','numeric_profile','numeric_profile_measured',
                      'numeric_profiles','set_numeric_mode','set_numeric_profile','require_identical'))


def applicable(public, method):
    return public.rsplit('.',1)[-1] in CLASSES|FUNCTIONS or method=='get_n_splits'


def check(public, method):
    import numpy as np
    from ._verify_api import package
    ml = package()
    cls = resolve(public)
    name = public.rsplit('.',1)[-1]
    if name=='Array':
        a=ml.Array.from_list([[.25,.5],[.75,1]],dtype='<f4')
        if method=='from_list': return ml.Array.from_list([[.125,.375]],dtype='<f4')
        if method=='__array__': return a.__array__()
        if method=='__array_interface__:get':
            view=a.__array_interface__
            assert view['shape']==(2,2) and view['typestr']=='<f4' and view['data'][0]>0
            return {'shape':view['shape'],'typestr':view['typestr'],'values':np.asarray(a)}
        if method=='__buffer__':
            view=a.__buffer__(0)
            assert view.shape==(2,2)
            return {'shape':view.shape,'format':view.format,'bytes':view.tobytes()}
        if method=='__bool__': return ml.Array.from_list([.25],dtype='<f4').__bool__()
        if method=='__getitem__': return a.__getitem__((slice(None),1))
        if method=='__iter__': return list(a.__iter__())
        if method=='__len__': return a.__len__()
        if method in ('__eq__','__ne__'): return getattr(a,method)(a)
        if method=='__reduce__':
            # (copyreg.__newobj__, (cls,), state): the state goes to __setstate__, as pickle does
            reducer,args,*state=a.__reduce__(); restored=reducer(*args)
            if state and state[0] is not None: restored.__setstate__(state[0])
            assert a.tobytes()==restored.tobytes()
            return restored
        if method=='flags:get': return a.flags
        return getattr(a,method)()
    if name in ('ConstantKernel','RBF','Matern','WhiteKernel'):
        kernel=cls()
        if method in ('__add__','__mul__'):
            combined=getattr(kernel,method)(ml.RBF(.5))
            x=np.array([[.125],[.5]],np.float32)
            m=ml.GaussianProcessRegressor(kernel=combined,optimizer=None)
            m.fit(x,np.array([.25,.75],np.float32))
            return m.predict(x)
        return getattr(kernel,method.split(':')[0])
    if method=='get_n_splits':
        splitter=factory(public,**({'test_size':.5} if public.endswith('StratifiedShuffleSplit') else {}))
        x=np.arange(16,dtype=np.float32).reshape(8,2); y=np.arange(8)%2; groups=np.arange(8)//2
        count=splitter.get_n_splits(x,y,groups)
        splits=list(splitter.split(x,y,groups))
        assert count==len(splits)
        return count
    if name in ('ParameterGrid','ParameterSampler'):
        obj=cls({'alpha':[.25,.5]}) if name=='ParameterGrid' else cls({'alpha':[.25,.5]},n_iter=2,random_state=7)
        if method=='__iter__':return list(obj.__iter__())
        if method=='__len__':return obj.__len__()
        return obj.__getitem__(0)
    if name=='Generator':
        gen=cls(7,numeric_mode='identical')
        if method=='next_stream':return gen.next_stream()
        if method=='dropout_backward':
            _, key=gen.dropout(np.ones((2,2),np.float32),.25)
            return gen.dropout_backward(np.ones((2,2),np.float32),key)
        kwargs={}
        if name.startswith('kaiming') or method.startswith('kaiming'):kwargs={'fan_in':2}
        if method.startswith('xavier'):kwargs={'fan_in':2,'fan_out':2}
        return getattr(gen,method)((2,2),**kwargs)
    if name in ('ConstantLR','WarmupLinearLR','WarmupCosineLR'):
        obj=cls(.125,warmup_steps=1,total_steps=3,min_lr=.03125)
        if method=='from_config':
            restored=cls.from_config(obj.config())
            return [restored.lr_at(t) for t in (1,2,3)]
        if method=='config':return obj.config()
        return [getattr(obj,method)(t) for t in (1,2,3)]
    if name in ('StepLR','ExponentialLR','OneCycleLR'):
        sig=inspect.signature(cls)
        values={'lr':.125,'base_lr':.125,'initial_lr':.125,'max_lr':.125,'step_size':2,'gamma':.5,'total_steps':4}
        kwargs={k:v for k,v in values.items() if k in sig.parameters}
        obj=cls(**kwargs)
        return [obj.lr_at(t) for t in (1,2,3)]
    if name in ('BootstrapResult','PermutationTestResult','MonteCarloResult'):
        if name == 'BootstrapResult':
            dist = np.array([.75, .25], np.float32)
            obj = cls(.55, dist, dist[::-1].copy(), .18, .25, .75, 0, 1)
            return [obj.point_estimate, obj.distribution, obj.sorted_distribution,
                    obj.standard_error, tuple(obj.confidence_interval), obj.order_low, obj.order_high]
        if name == 'PermutationTestResult':
            obj = cls(.625, np.array([.25, .5, .75], np.float32), .375, 2, 1)
            return [obj.statistic, obj.null_distribution, obj.pvalue, obj.count_less, obj.count_greater]
        obj = cls(.75, .375, 2., .625)
        return [obj.integral, obj.mean, obj.volume, obj.closed_form]
    if name=='set_numeric_mode':
        before=ml.runtime_report()['default_mode']
        try:
            result=cls('identical');assert ml.runtime_report()['default_mode']=='identical'
            try:cls('deterministic')
            except ValueError:pass
            else:raise AssertionError('retired mode must be refused')
            return {'identical_selected':True,'retired_refused':True}
        finally:cls(before)
    if name=='require_identical':
        cls();return {'identical_accepted':True}
    if name=='gpu_arch_how':
        result=cls();assert isinstance(result,(str,type(None)))
        return {'route_type_valid':True}
    if name=='set_numeric_profile':
        before=ml.numeric_profile()
        try:
            cls(before)
            try:cls('invalid_verifier_profile')
            except ValueError:pass
            else:raise AssertionError('unknown profile must be refused')
            return ml.numeric_profile()
        finally:cls(before)
    if name=='numeric_profile_measured':return cls(ml.numeric_profile())
    if name in FUNCTIONS:return cls()
    raise ValueError('unimplemented utility fixture: '+public+'.'+method)


def utility_checks(recorder, rows):
    for row in rows:
        recorder.check(row['endpoint'],lambda r=row:check(r['public'],r['op']))
