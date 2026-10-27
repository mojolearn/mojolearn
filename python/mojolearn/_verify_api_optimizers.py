"""One-update optimizer state, ownership, transfer and zeroing contracts."""
__mojolearn_user_boundary__ = True

from ._verify_api import resolve, encode

CLASSES=frozenset(('SGD','Adam','AdamW','Adagrad','Adamax','Adafactor','Lion','LAMB','NAdam','RMSprop'))


def applicable(public, method):
    return public.rsplit('.',1)[-1] in CLASSES and method not in ('get_params','set_params','__getattribute__')


def optimizer_checks(recorder, public, methods):
    import numpy as np
    cls=resolve(public)
    def check(method):
        param=np.array([[.25,-.5],[.75,1]],np.float32)
        gradient=np.array([[.125,.25],[-.125,.5]],np.float32)
        opt=cls([param],lr=.125,numeric_mode='identical')
        if method=='params_resident_available':
            available=opt.params_resident_available()
            assert type(available) is bool
            return {'resident_available':available}
        if method in ('params_to_device','params_to_host','grads_to_device','grads_to_host','param_device','step_device'):
            opt.params_to_device()
            if method=='params_to_device':return opt.params_to_host()
            if method=='params_to_host':return opt.params_to_host()
            if method=='param_device':
                value=opt.param_device(0)
                assert tuple(value.shape)==param.shape
                return {'shape':value.shape,'values':opt.params_to_host()}
            opt.grads_to_device([gradient])
            if method in ('grads_to_device','grads_to_host'):
                target=np.zeros_like(gradient)
                opt.grads_to_host([target])
                assert target.tobytes()==gradient.tobytes()
                return target
            opt.step_device()
            return opt.params_to_host()
        if method=='zero_grad':
            opt.zero_grad([gradient])
            assert gradient.tobytes()==bytes(gradient.nbytes)
            return gradient
        if method=='step_accumulated':
            opt.step_accumulated([[gradient]],tokens=1)
            return param
        opt.step([gradient])
        if method=='state_dict':return {'state':opt.state_dict(),'parameters':param}
        if method=='load_state_dict':
            state=opt.state_dict()
            restored=cls([param.copy()],lr=.125,numeric_mode='identical')
            restored.load_state_dict(state)
            assert encode(restored.state_dict())==encode(state)
            return restored.state_dict()
        value=getattr(opt,method.split(':')[0])
        return value() if callable(value) else value
    for method in methods:
        recorder.check(public+'.'+method,lambda m=method:check(m))
