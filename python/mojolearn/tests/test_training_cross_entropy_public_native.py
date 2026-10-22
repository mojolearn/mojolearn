"""The public loss prepares, calls and publishes outputs without Python runtime."""
import array
import warnings
import numpy as np
import pytest
from mojolearn import _training_impl as T, _backend
from mojolearn._array import Array
from mojolearn._verify_native_runtime import audit_native_call

def prepared(case):
    x=np.array([[.2,.7,-.3],[.9,-.2,.5],[-.7,.8,.1],[.3,.1,.2]],dtype='f4')
    y=np.array([2,0,-100,1],dtype='i4')
    kw={'numeric_mode':'identical'}
    category,name=case.split(':',1)
    if category=='target':
        if name in ['i1','u1','i2','u2','i4','u4','i8','u8','q','Q','>i4','>i8','bool','f4','f8','object','str','g']:
            vals=[2,0,2,1]
            y=np.array(vals,dtype=name)
        elif name=='list': y=[2,0,-100,1]
        elif name=='tuple': y=(2,0,-100,1)
        elif name=='float-list': y=[2.,0.,2.,1.]
        elif name=='npint-list': y=[np.int64(2),0,2,1]
        elif name=='npfloat-list': y=[np.float64(2),0,2,1]
        elif name=='bool-list': y=[True,False,True,False]
        elif name=='array': y=Array.from_buffer(y)
        elif name=='stdlib': y=array.array('q',y.tolist())
        elif name=='memoryview': y=memoryview(array.array('q',y.tolist()))
        elif name=='bytearray': y=bytearray([2,0,2,1])
        elif name=='negative': y=y[::-1]
        elif name=='strided': y=np.array([2,99,0,99,-100,99,1,99],dtype='i8')[::2]
        elif name=='readonly': y=np.frombuffer(y.tobytes(),dtype='i4')
        elif name=='empty': y=[]
        elif name=='rank2': y=y.reshape(2,2)
        elif name=='scalar': y=2
        elif name=='scalar-array': y=np.array(2)
        elif name=='rank2-list': y=[[2,0],[2,1]]
        elif name=='ragged': y=[[2,0],[1]]
        elif name=='overflow': y=np.array([2**31]*4,dtype='i8')
        elif name=='uint-overflow': y=np.array([2**63]*4,dtype='u8')
        elif name=='large-int-list': y=[2**63]*4
        elif name=='past-vocab': y=np.array([3]*4,dtype='i4')
        elif name=='negative-id': y=np.array([-1]*4,dtype='i4')
        elif name=='wrong-count': y=y[:3]
    elif category=='logits':
        if name=='fortran': x=np.asfortranarray(x)
        elif name=='negative': x=x[::-1]
        elif name=='strided':
            full=np.zeros((4,6),dtype='f4');full[:,::2]=x;x=full[:,::2]
        elif name=='readonly': x=np.frombuffer(x.tobytes(),dtype='f4').reshape(4,3)
        elif name=='array': x=Array.from_buffer(x)
        elif name in ['f2','f8','>f4']: x=x.astype(name)
        elif name=='rank1': x=x.ravel()
        elif name=='rank3': x=x.reshape(2,2,3)
        elif name=='scalar': x=np.array(1,dtype='f4')
        elif name=='scalar-array': x=Array.from_buffer(np.array(1,dtype='f4'))
        elif name=='list': x=x.tolist()
        elif name=='empty-rows': x=x[:0]
        elif name=='empty-vocab': x=x[:,:0]
        elif name=='nan': x[0,0]=np.nan
        elif name=='inf': x[0,0]=np.inf
    elif category=='mode':
        red,grad=name.split(',');kw.update(reduction=red,return_grad=grad=='True')
    elif category=='option':
        option,form=name.split(',')
        values={'float':.1,'negative':-.1,'one':1.,'nan':np.nan,'inf':np.inf,'string':'0.1','np-complex':np.complex64(.1+.2j),'object-array':np.array(.1,dtype=object),'array1d':np.array([.1]),'none':None,'int':2,'fractional':2.9,'int-string':'2','zero':0,'negative-int':-1,'np-int':np.int64(2),'int-array':np.array(2),'int-complex':np.complex64(2+1j),'false':False,'true':True,'empty-list':[],'list':[0],'bool-array':np.array(True),'bool-array1d':np.array([True]),'ambiguous-array':np.array([True,False]),'empty-array':np.array([]),'nonempty-string':'yes','empty-string':''}
        kw[option]=values[form]
    elif category=='reduction':
        kw['reduction']={'bad':'bad','none':None,'int':1,'bytes':b'mean','np-str':np.str_('mean'),'list':[],'array':np.array('mean')}[name]
    elif category=='kwargs': kw[name]=None
    return x,y,kw

cases=['target:'+n for n in ['i1','u1','i2','u2','i4','u4','i8','u8','q','Q','>i4','>i8','bool','f4','f8','object','str','g','list','tuple','float-list','npint-list','npfloat-list','bool-list','array','stdlib','memoryview','bytearray','negative','strided','readonly','empty','rank2','scalar','scalar-array','rank2-list','ragged','overflow','uint-overflow','large-int-list','past-vocab','negative-id','wrong-count']]

cases+=['logits:'+n for n in ['c','fortran','negative','strided','readonly','array','f2','f8','>f4','rank1','rank3','scalar','scalar-array','list','empty-rows','empty-vocab','nan','inf']]

cases+=['mode:'+r+','+str(g) for r in ['mean','sum','none'] for g in [False,True]]

cases+=['option:label_smoothing,'+f for f in ['float','negative','one','nan','inf','string','np-complex','object-array','array1d','none']]

cases+=['option:'+o+','+f for o in ['ignore_index','num_items'] for f in ['int','fractional','int-string','zero','negative-int','np-int','int-array','int-complex','none']]

cases+=['option:return_grad,'+f for f in ['false','true','float','empty-list','list','bool-array','bool-array1d','ambiguous-array','empty-array','nonempty-string','empty-string']]

cases+=['reduction:'+n for n in ['bad','none','int','bytes','np-str','list','array']]

cases+=['kwargs:'+n for n in ['weight','size_average','reduce','dropout','distributed','unknown']]

REFUSALS = {'target:>i4': 'TypeError', 'target:>i8': 'TypeError', 'target:bool': 'TypeError', 'target:f4': 'TypeError', 'target:f8': 'TypeError', 'target:object': 'TypeError', 'target:str': 'TypeError', 'target:g': 'TypeError', 'target:float-list': 'TypeError', 'target:npfloat-list': 'TypeError', 'target:empty': 'ValueError', 'target:rank2': 'ValueError', 'target:scalar': 'ValueError', 'target:scalar-array': 'ValueError', 'target:rank2-list': 'ValueError', 'target:ragged': 'TypeError', 'target:overflow': 'OverflowError', 'target:uint-overflow': 'OverflowError', 'target:large-int-list': 'TypeError', 'target:past-vocab': 'Exception', 'target:negative-id': 'Exception', 'target:wrong-count': 'ValueError', 'logits:f2': 'NotImplementedError', 'logits:f8': 'TypeError', 'logits:>f4': 'TypeError', 'logits:rank1': 'ValueError', 'logits:rank3': 'ValueError', 'logits:scalar': 'ValueError', 'logits:scalar-array': 'ValueError', 'logits:list': 'TypeError', 'logits:empty-rows': 'ValueError', 'logits:empty-vocab': 'ValueError', 'logits:nan': 'Exception', 'logits:inf': 'Exception', 'mode:none,True': 'NotImplementedError', 'option:label_smoothing,negative': 'Exception', 'option:label_smoothing,one': 'Exception', 'option:label_smoothing,nan': 'Exception', 'option:label_smoothing,inf': 'Exception', 'option:label_smoothing,array1d': 'TypeError', 'option:label_smoothing,none': 'TypeError', 'option:ignore_index,int': 'Exception', 'option:ignore_index,fractional': 'Exception', 'option:ignore_index,int-string': 'Exception', 'option:ignore_index,zero': 'Exception', 'option:ignore_index,negative-int': 'Exception', 'option:ignore_index,np-int': 'Exception', 'option:ignore_index,int-array': 'Exception', 'option:ignore_index,int-complex': 'Exception', 'option:ignore_index,none': 'TypeError', 'option:return_grad,ambiguous-array': 'ValueError', 'option:return_grad,empty-array': 'ValueError', 'reduction:bad': 'ValueError', 'reduction:none': 'ValueError', 'reduction:int': 'ValueError', 'reduction:bytes': 'ValueError', 'reduction:list': 'TypeError', 'reduction:array': 'TypeError', 'kwargs:weight': 'TypeError', 'kwargs:size_average': 'TypeError', 'kwargs:reduce': 'TypeError', 'kwargs:dropout': 'NotImplementedError', 'kwargs:distributed': 'NotImplementedError', 'kwargs:unknown': 'TypeError'}

WARNINGS_253 = {'option:label_smoothing,np-complex': [['ComplexWarning', 'Casting complex values to real discards the imaginary part']], 'option:ignore_index,int-complex': [['ComplexWarning', 'Casting complex values to real discards the imaginary part']], 'option:num_items,int-complex': [['ComplexWarning', 'Casting complex values to real discards the imaginary part']]}

WARNINGS_126 = {'option:label_smoothing,np-complex': [['ComplexWarning', 'Casting complex values to real discards the imaginary part']], 'option:label_smoothing,array1d': [['DeprecationWarning', 'Conversion of an array with ndim > 0 to a scalar is deprecated, and will error in future. Ensure you extract a single element from your array before performing this operation. (Deprecated NumPy 1.25.)']], 'option:ignore_index,int-complex': [['ComplexWarning', 'Casting complex values to real discards the imaginary part']], 'option:num_items,int-complex': [['ComplexWarning', 'Casting complex values to real discards the imaginary part']], 'option:return_grad,empty-array': [['DeprecationWarning', 'The truth value of an empty array is ambiguous. Returning False, but in future this will result in an error. Use `array.size > 0` to check that an array is not empty.'], ['DeprecationWarning', 'The truth value of an empty array is ambiguous. Returning False, but in future this will result in an error. Use `array.size > 0` to check that an array is not empty.'], ['DeprecationWarning', 'The truth value of an empty array is ambiguous. Returning False, but in future this will result in an error. Use `array.size > 0` to check that an array is not empty.'], ['DeprecationWarning', 'The truth value of an empty array is ambiguous. Returning False, but in future this will result in an error. Use `array.size > 0` to check that an array is not empty.']]}


RELEASE = tuple(int(v) for v in np.__version__.split('.')[:2])


def expected_category(case):
    name = REFUSALS.get(case)
    if case == 'option:label_smoothing,array1d' and RELEASE < (2, 4): name = None
    if case == 'option:return_grad,empty-array' and RELEASE < (2, 2): name = None
    return {'TypeError': TypeError, 'ValueError': ValueError, 'Exception': Exception,
            'NotImplementedError': NotImplementedError, 'OverflowError': OverflowError}.get(name)


def expected_warnings(case):
    if case == 'option:label_smoothing,array1d':
        return WARNINGS_126[case] if RELEASE < (2, 4) else []
    if case == 'option:return_grad,empty-array':
        return WARNINGS_126[case] if RELEASE < (2, 2) else []
    return WARNINGS_253.get(case, [])


def reference(x, y, options):
    # Independent test controller: byte-compatible preparation plus the
    # retained native forward/backward entry. The119-case old-public baseline
    # separately establishes this reference's admitted cases and refusals.
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        logits = np.ascontiguousarray(x, dtype='f4')
        targets = np.ascontiguousarray(list(y) if isinstance(y, bytearray) else y, dtype='i4')
        reduction = {'none':0, 'sum':1, 'mean':2}[options.get('reduction', 'mean')]
        want = bool(options.get('return_grad', False))
        rows, vocab = logits.shape
        loss, row = np.zeros(1, dtype='f4'), np.zeros(rows, dtype='f4')
        grad = np.zeros((rows, vocab) if want else (1,), dtype='f4')
        params = [rows, vocab, int(options.get('ignore_index', -100)), reduction,
                  int(options.get('num_items') or 0), int(want), float(options.get('label_smoothing', 0.))]
    binding = _backend.binding('_mojolearn_training', 'identical')
    binding.ce_loss(loss.ctypes.data, row.ctypes.data, grad.ctypes.data,
                    logits.ctypes.data, targets.ctypes.data, params)
    value = row if reduction == 0 else float(loss[0])
    return (value, grad) if want else value


def compare_output(value, expected):
    if isinstance(expected, tuple):
        assert isinstance(value, tuple) and len(value) == len(expected)
        for a, b in zip(value, expected): compare_output(a, b)
    elif isinstance(expected, float):
        assert type(value) is float
        assert np.float64(value).tobytes() == np.float64(expected).tobytes()
    else:
        assert type(value) is Array
        assert value._mv.obj is value._store
        actual = np.asarray(value)
        assert actual.dtype == expected.dtype and actual.shape == expected.shape
        assert actual.tobytes() == expected.tobytes()


def assert_refusal(audit, category):
    assert type(audit.exception) is category, audit.report()
    for field in ('python_frames', 'forbidden_native_calls', 'wrapper_work', 'unobserved_existing_threads'):
        assert not audit.report()[field], audit.report()


@pytest.mark.parametrize('case', cases)
def test_public_cross_entropy_preserves_retained_host_contract(case):
    x, y, options = prepared(case)
    category = expected_category(case)
    expected = None if category else reference(x, y, options)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        audit = audit_native_call(T.cross_entropy, x, y, **options)
    if category: assert_refusal(audit, category)
    else: compare_output(audit.assert_native(), expected)
    assert [[type(w.message).__name__, str(w.message)] for w in caught] == expected_warnings(case)


@pytest.mark.parametrize('case', ['target:i1', 'target:i2', 'target:i8', 'target:npint-list',
                                 'target:negative', 'target:strided', 'target:readonly',
                                 'logits:fortran', 'logits:negative', 'logits:strided',
                                 'logits:readonly', 'logits:array'])
@pytest.mark.parametrize('smoothing', [0., .2])
def test_public_cross_entropy_joint_gradient_matches_retained_kernel(case, smoothing):
    x, y, options = prepared(case)
    options.update(return_grad=True, label_smoothing=smoothing, num_items=3, reduction='sum')
    expected = reference(x, y, options)
    compare_output(audit_native_call(T.cross_entropy, x, y, **options).assert_native(), expected)


@pytest.mark.parametrize('slot', ['logits', 'targets', 'target-element', 'reduction', 'smoothing',
                                 'ignore_index', 'num_items', 'return_grad', 'object-return-grad'])
def test_public_cross_entropy_refuses_python_hooks_before_execution(slot):
    calls = []
    class Foreign:
        def __array__(self, *a, **k): calls.append('array'); return np.ones(1)
        def __buffer__(self, *a): calls.append('buffer')
        def __float__(self): calls.append('float'); return .1
        def __int__(self): calls.append('int'); return 1
        def __index__(self): calls.append('index'); return 1
        def __bool__(self): calls.append('bool'); return True
        def __hash__(self): calls.append('hash'); return 1
    x, y, options = prepared('logits:c')
    if slot == 'logits': x = Foreign()
    elif slot == 'targets': y = Foreign()
    elif slot == 'target-element': y = [Foreign(), 0, 1, 2]
    elif slot == 'smoothing': options['label_smoothing'] = Foreign()
    elif slot == 'object-return-grad':
        obj = np.empty((), dtype=object); obj[()] = Foreign(); options['return_grad'] = obj
    else: options[slot] = Foreign()
    assert_refusal(audit_native_call(T.cross_entropy, x, y, **options), TypeError)
    assert not calls


@pytest.mark.parametrize('entry', ['ce_loss', 'embedding_index_cast'])
def test_public_cross_entropy_refuses_replaced_native_entries(monkeypatch, entry):
    calls = []
    def hook(*a, **k): calls.append(a)
    if entry == 'ce_loss': monkeypatch.setitem(T._OPT_CONTEXT['bindings']['identical'], entry, hook)
    else: monkeypatch.setitem(T._OPT_CONTEXT, entry, hook)
    x, y, options = prepared('target:i8')
    assert_refusal(audit_native_call(T.cross_entropy, x, y, **options), TypeError)
    assert not calls


def test_public_cross_entropy_uses_no_python_preparation(monkeypatch):
    x, y, options = prepared('target:list')
    expected = reference(x, y, options)
    def hook(*a, **k): raise AssertionError('old loss preparation entered')
    for name in ('_load', '_refuse_unknown', '_is_buffer', 'probe', '_check_dtype', 'as_f32_c', 'as_i32_c', '_cross_entropy_dev'):
        monkeypatch.setattr(T, name, hook)
    compare_output(audit_native_call(T.cross_entropy, x, y, **options).assert_native(), expected)


@pytest.mark.parametrize('action', ['ignore', 'error'])
def test_public_cross_entropy_scalar_warning_actions_are_native(action):
    x, y, options = prepared('option:label_smoothing,np-complex')
    with warnings.catch_warnings():
        warnings.simplefilter(action)
        audit = audit_native_call(T.cross_entropy, x, y, **options)
    if action == 'error': assert_refusal(audit, np.exceptions.ComplexWarning)
    else: compare_output(audit.assert_native(), reference(x, y, options))


def test_public_cross_entropy_warning_hook_is_refused(monkeypatch):
    calls = []
    def hook(*a, **k): calls.append(a)
    x, y, options = prepared('option:label_smoothing,np-complex')
    with warnings.catch_warnings():
        warnings.simplefilter('always')
        monkeypatch.setattr(warnings, 'showwarning', hook)
        audit = audit_native_call(T.cross_entropy, x, y, **options)
    assert_refusal(audit, TypeError)
    assert not calls


@pytest.mark.parametrize('case', ['record', 'tuple-record', 'tuple-foreign'])
def test_public_cross_entropy_reduction_never_hashes_embedded_python_objects(case):
    calls = []
    class Foreign:
        def __hash__(self): calls.append('hash'); return 123
        def __eq__(self, other): calls.append('eq'); return False
    record = np.empty((), dtype=[('field', object)])
    record['field'][()] = Foreign()
    record.flags.writeable = False
    scalar = record[()]
    reduction = scalar if case == 'record' else ((scalar,) if case == 'tuple-record' else (Foreign(),))
    x, y, options = prepared('logits:c')
    options['reduction'] = reduction
    assert_refusal(audit_native_call(T.cross_entropy, x, y, **options), TypeError)
    assert not calls


@pytest.mark.parametrize('reduction,want,smoothing', [('mean',False,0.),('mean',True,0.),
    ('sum',True,.2),('none',False,.2)])
def test_public_cross_entropy_device_outputs_are_native_and_consumed(reduction, want, smoothing):
    if _backend.vendor() == 'cpu': pytest.skip('training device pool requires GPU')
    x, y, options = prepared('target:i8')
    options.update(reduction=reduction, return_grad=want, label_smoothing=smoothing)
    expected = reference(x, y, options)
    device = audit_native_call(T.train_to_device, x, numeric_mode='identical').assert_native()
    audit = audit_native_call(T.cross_entropy, device, y, **options)
    result = audit.assert_native()
    if want:
        loss, gradient = result
        actual = audit_native_call(gradient.to_host).assert_native()
        compare_output((loss, actual), expected)
        audit_native_call(T.TrainDeviceArray.__del__, gradient).assert_native()
    else: compare_output(result, expected)
    audit_native_call(T.TrainDeviceArray.__del__, device).assert_native()


def test_public_cross_entropy_device_failure_rollback_has_no_python_finalizer():
    if _backend.vendor() == 'cpu': pytest.skip('training device pool requires GPU')
    x, y, options = prepared('target:past-vocab')
    device = audit_native_call(T.train_to_device, x, numeric_mode='identical').assert_native()
    assert_refusal(audit_native_call(T.cross_entropy, device, y, return_grad=True, numeric_mode='identical'), Exception)
    # A subsequent real operation consumes both outputs after the failed
    # operation's temporary device owner is dropped by the native descriptor.
    good = np.array([2,0,2,1], dtype='i4')
    loss, grad = audit_native_call(T.cross_entropy, device, good, return_grad=True, numeric_mode='identical').assert_native()
    compare_output((loss, audit_native_call(grad.to_host).assert_native()), reference(x, good, {'return_grad':True}))
    audit_native_call(T.TrainDeviceArray.__del__, grad).assert_native()
    audit_native_call(T.TrainDeviceArray.__del__, device).assert_native()


TRUTH_CASES = ['np-float','np-complex','array-float','array-complex','array-object-zero',
    'array-object-list','unicode-empty','unicode-space','bytes-space','rawvoid-zero','rawvoid-one',
    'record-zero','record-one','record-object-zero','record-object-one','record-object-empty-array',
    'array-record-zero','array-record-one','array-record-object-zero','array-record-object-one',
    'native-array','native-array-scalar','native-array-empty','stdlib-empty','memoryview','set','dict']
if hasattr(np.dtypes, 'StringDType'): TRUTH_CASES += ['managed-empty','managed-missing-native','managed-missing-array']


def return_grad_value(case):
    if case == 'np-float': return np.float32(.1)
    if case == 'np-complex': return np.complex64(1j)
    if case == 'array-float': return np.array(.1)
    if case == 'array-complex': return np.array(1j)
    if case == 'array-object-zero': return np.array(0, dtype=object)
    if case == 'array-object-list':
        value = np.empty((),dtype=object);value[()] = [0];return value
    if case == 'unicode-empty': return np.array('',dtype='U3')
    if case == 'unicode-space': return np.array(' ',dtype='U3')
    if case == 'bytes-space': return np.array(b' ',dtype='S3')
    if case.startswith('rawvoid'): return np.void(bytes([0, 0] if case.endswith('zero') else [0, 97]))
    if 'record' in case:
        object_field = 'object' in case
        value = np.zeros((),dtype=[('left','i4'),('right',object if object_field else 'f4')])
        value['right'] = 1 if case.endswith('one') else 0
        if case.endswith('empty-array'): value['right'][()] = np.array([])
        return value if case.startswith('array-') else value[()]
    if case == 'native-array': return Array.from_buffer(np.ones(1,dtype='f4'))
    if case == 'native-array-scalar': return Array.from_buffer(np.ones((),dtype='f4'))
    if case == 'native-array-empty': return Array.from_buffer(np.zeros(0,dtype='f4'))
    if case == 'stdlib-empty': return array.array('i')
    if case == 'memoryview': return memoryview(b'a')
    if case == 'set': return {0}
    if case == 'dict': return {'ignored':False}
    if case == 'managed-empty': return np.array('',dtype=np.dtypes.StringDType())
    sentinel = 0 if case == 'managed-missing-native' else np.array(False)
    value = np.empty((),dtype=np.dtypes.StringDType(na_object=sentinel));value[()] = sentinel
    return value


@pytest.mark.parametrize('case', TRUTH_CASES)
def test_public_cross_entropy_native_truth_configuration_matches_builtin_contract(case):
    value = return_grad_value(case)
    x,y,options = prepared('logits:c')
    category = None
    with warnings.catch_warnings(record=True) as expected_warnings:
        warnings.simplefilter('always')
        try:
            for _ in range(4): want = bool(value)
        except Exception as error: category = type(error)
    with warnings.catch_warnings(record=True) as actual_warnings:
        warnings.simplefilter('always')
        audit = audit_native_call(T.cross_entropy,x,y,return_grad=value,numeric_mode='identical')
    if category: assert_refusal(audit,category)
    else: compare_output(audit.assert_native(),reference(x,y,{'return_grad':want}))
    assert [(type(w.message),str(w.message)) for w in actual_warnings] == [(type(w.message),str(w.message)) for w in expected_warnings]


@pytest.mark.parametrize('dtype', ['f2', 'f4', 'f8', 'i1', 'i4', 'i8', 'u1', 'u2', 'u4'])
@pytest.mark.parametrize('shape,nonzero', [pytest.param((), False, id='scalar-zero'),
    pytest.param((1,), True, id='vector-nonzero'),
    pytest.param((1, 1), False, id='matrix-zero'),
    pytest.param((1, 1, 1), True, id='rank3-nonzero')])
def test_public_cross_entropy_array_truth_reads_single_value(dtype, shape, nonzero):
    number = (-1 if dtype.startswith(('f', 'i')) else 1) if nonzero else 0
    value = Array.from_buffer(np.full(shape, number, dtype=dtype))
    # Independent existing public contract, outside the native-call audit.
    want = bool(value)
    assert want is nonzero
    x, y, _ = prepared('logits:c')
    audit = audit_native_call(T.cross_entropy, x, y, return_grad=value, numeric_mode='identical')
    compare_output(audit.assert_native(), reference(x, y, {'return_grad': want}))


@pytest.mark.parametrize('dtype', ['f2', 'f4', 'f8'])
@pytest.mark.parametrize('number', [-0.0, -1.5, np.inf, -np.inf, np.nan],
                         ids=['negative-zero', 'negative', 'inf', 'negative-inf', 'nan'])
def test_public_cross_entropy_array_truth_preserves_float_values(dtype, number):
    value = Array.from_buffer(np.array(number, dtype=dtype))
    want = bool(value)
    assert want is bool(number)
    x, y, _ = prepared('logits:c')
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        audit = audit_native_call(T.cross_entropy, x, y, return_grad=value, numeric_mode='identical')
    compare_output(audit.assert_native(), reference(x, y, {'return_grad': want}))
    assert not caught


@pytest.mark.parametrize('shape', [(0,), (1, 0), (2,), (1, 2)],
                         ids=['empty', 'matrix-empty', 'vector-many', 'matrix-many'])
def test_public_cross_entropy_array_truth_preserves_ambiguous_error(shape):
    value = Array.from_buffer(np.zeros(shape, dtype='f4'))
    with pytest.raises(ValueError) as expected:
        bool(value)
    x, y, _ = prepared('logits:c')
    audit = audit_native_call(T.cross_entropy, x, y, return_grad=value, numeric_mode='identical')
    assert_refusal(audit, ValueError)
    assert str(audit.exception) == str(expected.value)


@pytest.mark.parametrize('arm', ['arm_lossgrad','arm_lossvalue'])
def test_existing_cross_entropy_analytic_contract(arm):
    from mojolearn.tests import test_training_surface as existing
    report = existing.Report(True)
    getattr(existing, arm)(report)
    assert report.rows and not report.failures, report.failures


@pytest.mark.parametrize('case', ['targets-float','targets-rank','logits-rank','none-backward'])
def test_public_cross_entropy_device_preserves_refusal_categories(case):
    if _backend.vendor() == 'cpu': pytest.skip('training device pool requires GPU')
    x,y,options = prepared('logits:c')
    category = TypeError
    if case == 'targets-float': y = y.astype('f4')
    elif case == 'targets-rank': y = y.reshape(2,2)
    elif case == 'logits-rank': x,category = x.ravel(),ValueError
    else: options.update(reduction='none',return_grad=True);category = NotImplementedError
    device = audit_native_call(T.train_to_device,x,numeric_mode='identical').assert_native()
    assert_refusal(audit_native_call(T.cross_entropy,device,y,**options),category)
    audit_native_call(T.TrainDeviceArray.__del__,device).assert_native()
