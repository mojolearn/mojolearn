#!/usr/bin/env python3
"""Future full-workload AFCL runner. Authored only: NEVER RUN in this delivery.

Uses existing board builders, public estimator operations and independent
quality functions. Requires audited real or established synthetic full inputs. No compile,
identity rerun, dataset download, opponent fitting or board publication occurs.
"""
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
# This directory contains select.py, which must not shadow stdlib select when
# subprocess imports selectors. Import AFCL selection explicitly if needed.
sys.path[:] = [p for p in sys.path if Path(p or '.').resolve() != HERE]

import argparse
import hashlib
import importlib
import importlib.util
import json
import math
import os
import platform
import socket
import subprocess
import time
import traceback

CATALOG = HERE / 'full_workloads.json'
HARNESS = {
    'classical': 'tools/classical_two_datasets.py',
    'classical2': 'tools/bench_board_more.py',
    'algos': 'tools/bench_board_algos.py',
    'trees': 'bench/speed/forest_speed_arm.py',
}
GATES = ('full_dataset_coverage', 'actual_candidate_route',
         'task_quality_no_regression', 'api_contract', 'matched_full_operation')
CLASSICAL_CONSTRUCTORS = {
    'GaussianProcessRegressor', 'GaussianProcessClassifier', 'KernelDensity',
    'NearestNeighbors', 'KNeighborsClassifier', 'KNeighborsRegressor', 'ARIMA',
    'AutoARIMA', 'HoltWinters', 'ETS', 'SimpleImputer', 'IterativeImputer',
    'PowerTransformer', 'StandardScaler', 'MinMaxScaler', 'MaxAbsScaler',
    'TargetEncoder', 'SelectKBest', 'GaussianNB', 'CategoricalNB',
    'MultinomialNB', 'ComplementNB', 'RandomForestClassifier',
    'RandomForestRegressor', 'ExtraTreesClassifier', 'ExtraTreesRegressor',
    'IsolationForest', 'GradientBoosting', 'TreeExplainer', 'GaussianMixture', 'RBF',
    'WhiteKernel', 'ConstantKernel', 'DotProduct', 'Matern',
    'KMeans','MiniBatchKMeans','BisectingKMeans','MeanShift','PCA','TruncatedSVD',
    'DBSCAN','HDBSCAN','SpectralClustering','SpectralEmbedding','UMAP',
    'AgglomerativeClustering','IVFIndex','RadiusNeighbors','ExponentialSmoothing',
    'LinearRegression','LogisticRegression','Ridge','RidgeClassifier','RidgeCV',
    'ElasticNet','Lasso','LassoCV','ElasticNetCV','Lars','LassoLars','BayesianRidge',
    'ARDRegression','HuberRegressor','SGDClassifier','SGDRegressor','Perceptron',
    'PassiveAggressiveClassifier','PassiveAggressiveRegressor','SGDOneClassSVM',
    'QNRegressor','SVC','SVR','LinearSVC','LinearSVR','OneClassSVM','KernelRidge',
    'RBFSampler','Nystroem','NMF','FactorAnalysis','PLSRegression','PLSCanonical',
    'CCA','MinCovDet','EllipticEnvelope','LinearDiscriminantAnalysis',
    'QuadraticDiscriminantAnalysis','RandomTreesEmbedding','AlternatingLeastSquares',
}
CLASSICAL_METHODS = {'fit','fit_transform','fit_predict','transform','predict',
    'predict_proba','score','score_samples','decision_function','kneighbors',
    'path_lengths','shap_values','forecast','partial_fit','query','search','radius_neighbors'}
CLASSICAL_FUNCTIONS = {'mean_squared_error','root_mean_squared_error',
    'mean_absolute_error','mean_squared_log_error','median_absolute_error',
    'r2_score','explained_variance_score','max_error','mean_absolute_percentage_error',
    'roc_curve','precision_recall_curve','roc_auc_score','average_precision_score',
    'auc','weighted_percentile','resample','f_regression','r_regression','f_classif',
    'kpss_test','select_d','bootstrap','permutation_test','resample_indices',
    'randomized_svd','qr','svd','lstsq','covariance','cov'}


def normalized(value):
    return json.loads(json.dumps(value, sort_keys=True, default=str))


def require_owned_apple():
    # Reuse the maintained queue and Apple M3 Ultra timing-host guard. This
    # function is called only by a future execution, never during authoring.
    policy = load_module('_afcl_queue_policy', REPO / 'tools/performance_ideas.py')
    policy.require_queue_host('time', {'mode':'fast'}, 'apple')
    def sysctl(name):
        return subprocess.run(['sysctl','-n',name],capture_output=True,text=True,
                              check=True).stdout.strip()
    return {'hostname':socket.gethostname(), 'platform':platform.platform(),
            'chip':sysctl('machdep.cpu.brand_string'), 'hardware_model':sysctl('hw.model'),
            'queue_job':os.environ.get('MOJOLEARN_PERFORMANCE_QUEUE_JOB')}


def resolve_callable(target, reference=False):
    """Resolve only declared classical public calls; no eval or shell code."""
    parts = target.split('.')
    if reference:
        if parts[0] not in ('sklearn','scipy','numpy','statsmodels','statsforecast'):
            raise ValueError('Independent reference must use a declared CPU scientific library')
        for end in range(len(parts)-1, 0, -1):
            try:
                value = importlib.import_module('.'.join(parts[:end]))
            except ModuleNotFoundError:
                continue
            for name in parts[end:]:
                value = getattr(value, name)
            return value
        raise ValueError('Reference callable could not be resolved: '+target)
    if parts[0] == 'mojolearn':
        parts = parts[1:]
    if not parts or parts[-1] not in CLASSICAL_CONSTRUCTORS | CLASSICAL_FUNCTIONS:
        raise ValueError('Public operation is not in the classical AFCL allowlist: '+target)
    value = importlib.import_module('mojolearn')
    for index, name in enumerate(parts):
        try:
            value = getattr(value, name)
        except AttributeError:
            value = importlib.import_module('mojolearn.'+'.'.join(parts[:index+1]))
    return value


class PublicProgram:
    """Declarative benchmark shell for full public calls absent from board presets.

    No estimator math is implemented here. Arguments are forwarded unchanged;
    fit/variance/metric/resample output consumption is explicit in the program.
    """
    def __init__(self, program, arrays, reference=False):
        self.program, self.arrays, self.reference = program, arrays, reference
        self.values, self.result, self.calls = {}, {}, []
        self.info = {'library':'independent-reference' if reference else 'mojolearn',
                     'numeric_mode_env':os.environ.get('MOJOLEARN_NUMERIC_MODE'),
                     'operation_scope':'declared complete public-operation program'}

    def argument(self, value):
        if isinstance(value,dict):
            if set(value)=={'array'}:
                self.consumed_arrays.add(value['array'])
                return self.arrays[value['array']]
            if set(value)=={'ref'}:
                return self.values[value['ref']]
            if 'constructor' in value:
                constructor=resolve_callable(value['constructor'],self.reference)
                return constructor(**{key:self.argument(item) for key,item in value.get('kwargs',{}).items()})
            return {key:self.argument(item) for key,item in value.items()}
        if isinstance(value,list):
            return [self.argument(item) for item in value]
        return value

    def consume(self, key, value):
        import numpy as np
        if getattr(value,'format',None)=='csr':
            for name in ('data','indices','indptr'):
                self.consume(key+'.'+name,getattr(value,name))
            self.consume(key+'.shape',np.asarray(value.shape,dtype=np.int64))
        elif isinstance(value,(tuple,list)):
            for index,item in enumerate(value):
                self.consume(key+'.'+str(index),item)
        elif isinstance(value,dict):
            for name,item in value.items():
                self.consume(key+'.'+name,item)
        else:
            array=np.asarray(value)
            if array.dtype.hasobject:
                raise ValueError('Consume numeric outputs, not an estimator object: '+key)
            self.result[key]=np.ascontiguousarray(array)

    def call(self):
        self.values,self.result,self.calls={},{},[]
        self.consumed_arrays=set()
        for operation in self.program:
            kind=operation['kind']
            args=[self.argument(value) for value in operation.get('args',[])]
            kwargs={key:self.argument(value) for key,value in operation.get('kwargs',{}).items()}
            if kind=='method':
                if operation['method'] not in CLASSICAL_METHODS:
                    raise ValueError('Unsupported classical method: '+operation['method'])
                owner=self.values[operation['object']]
                function=getattr(owner,operation['method'])
            elif kind in ('construct','function'):
                function=resolve_callable(operation['target'],self.reference)
            else:
                raise ValueError('Public program kind must be construct, method or function')
            inputs={str(index):list(value.shape) for index,value in enumerate(args) if hasattr(value,'shape')}
            inputs.update({key:list(value.shape) for key,value in kwargs.items() if hasattr(value,'shape')})
            start=time.perf_counter_ns()
            value=function(*args,**kwargs)
            self.values[operation['id']]=value
            if operation.get('consume_fields'):
                for label,path in operation['consume_fields'].items():
                    selected=value
                    for field in path.split('.'):
                        if field.startswith('_'):
                            raise ValueError('Consume only named public result fields')
                        selected=getattr(selected,field)
                    self.consume(operation['id']+'.'+label,selected)
            elif operation.get('consume',kind=='function'):
                self.consume(operation['id'],value)
            self.calls.append({'id':operation['id'],'kind':kind,'inputs':inputs,
                'ms':(time.perf_counter_ns()-start)/1e6,'declaration':operation})
            if kind=='construct' and not self.reference:
                for field in ('numeric_mode_used','vendor_used'):
                    if hasattr(value,field):
                        self.info[field]=getattr(value,field)()
        if not self.result:
            raise ValueError('The complete public program consumed no numeric output')

    def outputs(self):
        return self.result

    def sync(self):
        # Supported public functions return host arrays/scalars. Numeric output
        # materialization in consume is inside the complete operation clock.
        return None


def compare_public_outputs(actual, reference):
    import numpy as np
    if set(actual)!=set(reference):
        raise ValueError('Independent reference output names differ from public program')
    result={}
    for key,value in actual.items():
        left,right=np.asarray(value),np.asarray(reference[key])
        if left.shape!=right.shape:
            raise ValueError('Independent reference shape differs for '+key)
        lf,rf=np.isfinite(left),np.isfinite(right)
        if not np.array_equal(lf,rf) or np.isnan(left).any() or np.isnan(right).any():
            raise ValueError('Nonfinite output disagrees with independent contract: '+key)
        if not np.array_equal(left[~lf],right[~rf]):
            raise ValueError('Infinite sentinel differs from independent reference: '+key)
        difference=left[lf].astype(np.float64)-right[rf].astype(np.float64)
        denominator=float(np.linalg.norm(right[rf].astype(np.float64)))
        result[key+'_max_abs_error']=float(np.max(np.abs(difference))) if difference.size else 0.0
        result[key+'_relative_l2_error']=float(np.linalg.norm(difference))/(denominator or 1.0)
    return result


def install_call_traces(module, rules, arrays):
    """Record actual arguments at selected public call boundaries, including caps."""
    observed=[]
    for rule in rules:
        parts=rule['target'].removeprefix('mojolearn.').split('.')
        owner=module
        for name in parts[:-1]:
            owner=getattr(owner,name)
        name=parts[-1]
        original=getattr(owner,name)
        bound=isinstance(owner,type)
        def wrapped(*args,_original=original,_rule=rule,_bound=bound,**kwargs):
            positional=args[1:] if _bound else args
            receipt={'target':_rule['target'],'arrays':{},'_values':{}}
            for position,array_name in _rule['arguments'].items():
                value=positional[int(position)] if position.isdigit() else kwargs[position]
                expected=arrays[array_name]
                if not hasattr(value,'shape') or tuple(value.shape)!=tuple(expected.shape):
                    raise ValueError('Public call slices an audited full input: '+_rule['target']+' '+array_name)
                receipt['arrays'][array_name]=list(value.shape)
                receipt['_values'][array_name]=value
            observed.append(receipt)
            return _original(*args,**kwargs)
        setattr(owner,name,wrapped)
    return observed


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + '\n')
    temporary.replace(path)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for part in iter(lambda: stream.read(8 << 20), b''):
            digest.update(part)
    return digest.hexdigest()


def file_receipt(value, base):
    if not isinstance(value, dict) or not value.get('path') or not value.get('sha256'):
        raise ValueError('File receipt needs path and sha256')
    path = Path(value['path']).expanduser()
    if not path.is_absolute():
        path = Path(base) / path
    path = path.resolve()
    actual = sha256(path)
    if actual != value['sha256']:
        raise ValueError('Artifact hash mismatch: ' + str(path))
    return path


def load_module(name, path):
    sys.path.insert(0, str(REPO / 'tools'))
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def array_receipts(arrays):
    import numpy as np
    out = {}
    for key, value in arrays.items():
        if hasattr(value, 'shape'):
            if getattr(value,'format',None)=='csr':
                out[key]={'shape':list(value.shape),'dtype':str(value.dtype),'format':'csr',
                    'components':array_receipts({name:getattr(value,name) for name in ('data','indices','indptr')})}
                continue
            data = np.ascontiguousarray(value)
            out[key] = {'shape': list(data.shape), 'dtype': str(data.dtype),
                        'sha256': hashlib.sha256(data.data).hexdigest()}
    return out


def lane_entries():
    entries = {}
    for path in sorted((HERE / 'lanes').glob('*.json')):
        for entry in read_json(path)['entries']:
            entries[entry['id']] = entry
    return entries


def arm_environment(arm, package_root, entries, shared):
    env = os.environ.copy()
    # Clear *every* AFCL card's runtime controls before adding this arm's
    # prerequisites. Otherwise an inherited B-only flag can silently reach A.
    keys = {'MOJOLEARN_ALGOS_SMOKE_ROWS', 'MOJOLEARN_SPEED_SIZE',
            'MOJOLEARN_NUMERIC_MODE', 'MOJOLEARN_VENDOR', 'MOJOLEARN_BENCH_THREADS'}
    for entry in entries.values():
        keys.update(entry.get('baseline_env', {}))
        keys.update(entry.get('candidate_env', {}))
    for key in tuple(env):
        if key in keys or key.startswith('MOJOLEARN_AFCL_'):
            env.pop(key, None)
    # Full machine policy on dedicated Apple hosts. Harness numerical
    # semantics (such as implicit's internal BLAS=1) remain its own policy.
    for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
                'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'BLIS_NUM_THREADS'):
        env.pop(key, None)
    env.update({str(k): str(v) for k, v in shared.items()})
    env.update({str(k): str(v) for k, v in arm.get('runtime_environment', {}).items()})
    env.update(MOJOLEARN_NUMERIC_MODE='fast', MOJOLEARN_VENDOR='metal',
               MOJOLEARN_BENCH_INSTALLED='1', MOJOLEARN_BENCH_WHOLE_OPERATION='1',
               MOJOLEARN_SPEED_SIZE='shipped', MOJOLEARN_SPEED_ROUNDS='1',
               MOJOLEARN_SPEED_EXPECTED_VENDOR='metal', PYTHONNOUSERSITE='1',
               PYTHONDONTWRITEBYTECODE='1')
    env['PYTHONPATH'] = os.pathsep.join((str(package_root), str(REPO / 'tools')))
    if os.environ.get('MOJOLEARN_AFCL_RECIPE_SHA256'):
        env['MOJOLEARN_AFCL_RECIPE_SHA256'] = os.environ['MOJOLEARN_AFCL_RECIPE_SHA256']
    return env


def audit_workload(workload, base, recipe):
    """Fail closed before any GPU worker; author declarations are not PASS."""
    kind = workload.get('dataset_kind')
    if kind not in ('real', 'synthetic'):
        raise ValueError('Declare real or established synthetic dataset_kind')
    if kind == 'synthetic':
        if not recipe.get('established_synthetic', False):
            raise ValueError('A synthetic replacement for a saved real task is not admissible')
        generator = workload.get('generator', {})
        if not generator.get('version') or 'seed' not in generator or not generator.get('full_dimensions'):
            raise ValueError('Established synthetic tasks require frozen generator/version/seed/full dimensions')
        file_receipt(generator['source'], base)
    if workload.get('input_mode') != 'audited_lane_inputs':
        raise ValueError('Use audited_lane_inputs; saved lane_arrays intrinsic caps cannot establish full coverage')
    identity = workload.get('dataset_identity', {})
    if not identity.get('name') or not identity.get('version'):
        raise ValueError('Dataset name/version must be explicit')
    sources = identity.get('sources', [])
    if not sources:
        raise ValueError('Dataset source/cache file hashes and paths are required')
    for source in sources:
        file_receipt(source, base)
    full = workload.get('full_dataset_audit', {})
    if full.get('complete') is not True or full.get('row_subsampling') is not False:
        raise ValueError('An explicit full-population, no-row-subsampling audit is required')
    if not full.get('source_populations') or not full.get('array_populations'):
        raise ValueError('Declare full fit/eval/index/query/series populations and their input-array axes')
    if not full.get('lineage') or not full.get('preprocessing'):
        raise ValueError('Record source split lineage and input preprocessing; full dimensions alone are insufficient')
    if not workload.get('expected_arrays') or not workload.get('expected_params'):
        raise ValueError('Exact input shape/dtype/hash and constructed estimator settings are required')
    if recipe['adapter'] != 'public' and not workload.get('trace_calls'):
        raise ValueError('Trace the actual public calls and their consumed input arrays; file shapes alone do not prove coverage')
    if recipe['adapter'] == 'public':
        if not workload.get('public_program'):
            raise ValueError('Public-operation adapter needs its exact declarative operation program')
        if not workload.get('reference_program') and not workload.get('reference_outputs'):
            raise ValueError('Public operations require independent reference outputs or a reference program')
    if not workload.get('quality_gates'):
        raise ValueError('Existing independent task metric floors are required before timing')
    for gate in workload['quality_gates']:
        file_receipt(gate['policy_evidence'], base)
        if not gate.get('metric') or gate.get('direction') not in ('higher', 'lower'):
            raise ValueError('Each task floor needs a metric and higher/lower direction')
        if 'floor' not in gate or 'max_degradation_abs' not in gate:
            raise ValueError('Existing floor and permitted absolute A/B degradation must be explicit')
        if (not math.isfinite(float(gate['floor'])) or
                not math.isfinite(float(gate['max_degradation_abs'])) or
                float(gate['max_degradation_abs']) < 0):
            raise ValueError('Quality floors and degradation limits must be finite and valid')
    if workload.get('operation_boundary') != 'prepare-fit-synchronize-infer-consume':
        raise ValueError('Whole-operation boundary must include preparation, fit, synchronization and consumed outputs')
    arrays_path = file_receipt(workload['arrays_npz'], base)
    record_path = file_receipt(workload['record_json'], base) if workload.get('record_json') else None
    for reference in workload.get('reference_outputs', []):
        file_receipt(reference, base)
        provenance=read_json(file_receipt(reference['provenance'], base))
        if provenance.get('arrays_npz_sha256') != workload['arrays_npz']['sha256']:
            raise ValueError('Independent reference belongs to different lane inputs')
        if not provenance.get('library_version') or not provenance.get('settings'):
            raise ValueError('Independent reference provenance needs library version and exact settings')
    return dict(workload, arrays_npz=dict(workload['arrays_npz'], path=str(arrays_path)),
                record_json=(dict(workload['record_json'], path=str(record_path)) if record_path else None),
                adapter=recipe['adapter'], lane=recipe['lane'], audit_base=str(base))


def load_lane_inputs(task):
    import numpy as np
    workload = task['workload']
    path = file_receipt(workload['arrays_npz'], workload['audit_base'])
    with np.load(path, allow_pickle=False) as archive:
        arrays = {name: np.ascontiguousarray(archive[name]) for name in archive.files}
    for name,descriptor in workload.get('sparse_inputs',{}).items():
        if name in arrays or descriptor.get('format')!='csr':
            raise ValueError('Sparse inputs require a distinct CSR name and exact component arrays')
        from scipy.sparse import csr_matrix
        parts=[arrays.pop(descriptor[key]) for key in ('data','indices','indptr')]
        arrays[name]=csr_matrix(tuple(parts),shape=tuple(descriptor['shape']),copy=False)
    actual = array_receipts(arrays)
    if actual != workload['expected_arrays']:
        raise ValueError('Actual lane-input shapes/dtypes/hashes differ from the audited contract')
    audit = workload['full_dataset_audit']
    visited = set()
    for key, descriptor in audit['array_populations'].items():
        if key not in arrays:
            raise ValueError('Population array missing: ' + key)
        population = descriptor['population']
        source = audit['source_populations'][population]
        count = int(source['full_count'])
        axis = int(descriptor.get('axis', 0))
        if count <= 0 or arrays[key].shape[axis] != count:
            raise ValueError('Full population is not consumed for ' + key)
        if not source.get('source_split'):
            raise ValueError('Source split or series/time population must be named')
        visited.add(population)
    if visited != set(audit['source_populations']):
        raise ValueError('At least one declared full population has no consumed input array')
    for key, value in workload.get('scalars', {}).items():
        if key in arrays:
            raise ValueError('Scalar must not overwrite an audited array: ' + key)
        arrays[key] = value
    record = read_json(workload['record_json']['path']) if workload.get('record_json') else {}
    record['dataset'] = workload['dataset_identity']['name']
    if record.get('smoke_max_rows'):
        raise ValueError('Smoke prepared record cannot be a full workload')
    if record.get('synthetic') and workload.get('dataset_kind') != 'synthetic':
        raise ValueError('Synthetic record cannot be passed as a real workload')
    return arrays, record, actual


def make_forest_data(module, arrays, workload):
    metadata = workload.get('forest_data', {})
    if metadata.get('task') not in ('regression', 'binary', 'multiclass', 'ranking', 'anomaly'):
        raise ValueError('Tree workload requires its exact existing task')
    data = module.spec.Data(workload['dataset_identity']['name'], arrays['X_train'],
        arrays['X_test'], arrays['y_train'], arrays['y_test'], metadata['task'],
        n_classes=int(metadata.get('n_classes', 0)), y_anom=arrays.get('y_anom'),
        cat_idx=metadata.get('cat_idx', []))
    for key in ('qid_train', 'qid_test'):
        if key in arrays:
            setattr(data, key, arrays[key])
    return data


def make_runner(module, workload, arrays, record):
    family, lane = workload['adapter'], workload['lane']
    if family == 'public':
        return PublicProgram(workload['public_program'], arrays, reference=False)
    if family == 'classical':
        return module.BUILDERS[(lane, 'ours')](arrays, record)
    if family == 'classical2':
        return module.build(lane, 'ours-fast', arrays, record)
    if family == 'algos':
        module._DATASET = workload['dataset_identity']['name']
        return module.build(lane, 'ours-fast', arrays)
    data = make_forest_data(module, arrays, workload)
    config = module.spec.lane_config(lane, 'shipped')
    if data.task == 'multiclass':
        config['n_classes'] = data.n_classes
    module.spec.prepare_cuml_labels(data)
    module.spec.prepare_anomaly_labels(lane, data)
    module.prepare_our_inputs(data)
    arms = module.build_ours(lane, config, data)
    if len(arms) != 1:
        raise RuntimeError('Existing forest arm refused construction')
    arm = arms[0]
    return {'arm': arm, 'data': data, 'model': arm.make(), 'config': config}


def parameters(runner, family, ctd):
    if family == 'public':
        return runner.program
    if family == 'trees':
        return ctd.params_record(runner['model'])
    if family == 'algos':
        return runner.record
    return ctd.params_record(getattr(runner, 'params_obj' if family == 'classical' else 'params', None))


def synchronize(runner, family):
    if family == 'trees':
        runner['arm'].sync()
    elif family == 'algos':
        runner.sync_for_receipt()
    else:
        runner.sync()


def one_operation(module, workload, arrays, record, ctd):
    family = workload['adapter']
    started = time.perf_counter_ns()
    # Benchmark preparation: each complete operation gets fresh full inputs.
    # Parsing the saved source corpus is separately documented CPU input work.
    arrays={key:value.copy() if hasattr(value,'shape') else value for key,value in arrays.items()}
    runner = make_runner(module, workload, arrays, record)
    synchronize(runner, family)
    params = normalized(parameters(runner, family, ctd))
    if params != workload['expected_params']:
        raise ValueError('Constructed settings differ from the audited exact settings readback')
    prepared = time.perf_counter_ns()
    if family == 'trees':
        runner['arm'].fit(runner['model'], runner['data'])
    elif family == 'algos':
        runner.fit()
    else:
        runner.call()
    synchronize(runner, family)
    fitted = time.perf_counter_ns()
    independent = None
    inference_scope = 'consumed outputs include inference where the saved worker does so'
    if family == 'trees':
        triples = list(runner['arm'].score(runner['model'], runner['data']))
        outputs = {str(name): prediction for name, _, prediction in triples}
        independent = {str(name): float(value) for name, value, _ in triples}
        inference_scope = 'existing forest scoring call, full heldout arrays, including independent score calculation'
    else:
        if family == 'algos':
            inferred = runner.infer()
            inference_scope = 'existing Runner.infer' if inferred else 'no separate saved inference call'
        outputs = runner.outputs()
    synchronize(runner, family)
    consumed = time.perf_counter_ns()
    info = (runner.info if family != 'trees' else {'library': 'mojolearn',
            'numeric_mode_used': runner['model'].numeric_mode_used(),
            'vendor_used': runner['model'].vendor_used()})
    return runner, outputs, independent, {
        'whole_operation_ms': (consumed-started)/1e6,
        'preparation_ms': (prepared-started)/1e6,
        'fit_or_primary_call_ms': (fitted-prepared)/1e6,
        'inference_and_output_consumption_ms': (consumed-fitted)/1e6,
        'inference_scope': inference_scope,
        'dataset_load_included': False, 'input_preprocessing': workload['full_dataset_audit']['preprocessing'],
        'boundary': 'prepare-fit-synchronize-infer-consume',
        'params': params, 'info': info,
        'public_calls': runner.calls if family == 'public' else [],
    }


def independent_quality(module, workload, arrays, record, outputs, forest_quality, evidence_dir):
    if workload['adapter'] == 'trees':
        return forest_quality
    import numpy as np
    all_outputs = {'ours': outputs}
    for reference in workload.get('reference_outputs', []):
        path = file_receipt(reference, workload['audit_base'])
        with np.load(path, allow_pickle=False) as archive:
            all_outputs[reference['arm']] = {key: np.ascontiguousarray(archive[key]) for key in archive.files}
    if workload['adapter'] == 'public':
        reference_outputs = all_outputs.get('independent-reference')
        if reference_outputs is None:
            reference = PublicProgram(workload['reference_program'], arrays, reference=True)
            reference.call()
            reference_outputs = reference.outputs()
            path=Path(evidence_dir)/'independent-reference.outputs.npz'
            np.savez(path, **reference_outputs)
            write_json(Path(evidence_dir)/'independent-reference.json', {
                'kind':'untimed_independent_reference','program':workload['reference_program'],
                'arrays_npz_sha256':workload['arrays_npz']['sha256'],
                'outputs_sha256':sha256(path), 'calls':reference.calls})
        return compare_public_outputs(outputs, reference_outputs)
    if workload['adapter'] == 'classical':
        values = module.quality(workload['lane'], arrays, all_outputs, record)
    else:
        values = module.quality(workload['lane'], arrays, all_outputs)
    return values.get('ours', {})


def worker(task_path, output):
    result = {'status': 'running', 'stage': 'load', 'quality_status': 'PENDING'}
    try:
        task = read_json(task_path)
        result['machine']=require_owned_apple()
        workload = task['workload']
        family = workload['adapter']
        arrays, record, actual = load_lane_inputs(task)
        import mojolearn
        package_path = Path(mojolearn.__file__).resolve()
        if not package_path.is_relative_to(Path(task['package_root']).resolve()):
            raise RuntimeError('An unintended mojolearn package was imported: ' + str(package_path))
        ctd = load_module('_afcl_ctd', REPO / HARNESS['classical'])
        module = (None if family == 'public' else ctd if family == 'classical' else
                  load_module('_afcl_workload', REPO / HARNESS[family]))
        traced=install_call_traces(mojolearn,workload.get('trace_calls',[]),arrays)
        result.update(actual_arrays=actual, package_path=str(package_path), rounds=[],
                      source_sha=task['source_sha'], workload_id=workload['id'], stage='operations')
        # One excluded first/cold operation, one scored repeated operation.
        # No competing arm/model stays live on the GPU when the next arm starts.
        runner = None
        outputs = None
        for sample in range(2):
            runner = None
            outputs = None
            traced.clear()
            runner, outputs, forest_quality, timing = one_operation(module, workload, arrays, record, ctd)
            expected_runtime={key for key,descriptor in workload['full_dataset_audit']['array_populations'].items()
                              if descriptor.get('usage','runtime')=='runtime'}
            consumed=(runner.consumed_arrays if family=='public' else
                      {key for receipt in traced for key in receipt['arrays']})
            if not expected_runtime.issubset(consumed):
                raise ValueError('Full runtime populations were not observed at public calls: '+
                                 ', '.join(sorted(expected_runtime-consumed)))
            expected_targets={rule['target'] for rule in workload.get('trace_calls',[])}
            if expected_targets != {receipt['target'] for receipt in traced}:
                raise ValueError('An audited public call was not observed')
            call_receipts=[]
            for receipt in traced:
                hashes=array_receipts(receipt.pop('_values'))
                for name,digest in hashes.items():
                    if digest != actual[name]:
                        raise ValueError('Public call changed, sampled or mutated an audited array: '+name)
                call_receipts.append(dict(receipt,input_receipts=hashes))
            timing.update(sample=sample, excluded_warmup=(sample == 0),
                          use='cold_first_operation_excluded' if sample == 0 else 'repeated_scored_operation',
                          outputs=array_receipts(outputs), actual_public_calls=call_receipts,
                          consumed_full_arrays=sorted(consumed))
            result['rounds'].append(timing)
            write_json(output, result)
        import numpy as np
        outputs_path = Path(output).with_suffix('.outputs.npz')
        np.savez(outputs_path, **outputs)
        result.update(stage='quality', outputs_path=str(outputs_path), outputs_sha256=sha256(outputs_path))
        result['quality_metrics'] = independent_quality(module, workload, arrays, record, outputs, forest_quality, Path(output).parent)
        result['quality_source'] = ('audited independent public-operation reference' if family=='public' else
            HARNESS[family] + ('::existing Arm.score' if family == 'trees' else '::quality'))
        result['runtime_environment'] = {key:value for key,value in os.environ.items()
            if key.startswith('MOJOLEARN_') or key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS',
                 'MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','PYTHONPATH')}
        try:
            from threadpoolctl import threadpool_info
            result['effective_thread_pools'] = threadpool_info()
        except ImportError:
            result['effective_thread_pools'] = 'unavailable; no utilization claim'
        result.update(status='completed_unqualified', stage='complete',
            actual_candidate_route='PENDING: kernel execution witness required',
            api_contract='PENDING: full API contract evidence is separate from saved worker outputs')
        write_json(output, result)
        return 0
    except BaseException as exc:
        traceback.print_exc()
        result.update(status='failed', error=repr(exc))
        write_json(output, result)
        return 1


def quality_gates(workload, baseline, candidate):
    details = []
    for gate in workload['quality_gates']:
        name = gate['metric']
        a = baseline.get('quality_metrics', {}).get(name)
        b = candidate.get('quality_metrics', {}).get(name)
        entry = {'metric':name,'baseline':a,'candidate':b,'policy_evidence':gate['policy_evidence']}
        if not isinstance(a,(int,float)) or not isinstance(b,(int,float)) or not math.isfinite(a) or not math.isfinite(b):
            entry.update(status='PENDING', reason='Independent metric missing or nonfinite')
        else:
            floor = float(gate['floor'])
            tolerance = float(gate['max_degradation_abs'])
            if gate['direction'] == 'higher':
                passed = a >= floor and b >= floor and b >= a - tolerance
            else:
                passed = a <= floor and b <= floor and b <= a + tolerance
            entry['status'] = 'PASS' if passed else 'FAIL'
        details.append(entry)
    status = ('FAIL' if any(x['status']=='FAIL' for x in details) else
              'PASS' if details and all(x['status']=='PASS' for x in details) else 'PENDING')
    return {'status':status,'metrics':details}


def resolve_package(build_root, arm):
    relative = arm.get('package_relative')
    package = (build_root / relative if relative else Path(arm['package_root'])).resolve()
    for name, receipt in arm.get('artifacts', {}).items():
        path = package / 'mojolearn' / name
        if sha256(path) != receipt['sha256']:
            raise ValueError('Paired staged binary changed: ' + str(path))
    if not arm.get('artifacts'):
        raise ValueError('Paired package has no retained binary receipts')
    for name, receipt in arm.get('runtime_libraries', {}).items():
        if sha256(package / 'mojolearn' / name) != receipt['sha256']:
            raise ValueError('Paired runtime library changed: ' + name)
    for name, digest in arm.get('python_sources_sha256', {}).items():
        if sha256(package / 'mojolearn' / name) != digest:
            raise ValueError('Paired Python API source changed: ' + name)
    return package


def proof_context(summary, idea):
    return {'id':idea, 'source_sha':summary['source_sha'], 'mode':'fast','vendor':'apple',
        'workloads_sha256':summary.get('workloads_sha256'),
        'paired_build_sha256':summary.get('paired_build_sha256'),
        'manifest_sha256':summary.get('recipe_digests',{}).get(idea),
        'artifact_hashes':summary.get('artifact_hashes')}


def load_external_proofs(audited, base, summary):
    """Reuse accepted same-freeze evidence; declarations and exit codes cannot pass.

    The bundle is an external acceptance receipt from the existing route/API
    evidence process, not a second quality implementation in this runner.
    It must retain immutable underlying execution evidence and its acceptance.
    A stable path in the workload config permits attaching later evidence
    without changing the audited workload hash or requiring another GPU run.
    """
    name=audited.get('evidence_bundle_path')
    if not name:
        return []
    path=Path(name)
    if not path.is_absolute():
        path=base/path
    if not path.exists():
        summary['external_evidence']={'path':str(path),'status':'PENDING'}
        return []
    bundle=read_json(path)
    if bundle.get('schema')!=1:
        raise ValueError('Unsupported external evidence bundle schema')
    accepted=[]
    for receipt in bundle.get('receipts',[]):
        idea=receipt.get('id')
        if idea not in summary['ideas']:
            continue
        if receipt.get('status')!='PASS':
            continue
        expected=proof_context(summary,idea)
        if not expected['manifest_sha256']:
            continue
        for key,value in expected.items():
            if receipt.get(key)!=value:
                raise ValueError('External evidence mismatches '+key+' for '+str(idea))
        if receipt.get('gate') not in ('actual_candidate_route','api_contract','full_dataset_coverage'):
            raise ValueError('External bundle cannot override computed task quality or matched-operation gates')
        if not receipt.get('accepted_by') or not receipt.get('execution_evidence'):
            raise ValueError('External evidence requires named acceptance and immutable execution artifacts')
        file_receipt(receipt['acceptance_evidence'],path.parent)
        for artifact in receipt['execution_evidence']:
            file_receipt(artifact,path.parent)
        if receipt['gate']=='actual_candidate_route':
            if not receipt.get('executed_candidate_symbols') or not receipt.get('baseline_control_witness'):
                raise ValueError('Route evidence needs candidate execution symbols and a baseline control witness')
            if receipt.get('machine',{}).get('chip')!=summary.get('machine',{}).get('chip'):
                raise ValueError('Route witness was not accepted for this Apple chip')
        elif receipt['gate']=='api_contract' and not receipt.get('checks'):
            raise ValueError('API receipt needs the retained actual contract checks')
        elif receipt['gate']=='full_dataset_coverage':
            if not receipt.get('covered_callers') or not receipt.get('covered_workload_ids'):
                raise ValueError('Scope resolution must map callers/settings to complete workload receipts')
        accepted.append(receipt)
    summary['external_evidence']={'path':str(path.resolve()),'sha256':sha256(path),
        'accepted_receipts':len(accepted),'status':'attached' if accepted else 'PENDING'}
    return accepted


def external_gate(proofs, gate, ideas, workload_id):
    found=[receipt for receipt in proofs if receipt['gate']==gate and
           receipt.get('workload_id')==workload_id and receipt['id'] in ideas]
    if {receipt['id'] for receipt in found}==set(ideas) and ideas:
        return {'status':'PASS','source':'accepted same-freeze external evidence',
                'evidence':found}
    return {'status':'PENDING','reason':'Missing source/build/workload/recipe-matched accepted '+gate+' evidence'}


def resolve_scope(summary, catalog_entries, proofs):
    completed={item['id'] for item in summary['workloads'] if item.get('gates',{}).get(
        'matched_full_operation',{}).get('status')=='PASS'}
    resolutions=[receipt for receipt in proofs if receipt['gate']=='full_dataset_coverage'
                 and set(receipt['covered_workload_ids']).issubset(completed)]
    unresolved={}
    missing={}
    for idea in summary['ideas']:
        own=[receipt for receipt in resolutions if receipt['id']==idea]
        unresolved[idea]=[note for note in catalog_entries[idea]['pending_callers_or_settings']
            if not any(receipt.get('pending_note')==note for receipt in own)]
        covered={item['recipe'] for item in summary['workloads'] if idea in item.get('ideas',[]) and
                 item['id'] in completed}
        missing[idea]=[recipe for recipe in catalog_entries[idea]['recipes']
            if recipe not in covered and not any(receipt.get('resolved_recipe')==recipe for receipt in own)]
    summary.update(unresolved_catalog=unresolved,missing_recipe_coverage=missing,
        original_catalog_notes={idea:catalog_entries[idea]['pending_callers_or_settings'] for idea in summary['ideas']},
        accepted_scope_resolutions=resolutions)


def write_quality_receipts(output, summary):
    paths={}
    for idea in summary.get('ideas',[]):
        gates={name:summary['gates'][name]['status'] for name in GATES}
        context=proof_context(summary,idea)
        passed=all(status=='PASS' for status in gates.values()) and bool(context['manifest_sha256'])
        receipt=dict(context,schema=1,status='PASS' if passed else 'PENDING',gates=gates,
            reason=None if passed else 'Required execution, coverage, API, quality or hydrated recipe evidence remains pending',
            source_result=str(output/'result.json'),machine=summary.get('machine'))
        if any(status=='FAIL' for status in gates.values()):
            receipt['status']='FAIL'
        path=output/idea/'quality-receipt.json'
        write_json(path,receipt)
        paths[idea]=str(path)
        if len(summary.get('ideas',[]))==1:
            write_json(output/'quality-receipt.json',receipt)
    summary['quality_receipts']=paths


def finish_receipts(output, summary, by_id, proofs):
    resolve_scope(summary,by_id,proofs)
    for gate in GATES:
        values=[item.get('gates',{}).get(gate,{}).get('status','PENDING') for item in summary['workloads']]
        status='FAIL' if 'FAIL' in values else 'PASS' if values and all(v=='PASS' for v in values) else 'PENDING'
        if gate=='full_dataset_coverage' and (any(summary['missing_recipe_coverage'].values()) or
                                            any(summary['unresolved_catalog'].values())):
            status='PENDING' if status!='FAIL' else status
        summary['gates'][gate]={'status':status}
    summary.update(status='measured_unqualified',promotion_status='BLOCKED')
    if all(gate['status']=='PASS' for gate in summary['gates'].values()):
        summary['status']='qualified_validation' if summary['stage']=='validate' else 'qualified_measurement'
        summary['promotion_status']='PENDING_OWNER_DECISION_NO_DEFAULT_CHANGE'
    write_quality_receipts(output,summary)
    write_json(output/'result.json',summary)


def attach_evidence(args):
    """Attach immutable accepted receipts to retained operations, without GPU work."""
    original=Path(args.attach_evidence_to).resolve()
    output=Path(args.output).resolve()
    output.mkdir(parents=True,exist_ok=False)
    summary=read_json(original/'result.json')
    summary['attachment_source']={'path':str(original/'result.json'),'sha256':sha256(original/'result.json')}
    try:
        audit_path=Path(args.workloads).resolve()
        if sha256(audit_path)!=summary['workloads_sha256'] or sha256(CATALOG)!=summary['source_catalog_sha256']:
            raise ValueError('Evidence attachment requires the same workload audit and source catalog')
        build_path=Path(summary['paired_build_path'])
        if sha256(build_path)!=summary['paired_build_sha256']:
            raise ValueError('Retained paired-build receipt changed')
        for arm in read_json(build_path)['arms'].values():
            resolve_package(build_path.parent,arm)
        for item in summary['workloads']:
            for arm in item['arms'].values():
                if arm.get('exit_code')!=0 or sha256(arm['result'])!=arm.get('result_sha256'):
                    raise ValueError('Retained worker evidence is failed, missing or changed')
                data=read_json(arm['result'])
                if sha256(data['outputs_path'])!=data['outputs_sha256']:
                    raise ValueError('Retained consumed output artifact changed')
                arm['data']=data
        proofs=load_external_proofs(read_json(audit_path),audit_path.parent,summary)
        for item in summary['workloads']:
            for gate in ('actual_candidate_route','api_contract'):
                item['gates'][gate]=external_gate(proofs,gate,item['ideas'],item['id'])
            item['status']=('qualified_workload' if all(gate['status']=='PASS' for gate in item['gates'].values())
                            else 'measured_unqualified')
        finish_receipts(output,summary,{entry['id']:entry for entry in read_json(CATALOG)['entries']},proofs)
        print('Evidence attached without device execution: '+str(output/'result.json'))
        return 0
    except BaseException as exc:
        summary.update(status='failed_or_pending',error=repr(exc),promotion_status='BLOCKED')
        summary['gates']={name:{'status':'PENDING'} for name in GATES}
        write_quality_receipts(output,summary)
        write_json(output/'result.json',summary)
        print('Evidence attachment refused: '+str(exc),file=sys.stderr)
        return 1


def require_prior_quality(args, audited, base, summary):
    if args.stage=='validate':
        return
    declarations=audited.get('prior_quality_receipts',{})
    for idea in summary['ideas']:
        name=(os.environ.get('MOJOLEARN_AFCL_QUALITY_RECEIPT') if len(summary['ideas'])==1 else None)
        name=name or declarations.get(idea)
        if not name:
            raise ValueError('Timing requires accepted same-freeze quality first; use --stage validate or attach evidence')
        path=Path(name)
        if not path.is_absolute():
            path=base/path
        receipt=read_json(path)
        expected=proof_context(summary,idea)
        if not expected['manifest_sha256'] or receipt.get('status')!='PASS':
            raise ValueError('Timing quality receipt is pending or lacks hydrated recipe digest')
        for key,value in expected.items():
            if receipt.get(key)!=value:
                raise ValueError('Prior quality receipt mismatches '+key)
        if any(receipt.get('gates',{}).get(gate)!='PASS' for gate in GATES):
            raise ValueError('Every required quality gate must pass before timing')
        summary.setdefault('prior_quality_receipts',{})[idea]={'path':str(path),'sha256':sha256(path)}


def run(args):
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    summary = {'schema':1, 'status':'planning', 'promotion_status':'BLOCKED',
               'stage':args.stage,'source_sha':args.source_sha,'workloads':[],
               'ideas':list(dict.fromkeys(item for value in args.idea for item in value.split(',') if item)),
               'recipe_digest':os.environ.get('MOJOLEARN_AFCL_RECIPE_SHA256'),
               'gates':{name:{'status':'PENDING'} for name in GATES}}
    destination = output / 'result.json'
    try:
        selected = summary['ideas']
        entries = lane_entries()
        if not selected or any(idea not in entries for idea in selected):
            raise ValueError('Select existing AFCL classical cards')
        if 'AFCL-G01' in selected and 'AFCL-G02' in selected:
            raise ValueError('G02 disables the MMA route G01 needs')
        build_path = Path(args.arms).resolve() / 'paired-build.json'
        build = read_json(build_path)
        summary.update(paired_build_path=str(build_path),paired_build_sha256=sha256(build_path),
            artifact_hashes={name:{key:value['sha256'] for key,value in arm.get('artifacts',{}).items()}
                             for name,arm in build.get('arms',{}).items()},
            build_provenance={'source_sha':build.get('source_sha'),'arms':build.get('arms',{})})
        if build.get('source_sha') != args.source_sha or build.get('vendor') != 'apple' or build.get('numeric_mode') != 'fast':
            raise ValueError('Paired build source/vendor/numeric mode does not match this frozen run')
        if build.get('status') != 'build_complete_unverified_unmeasured':
            raise ValueError('Both frozen packages must have complete build receipts before timing')
        if set(build.get('ideas', [])) != set(selected):
            raise ValueError('The selected cards must exactly match the paired packages')
        if not all(arm in build.get('arms', {}) for arm in ('baseline','candidate')):
            raise ValueError('This runner needs the baseline/candidate pair; factorial arms need their own pair selection')
        audit_path = Path(args.workloads).resolve()
        audited = read_json(audit_path)
        summary.update(workloads_path=str(audit_path),workloads_sha256=sha256(audit_path))
        recipe_digests=audited.get('recipe_digests',{})
        if len(selected)==1 and summary['recipe_digest']:
            recipe_digests=dict(recipe_digests,**{selected[0]:summary['recipe_digest']})
        summary['recipe_digests']={idea:recipe_digests.get(idea) for idea in selected}
        if audited.get('schema') != 1 or audited.get('status') != 'audited_full_workloads':
            raise ValueError('Supply an audited_full_workloads config; full_workloads.json is a pending catalog, not an execution audit')
        if audited.get('source_sha') != args.source_sha:
            raise ValueError('Workload settings audit belongs to another frozen source')
        catalog = read_json(CATALOG)
        by_id = {entry['id']:entry for entry in catalog['entries']}
        recipes = catalog['recipes']
        allowed = set().union(*(set(by_id[idea]['recipes']) for idea in selected))
        workloads = []
        for workload in audited.get('workloads', []):
            if not set(workload.get('ideas', [])).intersection(selected):
                continue
            if workload['recipe'] not in allowed and workload['recipe']!='public/full-classical-operation':
                raise ValueError('Workload is not a mapped classical recipe for the selected cards')
            if not workload.get('id') or Path(workload['id']).name != workload['id'] or workload['id'] in ('.','..'):
                raise ValueError('Workload id must be a single safe directory name')
            workloads.append(audit_workload(workload, audit_path.parent, recipes[workload['recipe']]))
        if not workloads:
            raise ValueError('No audited full workload was supplied; missing recipes remain pending')
        if len({workload['id'] for workload in workloads})!=len(workloads):
            raise ValueError('Workload ids must be unique')
        all_env_keys = set()
        for entry in entries.values():
            all_env_keys.update(entry.get('baseline_env', {}))
            all_env_keys.update(entry.get('candidate_env', {}))
        shared = audited.get('shared_environment', {})
        resource_caps={'MOJOLEARN_BENCH_THREADS','OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS',
                       'NUMEXPR_NUM_THREADS','VECLIB_MAXIMUM_THREADS','BLIS_NUM_THREADS','MOJOLEARN_ALGOS_SMOKE_ROWS'}
        if all_env_keys.intersection(shared) or any(k.startswith('MOJOLEARN_AFCL_') for k in shared):
            raise ValueError('Per-card controls belong to paired-build arm environments, not shared_environment')
        if resource_caps.intersection(shared):
            raise ValueError('Dedicated Apple workload environments must leave CPU libraries unrestricted')
        packages = {}
        for name in ('baseline','candidate'):
            arm = build['arms'][name]
            wanted_defines = set().union(*(set(entries[idea][name+'_defines']) for idea in selected))
            if set(arm['defines']) != wanted_defines:
                raise ValueError('Paired arm defines differ from the selected source controls: ' + name)
            wanted_environment={}
            for idea in selected:
                for key,value in entries[idea].get(name+'_env',{}).items():
                    if key in wanted_environment and wanted_environment[key]!=str(value):
                        raise ValueError('Conflicting runtime prerequisites: '+key)
                    wanted_environment[key]=str(value)
            if {key:str(value) for key,value in arm.get('runtime_environment',{}).items()}!=wanted_environment:
                raise ValueError('Paired runtime controls differ from exact arm prerequisites: '+name)
            packages[name] = resolve_package(build_path.parent, arm)
        summary.update(
            source_catalog_sha256=sha256(CATALOG), status='running',
            unresolved_catalog={idea:by_id[idea]['pending_callers_or_settings'] for idea in selected},
            missing_recipe_coverage=sorted(allowed-{workload['recipe'] for workload in workloads}))
        summary['machine']=require_owned_apple()
        proofs=load_external_proofs(audited,audit_path.parent,summary)
        require_prior_quality(args,audited,audit_path.parent,summary)
        write_json(destination,summary)
        # Serial process lifetimes ensure only one arm uses the owned GPU.
        for workload in workloads:
            item={'id':workload['id'],'recipe':workload['recipe'],'arms':{},'status':'running',
                  'ideas':sorted(set(workload['ideas']).intersection(selected)),
                  'dataset_identity':workload['dataset_identity'],'dataset_kind':workload['dataset_kind'],
                  'full_dataset_audit':workload['full_dataset_audit'],
                  'source_sha':args.source_sha,'machine':summary['machine'],
                  'artifact_hashes':summary['artifact_hashes']}
            summary['workloads'].append(item)
            for name in ('baseline','candidate'):
                directory=output / workload['id'] / name
                directory.mkdir(parents=True,exist_ok=False)
                task={'workload':workload,'source_sha':args.source_sha,'package_root':str(packages[name])}
                task_path=directory/'task.json'
                worker_output=directory/'worker.json'
                write_json(task_path,task)
                environment=arm_environment(build['arms'][name],packages[name],entries,shared)
                command=[args.python,str(Path(__file__).resolve()),'--worker',str(task_path),'--worker-output',str(worker_output)]
                started=time.monotonic()
                with (directory/'worker.log').open('wb') as log:
                    try:
                        process=subprocess.run(command,cwd=REPO,env=environment,stdout=log,stderr=subprocess.STDOUT,
                                               timeout=args.timeout,check=False)
                        code=process.returncode
                    except subprocess.TimeoutExpired:
                        code=124
                item['arms'][name]={'exit_code':code,'wall_seconds':time.monotonic()-started,
                    'log':str(directory/'worker.log'),'result':str(worker_output),
                    'result_sha256':sha256(worker_output) if worker_output.exists() else None,
                    'data':read_json(worker_output) if worker_output.exists() else {'status':'missing_worker_receipt'}}
                write_json(destination,summary)
                if code or item['arms'][name]['data'].get('status')!='completed_unqualified':
                    raise RuntimeError('Failed/incomplete workload '+workload['id']+' arm '+name+'; retained original evidence')
            a=item['arms']['baseline']['data'];b=item['arms']['candidate']['data']
            if a['actual_arrays'] != b['actual_arrays'] or a['rounds'][1]['params'] != b['rounds'][1]['params']:
                raise ValueError('A/B dataset/settings differ in '+workload['id'])
            q=quality_gates(workload,a,b)
            item.update(status='measured_unqualified',quality=q,
                timing_ratio_b_over_a=b['rounds'][1]['whole_operation_ms']/a['rounds'][1]['whole_operation_ms'],
                gates={'full_dataset_coverage':{'status':'PASS','scope':'audited workload only'},
                    'matched_full_operation':{'status':'PASS'},'task_quality_no_regression':q,
                    'actual_candidate_route':external_gate(proofs,'actual_candidate_route',item['ideas'],item['id']),
                    'api_contract':external_gate(proofs,'api_contract',item['ideas'],item['id'])})
            if all(gate['status']=='PASS' for gate in item['gates'].values()):
                item['status']='qualified_workload'
            write_json(destination,summary)
        # No blanket PASS: missing/transitive workload coverage and execution
        # route/API witnesses cannot be inferred from a successful process.
        finish_receipts(output,summary,by_id,proofs)
        print('AFCL evidence retained at '+str(destination)+'; '+summary['status'])
        return 0
    except BaseException as exc:
        summary.update(status='failed_or_pending',error=repr(exc),promotion_status='BLOCKED')
        write_quality_receipts(output,summary)
        write_json(destination,summary)
        print('AFCL did not complete: '+str(exc)+'; evidence '+str(destination),file=sys.stderr)
        return 1


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--idea',action='append',default=[])
    parser.add_argument('--arms')
    parser.add_argument('--output')
    parser.add_argument('--source-sha')
    parser.add_argument('--workloads',help='Explicit audited full-workload JSON; the catalog itself is not an audit')
    parser.add_argument('--stage',choices=('validate','time','run'),default='run')
    parser.add_argument('--attach-evidence-to',help='Retained result directory; attach same-freeze accepted proofs without device execution to a fresh --output')
    parser.add_argument('--python',default=sys.executable)
    parser.add_argument('--timeout',type=float,default=86400)
    parser.add_argument('--worker',help=argparse.SUPPRESS)
    parser.add_argument('--worker-output',help=argparse.SUPPRESS)
    args=parser.parse_args()
    if args.attach_evidence_to:
        if not args.output or not args.workloads:
            parser.error('Evidence attachment requires --output and the unchanged --workloads config')
        return attach_evidence(args)
    if args.worker:
        if not args.worker_output:
            parser.error('--worker-output is required for an internal worker')
        return worker(args.worker,args.worker_output)
    if not all((args.idea,args.arms,args.output,args.source_sha,args.workloads)):
        parser.error('--idea --arms --output --source-sha --workloads are required')
    return run(args)


if __name__=='__main__':
    raise SystemExit(main())
