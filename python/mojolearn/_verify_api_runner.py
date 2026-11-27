"""Shared endpoint registry and exact comparison for the ordinary verifier."""
import hashlib
import json
import tempfile
from pathlib import Path
from ._verification_api_targets import TARGETS
from ._verify_api import Recorder


def registry():
    from . import (_verify_api_metadata as metadata, _verify_api_estimators as estimators,
        _verify_api_utilities as utilities, _verify_api_optimizers as optimizers,
        _verify_api_neural as neural, _verify_api_training as training,
        _verify_api_programs as programs, _verify_api_models as models, _verify_api_misc as misc)
    families=(('utilities',utilities),('optimizers',optimizers),('neural',neural),
              ('training',training),('programs',programs),('models',models),('estimators',estimators),('misc',misc))
    routes={}
    for row in TARGETS:
        public,method=row['public'],row['op']
        if method in metadata.PARAMETER_METHODS:
            routes[row['endpoint']]=('metadata',metadata,public,method);continue
        for family,module in families:
            if module.applicable(public,[method] if family=='estimators' else method):
                routes[row['endpoint']]=(family,module,public,method);break
    return routes


def selected_endpoints(scope='all'):
    routes=registry()
    if scope=='all':return sorted(row['endpoint'] for row in TARGETS)
    if scope!='neural':raise ValueError('unknown public verification scope: '+scope)
    neural_families={'optimizers','neural','training','programs','models'}
    shared_names={'Array','BpeTokenizer','GPT2Tokenizer','TrainedBpeVocabulary'}
    public_names={public for family,_,public,_ in routes.values()
        if family in neural_families or '.training.' in public
        or public.rsplit('.',1)[-1] in shared_names
        or public.rsplit('.',1)[-1].startswith(('RNN','LSTM','GRU','MLP'))}
    # Include each selected class's parameter/attribute contracts as well as
    # its numerical doors. Classical estimator metadata stays out of this run.
    return sorted(endpoint for endpoint,(_,_,public,_) in routes.items() if public in public_names)


def sources():
    # Per-family closures allow unchanged groups to retain their evidence.
    root=Path(__file__).parent
    common=['_verify_api.py','_verify_api_metadata.py']
    result={}
    for family in ('metadata','estimators','utilities','optimizers','neural','training','programs','models','misc'):
        files=set(common+['_verify_api_'+family+'.py'])
        if family=='estimators':files.add('_verify_api_availability.py')
        if family=='misc':files.add('_verify_api_estimators.py')
        result[family]=hashlib.sha256(b''.join(name.encode()+b'\0'+(root/name).read_bytes() for name in sorted(files))).hexdigest()
    return result


#: Endpoints that exist only on a build with a device: the optimizers' device-resident
#: parameter I/O, Embedding's device upload and the byte-LM trainer run on one GPU
#: (devices=(0,); every method: its sessions live inside the GPU byte-LM binding). The product declares
#: a build without a device through `runtime_report()['cpu_only']`; there these endpoints
#: are recorded 'not applicable' and not run. On every device build they run and are judged
#: as before, and `admit` still requires the canonical NVIDIA capture to execute all of them.
OPTIMIZER_DEVICE_METHODS=frozenset(('params_to_device','params_to_host','grads_to_device',
    'grads_to_host','param_device','step_device'))
DEVICE_ONLY_CLASSES=frozenset(('ParallelByteLanguageModelTrainer',))
DEVICE_ONLY_METHODS=frozenset((('Embedding','to_device'),))
#: `devices=` drivers whose workers run on the installation's own set whatever
#: side their member carries (Andrew, 2026-11-24: the parallel drivers keep
#: `devices=`; verify takes their one digest and lists them single-sided).
#: On a CPU-only install that set is the CPU one and the CPU side runs them;
#: beside a GPU set they run on the GPU side and the CPU side records them
#: 'not applicable' (kind 'installation side'), never a GPU digest under the
#: CPU label.
#: gs-fix-7 (2026-11-26): SambaStack has no CPU route beside a GPU set: its blocks'
#: backward has no host training image loaded there, so the product refuses device='cpu'
#: on a GPU installation at construction (bindings/samba_route_unpack.mojo
#: _samba_cpu_refuse). A CPU-only installation runs it on the CPU side and records the
#: CPU column; the CPU side of a GPU installation records it 'not applicable'.
INSTALLATION_SIDE_CLASSES={
    'ParallelCausalLM':"workers follow the installation vendor (bindings/parallel_causal_native_run.mojo); "
                      "its CPU route runs on a CPU-only installation",
    'ParallelNeuralTrainer':"a devices= driver runs on this installation's GPU set; its one digest is the GPU side's",
    'SambaStack':"no CPU route beside a GPU set: its blocks' host training images are not loaded there "
                 "(bindings/samba_route_unpack.mojo); the CPU column is a CPU-only installation's",
}


def device_only(family, public, method):
    # Parameter/attribute contracts (the metadata family) are host contracts on every build.
    if family=='metadata':return False
    name=public.rsplit('.',1)[-1]
    return (name in DEVICE_ONLY_CLASSES or (name,method) in DEVICE_ONLY_METHODS
            or (family=='optimizers' and method in OPTIMIZER_DEVICE_METHODS))


def sides():
    """The sides this installation captures, CPU first: ['cpu'] on a build
    that declares no device (runtime_report cpu_only), else ['cpu', 'gpu'].
    Read once per run, before any endpoint runs."""
    from . import runtime_report
    return ['cpu'] if runtime_report().get('cpu_only') else ['cpu','gpu']


def capture(path=None, endpoints=None, *, artifacts=None, side=None):
    """One side's capture. `side` ("cpu" or "gpu") binds every resolved public
    class and function to device=`side` in this process (`_verify_device`);
    the CPU side records the device-only endpoints not applicable, as an
    installation without a device does. None is the installation's own side."""
    from . import __version__,_verify,runtime_report
    from . import _verify_device as vd
    routes=registry();expected={r['endpoint'] for r in TARGETS}
    selected=expected if endpoints is None else set(endpoints)
    if not selected or selected-expected:raise ValueError('unknown or empty endpoint selection')
    # Read once per capture, before any endpoint runs: the build's own declaration.
    cpu_only=bool(runtime_report().get('cpu_only'))
    if side=='gpu' and cpu_only:raise ValueError('no GPU side on this installation')
    no_device=cpu_only or side=='cpu'
    temporary=tempfile.TemporaryDirectory() if path is None else None
    scope=vd.side(side) if side else None
    if scope:scope.__enter__()
    try:
        recorder=Recorder(Path(temporary.name)/'api.json' if temporary else path,selected)
        groups={}
        for endpoint in sorted(selected & routes.keys()):
            family,module,public,method=routes[endpoint]
            if no_device and device_only(family,public,method):
                recorder.rows[endpoint]=dict(status='not applicable',kind='device only',
                    reason='the CPU side (device="cpu")' if side=='cpu' and not cpu_only
                    else 'this build declares no device (runtime_report cpu_only)')
                continue
            if side=='cpu' and not cpu_only and public.rsplit('.',1)[-1] in INSTALLATION_SIDE_CLASSES:
                recorder.rows[endpoint]=dict(status='not applicable',kind='installation side',
                    reason=INSTALLATION_SIDE_CLASSES[public.rsplit('.',1)[-1]])
                continue
            groups.setdefault((family,public),(module,[]))[1].append(method)
        for (family,public),(module,methods) in groups.items():
            if family=='metadata':module.parameter_checks(recorder,public)
            elif family=='utilities':module.utility_checks(recorder,[r for r in TARGETS if r['endpoint'] in selected and r['public']==public and routes[r['endpoint']][0]==family])
            else:
                name={'estimators':'estimator','optimizers':'optimizer','neural':'neural','training':'training',
                      'programs':'program','models':'model','misc':'misc'}[family]+'_checks'
                getattr(module,name)(recorder,public,methods)
        _runtime=runtime_report()
        report=dict(format='mojolearn.api-endpoints.v1',version=__version__,
            vendor='cpu' if side=='cpu' else _runtime['vendor'],side=side,
            numeric_mode=_runtime['default_mode'],expected=sorted(selected),endpoints=recorder.rows,
            pending=sorted(selected-recorder.rows.keys()),sources=sources(),
            families={e:routes[e][0] for e in selected if e in routes},bindings=_verify.binding_artifacts() if artifacts is None else artifacts)
        recorder.path.write_text(json.dumps(report,sort_keys=True,indent=1)+'\n')
        return report
    finally:
        if scope:scope.__exit__(None,None,None)
        if temporary:temporary.cleanup()


def output_contract(value):
    if isinstance(value,dict):
        if {'shape','dtype','bytes','sha256'} <= value.keys():
            raw=bytes.fromhex(value['bytes'])
            if hashlib.sha256(raw).hexdigest()!=value['sha256']:
                raise ValueError('output bytes do not match their digest')
            import numpy as np
            expected=np.prod(value['shape'],dtype=np.int64)*np.dtype(value['dtype']).itemsize
            if len(raw)!=expected:raise ValueError('output byte length does not match shape/dtype')
            return ('array',value['shape'],value['dtype'])
        return {k:output_contract(v) for k,v in value.items() if k not in ('sha256',)}
    if isinstance(value,list):return [output_contract(v) for v in value]
    return type(value).__name__


def differences(actual,expected):
    errors=[]
    if set(actual.get('expected',()))!=set(expected.get('expected',())):errors.append('endpoint scope differs')
    if actual.get('pending'):errors.append('unexecuted endpoints: '+str(len(actual['pending'])))
    for endpoint in actual.get('expected',()):
        a=actual.get('endpoints',{}).get(endpoint,{});b=expected.get('endpoints',{}).get(endpoint,{})
        if a.get('status')!='captured':errors.append('execution failed: '+endpoint);continue
        if b.get('status')!='captured':errors.append('reference missing: '+endpoint);continue
        try:
            if output_contract(a.get('output'))!=output_contract(b.get('output')):
                errors.append('output contract differs: '+endpoint)
        except (ValueError,TypeError,KeyError) as error:
            errors.append('invalid output evidence: '+endpoint+': '+str(error))
        if a.get('kind')!=b.get('kind'):errors.append('contract kind differs: '+endpoint)
        if a.get('output')!=b.get('output'):errors.append('output differs: '+endpoint)
    return errors


#: The columns an endpoint reference can validate: the CPU (host) column and the three GPU vendors.
COLUMNS=('cpu','metal','cuda','hip')


def admit(captures):
    """Build one shared reference; retain each other device's actual coverage.

    Reference availability is not release qualification. A complete NVIDIA
    capture supplies the bytes; partial checks on another device remain
    explicitly partial, and any observed output disagreement rejects admission.
    The CPU column ('cpu': a `capture(side="cpu")`, on a CPU-only installation
    or the CPU side of a GPU one) must reproduce the same bytes on every
    endpoint it executes; the device-only endpoints it records not applicable
    are not missing from it (they have no CPU form), so a CPU capture with no
    other gap validates the CPU column.
    """
    expected={row['endpoint'] for row in TARGETS}
    if 'cuda' not in captures or set(captures)-set(COLUMNS):
        raise ValueError('endpoint reference requires a complete NVIDIA capture')
    canonical=captures['cuda']
    validation={}
    for vendor,report in captures.items():
        if report.get('vendor')!=vendor or report.get('numeric_mode')!='identical':
            raise ValueError('incorrect reference vendor or numeric mode')
        # A CPU column is the CPU side; a GPU column is never a CPU-side capture.
        if (report.get('side')=='cpu')!=(vendor=='cpu'):
            raise ValueError('capture side does not match its column: '+vendor+' / '+str(report.get('side')))
        if not report.get('version') or report['version']!=canonical.get('version'):
            raise ValueError('endpoint references must belong to one release')
        if set(report.get('expected',()))!=expected or set(report.get('endpoints',{}))-expected:
            raise ValueError('incomplete endpoint reference scope')
        if report.get('sources')!=sources() or not report.get('bindings'):
            raise ValueError('stale fixtures or missing native artifact provenance')
        passed=sorted(e for e,row in report.get('endpoints',{}).items() if row.get('status')=='captured')
        not_applicable=sorted(e for e,row in report.get('endpoints',{}).items()
            if vendor=='cpu' and row.get('status')=='not applicable' and row.get('kind') in ('device only','installation side'))
        missing=sorted(expected-set(passed)-set(not_applicable))
        if vendor=='cuda' and (missing or report.get('pending')):
            raise ValueError('canonical reference must execute every endpoint successfully')
        actual=dict(expected=passed,endpoints=report['endpoints'],pending=[])
        reference=dict(expected=passed,endpoints=canonical['endpoints'],pending=[])
        errors=differences(actual,reference)
        if errors:raise ValueError('; '.join(errors))
        validation[vendor]=dict(compared=len(passed),expected=len(expected),
            incomplete=missing, not_applicable=not_applicable,
            failures={e:row.get('error','unexecuted') for e,row in report['endpoints'].items()
                if row.get('status')!='captured' and e not in not_applicable}, pending=report.get('pending',[]))
    complete=sorted(v for v,row in validation.items() if not row['incomplete'] and not row['pending'])
    return dict(format='mojolearn.api-reference.v1',witness=canonical,
        validated_vendors=complete,untested_vendors=sorted(set(COLUMNS)-set(captures)),
        device_validation=validation,release_qualified=False,
        witnesses={v:{k:r[k] for k in ('vendor','version','sources','bindings')} for v,r in captures.items()})


def unchanged_reference_inputs(clean, reference):
    """A release label alone must not invalidate executed identity evidence."""
    prior=reference.get('witnesses',{}).get(clean['vendor'],{})
    def binaries(report):
        return {row.get('module'):row.get('sha256') for row in report.get('bindings',())}
    actual=binaries(clean)
    families=set(clean['families'].values())
    return bool(actual) and actual==binaries(prior) and all(
        clean['sources'][family]==prior.get('sources',{}).get(family) for family in families)


def verify(*,self_test=True,reference=None,retain=None,scope='all'):
    """Both sides in this process (2026-11-24): the CPU side (device="cpu")
    and, on an installation with a GPU, the GPU side (device="gpu"), each
    captured and judged on its own. A side's own failed or unexecuted
    endpoints fail the run; each side is judged against its own column (the
    CPU side against 'cpu', the GPU side against its vendor), and a side whose
    column the reference does not validate reads UNTESTED (its column is owed).
    MATCH requires agreement on every requested side. Each applicable side
    runs its own fault and restoration controls."""
    from . import _verify_device as vd
    from ._verify_public import combined_status
    if reference is None:
        try:
            from ._verification_api_reference import DATA
            reference=DATA
        except ImportError:reference={}
    run_sides=sides()
    results={}
    for side in run_sides:
        clean=capture(endpoints=selected_endpoints(scope),side=side)
        if retain:retain('api-clean-'+side,clean)
        results[side]=_judge_side(clean,reference,scope)
        if self_test:
            from ._verify_api_faults import check_faults
            def retain_side(name, data):
                retain(f"api-{side}-{name}", data)
            with vd.side(side):
                faults = check_faults(clean, retain=retain_side if retain else None)
            results[side]['faults'] = faults
            if any(row['status'] not in ('CAUGHT_AND_RESTORED', 'NOT_APPLICABLE') for row in faults.values()):
                results[side]['errors'].append('endpoint fault controls did not all pass')
                results[side]['status'] = 'FAILED'
    primary=run_sides[-1]
    result=dict(results[primary], failures=dict(results[primary]['failures']))
    result['sides']={side:{k:v for k,v in r.items() if k!='evidence'} for side,r in results.items()}
    result['side']=primary
    for side,r in results.items():
        if side==primary:continue
        result['failures'].update({side+':'+e:m for e,m in r['failures'].items()})
        result['pending']=list(result['pending'])+[side+':'+e for e in r['pending']]
        result['errors']=list(result['errors'])+[side+' side: '+e for e in r['errors'] if r['status']!='UNTESTED']
    result['status'] = ('FAILED' if result['failures'] or result['pending']
                        else combined_status(r['status'] for r in results.values()))
    return result


def _judge_side(clean,reference,scope):
    """One side's capture against the endpoint reference (the judgement the
    single-process verifier made, unchanged), without the fault controls."""
    not_applicable=sorted(e for e,r in clean['endpoints'].items() if r['status']=='not applicable')
    result=dict(format='mojolearn.api-verification.v1',status='OWED',expected=len(clean['expected']),
        executed=len(clean['endpoints'])-len(not_applicable),not_applicable=not_applicable,failures={e:r['error'] for e,r in clean['endpoints'].items() if r['status']=='failed'},
        pending=clean['pending'],errors=[],faults={},vendor=clean['vendor'],side=clean.get('side'),scope='bounded endpoint fixtures',
        evidence=clean, scope_profile=scope, checks_by_kind={kind:sum(row['kind']==kind for row in clean['endpoints'].values())
            for kind in sorted({row['kind'] for row in clean['endpoints'].values()})})
    result['reference_device_validation']=reference.get('device_validation',{})
    result['reference_untested_vendors']=reference.get('untested_vendors',list(COLUMNS))
    witness=reference.get('witness',{})
    # A shipped reference whose fixture sources differ from the shipped fixtures is stale on
    # every device: the staleness is a property of the reference, not of the vendor running it.
    # r6-proving (2026-11-23): this test sat below the UNTESTED return, so the CPU column (the
    # source-tree self-test on the CPU box) passed while the L40S self-test of the same wheel
    # read OWED; the CPU run now fails the same way.
    stale=sorted(family for family in set(clean['families'].values())
                 if witness.get('sources',{}).get(family)!=clean['sources'][family])
    if stale:
        result['errors'].append('endpoint fixtures changed; current reference is stale ('
                                +', '.join(stale)+')')
    # The endpoint reference validates the columns it was admitted with (cpu, cuda, hip,
    # metal). A column it never validated (for example the CPU side of a reference admitted
    # without `--cpu`) is UNTESTED, never FAILED, on its outputs; with a
    # stale reference it reads OWED, as a validated vendor does. Its own execution failures
    # still fail it (2026-11-24: a side that refuses is a failure, never untested).
    if reference.get('validated_vendors') and clean['vendor'] not in reference['validated_vendors']:
        reason='endpoint reference validates '+', '.join(reference['validated_vendors'])+'; not '+clean['vendor']
        if result['failures'] or result['pending']:
            result.update(status='FAILED',untested_reason=reason)
        elif stale:
            result.update(status='OWED',untested_reason=reason)
        else:
            result.update(status='UNTESTED',errors=[],untested_reason=reason)
        return result
    reuse=witness.get('version')!=clean['version'] and unchanged_reference_inputs(clean,reference)
    if reuse:result['evidence_reuse']='unchanged loaded native bytes and numerical fixtures'
    if not stale:
        selected_witness=dict(witness,expected=[e for e in clean['expected'] if e not in not_applicable])
        result['errors'].extend(differences(dict(clean,expected=selected_witness['expected']),selected_witness))
        result['status']='MATCH' if not result['errors'] else 'FAILED'
        if witness.get('version')!=clean['version'] and not reuse:
            # Changed native bytes are judged by their outputs: the admitted witness holds the
            # bits every validated vendor agreed on, so reproducing it on every endpoint is the
            # same cross-vendor evidence. Only endpoints whose outputs moved owe new witnesses.
            if result['errors']:
                result['status']='OWED'
                result['errors'].insert(0,'changed native artifacts moved these endpoint outputs; they need current-release witnesses')
            else:
                result['evidence_reuse']='changed native bytes reproduce the admitted witness on every endpoint'
    if result['failures'] or result['pending']:result['status']='FAILED'
    return result
