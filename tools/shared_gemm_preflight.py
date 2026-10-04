"""Future-case native prerequisites. Metadata/capability checks, no model math."""
import hashlib
import json
from pathlib import Path


class InfrastructureError(RuntimeError):
    pass


DIAGNOSTICS = ('shared_gemm_count','shared_gemm_reset','shared_gemm_variant',
               'shared_gemm_mode','shared_gemm_vendor')
CASE_REQUIREMENTS = {
    'kmeans': ('core', ('kmeans_fit','kmeans_predict','kmeans_transform'), ('ibase',)),
    'knn': ('core', ('knn_search','knn_search_resident','knn_index_prepare','knn_index_release'), ()),
    'knn-wide-k': ('core', ('knn_search','knn_search_resident','knn_index_prepare','knn_index_release'), ()),
    'ols': ('estimators', ('ols_fit_resident','ols_normal_eq_default','ols_predict'), ()),
    'ridge': ('estimators', ('ridge_fit_resident','ridge_resident_default','ols_predict'), ()),
    'pca': ('estimators', ('pca_fit_full','pca_transform','inverse_transform'), ()),
    'kde': ('estimators', ('kde_fit_prepare','kde_fit_release','kde_score_samples_resident'), ()),
}
ARTIFACTS = {'core':'_mojolearn.so','estimators':'_mojolearn_estimators.so'}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def prerequisite_plan(case, source, verified_root):
    """Read/hash only; no imports or installed-package guesses."""
    if case not in CASE_REQUIREMENTS:
        raise InfrastructureError('case has no reviewed native prerequisite declaration: '+case)
    plan = {}
    for name in CASE_REQUIREMENTS[case][2]:
        directory = Path(verified_root)/source/name
        manifest_path = directory/'manifest.json'
        try:
            m = json.loads(manifest_path.read_text())
            required = dict(contract='shared-gemm-prerequisite-v1',source_sha=source,binding='ibase',
                module='_mojolearn',artifact='_mojolearn.so',install_path='identical/_mojolearn.so',
                numeric_mode='identical',defines='',builder='m2',compile_only=True,
                target_cpu='apple-m1',target_accelerator='metal:1',target_column='apple',required_exports=['all_finite_f32'])
            for key, expected in required.items():
                if m.get(key) != expected:
                    raise ValueError('manifest mismatch: '+key)
            binary = directory/m['artifact']
            if sha(binary) != m['sha256']:
                raise ValueError('artifact SHA mismatch')
            plan[name] = dict(manifest=m, manifest_sha=sha(manifest_path), artifact_path=str(binary))
        except (OSError, ValueError, KeyError, TypeError) as exc:
            raise InfrastructureError(f'{case} requires verified same-source {name}: {manifest_path}: {exc}') from exc
    return plan


def check_loaded(root, case, source, binary_sha, variant, plan):
    """Check actual loaded modules before fixture creation or estimator calls.

    Calls only native mode/vendor/variant metadata getters. Resolving the
    existing _buffer helper is checked without executing a finiteness scan.
    """
    try:
        from mojolearn import _backend
        family, capabilities, dependencies = CASE_REQUIREMENTS[case]
        artifact = ARTIFACTS[family]
        name = artifact[:-3]
        module = _backend.binding(name, 'fast')
        path = Path(module.__file__).resolve()
        expected = (Path(root)/'python/mojolearn'/artifact).resolve()
        if path != expected or sha(path) != binary_sha:
            raise ValueError('actual target path/hash mismatch')
        for capability in DIAGNOSTICS+capabilities:
            if not callable(getattr(module, capability, None)):
                raise ValueError('missing target capability '+name+'.'+capability)
        if module.shared_gemm_mode() != 0 or module.shared_gemm_vendor() != 'metal' or module.shared_gemm_variant() != variant:
            raise ValueError('target mode/vendor/variant mismatch')
        if set(plan) != set(dependencies):
            raise ValueError('prerequisite plan differs from reviewed case requirements')
        observed = dict(primary=dict(module=name,path=str(path),sha256=sha(path),capabilities=list(capabilities)),dependencies={})
        for dependency in dependencies:
            entry = plan[dependency]
            m = entry['manifest']
            if m['source_sha'] != source:
                raise ValueError('prerequisite compiled source mismatch')
            aux = _backend.binding(m['module'], m['numeric_mode'])
            aux_path = Path(aux.__file__).resolve()
            expected_path = (Path(root)/'python/mojolearn'/m['install_path']).resolve()
            if aux_path != expected_path or sha(aux_path) != m['sha256']:
                raise ValueError('actual prerequisite path/hash mismatch')
            if aux.mojolearn_numeric_mode() != 1 or aux.mojolearn_vendor() != 'metal':
                raise ValueError('prerequisite mode/vendor mismatch')
            for capability in m['required_exports']:
                if not callable(getattr(aux, capability, None)):
                    raise ValueError('missing prerequisite capability '+capability)
            from mojolearn import _buffer
            if _buffer._native('all_finite_f32') is not aux.all_finite_f32:
                raise ValueError('buffer helper resolved outside verified identical base')
            observed['dependencies'][dependency] = dict(path=str(aux_path),sha256=sha(aux_path),
                numeric_mode=m['numeric_mode'],manifest_sha=entry['manifest_sha'],capabilities=m['required_exports'])
        return observed
    except InfrastructureError:
        raise
    except Exception as exc:
        raise InfrastructureError(f'{case} native preflight failed before numerical case: {exc}') from exc
