#!/usr/bin/env python3
"""M3 serial quality-only installer. SOURCE TAG CASE [--features 11|65|220].
No build fallback, timing, remote operations, queue edits, or scored repeats.
"""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile

if not __debug__:
    raise RuntimeError('quality gates require Python assertions enabled')

FAMILY = {"kmeans": "core", "knn": "core", "knn-wide-k": "core", "ols": "estimators",
          "ridge": "estimators", "pca": "estimators", "kde": "estimators"}
ARTIFACT = {"core": "_mojolearn.so", "estimators": "_mojolearn_estimators.so"}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def record(path, data):
    with Path(path).open('x') as stream:
        json.dump(data, stream, indent=2, allow_nan=False)
        stream.write('\n')


def atomic_install(src, dst):
    fd, temp = tempfile.mkstemp(prefix='.shared-gemm-arm-', suffix='.so', dir=dst.parent)
    os.close(fd)
    try:
        shutil.copy2(src, temp)
        os.replace(temp, dst)
    finally:
        Path(temp).unlink(missing_ok=True)


def run_logged(command, log, env):
    with log.open('x') as stream:
        result = subprocess.run(command, env=env, stdout=stream, stderr=subprocess.STDOUT)
    if result.returncode:
        print('\n'.join(log.read_text(errors='replace').splitlines()[-20:]), flush=True)
    return result.returncode


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source'); parser.add_argument('tag')
    parser.add_argument('case', choices=tuple(FAMILY))
    parser.add_argument('--features', type=int, choices=(11, 65, 220), default=65)
    args = parser.parse_args()
    assert re.fullmatch('[0-9a-f]{40}', args.source)
    assert re.fullmatch('[A-Za-z0-9_.-]+', args.tag)
    root = Path(__file__).resolve().parents[1]
    os.chdir(root)
    from shared_gemm_source_contract import validate_source
    provenance = validate_source(root, args.source)
    config_path = root/'tools/shared_gemm_variant.json'
    config = json.loads(config_path.read_text())
    variant = config['variant']
    assert config['contract'] == 'shared-gemm-static-downstream-v1' and variant in (1, 5)
    family = FAMILY[args.case]
    artifact = ARTIFACT[family]
    arms = Path.home()/'mq/verified-arms'/args.source/family
    manifest_path = arms/'manifest.json'
    manifest = json.loads(manifest_path.read_text())
    for key in ('contract', 'base_source', 'variant', 'families', 'defines_A', 'defines_B'):
        assert manifest[key] == config[key], 'manifest/variant identity mismatch: '+key
    assert manifest['source_sha'] == args.source and manifest['binding'] == family
    assert manifest['artifact'] == artifact and manifest['builder'] == 'm2'
    assert manifest['numeric_mode'] == 'fast' and manifest['compile_only'] is True
    assert manifest['target_accelerator'] == 'metal:1' and manifest['target_column'] == 'apple'
    assert manifest['contract_sha'] == sha(config_path)
    assert config['defines_A'].split() == ['-D', 'MOJOLEARN_APPLE_FAST_SHARED_GEMM_COUNTERS']
    assert config['defines_B'].split() == config['defines_A'].split()+['-D', f'MOJOLEARN_APPLE_FAST_SHARED_GEMM_G{variant}']
    hashes = {arm: sha(arms/(arm+'.so')) for arm in ('A', 'B')}
    assert manifest['hashes'] == hashes
    # Additional guard against accidental parallel pair installers. The M3
    # manager must still serialize this with every other GPU job and transfer.
    lock = (Path.home()/'mq/shared-gemm-quality.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    out = Path.home()/'mq/out'/(args.tag+'-quality')
    out.mkdir(parents=True, exist_ok=False)
    shutil.copy2(manifest_path, out/'manifest.json')
    so = root/'python/mojolearn'/artifact
    assert not so.is_symlink(), 'refusing to replace a symlinked installed binding'
    original = out/'original.so'
    had_original = so.exists()
    original_sha = sha(so) if had_original else None
    if had_original:
        shutil.copy2(so, original)
        assert sha(original) == original_sha
    record(out/'intake.json', dict(source=args.source,provenance=provenance,case=args.case,features=args.features,
        variant=variant,family=family,artifact=artifact,hashes=hashes,manifest_sha=sha(manifest_path),
        original_sha=original_sha,scored=False))
    env = dict(os.environ, MOJOLEARN_NUMERIC_MODE='fast', MOJOLEARN_VENDOR='apple',
        MOJOLEARN_BENCH_INSTALLED='0', PYTHONPATH=str(root/'python'), OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1')
    env.pop('PYTHONOPTIMIZE', None)
    helper = str(root/'tools/shared_gemm_downstream_quality.py')
    success = False
    try:
        for arm, selected in (('A', 0), ('B', variant)):
            atomic_install(arms/(arm+'.so'), so)
            assert sha(so) == hashes[arm]
            command = [sys.executable, helper, 'dump', args.source, hashes[arm], str(selected),
                       args.case, str(out/(arm+'.npz')), '--features', str(args.features)]
            assert run_logged(command, out/(arm+'.log'), env) == 0, 'capture failed: '+arm
            meta = json.loads((out/(arm+'.json')).read_text())
            assert meta['source'] == args.source and meta['binary_sha'] == hashes[arm]
            assert meta['provenance'] == provenance
            assert meta['variant'] == selected and meta['case'] == args.case and meta['features'] == args.features
            assert Path(meta['binary']) == so.resolve(), 'capture loaded another package path'
        rc = run_logged([sys.executable, helper, 'compare', str(out/'A.npz'), str(out/'B.npz'),
                         str(out/'report.json')], out/'compare.log', env)
        report = json.loads((out/'report.json').read_text()) if (out/'report.json').exists() else {}
        success = rc == 0 and report.get('status') == 'PASS'
    finally:
        if had_original:
            atomic_install(original, so)
            assert sha(so) == original_sha, 'original restore hash mismatch'
        else:
            so.unlink(missing_ok=True)
        record(out/'restore.json', dict(restored=True,had_original=had_original,original_sha=original_sha))
    # No PASS receipt until restoration succeeds. HOLD and NO_REACH are rc1.
    receipt = dict(source_sha=args.source,provenance=provenance,variant=variant,case=args.case,features=args.features,
        family=family,hashes=hashes,manifest_sha=sha(manifest_path),scored=False,
        status=report.get('status', 'ERROR'),report_sha=sha(out/'report.json') if (out/'report.json').exists() else None)
    record(out/('PASS.json' if success else 'HOLD.json'), receipt)
    print('SHARED-GEMM-DOWNSTREAM '+json.dumps(receipt, sort_keys=True), flush=True)
    return 0 if success else 1


if __name__ == '__main__':
    sys.exit(main())
