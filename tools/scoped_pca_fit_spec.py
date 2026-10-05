#!/usr/bin/env python3
"""Generate metadata spec from clean harness, without native or cloud work."""
import argparse
import json
import os
from scoped_pca_fit import A_FLAGS, B_FLAGS, ROOT, source_contract
from apple_fast_job_policy import policy_for


def make_spec(harness, source, tag, action, data, data_sha, quality=None, quality_sha=None):
    args = [source, tag, action, data, data_sha]
    prerequisites = [dict(id='board-data', kind='file', path=data, sha256=data_sha)]
    requires = ['estimators', 'board-data']
    if action == 'timing':
        args += ['--quality-report', quality, '--quality-sha', quality_sha]
        prerequisites.append(dict(id='caller-quality', kind='file', path=quality, sha256=quality_sha,
                                  json_equals=dict(status='PASS', scored=False, source_sha=source,
                                                   harness_source=harness, data_sha=data_sha)))
        requires.append('caller-quality')
    policy_for(harness, 'tools/scoped_pca_fit.py', args)
    return dict(version=1, tag=tag, harness_source=harness, script='tools/scoped_pca_fit.py', args=args,
                artifacts=[dict(id='estimators', kind='pair', compiled_source=source, binding='estimators',
                                numeric_mode='fast', defines_A=A_FLAGS, defines_B=B_FLAGS)],
                prerequisites=prerequisites,
                cases=[dict(name='public-pca-fit-full-istella', requires=requires,
                            source_files=['bindings/_mojolearn_estimators.mojo',
                                          'python/mojolearn/decomposition.py',
                                          'decomposition/estimator.mojo',
                                          'decomposition/impl/linalg/detail/pca.mojo',
                                          'experiments/apple_fast/gemm/scoped_dispatch.mojo'])])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('source'); p.add_argument('tag'); p.add_argument('action', choices=('quality', 'timing'))
    p.add_argument('data'); p.add_argument('data_sha')
    p.add_argument('--quality-report'); p.add_argument('--quality-sha')
    a = p.parse_args()
    os.chdir(ROOT)
    harness = source_contract(a.source)
    print(json.dumps(make_spec(harness, a.source, a.tag, a.action, a.data, a.data_sha,
                               a.quality_report, a.quality_sha), indent=2))


if __name__ == '__main__':
    main()
