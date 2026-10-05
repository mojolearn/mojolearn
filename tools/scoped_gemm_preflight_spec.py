#!/usr/bin/env python3
"""Print a scoped adapter quality metadata spec. No cloud/native/queue activity.

Run from the clean final harness checkout: COMPILED_SOURCE TAG PROFILE.
The manager stages the JSON and runs apple_fast_job_preflight.py on M3.
"""
import argparse
import json
import os
from scoped_gemm_quality import AUDIT, BINDING, CASES, FLAGS, PCA_CASES, PROFILES, ROOT, verify_source
from apple_fast_job_policy import policy_for


def make_spec(harness, compiled, tag, profile):
    args = [compiled, tag, profile]
    script = 'tools/scoped_gemm_quality.py'
    policy_for(harness, script, args)
    mask = PROFILES[profile]
    a = '-D ' + AUDIT
    b = a + ''.join(' -D ' + flag for i, flag in enumerate(FLAGS) if mask & (1 << i))
    files = ['bindings/_mojolearn_scoped_gemm_probe.mojo',
             'experiments/apple_fast/gemm/scoped_dispatch.mojo',
             'x_decomp/device.mojo', 'decomposition/impl/linalg/detail/pca.mojo']
    return dict(version=1, tag=tag, harness_source=harness, script=script, args=args,
                artifacts=[dict(id='probe', kind='pair', compiled_source=compiled,
                                binding=BINDING, numeric_mode='fast', defines_A=a, defines_B=b)],
                prerequisites=[],
                cases=[dict(name=c[0], requires=['probe'], source_files=files)
                       for c in CASES + PCA_CASES])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('compiled_source')
    parser.add_argument('tag')
    parser.add_argument('profile', choices=PROFILES)
    args = parser.parse_args()
    os.chdir(ROOT)
    harness = verify_source(args.compiled_source)
    print(json.dumps(make_spec(harness, args.compiled_source, args.tag, args.profile), indent=2))


if __name__ == '__main__':
    main()
