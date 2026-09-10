#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Root host-only check of a freshly compiled binding. Never submits valid buffers."""
import argparse
import importlib.util
from pathlib import Path
from mojolearn import ByteLanguageModelConfig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('binding', type=Path)
    args = parser.parse_args()
    spec = importlib.util.spec_from_file_location('_mojolearn_byte_lm', args.binding)
    native = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(native)
    assert native.byte_lm_numeric_mode() == 1
    cases = [(2, 32, 32, 4, 2, 8, 64), (3, 7, 24, 3, 1, 8, 40),
             (1, 65, 48, 6, 3, 8, 96), (1, 256, 512, 8, 4, 64, 1024)]
    for fields in cases:
        cfg = ByteLanguageModelConfig(*fields)
        assert native.byte_lm_config_profile(list(fields)) == cfg.profile
    assert native.byte_lm_profile() == ByteLanguageModelConfig().profile
    rejected = 0
    for fields in ([0, 32, 32, 4, 2, 8, 64], [2, 8193, 32, 4, 2, 8, 64],
                   [1, 8192, 64, 32, 1, 2, 2], [1, 1, 1048576, 1, 1, 1048576, 2], [1], [True, 32, 32, 4, 2, 8, 64], [1.5, 32, 32, 4, 2, 8, 64]):
        try:
            native.byte_lm_config_profile(fields)
        except Exception:
            rejected += 1
        else:
            raise AssertionError('invalid native shape accepted')
    params = [0, 0, 2, .001, .9, .999, 1e-8, .01, 0., 0., 0, 0.]
    for configured in (False, True):
        try:
            if configured:
                native.byte_lm_run_configured([0] * 11, params, list(cases[1]))
            else:
                native.byte_lm_run([0] * 11, params)
        except Exception as exc:
            assert 'span' in str(exc), str(exc)
        else:
            raise AssertionError('null addresses accepted')
    print(f'PASS native host-only profile agreement: {len(cases)} profiles, {rejected} invalid shapes, both ABIs reject null spans')
    print(f'vendor={native.byte_lm_vendor()} numeric_mode={native.byte_lm_numeric_mode()}')
    print('No DeviceContext or model operation requested; numerical qualification remains owed.')


if __name__ == '__main__':
    main()
