# SPDX-License-Identifier: Apache-2.0
"""Testing-only reproducer; each selected case runs in a fresh interpreter."""
import argparse
import importlib.util
from pathlib import Path
import sys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--path', choices=('original', 'direct', 'cpython'), default='original')
    parser.add_argument('--case', choices=('instance', 'class', 'descriptor', 'missing'), required=True)
    args = parser.parse_args()
    # Import just the canonical connection declarations. Importing the product
    # would require the complete build and defeat this reduced test's purpose.
    name = 'mojolearn._native_loaded_modules'
    spec = importlib.util.spec_from_file_location(name, args.root / 'python/mojolearn/_native_loaded_modules.py')
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    import _mojo_item_lookup
    class Probe:
        pass
    model = Probe()
    token = object()
    if args.case == 'instance':
        model.value = token
    elif args.case == 'class':
        Probe.value = token
    elif args.case == 'descriptor':
        def forbidden(self):
            raise AssertionError('Python property must not execute')
        Probe.value = property(forbidden)
    print('begin', args.path, args.case, flush=True)
    try:
        if args.path == 'original':
            result = _mojo_item_lookup.lookup(model, vars(model), 'value', {'missing': object()}, 'TypeError')
        else:
            assert args.case in ('instance', 'missing'), 'dictionary controls only accept instance/missing cases'
            entry = getattr(_mojo_item_lookup, args.path + '_lookup')
            result = entry(vars(model), 'value')
    except Exception as error:
        if args.case not in ('descriptor', 'missing'):
            raise
        assert ('constructor parameter' in str(error) or 'Python properties' in str(error)
                or 'missing native dictionary key' in str(error) or isinstance(error, KeyError)), repr(error)
    else:
        assert args.case in ('instance', 'class')
        assert result is token
    print('passed', args.path, args.case, flush=True)


if __name__ == '__main__':
    main()
