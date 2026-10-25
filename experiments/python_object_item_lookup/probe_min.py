# SPDX-License-Identifier: Apache-2.0
"""Test-only independent reduction; imports no mojolearn package or helpers."""
import argparse
import _mojo_item_lookup_min as extension


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', choices=('original_shaped', 'conditional', 'candidate', 'dictionary_candidate'), required=True)
    parser.add_argument('--case', choices=('instance', 'class', 'descriptor', 'missing'), required=True)
    args = parser.parse_args()
    class Probe:
        pass
    model, token, missing = Probe(), object(), object()
    value = missing
    if args.case == 'instance':
        model.value = token
    elif args.case == 'class':
        Probe.value = token
        value = token
    elif args.case == 'descriptor':
        def forbidden(self):
            raise AssertionError('Python property must not execute')
        Probe.value = property(forbidden)
        value = Probe.__dict__['value']
    fields = vars(model)
    entry = getattr(extension, args.path)
    if args.path == 'original_shaped':
        call_args = model, fields, 'value', {'missing': missing}, 'TypeError'
    elif args.path == 'conditional':
        assert args.case in ('instance', 'missing')
        call_args = fields, 'value'
    elif args.path == 'candidate':
        call_args = fields, 'value', value, missing
    else:
        call_args = fields, 'value', value, {'missing': missing}
    print('begin', args.path, args.case, flush=True)
    try:
        result = entry(*call_args)
    except Exception as error:
        assert args.case in ('descriptor', 'missing'), repr(error)
        assert ('constructor parameter' in str(error) or 'Python properties' in str(error)
                or 'missing native dictionary key' in str(error)), repr(error)
    else:
        assert args.case in ('instance', 'class')
        assert result is token
    print('passed', args.path, args.case, flush=True)


if __name__ == '__main__':
    main()
