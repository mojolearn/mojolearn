"""Small installed CPython compatibility check; numerical suites run on 3.14."""
import os
from pathlib import Path
import sys

import mojolearn as ml
from mojolearn import _backend, _merged, host_surface


package = Path(ml.__file__).resolve().parent
assert package.is_relative_to(Path(sys.prefix).resolve()), package
exports = {family['binding']: family['exports'] for family in host_surface.FAMILIES}
for name in host_surface.wheel_bindings():
    path = Path(_merged.binding_path(str(package / 'host'), name)).resolve()
    assert path.exists(), path
    if os.environ.get('MOJOLEARN_MERGED_BINDINGS', '1') == '1':
        assert _merged.is_merged_file(str(path)), path
    assert Path(_backend.host_module_path(name)).resolve() == path
    module = _backend.load_host_module(name)
    assert all(callable(getattr(module, export, None)) for export in exports[name]), name
    prefix = name[len('_mojolearn_'):]
    assert getattr(module, prefix + '_vendor')() == 'cpu', name
    assert getattr(module, prefix + '_numeric_mode')() == 1, name
    assert str(getattr(module, prefix + '_column')()) == 'cpu', name

# Exercise argument conversion, native execution and returned Array ownership
# on every supported interpreter, without NumPy or an accuracy suite.
x = ml.Array.from_list([[1.0, 2.0], [3.0, 4.0]], '<f4')
assert x.tolist() == [[1.0, 2.0], [3.0, 4.0]]
if '--no-gpu' not in sys.argv:
    mode = os.environ['MOJOLEARN_NUMERIC_MODE']
    result = ml.StandardScaler(numeric_mode=mode).fit_transform(x)
    assert type(result) is ml.Array and result.shape == (2, 2)
    assert result.tolist() == [[-1.0, -1.0], [1.0, 1.0]]
    scope = 'native fit_transform passed'
else:
    scope = 'import/host loading/Array only; DEVICE NOT TESTED'
assert not any(name == 'numpy' or name.startswith('numpy.') for name in sys.modules)
print(f'{len(exports)} host families; {scope}; numerical suites reserved for Python 3.14')
