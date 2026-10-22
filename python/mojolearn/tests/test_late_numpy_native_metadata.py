"""Late optional NumPy connection must not invoke replaced Python factories."""
import subprocess
import sys

import pytest


@pytest.mark.parametrize("name", ["dtype", "ndarray"])
@pytest.mark.parametrize("operation", ["fit", "getattr"])
def test_late_numpy_factory_replacement_is_refused_before_callback(name, operation):
    script = r'''
import sys
import mojolearn as ml
from mojolearn._verify_native_runtime import audit_native_call
assert "numpy" not in sys.modules, "test must start before optional NumPy connection"
model = ml.StandardScaler()
# Retain the callable while the optional NumPy connection is still absent.
# Attribute access is itself a native operation and is audited separately.
fit = model.fit
assert "numpy" not in sys.modules
import numpy as np
values = np.arange(24, dtype="f4").reshape(8, 3)
name = sys.argv[1]
original = getattr(np, name)
calls = []
def replacement(*args, **kwargs):
    calls.append(name)
    raise AssertionError("replaced NumPy factory entered Python")
setattr(np, name, replacement)
try:
    if sys.argv[2] == "fit":
        result = audit_native_call(fit, values)
    else:
        result = audit_native_call(getattr, model, "fit")
finally:
    setattr(np, name, original)
assert type(result.exception) is TypeError, result.report()
assert not calls, calls
assert "native" in str(result.exception), str(result.exception)
report = result.report()
for key in ("python_frames", "forbidden_native_calls", "wrapper_work", "unobserved_existing_threads"):
    assert not report[key], report
print("late native metadata refusal:", name)
'''
    result = subprocess.run([sys.executable, "-c", script, name, operation], text=True,
                            capture_output=True, timeout=90)
    assert result.returncode == 0, (result.stdout, result.stderr)
