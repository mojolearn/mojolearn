# Final source checks

Source checkout on Metal, CPython3.14 with optional NumPy/pytest/sklearn test
oracles; PyTorch absent. Full command: `PYTHONPATH=python python -m pytest -q
python/mojolearn/tests` (995pass,53skip,89subtests). No-NumPy smoke is
`PYTHONPATH=python python3.N tools/check_numpy_free_runtime.py` forN=10,14.
Release harness: `python3 tools/test_macos_verify_wheel.py` (10pass).

Base extensions rebuilt using `MOJOLEARN_NUMERIC_MODE=MODE nice -n19
tools/with_build_lock.sh sh bindings/build.sh` in each mode. Native nonzero
gates: `MOJOLEARN_NUMERIC_MODE=MODE PYTHONPATH=python python -m pytest -q
python/mojolearn/tests/test_native_nonzero.py` (29pass per mode, zero skips).
Build logs remain /tmp/boundary-final-base-MODE.log; all builds succeeded.
These are source checks, not installed-wheel or cross-vendor certification.
