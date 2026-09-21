#!/bin/sh
# Run packaging/portable_math/test_wheel.py, the wheel's dependency and
# platform-math audit tests. They need lief (the pkg environment) and pytest
# (the test environment), and no single pixi environment has both.
set -eu
cd "$(dirname "$0")/.."
SP=$(.pixi/envs/test/bin/python -c 'import os, pytest; print(os.path.dirname(os.path.dirname(pytest.__file__)))')
exec .pixi/envs/pkg/bin/python -c "import sys; sys.path.append('$SP'); import pytest; sys.exit(pytest.main(['-q', '-p', 'no:cacheprovider', 'packaging/portable_math/test_wheel.py']))"
