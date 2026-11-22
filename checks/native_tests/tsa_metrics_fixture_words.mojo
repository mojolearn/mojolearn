# SPDX-License-Identifier: Apache-2.0
"""Fixture directories of the tsa and metrics native checks
(`tools/tsa_metrics_native_fixtures.py` writes them). Importable; no
`main()`. Paths are relative to the repository root, where the pixi tasks
run. The word readers are `checks/native_tests/testsuite_cases.mojo`
(`fixture_words`, `fixture_f32`, `fixture_i32`).
"""

comptime TSA_FIXTURE_DIR = "checks/native_tests/fixtures/tsa/"
comptime METRICS_FIXTURE_DIR = "checks/native_tests/fixtures/metrics/"
