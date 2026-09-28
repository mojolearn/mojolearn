#!/bin/sh
# tools/py_shared/make_base_patch.sh: write tools/py_shared/base.patch, the lane's diff
# against its base (default 0a11b50c7) without its own tooling, tests and progress
# file, so ab_job.sh can rebuild the base tree on a box by reversing it. Untracked:
# it travels with `nvidia_central.sh sync` and is never committed.
cd "$(dirname "$0")/../.."
git diff --binary "${1:-0a11b50c7}" -- . ':!tools/py_shared' ':!bench/py_shared_micro.py' \
  ':!python/mojolearn/tests/test_arena_ranges.py' ':!python/mojolearn/tests/test_portable_math_fast.py' \
  ':!docs/lanes/progress/py-shared.md' > tools/py_shared/base.patch
git diff --stat "${1:-0a11b50c7}" -- . ':!tools/py_shared' ':!bench/py_shared_micro.py' \
  ':!python/mojolearn/tests/test_arena_ranges.py' ':!python/mojolearn/tests/test_portable_math_fast.py' \
  ':!docs/lanes/progress/py-shared.md' | tail -1
