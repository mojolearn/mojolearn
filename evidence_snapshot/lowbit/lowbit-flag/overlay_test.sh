#!/bin/bash
# Run the flag tests against the RELEASED wheel with this lane's changed Python
# files laid over it. No build: the binaries are the release's.
set -uo pipefail
T=/root/mojolearn-lowbit-flag
V=/root/lowbit-flag-venv
cd "$T"
PY=$(ls "$T"/.pixi/envs/test/bin/python 2>/dev/null || command -v python3)
[ -x "$V/bin/python" ] || "$PY" -m venv "$V"
"$V/bin/python" -m pip install -q --upgrade pip >/dev/null 2>&1
"$V/bin/python" -m pip install -q "mojolearn[verify]==0.8.25" pytest 2>&1 | tail -3
SITE=$("$V/bin/python" -c 'import importlib.util,os; print(os.path.dirname(importlib.util.find_spec("mojolearn").origin))')
echo "installed package: $SITE"
"$V/bin/python" -m pip list 2>/dev/null | grep -i -E '^mojolearn|^numpy|^pytest'
# regression tests the wheel does not ship: taken from this lane's tree as they are
mkdir -p "$SITE/tests"; [ -f "$SITE/tests/__init__.py" ] || cp "$T/python/mojolearn/tests/__init__.py" "$SITE/tests/__init__.py" 2>/dev/null || : > "$SITE/tests/__init__.py"
cp "$T"/python/mojolearn/tests/*.py "$SITE/tests/" 2>/dev/null; ls "$SITE/tests" | wc -l
for f in _numeric_profile.py __init__.py _samba_impl.py _training_impl.py _byte_lm_host.py _byte_lm_impl.py models/causal_lm.py models/parallel_causal_lm.py tests/test_numeric_profile.py; do
    mkdir -p "$SITE/$(dirname "$f")"
    if [ -f "$SITE/$f" ]; then
        # the overlay is honest only if the release's file equals this lane's BASE file
        if git show "$(git rev-parse HEAD)":"python/mojolearn/$f" | cmp -s - "$SITE/$f"; then echo "base==release  $f"; else echo "BASE DIFFERS FROM RELEASE  $f"; fi
    else
        echo "new file       $f"
    fi
    cp "$T/python/mojolearn/$f" "$SITE/$f"
done
cd /root
export MOJOLEARN_NUMERIC_MODE=identical
"$V/bin/python" -c 'import mojolearn as ml; print("import ok", ml.__version__ if hasattr(ml,"__version__") else "", ml.numeric_profile(), ml.numeric_mode())'
echo "import_exit=$?"
TESTS="$SITE/tests/test_numeric_profile.py"
for t in test_models_loader.py test_parallel_causal_lm.py test_lowbit_weights.py test_training_primitives_surface.py test_samba_surface.py test_byte_lm_surface.py test_byte_lm_host_trainer.py; do [ -f "$SITE/tests/$t" ] && TESTS="$TESTS $SITE/tests/$t"; done
"$V/bin/python" -m pytest -q -rs -p no:cacheprovider $TESTS
echo "pytest_exit=$?"
