#!/bin/bash
# lane/exposure-leftovers: the chunked LM head v2 Python door on a CPU-only box.
set -u
: "${LEG_OUT:?LEG_OUT must be set by the leg}"
echo "### commit ${MOJOLEARN_COMMIT:-?}"

PYTHONPATH=packaging/portable_math python3 -c \
    "import pathlib, stage; print(stage.build(pathlib.Path('python/mojolearn/.libs/libMojolearnMath.so')))" \
    > "$LEG_OUT/portable_math.log" 2>&1
echo "portable_math exit $?"

echo "### 1. oracle check: done on the first leg (CHUNKED_LM_HEAD_V2_OK)"
TPY="$PWD/.pixi/envs/test/bin/python"
"$TPY" -c "import sys, numpy, pytest; print(sys.version.split()[0], numpy.__version__, pytest.__version__)" 2>&1 | tee "$LEG_OUT/test_env.txt"
python3 -c "import mojolearn, mojolearn._backend as b; print('cpu_only', b._CPU_ONLY is not None, 'built', b.host_families_built())" \
    2>&1 | tee "$LEG_OUT/backend.txt"

echo "### 2. pytest: the new door test and the training host tests beside it"
( cd python && "$TPY" -m pytest -q -rs -s mojolearn/tests/test_chunked_lm_head_door.py \
    mojolearn/tests/test_cpu_training_samba.py ) \
    > "$LEG_OUT/pytest.txt" 2>&1
echo "pytest exit $?"
tail -25 "$LEG_OUT/pytest.txt"

cat > /tmp/clh_hash.py <<'PY'
import hashlib, numpy as np
from mojolearn import training, _backend
lib = _backend.binding("_mojolearn_training", "identical")
print("sabotage", lib.training_host_sabotage())
rng = np.random.default_rng(23)
for rows, vocab, width in ((5, 513, 8), (16, 300, 12)):
    h = rng.uniform(-1, 1, (rows, width)).astype(np.float32)
    w = rng.uniform(-0.5, 0.5, (vocab, width)).astype(np.float32)
    t = (np.arange(rows) * 97 % vocab).astype(np.int64)
    loss = training.chunked_lm_head_loss(h, w, t)
    l2, dh, dw = training.chunked_lm_head_loss(h, w, t, return_grad=True)
    d = lambda a: hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()[:16]
    print(rows, vocab, width, "loss", np.float32(loss).tobytes().hex(), np.float32(l2).tobytes().hex(),
          "dh", d(np.asarray(dh)), "dw", d(np.asarray(dw)))
PY
echo "### 3. clean hashes"
( cd python && python3 /tmp/clh_hash.py ) 2>&1 | tee "$LEG_OUT/hash_clean.txt"
echo "### 4. sabotage hashes (MOJOLEARN_HOST_SABOTAGE training host)"
( cd python && MOJOLEARN_HOST_ALLOW_SABOTAGE=1 MOJOLEARN_HOST_DIR="$PWD/mojolearn/host-sabotage" python3 /tmp/clh_hash.py ) 2>&1 | tee "$LEG_OUT/hash_sabotage.txt"

echo "### 5. gates"
python3 tools/lane_accounting.py --check > "$LEG_OUT/lane_accounting_check.txt" 2>&1
echo "lane_accounting --check exit $?"
tail -4 "$LEG_OUT/lane_accounting_check.txt"
exit 0
