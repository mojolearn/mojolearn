import os, sys, struct, time
root = sys.argv[1]; os.environ['MOJOLEARN_BYTE_LM_HOST_BINARY'] = sys.argv[2]
sys.path.insert(0, root + '/python')
from mojolearn import LanguageModelInference
from mojolearn._buffer import frombytes
cap = root + '/bench/results/resume/2026-09-07-root-byte-lm-three-vendor/apple/full128'
m = LanguageModelInference.from_checkpoint(cap + '/final.checkpoint.json', threaded=True, threads=1)
flat = list(struct.unpack('<66i', open(cap + '/heldout-final/batch00.ids.i32', 'rb').read())) * 8
x = frombytes(struct.pack('<256i', *flat[:256]), '<i4', (8, 32))
y = frombytes(open(cap + '/heldout-final/batch00.ids.i32', 'rb').read(), '<i4', (2, 33))
print('pid', os.getpid(), flush=True)
end = time.time() + float(sys.argv[3])
while time.time() < end:
    m.logits(x); m.loss_bits(y)
