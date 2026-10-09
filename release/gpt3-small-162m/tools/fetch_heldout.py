"""Fetch N evenly spaced 2,049-token windows from the held-out validation range (R2)."""
import os, sys, json, hashlib, boto3
import numpy as np
N = int(sys.argv[1]); out = sys.argv[2]
man = json.load(open(sys.argv[3]))
lo, hi = man['validation_range']; W = 2049
PART = 2_000_000_000 // 4
s3 = boto3.client('s3', endpoint_url=f"https://{os.environ['R2_ACCOUNT_ID']}.r2.cloudflarestorage.com",
                  aws_access_key_id=os.environ['R2_ACCESS_KEY_ID'], aws_secret_access_key=os.environ['R2_SECRET_ACCESS_KEY'], region_name='auto')
key = 'corpus/fineweb-edu-10BT/tokens/mojolearn-bpe-fineweb-edu-50257-v1/tokens.i32.part%02d'
def read(start, n):
    parts = []
    while n:
        p, off = divmod(start, PART); take = min(n, PART - off)
        b = s3.get_object(Bucket=os.environ['R2_BUCKET'], Key=key % p, Range=f'bytes={off*4}-{(off+take)*4-1}')['Body'].read()
        assert len(b) == take * 4; parts.append(b); start += take; n -= take
    return b''.join(parts)
stride = (hi - lo - W) // (N - 1)
starts = [lo + i * stride for i in range(N)]
data = b''.join(read(s, W) for s in starts)
arr = np.frombuffer(data, '<i4').reshape(N, W)
assert arr.min() >= 0 and arr.max() < 50256
arr.tofile(out)
json.dump({'validation_range': [lo, hi], 'window': W, 'n_windows': N, 'stride': stride, 'starts': starts,
           'tokens_sha256': man['sha256'], 'windows_sha256': hashlib.sha256(arr.tobytes()).hexdigest()},
          open(out + '.json', 'w'), indent=1)
print('ok', arr.shape, hashlib.sha256(arr.tobytes()).hexdigest()[:16])
