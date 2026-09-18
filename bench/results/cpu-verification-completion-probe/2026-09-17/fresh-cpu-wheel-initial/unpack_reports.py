"""Reconstruct each original verifier JSON byte for byte, checking its digest."""
import gzip,hashlib,json,sys
from pathlib import Path
root=Path(__file__).resolve().parent
out=Path(sys.argv[1]);out.mkdir(parents=True,exist_ok=False)
bundle=json.loads(gzip.decompress((root/'reports.json.gz').read_bytes()))
for lane,delta in bundle['reports'].items():
 data=(json.dumps(dict(bundle['shared'],**delta),indent=1,sort_keys=True)+'\n').encode()
 assert hashlib.sha256(data).hexdigest()==bundle['sha256'][lane],lane
 (out/(lane+'.json')).write_bytes(data)
print(f"Reconstructed {len(bundle['reports'])} original reports")
