import hashlib,json
from pathlib import Path
from mojolearn import host_surface,_backend
rows=[]
for family in host_surface.FAMILIES:
 try:
  mod=_backend.load_host_module(family['binding'])
  missing=[name for name in family['exports'] if not hasattr(mod,name)]
  row={'family':family['family'],'missing':missing,'path':mod.__file__,'sha256':hashlib.sha256(Path(mod.__file__).read_bytes()).hexdigest()}
 except Exception as e:row={'family':family['family'],'error':str(e)}
 rows.append(row)
print(json.dumps(rows,indent=2))
raise SystemExit(int(any(r.get('missing') or r.get('error') for r in rows)))
