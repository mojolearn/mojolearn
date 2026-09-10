#!/usr/bin/env python3
"""Read-only lexical inventory; only --output is written."""
from pathlib import Path
import re, subprocess, json
import argparse
parser = argparse.ArgumentParser(description="Inventory named Mojo math calls without counting comments/docstrings/messages")
parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
root = args.root.resolve()
names=['erfc','expm1','log10','log2','asin','acos','atan','atan2','atanh','cbrt','sinh','cosh','hypot','lgamma','tgamma','log1p','erf','tanh','sqrt','pow']
records={name:{'calls':[],'imports':[],'mentions':0} for name in names}
def blank(m):return '\n'*m.group(0).count('\n')
files=subprocess.check_output(['rg','--files','-g','*.mojo'],cwd=root,text=True).splitlines()
for file in files:
 s=(root/file).read_text()
 code=re.sub(r'(?s)("""|\'\'\').*?\1',blank,s)
 code=re.sub(r'#[^\n]*','',code)
 ffi=code
 code=re.sub(r'"(?:\\.|[^"\\])*"|\'(?:\\.|[^\'\\])*\'',blank,code)
 category='production/library' if file in ['checks/numerics.mojo', 'spectral/checks/symmetric_eig_host.mojo'] else 'gate/oracle' if '/checks/' in '/'+file or file.startswith('checks/') else 'benchmark' if '/bench/' in '/'+file or file.startswith('bench/') else 'production/library'
 for name in names:
  records[name]['mentions']+=len(re.findall(r'\b'+name+r'\b',s))
  locations=set()
  for m in re.finditer(r'\b'+name+r'\s*\(',code):
   line=code.count('\n',0,m.start())+1
   raw=code.splitlines()[line-1]
   if re.search(r'\b(?:def|fn)\s+'+name+r'\s*\(',raw):continue
   locations.add(line)
  for m in re.finditer(r'external_call\s*\[\s*[\'\"]'+name+r'[\'\"]',ffi): locations.add(ffi.count('\n',0,m.start())+1)
  for line in sorted(locations):records[name]['calls'].append({'path':file,'line':line,'category':category,'source':s.splitlines()[line-1].strip()})
  for i,line in enumerate(code.splitlines(),1):
   if re.search(r'^\s*from\s+.*math\s+import\b',line) and re.search(r'\b'+name+r'\b',line):records[name]['imports'].append({'path':file,'line':i,'category':category})
revision = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip()
args.output.write_text(json.dumps({'source_revision': revision, 'scope': 'tracked/nonignored Mojo files; docstrings/comments/string literals removed for call detection; named external_call targets included; lexical categories are not reachability proofs', 'file_count': len(files), 'operations': records}, indent=2) + '\n')
for name,r in records.items():
 print(name,'mentions',r['mentions'],'calls',len(r['calls']),'production',sum(x['category']=='production/library' for x in r['calls']),'gate',sum(x['category']=='gate/oracle' for x in r['calls']),'bench',sum(x['category']=='benchmark' for x in r['calls']))
 if name not in ['sqrt','pow','log2','tanh','erf']:
  for c in r['calls']:print(' ',c['path'],c['line'],c['source'])
