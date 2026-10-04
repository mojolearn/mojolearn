"""Reviewed v2 proof for the be334 -> a4d launch-bound-only transition."""
import hashlib,json,pathlib,re,subprocess
OLD='be33488600e71ac49f6a42c364d6e45895fad7d3';NEW='a4d01a130c8d05ff0208a39989d36b86b69aa0ed'
CHANGED_CODE='x_linear/device.mojo'
def digest(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()
def graph(repo,sha):
 raw=subprocess.check_output(['git','-C',str(repo),'ls-tree','-rz','--full-tree',sha])
 entries={row.split(b'\t',1)[1].decode():row+b'\0' for row in raw.split(b'\0') if row}
 paths=set(entries);edges={};records=[]
 result=subprocess.run(['git','-C',str(repo),'grep','-n','-E',r'^[[:space:]]*(from|import)[[:space:]]',sha,'--','*.mojo'],capture_output=True,text=True)
 if result.returncode not in (0,1):raise RuntimeError('import graph scan failed')
 for line in result.stdout.splitlines():
  _,source,number,text=line.split(':',3)
  match=re.match(r'\s*from\s+([.\w]+)\s+import\s+(.*)',text)
  if match:specs=[(match[1],match[2])]
  else:
   match=re.match(r'\s*import\s+(.+)',text)
   if not match:continue  # prose in docstrings, not a syntactically valid import
   specs=[(part.strip().split()[0],'') for part in match[1].split(',') if part.strip()]
  for spec,names in specs:
   targets=set();bases=[]
   if spec.startswith('.'):
    dots=len(spec)-len(spec.lstrip('.'));parent=pathlib.PurePosixPath(source).parent
    for _ in range(dots-1):parent=parent.parent
    bases=[str(parent/spec[dots:].replace('.','/'))]
   else:bases=[spec.replace('.','/'),str(pathlib.PurePosixPath(source).parent/spec.replace('.','/'))]
   for base in bases:
    if base+'.mojo' in paths:targets.add(base+'.mojo')
    if base+'/__init__.mojo' in paths:
     # Conservative package resolution: every member, not guessed reexports.
     targets.update(p for p in paths if p.startswith(base+'/') and p.endswith('.mojo'))
   records.append({'source':source,'line':int(number),'module':spec,'members':names,'local_targets':sorted(targets)})
   edges.setdefault(source,set()).update(targets)
 return entries,{'sha':sha,'imports':records,'edges':{k:sorted(v) for k,v in sorted(edges.items())}}
def reachable(g,entry):
 seen=set();queue=[entry]
 while queue:
  node=queue.pop()
  if node in seen:continue
  seen.add(node);queue.extend(g['edges'].get(node,[]))
 return seen
def prove(repo,module,out,reviewed):
 if module=='x_linear':raise RuntimeError('affected GPU linear module cannot be reused')
 changed=subprocess.check_output(['git','-C',str(repo),'diff','--name-only',OLD,NEW],text=True).splitlines()
 if set(changed)!=set(reviewed) or CHANGED_CODE not in reviewed:raise RuntimeError('unreviewed changed file inventory')
 noncode=set(reviewed)-{CHANGED_CODE}
 # Exact reviewed external evidence/harness paths; no broad path exclusion.
 allowed={'bench/results/r2-index.tsv','docs/identical/optimization-ledger.json','docs/identical/toggle-inventory.json','tools/identical_wave_cnn_sgd_gate.py'}
 allowed|={'bench/results/identical-integration-20261004/'+x+'.json' for x in ['apple-callpath-bits','dart-full-column-bits','eigh-block-drop','gmm-power-bits','gram-prophet-bits','lu-nan-fix','m4-host-callpath-bits','manifest','pca-bits']}
 if noncode!=allowed:raise RuntimeError('unreviewed noncompiler changes')
 filename={'base':'_mojolearn','base_host':'_mojolearn_core_host'}.get(module,'_mojolearn_'+module)
 entry='bindings/'+filename+'.mojo';items=[]
 for sha in [OLD,NEW]:
  entries,g=graph(repo,sha)
  inbound={p for p,deps in g['edges'].items() if CHANGED_CODE in deps}
  if inbound!={'bindings/_mojolearn_x_linear.mojo'}:raise RuntimeError('changed helper has unexpected importers '+repr(inbound))
  reached=reachable(g,entry)
  if CHANGED_CODE in reached:raise RuntimeError('selected module reaches changed helper')
  excluded=noncode|{CHANGED_CODE}
  inputs={path:record.split(b'\t',1)[0].decode() for path,record in entries.items() if path not in excluded}
  graphpath=out/('dependency-graph-'+sha[:12]+'.json')
  graphpath.write_text(json.dumps(g,indent=2)+'\n')
  inputpath=out/('compile-inputs-'+module+'-'+sha[:12]+'.json')
  inputpath.write_text(json.dumps(inputs,sort_keys=True,indent=2)+'\n')
  items.append({'source_sha':sha,'graph_path':str(graphpath),'graph_sha256':hashlib.sha256(graphpath.read_bytes()).hexdigest(),
                'compile_inputs_path':str(inputpath),'compile_inputs_sha256':hashlib.sha256(inputpath.read_bytes()).hexdigest(),
                'compile_inputs_digest':digest(inputs),'reachable_mojo_files':sorted(reached),'inbound_to_changed_helper':sorted(inbound)})
 if items[0]['compile_inputs_digest']!=items[1]['compile_inputs_digest']:raise RuntimeError('compile inputs changed')
 return {'schema':2,'closure_algorithm':'reviewed-mojo-import-graph-and-all-other-tracked-inputs-v2','module':module,
         'old_source_sha':OLD,'new_source_sha':NEW,'changed_files':changed,'own_entrypoint':entry,
         'excluded_unreachable_mojo_files':[CHANGED_CODE],'reviewed_noncompiler_changes':sorted(noncode),
         'identical_closure_sha256':items[0]['compile_inputs_digest'],'sources':items}
