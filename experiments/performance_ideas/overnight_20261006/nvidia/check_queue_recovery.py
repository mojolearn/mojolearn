"""Operational queue publication/capture checks only; no algorithm executed."""
import concurrent.futures,json,pathlib,tempfile,threading,time
E=pathlib.Path(__file__).resolve().parent
with tempfile.TemporaryDirectory() as directory:
 root=pathlib.Path(directory);path=root/'queue.json';path.write_text(json.dumps([dict(key='retained')]))
 stop=threading.Event();errors=[];reads=[0]
 def reader():
  while not stop.is_set():
   try:
    value=json.loads(path.read_text());assert isinstance(value,list) and value[0]['key']=='retained';reads[0]+=1
   except Exception as error:errors.append(repr(error))
 thread=threading.Thread(target=reader);thread.start()
 def writer(i):
  raw=json.dumps([dict(key='retained'),dict(key='new-'+str(i),payload='x'*20000)]);tmp=root/('queue.incoming-'+str(i)+'-'+str(time.time_ns()))
  with tmp.open('w') as file:file.write(raw[:100]);file.flush();time.sleep(.002);file.write(raw[100:])
  tmp.replace(path)
 with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:list(pool.map(writer,range(30)))
 stop.set();thread.join();assert not errors,errors
current=json.loads((E/'default-capture/artifacts/repairs/results.json').read_text());old=None
for path in sorted((E/'default-repair-snapshots').glob('*.json'),key=lambda p:p.stat().st_mtime,reverse=True):
 data=json.loads(path.read_text())
 if len(data)==78:old=data;break
assert old is not None,'missing pre-crash snapshot'
assert all(current.get(key)==record for key,record in old.items()),'completed raw result changed during worker recovery'
print(json.dumps(dict(status='PASS',concurrent_publications=30,successful_reads=reads[0],parse_errors=0,retained_completed_records=len(old),current_records=len(current),policy='No algorithm execution or identity/compile validation')))
