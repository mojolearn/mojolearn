import os,sys,json,time,fcntl,pathlib,subprocess,hashlib
B=pathlib.Path('/Volumes/MojoPerfBuild20261005/six-lane-full-ab-20261006/apple');E=B/'drain-reservation';E.mkdir(exist_ok=True)
if (E/'status.json').exists():raise SystemExit('Reservation already exists; inspect retained status')
if os.fork():raise SystemExit(0)
os.setsid()
if os.fork():os._exit(0)
log=open(E/'owner.log','ab',buffering=0);os.dup2(log.fileno(),1);os.dup2(log.fileno(),2);fd=os.open('/dev/null',os.O_RDONLY);os.dup2(fd,0)
def write(d):
 p=E/'status.tmp';p.write_text(json.dumps(d,indent=2)+'\n');p.replace(E/'status.json')
s=dict(status='ACQUIRING',pid=os.getpid(),started_at=time.time(),guard_token=sys.argv[1]);write(s)
try:
 with open('/Users/ec2-user/mojolearn-full-f867b50e8/gpu.lock','r+b') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  processes=subprocess.check_output(['ps','-axo','pid=,ppid=,user=,command='],text=True);(E/'process-snapshot.txt').write_text(processes)
  suspect=[]
  for line in processes.splitlines():
   parts=line.strip().split(None,3)
   if len(parts)!=4:continue
   pid,ppid,user,command=parts
   if int(pid)==os.getpid() or user!='ec2-user':continue
   if any(x in command for x in ['drain-reservation.py','apple_steward.py','aggregate-status.py']):continue
   if any(x in command.lower() for x in ['python','mojo ','mojo-run','bench_board','classical_two_datasets','performance_full_ab_queue','six_lane_ab_worker']):suspect.append(line)
  if suspect:raise RuntimeError('Other possible jobs remain: '+repr(suspect))
  mount=subprocess.check_output(['df','-P',str(B),'/'],text=True);(E/'mount-mapping.txt').write_text(mount)
  disk=subprocess.check_output(['diskutil','info','-plist','/Volumes/MojoPerfBuild20261005']);(E/'external-volume-info.plist').write_bytes(disk)
  if not os.path.ismount('/Volumes/MojoPerfBuild20261005'):raise RuntimeError('Expected retained external volume not mounted')
  s.update(status='HELD',lock_exclusive=True,no_other_jobs=True,mount_mapping=mount,updated_at=time.time());write(s)
  while not (E/'release-reservation').exists():time.sleep(5)
  s.update(status='RELEASED',updated_at=time.time());write(s)
except BaseException as e:s.update(status='FAILED',error=repr(e),updated_at=time.time());write(s);raise
