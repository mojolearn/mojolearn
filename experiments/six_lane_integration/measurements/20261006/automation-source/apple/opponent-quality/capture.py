from pathlib import Path
import datetime,json,subprocess
BASE=Path(__file__).resolve().parent
stamp=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')
logs=BASE/'capture-logs';logs.mkdir(exist_ok=True)
cmd=['rsync','-az','--exclude=inputs/','--exclude=cache/','--exclude=tmp/','--exclude=work/','-e','ssh -i /Users/andrewhendel/.ssh/mambik-l8.pem -o BatchMode=yes -o ConnectTimeout=15','ec2-user@54.157.1.251:/Volumes/MojoPerfBuild20261005/six-lane-full-ab-20261006/apple/opponent-quality/',str(BASE/'captured')+'/']
with (logs/(stamp+'.log')).open('ab') as log:r=subprocess.run(cmd,stdout=log,stderr=log,timeout=90)
(BASE/'capture-status.json').write_text(json.dumps({'exit':r.returncode,'at':stamp,'retention':'All result/quality/resource JSON and complete logs mirrored; large saved outputs remain on persistent M3 under recorded board/work paths, never deleted.'},indent=2)+'\n')
raise SystemExit(r.returncode)
